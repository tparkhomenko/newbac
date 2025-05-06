import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import wandb
import logging
import gc
import argparse
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import time
from datetime import datetime

# Set environment variables for better GPU memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.benign_malignant_head import BenignMalignantHead
from sam.sam_encoder import SAMFeatureExtractor
from datasets.benign_malignant_dataset import BenignMalignantDataset

# Focal Loss implementation
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
        # Ensure alpha is float32 before passing to CrossEntropyLoss
        if alpha is not None and isinstance(alpha, torch.Tensor):
            alpha = alpha.float()
            
        self.cross_entropy = nn.CrossEntropyLoss(weight=alpha, reduction='none')
        
    def forward(self, inputs, targets):
        ce_loss = self.cross_entropy(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

def load_config():
    """Load configuration from config.yaml."""
    config_path = project_root / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_config(config_name):
    """Get configuration for the specified name."""
    configs = {
        "original": {
            "name": "benign-malignant-original",
            "description": "Training with original images only (~31.5k)",
            "original_only": True,
            "balanced_sampling": False,
            "max_samples_per_class": None,
            "model_path": "benign_malignant_original_best.pth"
        },
        "augmented": {
            "name": "benign-malignant-augmented",
            "description": "Training with all augmented images (~157.6k)",
            "original_only": False,
            "balanced_sampling": False,
            "max_samples_per_class": None,
            "model_path": "benign_malignant_augmented_best.pth"
        },
        "balanced": {
            "name": "benign-malignant-balanced",
            "description": "Training with balanced subset (2k per class)",
            "original_only": False,
            "balanced_sampling": False,
            "max_samples_per_class": 2000,
            "model_path": "benign_malignant_balanced_best.pth"
        }
    }
    
    # Default to balanced if not specified
    return configs.get(config_name, configs["balanced"])

def plot_confusion_matrix(cm, class_names, epoch, save_dir):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - Epoch {epoch}')
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the figure
    plt.savefig(f'{save_dir}/confusion_matrix_epoch_{epoch}.png', bbox_inches='tight')
    plt.close()

def setup_logging(config, model_name="benign_malignant"):
    """Setup logging directories and files."""
    # Create timestamp for unique run identification
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create log directories if they don't exist
    log_dir = project_root / 'logs' / model_name / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging to file
    log_file = log_dir / 'training.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Create CSV file for metrics
    metrics_file = log_dir / 'metrics.csv'
    with open(metrics_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch', 'batch', 'train_loss', 'train_acc',
            'val_loss', 'val_acc', 'learning_rate',
            'f1_score', 'precision', 'recall'
        ])
    
    return log_dir, metrics_file

def log_metrics(metrics_file, epoch, batch, train_loss, train_acc, 
                val_loss=None, val_acc=None, learning_rate=None,
                f1=None, precision=None, recall=None):
    """Log metrics to CSV file."""
    with open(metrics_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch, batch, train_loss, train_acc,
            val_loss if val_loss is not None else '',
            val_acc if val_acc is not None else '',
            learning_rate if learning_rate is not None else '',
            f1 if f1 is not None else '',
            precision if precision is not None else '',
            recall if recall is not None else ''
        ])

def train(config_name="balanced"):
    # Clear GPU memory
    torch.cuda.empty_cache()
    gc.collect()
    
    # Setup logging
    config = load_config()
    log_dir, metrics_file = setup_logging(config)
    logger = logging.getLogger(__name__)
    
    # Get experiment configuration
    experiment_config = get_config(config_name)
    logger.info(f"Running experiment: {experiment_config['description']}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Initialize wandb
    wandb.init(
        project="skin-lesion-classification",
        name=experiment_config["name"],
        config={
            "experiment": experiment_config["description"],
            "learning_rate": 0.001,
            "batch_size": 16,
            "epochs": 50,
            "early_stopping_patience": 5,
            "model": "BenignMalignantHead",
            "hidden_dims": [512, 256],
            "focal_loss_gamma": 2.0,
            "original_only": experiment_config["original_only"],
            "balanced_sampling": experiment_config["balanced_sampling"],
            "max_samples_per_class": experiment_config["max_samples_per_class"]
        }
    )
    
    # Create datasets
    metadata_path = config['training']['metadata_path']
    feature_cache_dir = os.path.join(config['data']['processed']['root_dir'], 'features')
    
    # Dataset configuration
    original_only = experiment_config["original_only"]
    balanced_sampling = experiment_config["balanced_sampling"]
    max_samples_per_class = experiment_config["max_samples_per_class"]
    
    logger.info(f"Dataset config - Original only: {original_only}, Balanced: {balanced_sampling}, Max samples: {max_samples_per_class}")
    
    # Create datasets with desired configuration
    train_dataset = BenignMalignantDataset(
        metadata_path=metadata_path,
        split='train',
        feature_cache_dir=feature_cache_dir,
        original_only=original_only,
        balanced_sampling=balanced_sampling,
        max_samples_per_class=max_samples_per_class
    )
    
    val_dataset = BenignMalignantDataset(
        metadata_path=metadata_path,
        split='val',
        feature_cache_dir=feature_cache_dir,
        original_only=original_only,
        balanced_sampling=True,  # Always balance validation for accurate metrics
        max_samples_per_class=2000  # Limit validation set size for memory efficiency
    )
    
    # Calculate class weights
    train_class_counts = train_dataset.class_counts
    total_samples = sum(train_class_counts.values())
    
    # Ensure class weights are float32
    class_weights = torch.tensor([
        float(total_samples) / (len(train_class_counts) * float(train_class_counts[0])),  # benign
        float(total_samples) / (len(train_class_counts) * float(train_class_counts[1]))   # malignant
    ], dtype=torch.float32).to(device)
    
    logger.info(f"Class weights: {class_weights}")
    
    # DataLoader configuration
    batch_size = 16  # Training batch size
    val_batch_size = 32  # Validation batch size (can be larger)
    log_interval = 10  # Log only every 10 batches to reduce output
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Create model
    input_dim = 256  # SAM feature dimension
    hidden_dims = [512, 256]
    output_dim = 2  # binary classification
    
    model = BenignMalignantHead(input_dim, hidden_dims, output_dim, dropout=0.3).to(device)
    
    # Add wandb model watching
    wandb.watch(model, log="all", log_freq=10)
    
    # Setup mixed precision
    scaler = torch.amp.GradScaler('cuda')
    
    # Setup loss and optimizer
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Early stopping
    early_stopping_patience = 5
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Class names for confusion matrix
    class_names = ['benign', 'malignant']
    
    # Setup confusion matrix directory
    confusion_matrix_dir = os.path.join(config['visualization']['save_dir'], 
                                        config['visualization']['subdirs']['confusion_matrices'],
                                        'benign_malignant')
    os.makedirs(confusion_matrix_dir, exist_ok=True)
    
    # Training loop
    best_val_acc = 0
    start_time = time.time()
    
    for epoch in range(wandb.config.epochs):
        epoch_start_time = time.time()
        
        # Train phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        all_train_preds = []
        all_train_targets = []
        
        # Training loop with progress bar
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch_idx, (features, targets) in enumerate(progress_bar):
            try:
                # Move data to device
                features = features.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                # Forward pass
                with torch.cuda.amp.autocast():
                    outputs = model(features)
                    loss = criterion(outputs, targets)
                
                # Backward pass
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                # Calculate accuracy
                with torch.no_grad():
                    _, predicted = outputs.max(1)
                    train_total += targets.size(0)
                    train_correct += predicted.eq(targets).sum().item()
                
                # Store predictions and targets for metrics
                all_train_preds.extend(predicted.cpu().numpy())
                all_train_targets.extend(targets.cpu().numpy())
                
                # Update total loss
                train_loss += loss.item()
                
                # Calculate batch metrics
                batch_loss = loss.item()
                batch_acc = 100. * train_correct / train_total
                current_lr = optimizer.param_groups[0]['lr']
                
                # Log metrics
                log_metrics(
                    metrics_file, epoch, batch_idx,
                    batch_loss, batch_acc,
                    learning_rate=current_lr
                )
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': train_loss/(batch_idx+1),
                    'Acc': batch_acc
                })
                
                # Clean up GPU memory
                del features, targets, outputs, loss
                torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Error in training batch {batch_idx}: {str(e)}")
                continue
        
        # Calculate training metrics
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        train_f1 = f1_score(all_train_targets, all_train_preds, average='weighted')
        train_precision = precision_score(all_train_targets, all_train_preds, average='weighted')
        train_recall = recall_score(all_train_targets, all_train_preds, average='weighted')
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_targets = []
        
        with torch.no_grad():
            for features, targets in tqdm(val_loader, desc='Validation'):
                features = features.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                with torch.cuda.amp.autocast():
                    outputs = model(features)
                    loss = criterion(outputs, targets)
                
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                val_loss += loss.item()
                
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_targets.extend(targets.cpu().numpy())
                
                del features, targets, outputs, loss
                torch.cuda.empty_cache()
        
        # Calculate validation metrics
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        val_f1 = f1_score(all_val_targets, all_val_preds, average='weighted')
        val_precision = precision_score(all_val_targets, all_val_preds, average='weighted')
        val_recall = recall_score(all_val_targets, all_val_preds, average='weighted')
        
        # Log epoch metrics
        log_metrics(
            metrics_file, epoch, 'end',
            train_loss, train_acc,
            val_loss, val_acc,
            optimizer.param_groups[0]['lr'],
            val_f1, val_precision, val_recall
        )
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Log epoch summary
        logger.info(
            f'Epoch {epoch} Summary:\n'
            f'Time: {epoch_time:.2f}s\n'
            f'Train Loss: {train_loss:.4f}\n'
            f'Train Acc: {train_acc:.2f}%\n'
            f'Train F1: {train_f1:.4f}\n'
            f'Val Loss: {val_loss:.4f}\n'
            f'Val Acc: {val_acc:.2f}%\n'
            f'Val F1: {val_f1:.4f}\n'
            f'Val Precision: {val_precision:.4f}\n'
            f'Val Recall: {val_recall:.4f}\n'
            f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}'
        )
        
        # Save confusion matrix plot
        cm = confusion_matrix(all_val_targets, all_val_preds)
        plot_confusion_matrix(cm, ['benign', 'malignant'], epoch, log_dir / 'confusion_matrices')
        
        # Save checkpoint if best validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = log_dir / experiment_config["model_path"]
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'train_loss': train_loss,
                'val_f1': val_f1,
                'val_precision': val_precision,
                'val_recall': val_recall,
                'learning_rate': optimizer.param_groups[0]['lr']
            }, checkpoint_path)
            logger.info(f'Saved best model checkpoint to {checkpoint_path}')
        
        # Clear memory after each epoch
        gc.collect()
        torch.cuda.empty_cache()
    
    # Log total training time
    total_time = time.time() - start_time
    logger.info(f'\nTotal training time: {total_time/3600:.2f} hours')
    logger.info(f'Best validation accuracy: {best_val_acc:.2f}%')
    
    # Save final model
    final_checkpoint_path = log_dir / f'benign_malignant_final.pth'
    torch.save({
        'epoch': wandb.config.epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'train_loss': train_loss,
        'val_f1': val_f1,
        'val_precision': val_precision,
        'val_recall': val_recall,
        'learning_rate': optimizer.param_groups[0]['lr']
    }, final_checkpoint_path)
    logger.info(f'Saved final model checkpoint to {final_checkpoint_path}')
    
    wandb.finish()
    logger.info("Training completed!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train benign/malignant classifier')
    parser.add_argument('--config', type=str, default='balanced', 
                        choices=['original', 'augmented', 'balanced'],
                        help='Configuration to use: original, augmented, or balanced')
    args = parser.parse_args()
    
    train(config_name=args.config) 