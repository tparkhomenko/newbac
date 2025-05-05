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

def train(config_name="balanced"):
    # Clear GPU memory
    torch.cuda.empty_cache()
    gc.collect()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load config
    config = load_config()
    
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
    for epoch in range(50):  # 50 epochs
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        all_targets = []
        all_predictions = []
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/50')
        for batch_idx, (features, targets) in enumerate(progress_bar):
            try:
                # Move data to device
                features = features.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                # Ensure data types are correct
                features = features.float()
                targets = targets.long()
                
                # Mixed precision forward pass
                with torch.amp.autocast('cuda'):
                    outputs = model(features)
                    loss = criterion(outputs, targets)
                
                # Mixed precision backward pass
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                # Calculate accuracy
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()
                
                # Update total loss
                train_loss += loss.item()
                
                # Log to wandb only every log_interval batches
                if batch_idx % log_interval == 0:
                    wandb.log({
                        'train_batch_loss': loss.item(),
                        'train_batch_acc': 100.*train_correct/train_total,
                        'learning_rate': optimizer.param_groups[0]['lr']
                    })
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': train_loss/(batch_idx+1),
                    'Acc': 100.*train_correct/train_total
                })
                
                # Collect targets and predictions for metrics
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                
                # Clean up GPU memory
                del features, targets, outputs, loss
                torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Error in training batch {batch_idx}: {str(e)}")
                continue
        
        # Calculate train metrics
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        train_f1 = f1_score(all_targets, all_predictions, average='weighted')
        train_precision = precision_score(all_targets, all_predictions, average='weighted')
        train_recall = recall_score(all_targets, all_predictions, average='weighted')
        
        # Log train metrics
        wandb.log({
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_f1': train_f1,
            'train_precision': train_precision,
            'train_recall': train_recall,
            'epoch': epoch+1
        })
        
        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_val_targets = []
        all_val_predictions = []
        
        with torch.no_grad():
            for features, targets in tqdm(val_loader, desc='Validation'):
                features = features.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                with torch.amp.autocast('cuda'):
                    outputs = model(features)
                    loss = criterion(outputs, targets)
                
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                val_loss += loss.item()
                
                # Collect targets and predictions for metrics
                all_val_targets.extend(targets.cpu().numpy())
                all_val_predictions.extend(predicted.cpu().numpy())
                
                del features, targets, outputs, loss
                torch.cuda.empty_cache()
        
        # Calculate validation metrics
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        val_f1 = f1_score(all_val_targets, all_val_predictions, average='weighted')
        val_precision = precision_score(all_val_targets, all_val_predictions, average='weighted')
        val_recall = recall_score(all_val_targets, all_val_predictions, average='weighted')
        
        # Generate and log confusion matrix
        cm = confusion_matrix(all_val_targets, all_val_predictions)
        wandb.log({
            "confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=all_val_targets,
                preds=all_val_predictions,
                class_names=class_names
            )
        })
        
        # Save confusion matrix
        plot_confusion_matrix(cm, class_names, epoch+1, confusion_matrix_dir)
        
        # Log validation metrics
        wandb.log({
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_f1': val_f1,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'epoch': epoch+1
        })
        
        # Print metrics
        logger.info(f'Epoch {epoch+1}: '
                   f'Train Loss: {train_loss:.4f} '
                   f'Train Acc: {train_acc:.2f}% '
                   f'Train F1: {train_f1:.4f} '
                   f'Val Loss: {val_loss:.4f} '
                   f'Val Acc: {val_acc:.2f}% '
                   f'Val F1: {val_f1:.4f}')
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Check if best model and save
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save checkpoint
            checkpoint_dir = os.path.join(project_root, 'saved_models', 'benign_malignant')
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, experiment_config["model_path"])
            
            # Convert class_weights to CPU before saving
            weights_cpu = class_weights.cpu().numpy().astype(np.float32)
            
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_f1': val_f1,
                'class_mapping': {0: 'benign', 1: 'malignant'},
                'class_weights': weights_cpu,
                'original_only': original_only,
                'balanced_sampling': balanced_sampling,
                'max_samples_per_class': max_samples_per_class,
                'experiment': experiment_config["description"]
            }, checkpoint_path)
            
            logger.info(f'Saved best model checkpoint to {checkpoint_path}')
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= early_stopping_patience:
            logger.info(f'Early stopping triggered after {epoch+1} epochs')
            break
        
        # Clear memory after each epoch
        gc.collect()
        torch.cuda.empty_cache()
    
    wandb.finish()
    logger.info("Training completed!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train benign/malignant classifier')
    parser.add_argument('--config', type=str, default='balanced', 
                        choices=['original', 'augmented', 'balanced'],
                        help='Configuration to use: original, augmented, or balanced')
    args = parser.parse_args()
    
    train(config_name=args.config) 