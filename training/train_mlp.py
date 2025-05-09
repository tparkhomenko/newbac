import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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
import random

# Set environment variables for better GPU memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import models and datasets
from models.skin_not_skin_head import SkinNotSkinClassifier
from models.lesion_type_head import LesionTypeHead
from models.benign_malignant_head import BenignMalignantHead
from datasets.unified_dataset import UnifiedDataset
from utils.data_split import DATASET_CONFIGS, log_dataset_statistics

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def parse_model_config(model_config):
    """Parse model configuration string to get hidden dimensions and dropout rate."""
    parts = model_config.split('_')
    hidden_dims = []
    dropout = 0.3  # default value
    
    for part in parts:
        if part.startswith('DO'):
            # Extract dropout rate: DO03 -> 0.3
            dropout = float('0.' + part[2:])
        else:
            try:
                # Parse hidden dimension
                hidden_dims.append(int(part))
            except ValueError:
                pass  # Ignore non-numeric parts
    
    return hidden_dims, dropout

def load_config():
    """Load configuration from config.yaml."""
    config_path = project_root / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_logging(config, model_name, model_config, dataset_config):
    """Setup logging directories and files."""
    # Create log directories if they don't exist
    log_dir = project_root / 'logs' / model_name
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create unique log file based on configuration
    metrics_file = log_dir / f"{dataset_config}_{model_config}_metrics.csv"
    
    # Setup logging to file
    log_file = log_dir / f"{dataset_config}_{model_config}_training.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Create CSV file for metrics
    with open(metrics_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch', 'train_loss', 'train_acc', 'train_f1',
            'val_loss', 'val_acc', 'val_f1', 'learning_rate',
            'precision', 'recall'
        ])
    
    return log_dir, metrics_file

def log_metrics(metrics_file, epoch, train_loss, train_acc, train_f1,
                val_loss=None, val_acc=None, val_f1=None, learning_rate=None,
                precision=None, recall=None):
    """Log metrics to CSV file."""
    with open(metrics_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch, train_loss, train_acc, train_f1,
            val_loss if val_loss is not None else '',
            val_acc if val_acc is not None else '',
            val_f1 if val_f1 is not None else '',
            learning_rate if learning_rate is not None else '',
            precision if precision is not None else '',
            recall if recall is not None else ''
        ])

def plot_confusion_matrix(cm, class_names, epoch, save_dir):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - Epoch {epoch}')
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the figure
    plt.savefig(f'{save_dir}/confusion_matrix_epoch_{epoch}.png', bbox_inches='tight')
    plt.close()

def create_model(model_type, model_config, num_classes):
    """Create model based on type and configuration."""
    input_dim = 256  # SAM feature dimension
    hidden_dims, dropout = parse_model_config(model_config)
    
    if model_type == 'mlp1':
        model = SkinNotSkinClassifier()
    elif model_type == 'mlp2':
        model = LesionTypeHead(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=num_classes, dropout=dropout)
    elif model_type == 'mlp3':
        model = BenignMalignantHead(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=num_classes, dropout=dropout)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model

def get_class_names(dataset_type):
    """Get class names based on dataset type."""
    if dataset_type == 'skin_classification':
        return ['not-skin', 'skin']
    elif dataset_type == 'lesion_type':
        return ['melanocytic', 'non-melanocytic carcinoma', 'keratosis', 'fibrous', 'vascular']
    elif dataset_type == 'benign_malignant':
        return ['benign', 'malignant']
    else:
        return [f'class_{i}' for i in range(10)]  # Default

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(model_config, dataset_config, dry_run=False, wandb_project='skin-lesion-classification'):
    logger = logging.getLogger(__name__)
    # Set fixed random seed for reproducibility
    SEED = 42
    set_global_seed(SEED)
    logger.info(f"Random seed set to {SEED}")
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    gc.collect()
    
    # Load config
    config = load_config()
    
    # Get dataset configuration
    if dataset_config not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset config: {dataset_config}")
    
    dataset_config_dict = DATASET_CONFIGS[dataset_config]
    model_type = dataset_config_dict['model']
    dataset_type = dataset_config_dict['dataset_type']
    
    # Setup logging
    log_dir, metrics_file = setup_logging(config, model_type, model_config, dataset_config)
    
    # Parse model configuration
    hidden_dims, dropout = parse_model_config(model_config)
    
    # Log training configuration
    logger.info(f"\n{'='*60}")
    logger.info(f"TRAINING CONFIGURATION")
    logger.info(f"{'='*60}")
    logger.info(f"Model Type: {model_type}")
    logger.info(f"Model Config: {model_config}")
    logger.info(f"Hidden Dimensions: {hidden_dims}")
    logger.info(f"Dropout Rate: {dropout}")
    logger.info(f"Dataset Config: {dataset_config}")
    logger.info(f"Dataset Type: {dataset_type}")
    logger.info(f"Description: {dataset_config_dict['description']}")
    logger.info(f"{'='*60}\n")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Setup feature cache directory
    feature_cache_dir = os.path.join(config['data']['processed']['root_dir'], 'features')
    
    # Create datasets
    train_dataset = UnifiedDataset(
        dataset_config=dataset_config,
        split='train',
        feature_cache_dir=feature_cache_dir
    )
    
    val_dataset = UnifiedDataset(
        dataset_config=dataset_config,
        split='val',
        feature_cache_dir=feature_cache_dir
    )
    
    # Exit if dry run
    if dry_run:
        logger.info("Dry run complete. Exiting without training.")
        return
    
    # Initialize wandb
    wandb.init(
        project=wandb_project,
        name=f"{model_type}-{dataset_config}-{model_config}",
        config={
            "model_type": model_type,
            "model_config": model_config,
            "dataset_config": dataset_config,
            "hidden_dims": hidden_dims,
            "dropout": dropout,
            "learning_rate": 0.001,
            "weight_decay": 0.01,
            "batch_size": 16,
            "epochs": 50,
            "early_stopping_patience": 5
        }
    )
    
    # Log dataset statistics to wandb
    train_df = pd.DataFrame({'split': ['train'] * len(train_dataset), 'label': [y for _, y in train_dataset]})
    val_df = pd.DataFrame({'split': ['val'] * len(val_dataset), 'label': [y for _, y in val_dataset]})
    test_df = pd.DataFrame({'split': ['test'] * 0, 'label': []})  # Placeholder for test set
    
    # Get number of classes
    num_classes = len(set(train_df['label']))
    
    # Get class names
    class_names = get_class_names(dataset_type)
    if len(class_names) < num_classes:
        class_names = [f'class_{i}' for i in range(num_classes)]
    
    # Log dataset statistics
    log_dataset_statistics(
        train_df, val_df, test_df, 
        'label', dataset_config, 
        output_dir=os.path.join(log_dir, 'dataset_stats'),
        log_wandb=True
    )
    
    # DataLoader configuration
    batch_size = 16
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Create model
    model = create_model(model_type, model_config, num_classes)
    model = model.to(device)
    
    # Setup mixed precision
    scaler = torch.cuda.amp.GradScaler()
    
    # Compute class weights for loss function
    label_counts = train_df['label'].value_counts().sort_index()
    if len(label_counts) != num_classes:
        temp_counts = pd.Series(0, index=range(num_classes))
        for idx, count in label_counts.items():
            temp_counts[idx] = count
        label_counts = temp_counts
    total_samples = len(train_dataset)
    class_weights = torch.tensor([
        total_samples / (num_classes * count) for count in label_counts
    ], dtype=torch.float32).to(device)
    logger.info(f"Class weights: {class_weights.cpu().numpy()}")
    
    # Setup loss and optimizer based on dataset type
    if dataset_type in ['benign_malignant', 'lesion_type']:
        # Use focal loss for imbalanced datasets
        criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    else:
        # Use weighted cross-entropy for balanced datasets
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Learning rate scheduler - reduce on plateau for better convergence
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Add wandb model watching
    wandb.watch(model, log="all", log_freq=10)
    
    # Setup paths for saving models and results
    model_save_dir = project_root / 'saved_models' / model_type
    model_save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_save_dir / f"{dataset_config}_{model_config}_best.pth"
    
    confusion_matrix_dir = os.path.join(
        config['visualization']['save_dir'], 
        config['visualization']['subdirs']['confusion_matrices'],
        model_type
    )
    os.makedirs(confusion_matrix_dir, exist_ok=True)
    
    # Training loop
    best_val_acc = 0
    best_val_f1 = 0
    patience_counter = 0
    early_stopping_patience = 5
    start_time = time.time()
    
    for epoch in range(50):  # Maximum 50 epochs
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
                
                # Forward pass with mixed precision
                with torch.amp.autocast('cuda'):
                    outputs = model(features)
                    loss = criterion(outputs, targets)
                
                # Backward pass with gradient scaling
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
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': train_loss/(batch_idx+1),
                    'Acc': 100. * train_correct / train_total
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
                
                # Forward pass with mixed precision
                with torch.amp.autocast('cuda'):
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
        
        # Log metrics
        log_metrics(
            metrics_file, epoch,
            train_loss, train_acc, train_f1,
            val_loss, val_acc, val_f1,
            optimizer.param_groups[0]['lr'],
            val_precision, val_recall
        )
        
        # Log to wandb
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_f1': train_f1,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_f1': val_f1,
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        
        # Calculate confusion matrix
        cm = confusion_matrix(all_val_targets, all_val_preds)
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
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
        
        # Save checkpoint if best validation accuracy
        # Save checkpoint if best validation metrics (try accuracy first, then F1)
        is_best = False
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            is_best = True
            logger.info(f"New best model (accuracy): {val_acc:.2f}%")
        elif val_acc == best_val_acc and val_f1 > best_val_f1:
            best_val_f1 = val_f1
            is_best = True
            logger.info(f"New best model (F1 score): {val_f1:.4f}")
        
        if is_best:
            # Save model checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'train_f1': train_f1,
                'train_loss': train_loss,
                'class_names': class_names,
                'dataset_config': dataset_config,
                'model_config': model_config
            }, checkpoint_path)
            logger.info(f'Saved best model checkpoint to {checkpoint_path}')
            
            # Save final confusion matrix to results directory
            final_cm_path = os.path.join(confusion_matrix_dir, f"{dataset_config}_{model_config}_cm.png")
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                      xticklabels=class_names[:num_classes], 
                      yticklabels=class_names[:num_classes])
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'{model_type.upper()} - {dataset_config} - {model_config}\nConfusion Matrix (Epoch {epoch})')
            plt.savefig(final_cm_path, bbox_inches='tight')
            plt.close()
            
            # Reset patience counter
            patience_counter = 0
        else:
            # Increment patience counter
            patience_counter += 1
        
        # Early stopping check
        if patience_counter >= early_stopping_patience:
            logger.info(f'Early stopping triggered after {epoch+1} epochs')
            break
        
        # Clean memory after each epoch
        gc.collect()
        torch.cuda.empty_cache()
    
    # Log total training time
    total_time = time.time() - start_time
    logger.info(f'\nTotal training time: {total_time/3600:.2f} hours')
    logger.info(f'Best validation accuracy: {best_val_acc:.2f}%')
    logger.info(f'Best validation F1 score: {best_val_f1:.4f}')
    
    # Save final model
    final_checkpoint_path = model_save_dir / f"{dataset_config}_{model_config}_final.pth"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'val_f1': val_f1,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'train_f1': train_f1,
        'train_loss': train_loss,
        'class_names': class_names,
        'dataset_config': dataset_config,
        'model_config': model_config
    }, final_checkpoint_path)
    logger.info(f'Saved final model checkpoint to {final_checkpoint_path}')
    
    # Finish up wandb
    wandb.finish()
    logger.info(f"Training completed for {model_type} with {dataset_config}_{model_config}!")
    
    return {
        'best_val_acc': best_val_acc,
        'best_val_f1': best_val_f1,
        'total_time': total_time,
        'model_path': str(checkpoint_path)
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MLP models with different configurations')
    
    parser.add_argument('--model_config', type=str, default='256_512_256_DO03',
                        choices=['256_512_256_DO03', '128_64_16_DO01', '64_16_DO01', '512_256_DO03', '512_256_DO01'],
                        help='Model architecture configuration')
    
    parser.add_argument('--dataset_config', type=str, required=True,
                        choices=list(DATASET_CONFIGS.keys()),
                        help='Dataset configuration to use')
    
    parser.add_argument('--dry_run', action='store_true',
                        help='Preview dataset without training')

    parser.add_argument('--wandb_project', type=str, default='skin-lesion-classification',
                        help='wandb project name to log runs to')
    
    args = parser.parse_args()
    
    # Run training or preview
    train(
        model_config=args.model_config,
        dataset_config=args.dataset_config,
        dry_run=args.dry_run,
        wandb_project=args.wandb_project
    ) 