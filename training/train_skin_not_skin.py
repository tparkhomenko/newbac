import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import yaml
from pathlib import Path
import logging
from tqdm import tqdm
import sys
import gc
import csv
import time
from datetime import datetime
import argparse

# Set environment variables for better GPU memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from datasets.skin_dataset import SkinDataset
from models.skin_not_skin_head import SkinNotSkinClassifier
from sam.sam_encoder import SAMFeatureExtractor

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

def setup_logging(config, model_config):
    """Setup logging directories and files."""
    # Create log directories if they don't exist
    log_dir = project_root / 'logs' / 'skin_not_skin'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Log file specific to model configuration
    metrics_file = log_dir / f"{model_config}_metrics.csv"
    
    # Setup logging to file
    log_file = log_dir / f"{model_config}_training.log"
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
            'val_loss', 'val_acc', 'val_f1', 'learning_rate'
        ])
    
    return log_dir, metrics_file

def log_metrics(metrics_file, epoch, train_loss, train_acc, train_f1=None,
                val_loss=None, val_acc=None, val_f1=None, learning_rate=None):
    """Log metrics to CSV file."""
    with open(metrics_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch, train_loss, train_acc, train_f1 if train_f1 is not None else '',
            val_loss if val_loss is not None else '',
            val_acc if val_acc is not None else '',
            val_f1 if val_f1 is not None else '',
            learning_rate if learning_rate is not None else ''
        ])

def load_config():
    """Load configuration from config.yaml."""
    config_path = project_root / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main(model_config="256_512_256_DO03"):
    # Clear GPU memory
    torch.cuda.empty_cache()
    gc.collect()
    
    # Load config
    config = load_config()
    train_config = config['training']['skin_not_skin']
    
    # Parse model configuration
    hidden_dims, dropout = parse_model_config(model_config)
    
    # Setup logging
    log_dir, metrics_file = setup_logging(config, model_config)
    logger = logging.getLogger(__name__)
    
    # Log training configuration
    logger.info("Training Configuration:")
    logger.info(f"Model Config: {model_config}")
    logger.info(f"Hidden Dimensions: {hidden_dims}")
    logger.info(f"Dropout Rate: {dropout}")
    for key, value in train_config.items():
        logger.info(f"{key}: {value}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Initialize wandb
    wandb.init(
        project="skin-lesion-classification",
        name=f"skin-not-skin-classifier-{model_config}",
        config={
            **train_config,
            'model_config': model_config,
            'hidden_dims': hidden_dims,
            'dropout': dropout
        }
    )
    
    # Create datasets with 2000 samples per class
    subset_size = 2000
    logger.info(f"Using subset of {subset_size} samples per class")
    
    train_dataset = SkinDataset(split='train', subset_size=subset_size)
    val_dataset = SkinDataset(split='val', subset_size=subset_size//5)  # 400 samples per class for validation
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    # DataLoader configuration - single process mode
    batch_size = 64
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # No multiprocessing
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=0,  # No multiprocessing
        pin_memory=True
    )
    
    # Create model with mixed precision support and specified architecture
    model = SkinNotSkinClassifier(hidden_dims=hidden_dims, dropout=dropout).to(device)
    scaler = torch.cuda.amp.GradScaler()
    
    # Setup loss and optimizer
    class_weights = torch.tensor(train_config['class_weights']).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_config['learning_rate'],
        weight_decay=train_config['weight_decay']
    )
    
    # Learning rate scheduler with warmup
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=train_config['learning_rate'],
        epochs=train_config['epochs'],
        steps_per_epoch=len(train_loader),
        pct_start=0.1
    )
    
    # Create directories for saving models and results
    model_save_dir = project_root / 'saved_models' / 'skin_not_skin'
    model_save_dir.mkdir(parents=True, exist_ok=True)
    
    confusion_matrix_dir = project_root / 'results' / 'confusion_matrices' / 'skin_not_skin'
    confusion_matrix_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    best_val_acc = 0
    patience_counter = 0
    early_stopping_patience = 5
    start_time = time.time()
    
    for epoch in range(train_config['epochs']):
        epoch_start_time = time.time()
        
        # Train with mixed precision
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
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
                
                # Mixed precision backward pass
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                # Update learning rate
                scheduler.step()
                
                # Calculate accuracy
                with torch.no_grad():
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                
                # Update total loss
                total_loss += loss.item()
                
                # Update progress bar
                batch_acc = 100. * correct / total
                progress_bar.set_postfix({
                    'Loss': total_loss/(batch_idx+1),
                    'Acc': batch_acc
                })
                
                # Clean up GPU memory
                del features, targets, outputs, loss
                torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Error in training batch {batch_idx}: {str(e)}")
                continue
        
        # Calculate epoch metrics
        train_loss = total_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
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
                
                del features, targets, outputs, loss
                torch.cuda.empty_cache()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Log epoch metrics
        log_metrics(
            metrics_file, epoch,
            train_loss, train_acc,
            val_loss=val_loss, val_acc=val_acc,
            learning_rate=optimizer.param_groups[0]['lr']
        )
        
        # Log to wandb
        wandb.log({
            'epoch': epoch,
            'train_epoch_loss': train_loss,
            'train_epoch_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Log epoch summary
        logger.info(
            f'Epoch {epoch} Summary:\n'
            f'Time: {epoch_time:.2f}s\n'
            f'Train Loss: {train_loss:.4f}\n'
            f'Train Acc: {train_acc:.2f}%\n'
            f'Val Loss: {val_loss:.4f}\n'
            f'Val Acc: {val_acc:.2f}%\n'
            f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}'
        )
        
        # Save checkpoint if best validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = model_save_dir / f'{model_config}_best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'train_loss': train_loss,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'model_config': model_config
            }, checkpoint_path)
            logger.info(f'Saved best model checkpoint to {checkpoint_path}')
            
            # Reset early stopping counter
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Early stopping check
        if patience_counter >= early_stopping_patience:
            logger.info(f'Early stopping triggered after {epoch+1} epochs')
            break
        
        # Clear memory after each epoch
        gc.collect()
        torch.cuda.empty_cache()
    
    # Log total training time
    total_time = time.time() - start_time
    logger.info(f'\nTotal training time: {total_time/3600:.2f} hours')
    logger.info(f'Best validation accuracy: {best_val_acc:.2f}%')
    
    # Save final model
    final_checkpoint_path = model_save_dir / f'{model_config}_final.pth'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'train_loss': train_loss,
        'learning_rate': optimizer.param_groups[0]['lr'],
        'model_config': model_config
    }, final_checkpoint_path)
    logger.info(f'Saved final model checkpoint to {final_checkpoint_path}')
    
    wandb.finish()
    logger.info(f"Training completed for model config: {model_config}!")

if __name__ == '__main__':
    # Add command-line arguments
    parser = argparse.ArgumentParser(description='Train skin/not-skin classifier')
    parser.add_argument('--model_config', type=str, default='256_512_256_DO03',
                        choices=['256_512_256_DO03', '128_64_16_DO01', '64_16_DO01'],
                        help='Model architecture configuration')
    args = parser.parse_args()
    
    main(model_config=args.model_config) 