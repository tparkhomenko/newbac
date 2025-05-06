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

# Set environment variables for better GPU memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from datasets.skin_dataset import SkinDataset
from models.skin_not_skin_head import SkinNotSkinClassifier
from sam.sam_encoder import SAMFeatureExtractor

def setup_logging(config):
    """Setup logging directories and files."""
    # Create timestamp for unique run identification
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create log directories if they don't exist
    log_dir = project_root / 'logs' / 'skin_not_skin' / timestamp
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
            'val_loss', 'val_acc', 'learning_rate'
        ])
    
    return log_dir, metrics_file

def log_metrics(metrics_file, epoch, batch, train_loss, train_acc, 
                val_loss=None, val_acc=None, learning_rate=None):
    """Log metrics to CSV file."""
    with open(metrics_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch, batch, train_loss, train_acc, 
            val_loss if val_loss is not None else '',
            val_acc if val_acc is not None else '',
            learning_rate if learning_rate is not None else ''
        ])

def load_config():
    """Load configuration from config.yaml."""
    config_path = project_root / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # Clear GPU memory
    torch.cuda.empty_cache()
    gc.collect()
    
    # Load config
    config = load_config()
    train_config = config['training']['skin_not_skin']
    
    # Setup logging
    log_dir, metrics_file = setup_logging(config)
    logger = logging.getLogger(__name__)
    
    # Log training configuration
    logger.info("Training Configuration:")
    for key, value in train_config.items():
        logger.info(f"{key}: {value}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Initialize wandb
    wandb.init(
        project="skin-lesion-classification",
        name="skin-not-skin-classifier-2k",
        config=train_config
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
    
    # Create model with mixed precision support
    model = SkinNotSkinClassifier().to(device)
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
    
    # Training loop
    best_val_acc = 0
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
                
                # Mixed precision forward pass
                with torch.cuda.amp.autocast():
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
                
                # Calculate batch metrics
                batch_loss = loss.item()
                batch_acc = 100. * correct / total
                current_lr = optimizer.param_groups[0]['lr']
                
                # Log metrics
                log_metrics(
                    metrics_file, epoch, batch_idx,
                    batch_loss, batch_acc,
                    learning_rate=current_lr
                )
                
                # Log to wandb
                wandb.log({
                    'train_batch_loss': batch_loss,
                    'train_batch_acc': batch_acc,
                    'learning_rate': current_lr
                })
                
                # Update progress bar
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
                
                with torch.cuda.amp.autocast():
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
            metrics_file, epoch, 'end',
            train_loss, train_acc,
            val_loss, val_acc,
            optimizer.param_groups[0]['lr']
        )
        
        # Log to wandb
        wandb.log({
            'train_epoch_loss': train_loss,
            'train_epoch_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
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
            checkpoint_path = log_dir / f'skin_not_skin_best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'train_loss': train_loss,
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
    final_checkpoint_path = log_dir / f'skin_not_skin_final.pth'
    torch.save({
        'epoch': train_config['epochs'] - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'train_loss': train_loss,
        'learning_rate': optimizer.param_groups[0]['lr']
    }, final_checkpoint_path)
    logger.info(f'Saved final model checkpoint to {final_checkpoint_path}')

if __name__ == '__main__':
    main() 