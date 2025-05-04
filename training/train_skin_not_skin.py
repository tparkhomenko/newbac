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

# Set environment variables for better GPU memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from datasets.skin_dataset import SkinDataset
from models.skin_not_skin_head import SkinNotSkinClassifier
from sam.sam_encoder import SAMFeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    for epoch in range(train_config['epochs']):
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
                
                # Log to wandb
                wandb.log({
                    'train_batch_loss': loss.item(),
                    'train_batch_acc': 100.*correct/total,
                    'learning_rate': optimizer.param_groups[0]['lr']
                })
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': total_loss/(batch_idx+1),
                    'Acc': 100.*correct/total
                })
                
                # Clean up GPU memory
                del features, targets, outputs, loss
                torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Error in training batch {batch_idx}: {str(e)}")
                continue
        
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
        
        # Log validation metrics
        wandb.log({
            'val_loss': val_loss,
            'val_acc': val_acc
        })
        
        # Save checkpoint if best validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(
                train_config['checkpoint_dir'],
                f'skin_not_skin_10k_best.pth'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, checkpoint_path)
            logger.info(f'Saved best model checkpoint to {checkpoint_path}')
        
        # Log epoch metrics
        logger.info(f'Epoch {epoch}: '
                   f'Train Loss: {total_loss/len(train_loader):.4f} '
                   f'Train Acc: {100.*correct/total:.2f}% '
                   f'Val Loss: {val_loss:.4f} '
                   f'Val Acc: {val_acc:.2f}%')
        
        # Clear memory after each epoch
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main() 