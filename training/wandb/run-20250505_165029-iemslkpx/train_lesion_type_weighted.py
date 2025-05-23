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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set environment variables for better GPU memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from datasets.skin_dataset import SkinDataset
from models.lesion_type_head import LesionTypeClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from config.yaml."""
    config_path = project_root / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def compute_class_weights(dataset):
    """Compute class weights based on class distribution."""
    class_counts = {}
    logger.info("Computing class weights...")
    
    # Initialize counts for all classes to 0
    for class_name in dataset.LESION_GROUP_MAP.keys():
        class_idx = dataset.LESION_GROUP_MAP[class_name]
        class_counts[class_idx] = 0
    
    # Create a DataLoader for batch processing
    loader = DataLoader(dataset, batch_size=1024, num_workers=0, shuffle=False)
    
    # Count samples in batches
    for _, targets in tqdm(loader, desc="Counting samples"):
        for target in targets:
            class_counts[target.item()] += 1
    
    # Calculate weights inversely proportional to class frequencies
    total_samples = sum(class_counts.values())
    weights = torch.zeros(len(dataset.LESION_GROUP_MAP))
    
    # Handle missing classes by setting their weight to 0
    for idx, count in class_counts.items():
        if count > 0:
            weights[idx] = total_samples / (len([c for c in class_counts.values() if c > 0]) * count)
        else:
            weights[idx] = 0.0
            logger.warning(f"Class {list(dataset.LESION_GROUP_MAP.keys())[idx]} has no samples, setting weight to 0")
    
    return weights

def create_class_index_mapping(class_weights):
    """
    Create a mapping from original class indices to new consecutive indices.
    
    Args:
        class_weights: Tensor of class weights with zeros for missing classes
        
    Returns:
        original_to_new: Dictionary mapping original indices to new indices
        new_to_original: Dictionary mapping new indices to original indices
    """
    # Find indices of active classes (non-zero weights)
    active_classes = torch.nonzero(class_weights).squeeze().tolist()
    if not isinstance(active_classes, list):
        active_classes = [active_classes]
    
    # Create mappings
    original_to_new = {orig_idx: new_idx for new_idx, orig_idx in enumerate(active_classes)}
    new_to_original = {new_idx: orig_idx for new_idx, orig_idx in enumerate(active_classes)}
    
    return original_to_new, new_to_original

def log_class_metrics(outputs, targets, class_names, prefix=''):
    """Log per-class accuracy and other metrics."""
    with torch.no_grad():
        _, predicted = outputs.max(1)
        
        # Compute per-class metrics
        metrics = {}
        
        # Create confusion matrix
        conf_matrix = torch.zeros(len(class_names), len(class_names))
        for t, p in zip(targets.view(-1), predicted.view(-1)):
            conf_matrix[t.long(), p.long()] += 1
            
        # Log confusion matrix
        if prefix.startswith('val_'):
            fig = plt.figure(figsize=(10, 10))
            sns.heatmap(conf_matrix.cpu().numpy(), 
                       xticklabels=class_names,
                       yticklabels=class_names,
                       annot=True, fmt='g')
            plt.title('Confusion Matrix')
            metrics[f'{prefix}confusion_matrix'] = wandb.Image(fig)
            plt.close()
        
        for i, class_name in enumerate(class_names):
            mask = targets == i
            if mask.sum() > 0:  # Only compute metrics if class exists
                class_correct = predicted[mask].eq(targets[mask]).sum().item()
                class_total = mask.sum().item()
                class_acc = 100. * class_correct / class_total
                
                # Calculate precision and recall
                true_positives = conf_matrix[i, i].item()
                false_positives = conf_matrix[:, i].sum().item() - true_positives
                false_negatives = conf_matrix[i, :].sum().item() - true_positives
                
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                metrics[f'{prefix}acc_class_{class_name}'] = class_acc
                metrics[f'{prefix}precision_class_{class_name}'] = precision * 100
                metrics[f'{prefix}recall_class_{class_name}'] = recall * 100
                metrics[f'{prefix}f1_class_{class_name}'] = f1 * 100
                metrics[f'{prefix}samples_class_{class_name}'] = class_total
        
        return metrics

def main():
    # Clear GPU memory
    torch.cuda.empty_cache()
    gc.collect()
    
    # Load config
    config = load_config()
    train_config = config['training']['lesion_type']
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Initialize wandb with more config details
    run_name = "lesion-type-classifier-weighted"
    wandb.init(
        project="skin-lesion-classification",
        name=run_name,
        config={
            **train_config,
            'architecture': 'MLP2',
            'feature_extractor': 'SAM-ViT-H (frozen)',
            'optimizer': 'AdamW',
            'scheduler': 'OneCycleLR',
            'sampling_strategy': 'weighted_full'
        }
    )
    
    # Create datasets without balanced sampling
    logger.info("Creating datasets with full data and class weighting")
    
    train_dataset = SkinDataset(
        split='train',
        use_cache=True,
        subset_size=None,  # Use all available data
        skin_only=True,
        target_type='lesion_type'
    )
    
    val_dataset = SkinDataset(
        split='val',
        use_cache=True,
        subset_size=None,  # Use all available data
        skin_only=True,
        target_type='lesion_type'
    )
    
    # DataLoader configuration - single process mode
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config['batch_size'],
        shuffle=True,
        num_workers=0,  # No multiprocessing
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config['batch_size'] * 2,
        shuffle=False,
        num_workers=0,  # No multiprocessing
        pin_memory=True
    )
    
    # Create model
    model = LesionTypeClassifier().to(device)
    
    # Enable wandb model tracking
    wandb.watch(model, log='all', log_freq=100)
    
    # Setup mixed precision training
    scaler = torch.amp.GradScaler()
    
    # Compute class weights for loss function
    class_weights = compute_class_weights(train_dataset).to(device)
    logger.info("Class weights for loss function:")
    for i, weight in enumerate(class_weights):
        class_name = list(train_dataset.LESION_GROUP_MAP.keys())[i]
        logger.info(f"{class_name}: {weight:.4f}")
    
    # Create mapping from original class indices to new consecutive indices
    original_to_new_idx, new_to_original_idx = create_class_index_mapping(class_weights)
    
    # Log the index mapping
    logger.info("Class index mapping (original → new):")
    for orig_idx, new_idx in original_to_new_idx.items():
        class_name = list(train_dataset.LESION_GROUP_MAP.keys())[orig_idx]
        logger.info(f"Original index {orig_idx} ({class_name}) → New index {new_idx}")
    
    # Remove classes with zero weight from the model's output
    active_classes = torch.nonzero(class_weights).squeeze().tolist()
    if not isinstance(active_classes, list):
        active_classes = [active_classes]
    
    logger.info(f"Training with {len(active_classes)} active classes")
    
    # Modify model's output layer for active classes only
    model.fc2 = nn.Linear(model.fc2.in_features, len(active_classes)).to(device)
    
    # Extract weights for active classes only (in the new order)
    filtered_weights = torch.zeros(len(active_classes), device=device)
    for orig_idx, new_idx in original_to_new_idx.items():
        filtered_weights[new_idx] = class_weights[orig_idx]
    
    # Setup loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=filtered_weights)
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
    
    # Get class names for logging (only active classes in the new order)
    class_names = []
    for new_idx in range(len(active_classes)):
        orig_idx = new_to_original_idx[new_idx]
        class_name = list(train_dataset.LESION_GROUP_MAP.keys())[orig_idx]
        class_names.append(class_name)
    
    # Training loop
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(train_config['epochs']):
        # Train
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Track epoch-level metrics
        train_outputs = []
        train_targets = []
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch_idx, (features, targets) in enumerate(progress_bar):
            try:
                # Move data to device
                features = features.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                # Remap targets to new consecutive indices
                remapped_targets = torch.tensor([original_to_new_idx.get(t.item(), 0) for t in targets], 
                                             device=device, dtype=torch.long)
                
                # Mixed precision forward pass
                with torch.amp.autocast('cuda'):
                    outputs = model(features)
                    loss = criterion(outputs, remapped_targets)
                
                # Mixed precision backward pass
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                # Update learning rate
                scheduler.step()
                
                # Track outputs and targets for epoch-level metrics
                train_outputs.append(outputs.detach())
                train_targets.append(remapped_targets.detach())
                
                # Calculate accuracy
                with torch.no_grad():
                    _, predicted = outputs.max(1)
                    total += remapped_targets.size(0)
                    correct += predicted.eq(remapped_targets).sum().item()
                
                # Update total loss
                total_loss += loss.item()
                
                # Log batch metrics
                metrics = {
                    'train_batch_loss': loss.item(),
                    'train_batch_acc': 100.*correct/total,
                    'learning_rate': optimizer.param_groups[0]['lr']
                }
                wandb.log(metrics, commit=False)  # Don't commit yet, wait for class metrics
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': total_loss/(batch_idx+1),
                    'Acc': 100.*correct/total
                })
                
                # Clean up GPU memory
                del features, outputs, loss
                torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Error in training batch {batch_idx}: {str(e)}")
                continue
        
        # Compute and log epoch-level training metrics
        train_outputs = torch.cat(train_outputs)
        train_targets = torch.cat(train_targets)
        train_metrics = {
            'train_epoch_loss': total_loss/len(train_loader),
            'train_epoch_acc': 100.*correct/total,
            'epoch': epoch
        }
        train_metrics.update(log_class_metrics(train_outputs, train_targets, class_names, prefix='train_epoch_'))
        wandb.log(train_metrics, commit=False)  # Don't commit yet, wait for validation metrics
        
        # Clean up train metrics
        del train_outputs, train_targets
        torch.cuda.empty_cache()
        
        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        # Track per-class metrics
        val_outputs = []
        val_targets = []
        
        with torch.no_grad():
            for features, targets in tqdm(val_loader, desc='Validation'):
                features = features.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                # Remap targets to new consecutive indices
                remapped_targets = torch.tensor([original_to_new_idx.get(t.item(), 0) for t in targets], 
                                             device=device, dtype=torch.long)
                
                with torch.amp.autocast('cuda'):
                    outputs = model(features)
                    loss = criterion(outputs, remapped_targets)
                
                val_outputs.append(outputs)
                val_targets.append(remapped_targets)
                
                _, predicted = outputs.max(1)
                val_total += remapped_targets.size(0)
                val_correct += predicted.eq(remapped_targets).sum().item()
                val_loss += loss.item()
                
                del features, outputs, loss
                torch.cuda.empty_cache()
        
        # Compute validation metrics
        val_outputs = torch.cat(val_outputs)
        val_targets = torch.cat(val_targets)
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Log validation metrics
        val_metrics = {
            'val_epoch_loss': val_loss,
            'val_epoch_acc': val_acc,
            'epoch': epoch
        }
        val_metrics.update(log_class_metrics(val_outputs, val_targets, class_names, prefix='val_epoch_'))
        wandb.log({**train_metrics, **val_metrics}, commit=True)  # Now commit all metrics
        
        # Save checkpoint if best validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            checkpoint_path = os.path.join(
                train_config['checkpoint_dir'],
                'lesion_type_weighted_best.pth'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_metrics': val_metrics,
                'class_mapping': {
                    'original_to_new': original_to_new_idx,
                    'new_to_original': new_to_original_idx,
                    'class_names': class_names
                }
            }, checkpoint_path)
            logger.info(f'Saved best model checkpoint to {checkpoint_path}')
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= train_config['lr_patience']:
            logger.info(f'Early stopping triggered after {patience_counter} epochs without improvement')
            break
        
        # Log epoch metrics
        logger.info(f'Epoch {epoch}: '
                   f'Train Loss: {total_loss/len(train_loader):.4f} '
                   f'Train Acc: {100.*correct/total:.2f}% '
                   f'Val Loss: {val_loss:.4f} '
                   f'Val Acc: {val_acc:.2f}%')
        
        # Log per-class metrics
        logger.info('Per-class validation accuracy:')
        for i, class_name in enumerate(class_names):
            acc = val_metrics.get(f'val_epoch_acc_class_{class_name}', float('nan'))
            samples = val_metrics.get(f'val_epoch_samples_class_{class_name}', 0)
            logger.info(f"{class_name}: {acc:.2f}% ({samples} samples)")
        
        # Clear memory after each epoch
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main() 