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
import pandas as pd
from sklearn.manifold import TSNE
from datetime import datetime

# Set environment variables for better GPU memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from datasets.skin_dataset import SkinLesionDataset
from models.lesion_type_head import LesionTypeHead
from utils.augmentation import cutmix, mixup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from config.yaml."""
    config_path = project_root / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def clean_gpu_memory():
    """Clean up GPU memory."""
    torch.cuda.empty_cache()
    gc.collect()

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

def plot_confusion_matrix(cm, class_names, epoch, save_dir):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    
    # Normalize by row (true labels)
    row_sums = cm.sum(axis=1)
    cm_norm = cm / row_sums[:, np.newaxis]
    
    # Plot
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
               xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Normalized Confusion Matrix - Epoch {epoch}')
    
    # Save
    save_path = os.path.join(save_dir, f'confusion_matrix_epoch_{epoch}.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    return save_path

def visualize_tsne(features, labels, class_names, epoch, save_dir):
    """Visualize feature space using t-SNE."""
    # Convert features to numpy array
    features_np = features.cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    features_tsne = tsne.fit_transform(features_np)
    
    # Plot
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        mask = labels_np == i
        if np.any(mask):
            plt.scatter(features_tsne[mask, 0], features_tsne[mask, 1], label=class_name, alpha=0.6)
    
    plt.legend()
    plt.title(f't-SNE Visualization - Epoch {epoch}')
    
    # Save
    save_path = os.path.join(save_dir, f'tsne_features_epoch_{epoch}.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    return save_path

class FocalLoss(nn.Module):
    """Focal Loss implementation with customizable gamma."""
    def __init__(self, alpha=None, gamma=3.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(weight=alpha, reduction='none')
    
    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def train_model():
    # Configuration
    config = load_config()
    training_config = config['training']
    
    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path(project_root) / "results" / f"lesion_type_exp2_{timestamp}"
    os.makedirs(result_dir, exist_ok=True)
    
    # Define max samples per class regardless of what's in the input files
    MAX_SAMPLES_PER_CLASS = 2000
    logger.info(f"Setting maximum samples per class to {MAX_SAMPLES_PER_CLASS}")
    
    # Device
    device = torch.device(training_config['device'])
    
    # Random seed
    torch.manual_seed(training_config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(training_config['seed'])
    
    # Initialize wandb
    run_name = f"lesion_type_max{MAX_SAMPLES_PER_CLASS}_{timestamp}"
    wandb.init(
        project=training_config['wandb']['project'],
        name=f"lesion_type_max{MAX_SAMPLES_PER_CLASS}_{timestamp}",
        tags=training_config['wandb']['tags'] + ["balanced", "focal_loss", "cutmix", "mixup", f"max{MAX_SAMPLES_PER_CLASS}"],
        config={
            **training_config,
            'dropout': 0.5,
            'gamma': 3.0,
            'weight_decay': 0.1,
            'balanced_data': True,
            'max_samples_per_class': MAX_SAMPLES_PER_CLASS,
            'augmentation': {
                'cutmix': True,
                'mixup': True,
                'cutmix_alpha': 1.0,
                'mixup_alpha': 0.2
            },
            'class_weights': [1.5, 1.5, 1.5, 1.2, 1.2]
        }
    )
    
    # Load full dataset to sample from
    full_dataset = SkinLesionDataset(
        metadata_path=training_config['metadata_path'],
        feature_cache_dir=training_config['train_features_dir'],
        skin_only=True,
        original_only=False,
        subset_fraction=None,
        per_class_fraction=None
    )
    
    # Force maximum 2000 samples per class regardless of what's in the input files
    # Create balanced dataset by sampling
    class_dfs = {}
    
    # Get all samples per class
    class_counts = {}
    for group in ['melanocytic', 'non-melanocytic carcinoma', 'keratosis', 'fibrous', 'vascular']:
        group_df = full_dataset.metadata[full_dataset.metadata['lesion_group'] == group]
        class_counts[group] = len(group_df)
        
        # Downsample to MAX_SAMPLES_PER_CLASS or use all available
        if len(group_df) > MAX_SAMPLES_PER_CLASS:
            class_dfs[group] = group_df.sample(n=MAX_SAMPLES_PER_CLASS, random_state=42)
            logger.info(f"Class {group}: Downsampled from {len(group_df)} to {MAX_SAMPLES_PER_CLASS} samples")
        else:
            class_dfs[group] = group_df
            logger.info(f"Class {group}: Using all {len(group_df)} samples")
    
    # Print a summary table
    logger.info("Class distribution summary:")
    logger.info("| Class | Original Count | Final Count |")
    logger.info("|-------|---------------|------------|")
    for group in ['melanocytic', 'non-melanocytic carcinoma', 'keratosis', 'fibrous', 'vascular']:
        logger.info(f"| {group} | {class_counts[group]} | {len(class_dfs[group])} |")
    
    # Combine into balanced dataset
    balanced_metadata = pd.concat(list(class_dfs.values())).sample(frac=1, random_state=42).reset_index(drop=True)
    logger.info(f"Created balanced dataset with {len(balanced_metadata)} samples")
    
    # Apply to dataset
    full_dataset.metadata = balanced_metadata
    train_dataset = full_dataset
    
    # Create DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,  # Increased batch size
        shuffle=True,
        num_workers=0,
        pin_memory=training_config['pin_memory']
    )
    
    # Class weights for loss function
    # More balanced weights across all classes, with slightly higher weights for larger classes
    manual_class_weights = torch.tensor([1.5, 1.5, 1.5, 1.2, 1.2], device=device, dtype=torch.float)
    
    # Create mapping from original class indices to new consecutive indices
    original_to_new_idx, new_to_original_idx = create_class_index_mapping(manual_class_weights)
    
    # Log the index mapping
    logger.info("Class index mapping (original → new):")
    for orig_idx, new_idx in original_to_new_idx.items():
        class_name = list(training_config['class_names'])[orig_idx]
        logger.info(f"Original index {orig_idx} ({class_name}) → New index {new_idx}")
    
    # Get active classes (non-zero weights)
    active_classes = torch.nonzero(manual_class_weights).squeeze().tolist()
    if not isinstance(active_classes, list):
        active_classes = [active_classes]
    
    # Create filtered weights for active classes
    filtered_weights = torch.zeros(len(active_classes), device=device)
    for orig_idx, new_idx in original_to_new_idx.items():
        filtered_weights[new_idx] = manual_class_weights[orig_idx]
    
    # Create model with higher dropout
    model = LesionTypeHead(
        input_dim=256,
        hidden_dims=training_config['model']['hidden_dims'],
        output_dim=len(active_classes),
        dropout=0.5  # Increased dropout
    ).to(device)
    
    # Get class names for active classes in the new order
    class_names = []
    for new_idx in range(len(active_classes)):
        orig_idx = new_to_original_idx[new_idx]
        class_name = list(training_config['class_names'])[orig_idx]
        class_names.append(class_name)
    
    # Enable wandb model tracking
    wandb.watch(model, log="all", log_freq=10)
    
    # Use FocalLoss with higher gamma and class weights
    criterion = FocalLoss(alpha=filtered_weights, gamma=3.0)
    
    # Optimizer with higher weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=training_config['learning_rate'],
        weight_decay=0.1  # Increased weight decay
    )
    
    # Learning rate scheduler with warmup
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=training_config['learning_rate'],
        epochs=training_config['num_epochs'],
        steps_per_epoch=len(train_loader),
        pct_start=0.1,  # 10% warmup
        div_factor=25,
        final_div_factor=1000
    )
    
    # Initialize gradient scaler for mixed precision training
    scaler = torch.amp.GradScaler()
    
    # Load validation dataset
    # For validation, use balanced subset with max 300 samples per class
    val_dataset = SkinLesionDataset(
        metadata_path=training_config.get('val_metadata_path', training_config['metadata_path']),
        feature_cache_dir=training_config['val_features_dir'],
        skin_only=True,
        original_only=True,
        subset_fraction=None
    )
    
    # Create balanced validation set
    val_class_dfs = {}
    VAL_SAMPLES_PER_CLASS = 300  # Use 300 samples per class for validation
    logger.info(f"Setting validation samples per class to {VAL_SAMPLES_PER_CLASS}")
    
    # Get validation counts
    val_class_counts = {}
    for group in ['melanocytic', 'non-melanocytic carcinoma', 'keratosis', 'fibrous', 'vascular']:
        group_df = val_dataset.metadata[val_dataset.metadata['lesion_group'] == group]
        val_class_counts[group] = len(group_df)
        
        # Downsample or use all available
        if len(group_df) > VAL_SAMPLES_PER_CLASS:
            val_class_dfs[group] = group_df.sample(n=VAL_SAMPLES_PER_CLASS, random_state=42)
            logger.info(f"Validation class {group}: Downsampled from {len(group_df)} to {VAL_SAMPLES_PER_CLASS} samples")
        else:
            val_class_dfs[group] = group_df
            logger.info(f"Validation class {group}: Using all {len(group_df)} samples")
    
    # Combine into balanced validation dataset
    val_balanced_metadata = pd.concat(list(val_class_dfs.values())).sample(frac=1, random_state=42).reset_index(drop=True)
    logger.info(f"Created balanced validation dataset with {len(val_balanced_metadata)} samples")
    
    # Apply to validation dataset
    val_dataset.metadata = val_balanced_metadata
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,  # Increased batch size
        shuffle=False,
        num_workers=0,
        pin_memory=training_config['pin_memory']
    )
    
    # Training loop
    num_epochs = training_config.get('debug_epochs', training_config['num_epochs']) if training_config.get('debug_epochs', 0) > 0 else training_config['num_epochs']
    patience = training_config.get('early_stopping_patience', 7)  # Increased patience
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Collect features and labels for t-SNE visualization
        all_features = []
        all_labels = []
        
        # Collect outputs and labels for per-class metrics
        train_outputs = []
        train_labels = []
        
        # Use tqdm for progress bar
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for features, labels in train_pbar:
            features, labels = features.to(device), labels.to(device)
            
            # Remap labels to new consecutive indices
            remapped_labels = torch.tensor([original_to_new_idx.get(t.item(), 0) for t in labels], 
                                         device=device, dtype=torch.long)
            
            # Apply CutMix or MixUp with 50% probability each
            r = np.random.rand()
            if r < 0.25 and epoch > 2:  # Apply CutMix after a few epochs
                features, remapped_labels_a, remapped_labels_b, lam = cutmix(
                    features, remapped_labels, alpha=1.0
                )
                cutmix_applied = True
            elif r < 0.5 and epoch > 2:  # Apply MixUp after a few epochs
                features, remapped_labels_a, remapped_labels_b, lam = mixup(
                    features, remapped_labels, alpha=0.2
                )
                mixup_applied = True
            else:
                cutmix_applied = False
                mixup_applied = False
            
            # Clean GPU memory periodically
            if train_total % 1000 == 0:
                clean_gpu_memory()
            
            optimizer.zero_grad()
            
            # Use mixed precision training
            with torch.amp.autocast('cuda'):
                outputs = model(features)
                
                # Apply loss according to augmentation
                if cutmix_applied or mixup_applied:
                    loss = lam * criterion(outputs, remapped_labels_a) + (1 - lam) * criterion(outputs, remapped_labels_b)
                else:
                    loss = criterion(outputs, remapped_labels)
            
            # Scale gradients and optimize
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Update learning rate
            scheduler.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += remapped_labels.size(0)
            
            # For accuracy, only count non-augmented samples
            if not cutmix_applied and not mixup_applied:
                train_correct += predicted.eq(remapped_labels).sum().item()
                train_outputs.extend(predicted.cpu().numpy())
                train_labels.extend(remapped_labels.cpu().numpy())
                
                # Store features and labels for t-SNE
                if epoch % 5 == 0 and len(all_features) < 1000:  # Limit to 1000 samples for memory
                    # Get penultimate layer features (before final classification layer)
                    with torch.no_grad():
                        # Extract the feature vector before the last layer
                        penultimate = model.model[:-1](features)
                        all_features.append(penultimate.detach())
                        all_labels.append(remapped_labels.detach())
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * train_correct / train_total:.2f}%',
                'lr': f'{scheduler.get_last_lr()[0]:.6f}'
            })
        
        # Calculate epoch statistics
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Create t-SNE visualization if we have features
        if all_features and all_labels:
            all_features = torch.cat(all_features)
            all_labels = torch.cat(all_labels)
            tsne_path = visualize_tsne(all_features, all_labels, class_names, epoch, result_dir)
            wandb.log({"tsne_visualization": wandb.Image(tsne_path)})
        
        # Per-class metrics for training
        if train_outputs and train_labels:
            # Convert to numpy arrays
            train_outputs = np.array(train_outputs)
            train_labels = np.array(train_labels)
            
            # Compute confusion matrix
            cm = np.zeros((len(class_names), len(class_names)))
            for t, p in zip(train_labels, train_outputs):
                cm[t][p] += 1
            
            # Per-class metrics
            train_f1 = []
            train_precision = []
            train_recall = []
            
            for i in range(len(class_names)):
                # True positives, false positives, false negatives
                tp = cm[i, i]
                fp = cm[:, i].sum() - tp
                fn = cm[i, :].sum() - tp
                
                # Precision, recall, F1
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                train_precision.append(precision)
                train_recall.append(recall)
                train_f1.append(f1)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # Collect for per-class metrics
        val_outputs = []
        val_labels = []
        
        # Collect features for t-SNE
        val_features = []
        val_label_list = []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for features, labels in val_pbar:
                features, labels = features.to(device), labels.to(device)
                
                # Remap labels to new consecutive indices
                remapped_labels = torch.tensor([original_to_new_idx.get(t.item(), -1) for t in labels], 
                                             device=device, dtype=torch.long)
                
                # Skip samples with unknown label mapping
                valid_mask = remapped_labels >= 0
                if not torch.all(valid_mask):
                    features = features[valid_mask]
                    remapped_labels = remapped_labels[valid_mask]
                    if len(remapped_labels) == 0:
                        continue
                
                # Use mixed precision for validation too
                with torch.amp.autocast('cuda'):
                    outputs = model(features)
                    loss = criterion(outputs, remapped_labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += remapped_labels.size(0)
                val_correct += predicted.eq(remapped_labels).sum().item()
                
                # Store outputs and labels for metrics
                val_outputs.extend(predicted.cpu().numpy())
                val_labels.extend(remapped_labels.cpu().numpy())
                
                # Store features for t-SNE
                if epoch % 5 == 0 and len(val_features) < 500:  # Limit to 500 samples
                    # Get penultimate layer features
                    penultimate = model.model[:-1](features)
                    val_features.append(penultimate)
                    val_label_list.append(remapped_labels)
                
                # Update progress bar
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100. * val_correct / val_total:.2f}%'
                })
        
        # Calculate validation statistics
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Create t-SNE visualization for validation features
        if val_features and val_label_list:
            val_features = torch.cat(val_features)
            val_label_list = torch.cat(val_label_list)
            val_tsne_path = visualize_tsne(val_features, val_label_list, class_names, epoch, result_dir)
            wandb.log({"val_tsne_visualization": wandb.Image(val_tsne_path)})
        
        # Per-class metrics for validation
        if val_outputs and val_labels:
            # Convert to numpy arrays
            val_outputs = np.array(val_outputs)
            val_labels = np.array(val_labels)
            
            # Compute confusion matrix
            val_cm = np.zeros((len(class_names), len(class_names)))
            for t, p in zip(val_labels, val_outputs):
                val_cm[t][p] += 1
            
            # Save confusion matrix visualization
            cm_path = plot_confusion_matrix(val_cm, class_names, epoch, result_dir)
            wandb.log({"confusion_matrix": wandb.Image(cm_path)})
            
            # Per-class metrics
            val_f1 = []
            val_precision = []
            val_recall = []
            
            for i in range(len(class_names)):
                # True positives, false positives, false negatives
                tp = val_cm[i, i]
                fp = val_cm[:, i].sum() - tp
                fn = val_cm[i, :].sum() - tp
                
                # Precision, recall, F1
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                val_precision.append(precision)
                val_recall.append(recall)
                val_f1.append(f1)
        
        # Log metrics to wandb
        metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'learning_rate': scheduler.get_last_lr()[0]
        }
        
        # Add per-class metrics
        for i, class_name in enumerate(class_names):
            if i < len(train_f1):
                metrics[f'train_f1_{class_name}'] = train_f1[i]
                metrics[f'train_precision_{class_name}'] = train_precision[i]
                metrics[f'train_recall_{class_name}'] = train_recall[i]
            
            if i < len(val_f1):
                metrics[f'val_f1_{class_name}'] = val_f1[i]
                metrics[f'val_precision_{class_name}'] = val_precision[i]
                metrics[f'val_recall_{class_name}'] = val_recall[i]
        
        wandb.log(metrics)
        
        # Log to console
        logger.info(f'Epoch [{epoch+1}/{num_epochs}]')
        logger.info(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        logger.info(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        
        if val_f1:
            logger.info(f'Val F1 Scores:')
            for i, class_name in enumerate(class_names):
                if i < len(val_f1):
                    logger.info(f'  {class_name}: {val_f1[i]:.4f}')
        
        # Print confusion matrix
        if val_outputs and val_labels:
            logger.info('Validation Confusion Matrix:')
            logger.info(val_cm)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'class_mapping': {
                    'original_to_new': original_to_new_idx,
                    'new_to_original': new_to_original_idx,
                    'class_names': class_names
                }
            }
            
            checkpoint_path = Path(training_config['model_save_dir']) / 'lesion_type_balanced_best.pth'
            torch.save(checkpoint, checkpoint_path)
            logger.info(f'Saved best model checkpoint to {checkpoint_path}')
            
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f'Early stopping triggered after {patience_counter} epochs without improvement')
                break
        
        # Clean GPU memory after each epoch
        clean_gpu_memory()
    
    wandb.finish()
    logger.info('Training completed!')
    
    # Save the final experiment details
    final_results = {
        'train_acc': train_acc,
        'val_acc': val_acc,
        'best_val_loss': best_val_loss,
        'epochs_trained': epoch + 1,
        'early_stopping_triggered': patience_counter >= patience,
        'class_mapping': {
            'original_to_new': original_to_new_idx,
            'new_to_original': new_to_original_idx,
            'class_names': class_names
        }
    }
    
    # Add per-class metrics
    if val_f1:
        for i, class_name in enumerate(class_names):
            if i < len(val_f1):
                final_results[f'val_f1_{class_name}'] = val_f1[i]
                final_results[f'val_precision_{class_name}'] = val_precision[i]
                final_results[f'val_recall_{class_name}'] = val_recall[i]
    
    # Save results
    result_path = Path(result_dir) / 'results.txt'
    with open(result_path, 'w') as f:
        f.write(f"Experiment completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Results:\n")
        for key, value in final_results.items():
            if not isinstance(value, dict):
                f.write(f"{key}: {value}\n")
        
        f.write("\nClass Mapping:\n")
        for orig_idx, new_idx in original_to_new_idx.items():
            class_name = list(training_config['class_names'])[orig_idx]
            f.write(f"Original index {orig_idx} ({class_name}) → New index {new_idx}\n")
        
        if val_outputs and val_labels:
            f.write("\nConfusion Matrix:\n")
            f.write(str(val_cm))
    
    logger.info(f"Final results saved to {result_path}")

if __name__ == "__main__":
    train_model() 