import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import wandb
import yaml
from pathlib import Path
import logging
from tqdm import tqdm
import sys
import gc
import numpy as np
from torch.amp import autocast, GradScaler
import pandas as pd
from tabulate import tabulate  # pip install tabulate
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import csv
import time
from datetime import datetime

# Set environment variables for better GPU memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from datasets.skin_dataset import SkinLesionDataset
from models.lesion_type_head import LesionTypeHead

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
    
def log_class_distribution(title, class_counts):
    table = [(k, v) for k, v in class_counts.items()]
    logger.info(f"\n{'='*40}\n{title}\n{'='*40}\n"
                f"Total images: {class_counts.sum()}\n"
                + tabulate(table, headers=['Lesion Group', 'Count'], tablefmt='github') + "\n")

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

# If not installed, you may need to implement FocalLoss below
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction='none')
    def forward(self, input, target):
        ce_loss = self.ce(input, target)
        pt = torch.exp(-ce_loss)
        if isinstance(self.alpha, torch.Tensor):
            at = self.alpha.gather(0, target)
        else:
            at = self.alpha
        focal_loss = at * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        else:
            return focal_loss.sum()

def setup_logging(config, model_name="lesion_type"):
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

def train_model():
    # Load configuration
    config = load_config()
    training_config = config['training']
    
    # Set device
    device = torch.device(training_config['device'])
    
    # Set random seed
    torch.manual_seed(training_config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(training_config['seed'])
    
    # Get max samples per class from config
    MAX_SAMPLES_PER_CLASS = training_config.get('max_samples_per_class', 2000)
    logger.info(f"Setting maximum samples per class to {MAX_SAMPLES_PER_CLASS}")
    
    # Initialize wandb with run name that includes max samples per class
    run_name = f"lesion_type_max{MAX_SAMPLES_PER_CLASS}"
    
    wandb.init(
        project=training_config['wandb']['project'],
        name=run_name,
        tags=training_config['wandb']['tags'] + [f"max{MAX_SAMPLES_PER_CLASS}"],
        config={
            **training_config,
            'max_samples_per_class': MAX_SAMPLES_PER_CLASS,
            'class_weights': training_config.get('class_weights', [0.49, 0.49, 0.49, 1.70, 1.84])
        }
    )
    
    # Set up logging level temporarily to suppress misleading dataset statistics
    datasets_logger = logging.getLogger('datasets.skin_dataset')
    original_level = datasets_logger.level
    datasets_logger.setLevel(logging.WARNING)  # Suppress INFO logs
    
    # Balanced dataset: all samples from minority classes, random samples from majority classes
    minority_classes = ['fibrous', 'vascular']
    majority_classes = ['melanocytic', 'non-melanocytic carcinoma', 'keratosis']
    full_dataset = SkinLesionDataset(
        metadata_path=training_config['metadata_path'],
        feature_cache_dir=training_config['train_features_dir'],
        skin_only=True,
        original_only=False,  # Use augmentations for rare classes
        subset_fraction=None,
        per_class_fraction=None
    )
    
    # Restore original logging level
    datasets_logger.setLevel(original_level)
    
    # Target counts for each class - use max_samples_per_class from config
    target_counts = {
        'melanocytic': MAX_SAMPLES_PER_CLASS,
        'non-melanocytic carcinoma': MAX_SAMPLES_PER_CLASS,
        'keratosis': MAX_SAMPLES_PER_CLASS,
        'fibrous': 1650,   # all
        'vascular': 1785   # all
    }
    
    logger.info(f"Using target counts: {target_counts}")
    
    # Build training set
    sampled_metadata = []
    for group, target in target_counts.items():
        group_df = full_dataset.metadata[full_dataset.metadata['lesion_group'] == group]
        
        # Store original count
        original_count = len(group_df)
        
        # For all classes, apply max_samples_per_class limit
        if len(group_df) > target:
            group_df = group_df.sample(n=target, random_state=42)
            logger.info(f"Class {group}: FORCE downsampled from {original_count} to {len(group_df)} samples")
        else:
            logger.info(f"Class {group}: Using all {len(group_df)} samples")
        
        sampled_metadata.append(group_df)
    train_metadata = pd.concat(sampled_metadata).sample(frac=1, random_state=42).reset_index(drop=True)
    # Create more comprehensive dataset statistics
    print(f"\n--- Comprehensive Dataset Statistics (max_samples_per_class={MAX_SAMPLES_PER_CLASS}) ---")
    
    # Call the generate_stats function
    generate_dataset_statistics()
    print(f"--- End of Statistics ---\n")
    full_dataset.metadata = train_metadata
    train_dataset = full_dataset
    
    # Create weights dictionary for mapping to sample_weights later
    weights_dict = {}
    
    # Class weights - either from config or computed from dataset
    if 'class_weights' in training_config:
        # Use weights from config
        config_weights = training_config['class_weights']
        logger.info(f"Using class weights from config: {config_weights}")
        class_weights_tensor = torch.tensor(config_weights, dtype=torch.float, device=device)
        
        # Create mapping for sample weights
        for i, group in enumerate(target_counts.keys()):
            weights_dict[group] = config_weights[i]
    else:
        # Compute weights from dataset
        class_counts = train_dataset.metadata['lesion_group'].value_counts().reindex(list(target_counts.keys()), fill_value=0)
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights = class_weights / class_weights.sum() * len(class_counts)
        class_weights_tensor = torch.tensor(class_weights.values, dtype=torch.float, device=device)
        logger.info(f"Computed class weights from dataset: {class_weights.values.tolist()}")
        
        # Create mapping for sample weights
        for i, group in enumerate(target_counts.keys()):
            weights_dict[group] = class_weights[group]
    
    # Create mapping from original class indices to new consecutive indices
    original_to_new_idx, new_to_original_idx = create_class_index_mapping(class_weights_tensor)
    
    # Log the index mapping
    logger.info("Class index mapping (original → new):")
    for orig_idx, new_idx in original_to_new_idx.items():
        class_name = list(target_counts.keys())[orig_idx]
        logger.info(f"Original index {orig_idx} ({class_name}) → New index {new_idx}")
    
    # Oversample rare classes using WeightedRandomSampler
    sample_weights = train_dataset.metadata['lesion_group'].map(weights_dict).values
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        sampler=sampler,
        num_workers=0,
        pin_memory=training_config['pin_memory']
    )
    
    # Get active classes (non-zero weights)
    active_classes = torch.nonzero(class_weights_tensor).squeeze().tolist()
    if not isinstance(active_classes, list):
        active_classes = [active_classes]
    
    # Create filtered weights for active classes
    filtered_weights = torch.zeros(len(active_classes), device=device)
    for orig_idx, new_idx in original_to_new_idx.items():
        filtered_weights[new_idx] = class_weights_tensor[orig_idx]
    
    # Use FocalLoss with class weights
    criterion = FocalLoss(alpha=filtered_weights, gamma=2)
    
    # Create model with the correct number of output classes
    model = LesionTypeHead(
        input_dim=256,
        hidden_dims=training_config['model']['hidden_dims'],
        output_dim=len(active_classes),  # Only active classes
        dropout=training_config['model']['dropout']
    ).to(device)
    
    # Get class names for active classes in the new order
    class_names = []
    for new_idx in range(len(active_classes)):
        orig_idx = new_to_original_idx[new_idx]
        class_name = list(target_counts.keys())[orig_idx]
        class_names.append(class_name)
    
    # Loss function and optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=training_config['learning_rate'],
        weight_decay=training_config['weight_decay']
    )
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()
    
    # After train_loader is created, define val_dataset and val_loader
    # Suppress dataset statistics again
    datasets_logger.setLevel(logging.WARNING)
    
    val_dataset = SkinLesionDataset(
        metadata_path=training_config.get('val_metadata_path', training_config['metadata_path']),
        feature_cache_dir=training_config['val_features_dir'],
        skin_only=True,
        original_only=True,
        subset_fraction=training_config['data_fraction'],
        per_class_fraction=None
    )
    
    # Restore original logging level
    datasets_logger.setLevel(original_level)
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=training_config['pin_memory']
    )
    
    # Setup logging
    log_dir, metrics_file = setup_logging(config, "lesion_type")
    logger = logging.getLogger(__name__)
    
    # Log training configuration
    logger.info("Training Configuration:")
    for key, value in training_config.items():
        logger.info(f"{key}: {value}")
    
    # Training loop
    best_val_acc = 0
    start_time = time.time()
    
    for epoch in range(training_config['epochs']):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Collect outputs and labels for per-class metrics
        train_outputs = []
        train_labels = []
        
        # Use tqdm for progress bar
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{training_config["epochs"]} [Train]')
        for batch_idx, (features, labels) in enumerate(train_pbar):
            features, labels = features.to(device), labels.to(device)
            
            # Remap labels to new consecutive indices
            remapped_labels = torch.tensor([original_to_new_idx.get(t.item(), 0) for t in labels], 
                                           device=device, dtype=torch.long)
            
            # Clean GPU memory periodically
            if batch_idx % 1000 == 0:
                clean_gpu_memory()
            
            optimizer.zero_grad()
            
            # Use mixed precision training
            with autocast('cuda'):
                outputs = model(features)
                loss = criterion(outputs, remapped_labels)
            # Scale gradients and optimize
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += remapped_labels.size(0)
            train_correct += predicted.eq(remapped_labels).sum().item()
            train_outputs.extend(predicted.cpu().numpy())
            train_labels.extend(remapped_labels.cpu().numpy())
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * train_correct / train_total:.2f}%'
            })
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Per-class metrics for training
        train_f1 = f1_score(train_labels, train_outputs, average=None)
        train_precision = precision_score(train_labels, train_outputs, average=None, zero_division=0)
        train_recall = recall_score(train_labels, train_outputs, average=None, zero_division=0)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_outputs = []
            val_labels = []
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{training_config["epochs"]} [Val]')
            for features, labels in val_pbar:
                features, labels = features.to(device), labels.to(device)
                
                # Remap labels to new consecutive indices
                remapped_labels = torch.tensor([original_to_new_idx.get(t.item(), 0) for t in labels], 
                                              device=device, dtype=torch.long)
                
                # Use mixed precision for validation too
                with autocast('cuda'):
                    outputs = model(features)
                    loss = criterion(outputs, remapped_labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += remapped_labels.size(0)
                val_correct += predicted.eq(remapped_labels).sum().item()
                val_outputs.extend(predicted.cpu().numpy())
                val_labels.extend(remapped_labels.cpu().numpy())
                
                # Update progress bar
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100. * val_correct / val_total:.2f}%'
                })
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Per-class metrics for validation
        val_f1 = f1_score(val_labels, val_outputs, average=None)
        val_precision = precision_score(val_labels, val_outputs, average=None, zero_division=0)
        val_recall = recall_score(val_labels, val_outputs, average=None, zero_division=0)
        val_cm = confusion_matrix(val_labels, val_outputs)
        
        # Log batch metrics
        batch_loss = loss.item()
        batch_acc = 100. * train_correct / train_total
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log metrics
        log_metrics(
            metrics_file, epoch, batch_idx,
            batch_loss, batch_acc,
            learning_rate=current_lr
        )
        
        # Calculate additional metrics
        val_f1 = f1_score(val_labels, val_outputs, average='weighted')
        val_precision = precision_score(val_labels, val_outputs, average='weighted')
        val_recall = recall_score(val_labels, val_outputs, average='weighted')
        
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
            f'Val Loss: {val_loss:.4f}\n'
            f'Val Acc: {val_acc:.2f}%\n'
            f'Val F1: {val_f1:.4f}\n'
            f'Val Precision: {val_precision:.4f}\n'
            f'Val Recall: {val_recall:.4f}\n'
            f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}'
        )
        
        # Save checkpoint if best validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = log_dir / f'lesion_type_best.pth'
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
        
        # Clean GPU memory after each epoch
        clean_gpu_memory()
    
    # Log total training time
    total_time = time.time() - start_time
    logger.info(f'\nTotal training time: {total_time/3600:.2f} hours')
    logger.info(f'Best validation accuracy: {best_val_acc:.2f}%')
    
    # Save final model
    final_checkpoint_path = log_dir / f'lesion_type_final.pth'
    torch.save({
        'epoch': training_config['epochs'] - 1,
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
    logger.info('Training completed!')

    # After training, evaluate on validation set
    all_labels = []
    all_preds = []
    model.eval()
    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)
            
            # Remap labels to new consecutive indices
            remapped_labels = torch.tensor([original_to_new_idx.get(t.item(), 0) for t in labels], 
                                          device=device, dtype=torch.long)
            
            outputs = model(features)
            _, predicted = outputs.max(1)
            all_labels.extend(remapped_labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)
    print("Per-class F1:", f1_score(all_labels, all_preds, average=None))
    print("Per-class Precision:", precision_score(all_labels, all_preds, average=None, zero_division=0))
    print("Per-class Recall:", recall_score(all_labels, all_preds, average=None, zero_division=0))
    # Document class-to-index mapping
    print("Class-to-index mapping:")
    for new_idx, class_name in enumerate(class_names):
        print(f"{new_idx}: {class_name}")

def generate_dataset_statistics():
    """
    Generate and print dataset statistics in a consistent format.
    This function replicates the logic from generate_stats.py to ensure consistent output.
    """
    # Load configuration
    config = load_config()
    MAX_SAMPLES_PER_CLASS = config.get('training', {}).get('max_samples_per_class', 2000)
    
    # For validation and test sets, we typically use a percentage of the training samples
    VAL_SAMPLES_PER_CLASS = int(MAX_SAMPLES_PER_CLASS * 0.15)  # 15% for validation
    TEST_SAMPLES_PER_CLASS = int(MAX_SAMPLES_PER_CLASS * 0.15)  # 15% for testing
    
    # Load full dataset
    df = pd.read_csv(project_root / 'datasets/metadata/unified_augmented.csv')
    
    # Define classes of interest - only skin lesion classes
    skin_classes = ['melanocytic', 'non-melanocytic carcinoma', 'keratosis', 'fibrous', 'vascular']
    
    # Print the table using fixed-width formatting - changed column order
    header = "| {:<25} | {:<11} | {:<9} | {:<16} | {:<9} | {:<9} | {:<9} |".format(
        "Class", "Total in DB", "Originals", "Used in Training", "Train", "Test", "Val"
    )
    separator = "| {:-<25} | {:-<11} | {:-<9} | {:-<16} | {:-<9} | {:-<9} | {:-<9} |".format(
        "", "", "", "", "", "", ""
    )
    
    print(header)
    print(separator)
    
    # Track totals
    total_db = 0
    total_originals = 0
    total_train_used = 0
    total_val_used = 0
    total_test_used = 0
    total_used = 0
    
    # Process skin lesion classes
    for lesion_class in skin_classes:
        # Total in database
        class_total = len(df[df['lesion_group'] == lesion_class])
        total_db += class_total
        
        # Original (non-augmented) images
        originals = len(df[(df['lesion_group'] == lesion_class) & 
                         (df['image'].str.contains('_original.jpg'))])
        total_originals += originals
        
        # Split counts - total available
        train_count_total = len(df[(df['lesion_group'] == lesion_class) & (df['split'] == 'train')])
        val_count_total = len(df[(df['lesion_group'] == lesion_class) & (df['split'] == 'val')])
        test_count_total = len(df[(df['lesion_group'] == lesion_class) & (df['split'] == 'test')])
        
        # Use the same logic as generate_stats.py
        # Currently used in training (based on configuration)
        used_train = min(train_count_total, MAX_SAMPLES_PER_CLASS)
        
        # For small classes, we use all available samples
        if train_count_total <= MAX_SAMPLES_PER_CLASS:
            used_val = val_count_total
            used_test = test_count_total
        else:
            # For larger classes, we use a proportion based on MAX_SAMPLES_PER_CLASS
            used_val = min(val_count_total, VAL_SAMPLES_PER_CLASS)
            used_test = min(test_count_total, TEST_SAMPLES_PER_CLASS)
        
        # Update totals
        total_train_used += used_train
        total_val_used += used_val
        total_test_used += used_test
        
        # For "Used in Training" column, use only the train count to match generate_stats.py
        total_used += used_train
        
        # Print the row with fixed width formatting using string format - changed column order
        row = "| {:<25} | {:<11} | {:<9} | {:<16} | {:<9} | {:<9} | {:<9} |".format(
            lesion_class, class_total, originals, used_train, used_train, used_test, used_val
        )
        print(row)
    
    # Print the separator and total row
    print(separator)
    total_row = "| {:<25} | {:<11} | {:<9} | {:<16} | {:<9} | {:<9} | {:<9} |".format(
        "TOTAL", total_db, total_originals, total_train_used, total_train_used, total_test_used, total_val_used
    )
    print(total_row)

if __name__ == "__main__":
    train_model() 