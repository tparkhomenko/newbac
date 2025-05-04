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
    
    # Initialize wandb
    wandb.init(
        project=training_config['wandb']['project'],
        name=training_config['wandb']['name'],
        tags=training_config['wandb']['tags'],
        config=training_config
    )
    
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
    # Target counts for each class
    target_counts = {
        'melanocytic': 5000,
        'non-melanocytic carcinoma': 5000,
        'keratosis': 5000,
        'fibrous': 1650,   # all
        'vascular': 1785   # all
    }
    # Build training set
    sampled_metadata = []
    for group, target in target_counts.items():
        group_df = full_dataset.metadata[full_dataset.metadata['lesion_group'] == group]
        if group in ['fibrous', 'vascular']:
            # Use all available (augmented)
            sampled_metadata.append(group_df)
        else:
            # Downsample to 5,000 (random sample, original + augmented)
            if len(group_df) > target:
                group_df = group_df.sample(n=target, random_state=42)
            sampled_metadata.append(group_df)
    train_metadata = pd.concat(sampled_metadata).sample(frac=1, random_state=42).reset_index(drop=True)
    # Print class balancing table
    print("\n| Class                     | Target Count | Actual Count |")
    print("| ------------------------- | ------------ | ------------ |")
    for group, target in target_counts.items():
        actual = (train_metadata['lesion_group'] == group).sum()
        print(f"| {group:<25} | {target:<12} | {actual:<12} |")
    full_dataset.metadata = train_metadata
    train_dataset = full_dataset
    # Class weights for FocalLoss
    class_counts = train_dataset.metadata['lesion_group'].value_counts().reindex(list(target_counts.keys()), fill_value=0)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights_tensor = torch.tensor(class_weights.values, dtype=torch.float, device=device)
    # Oversample rare classes using WeightedRandomSampler
    sample_weights = train_dataset.metadata['lesion_group'].map(class_weights).values
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        sampler=sampler,
        num_workers=0,
        pin_memory=training_config['pin_memory']
    )
    # Use FocalLoss with class weights
    criterion = FocalLoss(alpha=class_weights_tensor, gamma=2)
    
    # Create model
    model = LesionTypeHead(
        input_dim=256,
        hidden_dims=training_config['model']['hidden_dims'],
        output_dim=len(training_config['class_names']),
        dropout=training_config['model']['dropout']
    ).to(device)
    
    # Loss function and optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=training_config['learning_rate'],
        weight_decay=training_config['weight_decay']
    )
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()
    
    # After train_loader is created, define val_dataset and val_loader
    val_dataset = SkinLesionDataset(
        metadata_path=training_config['metadata_path'],
        feature_cache_dir=training_config['val_features_dir'],
        skin_only=True,
        original_only=True,
        subset_fraction=training_config['data_fraction'],
        per_class_fraction=None
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=training_config['pin_memory']
    )
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(training_config['num_epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Use tqdm for progress bar
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{training_config["num_epochs"]} [Train]')
        for features, labels in train_pbar:
            features, labels = features.to(device), labels.to(device)
            
            # Clean GPU memory periodically
            if train_total % 1000 == 0:
                clean_gpu_memory()
            
            optimizer.zero_grad()
            
            # Use mixed precision training
            with autocast('cuda'):
                outputs = model(features)
                loss = criterion(outputs, labels)
            # Scale gradients and optimize
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * train_correct / train_total:.2f}%'
            })
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{training_config["num_epochs"]} [Val]')
            for features, labels in val_pbar:
                features, labels = features.to(device), labels.to(device)
                
                # Use mixed precision for validation too
                with autocast('cuda'):
                    outputs = model(features)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                # Update progress bar
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100. * val_correct / val_total:.2f}%'
                })
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Log metrics
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })
        
        logger.info(f'Epoch [{epoch+1}/{training_config["num_epochs"]}]')
        logger.info(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        logger.info(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 
                      Path(training_config['model_save_dir']) / 'lesion_type_best.pth')
            logger.info('Saved best model checkpoint')
        
        # Clean GPU memory after each epoch
        clean_gpu_memory()
    
    wandb.finish()
    logger.info('Training completed!')

    # After training, evaluate on validation set
    all_labels = []
    all_preds = []
    model.eval()
    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, predicted = outputs.max(1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)
    print("Per-class F1:", f1_score(all_labels, all_preds, average=None))
    print("Per-class Precision:", precision_score(all_labels, all_preds, average=None, zero_division=0))
    print("Per-class Recall:", recall_score(all_labels, all_preds, average=None, zero_division=0))

if __name__ == "__main__":
    train_model() 