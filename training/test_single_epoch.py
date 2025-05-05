#!/usr/bin/env python3
"""
Script to run a single epoch of the MLP3 (benign/malignant classifier) for testing.
"""

import os
import sys
import torch
import logging
from pathlib import Path
import yaml
import gc
import time
from datetime import datetime
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mlp3_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

from models.benign_malignant_head import BenignMalignantHead
from datasets.benign_malignant_dataset import BenignMalignantDataset
from training.train_benign_malignant import FocalLoss, get_config

def load_config():
    """Load configuration from config.yaml."""
    config_path = project_root / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def test_single_epoch(config_name="balanced", max_samples=500):
    """Run a single epoch with the specified configuration."""
    start_time = time.time()
    
    # Disable wandb
    os.environ["WANDB_MODE"] = "disabled"
    
    # Get experiment configuration
    experiment_config = get_config(config_name)
    logger.info(f"Running test epoch for: {experiment_config['description']}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Create tiny datasets
    config = load_config()
    metadata_path = config['training']['metadata_path']
    feature_cache_dir = os.path.join(config['data']['processed']['root_dir'], 'features')
    
    # Dataset configuration
    original_only = experiment_config["original_only"]
    balanced_sampling = experiment_config["balanced_sampling"]
    
    logger.info(f"Creating small test dataset with max {max_samples} samples per class")
    logger.info(f"Configuration: original_only={original_only}, balanced_sampling={balanced_sampling}")
    
    # Create datasets with desired configuration but limited size
    train_dataset = BenignMalignantDataset(
        metadata_path=metadata_path,
        split='train',
        feature_cache_dir=feature_cache_dir,
        original_only=original_only,
        balanced_sampling=balanced_sampling,
        max_samples_per_class=max_samples
    )
    
    val_dataset = BenignMalignantDataset(
        metadata_path=metadata_path,
        split='val',
        feature_cache_dir=feature_cache_dir,
        original_only=original_only,
        balanced_sampling=True,
        max_samples_per_class=max_samples // 2
    )
    
    # DataLoader configuration - very small batches
    batch_size = 8
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Create model
    input_dim = 256
    hidden_dims = [512, 256]
    output_dim = 2
    
    model = BenignMalignantHead(input_dim, hidden_dims, output_dim, dropout=0.3).to(device)
    
    # Calculate class weights
    train_class_counts = train_dataset.class_counts
    total_samples = sum(train_class_counts.values())
    class_weights = torch.tensor([
        float(total_samples) / (len(train_class_counts) * float(train_class_counts[0])),  # benign
        float(total_samples) / (len(train_class_counts) * float(train_class_counts[1]))   # malignant
    ], dtype=torch.float32).to(device)
    
    logger.info(f"Class weights: {class_weights}")
    
    # Setup mixed precision
    scaler = torch.amp.GradScaler('cuda')
    
    # Setup loss and optimizer
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Run a single epoch
    logger.info("Starting single epoch training...")
    
    # Train phase
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    train_start = time.time()
    
    for batch_idx, (features, targets) in enumerate(train_loader):
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
            
            # Log progress
            if batch_idx % 5 == 0:
                logger.info(f"Batch {batch_idx+1}/{len(train_loader)}: Loss {loss.item():.4f}, "
                           f"Acc {100.*predicted.eq(targets).sum().item()/targets.size(0):.2f}%")
            
            # Clean up GPU memory
            del features, targets, outputs, loss
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Error in training batch {batch_idx}: {str(e)}")
            continue
    
    train_time = time.time() - train_start
    train_loss = train_loss / len(train_loader)
    train_acc = 100. * train_correct / train_total
    
    logger.info(f"Training: Loss {train_loss:.4f}, Acc {train_acc:.2f}%, Time: {train_time:.2f}s")
    
    # Validation phase
    logger.info("Starting validation...")
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    val_start = time.time()
    
    with torch.no_grad():
        for batch_idx, (features, targets) in enumerate(val_loader):
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                outputs = model(features)
                loss = criterion(outputs, targets)
            
            _, predicted = outputs.max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()
            val_loss += loss.item()
            
            # Log progress
            if batch_idx % 5 == 0:
                logger.info(f"Val Batch {batch_idx+1}/{len(val_loader)}: Loss {loss.item():.4f}, "
                           f"Acc {100.*predicted.eq(targets).sum().item()/targets.size(0):.2f}%")
            
            del features, targets, outputs, loss
            torch.cuda.empty_cache()
    
    val_time = time.time() - val_start
    val_loss = val_loss / len(val_loader)
    val_acc = 100. * val_correct / val_total
    
    logger.info(f"Validation: Loss {val_loss:.4f}, Acc {val_acc:.2f}%, Time: {val_time:.2f}s")
    
    total_time = time.time() - start_time
    logger.info(f"Total test time: {total_time:.2f}s")
    logger.info("Test completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test single epoch of MLP3')
    parser.add_argument('--config', type=str, default='balanced',
                      choices=['original', 'augmented', 'balanced'],
                      help='Configuration to test')
    parser.add_argument('--samples', type=int, default=500,
                      help='Max samples per class')
    args = parser.parse_args()
    
    test_single_epoch(config_name=args.config, max_samples=args.samples) 