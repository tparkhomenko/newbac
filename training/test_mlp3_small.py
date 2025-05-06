"""
Small test script for MLP3 (benign/malignant classifier) training.
Uses a very small subset of data and only a few epochs to verify code works.
"""

import os
import sys
from pathlib import Path
import torch
import logging
import yaml
import wandb

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import the main training function but customize for testing
from training.train_benign_malignant import FocalLoss, plot_confusion_matrix

# Import dataset and model
from datasets.benign_malignant_dataset import BenignMalignantDataset
from models.benign_malignant_head import BenignMalignantHead

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from config.yaml."""
    config_path = project_root / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def test_mlp3_small():
    """Run a small test of MLP3 to verify code functionality."""
    
    # Load config
    config = load_config()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Skip wandb for this test
    os.environ["WANDB_MODE"] = "disabled"
    
    # Create tiny dataset (100 samples per class)
    metadata_path = config['training']['metadata_path']
    feature_cache_dir = os.path.join(config['data']['processed']['root_dir'], 'features')
    
    logger.info("Creating tiny test datasets (100 samples per class)...")
    train_dataset = BenignMalignantDataset(
        metadata_path=metadata_path,
        split='train',
        feature_cache_dir=feature_cache_dir,
        original_only=True,  # Use only original images for faster test
        max_samples_per_class=100  # Tiny dataset
    )
    
    val_dataset = BenignMalignantDataset(
        metadata_path=metadata_path,
        split='val',
        feature_cache_dir=feature_cache_dir,
        original_only=True,
        max_samples_per_class=50  # Even smaller validation
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
    
    # DataLoader with small batch size
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
    input_dim = 256  # SAM feature dimension
    hidden_dims = [512, 256]
    output_dim = 2  # binary classification
    
    model = BenignMalignantHead(input_dim, hidden_dims, output_dim, dropout=0.3).to(device)
    
    # Setup mixed precision
    scaler = torch.amp.GradScaler('cuda')
    
    # Setup loss and optimizer
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Run only 2 epochs
    num_epochs = 2
    logger.info(f"Running {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Train phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
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
                
                logger.info(f"Batch {batch_idx+1}/{len(train_loader)}: Loss {loss.item():.4f}, "
                           f"Acc {100.*predicted.eq(targets).sum().item()/targets.size(0):.2f}%")
                
                # Clean up GPU memory
                del features, targets, outputs, loss
                torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Error in training batch {batch_idx}: {str(e)}")
                continue
                
        # Calculate average loss
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        logger.info(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.2f}%")
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                with torch.amp.autocast('cuda'):
                    outputs = model(features)
                    loss = criterion(outputs, targets)
                
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                val_loss += loss.item()
                
                del features, targets, outputs, loss
                torch.cuda.empty_cache()
        
        # Calculate validation metrics
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        logger.info(f"Epoch {epoch+1}: Val Loss {val_loss:.4f}, Val Acc {val_acc:.2f}%")
    
    logger.info("Test complete - model trained successfully!")
    
    # Save tiny test model to verify saving works
    checkpoint_dir = os.path.join(project_root, 'saved_models', 'benign_malignant')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'benign_malignant_test.pth')
    
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_acc': val_acc,
        'class_mapping': {0: 'benign', 1: 'malignant'},
        'class_weights': class_weights.cpu().numpy().astype(float),
    }, checkpoint_path)
    
    logger.info(f"Test model saved to {checkpoint_path}")

if __name__ == "__main__":
    test_mlp3_small() 