#!/usr/bin/env python3
"""
Test script for unified dataset implementation.
This script tests loading different dataset configurations
and verifies that the splits are correct.
"""
import os
import sys
import logging
from pathlib import Path
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from datasets.unified_dataset import UnifiedDataset
from utils.data_split import DATASET_CONFIGS, get_available_configs

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def test_dataset(config_name, batch_size=8, num_workers=0, verbose=False):
    """
    Test a specific dataset configuration.
    
    Args:
        config_name: Name of the dataset configuration to test
        batch_size: Batch size for dataloader
        num_workers: Number of workers for dataloader
        verbose: Print additional information
    
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Testing dataset: {config_name}")
    
    try:
        # Get configuration
        if config_name not in DATASET_CONFIGS:
            logger.error(f"Unknown dataset config: {config_name}")
            return False
        
        config = DATASET_CONFIGS[config_name]
        
        # Create datasets for each split
        splits = ['train', 'val', 'test']
        datasets = {}
        
        for split in splits:
            logger.info(f"Loading {split} split...")
            datasets[split] = UnifiedDataset(
                dataset_config=config_name,
                split=split,
                use_cache=False  # Don't use cache for testing
            )
        
        # Test dataloaders
        for split, dataset in datasets.items():
            if len(dataset) == 0:
                logger.error(f"Empty dataset for {split} split!")
                return False
            
            logger.info(f"{split.capitalize()} split: {len(dataset)} samples")
            
            # Create dataloader
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers
            )
            
            # Load a batch
            logger.info(f"Loading a batch from {split} dataloader...")
            features, labels = next(iter(dataloader))
            
            if verbose:
                logger.info(f"Features shape: {features.shape}")
                logger.info(f"Labels shape: {labels.shape}")
                logger.info(f"Label values: {labels.tolist()}")
            
            # Check that features and labels are valid
            if torch.isnan(features).any():
                logger.error(f"NaN values found in features for {split} split!")
                return False
            
            if torch.isnan(labels).any():
                logger.error(f"NaN values found in labels for {split} split!")
                return False
            
            logger.info(f"✅ {split.capitalize()} split dataloader test passed")
        
        # Verify the sum of splits matches expected
        total_samples = sum(len(dataset) for dataset in datasets.values())
        logger.info(f"Total samples across all splits: {total_samples}")
        
        # Print split percentages
        train_pct = len(datasets['train']) / total_samples * 100
        val_pct = len(datasets['val']) / total_samples * 100
        test_pct = len(datasets['test']) / total_samples * 100
        
        logger.info(f"Split percentages - Train: {train_pct:.1f}%, Val: {val_pct:.1f}%, Test: {test_pct:.1f}%")
        
        if abs(train_pct - 70.0) > 3.0 or abs(val_pct - 15.0) > 3.0 or abs(test_pct - 15.0) > 3.0:
            logger.warning("Split percentages deviate by more than 3% from expected 70/15/15 split!")
        
        logger.info(f"✅ All tests passed for {config_name}")
        return True
    
    except Exception as e:
        logger.error(f"Error testing {config_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_all_configs():
    """Test all available dataset configurations."""
    results = {}
    
    for config_name in get_available_configs():
        logger.info(f"\n{'-'*80}")
        logger.info(f"Testing {config_name}...")
        success = test_dataset(config_name)
        results[config_name] = "✅ Success" if success else "❌ Failed"
        logger.info(f"{'-'*80}\n")
    
    # Print summary
    logger.info("\nTEST SUMMARY")
    logger.info("=" * 50)
    for config_name, result in results.items():
        logger.info(f"{config_name}: {result}")
    logger.info("=" * 50)
    
    # Check if all tests passed
    all_passed = all(result == "✅ Success" for result in results.values())
    
    if all_passed:
        logger.info("✅ All dataset configurations passed!")
    else:
        logger.error("❌ Some dataset configurations failed!")
    
    return all_passed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Unified Dataset Implementation")
    parser.add_argument("--config", type=str, help="Specific configuration to test")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for testing")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for dataloader")
    parser.add_argument("--verbose", action="store_true", help="Print additional information")
    
    args = parser.parse_args()
    
    if args.config:
        test_dataset(args.config, args.batch_size, args.num_workers, args.verbose)
    else:
        test_all_configs() 