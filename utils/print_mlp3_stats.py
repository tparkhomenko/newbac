#!/usr/bin/env python3
"""
Script to print statistics for MLP3 (benign/malignant classifier) dataset configurations.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from tabulate import tabulate
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from config.yaml."""
    config_path = project_root / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def print_mlp3_stats():
    """
    Print statistics for MLP3 (benign/malignant classifier) dataset configurations.
    Shows counts for different setups: original-only, augmented, and balanced subset.
    """
    # Load config
    config = load_config()
    metadata_path = config.get('training', {}).get('metadata_path')
    
    if not metadata_path:
        logger.error("Metadata path not found in config.yaml")
        return
    
    # Load metadata
    metadata = pd.read_csv(metadata_path)
    logger.info(f"Loaded metadata with {len(metadata)} entries")
    
    # Filter for skin images with valid malignancy labels
    skin_data = metadata[(metadata['skin'] == 1) & 
                         (metadata['malignancy'].isin(['benign', 'malignant']))]
    
    # Count originals
    originals = skin_data[skin_data['image'].str.contains('_original.jpg')]
    
    # Split counts
    train_data = skin_data[skin_data['split'] == 'train']
    val_data = skin_data[skin_data['split'] == 'val']
    test_data = skin_data[skin_data['split'] == 'test']
    
    # Original split counts
    train_originals = originals[originals['split'] == 'train']
    val_originals = originals[originals['split'] == 'val']
    test_originals = originals[originals['split'] == 'test']
    
    # Count per malignancy
    total_benign = len(skin_data[skin_data['malignancy'] == 'benign'])
    total_malignant = len(skin_data[skin_data['malignancy'] == 'malignant'])
    
    original_benign = len(originals[originals['malignancy'] == 'benign'])
    original_malignant = len(originals[originals['malignancy'] == 'malignant'])
    
    train_benign = len(train_data[train_data['malignancy'] == 'benign'])
    train_malignant = len(train_data[train_data['malignancy'] == 'malignant'])
    
    val_benign = len(val_data[val_data['malignancy'] == 'benign'])
    val_malignant = len(val_data[val_data['malignancy'] == 'malignant'])
    
    test_benign = len(test_data[test_data['malignancy'] == 'benign'])
    test_malignant = len(test_data[test_data['malignancy'] == 'malignant'])
    
    # Original split counts per malignancy
    train_original_benign = len(train_originals[train_originals['malignancy'] == 'benign'])
    train_original_malignant = len(train_originals[train_originals['malignancy'] == 'malignant'])
    
    val_original_benign = len(val_originals[val_originals['malignancy'] == 'benign'])
    val_original_malignant = len(val_originals[val_originals['malignancy'] == 'malignant'])
    
    test_original_benign = len(test_originals[test_originals['malignancy'] == 'benign'])
    test_original_malignant = len(test_originals[test_originals['malignancy'] == 'malignant'])
    
    # Generate tables
    
    # Full dataset stats
    full_table = [
        ["Class", "Total in DB", "Originals", "Train", "Val", "Test"],
        ["benign", total_benign, original_benign, train_benign, val_benign, test_benign],
        ["malignant", total_malignant, original_malignant, train_malignant, val_malignant, test_malignant],
        ["TOTAL", total_benign + total_malignant, original_benign + original_malignant, 
         train_benign + train_malignant, val_benign + val_malignant, test_benign + test_malignant]
    ]
    
    # Original-only configuration (Config A)
    original_table = [
        ["Class", "Total in DB", "Used in Training", "Train", "Val", "Test"],
        ["benign", original_benign, train_original_benign, train_original_benign, val_original_benign, test_original_benign],
        ["malignant", original_malignant, train_original_malignant, train_original_malignant, val_original_malignant, test_original_malignant],
        ["TOTAL", original_benign + original_malignant, 
         train_original_benign + train_original_malignant,
         train_original_benign + train_original_malignant,
         val_original_benign + val_original_malignant,
         test_original_benign + test_original_malignant]
    ]
    
    # Balanced 2k configuration (Config C)
    balanced_train_benign = min(2000, train_benign)
    balanced_train_malignant = min(2000, train_malignant)
    balanced_val_benign = min(2000, val_benign)
    balanced_val_malignant = min(2000, val_malignant)
    
    balanced_table = [
        ["Class", "Total in DB", "Originals", "Used in Training", "Train", "Val (Max 2k/class)"],
        ["benign", total_benign, original_benign, balanced_train_benign, balanced_train_benign, balanced_val_benign],
        ["malignant", total_malignant, original_malignant, balanced_train_malignant, balanced_train_malignant, balanced_val_malignant],
        ["TOTAL", total_benign + total_malignant, original_benign + original_malignant, 
         balanced_train_benign + balanced_train_malignant, 
         balanced_train_benign + balanced_train_malignant,
         balanced_val_benign + balanced_val_malignant]
    ]
    
    # Print tables
    logger.info("\n" + "="*80)
    logger.info("MLP3 BENIGN/MALIGNANT CLASSIFIER DATASET STATISTICS")
    logger.info("="*80)
    
    logger.info("\nFULL DATASET OVERVIEW - All Images (Augmented)")
    logger.info(tabulate(full_table, headers="firstrow", tablefmt="grid"))
    
    logger.info("\nCONFIGURATION A - Original Images Only")
    logger.info(tabulate(original_table, headers="firstrow", tablefmt="grid"))
    
    logger.info("\nCONFIGURATION B - Full Augmented Dataset")
    logger.info("Uses all available images in the training split:")
    logger.info(f"- Training: {train_benign} benign, {train_malignant} malignant ({train_benign + train_malignant} total)")
    logger.info(f"- Validation: Up to 2000 samples per class (balanced)")
    
    logger.info("\nCONFIGURATION C - Balanced Subset (2k per class)")
    logger.info(tabulate(balanced_table, headers="firstrow", tablefmt="grid"))
    
    logger.info("\nThese configurations are implemented in training/train_benign_malignant.py")
    logger.info("To run all configurations: python -m training.run_all_mlp3_configs")
    logger.info("To run a specific configuration: python -m training.train_benign_malignant --config [original|augmented|balanced]")
    logger.info("="*80)

if __name__ == "__main__":
    print_mlp3_stats() 