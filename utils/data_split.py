import os
import sys
import pandas as pd
import numpy as np
import logging
import yaml
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import wandb
from sklearn.model_selection import train_test_split
from tabulate import tabulate

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Predefined dataset configurations
DATASET_CONFIGS = {
    # MLP1 configurations
    'mlp1_balanced': {
        'model': 'mlp1',
        'description': 'Binary skin vs non-skin, balanced (2000 per class)',
        'dataset_type': 'skin_classification',
        'samples_per_class': 2000,
        'augmented': True,
        'balanced': True,
        'include_dtd': True,
    },
    
    # MLP2 configurations
    'mlp2_augmented': {
        'model': 'mlp2',
        'description': 'Lesion type (5-class), augmented, max 2000 per class',
        'dataset_type': 'lesion_type',
        'samples_per_class': 2000,
        'augmented': True,
        'balanced': True,
        'include_dtd': False,
    },
    'mlp2_original': {
        'model': 'mlp2',
        'description': 'Lesion type (5-class), original only, balanced',
        'dataset_type': 'lesion_type',
        'samples_per_class': None,  # Use all available
        'augmented': False,
        'balanced': True,
        'include_dtd': False,
    },
    
    # MLP3 configurations
    'mlp3_augmented': {
        'model': 'mlp3',
        'description': 'Benign vs malignant, augmented, max 2000 per class',
        'dataset_type': 'benign_malignant',
        'samples_per_class': 2000,
        'augmented': True,
        'balanced': True,
        'include_dtd': False,
    },
    'mlp3_original': {
        'model': 'mlp3',
        'description': 'Benign vs malignant, original only, balanced',
        'dataset_type': 'benign_malignant',
        'samples_per_class': None,  # Use all available
        'augmented': False,
        'balanced': True,
        'include_dtd': False,
    },
    'mlp3_augmented_full': {
        'model': 'mlp3',
        'description': 'Benign vs malignant, all augmented, balanced',
        'dataset_type': 'benign_malignant',
        'samples_per_class': None,  # Use all available
        'augmented': True,
        'balanced': True,
        'include_dtd': False,
    },
}

def load_config():
    """Load configuration from config.yaml."""
    config_path = project_root / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_metadata(metadata_path=None):
    """Load the unified metadata file."""
    if metadata_path is None:
        config = load_config()
        metadata_path = config.get('unified_metadata_path', 'datasets/metadata/unified_augmented.csv')
    
    logger.info(f"Loading metadata from: {metadata_path}")
    df = pd.read_csv(metadata_path)
    logger.info(f"Loaded {len(df)} entries")
    return df

def filter_metadata(df, dataset_config):
    """
    Filter metadata according to the dataset configuration.
    
    Args:
        df: DataFrame containing metadata
        dataset_config: Dict or str of dataset configuration
    
    Returns:
        Filtered DataFrame
    """
    # If string config name provided, get the actual config
    if isinstance(dataset_config, str):
        if dataset_config not in DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset config: {dataset_config}")
        dataset_config = DATASET_CONFIGS[dataset_config]
    
    dataset_type = dataset_config['dataset_type']
    samples_per_class = dataset_config['samples_per_class']
    augmented = dataset_config['augmented']
    balanced = dataset_config['balanced']
    include_dtd = dataset_config['include_dtd']
    
    # Keep original dataframe intact
    filtered_df = df.copy()
    
    # Filter by image type (original or augmented)
    if not augmented:
        logger.info("Filtering for original images only")
        filtered_df = filtered_df[filtered_df['image'].str.contains('_original.jpg')]
    
    # Apply dataset-specific filtering
    if dataset_type == 'skin_classification':
        # For skin classification, include DTD (non-skin) if specified
        if not include_dtd:
            filtered_df = filtered_df[filtered_df['skin'] == 1]
        
        # Create binary label column for skin vs not-skin
        filtered_df['label'] = filtered_df['skin']
        stratify_col = 'skin'
    
    elif dataset_type == 'lesion_type':
        # For lesion type, filter for skin-only and remove unknown lesion groups
        filtered_df = filtered_df[
            (filtered_df['skin'] == 1) & 
            (filtered_df['lesion_group'] != 'unknown') &
            (filtered_df['lesion_group'] != 'not_skin_texture')
        ]
        
        # Map lesion groups to class indices
        lesion_map = {
            'melanocytic': 0,
            'non-melanocytic carcinoma': 1,
            'keratosis': 2,
            'fibrous': 3,
            'vascular': 4
        }
        filtered_df['label'] = filtered_df['lesion_group'].map(lesion_map)
        stratify_col = 'lesion_group'
    
    elif dataset_type == 'benign_malignant':
        # For benign/malignant, filter for skin-only and valid malignancy labels
        filtered_df = filtered_df[
            (filtered_df['skin'] == 1) & 
            (filtered_df['malignancy'].isin(['benign', 'malignant']))
        ]
        
        # Map malignancy to binary class index
        filtered_df['label'] = filtered_df['malignancy'].map({'benign': 0, 'malignant': 1})
        stratify_col = 'malignancy'
    
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    logger.info(f"After filtering for {dataset_type}: {len(filtered_df)} entries")
    
    # Balance classes if specified
    if balanced:
        logger.info("Balancing classes...")
        filtered_df = balance_classes(filtered_df, stratify_col, samples_per_class)
    
    return filtered_df

def balance_classes(df, stratify_col, samples_per_class=None):
    """
    Balance classes to have equal representation or limit to max samples per class.
    
    Args:
        df: DataFrame to balance
        stratify_col: Column to stratify by
        samples_per_class: Maximum samples per class, or None to use min class size
    
    Returns:
        Balanced DataFrame
    """
    # Get class counts
    class_counts = df[stratify_col].value_counts()
    logger.info(f"Class distribution before balancing:\n{class_counts}")
    
    balanced_dfs = []
    
    for class_value, count in class_counts.items():
        class_df = df[df[stratify_col] == class_value]
        
        if samples_per_class is not None:
            # If max samples specified, cap at that number
            if count > samples_per_class:
                logger.info(f"Downsampling class {class_value} from {count} to {samples_per_class}")
                class_df = class_df.sample(samples_per_class, random_state=42)
        else:
            # If no max specified, use all samples for minority classes
            logger.info(f"Using all {count} samples for class {class_value}")
        
        balanced_dfs.append(class_df)
    
    # Combine all balanced classes
    balanced_df = pd.concat(balanced_dfs).reset_index(drop=True)
    
    # Log final class distribution
    final_counts = balanced_df[stratify_col].value_counts()
    logger.info(f"Class distribution after balancing:\n{final_counts}")
    
    return balanced_df

def split_dataset(df, stratify_col='label', test_size=0.15, val_size=0.15):
    """
    Split dataset into train, validation, and test sets with stratification.
    
    Args:
        df: DataFrame to split
        stratify_col: Column to stratify by
        test_size: Fraction for test set
        val_size: Fraction for validation set
    
    Returns:
        train_df, val_df, test_df
    """
    # Calculate validation size relative to what remains after test split
    val_from_train = val_size / (1 - test_size)
    
    # First split off test set
    train_val_df, test_df = train_test_split(
        df, 
        test_size=test_size,
        stratify=df[stratify_col],
        random_state=42
    )
    
    # Then split validation from training
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_from_train,
        stratify=train_val_df[stratify_col],
        random_state=42
    )
    
    # Assign split column
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'
    
    # Log split sizes
    logger.info(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    return train_df, val_df, test_df

def log_dataset_statistics(train_df, val_df, test_df, stratify_col, dataset_config, output_dir=None, log_wandb=True):
    """
    Log dataset statistics to console, CSV, and wandb.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        stratify_col: Column used for stratification
        dataset_config: Dataset configuration dict or name
        output_dir: Directory to save CSV logs
        log_wandb: Whether to log to wandb
    """
    # If string config name provided, get the actual config
    if isinstance(dataset_config, str):
        config_name = dataset_config
        if dataset_config not in DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset config: {dataset_config}")
        dataset_config = DATASET_CONFIGS[dataset_config]
    else:
        config_name = next((k for k, v in DATASET_CONFIGS.items() if v == dataset_config), "custom")
    
    # Get class counts for each split
    train_counts = train_df[stratify_col].value_counts().to_dict()
    val_counts = val_df[stratify_col].value_counts().to_dict()
    test_counts = test_df[stratify_col].value_counts().to_dict()
    
    # Create a summary table with all classes
    all_classes = sorted(set(list(train_counts.keys()) + list(val_counts.keys()) + list(test_counts.keys())))
    
    stats_data = []
    for class_name in all_classes:
        stats_data.append({
            'Class': class_name,
            'Train': train_counts.get(class_name, 0),
            'Val': val_counts.get(class_name, 0),
            'Test': test_counts.get(class_name, 0),
            'Total': train_counts.get(class_name, 0) + val_counts.get(class_name, 0) + test_counts.get(class_name, 0)
        })
    
    # Convert to DataFrame
    stats_df = pd.DataFrame(stats_data)
    
    # Add totals row
    totals = {
        'Class': 'TOTAL',
        'Train': sum(train_counts.values()),
        'Val': sum(val_counts.values()),
        'Test': sum(test_counts.values()),
        'Total': sum(train_counts.values()) + sum(val_counts.values()) + sum(test_counts.values())
    }
    stats_df = pd.concat([stats_df, pd.DataFrame([totals])], ignore_index=True)
    
    # Log to console
    logger.info(f"\n{'='*60}")
    logger.info(f"Dataset Statistics for {config_name}: {dataset_config['description']}")
    logger.info(f"{'='*60}")
    logger.info(f"\n{tabulate(stats_df, headers='keys', tablefmt='github', showindex=False)}")
    logger.info(f"{'='*60}\n")
    
    # Create directory for saving if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save to CSV
        csv_path = os.path.join(output_dir, f"{config_name}_statistics.csv")
        stats_df.to_csv(csv_path, index=False)
        logger.info(f"Saved statistics to {csv_path}")
        
        # Save split distribution plot
        plot_path = os.path.join(output_dir, f"{config_name}_distribution.png")
        plot_class_distribution(stats_df, config_name, plot_path)
    
    # Log to wandb if active
    if log_wandb and wandb.run is not None:
        # Remove 'TOTAL' row for wandb Table
        wandb_stats_df = stats_df[stats_df['Class'] != 'TOTAL'].copy()
        wandb_table = wandb.Table(dataframe=wandb_stats_df)
        wandb.log({f"dataset_info/{config_name}/class_distribution": wandb_table})
        
        # Log key metrics
        wandb.run.summary.update({
            f"dataset_info/{config_name}/total_samples": totals['Total'],
            f"dataset_info/{config_name}/train_samples": totals['Train'],
            f"dataset_info/{config_name}/val_samples": totals['Val'],
            f"dataset_info/{config_name}/test_samples": totals['Test'],
            f"dataset_info/{config_name}/num_classes": len(all_classes)
        })
    
    return stats_df

def plot_class_distribution(stats_df, config_name, save_path=None):
    """
    Plot class distribution across splits.
    
    Args:
        stats_df: DataFrame with class statistics
        config_name: Configuration name for title
        save_path: Path to save plot
    """
    # Drop the totals row
    plot_df = stats_df[stats_df['Class'] != 'TOTAL'].copy()
    
    # Melt the dataframe for seaborn
    melted_df = pd.melt(
        plot_df, 
        id_vars=['Class'], 
        value_vars=['Train', 'Val', 'Test'],
        var_name='Split', value_name='Count'
    )
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x='Class', y='Count', hue='Split', data=melted_df)
    
    # Add labels
    plt.title(f"Class Distribution for {config_name}")
    plt.xlabel("Class")
    plt.ylabel("Sample Count")
    
    # Add labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%d')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved distribution plot to {save_path}")
    
    plt.close()

def get_dataset_splits(dataset_config, return_df=False, log_stats=True, output_dir=None, log_wandb=True):
    """
    Get train, validation, and test splits for a given dataset configuration.
    
    Args:
        dataset_config: Dataset configuration name or dict
        return_df: Whether to return the DataFrames or just log statistics
        log_stats: Whether to log statistics
        output_dir: Directory to save statistics
        log_wandb: Whether to log to wandb
    
    Returns:
        If return_df=True: train_df, val_df, test_df
        Otherwise: None
    """
    # If string config name provided, get the actual config
    if isinstance(dataset_config, str):
        if dataset_config not in DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset config: {dataset_config}")
        dataset_config = DATASET_CONFIGS[dataset_config]
    
    # Load and filter metadata
    metadata_df = load_metadata()
    filtered_df = filter_metadata(metadata_df, dataset_config)
    
    # Determine stratify column based on dataset type
    if dataset_config['dataset_type'] == 'skin_classification':
        stratify_col = 'skin'
    elif dataset_config['dataset_type'] == 'lesion_type':
        stratify_col = 'lesion_group'
    elif dataset_config['dataset_type'] == 'benign_malignant':
        stratify_col = 'malignancy'
    else:
        stratify_col = 'label'
    
    # Split dataset
    train_df, val_df, test_df = split_dataset(filtered_df, stratify_col=stratify_col)
    
    # Log statistics if requested
    if log_stats:
        log_dataset_statistics(
            train_df, val_df, test_df, 
            stratify_col, dataset_config, 
            output_dir=output_dir, 
            log_wandb=log_wandb
        )
    
    if return_df:
        return train_df, val_df, test_df
    else:
        return None

def preview_dataset_config(dataset_config, output_dir=None):
    """
    Preview a dataset configuration without training.
    Useful for dry-runs.
    
    Args:
        dataset_config: Dataset configuration name or dict
        output_dir: Directory to save statistics
    """
    # If output directory not specified, use a default
    if output_dir is None:
        config = load_config()
        output_dir = config.get('dataset_stats_dir', 'results/dataset_stats')
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get dataset splits and log statistics
    logger.info(f"Previewing dataset configuration: {dataset_config}")
    get_dataset_splits(
        dataset_config, 
        return_df=False, 
        log_stats=True, 
        output_dir=output_dir,
        log_wandb=False
    )
    logger.info(f"Preview complete. Statistics saved to {output_dir}")

def save_to_cache(df, cache_path):
    """Save DataFrame to cache file."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_csv(cache_path, index=False)
    logger.info(f"Saved dataset to cache: {cache_path}")

def load_from_cache(cache_path):
    """Load DataFrame from cache file."""
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
        logger.info(f"Loaded dataset from cache: {cache_path} ({len(df)} entries)")
        return df
    return None

def get_available_configs():
    """Return list of available dataset configurations."""
    return list(DATASET_CONFIGS.keys())

def describe_config(config_name):
    """Print description of a specific configuration."""
    if config_name not in DATASET_CONFIGS:
        logger.error(f"Unknown dataset config: {config_name}")
        return
    
    config = DATASET_CONFIGS[config_name]
    logger.info(f"\nDataset Configuration: {config_name}")
    logger.info(f"{'='*40}")
    logger.info(f"Description: {config['description']}")
    logger.info(f"Model: {config['model']}")
    logger.info(f"Dataset Type: {config['dataset_type']}")
    logger.info(f"Samples per class: {config['samples_per_class'] or 'All available'}")
    logger.info(f"Augmented: {config['augmented']}")
    logger.info(f"Balanced: {config['balanced']}")
    logger.info(f"Include DTD: {config['include_dtd']}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Dataset Management Utility")
    parser.add_argument('--list', action='store_true', help='List all available dataset configurations')
    parser.add_argument('--describe', type=str, help='Describe a specific dataset configuration')
    parser.add_argument('--preview', type=str, help='Preview a dataset configuration')
    parser.add_argument('--output', type=str, help='Output directory for statistics', default='results/dataset_stats')
    
    args = parser.parse_args()
    
    if args.list:
        logger.info("\nAvailable Dataset Configurations:")
        for name in get_available_configs():
            config = DATASET_CONFIGS[name]
            logger.info(f"- {name}: {config['description']}")
    
    if args.describe:
        describe_config(args.describe)
    
    if args.preview:
        preview_dataset_config(args.preview, args.output)
    
    # If no arguments given, show help
    if not (args.list or args.describe or args.preview):
        parser.print_help()