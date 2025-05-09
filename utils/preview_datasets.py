import os
import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.data_split import (
    DATASET_CONFIGS, 
    load_metadata, 
    filter_metadata, 
    split_dataset, 
    log_dataset_statistics
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def preview_dataset(dataset_config, output_dir=None):
    """
    Preview a dataset configuration without training.
    
    Args:
        dataset_config: Dataset configuration name
        output_dir: Directory to save statistics
    """
    if dataset_config not in DATASET_CONFIGS:
        logger.error(f"Unknown dataset config: {dataset_config}")
        return False
    
    config_dict = DATASET_CONFIGS[dataset_config]
    
    logger.info(f"\n{'='*60}")
    logger.info(f"PREVIEWING DATASET: {dataset_config}")
    logger.info(f"{'='*60}")
    logger.info(f"Description: {config_dict['description']}")
    logger.info(f"Model: {config_dict['model']}")
    logger.info(f"Dataset Type: {config_dict['dataset_type']}")
    logger.info(f"Samples per class: {config_dict['samples_per_class'] or 'All available'}")
    logger.info(f"Augmented: {config_dict['augmented']}")
    logger.info(f"Balanced: {config_dict['balanced']}")
    logger.info(f"Include DTD: {config_dict['include_dtd']}")
    logger.info(f"{'='*60}\n")
    
    # Load metadata
    metadata_df = load_metadata()
    
    # Filter metadata based on config
    filtered_df = filter_metadata(metadata_df, config_dict)
    
    # Determine stratify column based on dataset type
    dataset_type = config_dict['dataset_type']
    if dataset_type == 'skin_classification':
        stratify_col = 'skin'
    elif dataset_type == 'lesion_type':
        stratify_col = 'lesion_group'
    elif dataset_type == 'benign_malignant':
        stratify_col = 'malignancy'
    else:
        stratify_col = 'label'
    
    # Split dataset
    train_df, val_df, test_df = split_dataset(filtered_df, stratify_col=stratify_col)
    
    # Log dataset statistics
    if output_dir is None:
        output_dir = os.path.join(project_root, 'results', 'dataset_stats')
    
    os.makedirs(output_dir, exist_ok=True)
    
    stats_df = log_dataset_statistics(
        train_df, val_df, test_df, 
        stratify_col, dataset_config, 
        output_dir=output_dir,
        log_wandb=False
    )
    
    # Save detailed distribution file
    if stratify_col in filtered_df.columns:
        plot_detailed_distribution(
            filtered_df, 
            dataset_config, 
            stratify_col, 
            os.path.join(output_dir, f"{dataset_config}_detailed.png")
        )
    
    return True

def plot_detailed_distribution(df, config_name, column, save_path):
    """
    Plot a detailed distribution of the dataset.
    
    Args:
        df: DataFrame to analyze
        config_name: Configuration name
        column: Column to analyze
        save_path: Path to save the plot
    """
    plt.figure(figsize=(14, 10))
    
    # Plot class distribution
    class_counts = df[column].value_counts()
    ax = sns.barplot(x=class_counts.index, y=class_counts.values)
    
    # Add value labels
    for i, count in enumerate(class_counts.values):
        ax.text(i, count + 0.1, str(count), ha='center')
    
    # Add labels
    plt.title(f"Detailed Class Distribution - {config_name}")
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    logger.info(f"Saved detailed distribution plot to {save_path}")

def preview_all_datasets(output_dir=None):
    """Preview all available dataset configurations."""
    results = {}
    
    for config_name in DATASET_CONFIGS:
        logger.info(f"Previewing {config_name}...")
        success = preview_dataset(config_name, output_dir)
        results[config_name] = "✅ Success" if success else "❌ Failed"
    
    # Print summary
    logger.info("\nPREVIEW SUMMARY")
    logger.info("=" * 50)
    for config_name, result in results.items():
        logger.info(f"{config_name}: {result}")
    logger.info("=" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset Preview Utility")
    parser.add_argument("--config", type=str, help="Dataset configuration to preview")
    parser.add_argument("--output", type=str, help="Output directory for statistics", default="results/dataset_stats")
    parser.add_argument("--all", action="store_true", help="Preview all dataset configurations")
    
    args = parser.parse_args()
    
    if args.all:
        preview_all_datasets(args.output)
    elif args.config:
        preview_dataset(args.config, args.output)
    else:
        logger.error("Please provide either a --config name or use --all to preview all configurations")
        parser.print_help() 