#!/usr/bin/env python3
"""
Script to summarize all dataset configurations.
This creates a comprehensive summary of all configurations,
including sample counts, class distributions, and paths.
"""
import os
import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.data_split import (
    DATASET_CONFIGS,
    load_metadata,
    filter_metadata,
    split_dataset,
    get_available_configs
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def create_summary_table(output_path=None, include_plots=True):
    """
    Create a summary table of all dataset configurations.
    
    Args:
        output_path: Path to save CSV and markdown summaries
        include_plots: Whether to generate distribution plots
    """
    # Load metadata
    metadata = load_metadata()
    logger.info(f"Loaded metadata with {len(metadata)} entries")
    
    # Create summary table
    summary_data = []
    
    # Process each configuration
    for config_name in sorted(get_available_configs()):
        config = DATASET_CONFIGS[config_name]
        
        logger.info(f"Processing {config_name}...")
        
        # Filter metadata based on configuration
        filtered_df = filter_metadata(metadata, config)
        
        # Determine stratify column
        dataset_type = config['dataset_type']
        if dataset_type == 'skin_classification':
            stratify_col = 'skin'
        elif dataset_type == 'lesion_type':
            stratify_col = 'lesion_group'
        elif dataset_type == 'benign_malignant':
            stratify_col = 'malignancy'
        else:
            stratify_col = 'label'
        
        # Get column data
        num_classes = len(filtered_df[stratify_col].unique())
        class_counts = filtered_df[stratify_col].value_counts().to_dict()
        
        # Format class counts for display
        class_counts_str = ", ".join([f"{k}: {v}" for k, v in class_counts.items()])
        
        # Split dataset
        train_df, val_df, test_df = split_dataset(filtered_df, stratify_col=stratify_col)
        
        # Add to summary data
        summary_data.append({
            'Config Name': config_name,
            'Model': config['model'].upper(),
            'Description': config['description'],
            'Dataset Type': dataset_type,
            'Samples per Class': config['samples_per_class'] or "All available",
            'Augmented': "Yes" if config['augmented'] else "No",
            'Include DTD': "Yes" if config['include_dtd'] else "No",
            'Total Samples': len(filtered_df),
            'Train Samples': len(train_df),
            'Val Samples': len(val_df),
            'Test Samples': len(test_df),
            'Number of Classes': num_classes,
            'Class Counts': class_counts_str,
            'Class Column': stratify_col
        })
        
        # Generate distribution plot if requested
        if include_plots:
            plot_class_distribution(
                train_df, val_df, test_df,
                stratify_col, config_name,
                output_dir=output_path
            )
    
    # Convert to DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary to CSV if output path provided
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        csv_path = os.path.join(output_path, "dataset_configs_summary.csv")
        summary_df.to_csv(csv_path, index=False)
        logger.info(f"Saved summary to {csv_path}")
        
        # Also save as markdown table
        md_path = os.path.join(output_path, "dataset_configs_summary.md")
        
        with open(md_path, 'w') as f:
            f.write("# Dataset Configurations Summary\n\n")
            f.write(tabulate(summary_df, headers='keys', tablefmt='github', showindex=False))
            f.write("\n\n")
            
            # Add extra details
            f.write("## Dataset Type Breakdown\n\n")
            type_counts = summary_df['Dataset Type'].value_counts()
            f.write("- " + "\n- ".join([f"{dtype}: {count} configurations" for dtype, count in type_counts.items()]))
            f.write("\n\n")
            
            f.write("## Model Type Breakdown\n\n")
            model_counts = summary_df['Model'].value_counts()
            f.write("- " + "\n- ".join([f"{model}: {count} configurations" for model, count in model_counts.items()]))
            
        logger.info(f"Saved markdown summary to {md_path}")
    
    # Print summary to console
    print("\n" + "="*100)
    print("DATASET CONFIGURATIONS SUMMARY")
    print("="*100)
    print(tabulate(summary_df, headers='keys', tablefmt='github', showindex=False))
    print("="*100)
    
    return summary_df

def plot_class_distribution(train_df, val_df, test_df, stratify_col, config_name, output_dir=None):
    """
    Plot class distribution across splits.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        stratify_col: Column to stratify by
        config_name: Configuration name
        output_dir: Directory to save plot
    """
    splits_dfs = {'Train': train_df, 'Val': val_df, 'Test': test_df}
    class_counts = {}
    
    for split_name, df in splits_dfs.items():
        counts = df[stratify_col].value_counts().to_dict()
        class_counts[split_name] = counts
    
    # Collect all classes
    all_classes = set()
    for counts in class_counts.values():
        all_classes.update(counts.keys())
    all_classes = sorted(all_classes)
    
    # Create plot data
    plot_data = []
    for split_name, counts in class_counts.items():
        for cls in all_classes:
            plot_data.append({
                'Split': split_name,
                'Class': str(cls),  # Convert to string to ensure consistent display
                'Count': counts.get(cls, 0)
            })
    
    # Create DataFrame for plotting
    plot_df = pd.DataFrame(plot_data)
    
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(x='Class', y='Count', hue='Split', data=plot_df)
    plt.title(f'{config_name} - Class Distribution Across Splits')
    plt.xlabel(stratify_col)
    plt.ylabel('Count')
    
    # Add counts on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%d')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save plot if output directory provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"{config_name}_distribution.png"))
    
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset Configurations Summary")
    parser.add_argument("--output", type=str, default="results/dataset_summary",
                        help="Output directory for summary and plots")
    parser.add_argument("--no-plots", action="store_true",
                        help="Disable generation of distribution plots")
    
    args = parser.parse_args()
    
    # Create summary
    summary_df = create_summary_table(args.output, not args.no_plots)