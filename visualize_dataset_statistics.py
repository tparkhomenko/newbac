import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def load_data():
    """Load the dataset with splits."""
    df = pd.read_csv('data/metadata/unified_labels_with_splits.csv')
    return df

def create_source_distribution_plot(df):
    """Create a pie chart showing source dataset distribution."""
    source_counts = df['source_csv'].value_counts()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Pie chart
    colors = plt.cm.Set3(np.linspace(0, 1, len(source_counts)))
    wedges, texts, autotexts = ax1.pie(source_counts.values, labels=source_counts.index, 
                                        autopct='%1.1f%%', colors=colors, startangle=90)
    ax1.set_title('Source Dataset Distribution', fontsize=16, fontweight='bold')
    
    # Bar chart
    source_counts.plot(kind='bar', ax=ax2, color=colors)
    ax2.set_title('Source Dataset Counts', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Source Dataset')
    ax2.set_ylabel('Number of Images')
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_xticklabels(ax2.get_xticklabels(), ha='right')
    
    plt.tight_layout()
    plt.savefig('results/dataset_stats/source_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_label_distribution_plot(df):
    """Create plots showing label distribution."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Overall label distribution
    label_counts = df['unified_label'].value_counts()
    colors = plt.cm.Set3(np.linspace(0, 1, len(label_counts)))
    label_counts.plot(kind='bar', ax=ax1, color=colors)
    ax1.set_title('Overall Label Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Label')
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=45)
    
    # Label distribution by source
    label_source = pd.crosstab(df['unified_label'], df['source_csv'])
    label_source.plot(kind='bar', ax=ax2, stacked=True, colormap='Set3')
    ax2.set_title('Label Distribution by Source', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Label')
    ax2.set_ylabel('Count')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Pie chart of labels
    wedges, texts, autotexts = ax3.pie(label_counts.values, labels=label_counts.index, 
                                        autopct='%1.1f%%', colors=colors, startangle=90)
    ax3.set_title('Label Distribution (Pie Chart)', fontsize=14, fontweight='bold')
    
    # Heatmap of label vs source
    sns.heatmap(label_source, annot=True, fmt='d', cmap='YlOrRd', ax=ax4)
    ax4.set_title('Label vs Source Heatmap', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Source Dataset')
    ax4.set_ylabel('Label')
    
    plt.tight_layout()
    plt.savefig('results/dataset_stats/label_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_split_analysis_plot(df):
    """Create plots showing split analysis."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Split counts
    split_columns = ['split1', 'split2', 'split3', 'split4', 'split5']
    split_counts = {}
    for split_col in split_columns:
        split_counts[split_col] = df[split_col].sum()
    
    split_counts_series = pd.Series(split_counts)
    colors = plt.cm.Set3(np.linspace(0, 1, len(split_counts)))
    split_counts_series.plot(kind='bar', ax=ax1, color=colors)
    ax1.set_title('Images per Split', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Split')
    ax1.set_ylabel('Number of Images')
    ax1.tick_params(axis='x', rotation=45)
    
    # Split overlap analysis
    overlap_matrix = np.zeros((len(split_columns), len(split_columns)))
    for i, split1 in enumerate(split_columns):
        for j, split2 in enumerate(split_columns):
            overlap = len(df[(df[split1] == 1) & (df[split2] == 1)])
            overlap_matrix[i, j] = overlap
    
    im = ax2.imshow(overlap_matrix, cmap='Blues', aspect='auto')
    ax2.set_xticks(range(len(split_columns)))
    ax2.set_yticks(range(len(split_columns)))
    ax2.set_xticklabels(split_columns)
    ax2.set_yticklabels(split_columns)
    ax2.set_title('Split Overlap Matrix', fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(len(split_columns)):
        for j in range(len(split_columns)):
            text = ax2.text(j, i, int(overlap_matrix[i, j]),
                           ha="center", va="center", color="white" if overlap_matrix[i, j] > overlap_matrix.max()/2 else "black")
    
    plt.colorbar(im, ax=ax2)
    
    # Label distribution in splits
    split_label_data = []
    for split_col in split_columns:
        split_data = df[df[split_col] == 1]
        label_counts = split_data['unified_label'].value_counts()
        for label, count in label_counts.items():
            split_label_data.append({'Split': split_col, 'Label': label, 'Count': count})
    
    split_label_df = pd.DataFrame(split_label_data)
    pivot_data = split_label_df.pivot(index='Label', columns='Split', values='Count').fillna(0)
    
    pivot_data.plot(kind='bar', ax=ax3, colormap='Set3')
    ax3.set_title('Label Distribution Across Splits', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Label')
    ax3.set_ylabel('Count')
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Source distribution in splits
    split_source_data = []
    for split_col in split_columns:
        split_data = df[df[split_col] == 1]
        source_counts = split_data['source_csv'].value_counts()
        for source, count in source_counts.items():
            split_source_data.append({'Split': split_col, 'Source': source, 'Count': count})
    
    split_source_df = pd.DataFrame(split_source_data)
    pivot_source = split_source_df.pivot(index='Source', columns='Split', values='Count').fillna(0)
    
    pivot_source.plot(kind='bar', ax=ax4, colormap='Set3')
    ax4.set_title('Source Distribution Across Splits', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Source Dataset')
    ax4.set_ylabel('Count')
    ax4.tick_params(axis='x', rotation=45)
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('results/dataset_stats/split_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_stratified_split_visualization():
    """Create visualization for stratified splits."""
    # Load the stratified data
    try:
        df_stratified = pd.read_csv('data/metadata/unified_labels_with_stratified_splits.csv')
        
        # Count stratified assignments for each split
        split_columns = ['split1', 'split2', 'split3', 'split4', 'split5']
        
        # Create a figure with 5 subplots (2 rows, 3 columns)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()  # Flatten to 1D array
        
        for i, split_col in enumerate(split_columns):
            if split_col in df_stratified.columns and i < len(axes):
                # Get non-empty values (actual splits)
                split_data = df_stratified[df_stratified[split_col] != '']
                split_counts = split_data[split_col].value_counts()
                
                ax = axes[i]
                colors = plt.cm.Set3(np.linspace(0, 1, len(split_counts)))
                split_counts.plot(kind='bar', ax=ax, color=colors)
                ax.set_title(f'{split_col.upper()} - Stratified Distribution', fontsize=12, fontweight='bold')
                ax.set_xlabel('Split Type')
                ax.set_ylabel('Count')
                ax.tick_params(axis='x', rotation=45)
                
                # Add percentage labels
                total = split_counts.sum()
                for j, (split_type, count) in enumerate(split_counts.items()):
                    percentage = (count / total) * 100
                    ax.text(j, count + total * 0.01, f'{percentage:.1f}%', 
                           ha='center', va='bottom', fontweight='bold')
        
        # Hide the last subplot if we have 5 splits
        if len(axes) > 5:
            axes[5].axis('off')
        
        plt.tight_layout()
        plt.savefig('results/dataset_stats/stratified_splits.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    except FileNotFoundError:
        print("Stratified splits file not found. Skipping stratified visualization.")

def create_comprehensive_summary_plot(df):
    """Create a comprehensive summary plot."""
    fig = plt.figure(figsize=(20, 12))
    
    # Create a grid layout
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # 1. Source distribution (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    source_counts = df['source_csv'].value_counts()
    colors = plt.cm.Set3(np.linspace(0, 1, len(source_counts)))
    source_counts.plot(kind='bar', ax=ax1, color=colors)
    ax1.set_title('Source Dataset Distribution', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Source')
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=45)
    ax1.set_xticklabels(ax1.get_xticklabels(), ha='right')
    
    # 2. Label distribution (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    label_counts = df['unified_label'].value_counts()
    label_counts.plot(kind='bar', ax=ax2, color=colors)
    ax2.set_title('Label Distribution', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Label')
    ax2.set_ylabel('Count')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Split counts (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    split_columns = ['split1', 'split2', 'split3', 'split4', 'split5']
    split_counts = {}
    for split_col in split_columns:
        split_counts[split_col] = df[split_col].sum()
    split_counts_series = pd.Series(split_counts)
    split_counts_series.plot(kind='bar', ax=ax3, color=colors)
    ax3.set_title('Images per Split', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Split')
    ax3.set_ylabel('Count')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Label vs Source heatmap (middle left)
    ax4 = fig.add_subplot(gs[1, 0])
    label_source = pd.crosstab(df['unified_label'], df['source_csv'])
    sns.heatmap(label_source, annot=True, fmt='d', cmap='YlOrRd', ax=ax4)
    ax4.set_title('Label vs Source Heatmap', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Source Dataset')
    ax4.set_ylabel('Label')
    
    # 5. Split overlap matrix (middle middle)
    ax5 = fig.add_subplot(gs[1, 1])
    overlap_matrix = np.zeros((len(split_columns), len(split_columns)))
    for i, split1 in enumerate(split_columns):
        for j, split2 in enumerate(split_columns):
            overlap = len(df[(df[split1] == 1) & (df[split2] == 1)])
            overlap_matrix[i, j] = overlap
    
    im = ax5.imshow(overlap_matrix, cmap='Blues', aspect='auto')
    ax5.set_xticks(range(len(split_columns)))
    ax5.set_yticks(range(len(split_columns)))
    ax5.set_xticklabels(split_columns)
    ax5.set_yticklabels(split_columns)
    ax5.set_title('Split Overlap Matrix', fontsize=12, fontweight='bold')
    
    # Add text annotations
    for i in range(len(split_columns)):
        for j in range(len(split_columns)):
            text = ax5.text(j, i, int(overlap_matrix[i, j]),
                           ha="center", va="center", color="white" if overlap_matrix[i, j] > overlap_matrix.max()/2 else "black")
    
    # 6. Label distribution in splits (middle right)
    ax6 = fig.add_subplot(gs[1, 2])
    split_label_data = []
    for split_col in split_columns:
        split_data = df[df[split_col] == 1]
        label_counts = split_data['unified_label'].value_counts()
        for label, count in label_counts.items():
            split_label_data.append({'Split': split_col, 'Label': label, 'Count': count})
    
    split_label_df = pd.DataFrame(split_label_data)
    pivot_data = split_label_df.pivot(index='Label', columns='Split', values='Count').fillna(0)
    
    pivot_data.plot(kind='bar', ax=ax6, colormap='Set3')
    ax6.set_title('Label Distribution Across Splits', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Label')
    ax6.set_ylabel('Count')
    ax6.tick_params(axis='x', rotation=45)
    ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 7. Source distribution in splits (bottom left)
    ax7 = fig.add_subplot(gs[2, 0])
    split_source_data = []
    for split_col in split_columns:
        split_data = df[df[split_col] == 1]
        source_counts = split_data['source_csv'].value_counts()
        for source, count in source_counts.items():
            split_source_data.append({'Split': split_col, 'Source': source, 'Count': count})
    
    split_source_df = pd.DataFrame(split_source_data)
    pivot_source = split_source_df.pivot(index='Source', columns='Split', values='Count').fillna(0)
    
    pivot_source.plot(kind='bar', ax=ax7, colormap='Set3')
    ax7.set_title('Source Distribution Across Splits', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Source Dataset')
    ax7.set_ylabel('Count')
    ax7.tick_params(axis='x', rotation=45)
    ax7.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 8. Summary statistics (bottom middle and right)
    ax8 = fig.add_subplot(gs[2, 1:])
    ax8.axis('off')
    
    # Create summary text
    summary_text = f"""
    DATASET SUMMARY STATISTICS
    
    Total Images: {len(df):,}
    
    Source Datasets: {len(df['source_csv'].unique())}
    - ISIC_2020_Training: {len(df[df['source_csv'] == 'ISIC_2020_Training_GroundTruth.csv']):,} images
    - ISIC_2019_Training: {len(df[df['source_csv'] == 'ISIC_2019_Training_GroundTruth.csv']):,} images
    - ISIC2018_Task3_Training: {len(df[df['source_csv'] == 'ISIC2018_Task3_Training_GroundTruth.csv']):,} images
    - ISIC_2019_Test: {len(df[df['source_csv'] == 'ISIC_2019_Test_GroundTruth.csv']):,} images
    - imagenet_ood: {len(df[df['source_csv'] == 'imagenet_ood']):,} images
    - ISIC2018_Task3_Validation: {len(df[df['source_csv'] == 'ISIC2018_Task3_Validation_GroundTruth.csv']):,} images
    
    Labels: {len(df['unified_label'].unique())}
    - nevus: {len(df[df['unified_label'] == 'nevus']):,} images
    - other: {len(df[df['unified_label'] == 'other']):,} images
    - melanoma: {len(df[df['unified_label'] == 'melanoma']):,} images
    - bcc: {len(df[df['unified_label'] == 'bcc']):,} images
    - bkl: {len(df[df['unified_label'] == 'bkl']):,} images
    - ak: {len(df[df['unified_label'] == 'ak']):,} images
    - scc: {len(df[df['unified_label'] == 'scc']):,} images
    - df: {len(df[df['unified_label'] == 'df']):,} images
    - vascular: {len(df[df['unified_label'] == 'vascular']):,} images
    - non_skin: {len(df[df['unified_label'] == 'non_skin']):,} images
    
    Splits: 5 experimental splits created
    - split1: {df['split1'].sum():,} images (full dataset)
    - split2: {df['split2'].sum():,} images (balanced subset)
    - split3: {df['split3'].sum():,} images (rare-first balancing)
    - split4: {df['split4'].sum():,} images (maximize minorities)
    - split5: {df['split5'].sum():,} images (small classes only)
    """
    
    ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, fontsize=10,
              verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('results/dataset_stats/comprehensive_summary.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to create all visualizations."""
    print("Loading dataset...")
    df = load_data()
    
    # Create results directory
    Path('results/dataset_stats').mkdir(parents=True, exist_ok=True)
    
    print("Creating visualizations...")
    
    # Create individual plots
    print("1. Creating source distribution plot...")
    create_source_distribution_plot(df)
    
    print("2. Creating label distribution plot...")
    create_label_distribution_plot(df)
    
    print("3. Creating split analysis plot...")
    create_split_analysis_plot(df)
    
    print("4. Creating stratified split visualization...")
    create_stratified_split_visualization()
    
    print("5. Creating comprehensive summary plot...")
    create_comprehensive_summary_plot(df)
    
    print("All visualizations saved to results/dataset_stats/")
    print("Generated files:")
    print("- source_distribution.png")
    print("- label_distribution.png") 
    print("- split_analysis.png")
    print("- stratified_splits.png")
    print("- comprehensive_summary.png")

if __name__ == "__main__":
    main() 