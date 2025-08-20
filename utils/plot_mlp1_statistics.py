import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')

def load_and_analyze_data():
    """Load and analyze the MLP1 metadata"""
    # Load data
    df = pd.read_csv('datasets/metadata/metadata_skin_not_skin_split.csv')
    
    print(f"Dataset Overview:")
    print(f"Total images: {len(df):,}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return df

def create_comprehensive_plots(df):
    """Create comprehensive statistics plots"""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 24))
    
    # 1. Overall dataset distribution
    ax1 = plt.subplot(4, 3, 1)
    skin_counts = df['skin'].value_counts()
    colors = ['#ff9999', '#66b3ff']
    labels = ['Non-Skin (DTD)', 'Skin (ISIC)']
    plt.pie(skin_counts.values, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    plt.title('Overall Dataset Distribution\n(Skin vs Non-Skin)', fontsize=14, fontweight='bold')
    
    # 2. Split distribution
    ax2 = plt.subplot(4, 3, 2)
    split_counts = df['split'].value_counts()
    colors_split = ['#2ecc71', '#f39c12', '#e74c3c']
    bars = plt.bar(split_counts.index, split_counts.values, color=colors_split, alpha=0.8)
    plt.title('Dataset Split Distribution', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Images')
    
    # Add value labels on bars
    for bar, value in zip(bars, split_counts.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, 
                f'{value:,}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Split distribution by skin/non-skin
    ax3 = plt.subplot(4, 3, 3)
    split_skin = pd.crosstab(df['split'], df['skin'], normalize='index') * 100
    split_skin.plot(kind='bar', stacked=True, ax=ax3, color=['#ff9999', '#66b3ff'])
    plt.title('Split Distribution by Class', fontsize=14, fontweight='bold')
    plt.ylabel('Percentage (%)')
    plt.xlabel('Split')
    plt.legend(['Non-Skin', 'Skin'])
    plt.xticks(rotation=0)
    
    # 4. Lesion group distribution (for skin images only)
    ax4 = plt.subplot(4, 3, 4)
    skin_df = df[df['skin'] == 1]
    lesion_counts = skin_df['lesion_group'].value_counts()
    plt.barh(range(len(lesion_counts)), lesion_counts.values, color='skyblue', alpha=0.8)
    plt.yticks(range(len(lesion_counts)), lesion_counts.index)
    plt.title('Lesion Group Distribution\n(Skin Images Only)', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Images')
    
    # Add value labels
    for i, v in enumerate(lesion_counts.values):
        plt.text(v + 100, i, f'{v:,}', va='center', fontweight='bold')
    
    # 5. Malignancy distribution (for skin images only)
    ax5 = plt.subplot(4, 3, 5)
    malignancy_counts = skin_df['malignancy'].value_counts()
    colors_mal = ['#2ecc71', '#e74c3c', '#95a5a6']
    plt.pie(malignancy_counts.values, labels=malignancy_counts.index, autopct='%1.1f%%', 
            colors=colors_mal, startangle=90)
    plt.title('Malignancy Distribution\n(Skin Images Only)', fontsize=14, fontweight='bold')
    
    # 6. Split distribution by lesion group
    ax6 = plt.subplot(4, 3, 6)
    lesion_split = pd.crosstab(skin_df['lesion_group'], skin_df['split'])
    lesion_split.plot(kind='bar', ax=ax6, color=['#2ecc71', '#f39c12', '#e74c3c'])
    plt.title('Split Distribution by Lesion Group', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Images')
    plt.xlabel('Lesion Group')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Split')
    
    # 7. Augmentation analysis
    ax7 = plt.subplot(4, 3, 7)
    # Extract augmentation type from image names
    df['augmentation'] = df['image'].str.extract(r'_([^_]+)\.jpg$').fillna('original')
    aug_counts = df['augmentation'].value_counts()
    plt.bar(aug_counts.index, aug_counts.values, color='lightcoral', alpha=0.8)
    plt.title('Augmentation Distribution', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Images')
    plt.xlabel('Augmentation Type')
    plt.xticks(rotation=45)
    
    # Add value labels
    for i, v in enumerate(aug_counts.values):
        plt.text(i, v + 100, f'{v:,}', ha='center', va='bottom', fontweight='bold')
    
    # 8. Augmentation by class
    ax8 = plt.subplot(4, 3, 8)
    aug_skin = pd.crosstab(df['augmentation'], df['skin'], normalize='index') * 100
    aug_skin.plot(kind='bar', stacked=True, ax=ax8, color=['#ff9999', '#66b3ff'])
    plt.title('Augmentation Distribution by Class', fontsize=14, fontweight='bold')
    plt.ylabel('Percentage (%)')
    plt.xlabel('Augmentation Type')
    plt.legend(['Non-Skin', 'Skin'])
    plt.xticks(rotation=45)
    
    # 9. Detailed split statistics
    ax9 = plt.subplot(4, 3, 9)
    split_stats = df.groupby(['split', 'skin']).size().unstack(fill_value=0)
    split_stats.plot(kind='bar', ax=ax9, color=['#ff9999', '#66b3ff'])
    plt.title('Detailed Split Statistics', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Images')
    plt.xlabel('Split')
    plt.legend(['Non-Skin', 'Skin'])
    plt.xticks(rotation=0)
    
    # Add value labels
    for i, (split, row) in enumerate(split_stats.iterrows()):
        for j, value in enumerate(row):
            plt.text(i, value + 50, f'{value:,}', ha='center', va='bottom', fontweight='bold')
    
    # 10. Lesion type distribution in each split
    ax10 = plt.subplot(4, 3, 10)
    lesion_split_percent = pd.crosstab(skin_df['lesion_group'], skin_df['split'], normalize='columns') * 100
    lesion_split_percent.plot(kind='bar', ax=ax10, color=['#2ecc71', '#f39c12', '#e74c3c'])
    plt.title('Lesion Group Distribution by Split (%)', fontsize=14, fontweight='bold')
    plt.ylabel('Percentage (%)')
    plt.xlabel('Lesion Group')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Split')
    
    # 11. Malignancy distribution in each split
    ax11 = plt.subplot(4, 3, 11)
    mal_split_percent = pd.crosstab(skin_df['malignancy'], skin_df['split'], normalize='columns') * 100
    mal_split_percent.plot(kind='bar', ax=ax11, color=['#2ecc71', '#e74c3c', '#95a5a6'])
    plt.title('Malignancy Distribution by Split (%)', fontsize=14, fontweight='bold')
    plt.ylabel('Percentage (%)')
    plt.xlabel('Malignancy')
    plt.xticks(rotation=0)
    plt.legend(title='Split')
    
    # 12. Summary statistics table
    ax12 = plt.subplot(4, 3, 12)
    ax12.axis('tight')
    ax12.axis('off')
    
    # Calculate summary statistics
    total_images = len(df)
    skin_images = len(df[df['skin'] == 1])
    non_skin_images = len(df[df['skin'] == 0])
    
    train_images = len(df[df['split'] == 'train'])
    val_images = len(df[df['split'] == 'val'])
    test_images = len(df[df['split'] == 'test'])
    
    # Create summary table
    summary_data = [
        ['Total Images', f'{total_images:,}'],
        ['Skin Images', f'{skin_images:,} ({skin_images/total_images*100:.1f}%)'],
        ['Non-Skin Images', f'{non_skin_images:,} ({non_skin_images/total_images*100:.1f}%)'],
        ['', ''],
        ['Train Split', f'{train_images:,} ({train_images/total_images*100:.1f}%)'],
        ['Val Split', f'{val_images:,} ({val_images/total_images*100:.1f}%)'],
        ['Test Split', f'{test_images:,} ({test_images/total_images*100:.1f}%)'],
        ['', ''],
        ['Unique Lesion Groups', f'{len(skin_df["lesion_group"].unique())}'],
        ['Augmentation Types', f'{len(df["augmentation"].unique())}']
    ]
    
    table = ax12.table(cellText=summary_data, colLabels=['Metric', 'Value'], 
                      cellLoc='left', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(summary_data)):
        for j in range(2):
            cell = table[(i+1, j)]
            if i in [3, 7]:  # Empty rows
                cell.set_facecolor('#f0f0f0')
            else:
                cell.set_facecolor('#ffffff')
    
    # Style header
    for j in range(2):
        table[(0, j)].set_facecolor('#4CAF50')
        table[(0, j)].set_text_props(weight='bold', color='white')
    
    plt.title('Dataset Summary Statistics', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig

def create_additional_analysis(df):
    """Create additional detailed analysis"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Class balance in each split
    ax1 = axes[0, 0]
    split_balance = df.groupby(['split', 'skin']).size().unstack(fill_value=0)
    split_balance_percent = split_balance.div(split_balance.sum(axis=1), axis=0) * 100
    
    bars = split_balance_percent.plot(kind='bar', ax=ax1, color=['#ff9999', '#66b3ff'])
    plt.title('Class Balance in Each Split', fontsize=14, fontweight='bold')
    plt.ylabel('Percentage (%)')
    plt.xlabel('Split')
    plt.legend(['Non-Skin', 'Skin'])
    plt.xticks(rotation=0)
    
    # Add percentage labels
    for i, (split, row) in enumerate(split_balance_percent.iterrows()):
        for j, value in enumerate(row):
            plt.text(i, value + 1, f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 2. Lesion group distribution with counts
    ax2 = axes[0, 1]
    skin_df = df[df['skin'] == 1]
    lesion_counts = skin_df['lesion_group'].value_counts()
    
    bars = plt.bar(range(len(lesion_counts)), lesion_counts.values, color='lightblue', alpha=0.8)
    plt.title('Lesion Group Distribution (Counts)', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Images')
    plt.xlabel('Lesion Group')
    plt.xticks(range(len(lesion_counts)), lesion_counts.index, rotation=45, ha='right')
    
    # Add value labels
    for i, v in enumerate(lesion_counts.values):
        plt.text(i, v + 50, f'{v:,}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Augmentation distribution by dataset source
    ax3 = axes[1, 0]
    df['source'] = df['image'].apply(lambda x: 'ISIC' if 'ISIC' in x else 'DTD')
    aug_source = pd.crosstab(df['augmentation'], df['source'])
    aug_source.plot(kind='bar', ax=ax3, color=['#ff6b6b', '#4ecdc4'])
    plt.title('Augmentation Distribution by Source', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Images')
    plt.xlabel('Augmentation Type')
    plt.legend(['DTD', 'ISIC'])
    plt.xticks(rotation=45)
    
    # 4. Malignancy distribution by lesion group
    ax4 = axes[1, 1]
    mal_lesion = pd.crosstab(skin_df['lesion_group'], skin_df['malignancy'])
    mal_lesion.plot(kind='bar', ax=ax4, color=['#2ecc71', '#e74c3c', '#95a5a6'])
    plt.title('Malignancy Distribution by Lesion Group', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Images')
    plt.xlabel('Lesion Group')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Malignancy')
    
    plt.tight_layout()
    return fig

def main():
    """Main function to create all plots"""
    print("Loading MLP1 metadata...")
    df = load_and_analyze_data()
    
    print("\nCreating comprehensive statistics plots...")
    
    # Create main comprehensive plot
    fig1 = create_comprehensive_plots(df)
    
    # Save the main plot
    output_dir = Path('results/statistics')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig1.savefig(str(output_dir / 'mlp1_comprehensive_statistics.png'), dpi=300, bbox_inches='tight')
    print(f"Main statistics plot saved to: {output_dir / 'mlp1_comprehensive_statistics.png'}")
    
    # Create additional analysis
    fig2 = create_additional_analysis(df)
    fig2.savefig(str(output_dir / 'mlp1_additional_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"Additional analysis plot saved to: {output_dir / 'mlp1_additional_analysis.png'}")
    
    # Print detailed statistics
    print("\n" + "="*60)
    print("DETAILED DATASET STATISTICS")
    print("="*60)
    
    print(f"\nOverall Statistics:")
    print(f"Total images: {len(df):,}")
    print(f"Skin images: {len(df[df['skin'] == 1]):,} ({len(df[df['skin'] == 1])/len(df)*100:.1f}%)")
    print(f"Non-skin images: {len(df[df['skin'] == 0]):,} ({len(df[df['skin'] == 0])/len(df)*100:.1f}%)")
    
    print(f"\nSplit Distribution:")
    for split in ['train', 'val', 'test']:
        split_df = df[df['split'] == split]
        skin_count = len(split_df[split_df['skin'] == 1])
        non_skin_count = len(split_df[split_df['skin'] == 0])
        print(f"{split.capitalize()}: {len(split_df):,} images ({skin_count:,} skin, {non_skin_count:,} non-skin)")
    
    print(f"\nLesion Group Distribution (Skin Images Only):")
    skin_df = df[df['skin'] == 1]
    for group, count in skin_df['lesion_group'].value_counts().items():
        print(f"  {group}: {count:,} images")
    
    print(f"\nAugmentation Distribution:")
    df['augmentation'] = df['image'].str.extract(r'_([^_]+)\.jpg$').fillna('original')
    for aug, count in df['augmentation'].value_counts().items():
        print(f"  {aug}: {count:,} images")
    
    plt.show()

if __name__ == "__main__":
    main()