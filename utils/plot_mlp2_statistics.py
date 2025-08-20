import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')

def load_and_analyze_data():
    df = pd.read_csv('datasets/metadata/train_balanced_max.csv')
    print(f"Dataset Overview:")
    print(f"Total images: {len(df):,}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    return df

def create_comprehensive_plots(df):
    fig = plt.figure(figsize=(20, 24))
    # 1. Lesion group distribution
    ax1 = plt.subplot(4, 3, 1)
    lesion_counts = df['lesion_group'].value_counts()
    plt.pie(lesion_counts.values, labels=lesion_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('Lesion Group Distribution', fontsize=14, fontweight='bold')
    # 2. Split distribution
    ax2 = plt.subplot(4, 3, 2)
    split_counts = df['split'].value_counts()
    bars = plt.bar(split_counts.index, split_counts.values, color=['#2ecc71', '#f39c12', '#e74c3c'], alpha=0.8)
    plt.title('Dataset Split Distribution', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Images')
    for bar, value in zip(bars, split_counts.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, f'{value:,}', ha='center', va='bottom', fontweight='bold')
    # 3. Split distribution by lesion group
    ax3 = plt.subplot(4, 3, 3)
    lesion_split = pd.crosstab(df['split'], df['lesion_group'], normalize='index') * 100
    lesion_split.plot(kind='bar', stacked=True, ax=ax3)
    plt.title('Split Distribution by Lesion Group', fontsize=14, fontweight='bold')
    plt.ylabel('Percentage (%)')
    plt.xlabel('Split')
    plt.legend(title='Lesion Group')
    plt.xticks(rotation=0)
    # 4. Malignancy distribution
    ax4 = plt.subplot(4, 3, 4)
    mal_counts = df['malignancy'].value_counts()
    plt.pie(mal_counts.values, labels=mal_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('Malignancy Distribution', fontsize=14, fontweight='bold')
    # 5. Lesion group by split (counts)
    ax5 = plt.subplot(4, 3, 5)
    lesion_split_counts = pd.crosstab(df['lesion_group'], df['split'])
    lesion_split_counts.plot(kind='bar', ax=ax5)
    plt.title('Lesion Group by Split (Counts)', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Images')
    plt.xlabel('Lesion Group')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Split')
    # 6. Augmentation analysis
    ax6 = plt.subplot(4, 3, 6)
    df['augmentation'] = df['image'].str.extract(r'_([^_]+)\.jpg$').fillna('original')
    aug_counts = df['augmentation'].value_counts()
    plt.bar(aug_counts.index, aug_counts.values, color='lightcoral', alpha=0.8)
    plt.title('Augmentation Distribution', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Images')
    plt.xlabel('Augmentation Type')
    plt.xticks(rotation=45)
    for i, v in enumerate(aug_counts.values):
        plt.text(i, v + 100, f'{v:,}', ha='center', va='bottom', fontweight='bold')
    # 7. Augmentation by lesion group
    ax7 = plt.subplot(4, 3, 7)
    aug_lesion = pd.crosstab(df['augmentation'], df['lesion_group'], normalize='index') * 100
    aug_lesion.plot(kind='bar', stacked=True, ax=ax7)
    plt.title('Augmentation Distribution by Lesion Group', fontsize=14, fontweight='bold')
    plt.ylabel('Percentage (%)')
    plt.xlabel('Augmentation Type')
    plt.legend(title='Lesion Group')
    plt.xticks(rotation=45)
    # 8. Malignancy by lesion group
    ax8 = plt.subplot(4, 3, 8)
    mal_lesion = pd.crosstab(df['lesion_group'], df['malignancy'], normalize='index') * 100
    mal_lesion.plot(kind='bar', stacked=True, ax=ax8)
    plt.title('Malignancy by Lesion Group (%)', fontsize=14, fontweight='bold')
    plt.ylabel('Percentage (%)')
    plt.xlabel('Lesion Group')
    plt.legend(title='Malignancy')
    plt.xticks(rotation=45, ha='right')
    # 9. Detailed split statistics
    ax9 = plt.subplot(4, 3, 9)
    split_stats = df.groupby(['split', 'lesion_group']).size().unstack(fill_value=0)
    split_stats.plot(kind='bar', ax=ax9)
    plt.title('Detailed Split Statistics', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Images')
    plt.xlabel('Split')
    plt.legend(title='Lesion Group')
    plt.xticks(rotation=0)
    # 10. Lesion group distribution in each split (%)
    ax10 = plt.subplot(4, 3, 10)
    lesion_split_percent = pd.crosstab(df['lesion_group'], df['split'], normalize='columns') * 100
    lesion_split_percent.plot(kind='bar', ax=ax10)
    plt.title('Lesion Group Distribution by Split (%)', fontsize=14, fontweight='bold')
    plt.ylabel('Percentage (%)')
    plt.xlabel('Lesion Group')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Split')
    # 11. Malignancy distribution in each split (%)
    ax11 = plt.subplot(4, 3, 11)
    mal_split_percent = pd.crosstab(df['malignancy'], df['split'], normalize='columns') * 100
    mal_split_percent.plot(kind='bar', ax=ax11)
    plt.title('Malignancy Distribution by Split (%)', fontsize=14, fontweight='bold')
    plt.ylabel('Percentage (%)')
    plt.xlabel('Malignancy')
    plt.xticks(rotation=0)
    plt.legend(title='Split')
    # 12. Summary statistics table
    ax12 = plt.subplot(4, 3, 12)
    ax12.axis('tight')
    ax12.axis('off')
    total_images = len(df)
    lesion_groups = len(df['lesion_group'].unique())
    mal_types = len(df['malignancy'].unique())
    train_images = len(df[df['split'] == 'train'])
    val_images = len(df[df['split'] == 'val'])
    test_images = len(df[df['split'] == 'test'])
    summary_data = [
        ['Total Images', f'{total_images:,}'],
        ['Unique Lesion Groups', f'{lesion_groups}'],
        ['Malignancy Types', f'{mal_types}'],
        ['', ''],
        ['Train Split', f'{train_images:,} ({train_images/total_images*100:.1f}%)'],
        ['Val Split', f'{val_images:,} ({val_images/total_images*100:.1f}%)'],
        ['Test Split', f'{test_images:,} ({test_images/total_images*100:.1f}%)'],
        ['', ''],
        ['Augmentation Types', f'{len(df["augmentation"].unique())}']
    ]
    table = ax12.table(cellText=summary_data, colLabels=['Metric', 'Value'], cellLoc='left', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    for i in range(len(summary_data)):
        for j in range(2):
            cell = table[(i+1, j)]
            if i in [3, 7]:
                cell.set_facecolor('#f0f0f0')
            else:
                cell.set_facecolor('#ffffff')
    for j in range(2):
        table[(0, j)].set_facecolor('#4CAF50')
        table[(0, j)].set_text_props(weight='bold', color='white')
    plt.title('Dataset Summary Statistics', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    return fig

def create_additional_analysis(df):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    # 1. Lesion group balance in each split
    ax1 = axes[0, 0]
    split_balance = df.groupby(['split', 'lesion_group']).size().unstack(fill_value=0)
    split_balance_percent = split_balance.div(split_balance.sum(axis=1), axis=0) * 100
    split_balance_percent.plot(kind='bar', ax=ax1)
    plt.title('Lesion Group Balance in Each Split', fontsize=14, fontweight='bold')
    plt.ylabel('Percentage (%)')
    plt.xlabel('Split')
    plt.legend(title='Lesion Group')
    plt.xticks(rotation=0)
    # 2. Lesion group distribution with counts
    ax2 = axes[0, 1]
    lesion_counts = df['lesion_group'].value_counts()
    bars = plt.bar(range(len(lesion_counts)), lesion_counts.values, color='lightblue', alpha=0.8)
    plt.title('Lesion Group Distribution (Counts)', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Images')
    plt.xlabel('Lesion Group')
    plt.xticks(range(len(lesion_counts)), lesion_counts.index, rotation=45, ha='right')
    for i, v in enumerate(lesion_counts.values):
        plt.text(i, v + 50, f'{v:,}', ha='center', va='bottom', fontweight='bold')
    # 3. Augmentation distribution by lesion group
    ax3 = axes[1, 0]
    aug_lesion = pd.crosstab(df['augmentation'], df['lesion_group'])
    aug_lesion.plot(kind='bar', ax=ax3)
    plt.title('Augmentation Distribution by Lesion Group', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Images')
    plt.xlabel('Augmentation Type')
    plt.legend(title='Lesion Group')
    plt.xticks(rotation=45)
    # 4. Malignancy distribution by lesion group
    ax4 = axes[1, 1]
    mal_lesion = pd.crosstab(df['lesion_group'], df['malignancy'])
    mal_lesion.plot(kind='bar', ax=ax4)
    plt.title('Malignancy Distribution by Lesion Group', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Images')
    plt.xlabel('Lesion Group')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Malignancy')
    plt.tight_layout()
    return fig

def main():
    print("Loading MLP2 metadata...")
    df = load_and_analyze_data()
    print("\nCreating comprehensive statistics plots...")
    fig1 = create_comprehensive_plots(df)
    output_dir = Path('results/statistics')
    output_dir.mkdir(parents=True, exist_ok=True)
    fig1.savefig(str(output_dir / 'mlp2_comprehensive_statistics.png'), dpi=300, bbox_inches='tight')
    print(f"Main statistics plot saved to: {output_dir / 'mlp2_comprehensive_statistics.png'}")
    fig2 = create_additional_analysis(df)
    fig2.savefig(str(output_dir / 'mlp2_additional_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"Additional analysis plot saved to: {output_dir / 'mlp2_additional_analysis.png'}")
    print("\n" + "="*60)
    print("DETAILED DATASET STATISTICS")
    print("="*60)
    print(f"\nOverall Statistics:")
    print(f"Total images: {len(df):,}")
    print(f"Lesion groups: {df['lesion_group'].unique()}")
    print(f"Malignancy types: {df['malignancy'].unique()}")
    print(f"\nSplit Distribution:")
    for split in ['train', 'val', 'test']:
        split_df = df[df['split'] == split]
        print(f"{split.capitalize()}: {len(split_df):,} images")
    print(f"\nAugmentation Distribution:")
    for aug, count in df['augmentation'].value_counts().items():
        print(f"  {aug}: {count:,} images")
    plt.show()

if __name__ == "__main__":
    main() 