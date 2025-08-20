import os
import sys
import argparse

# Use a non-interactive backend for headless environments
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def generate_class_distribution_all_splits(csv_path: str, output_dir: str) -> str:
    """Recreate class_distribution_all_splits.png using the original notebook logic.

    Reads unified labels with stratified splits and produces a stacked bar chart of
    class counts per split group (train/val/test for split1..split5).
    """
    sns.set(style="whitegrid")

    df = pd.read_csv(csv_path)

    split_names = [f'split{i}' for i in range(1, 6)]
    groups = ['train', 'val', 'test']

    # Compute counts per class for each split/group combination
    from collections import defaultdict
    split_class_counts = defaultdict(lambda: defaultdict(int))
    for split in split_names:
        if split not in df.columns:
            continue
        for g in groups:
            mask = df[split] == g
            counts = df.loc[mask, 'unified_label'].value_counts()
            for label, count in counts.items():
                split_class_counts[f'{split}_{g}'][label] = count

    # Prepare DataFrame for plotting
    plot_df = pd.DataFrame(split_class_counts).fillna(0).astype(int)
    plot_df = plot_df.T  # rows: split_group, columns: class
    plot_df.index.name = 'split_group'

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'class_distribution_all_splits.png')

    fig, ax = plt.subplots(figsize=(14, 7))
    plot_df.plot(kind='bar', stacked=True, ax=ax, colormap='tab10')
    plt.title('Stacked Class Distribution per Split')
    plt.xlabel('Split and Group')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Diagnosis', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


def main():
    parser = argparse.ArgumentParser(description='Recreate class_distribution_all_splits.png using notebook logic')
    parser.add_argument('--csv', default='data/metadata/unified_labels_with_stratified_splits.csv', help='Path to unified labels with stratified splits CSV')
    parser.add_argument('--out', default='data/metadata', help='Directory to save the plot')
    args = parser.parse_args()

    output_path = generate_class_distribution_all_splits(args.csv, args.out)
    print(f'Saved: {output_path}')


if __name__ == '__main__':
    main()


