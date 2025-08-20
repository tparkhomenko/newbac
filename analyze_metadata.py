import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set(style="whitegrid")

# File paths
metadata_path = "data/metadata/metadata.csv"
output_dir = "data/metadata/"
os.makedirs(output_dir, exist_ok=True)

print("=== METADATA.CSV ANALYSIS WITH EXPERIMENTAL SPLITS ===")

# Read the metadata.csv with experimental splits
metadata_df = pd.read_csv(metadata_path)

print(f"\nTotal images in metadata.csv: {len(metadata_df):,}")
print(f"Columns: {list(metadata_df.columns)}")

# Basic statistics for each experimental split
exp_splits = ['exp1', 'exp2', 'exp3', 'exp4', 'exp5']
print(f"\n=== EXPERIMENTAL SPLIT STATISTICS ===")

for exp in exp_splits:
    count = metadata_df[exp].sum()
    print(f"{exp}: {count:,} images")

# Diagnosis distribution across all data
print(f"\n=== DIAGNOSIS DISTRIBUTION (ALL DATA) ===")
diagnosis_counts = metadata_df['diagnosis'].value_counts()
for diagnosis, count in diagnosis_counts.items():
    print(f"{diagnosis}: {count:,} images")

# Unified label distribution across all data
print(f"\n=== UNIFIED LABEL DISTRIBUTION (ALL DATA) ===")
label_counts = metadata_df['unified_label'].value_counts()
for label, count in label_counts.items():
    print(f"{label}: {count:,} images")

# Source dataset distribution
print(f"\n=== SOURCE DATASET DISTRIBUTION ===")
source_counts = metadata_df['source_csv'].value_counts()
for source, count in source_counts.items():
    print(f"{source}: {count:,} images")

# Experimental split overlap analysis
print(f"\n=== EXPERIMENTAL SPLIT OVERLAP ANALYSIS ===")
for i, exp1 in enumerate(exp_splits):
    for j, exp2 in enumerate(exp_splits):
        if i < j:
            overlap = (metadata_df[exp1] & metadata_df[exp2]).sum()
            print(f"{exp1} ∩ {exp2}: {overlap:,} images")

# Class distribution within each experimental split
print(f"\n=== CLASS DISTRIBUTION WITHIN EXPERIMENTAL SPLITS ===")
for exp in exp_splits:
    print(f"\n{exp.upper()} - Class Distribution:")
    exp_data = metadata_df[metadata_df[exp] == 1]
    class_counts = exp_data['unified_label'].value_counts()
    for label, count in class_counts.items():
        print(f"  {label}: {count:,} images")

# Source distribution within each experimental split
print(f"\n=== SOURCE DISTRIBUTION WITHIN EXPERIMENTAL SPLITS ===")
for exp in exp_splits:
    print(f"\n{exp.upper()} - Source Distribution:")
    exp_data = metadata_df[metadata_df[exp] == 1]
    source_counts = exp_data['source_csv'].value_counts()
    for source, count in source_counts.items():
        print(f"  {source}: {count:,} images")

# Create visualizations for experimental splits
print(f"\n=== CREATING VISUALIZATIONS ===")

# 1. Experimental split sizes
plt.figure(figsize=(10,6))
exp_counts = [metadata_df[exp].sum() for exp in exp_splits]
plt.bar(exp_splits, exp_counts, color='skyblue')
plt.title('Experimental Split Sizes')
plt.xlabel('Experimental Split')
plt.ylabel('Number of Images')
plt.xticks(rotation=45)
for i, v in enumerate(exp_counts):
    plt.text(i, v + 1000, f'{v:,}', ha='center', va='bottom')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'experimental_split_sizes.png'))
plt.close()

# 2. Class distribution heatmap across experimental splits
exp_class_data = []
for exp in exp_splits:
    exp_data = metadata_df[metadata_df[exp] == 1]
    class_counts = exp_data['unified_label'].value_counts()
    exp_class_data.append([class_counts.get(label, 0) for label in sorted(metadata_df['unified_label'].unique())])

exp_class_df = pd.DataFrame(exp_class_data, 
                           columns=sorted(metadata_df['unified_label'].unique()), 
                           index=exp_splits)

plt.figure(figsize=(14,8))
sns.heatmap(exp_class_df, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Image Count'})
plt.title('Class Distribution Across Experimental Splits')
plt.xlabel('Unified Label (Diagnosis)')
plt.ylabel('Experimental Split')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'experimental_split_class_heatmap.png'))
plt.close()

# 3. Source distribution across experimental splits
exp_source_data = []
for exp in exp_splits:
    exp_data = metadata_df[metadata_df[exp] == 1]
    source_counts = exp_data['source_csv'].value_counts()
    exp_source_data.append([source_counts.get(source, 0) for source in sorted(metadata_df['source_csv'].unique())])

exp_source_df = pd.DataFrame(exp_source_data, 
                            columns=sorted(metadata_df['source_csv'].unique()), 
                            index=exp_splits)

plt.figure(figsize=(14,8))
sns.heatmap(exp_source_df, annot=True, fmt='d', cmap='Greens', cbar_kws={'label': 'Image Count'})
plt.title('Source Distribution Across Experimental Splits')
plt.xlabel('Source Dataset')
plt.ylabel('Experimental Split')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'experimental_split_source_heatmap.png'))
plt.close()

print(f"\nVisualizations saved to {output_dir}")
print("=== METADATA.CSV ANALYSIS COMPLETE ===") 