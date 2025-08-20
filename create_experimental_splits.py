import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
np.random.seed(42)

# Load current metadata
df = pd.read_csv('data/metadata/metadata.csv')
print(f"Loaded metadata with {len(df)} images")

# Initialize exp columns
df['exp1'] = 0
df['exp2'] = 0
df['exp3'] = 0
df['exp4'] = 0
df['exp5'] = 0

# exp1: Full Dataset (all 104,397 images)
df['exp1'] = 1
print(f"exp1 (Full Dataset): {df['exp1'].sum()} images")

# exp2: Balanced Subset (~13,782 images - following protocol)
# Sample balanced representation from each source
exp2_indices = []
for source in df['source_csv'].unique():
    source_data = df[df['source_csv'] == source]
    if source == 'ImageNet_Sampled':
        # Include all ImageNet
        exp2_indices.extend(source_data.index.tolist())
    else:
        # Sample proportionally from ISIC sources
        target_size = int(13782 * len(source_data) / len(df))
        sampled = source_data.sample(n=min(target_size, len(source_data)), random_state=42)
        exp2_indices.extend(sampled.index.tolist())

df.loc[exp2_indices, 'exp2'] = 1
print(f"exp2 (Balanced Subset): {df['exp2'].sum()} images")

# exp3: Rare-First Balancing (~4,316 images - following protocol)
# Focus on rare classes and ImageNet
exp3_indices = []
# Include all ImageNet
imagenet_indices = df[df['source_csv'] == 'ImageNet_Sampled'].index.tolist()
exp3_indices.extend(imagenet_indices)

# Add rare medical classes
rare_classes = ['melanoma', 'bcc', 'ak', 'akiec', 'scc', 'vasc', 'df']
for class_name in rare_classes:
    class_data = df[(df['unified_label'] == class_name) & (df['source_csv'] != 'ImageNet_Sampled')]
    if len(class_data) > 0:
        exp3_indices.extend(class_data.index.tolist())

# Limit to target size
if len(exp3_indices) > 4316:
    exp3_indices = np.random.choice(exp3_indices, 4316, replace=False)

df.loc[exp3_indices, 'exp3'] = 1
print(f"exp3 (Rare-First Balancing): {df['exp3'].sum()} images")

# exp4: Maximize Minorities (~22,971 images - following protocol)
# Focus on minority classes and ImageNet
exp4_indices = []
# Include all ImageNet
exp4_indices.extend(imagenet_indices)

# Add minority classes (not nevus/unknown)
minority_classes = ['melanoma', 'bcc', 'ak', 'akiec', 'scc', 'vasc', 'df', 'bkl', 'lentigo']
for class_name in minority_classes:
    class_data = df[(df['unified_label'] == class_name) & (df['source_csv'] != 'ImageNet_Sampled')]
    if len(class_data) > 0:
        exp4_indices.extend(class_data.index.tolist())

# Limit to target size
if len(exp4_indices) > 22971:
    exp4_indices = np.random.choice(exp4_indices, 22971, replace=False)

df.loc[exp4_indices, 'exp4'] = 1
print(f"exp4 (Maximize Minorities): {df['exp4'].sum()} images")

# exp5: Small Classes Only (~3,782 images - following protocol)
# Focus on very small classes and ImageNet
exp5_indices = []
# Include all ImageNet
exp5_indices.extend(imagenet_indices)

# Add only very small classes
small_classes = ['ak', 'akiec', 'scc', 'vasc', 'df', 'lentigo', 'cafe_au_lait', 'atypical_melanocytic']
for class_name in small_classes:
    class_data = df[(df['unified_label'] == class_name) & (df['source_csv'] != 'ImageNet_Sampled')]
    if len(class_data) > 0:
        exp5_indices.extend(class_data.index.tolist())

# Limit to target size
if len(exp5_indices) > 3782:
    exp5_indices = np.random.choice(exp5_indices, 3782, replace=False)

df.loc[exp5_indices, 'exp5'] = 1
print(f"exp5 (Small Classes Only): {df['exp5'].sum()} images")

# Save updated metadata
df.to_csv('data/metadata/metadata.csv', index=False)

print(f"\nExperimental splits created and saved!")
print(f"exp1: {df['exp1'].sum()} images")
print(f"exp2: {df['exp2'].sum()} images")
print(f"exp3: {df['exp3'].sum()} images")
print(f"exp4: {df['exp4'].sum()} images")
print(f"exp5: {df['exp5'].sum()} images")

# Show overlap analysis
print(f"\nOverlap Analysis:")
print(f"exp1 ∩ exp2: {(df['exp1'] & df['exp2']).sum()} images")
print(f"exp1 ∩ exp3: {(df['exp1'] & df['exp3']).sum()} images")
print(f"exp2 ∩ exp3: {(df['exp2'] & df['exp3']).sum()} images")
print(f"exp2 ∩ exp4: {(df['exp2'] & df['exp4']).sum()} images")
print(f"exp3 ∩ exp4: {(df['exp3'] & df['exp4']).sum()} images")
print(f"exp3 ∩ exp5: {(df['exp3'] & df['exp5']).sum()} images") 