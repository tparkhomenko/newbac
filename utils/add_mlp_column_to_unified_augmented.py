import pandas as pd

# Load the CSV
csv_path = 'datasets/metadata/unified_augmented.csv'
df = pd.read_csv(csv_path)

mlp_labels = []
for idx, row in df.iterrows():
    mlps = ['MLP1']  # All images are used for MLP1 (skin/not-skin)
    # MLP2: skin lesion, valid lesion group
    if row['skin'] == 1 and row['lesion_group'] not in ['unknown', 'not_skin_texture']:
        mlps.append('MLP2')
    # MLP3: skin lesion, valid malignancy
    if row['skin'] == 1 and row['malignancy'] in ['benign', 'malignant']:
        mlps.append('MLP3')
    mlp_labels.append(';'.join(mlps))

df['MLP'] = mlp_labels

# Save to new CSV
out_path = 'datasets/metadata/unified_augmented_with_mlp.csv'
df.to_csv(out_path, index=False)
print(f"Saved with MLP column to {out_path}") 