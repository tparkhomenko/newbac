import pandas as pd

# Load the CSV
csv_path = 'datasets/metadata/unified_augmented.csv'
df = pd.read_csv(csv_path)

mlp1 = []
mlp2 = []
mlp3 = []
for idx, row in df.iterrows():
    # MLP1: all images
    mlp1.append(1)
    # MLP2: skin lesion, valid lesion group
    mlp2.append(1 if row['skin'] == 1 and row['lesion_group'] not in ['unknown', 'not_skin_texture'] else 0)
    # MLP3: skin lesion, valid malignancy
    mlp3.append(1 if row['skin'] == 1 and row['malignancy'] in ['benign', 'malignant'] else 0)

df['MLP1'] = mlp1
df['MLP2'] = mlp2
df['MLP3'] = mlp3

# Save to new CSV
out_path = 'datasets/metadata/unified_augmented_with_mlp_binary.csv'
df.to_csv(out_path, index=False)
print(f"Saved with MLP1/2/3 binary columns to {out_path}") 