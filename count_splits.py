import pandas as pd

# Read the CSV file with splits
df = pd.read_csv('data/metadata/unified_labels_with_splits.csv')

# Get the total number of images
total_images = len(df)

# Count images in each split column (where value equals 1)
split_counts = {}
for i in range(1, 6):
    column_name = f'split{i}'
    if column_name in df.columns:
        count = df[column_name].sum()  # Sum of 1s in the column
        split_counts[column_name] = count
    else:
        print(f"Warning: Column '{column_name}' not found in the CSV file")

# Print the summary
print("Split column counts:")
for split_name, count in split_counts.items():
    print(f"{split_name}: {count}/{total_images}") 