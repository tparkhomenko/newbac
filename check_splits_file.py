import pandas as pd

# Read the CSV file with splits
df = pd.read_csv('data/metadata/unified_labels_with_splits.csv')

# Print column names
print("Available columns:")
for col in df.columns:
    print(f"  {col}")

# Print first few rows to see the data structure
print("\nFirst 5 rows:")
print(df.head()) 