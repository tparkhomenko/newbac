import pandas as pd
import os

# Diagnosis mappings for ISIC2019 (slightly different from ISIC2018)
diagnosis_mapping = {
    'MEL': 'melanoma',
    'NV': 'nevus', 
    'BCC': 'bcc',
    'AK': 'ak',  # Note: AK instead of AKIEC in ISIC2019
    'BKL': 'bkl',
    'DF': 'df',
    'VASC': 'vasc',
    'SCC': 'scc',  # New in ISIC2019
    'UNK': 'unknown'  # New in ISIC2019
}

# Input and output files
input_file = 'data/metadata/metadata.csv'
output_file = 'data/metadata/metadata.csv'

# Process each ISIC2019 CSV file
csv_files = [
    ('data/raw/csv/ISIC_2019_Training_GroundTruth.csv', 'ISIC_2019_Training_GroundTruth.csv'),
    ('data/raw/csv/ISIC_2019_Test_GroundTruth.csv', 'ISIC_2019_Test_GroundTruth.csv')
]

all_data = []

for csv_path, source_csv in csv_files:
    print(f"Processing {source_csv}...")
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Process each row
    for _, row in df.iterrows():
        image_id = row['image']
        
        # Find which diagnosis column has value 1.0
        diagnosis_cols = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
        diagnosis = None
        
        for col in diagnosis_cols:
            if col in row and row[col] == 1.0:
                diagnosis = col
                break
        
        if diagnosis:
            unified_label = diagnosis_mapping[diagnosis]
            
            # Add to data list
            all_data.append({
                'diagnosis': diagnosis,
                'unified_label': unified_label,
                'source_csv': source_csv,
                'exp1': '',
                'exp2': '',
                'exp3': '',
                'exp4': '',
                'exp5': ''
            })
        else:
            print(f"Warning: No diagnosis found for {image_id}")

# Read existing metadata
existing_df = pd.read_csv(input_file)
print(f"Existing metadata has {len(existing_df)} rows")

# Append new data
result_df = pd.concat([existing_df, pd.DataFrame(all_data)], ignore_index=True)

# Save combined result
result_df.to_csv(output_file, index=False)

print(f"\nISIC2019 conversion complete!")
print(f"New ISIC2019 images added: {len(all_data)}")
print(f"Total images in metadata.csv: {len(result_df)}")
print(f"\nISIC2019 Diagnosis distribution:")
isic2019_data = pd.DataFrame(all_data)
print(isic2019_data['diagnosis'].value_counts())
print(f"\nISIC2019 Unified label distribution:")
print(isic2019_data['unified_label'].value_counts())
print(f"\nComplete dataset diagnosis distribution:")
print(result_df['diagnosis'].value_counts()) 