import pandas as pd
import os

# Diagnosis mappings for ISIC2020 (text-based diagnoses)
diagnosis_mapping = {
    'nevus': 'nevus',
    'melanoma': 'melanoma',
    'seborrheic keratosis': 'bkl',  # Maps to BKL (benign keratosis-like lesions)
    'lentigo NOS': 'lentigo',
    'lichenoid keratosis': 'bkl',   # Maps to BKL
    'solar lentigo': 'lentigo',
    'cafe-au-lait macule': 'cafe_au_lait',
    'atypical melanocytic proliferation': 'atypical_melanocytic',
    'unknown': 'unknown'
}

# Input and output files
input_file = 'data/metadata/metadata.csv'
output_file = 'data/metadata/metadata.csv'

# Process each ISIC2020 CSV file
csv_files = [
    ('data/raw/csv/ISIC_2020_Training_GroundTruth.csv', 'ISIC_2020_Training_GroundTruth.csv'),
    ('data/raw/csv/ISIC_2020_Test_Metadata.csv', 'ISIC_2020_Test_Metadata.csv')
]

all_data = []

for csv_path, source_csv in csv_files:
    print(f"Processing {source_csv}...")
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Process each row
    for _, row in df.iterrows():
        if source_csv == 'ISIC_2020_Training_GroundTruth.csv':
            # Training file has diagnosis column
            image_id = row['image_name']
            diagnosis_text = row['diagnosis']
            
            if diagnosis_text in diagnosis_mapping:
                diagnosis = diagnosis_text
                unified_label = diagnosis_mapping[diagnosis_text]
            else:
                print(f"Warning: Unknown diagnosis '{diagnosis_text}' for {image_id}")
                continue
                
        elif source_csv == 'ISIC_2020_Test_Metadata.csv':
            # Test file has no diagnosis - mark as unknown
            image_id = row['image']
            diagnosis = 'unknown'
            unified_label = 'unknown'
        
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

# Read existing metadata
existing_df = pd.read_csv(input_file)
print(f"Existing metadata has {len(existing_df)} rows")

# Append new data
result_df = pd.concat([existing_df, pd.DataFrame(all_data)], ignore_index=True)

# Save combined result
result_df.to_csv(output_file, index=False)

print(f"\nISIC2020 conversion complete!")
print(f"New ISIC2020 images added: {len(all_data)}")
print(f"Total images in metadata.csv: {len(result_df)}")
print(f"\nISIC2020 Diagnosis distribution:")
isic2020_data = pd.DataFrame(all_data)
print(isic2020_data['diagnosis'].value_counts())
print(f"\nISIC2020 Unified label distribution:")
print(isic2020_data['unified_label'].value_counts())
print(f"\nComplete dataset diagnosis distribution:")
print(result_df['diagnosis'].value_counts())
print(f"\nTotal images by source:")
print(result_df['source_csv'].value_counts()) 