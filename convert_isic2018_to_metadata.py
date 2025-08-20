import pandas as pd
import os

# Diagnosis mappings based on your specification
diagnosis_mapping = {
    'MEL': 'melanoma',
    'NV': 'nevus', 
    'BCC': 'bcc',
    'AKIEC': 'ak',
    'BKL': 'bkl',
    'DF': 'df',
    'VASC': 'vasc'
}

# Output file
output_file = 'data/metadata/metadata.csv'

# Process each ISIC2018 CSV file
csv_files = [
    ('data/raw/csv/ISIC2018_Task3_Training_GroundTruth.csv', 'ISIC2018_Task3_Training_GroundTruth.csv'),
    ('data/raw/csv/ISIC2018_Task3_Validation_GroundTruth.csv', 'ISIC2018_Task3_Validation_GroundTruth.csv'),
    ('data/raw/csv/ISIC2018_Task3_Test_GroundTruth.csv', 'ISIC2018_Task3_Test_GroundTruth.csv')
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
        diagnosis_cols = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
        diagnosis = None
        
        for col in diagnosis_cols:
            if row[col] == 1.0:
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

# Create DataFrame and save
result_df = pd.DataFrame(all_data)
result_df.to_csv(output_file, index=False)

print(f"\nConversion complete!")
print(f"Total images processed: {len(result_df)}")
print(f"Output saved to: {output_file}")
print(f"\nDiagnosis distribution:")
print(result_df['diagnosis'].value_counts())
print(f"\nUnified label distribution:")
print(result_df['unified_label'].value_counts()) 