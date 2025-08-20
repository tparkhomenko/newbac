#!/usr/bin/env python3
"""
Script to create a new metadata file with individual image filenames
by processing the source CSV files and mapping them to the unified labels.
"""

import pandas as pd
import os
from pathlib import Path
import sys

def load_source_csvs():
    """Load all source CSV files and extract image-diagnosis mappings."""
    
    csv_dir = Path('data/raw/csv')
    source_data = {}
    
    print("Loading source CSV files...")
    
    # ISIC 2018 files
    isic2018_files = [
        'ISIC2018_Task3_Training_GroundTruth.csv',
        'ISIC2018_Task3_Validation_GroundTruth.csv',
        'ISIC2018_Task3_Test_GroundTruth.csv'
    ]
    
    for filename in isic2018_files:
        filepath = csv_dir / filename
        if filepath.exists():
            print(f"  Loading {filename}...")
            df = pd.read_csv(filepath)
            
            # ISIC 2018 has binary columns for each diagnosis
            # Convert to unified format
            for idx, row in df.iterrows():
                image_name = row['image']
                
                # Find which diagnosis is 1.0
                for col in ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']:
                    if col in df.columns and row[col] == 1.0:
                        # Map to unified labels
                        if col == 'MEL':
                            diagnosis = 'MEL'
                            unified_label = 'melanoma'
                        elif col == 'NV':
                            diagnosis = 'NV'
                            unified_label = 'nevus'
                        elif col == 'BCC':
                            diagnosis = 'BCC'
                            unified_label = 'bcc'
                        elif col == 'AKIEC':
                            diagnosis = 'AKIEC'
                            unified_label = 'akiec'
                        elif col == 'BKL':
                            diagnosis = 'BKL'
                            unified_label = 'bkl'
                        elif col == 'DF':
                            diagnosis = 'DF'
                            unified_label = 'df'
                        elif col == 'VASC':
                            diagnosis = 'VASC'
                            unified_label = 'vascular'
                        
                        source_data[image_name] = {
                            'image': image_name,
                            'diagnosis': diagnosis,
                            'unified_label': unified_label,
                            'source_csv': filename
                        }
                        break
    
    # ISIC 2019 files - also use binary columns
    isic2019_files = [
        'ISIC_2019_Training_GroundTruth.csv',
        'ISIC_2019_Test_GroundTruth.csv'
    ]
    
    for filename in isic2019_files:
        filepath = csv_dir / filename
        if filepath.exists():
            print(f"  Loading {filename}...")
            df = pd.read_csv(filepath)
            
            # ISIC 2019 has binary columns for each diagnosis
            for idx, row in df.iterrows():
                image_name = row['image']
                
                # Find which diagnosis is 1.0
                for col in ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']:
                    if col in df.columns and row[col] == 1.0:
                        # Map to unified labels
                        if col == 'MEL':
                            diagnosis = 'MEL'
                            unified_label = 'melanoma'
                        elif col == 'NV':
                            diagnosis = 'NV'
                            unified_label = 'nevus'
                        elif col == 'BCC':
                            diagnosis = 'BCC'
                            unified_label = 'bcc'
                        elif col == 'AK':
                            diagnosis = 'AK'
                            unified_label = 'ak'
                        elif col == 'BKL':
                            diagnosis = 'BKL'
                            unified_label = 'bkl'
                        elif col == 'DF':
                            diagnosis = 'DF'
                            unified_label = 'df'
                        elif col == 'VASC':
                            diagnosis = 'VASC'
                            unified_label = 'vascular'
                        elif col == 'SCC':
                            diagnosis = 'SCC'
                            unified_label = 'scc'
                        elif col == 'UNK':
                            diagnosis = 'UNK'
                            unified_label = 'unknown'
                        
                        source_data[image_name] = {
                            'image': image_name,
                            'diagnosis': diagnosis,
                            'unified_label': unified_label,
                            'source_csv': filename
                        }
                        break
    
    # ISIC 2020 files
    isic2020_files = [
        'ISIC_2020_Training_GroundTruth.csv',
        'ISIC_2020_Training_GroundTruth_v2.csv'
    ]
    
    for filename in isic2020_files:
        filepath = csv_dir / filename
        if filepath.exists():
            print(f"  Loading {filename}...")
            df = pd.read_csv(filepath)
            
            # Check if image_name column exists, otherwise try 'image'
            image_col = 'image_name' if 'image_name' in df.columns else 'image'
            
            # ISIC 2020 has diagnosis column
            for idx, row in df.iterrows():
                image_name = row[image_col]
                diagnosis = row['diagnosis']
                
                # Map to unified labels
                if diagnosis == 'melanoma':
                    unified_label = 'melanoma'
                    diagnosis_code = 'MEL'
                elif diagnosis == 'nevus':
                    unified_label = 'nevus'
                    diagnosis_code = 'NV'
                elif diagnosis == 'bcc':
                    unified_label = 'bcc'
                    diagnosis_code = 'BCC'
                elif diagnosis == 'akiec':
                    unified_label = 'akiec'
                    diagnosis_code = 'AKIEC'
                elif diagnosis == 'bkl':
                    unified_label = 'bkl'
                    diagnosis_code = 'BKL'
                elif diagnosis == 'df':
                    unified_label = 'df'
                    diagnosis_code = 'DF'
                elif diagnosis == 'vasc':
                    unified_label = 'vascular'
                    diagnosis_code = 'VASC'
                elif diagnosis == 'unknown':
                    unified_label = 'unknown'
                    diagnosis_code = 'UNK'
                else:
                    unified_label = 'other'
                    diagnosis_code = diagnosis
                
                source_data[image_name] = {
                    'image': image_name,
                    'diagnosis': diagnosis_code,
                    'unified_label': unified_label,
                    'source_csv': filename
                }
    
    # Add imagenet images (if they exist)
    imagenet_dir = Path('data/raw/imagenet')
    if imagenet_dir.exists():
        print("  Loading imagenet images...")
        for img_file in imagenet_dir.glob('*.jpg'):
            image_name = img_file.stem
            source_data[image_name] = {
                'image': image_name,
                'diagnosis': 'IMAGENET',
                'unified_label': 'non_skin',
                'source_csv': 'imagenet_ood'
            }
    
    return source_data

def create_new_metadata():
    """Create new metadata file with individual image filenames."""
    
    print("Creating new metadata with filenames...")
    
    # Load source data
    source_data = load_source_csvs()
    
    # Convert to DataFrame
    df = pd.DataFrame(list(source_data.values()))
    
    print(f"Total images found: {len(df):,}")
    
    # Check coverage in cleaned_resized
    cleaned_dir = Path('data/cleaned_resized')
    missing_images = []
    
    print("\nChecking coverage in cleaned_resized...")
    for idx, row in df.iterrows():
        image_name = row['image']
        source = row['source_csv']
        
        # Determine which folder this image should be in
        if 'ISIC2018' in source:
            folder = 'isic2018_512'
        elif 'ISIC_2019' in source:
            folder = 'isic2019_512'
        elif 'ISIC_2020' in source:
            folder = 'isic2020_512'
        elif 'imagenet' in source:
            folder = 'plausibility_check_512'
        else:
            continue
        
        image_path = cleaned_dir / folder / f'{image_name}.jpg'
        if not image_path.exists():
            missing_images.append(image_name)
    
    print(f"Missing from cleaned_resized: {len(missing_images):,}")
    
    # Add experimental split columns (placeholder for now)
    df['exp1'] = 1
    df['exp2'] = 0
    df['exp3'] = 0
    df['exp4'] = 0
    df['exp5'] = 0
    
    # Save new metadata
    output_path = 'data/metadata/metadata_with_filenames.csv'
    df.to_csv(output_path, index=False)
    print(f"\nSaved new metadata to: {output_path}")
    
    # Summary
    print(f"\n{'='*50}")
    print(f"SUMMARY")
    print(f"{'='*50}")
    print(f"Total images in new metadata: {len(df):,}")
    print(f"Missing from cleaned_resized: {len(missing_images):,}")
    print(f"Coverage: {((len(df) - len(missing_images)) / len(df) * 100):.1f}%")
    
    if missing_images:
        print(f"\nFirst 10 missing images:")
        for img in missing_images[:10]:
            print(f"  {img}")
    
    return df, missing_images

if __name__ == "__main__":
    create_new_metadata()
