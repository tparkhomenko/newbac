#!/usr/bin/env python3
"""
Unify CSV labels from all ISIC datasets into a single format - FIXED VERSION.
"""

import pandas as pd
import os
import glob
from pathlib import Path

def get_label_from_onehot(row, label_columns):
    """Extract label from one-hot encoded columns."""
    for col in label_columns:
        if row[col] == 1.0:
            return col
    return None

def unify_labels(diagnosis):
    """Convert diagnosis to unified label format."""
    label_map = {
        "MEL": "melanoma", "melanoma": "melanoma",
        "NV": "nevus", "nevus": "nevus",
        "BCC": "bcc", "basal cell carcinoma": "bcc",
        "AK": "ak", "AKIEC": "ak", "actinic keratosis": "ak",
        "BKL": "bkl", "seborrheic keratosis": "bkl", "solar lentigo": "bkl", "lichenoid keratosis": "bkl", "lentigo NOS": "bkl",
        "DF": "df", "dermatofibroma": "df",
        "VASC": "vascular", "vascular lesion": "vascular",
        "SCC": "scc", "squamous cell carcinoma": "scc",
        "atypical melanocytic proliferation": "other", "unknown": "other", "UNK": "other",
    }
    
    if pd.isna(diagnosis) or diagnosis == "":
        return "other"
    
    diagnosis_str = str(diagnosis).lower().strip()
    
    # Direct mapping
    if diagnosis_str in label_map:
        return label_map[diagnosis_str]
    
    # Check for partial matches
    for key, value in label_map.items():
        if key.lower() in diagnosis_str or diagnosis_str in key.lower():
            return value
    
    return "other"

def process_csv_file(file_path):
    """Process a single CSV file and return unified DataFrame."""
    print(f"Processing: {os.path.basename(file_path)}")
    
    df = pd.read_csv(file_path)
    source_csv = os.path.basename(file_path)
    
    # Add source column
    df['source_csv'] = source_csv
    
    # Determine image column name
    image_col = None
    for col in ['image', 'image_name']:
        if col in df.columns:
            image_col = col
            break
    
    if image_col is None:
        print(f"Warning: No image column found in {file_path}")
        return None
    
    # Rename image column to standard 'image'
    if image_col != 'image':
        df = df.rename(columns={image_col: 'image'})
    
    # Extract diagnosis based on file type
    diagnosis_col = None
    
    # Check if there's a diagnosis column
    for col in ['diagnosis']:
        if col in df.columns:
            diagnosis_col = col
            break
    
    # If no diagnosis column, look for one-hot encoded labels
    if diagnosis_col is None:
        # Define possible label columns
        label_columns = ['MEL', 'NV', 'BCC', 'AKIEC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
        available_labels = [col for col in label_columns if col in df.columns]
        
        if available_labels:
            print(f"  Found one-hot labels: {available_labels}")
            # Extract diagnosis from one-hot encoding
            df['diagnosis'] = df.apply(lambda row: get_label_from_onehot(row, available_labels), axis=1)
        else:
            # Check if this is a metadata-only file
            metadata_columns = ['age_approx', 'anatom_site_general', 'sex', 'patient', 'patient_id']
            if any(col in df.columns for col in metadata_columns):
                print(f"  This appears to be a metadata-only file (no diagnoses)")
                return None  # Skip metadata-only files
            else:
                print(f"  Warning: No diagnosis or label columns found in {file_path}")
                df['diagnosis'] = 'unknown'
    
    # Apply unified label mapping
    df['unified_label'] = df['diagnosis'].apply(unify_labels)
    
    # Keep only essential columns
    result_df = df[['image', 'diagnosis', 'unified_label', 'source_csv']].copy()
    
    print(f"  Processed {len(result_df)} rows")
    print(f"  Label distribution: {result_df['unified_label'].value_counts().to_dict()}")
    
    return result_df

def main():
    """Main function to process all CSV files."""
    csv_dir = "data/raw/csv"
    output_dir = "data/metadata"
    
    # Get all CSV files
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    print(f"Found {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"  - {os.path.basename(f)}")
    
    # Process each file
    all_dfs = []
    for csv_file in csv_files:
        df = process_csv_file(csv_file)
        if df is not None:
            all_dfs.append(df)
    
    # Combine all DataFrames
    if all_dfs:
        unified_df = pd.concat(all_dfs, ignore_index=True)
        
        # Remove duplicates based on image name
        print(f"\nBefore removing duplicates: {len(unified_df)} rows")
        unified_df = unified_df.drop_duplicates(subset=['image'], keep='first')
        print(f"After removing duplicates: {len(unified_df)} rows")
        
        # Save unified CSV
        output_path = os.path.join(output_dir, "unified_labels_fixed.csv")
        unified_df.to_csv(output_path, index=False)
        print(f"\nSaved unified CSV to: {output_path}")
        
        # Display final statistics
        print("\n" + "="*50)
        print("FINAL STATISTICS")
        print("="*50)
        
        print("\nLabel distribution:")
        label_counts = unified_df['unified_label'].value_counts()
        for label, count in label_counts.items():
            print(f"  {label}: {count}")
        
        print(f"\nTotal unique images: {len(unified_df)}")
        
        print("\nSource CSV distribution:")
        source_counts = unified_df['source_csv'].value_counts()
        for source, count in source_counts.items():
            print(f"  {source}: {count}")
        
        # Display sample of unified data
        print("\nSample of unified data:")
        print(unified_df.head(10).to_string(index=False))
        
    else:
        print("No valid CSV files processed!")

if __name__ == "__main__":
    main() 