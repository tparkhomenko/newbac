#!/usr/bin/env python3
"""
Script to check for duplicate images in raw and resized directories based on CSV file.
This script only shows what would be deleted without actually deleting anything.
"""

import os
import pandas as pd
from pathlib import Path

def check_duplicates():
    """
    Check for duplicate images in both raw and resized directories based on CSV file.
    """
    # Define paths
    csv_file = "data/raw/csv/ISIC_2020_Training_Duplicates.csv"
    raw_dir = "data/raw/isic2020/ISIC_2020_Training_JPEG/train"
    resized_dir = "data/cleaned_resized/isic2020_512"
    
    # Check if CSV file exists
    if not os.path.exists(csv_file):
        print(f"Error: CSV file not found at {csv_file}")
        return
    
    # Read the CSV file
    try:
        df = pd.read_csv(csv_file)
        print(f"Found {len(df)} duplicate pairs in CSV file")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    # Track statistics
    raw_found = 0
    resized_found = 0
    raw_not_found = 0
    resized_not_found = 0
    
    raw_files_to_delete = []
    resized_files_to_delete = []
    
    # Process each duplicate pair
    for index, row in df.iterrows():
        image1 = row['image_name_1']
        image2 = row['image_name_2']
        
        # Add .jpg extension
        image1_jpg = f"{image1}.jpg"
        image2_jpg = f"{image2}.jpg"
        
        # Define file paths
        raw_path1 = os.path.join(raw_dir, image1_jpg)
        raw_path2 = os.path.join(raw_dir, image2_jpg)
        resized_path1 = os.path.join(resized_dir, image1_jpg)
        resized_path2 = os.path.join(resized_dir, image2_jpg)
        
        # Check raw directory
        if os.path.exists(raw_path1):
            raw_files_to_delete.append(raw_path1)
            raw_found += 1
        else:
            raw_not_found += 1
            
        if os.path.exists(raw_path2):
            raw_files_to_delete.append(raw_path2)
            raw_found += 1
        else:
            raw_not_found += 1
        
        # Check resized directory
        if os.path.exists(resized_path1):
            resized_files_to_delete.append(resized_path1)
            resized_found += 1
        else:
            resized_not_found += 1
            
        if os.path.exists(resized_path2):
            resized_files_to_delete.append(resized_path2)
            resized_found += 1
        else:
            resized_not_found += 1
    
    # Print summary
    print("\n" + "="*60)
    print("DUPLICATE CHECK SUMMARY")
    print("="*60)
    print(f"Total duplicate pairs in CSV: {len(df)}")
    print(f"Raw images found: {raw_found}")
    print(f"Raw images not found: {raw_not_found}")
    print(f"Resized images found: {resized_found}")
    print(f"Resized images not found: {resized_not_found}")
    print("="*60)
    
    # Show files that would be deleted
    if raw_files_to_delete:
        print(f"\nRAW IMAGES THAT WOULD BE DELETED ({len(raw_files_to_delete)} files):")
        print("-" * 50)
        for file_path in raw_files_to_delete[:10]:  # Show first 10
            print(f"  {os.path.basename(file_path)}")
        if len(raw_files_to_delete) > 10:
            print(f"  ... and {len(raw_files_to_delete) - 10} more")
    
    if resized_files_to_delete:
        print(f"\nRESIZED IMAGES THAT WOULD BE DELETED ({len(resized_files_to_delete)} files):")
        print("-" * 50)
        for file_path in resized_files_to_delete[:10]:  # Show first 10
            print(f"  {os.path.basename(file_path)}")
        if len(resized_files_to_delete) > 10:
            print(f"  ... and {len(resized_files_to_delete) - 10} more")
    
    # Calculate disk space that would be freed
    total_size_raw = sum(os.path.getsize(f) for f in raw_files_to_delete if os.path.exists(f))
    total_size_resized = sum(os.path.getsize(f) for f in resized_files_to_delete if os.path.exists(f))
    total_size_mb = (total_size_raw + total_size_resized) / (1024 * 1024)
    
    print(f"\nDISK SPACE THAT WOULD BE FREED: {total_size_mb:.2f} MB")
    print("="*60)

if __name__ == "__main__":
    check_duplicates() 