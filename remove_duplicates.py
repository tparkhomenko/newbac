#!/usr/bin/env python3
"""
Script to remove duplicate images from raw and resized directories based on CSV file.
"""

import os
import pandas as pd
import shutil
from pathlib import Path

def remove_duplicates():
    """
    Remove duplicate images from both raw and resized directories based on CSV file.
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
    raw_deleted = 0
    resized_deleted = 0
    raw_not_found = 0
    resized_not_found = 0
    
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
        
        # Remove from raw directory
        if os.path.exists(raw_path1):
            try:
                os.remove(raw_path1)
                print(f"Deleted raw image: {image1_jpg}")
                raw_deleted += 1
            except Exception as e:
                print(f"Error deleting raw image {image1_jpg}: {e}")
        else:
            print(f"Raw image not found: {image1_jpg}")
            raw_not_found += 1
            
        if os.path.exists(raw_path2):
            try:
                os.remove(raw_path2)
                print(f"Deleted raw image: {image2_jpg}")
                raw_deleted += 1
            except Exception as e:
                print(f"Error deleting raw image {image2_jpg}: {e}")
        else:
            print(f"Raw image not found: {image2_jpg}")
            raw_not_found += 1
        
        # Remove from resized directory
        if os.path.exists(resized_path1):
            try:
                os.remove(resized_path1)
                print(f"Deleted resized image: {image1_jpg}")
                resized_deleted += 1
            except Exception as e:
                print(f"Error deleting resized image {image1_jpg}: {e}")
        else:
            print(f"Resized image not found: {image1_jpg}")
            resized_not_found += 1
            
        if os.path.exists(resized_path2):
            try:
                os.remove(resized_path2)
                print(f"Deleted resized image: {image2_jpg}")
                resized_deleted += 1
            except Exception as e:
                print(f"Error deleting resized image {image2_jpg}: {e}")
        else:
            print(f"Resized image not found: {image2_jpg}")
            resized_not_found += 1
    
    # Print summary
    print("\n" + "="*50)
    print("DUPLICATE REMOVAL SUMMARY")
    print("="*50)
    print(f"Total duplicate pairs processed: {len(df)}")
    print(f"Raw images deleted: {raw_deleted}")
    print(f"Raw images not found: {raw_not_found}")
    print(f"Resized images deleted: {resized_deleted}")
    print(f"Resized images not found: {resized_not_found}")
    print("="*50)

if __name__ == "__main__":
    # Ask for confirmation before proceeding
    print("This script will delete duplicate images from both raw and resized directories.")
    print("Based on the CSV file: data/raw/csv/ISIC_2020_Training_Duplicates.csv")
    print("\nWARNING: This action cannot be undone!")
    
    response = input("\nDo you want to proceed? (yes/no): ").lower().strip()
    
    if response in ['yes', 'y']:
        remove_duplicates()
    else:
        print("Operation cancelled.") 