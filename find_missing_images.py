#!/usr/bin/env python3
"""
Find images that are on disk but missing from the CSV file.
"""

import pandas as pd
import os
from pathlib import Path

def find_missing_images():
    """Find images that are on disk but not in the CSV."""
    
    # Load CSV
    csv_path = "data/metadata/unified_labels_with_stratified_splits.csv"
    df = pd.read_csv(csv_path)
    csv_images = set(df['image'].tolist())
    print(f"Images in CSV: {len(csv_images)}")
    
    # Find all images on disk
    disk_images = set()
    image_root = Path("data/cleaned_resized")
    
    for img_file in image_root.rglob("*.jpg"):
        # Get filename without extension
        img_name = img_file.stem
        disk_images.add(img_name)
    
    print(f"Images on disk: {len(disk_images)}")
    
    # Find missing images
    missing_images = disk_images - csv_images
    print(f"Missing from CSV: {len(missing_images)}")
    
    # Show some examples
    print("\nFirst 20 missing images:")
    for i, img in enumerate(sorted(missing_images)[:20]):
        print(f"{i+1:2d}. {img}")
    
    # Check by directory
    print("\nMissing images by directory:")
    for dataset_dir in image_root.iterdir():
        if dataset_dir.is_dir():
            dataset_missing = []
            for img_file in dataset_dir.rglob("*.jpg"):
                img_name = img_file.stem
                if img_name not in csv_images:
                    dataset_missing.append(img_name)
            
            if dataset_missing:
                print(f"{dataset_dir.name}: {len(dataset_missing)} missing images")
                print(f"  Examples: {dataset_missing[:5]}")
    
    return missing_images

if __name__ == "__main__":
    missing = find_missing_images() 