#!/usr/bin/env python3
"""
Script to check if all images from metadata CSV are present in cleaned_resized folders.
"""

import pandas as pd
import os
from pathlib import Path
import sys

def check_image_coverage():
    """Check coverage of images from metadata in cleaned_resized folders."""
    
    # Load metadata
    print("Loading metadata...")
    try:
        df = pd.read_csv('data/metadata/unified_labels_with_stratified_splits.csv')
        print(f"✓ Loaded metadata with {len(df):,} images")
    except FileNotFoundError:
        print("❌ Error: unified_labels_with_stratified_splits.csv not found")
        return
    
    # Check cleaned_resized folders
    cleaned_dir = Path('data/cleaned_resized')
    if not cleaned_dir.exists():
        print("❌ Error: data/cleaned_resized folder not found")
        return
    
    print(f"\nChecking cleaned_resized folders...")
    folder_counts = {}
    total_cleaned = 0
    
    for folder in cleaned_dir.iterdir():
        if folder.is_dir():
            jpg_count = len(list(folder.glob('*.jpg')))
            folder_counts[folder.name] = jpg_count
            total_cleaned += jpg_count
            print(f"  {folder.name}: {jpg_count:,} images")
    
    print(f"Total images in cleaned_resized: {total_cleaned:,}")
    
    # Check for missing images
    print(f"\nChecking for missing images...")
    missing_count = 0
    missing_examples = []
    
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
            print(f"⚠️  Unknown source: {source}")
            continue
        
        image_path = cleaned_dir / folder / f'{image_name}.jpg'
        if not image_path.exists():
            missing_count += 1
            if len(missing_examples) < 10:  # Show first 10 missing
                missing_examples.append((image_name, source))
    
    print(f"Missing images: {missing_count:,}")
    if missing_examples:
        print("Examples of missing images:")
        for img, src in missing_examples:
            print(f"  {img} (from {src})")
    
    # Check for extra images (images in folders but not in metadata)
    print(f"\nChecking for extra images...")
    metadata_images = set(df['image'].tolist())
    extra_count = 0
    extra_examples = []
    
    for folder in cleaned_dir.iterdir():
        if folder.is_dir():
            for jpg_file in folder.glob('*.jpg'):
                image_name = jpg_file.stem  # Remove .jpg extension
                if image_name not in metadata_images:
                    extra_count += 1
                    if len(extra_examples) < 10:  # Show first 10 extra
                        extra_examples.append((image_name, folder.name))
    
    print(f"Extra images: {extra_count:,}")
    if extra_examples:
        print("Examples of extra images:")
        for img, folder in extra_examples:
            print(f"  {img} in {folder}")
    
    # Summary
    print(f"\n{'='*50}")
    print(f"SUMMARY")
    print(f"{'='*50}")
    print(f"Metadata images: {len(df):,}")
    print(f"Cleaned resized images: {total_cleaned:,}")
    print(f"Missing from cleaned_resized: {missing_count:,}")
    print(f"Extra in cleaned_resized: {extra_count:,}")
    
    if missing_count == 0:
        print(f"✅ All metadata images are present in cleaned_resized!")
    else:
        coverage = ((len(df) - missing_count) / len(df) * 100)
        print(f"⚠️  Coverage: {coverage:.1f}% ({len(df) - missing_count:,}/{len(df):,})")
    
    if extra_count > 0:
        print(f"ℹ️  There are {extra_count:,} extra images in cleaned_resized not in metadata")

if __name__ == "__main__":
    check_image_coverage()

