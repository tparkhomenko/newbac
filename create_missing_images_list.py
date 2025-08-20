#!/usr/bin/env python3
"""
Script to create a complete list of missing images from cleaned_resized folders.
"""

import pandas as pd
from pathlib import Path

def create_missing_images_list():
    """Create a complete list of missing images."""
    
    # Load the new metadata
    df = pd.read_csv('data/metadata/metadata_with_filenames.csv')
    print(f"Loaded metadata with {len(df):,} images")
    
    # Check coverage in cleaned_resized
    cleaned_dir = Path('data/cleaned_resized')
    missing_images = []
    
    print("Checking coverage in cleaned_resized...")
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
            missing_images.append({
                'image': image_name,
                'source_csv': source,
                'diagnosis': row['diagnosis'],
                'unified_label': row['unified_label']
            })
    
    print(f"Found {len(missing_images):,} missing images")
    
    # Create missing images list
    missing_df = pd.DataFrame(missing_images)
    
    # Save to CSV
    missing_csv_path = 'data/metadata/missing_images_list.csv'
    missing_df.to_csv(missing_csv_path, index=False)
    print(f"Saved missing images list to: {missing_csv_path}")
    
    # Save to text file for easy reading
    missing_txt_path = 'data/metadata/missing_images_list.txt'
    with open(missing_txt_path, 'w') as f:
        f.write(f"Missing Images List - Total: {len(missing_images):,}\n")
        f.write("="*60 + "\n\n")
        
        # Group by source
        for source in missing_df['source_csv'].unique():
            source_missing = missing_df[missing_df['source_csv'] == source]
            f.write(f"Source: {source} - {len(source_missing):,} missing images\n")
            f.write("-" * 40 + "\n")
            
            for _, row in source_missing.iterrows():
                f.write(f"  {row['image']} ({row['diagnosis']} -> {row['unified_label']})\n")
            f.write("\n")
    
    print(f"Saved missing images list to: {missing_txt_path}")
    
    # Summary by source
    print(f"\nMissing images by source:")
    for source in missing_df['source_csv'].unique():
        count = len(missing_df[missing_df['source_csv'] == source])
        print(f"  {source}: {count:,}")
    
    return missing_df

if __name__ == "__main__":
    create_missing_images_list()
