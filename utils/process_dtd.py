import os
import shutil
from pathlib import Path
import pandas as pd
from tqdm import tqdm

def process_dtd_dataset():
    """
    Process DTD dataset:
    1. Copy all images from nested folders to a single directory
    2. Create metadata with skin=0 label
    """
    # Setup paths
    source_dir = Path('data/dtd/images')
    target_dir = Path('datasets/dtd')
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize metadata list
    metadata = []
    
    print("Processing DTD images...")
    
    # Process each texture category
    for category_dir in tqdm(list(source_dir.glob('*'))):
        if not category_dir.is_dir():
            continue
            
        # Process each image in category
        for img_path in category_dir.glob('*.jpg'):
            # Create target path
            target_path = target_dir / f"{category_dir.name}_{img_path.name}"
            
            # Copy image
            shutil.copy2(img_path, target_path)
            
            # Add to metadata
            metadata.append({
                'image': target_path.name,
                'texture': category_dir.name,
                'skin': 0  # Not skin
            })
    
    # Create metadata DataFrame
    metadata_df = pd.DataFrame(metadata)
    
    # Save metadata
    metadata_path = target_dir / 'metadata_dtd.csv'
    metadata_df.to_csv(metadata_path, index=False)
    
    print(f"\nProcessed {len(metadata)} DTD images")
    print(f"Images and metadata saved to: {target_dir}")
    print("\nTexture distribution:")
    print(metadata_df['texture'].value_counts())

if __name__ == "__main__":
    process_dtd_dataset() 