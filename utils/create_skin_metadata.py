import os
import pandas as pd
from pathlib import Path

def create_combined_metadata():
    # Define paths
    base_dir = Path('/home/parkhomenko/Documents/new_project')
    isic_metadata_path = base_dir / 'data/metadata/metadata_isic2019.csv'
    dtd_aug_dir = base_dir / 'datasets/dtd_aug'
    output_path = base_dir / 'datasets/metadata/metadata_skin_not_skin.csv'

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load ISIC metadata
    print("Loading ISIC metadata...")
    isic_df = pd.read_csv(isic_metadata_path)
    
    # Ensure ISIC data has skin=1
    isic_df['skin'] = 1

    # Get list of all DTD augmented images
    print("Processing DTD images...")
    dtd_images = []
    for img_path in dtd_aug_dir.glob('*.jpg'):
        dtd_images.append({
            'image': img_path.name,
            'lesion_group': 'not_skin_texture',
            'lesion': 'DTD',
            'malignancy': 'not_applicable',
            'skin': 0
        })

    # Create DTD DataFrame
    dtd_df = pd.DataFrame(dtd_images)

    # Combine both datasets
    print("Combining datasets...")
    combined_df = pd.concat([isic_df, dtd_df], ignore_index=True)

    # Ensure all required columns are present
    required_columns = ['image', 'lesion_group', 'lesion', 'malignancy', 'skin']
    assert all(col in combined_df.columns for col in required_columns), "Missing required columns"

    # Save combined metadata
    print(f"Saving combined metadata to {output_path}...")
    combined_df.to_csv(output_path, index=False)
    
    # Print statistics
    print("\nDataset Statistics:")
    print(f"Total images: {len(combined_df)}")
    print(f"ISIC images: {len(isic_df)}")
    print(f"DTD images: {len(dtd_df)}")
    print("\nSkin distribution:")
    print(combined_df['skin'].value_counts())
    print("\nLesion group distribution:")
    print(combined_df['lesion_group'].value_counts())

if __name__ == '__main__':
    create_combined_metadata() 