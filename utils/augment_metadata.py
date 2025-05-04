import pandas as pd
from pathlib import Path

def augment_metadata():
    """
    Create augmented metadata entries for each augmented image.
    Each original image has 4 versions:
    - original
    - flip_h (horizontal flip)
    - rot_90, rot_180, rot_270 (rotations)
    """
    # Read the original metadata
    metadata = pd.read_csv('data/metadata/metadata_isic2019.csv')
    
    # Filter out unknown lesions
    metadata = metadata[metadata['lesion_group'] != 'unknown'].copy()
    
    # List of augmentation suffixes that match the image processing
    augmentations = ['original', 'flip_h', 'rot_90', 'rot_180', 'rot_270']
    
    # Create new augmented metadata
    augmented_data = []
    
    for _, row in metadata.iterrows():
        # For each original image, create entries for all augmented versions
        for aug in augmentations:
            augmented_data.append({
                'image': f"{row['image']}_{aug}",
                'lesion_group': row['lesion_group'],
                'lesion': row['lesion'],
                'malignancy': row['malignancy'],
                'skin': 1  # All ISIC images are skin lesions
            })
    
    # Create new dataframe with augmented entries
    augmented_df = pd.DataFrame(augmented_data)
    
    # Save augmented metadata
    output_dir = Path('datasets/isic2019')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'metadata_augmented.csv'
    augmented_df.to_csv(output_path, index=False)
    
    print(f"Original metadata entries: {len(metadata)}")
    print(f"Augmented metadata entries: {len(augmented_df)}")
    print(f"Augmented metadata saved to: {output_path}")
    
    # Print distribution statistics
    print("\nLesion group distribution in augmented dataset:")
    print(augmented_df['lesion_group'].value_counts())
    print("\nMalignancy distribution in augmented dataset:")
    print(augmented_df['malignancy'].value_counts())
    print("\nSkin label distribution:")
    print(augmented_df['skin'].value_counts())

if __name__ == "__main__":
    augment_metadata() 