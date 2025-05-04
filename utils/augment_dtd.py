import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def resize_image(image, target_size=(512, 512)):
    """Resize image to target size while maintaining aspect ratio"""
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def apply_augmentations(image):
    """Apply basic augmentations (flip and rotation)"""
    augmented = []
    
    # Original image
    augmented.append(('original', image))
    
    # Horizontal flip
    flipped_h = cv2.flip(image, 1)
    augmented.append(('flip_h', flipped_h))
    
    # Vertical flip
    flipped_v = cv2.flip(image, 0)
    augmented.append(('flip_v', flipped_v))
    
    # 90 and 270 degree rotations
    for angle in [90, 270]:
        matrix = cv2.getRotationMatrix2D((image.shape[1]/2, image.shape[0]/2), angle, 1.0)
        rotated = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))
        augmented.append((f'rot_{angle}', rotated))
    
    return augmented

def process_dtd_dataset():
    """Process and augment DTD dataset"""
    # Setup paths
    source_dir = Path('datasets/dtd')
    target_dir = Path('datasets/dtd_aug')
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Read original metadata
    metadata = pd.read_csv(source_dir / 'metadata_dtd.csv')
    
    # Initialize augmented metadata list
    augmented_data = []
    
    print("Processing and augmenting DTD images...")
    processed_count = 0
    
    # Process each image
    for _, row in tqdm(metadata.iterrows(), total=len(metadata)):
        # Read image
        img_path = source_dir / row['image']
        image = cv2.imread(str(img_path))
        
        if image is None:
            print(f"Failed to read {img_path}")
            continue
            
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        resized = resize_image(image)
        
        # Apply augmentations
        augmented_images = apply_augmentations(resized)
        
        # Save all versions
        for aug_name, aug_image in augmented_images:
            # Create augmented image name
            aug_image_name = f"{img_path.stem}_{aug_name}.jpg"
            save_path = target_dir / aug_image_name
            
            # Save image
            cv2.imwrite(str(save_path), cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
            
            # Add to metadata
            augmented_data.append({
                'image': aug_image_name,
                'texture': row['texture'],
                'skin': 0  # Not skin
            })
            processed_count += 1
    
    # Create augmented metadata DataFrame
    augmented_df = pd.DataFrame(augmented_data)
    
    # Save augmented metadata
    metadata_path = target_dir / 'metadata_dtd_augmented.csv'
    augmented_df.to_csv(metadata_path, index=False)
    
    print(f"\nProcessing complete!")
    print(f"Original images: {len(metadata)}")
    print(f"Augmented images: {processed_count}")
    print(f"Images and metadata saved to: {target_dir}")
    print("\nTexture distribution in augmented dataset:")
    print(augmented_df['texture'].value_counts().head())

if __name__ == "__main__":
    process_dtd_dataset() 