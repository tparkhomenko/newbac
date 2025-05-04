import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil
import yaml

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def create_processed_dirs():
    """Create necessary directories for processed data"""
    processed_dir = Path('data/processed/isic2020')
    processed_dir.mkdir(parents=True, exist_ok=True)
    return processed_dir

def resize_image(image, target_size=(512, 512)):
    """Resize image to target size while maintaining aspect ratio"""
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def apply_augmentations(image):
    """Apply basic augmentations (flip and rotation)"""
    augmented = []
    
    # Original image
    augmented.append(('original', image))
    
    # Horizontal flip
    flipped = cv2.flip(image, 1)
    augmented.append(('flip_h', flipped))
    
    # 90, 180, 270 degree rotations
    for angle in [90, 180, 270]:
        matrix = cv2.getRotationMatrix2D((image.shape[1]/2, image.shape[0]/2), angle, 1.0)
        rotated = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))
        augmented.append((f'rot_{angle}', rotated))
    
    return augmented

def process_dataset():
    """Main function to process the ISIC dataset"""
    config = load_config()
    processed_dir = create_processed_dirs()
    
    # Process training data
    train_dir = Path('data/isic2020/ISIC_2020_Training_JPEG/train')
    
    print("Processing training images...")
    for img_path in tqdm(list(train_dir.glob('*.jpg'))):
        # Read image
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
            save_path = processed_dir / f"{img_path.stem}_{aug_name}.jpg"
            cv2.imwrite(str(save_path), cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    process_dataset() 