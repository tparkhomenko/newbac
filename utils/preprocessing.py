import os
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import logging
from typing import List, Tuple, Dict
import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Unified preprocessor for ISIC and DTD datasets."""
    
    def __init__(self):
        self.image_size = 1024  # SAM format
        self.base_path = Path("/home/parkhomenko/Documents/new_project")
        
        # Input paths
        self.isic_path = self.base_path / "data/ISIC2019"
        self.dtd_path = self.base_path / "data/dtd/images"
        
        # Output paths
        self.isic_aug_path = self.base_path / "datasets/isic2019_aug"
        self.dtd_aug_path = self.base_path / "datasets/dtd_aug"
        
        # Create output directories
        self.isic_aug_path.mkdir(parents=True, exist_ok=True)
        self.dtd_aug_path.mkdir(parents=True, exist_ok=True)
        
        # Metadata paths
        self.metadata_path = self.base_path / "datasets/metadata"
        self.metadata_path.mkdir(parents=True, exist_ok=True)

    def resize_image(self, image: Image.Image) -> Image.Image:
        """Resize image to target size."""
        return image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)

    def apply_augmentations(self, image: Image.Image) -> List[Tuple[Image.Image, str]]:
        """Apply unified augmentation rules to image."""
        augmented = []
        
        # Original
        augmented.append((image, "original"))
        
        # Horizontal flip
        h_flip = image.transpose(Image.FLIP_LEFT_RIGHT)
        augmented.append((h_flip, "h_flip"))
        
        # Vertical flip
        v_flip = image.transpose(Image.FLIP_TOP_BOTTOM)
        augmented.append((v_flip, "v_flip"))
        
        # 90° rotation
        rot_90 = image.rotate(90, expand=True)
        augmented.append((rot_90, "rot_90"))
        
        # 270° rotation
        rot_270 = image.rotate(270, expand=True)
        augmented.append((rot_270, "rot_270"))
        
        return augmented

    def process_isic_dataset(self) -> pd.DataFrame:
        """Process ISIC dataset with augmentations."""
        metadata_rows = []
        
        # Get ISIC metadata
        isic_meta = pd.read_csv(self.base_path / "data/metadata/metadata_isic2019.csv")
        
        for idx, row in tqdm(isic_meta.iterrows(), desc="Processing ISIC images"):
            try:
                # Load and resize image
                img_path = self.isic_path / f"{row['image']}.jpg"
                image = Image.open(img_path).convert('RGB')
                image = self.resize_image(image)
                
                # Apply augmentations
                augmented_images = self.apply_augmentations(image)
                
                # Save augmented images and create metadata
                for aug_img, aug_type in augmented_images:
                    out_filename = f"{row['image']}_{aug_type}.jpg"
                    out_path = self.isic_aug_path / out_filename
                    aug_img.save(out_path, quality=95)
                    
                    metadata_rows.append({
                        'image': f"isic2019_aug/{out_filename}",
                        'lesion_group': row['lesion_group'],
                        'lesion': row['lesion'],
                        'malignancy': row['malignancy'],
                        'skin': 1
                    })
                    
            except Exception as e:
                logger.error(f"Error processing ISIC image {row['image']}: {str(e)}")
                continue
        
        return pd.DataFrame(metadata_rows)

    def process_dtd_dataset(self) -> pd.DataFrame:
        """Process DTD dataset with augmentations."""
        metadata_rows = []
        
        for texture_dir in tqdm(list(self.dtd_path.iterdir()), desc="Processing DTD images"):
            if not texture_dir.is_dir():
                continue
                
            for img_path in texture_dir.glob("*.jpg"):
                try:
                    # Load and resize image
                    image = Image.open(img_path).convert('RGB')
                    image = self.resize_image(image)
                    
                    # Apply augmentations
                    augmented_images = self.apply_augmentations(image)
                    
                    # Save augmented images and create metadata
                    for aug_img, aug_type in augmented_images:
                        out_filename = f"{img_path.stem}_{aug_type}.jpg"
                        out_path = self.dtd_aug_path / out_filename
                        aug_img.save(out_path, quality=95)
                        
                        metadata_rows.append({
                            'image': f"dtd_aug/{out_filename}",
                            'lesion_group': 'not_skin_texture',
                            'lesion': 'DTD',
                            'malignancy': 'not_applicable',
                            'skin': 0
                        })
                        
                except Exception as e:
                    logger.error(f"Error processing DTD image {img_path}: {str(e)}")
                    continue
        
        return pd.DataFrame(metadata_rows)

    def process_all(self):
        """Process both datasets and create unified metadata."""
        logger.info("Starting preprocessing...")
        
        # Process datasets
        isic_metadata = self.process_isic_dataset()
        dtd_metadata = self.process_dtd_dataset()
        
        # Combine metadata
        unified_metadata = pd.concat([isic_metadata, dtd_metadata], ignore_index=True)
        
        # Save metadata
        unified_metadata.to_csv(self.metadata_path / "unified_augmented.csv", index=False)
        logger.info(f"Saved unified metadata with {len(unified_metadata)} entries")
        
        # Print statistics
        logger.info("\nDataset Statistics:")
        logger.info(f"Total images: {len(unified_metadata)}")
        logger.info(f"ISIC images: {len(isic_metadata)}")
        logger.info(f"DTD images: {len(dtd_metadata)}")
        logger.info("\nSkin distribution:")
        logger.info(unified_metadata['skin'].value_counts())
        
        return unified_metadata

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    metadata = preprocessor.process_all() 