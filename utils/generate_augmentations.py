import os
import sys
from pathlib import Path
import logging
import shutil
from PIL import Image
import torchvision.transforms.functional as TF

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_augmentations(image_path, output_dir):
    """Create augmented versions of an image.
    
    Args:
        image_path: Path to original image
        output_dir: Directory to save augmented images
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    base_name = Path(image_path).stem
    
    # Original
    image.save(output_dir / f"{base_name}_original.jpg")
    
    # Horizontal flip
    TF.hflip(image).save(output_dir / f"{base_name}_h_flip.jpg")
    
    # Vertical flip
    TF.vflip(image).save(output_dir / f"{base_name}_v_flip.jpg")
    
    # 90° rotation
    TF.rotate(image, 90).save(output_dir / f"{base_name}_rot_90.jpg")
    
    # 270° rotation
    TF.rotate(image, 270).save(output_dir / f"{base_name}_rot_270.jpg")

def main():
    """Generate augmented datasets."""
    try:
        # Setup paths
        data_root = project_root / "data"
        datasets_root = project_root / "datasets"
        isic_dir = data_root / "ISIC2019"
        aug_dir = datasets_root / "isic2019_aug"
        
        # Create output directory
        aug_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created augmented dataset directory: {aug_dir}")
        
        # Process ISIC images
        total_images = len(list(isic_dir.glob("*.jpg")))
        logger.info(f"Found {total_images} ISIC images to process")
        
        for i, image_path in enumerate(isic_dir.glob("*.jpg"), 1):
            if i % 100 == 0:
                logger.info(f"Processing image {i}/{total_images}")
            create_augmentations(image_path, aug_dir)
        
        logger.info("✅ Successfully generated augmented dataset!")
        logger.info(f"Augmented images saved to: {aug_dir}")
        
    except Exception as e:
        logger.error(f"❌ Failed to generate augmented dataset: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 