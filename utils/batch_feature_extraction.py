#!/usr/bin/env python3
"""
Batch feature extraction script for SAM2 encoder.
Extracts features from all images in data/cleaned_resized and saves them as .npy files.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import time
import gc
from PIL import Image
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from sam.sam_encoder import SAMFeatureExtractor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def find_all_images(input_dir):
    """Find all image files in the directory and subdirectories."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                image_files.append(os.path.join(root, file))
    
    return sorted(image_files)

def extract_features_batch(
    input_dir,
    output_dir,
    batch_size=4,
    device='cuda',
    model_type='vit_h',
    checkpoint_path=None
):
    """
    Extract SAM features from all images in input_dir and save to output_dir.
    
    Args:
        input_dir: Directory containing images
        output_dir: Directory to save .npy feature files
        batch_size: Number of images to process at once
        device: Device to run on ('cuda' or 'cpu')
        model_type: SAM model type ('vit_h', 'vit_l', 'vit_b')
        checkpoint_path: Path to SAM checkpoint (optional)
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all images
    logger.info(f"Scanning for images in {input_dir}...")
    image_files = find_all_images(input_dir)
    logger.info(f"Found {len(image_files)} images")
    
    if len(image_files) == 0:
        logger.error(f"No images found in {input_dir}")
        return
    
    # Initialize SAM feature extractor
    logger.info(f"Initializing SAM {model_type} feature extractor...")
    try:
        feature_extractor = SAMFeatureExtractor(
            model_type=model_type,
            checkpoint_path=checkpoint_path,
            device=device
        )
        logger.info("✅ SAM feature extractor initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize SAM feature extractor: {str(e)}")
        return
    
    # Process images in batches
    processed_count = 0
    failed_count = 0
    skipped_count = 0
    
    logger.info(f"Starting batch processing with batch_size={batch_size}")
    
    for i in tqdm(range(0, len(image_files), batch_size), desc="Processing batches"):
        batch_files = image_files[i:i + batch_size]
        batch_images = []
        batch_names = []
        
        # Load batch of images
        for image_path in batch_files:
            try:
                # Load image
                image = Image.open(image_path).convert('RGB')
                batch_images.append(image)
                
                # Get output filename
                image_name = Path(image_path).stem
                batch_names.append(image_name)
                
            except Exception as e:
                logger.warning(f"Failed to load {image_path}: {str(e)}")
                failed_count += 1
                continue
        
        if not batch_images:
            continue
        
        try:
            # Extract features for the batch
            start_time = time.time()
            features = feature_extractor.extract_features(batch_images)
            extraction_time = time.time() - start_time
            
            # Save features for each image
            for j, (image_name, feature) in enumerate(zip(batch_names, features)):
                output_path = os.path.join(output_dir, f"{image_name}.npy")
                
                # Check if file already exists
                if os.path.exists(output_path):
                    logger.debug(f"Skipping {image_name} (already exists)")
                    skipped_count += 1
                    continue
                
                # Save feature
                feature_np = feature.cpu().numpy()
                np.save(output_path, feature_np)
                processed_count += 1
            
            # Clean up batch
            for image in batch_images:
                image.close()
            del batch_images, features
            torch.cuda.empty_cache()
            gc.collect()
            
            # Log progress
            if (i // batch_size + 1) % 10 == 0:
                logger.info(f"Processed {processed_count} images, failed {failed_count}, skipped {skipped_count}")
                logger.info(f"Last batch extraction time: {extraction_time:.2f}s")
                
        except Exception as e:
            logger.error(f"Failed to process batch starting at index {i}: {str(e)}")
            failed_count += len(batch_images)
            # Clean up on error
            for image in batch_images:
                image.close()
            torch.cuda.empty_cache()
            gc.collect()
    
    # Final summary
    logger.info("=" * 60)
    logger.info("FEATURE EXTRACTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total images found: {len(image_files)}")
    logger.info(f"Successfully processed: {processed_count}")
    logger.info(f"Failed: {failed_count}")
    logger.info(f"Skipped (already existed): {skipped_count}")
    logger.info(f"Output directory: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Batch SAM feature extraction")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/cleaned_resized",
        help="Directory containing images to process"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/features",
        help="Directory to save extracted features"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of images to process at once"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run on"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="vit_h",
        choices=["vit_h", "vit_l", "vit_b"],
        help="SAM model type"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to SAM checkpoint (optional)"
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        logger.error(f"Input directory does not exist: {args.input_dir}")
        return
    
    # Check if output directory exists and warn about overwriting
    if os.path.exists(args.output_dir):
        existing_files = len([f for f in os.listdir(args.output_dir) if f.endswith('.npy')])
        if existing_files > 0:
            logger.warning(f"Output directory {args.output_dir} already contains {existing_files} .npy files")
            logger.warning("Existing files will be skipped (not overwritten)")
    
    # Run feature extraction
    logger.info("Starting batch SAM feature extraction...")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Model type: {args.model_type}")
    
    extract_features_batch(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device,
        model_type=args.model_type,
        checkpoint_path=args.checkpoint_path
    )

if __name__ == "__main__":
    main() 