#!/usr/bin/env python3
"""
Extract SAM features for Exp6 dataset splits.
Uses the existing SAMFeatureExtractor to extract 256-dim vectors for each image in exp6.
"""

import os
import sys
import argparse
import pandas as pd
import torch
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import pickle
from typing import Dict, List, Optional

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from sam.sam_encoder import SAMFeatureExtractor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_image_path(image_root: Path, image_name: str) -> Optional[str]:
    """Find the full path for an image by name."""
    # Try different possible extensions and name variations
    possible_names = [
        image_name,
        f"{image_name}.jpg",
        f"{image_name}.jpeg", 
        f"{image_name}.png",
    ]
    
    for name in possible_names:
        # Search recursively in image_root
        for root, dirs, files in os.walk(image_root):
            if name in files:
                return os.path.join(root, name)
    
    return None


def extract_features_for_split(
    df_split: pd.DataFrame,
    image_root: Path,
    extractor: SAMFeatureExtractor,
    batch_size: int = 8,
    device: str = 'cuda'
) -> Dict[str, np.ndarray]:
    """Extract features for a single split."""
    features_dict = {}
    failed_images = []
    
    # Process in batches
    for i in tqdm(range(0, len(df_split), batch_size), desc=f"Extracting features"):
        batch_df = df_split.iloc[i:i+batch_size]
        batch_paths = []
        batch_names = []
        
        # Prepare batch
        for _, row in batch_df.iterrows():
            image_name = str(row['image_name']).strip()
            image_path = find_image_path(image_root, image_name)
            
            if image_path and os.path.exists(image_path):
                batch_paths.append(image_path)
                batch_names.append(image_name)
            else:
                logger.warning(f"Image not found: {image_name}")
                failed_images.append(image_name)
        
        if batch_paths:
            try:
                # Extract features for this batch
                with torch.no_grad():
                    features = extractor.extract_features(batch_paths)
                
                # Store features
                for j, name in enumerate(batch_names):
                    features_dict[name] = features[j].cpu().numpy()
                    
            except Exception as e:
                logger.error(f"Failed to extract features for batch {i//batch_size}: {e}")
                for name in batch_names:
                    failed_images.append(name)
    
    logger.info(f"Successfully extracted features for {len(features_dict)} images")
    if failed_images:
        logger.warning(f"Failed to process {len(failed_images)} images")
        logger.warning(f"Failed images: {failed_images[:10]}...")  # Show first 10
    
    return features_dict


def main():
    parser = argparse.ArgumentParser(description='Extract SAM features for Exp6 dataset')
    parser.add_argument('--force', action='store_true', help='Force re-extraction even if files exist')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for feature extraction')
    parser.add_argument('--model-type', type=str, default='vit_h', choices=['vit_h', 'vit_l', 'vit_b'], 
                       help='SAM model type')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    metadata_path = project_root / 'data/metadata/metadata.csv'
    image_root = project_root / 'data/cleaned_resized'
    features_dir = project_root / 'data/processed/features'
    
    # Create output directory
    features_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Check if features already exist
    output_files = {
        'train': features_dir / 'sam_features_exp6_train.pkl',
        'val': features_dir / 'sam_features_exp6_val.pkl', 
        'test': features_dir / 'sam_features_exp6_test.pkl'
    }
    
    if not args.force:
        existing_files = [f for f in output_files.values() if f.exists()]
        if existing_files:
            logger.info(f"Found existing feature files: {[f.name for f in existing_files]}")
            logger.info("Use --force to re-extract")
            return
    
    # Load metadata
    logger.info(f"Loading metadata from {metadata_path}")
    df = pd.read_csv(metadata_path)
    
    # Filter for exp6 rows
    df_exp6 = df[df['exp6'].isin(['train', 'val', 'test'])].copy()
    logger.info(f"Found {len(df_exp6)} images in exp6")
    
    # Print split counts
    split_counts = df_exp6['exp6'].value_counts()
    logger.info("Split counts:")
    for split, count in split_counts.items():
        logger.info(f"  {split}: {count}")
    
    # Initialize SAM feature extractor
    logger.info(f"Initializing SAM feature extractor with model type: {args.model_type}")
    extractor = SAMFeatureExtractor(model_type=args.model_type, device=device)
    
    # Extract features for each split
    for split in ['train', 'val', 'test']:
        logger.info(f"\nProcessing {split} split...")
        
        df_split = df_exp6[df_exp6['exp6'] == split].copy()
        if len(df_split) == 0:
            logger.warning(f"No images found for {split} split")
            continue
            
        logger.info(f"Extracting features for {len(df_split)} images in {split} split")
        
        # Extract features
        features_dict = extract_features_for_split(
            df_split, image_root, extractor, args.batch_size, device
        )
        
        # Save features
        output_path = output_files[split]
        logger.info(f"Saving {len(features_dict)} features to {output_path}")
        
        with open(output_path, 'wb') as f:
            pickle.dump(features_dict, f)
        
        # Print summary
        if features_dict:
            # Get feature shape from first item
            first_feature = next(iter(features_dict.values()))
            feature_shape = first_feature.shape
            logger.info(f"Feature matrix shape for {split}: {len(features_dict)} x {feature_shape[0]}")
        else:
            logger.warning(f"No features extracted for {split} split")
    
    logger.info("\nFeature extraction completed!")
    
    # Print final summary
    logger.info("\nFinal summary:")
    for split in ['train', 'val', 'test']:
        output_path = output_files[split]
        if output_path.exists():
            with open(output_path, 'rb') as f:
                features_dict = pickle.load(f)
            logger.info(f"{split}: {len(features_dict)} images, feature dim: {next(iter(features_dict.values())).shape[0] if features_dict else 'N/A'}")


if __name__ == "__main__":
    main()

