#!/usr/bin/env python3
"""
Resume SAM feature extraction from where we left off.
Loads existing incremental features and continues processing.
"""

import os
import pandas as pd
import torch
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import pickle
from typing import Dict, List, Optional
import argparse

# Add the project root to the path
import sys
sys.path.append(str(Path(__file__).parent))

from sam.sam_encoder import SAMFeatureExtractor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SAMFeatureExtractorResume:
    """Resume SAM feature extraction from existing incremental file."""
    
    def __init__(self, 
                 csv_path: str = "data/metadata/unified_labels_with_stratified_splits.csv",
                 image_root: str = "data/cleaned_resized",
                 output_dir: str = "data/processed/features",
                 model_type: str = "vit_b",
                 batch_size: int = 1,
                 device: Optional[str] = None):
        """
        Initialize the feature extraction pipeline.
        """
        self.csv_path = csv_path
        self.image_root = Path(image_root)
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize SAM feature extractor
        logger.info(f"Initializing SAM feature extractor with model type: {model_type}")
        self.extractor = SAMFeatureExtractor(model_type=model_type, device=self.device)
        
        # Load metadata
        self.load_metadata()
        
        # Find all image files
        self.find_image_files()
        
        # Load existing features
        self.load_existing_features()
        
    def load_metadata(self):
        """Load the CSV metadata file."""
        logger.info(f"Loading metadata from {self.csv_path}")
        self.df = pd.read_csv(self.csv_path)
        logger.info(f"Loaded {len(self.df)} image records")
        
    def find_image_files(self):
        """Find all image files in the cleaned_resized directory."""
        logger.info("Scanning for image files...")
        
        self.image_files = {}
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        # Scan all subdirectories in cleaned_resized
        for dataset_dir in self.image_root.iterdir():
            if dataset_dir.is_dir():
                logger.info(f"Scanning {dataset_dir.name}")
                for img_file in dataset_dir.rglob("*"):
                    if img_file.is_file() and img_file.suffix.lower() in image_extensions:
                        # Use filename without extension as key
                        img_key = img_file.stem
                        self.image_files[img_key] = str(img_file)
        
        logger.info(f"Found {len(self.image_files)} image files")
        
    def load_existing_features(self):
        """Load existing incremental features."""
        # Prefer 'all' incremental files if present, otherwise fall back to train-specific files
        all_incremental_file = self.output_dir / "sam_features_all_incremental.pkl"
        all_metadata_file = self.output_dir / "sam_features_all_incremental_metadata.pkl"
        train_incremental_file = self.output_dir / "sam_features_train_incremental.pkl"
        train_metadata_file = self.output_dir / "sam_features_train_incremental_metadata.pkl"

        incremental_file = None
        metadata_file = None
        if all_incremental_file.exists():
            incremental_file = all_incremental_file
            metadata_file = all_metadata_file if all_metadata_file.exists() else None
        elif train_incremental_file.exists():
            incremental_file = train_incremental_file
            metadata_file = train_metadata_file if train_metadata_file.exists() else None

        self.features_dict = {}
        self.metadata = {}

        if incremental_file:
            # Load features dict regardless of metadata state
            try:
                with open(incremental_file, 'rb') as f:
                    self.features_dict = pickle.load(f)
                logger.info(f"Loaded existing features from {incremental_file.name}: {len(self.features_dict)} images")
            except Exception as e:
                logger.warning(f"Could not load existing features from {incremental_file.name}: {e}")
                self.features_dict = {}

            # Best-effort load metadata
            last_batch = 0
            if metadata_file:
                try:
                    with open(metadata_file, 'rb') as f:
                        self.metadata = pickle.load(f)
                    last_batch = int(self.metadata.get('last_batch', 0))
                except Exception as e:
                    logger.warning(f"Could not load metadata from {metadata_file.name}: {e}")
                    self.metadata = {}
            self.metadata['last_batch'] = last_batch
            logger.info(f"Last batch: {self.metadata.get('last_batch', 0)}")
        else:
            logger.info("No existing features found, starting fresh")
    
    def get_image_path(self, image_name: str) -> Optional[str]:
        """Get the full path for an image by name."""
        # Try different possible extensions and name variations
        possible_keys = [
            image_name,
            image_name.replace('.jpg', ''),
            image_name.replace('.jpeg', ''),
            image_name.replace('.png', ''),
        ]
        
        for key in possible_keys:
            if key in self.image_files:
                return self.image_files[key]
        
        return None
    
    def extract_features_batch(self, image_paths: List[str]) -> torch.Tensor:
        """Extract features for a batch of images."""
        try:
            features = self.extractor.extract_features(image_paths)
            return features
        except Exception as e:
            logger.error(f"Error extracting features for batch: {e}")
            # Return zero features for failed batch
            return torch.zeros(len(image_paths), 256, device=self.device)
    
    def save_features_incremental(self, features_dict: Dict[str, np.ndarray], split_name: str = None, batch_num: int = 0):
        """Save extracted features incrementally to avoid losing progress."""
        if split_name:
            output_file = self.output_dir / f"sam_features_{split_name}_incremental.pkl"
        else:
            output_file = self.output_dir / "sam_features_all_incremental.pkl"
        
        # Save features incrementally
        with open(output_file, 'wb') as f:
            pickle.dump(features_dict, f)
        
        logger.info(f"Saved incremental features to {output_file} (batch {batch_num}, {len(features_dict)} images)")
        
        # Also save metadata
        if split_name:
            metadata_file = self.output_dir / f"sam_features_{split_name}_incremental_metadata.pkl"
        else:
            metadata_file = self.output_dir / "sam_features_all_incremental_metadata.pkl"
        
        metadata = {
            'feature_dim': 256,
            'model_type': 'vit_b',
            'num_images': len(features_dict),
            'image_names': list(features_dict.keys()),
            'last_batch': batch_num,
            'last_save_time': str(pd.Timestamp.now())
        }
        
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
    
    def process_dataset_resume(self, split_name: str = None):
        """Process remaining images in the dataset and extract features."""
        logger.info(f"Resuming feature extraction for {len(self.df)} images")
        
        # Filter by split if specified and not 'all'
        if split_name:
            split_columns = [col for col in self.df.columns if col.startswith('split')]
            split_mask = self.df[split_columns].apply(lambda x: x == split_name).any(axis=1)
            df_to_process = self.df[split_mask].copy()
            logger.info(f"Processing {len(df_to_process)} images for split: {split_name}")
        else:
            df_to_process = self.df.copy()
        
        # Filter out already processed images
        already_processed = set(self.features_dict.keys())
        df_to_process = df_to_process[~df_to_process['image_name'].isin(already_processed)]
        logger.info(f"Remaining images to process: {len(df_to_process)}")
        
        # Prepare results storage
        failed_images = []
        
        # Process in batches
        start_batch = self.metadata.get('last_batch', 0) + 1
        for i in tqdm(range(0, len(df_to_process), self.batch_size), desc="Extracting features"):
            batch_df = df_to_process.iloc[i:i+self.batch_size]
            batch_paths = []
            batch_names = []
            
            # Prepare batch
            for _, row in batch_df.iterrows():
                image_name = row['image_name']
                image_path = self.get_image_path(image_name)
                
                if image_path and os.path.exists(image_path):
                    batch_paths.append(image_path)
                    batch_names.append(image_name)
                else:
                    logger.warning(f"Image not found: {image_name}")
                    failed_images.append(image_name)
            
            if batch_paths:
                # Extract features for this batch
                features = self.extract_features_batch(batch_paths)
                
                # Store features
                for j, name in enumerate(batch_names):
                    self.features_dict[name] = features[j].cpu().numpy()
                
                # Save incrementally every batch (since batch_size=1)
                batch_num = start_batch + (i // self.batch_size)
                self.save_features_incremental(self.features_dict, split_name, batch_num)
        
        # Log statistics
        logger.info(f"Successfully extracted features for {len(self.features_dict)} images")
        if failed_images:
            logger.warning(f"Failed to process {len(failed_images)} images")
            logger.warning(f"Failed images: {failed_images[:10]}...")  # Show first 10
        
        return self.features_dict, failed_images

def main():
    parser = argparse.ArgumentParser(description="Resume SAM feature extraction")
    parser.add_argument("--csv-path", default="data/metadata/unified_labels_with_stratified_splits.csv",
                       help="Path to CSV file with image metadata")
    parser.add_argument("--image-root", default="data/cleaned_resized",
                       help="Root directory containing image datasets")
    parser.add_argument("--output-dir", default="data/processed/features",
                       help="Directory to save extracted features")
    parser.add_argument("--model-type", default="vit_b", choices=["vit_h", "vit_l", "vit_b"],
                       help="SAM model type to use")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Batch size for feature extraction")
    parser.add_argument("--device", default=None,
                       help="Device to use ('cuda', 'cpu', or None for auto)")
    parser.add_argument("--split", default="train",
                       help="Specific split to process (e.g., 'train', 'val', 'test', or 'all' for all images)")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = SAMFeatureExtractorResume(
        csv_path=args.csv_path,
        image_root=args.image_root,
        output_dir=args.output_dir,
        model_type=args.model_type,
        batch_size=args.batch_size,
        device=args.device
    )
    
    # Process dataset
    split_arg = args.split if args.split != 'all' else None
    features_dict, failed_images = pipeline.process_dataset_resume(split_arg)
    
    # Print summary
    print(f"\nFeature extraction completed!")
    print(f"Successfully processed: {len(features_dict)} images")
    print(f"Failed to process: {len(failed_images)} images")
    if failed_images:
        print(f"First 5 failed images: {failed_images[:5]}")

if __name__ == "__main__":
    main() 