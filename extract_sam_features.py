#!/usr/bin/env python3
"""
Extract SAM features from all images in the dataset for training preparation.
Uses the existing SAM encoder to extract features from images in data/cleaned_resized.
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

class SAMFeatureExtractorPipeline:
    """Pipeline for extracting SAM features from all images in the dataset."""
    
    def __init__(self, 
                 csv_path: str = "data/metadata/unified_labels_with_stratified_splits.csv",
                 image_root: str = "data/cleaned_resized",
                 output_dir: str = "data/processed/features",
                 model_type: str = "vit_h",
                 batch_size: int = 8,
                 device: Optional[str] = None):
        """
        Initialize the feature extraction pipeline.
        
        Args:
            csv_path: Path to the CSV file with image metadata
            image_root: Root directory containing the image datasets
            output_dir: Directory to save extracted features
            model_type: SAM model type ('vit_h', 'vit_l', 'vit_b')
            batch_size: Batch size for feature extraction
            device: Device to run on ('cuda', 'cpu', or None for auto)
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
        
    def load_metadata(self):
        """Load the CSV metadata file."""
        logger.info(f"Loading metadata from {self.csv_path}")
        self.df = pd.read_csv(self.csv_path)
        logger.info(f"Loaded {len(self.df)} image records")
        
        # Display basic info
        logger.info(f"Columns: {list(self.df.columns)}")
        logger.info(f"Unique diagnoses: {self.df['diagnosis'].unique()}")
        logger.info(f"Unique unified labels: {self.df['unified_label'].unique()}")
        
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
    
    def process_dataset(self, split_name: str = None):
        """Process all images in the dataset and extract features."""
        logger.info(f"Starting feature extraction for {len(self.df)} images")
        
        # Filter by split if specified
        if split_name:
            split_columns = [col for col in self.df.columns if col.startswith('split')]
            split_mask = self.df[split_columns].apply(lambda x: x == split_name).any(axis=1)
            df_to_process = self.df[split_mask].copy()
            logger.info(f"Processing {len(df_to_process)} images for split: {split_name}")
        else:
            df_to_process = self.df.copy()
        
        # Prepare results storage
        features_dict = {}
        failed_images = []
        
        # Process in batches
        for i in tqdm(range(0, len(df_to_process), self.batch_size), desc="Extracting features"):
            batch_df = df_to_process.iloc[i:i+self.batch_size]
            batch_paths = []
            batch_names = []
            
            # Prepare batch
            for _, row in batch_df.iterrows():
                image_name = row['image']
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
                    features_dict[name] = features[j].cpu().numpy()
                
                # Save incrementally every 100 batches (or every batch if batch_size=1)
                if (i // self.batch_size) % 100 == 0 or self.batch_size == 1:
                    self.save_features_incremental(features_dict, split_name, i // self.batch_size)
        
        # Save final results
        self.save_features(features_dict, split_name)
        
        # Log statistics
        logger.info(f"Successfully extracted features for {len(features_dict)} images")
        if failed_images:
            logger.warning(f"Failed to process {len(failed_images)} images")
            logger.warning(f"Failed images: {failed_images[:10]}...")  # Show first 10
        
        return features_dict, failed_images
    
    def save_features(self, features_dict: Dict[str, np.ndarray], split_name: str = None):
        """Save extracted features to disk."""
        if split_name:
            output_file = self.output_dir / f"sam_features_{split_name}.pkl"
        else:
            output_file = self.output_dir / "sam_features_all.pkl"
        
        # Save features
        with open(output_file, 'wb') as f:
            pickle.dump(features_dict, f)
        
        logger.info(f"Saved features to {output_file}")
        
        # Also save as numpy array with metadata
        if split_name:
            metadata_file = self.output_dir / f"sam_features_{split_name}_metadata.pkl"
        else:
            metadata_file = self.output_dir / "sam_features_all_metadata.pkl"
        
        metadata = {
            'feature_dim': 256,
            'model_type': 'vit_h',
            'num_images': len(features_dict),
            'image_names': list(features_dict.keys())
        }
        
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Saved metadata to {metadata_file}")
    
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
            'model_type': 'vit_b',  # Updated to match current model
            'num_images': len(features_dict),
            'image_names': list(features_dict.keys()),
            'last_batch': batch_num,
            'last_save_time': str(pd.Timestamp.now())
        }
        
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
    
    def process_all_splits(self):
        """Process all splits separately."""
        split_columns = [col for col in self.df.columns if col.startswith('split')]
        
        for split_col in split_columns:
            unique_splits = self.df[split_col].dropna().unique()
            for split_name in unique_splits:
                logger.info(f"Processing split: {split_name} from column: {split_col}")
                self.process_dataset(split_name)

def main():
    parser = argparse.ArgumentParser(description="Extract SAM features from dataset images")
    parser.add_argument("--csv-path", default="data/metadata/unified_labels_with_stratified_splits.csv",
                       help="Path to CSV file with image metadata")
    parser.add_argument("--image-root", default="data/cleaned_resized",
                       help="Root directory containing image datasets")
    parser.add_argument("--output-dir", default="data/processed/features",
                       help="Directory to save extracted features")
    parser.add_argument("--model-type", default="vit_h", choices=["vit_h", "vit_l", "vit_b"],
                       help="SAM model type to use")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size for feature extraction")
    parser.add_argument("--device", default=None,
                       help="Device to use ('cuda', 'cpu', or None for auto)")
    parser.add_argument("--split", default=None,
                       help="Specific split to process (e.g., 'train', 'val', 'test')")
    parser.add_argument("--all-splits", action="store_true",
                       help="Process all splits separately")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = SAMFeatureExtractorPipeline(
        csv_path=args.csv_path,
        image_root=args.image_root,
        output_dir=args.output_dir,
        model_type=args.model_type,
        batch_size=args.batch_size,
        device=args.device
    )
    
    # Process dataset
    if args.all_splits:
        pipeline.process_all_splits()
    else:
        features_dict, failed_images = pipeline.process_dataset(args.split)
        
        # Print summary
        print(f"\nFeature extraction completed!")
        print(f"Successfully processed: {len(features_dict)} images")
        print(f"Failed to process: {len(failed_images)} images")
        if failed_images:
            print(f"First 5 failed images: {failed_images[:5]}")

if __name__ == "__main__":
    main() 