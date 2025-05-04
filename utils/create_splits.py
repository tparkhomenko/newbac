import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_base_image_name(filename):
    """Extract base image name without augmentation suffix."""
    # Remove augmentation suffixes (_original, _h_flip, etc.)
    parts = filename.split('/')[-1].split('_')
    if 'ISIC' in filename:
        return parts[0] + '_' + parts[1]  # ISIC_XXXXXXX
    else:
        return '_'.join(parts[:-2])  # texture_name_XXXX

def create_splits():
    # Load metadata
    metadata_path = Path("/home/parkhomenko/Documents/new_project/datasets/metadata/unified_augmented.csv")
    df = pd.read_csv(metadata_path)
    logger.info(f"Loaded {len(df)} images from metadata")
    
    # Extract base image names and dataset type
    df['base_image'] = df['image'].apply(get_base_image_name)
    df['dataset'] = df['image'].apply(lambda x: 'isic' if 'isic2019_aug' in x else 'dtd')
    
    # Process ISIC and DTD separately
    split_mapping = {}
    
    for dataset in ['isic', 'dtd']:
        # Get unique base images for this dataset
        dataset_df = df[df['dataset'] == dataset].drop_duplicates('base_image')
        bases = dataset_df['base_image'].values
        logger.info(f"\nProcessing {dataset.upper()} dataset:")
        logger.info(f"Found {len(bases)} unique base images")
        
        # Random shuffle
        np.random.seed(42)
        np.random.shuffle(bases)
        
        # Split indices
        n_total = len(bases)
        n_train = int(0.7 * n_total)
        n_val = int(0.15 * n_total)
        
        # Assign splits
        for img in bases[:n_train]:
            split_mapping[img] = 'train'
        for img in bases[n_train:n_train + n_val]:
            split_mapping[img] = 'val'
        for img in bases[n_train + n_val:]:
            split_mapping[img] = 'test'
        
        # Log split sizes for this dataset
        logger.info(f"Train: {n_train} images")
        logger.info(f"Val: {n_val} images")
        logger.info(f"Test: {len(bases) - n_train - n_val} images")
    
    # Add splits to main dataframe
    df['split'] = df['base_image'].map(split_mapping)
    
    # Remove temporary columns
    df = df.drop(['base_image', 'dataset'], axis=1)
    
    # Save updated metadata
    df.to_csv(metadata_path, index=False)
    
    # Print statistics
    logger.info("\nFinal Split Statistics:")
    logger.info(df['split'].value_counts())
    logger.info("\nSplit Statistics by Skin/Non-Skin:")
    logger.info(pd.crosstab(df['split'], df['skin']))
    logger.info("\nSplit Statistics by Lesion Group:")
    logger.info(pd.crosstab(df['split'], df['lesion_group']))

def filter_balanced_training_set():
    """Filter metadata for balanced training: all augmented for fibrous/vascular, downsampled originals for others."""
    metadata_path = Path("/home/parkhomenko/Documents/new_project/datasets/metadata/unified_augmented.csv")
    out_path = Path("/home/parkhomenko/Documents/new_project/datasets/metadata/balanced_train_metadata.csv")
    df = pd.read_csv(metadata_path)
    # Classes to use all augmentations
    aug_classes = ["fibrous", "vascular"]
    # Classes to use only original images
    orig_classes = ["melanocytic", "non-melanocytic carcinoma", "keratosis", "not_skin_texture", "unknown"]
    # Get all augmented for fibrous/vascular
    aug_df = df[df["lesion_group"].isin(aug_classes) & df["split"].eq("train")]
    # Get only originals for other classes
    orig_df = df[df["lesion_group"].isin(orig_classes) & df["image"].str.contains("_original.jpg") & df["split"].eq("train")]
    # Downsample all to the minimum class size
    min_count = orig_df["lesion_group"].value_counts().min()
    balanced_orig = orig_df.groupby("lesion_group").sample(n=min_count, random_state=42)
    # Combine
    balanced = pd.concat([aug_df, balanced_orig], ignore_index=True)
    balanced.to_csv(out_path, index=False)
    print(f"Saved balanced training metadata: {len(balanced)} images to {out_path}")

if __name__ == "__main__":
    create_splits()
    filter_balanced_training_set() 