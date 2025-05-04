import sys
import os
from pathlib import Path
import torch
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from sam.sam_encoder import SAMFeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_sam_encoder():
    """Test SAM encoder initialization and feature extraction."""
    try:
        # Initialize feature extractor
        logger.info("Initializing SAM Feature Extractor...")
        extractor = SAMFeatureExtractor()
        
        # Test with random input
        logger.info("Testing with random input...")
        dummy_batch = torch.randn(2, 3, 1024, 1024)  # 2 images, RGB, 1024x1024
        features = extractor.extract_features(dummy_batch)
        
        # Print feature shape
        logger.info(f"Successfully extracted features with shape: {features.shape}")
        logger.info(f"Features device: {features.device}")
        logger.info(f"Features dtype: {features.dtype}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing SAM encoder: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_sam_encoder()
    if success:
        logger.info("✅ SAM encoder test passed!")
    else:
        logger.error("❌ SAM encoder test failed!") 