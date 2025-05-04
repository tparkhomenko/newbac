import torch
import torch.nn as nn
from segment_anything import sam_model_registry
from typing import Optional, Union, List
import logging
import os
import yaml
import urllib.request
from pathlib import Path
import numpy as np
from PIL import Image
import torchvision.transforms as T

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def download_model(url: str, save_path: str):
    """Download SAM model checkpoint if not exists."""
    if os.path.exists(save_path):
        logger.info(f"Model checkpoint already exists at {save_path}")
        return
    
    logger.info(f"Downloading model checkpoint from {url}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    try:
        urllib.request.urlretrieve(url, save_path)
        logger.info(f"Successfully downloaded model to {save_path}")
    except Exception as e:
        logger.error(f"Failed to download model: {str(e)}")
        raise

class SAMFeatureExtractor:
    """SAM2 feature extractor that returns frozen image embeddings."""
    
    def __init__(
        self,
        model_type: str = "vit_h",
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        """Initialize SAM feature extractor.
        
        Args:
            model_type: Type of SAM model ('vit_h', 'vit_l', 'vit_b')
            checkpoint_path: Path to model checkpoint
            device: Device to run model on ('cuda', 'cpu', or None for auto)
        """
        self.config = load_config()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Setup checkpoint path and download if needed
        if checkpoint_path is None:
            checkpoint_dir = Path(self.config['sam']['encoder']['checkpoint_dir'])
            checkpoint_path = checkpoint_dir / f"sam_{model_type}.pth"
            if not os.path.exists(checkpoint_path):
                url = self.config['sam']['encoder']['checkpoint_urls'][model_type]
                download_model(url, str(checkpoint_path))
        
        # Load SAM model
        try:
            self.model = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
            self.model.to(self.device)
            logger.info(f"Successfully loaded {model_type} model")
        except Exception as e:
            logger.error(f"Failed to load SAM model: {str(e)}")
            raise
            
        # Freeze model weights
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        logger.info("Model weights frozen")

        # Get image preprocessing size from config
        self.image_size = self.config['sam']['encoder']['image_size']
        
        # Setup image preprocessing
        self.transform = T.Compose([
            T.Resize((self.image_size, self.image_size), antialias=True),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def preprocess_image(self, image: Union[str, Image.Image, np.ndarray]) -> torch.Tensor:
        """Preprocess image for SAM model.
        
        Args:
            image: Input image as PIL Image, file path, or numpy array
            
        Returns:
            Preprocessed image tensor [C, H, W]
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise ValueError(f"Unsupported image type: {type(image)}")
            
        return self.transform(image)

    @torch.no_grad()
    def extract_features(self, images: Union[torch.Tensor, List[Union[str, Image.Image, np.ndarray]]]) -> torch.Tensor:
        """Extract features from images using SAM encoder.
        
        Args:
            images: Either a batch of preprocessed tensors [B, C, H, W] or
                   a list of images (as PIL Images, file paths, or numpy arrays)
            
        Returns:
            Image features tensor [B, D] where D is the feature dimension
        """
        # Handle different input types
        if isinstance(images, list):
            # Preprocess list of images
            processed = torch.stack([self.preprocess_image(img) for img in images])
        else:
            processed = images
            
        # Ensure we have a batch dimension
        if processed.dim() == 3:
            processed = processed.unsqueeze(0)
            
        # Move to device
        processed = processed.to(self.device)
        
        # Extract features
        features = self.model.image_encoder(processed)
        
        # Average pool spatial dimensions to get [B, D] tensor
        features = torch.mean(features, dim=(-2, -1))
        
        return features

if __name__ == "__main__":
    # Test model loading and feature extraction
    try:
        extractor = SAMFeatureExtractor()
        print("Successfully initialized SAMFeatureExtractor")
        
        # Create a dummy image for testing
        dummy_image = torch.randn(1, 3, 1024, 1024)
        features = extractor.extract_features(dummy_image)
        print(f"Extracted features shape: {features.shape}")
        
    except Exception as e:
        print(f"Error: {str(e)}") 