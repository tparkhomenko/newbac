import torch
import numpy as np

def cutmix(features, targets, alpha=1.0):
    """
    Implements CutMix augmentation for feature vectors.
    
    Args:
        features: Input features tensor of shape [batch_size, feature_dim]
        targets: Target labels tensor of shape [batch_size]
        alpha: CutMix hyperparameter
        
    Returns:
        mixed_features: Features after applying CutMix
        targets_a: First set of targets
        targets_b: Second set of targets
        lam: Mixing coefficient
    """
    batch_size = features.size(0)
    
    # Generate random indices for the second samples
    indices = torch.randperm(batch_size).to(features.device)
    
    # Generate random mixing coefficient
    lam = np.random.beta(alpha, alpha)
    
    # Convert to feature-wise mixing
    # Each feature has lam probability of coming from the first sample
    # and (1-lam) probability from the second
    mask = torch.bernoulli(torch.full((features.size(1),), lam)).to(features.device)
    
    # Expand mask to match features shape
    mask = mask.unsqueeze(0).expand(batch_size, -1)
    
    # Create mixed features
    mixed_features = features * mask + features[indices] * (1 - mask)
    
    return mixed_features, targets, targets[indices], lam

def mixup(features, targets, alpha=0.2):
    """
    Implements MixUp augmentation for feature vectors.
    
    Args:
        features: Input features tensor of shape [batch_size, feature_dim]
        targets: Target labels tensor of shape [batch_size]
        alpha: MixUp hyperparameter
        
    Returns:
        mixed_features: Features after applying MixUp
        targets_a: First set of targets
        targets_b: Second set of targets
        lam: Mixing coefficient
    """
    batch_size = features.size(0)
    
    # Generate random indices for the second samples
    indices = torch.randperm(batch_size).to(features.device)
    
    # Generate random mixing coefficient
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1-lam)  # Ensure lambda is at least 0.5 to preserve more of the primary feature
    
    # Mix features
    mixed_features = lam * features + (1 - lam) * features[indices]
    
    return mixed_features, targets, targets[indices], lam

def create_saliency_map(model, features, target_class, device):
    """
    Creates a saliency map showing which features most influence the prediction.
    
    Args:
        model: Trained model
        features: Input features tensor of shape [feature_dim]
        target_class: Target class index
        device: Computation device
        
    Returns:
        saliency: Saliency map
    """
    # Reset gradients
    model.zero_grad()
    
    # Create input with gradient tracking
    features = features.unsqueeze(0).to(device)
    features.requires_grad_()
    
    # Forward pass
    output = model(features)
    
    # Get target class score
    target_score = output[0, target_class]
    
    # Backward pass
    target_score.backward()
    
    # Get gradients
    saliency = features.grad.abs().squeeze().cpu().numpy()
    
    return saliency 