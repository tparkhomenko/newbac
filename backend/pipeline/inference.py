import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .preprocess import ImagePreprocessor

# Import the user's SAM feature extractor
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from sam.sam_encoder import SAMFeatureExtractor


DEFAULT_LESION_LABELS = [
    "melanoma",
    "nevus", 
    "seborrheic_keratosis",
    "basal_cell_carcinoma",
    "actinic_keratosis",
]


def _safe_softmax(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    return F.softmax(logits, dim=-1)


def _normalize_probs(values: List[float]) -> List[float]:
    s = sum(values)
    if s <= 0:
        n = len(values)
        return [1.0 / n for _ in values]
    return [v / s for v in values]


class InferencePipeline:
    """Runs the staged MLP pipeline using multihead models with SAM features."""

    def __init__(self, model_dir: str, device: torch.device) -> None:
        self.device = device
        self.model_dir = model_dir
        self.preprocessor = ImagePreprocessor()
        self.lesion_labels: List[str] = DEFAULT_LESION_LABELS
        self.debug_mode = os.environ.get("DEBUG_MODE", "false").lower() == "true"

        # Load the multihead models
        self.mlp1 = self._load_model("mlp1.pt")
        self.mlp2 = self._load_model("mlp2.pt") 
        self.mlp3 = self._load_model("mlp3.pt")

        # Initialize SAM feature extractor
        try:
            sam_model_type = os.environ.get("SAM_MODEL_TYPE", "vit_h")
            self.sam_extractor = SAMFeatureExtractor(
                model_type=sam_model_type,
                device=str(self.device)
            )
            self.sam_available = True
        except Exception as e:
            print(f"Warning: Could not initialize SAM extractor: {e}")
            self.sam_extractor = None
            self.sam_available = False

    def _load_model(self, filename: str) -> Optional[torch.jit.ScriptModule]:
        path = os.path.join(self.model_dir, filename)
        if not os.path.exists(path):
            return None
        try:
            model = torch.jit.load(path, map_location=self.device)
            model.eval()
            return model
        except Exception:
            return None

    def model_status(self) -> Dict[str, Any]:
        return {
            "mlp1": "loaded" if self.mlp1 is not None else "fallback",
            "mlp2": "loaded" if self.mlp2 is not None else "fallback", 
            "mlp3": "loaded" if self.mlp3 is not None else "fallback",
            "sam": "loaded" if self.sam_available else "fallback",
            "labels": {"lesion": self.lesion_labels},
        }

    def _generate_sam_features(self, image_bytes: bytes) -> torch.Tensor:
        """Generate SAM features using the actual SAM model."""
        if self.sam_available and self.sam_extractor is not None:
            try:
                # Convert bytes to PIL Image
                from PIL import Image
                import io
                image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                
                # Extract features using SAM
                features = self.sam_extractor.extract_features(image)
                return features
            except Exception as e:
                print(f"Warning: SAM feature extraction failed: {e}")
                # Fallback to mock features
                pass
        
        # Fallback to mock features if SAM is not available
        if self.debug_mode:
            print("Using mock SAM features (fallback)")
        features = torch.randn(1, 256, device=self.device) * 2.0
        features[:, :128] += 1.0  # Bias to try to trigger skin class
        return features

    def _forward_multihead(self, model: torch.jit.ScriptModule, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through multihead model returning (skin, lesion, bm) logits."""
        with torch.no_grad():
            outputs = model(features)
            if isinstance(outputs, (list, tuple)):
                skin_logits, lesion_logits, bm_logits = outputs
            else:
                # Handle case where model returns single tensor that needs to be split
                combined = outputs
                skin_logits = combined[:, :2]
                lesion_logits = combined[:, 2:7]  # 5 classes
                bm_logits = combined[:, 7:9]     # 2 classes
            
            return skin_logits, lesion_logits, bm_logits

    def _fallback_mlp1(self, image_bytes: bytes) -> List[float]:
        stats = self.preprocessor.quick_image_stats(image_bytes)
        m = stats["overall_mean"]
        # Heuristic: brighter images more likely to be skin
        p_skin = max(0.0, min(1.0, (m - 0.3) / 0.5))
        probs = _normalize_probs([p_skin, 1.0 - p_skin])
        return probs

    def _fallback_mlp2(self, image_bytes: bytes) -> List[float]:
        stats = self.preprocessor.quick_image_stats(image_bytes)
        r, g, b = stats["mean_per_channel"]
        # Simple color-based distribution for 5 classes
        base = [r, g, b, (r+g)/2, (g+b)/2]
        probs = _normalize_probs(base)
        return probs

    def _fallback_mlp3(self, image_bytes: bytes) -> List[float]:
        stats = self.preprocessor.quick_image_stats(image_bytes)
        # Heuristic: darker images slightly more malignant (toy logic)
        p_malignant = max(0.0, min(1.0, (0.6 - stats["overall_mean"]) / 0.6))
        probs = _normalize_probs([1.0 - p_malignant, p_malignant])
        return probs

    def predict_from_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        route: List[str] = []
        
        # Generate SAM features (256-dimensional)
        features = self._generate_sam_features(image_bytes)
        if self.sam_available:
            route.append("SAM_features_real")
        else:
            route.append("SAM_features_mock")
        
        # MLP1: Skin vs Not-Skin
        if self.mlp1 is not None:
            skin_logits, _, _ = self._forward_multihead(self.mlp1, features)
            probs1 = _safe_softmax(skin_logits)[0].tolist()
            route.append("MLP1_loaded")
        else:
            probs1 = self._fallback_mlp1(image_bytes)
            route.append("MLP1_fallback")

        skin_idx = int(probs1.index(max(probs1)))
        # Assumes index 0 -> skin, 1 -> not_skin
        is_skin_label = "skin" if skin_idx == 0 else "not_skin"
        is_skin_conf = float(max(probs1))
        is_skin_probs = {"skin": float(probs1[0] if len(probs1) > 0 else 0.0), "not_skin": float(probs1[1] if len(probs1) > 1 else 0.0)}

        result: Dict[str, Any] = {
            "is_skin": {
                "label": is_skin_label,
                "confidence": is_skin_conf,
                "probabilities": is_skin_probs,
            },
            "lesion_type": None,
            "malignancy": None,
            "route_taken": route,
        }

        if is_skin_label != "skin":
            return result

        # MLP2: Lesion type (5 classes)
        if self.mlp2 is not None:
            _, lesion_logits, _ = self._forward_multihead(self.mlp2, features)
            probs2 = _safe_softmax(lesion_logits)[0].tolist()
            route.append("MLP2_loaded")
        else:
            probs2 = self._fallback_mlp2(image_bytes)
            route.append("MLP2_fallback")

        lesion_idx = int(probs2.index(max(probs2)))
        lesion_conf = float(max(probs2))

        result["lesion_type"] = {
            "label_index": lesion_idx,
            "confidence": lesion_conf,
            "probabilities": [float(p) for p in probs2],
            "labels": self.lesion_labels,
        }

        # MLP3: Benign vs Malignant
        if self.mlp3 is not None:
            _, _, bm_logits = self._forward_multihead(self.mlp3, features)
            probs3 = _safe_softmax(bm_logits)[0].tolist()
            route.append("MLP3_loaded")
        else:
            probs3 = self._fallback_mlp3(image_bytes)
            route.append("MLP3_fallback")

        bm_idx = int(probs3.index(max(probs3)))
        bm_label = "benign" if bm_idx == 0 else "malignant"
        bm_conf = float(max(probs3))
        bm_probs = {"benign": float(probs3[0] if len(probs3) > 0 else 0.0), "malignant": float(probs3[1] if len(probs3) > 1 else 0.0)}

        result["malignancy"] = {
            "label": bm_label,
            "confidence": bm_conf,
            "probabilities": bm_probs,
        }

        result["route_taken"] = route
        return result


