import io
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch


def _parse_float_list(env_value: Optional[str]) -> Optional[List[float]]:
    if not env_value:
        return None
    try:
        return [float(x.strip()) for x in env_value.split(",")]
    except Exception:
        return None


class ImagePreprocessor:
    """Converts raw image bytes into a normalized tensor suitable for model inference."""

    def __init__(
        self,
        image_size: int = 224,
        mean: Optional[Sequence[float]] = None,
        std: Optional[Sequence[float]] = None,
    ) -> None:
        self.image_size = int(os.environ.get("PREPROCESS_SIZE", str(image_size)))
        mean_from_env = _parse_float_list(os.environ.get("PREPROCESS_MEAN"))
        std_from_env = _parse_float_list(os.environ.get("PREPROCESS_STD"))

        # Defaults to common ImageNet normalization if not specified
        default_mean = [0.485, 0.456, 0.406]
        default_std = [0.229, 0.224, 0.225]

        self.mean: Tuple[float, float, float] = tuple(
            (mean_from_env or list(mean or default_mean))[:3]
        )  # type: ignore[assignment]
        self.std: Tuple[float, float, float] = tuple(
            (std_from_env or list(std or default_std))[:3]
        )  # type: ignore[assignment]

    def _load_image(self, image_bytes: bytes) -> Image.Image:
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image

    def _resize(self, image: Image.Image) -> Image.Image:
        return image.resize((self.image_size, self.image_size), Image.BILINEAR)

    def _to_tensor(self, image: Image.Image) -> torch.Tensor:
        array = np.asarray(image).astype("float32") / 255.0  # H W C in [0,1]
        array = np.transpose(array, (2, 0, 1))  # C H W
        tensor = torch.from_numpy(array)
        return tensor

    def _normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        for c in range(3):
            tensor[c] = (tensor[c] - self.mean[c]) / self.std[c]
        return tensor

    def preprocess_bytes(self, image_bytes: bytes) -> torch.Tensor:
        image = self._load_image(image_bytes)
        image = self._resize(image)
        tensor = self._to_tensor(image)
        tensor = self._normalize(tensor)
        return tensor.unsqueeze(0)  # 1 C H W

    def quick_image_stats(self, image_bytes: bytes) -> dict:
        image = self._load_image(image_bytes)
        array = np.asarray(image).astype("float32") / 255.0
        mean_per_channel = array.reshape(-1, 3).mean(axis=0).tolist()
        overall_mean = float(array.mean())
        return {
            "mean_per_channel": mean_per_channel,
            "overall_mean": overall_mean,
            "width": int(image.width),
            "height": int(image.height),
        }


