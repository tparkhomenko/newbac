import io
import json
import os
import csv
import glob
from typing import Any, Dict, List, Optional

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .pipeline.inference import InferencePipeline


class BinaryPrediction(BaseModel):
    label: str = Field(description="Predicted class label, e.g., 'skin' or 'not_skin'")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence for the predicted label")
    probabilities: Dict[str, float] = Field(description="Per-class probabilities")


class MultiClassPrediction(BaseModel):
    label_index: int = Field(description="Index of the predicted class")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence for the predicted class")
    probabilities: List[float] = Field(description="Per-class probabilities aligned with indices")
    labels: Optional[List[str]] = Field(default=None, description="Optional class labels for indices")


class MetadataInfo(BaseModel):
    image_name: str = Field(description="Image filename without extension")
    csv_source: str = Field(description="Source CSV file")
    diagnosis_from_csv: str = Field(description="Original diagnosis from CSV")
    unified_diagnosis: str = Field(description="Unified diagnosis label")
    exp1: str = Field(description="Experiment 1 split")
    exp2: str = Field(description="Experiment 2 split")
    exp3: str = Field(description="Experiment 3 split")
    exp4: str = Field(description="Experiment 4 split")
    exp5: str = Field(description="Experiment 5 split")


class PredictionResponse(BaseModel):
    is_skin: BinaryPrediction
    lesion_type: Optional[MultiClassPrediction] = None
    malignancy: Optional[BinaryPrediction] = None
    route_taken: List[str]
    metadata: Optional[MetadataInfo] = None


class ModelInfo(BaseModel):
    current_model: str = Field(description="Currently active architecture (parallel|multi)")
    available: List[str] = Field(description="Available architectures")
    stats: Dict[str, Any] = Field(description="Per-MLP stats parsed from logs for the active architecture")


class ModelSwitchRequest(BaseModel):
    architecture: str = Field(description="Architecture to switch to: 'parallel' or 'multi'")


class QuickTestResult(BaseModel):
    total_images: int = Field(description="Total number of images tested")
    accuracy: float = Field(description="Overall accuracy")
    f1_score: float = Field(description="F1 score")
    per_class_accuracy: Dict[str, float] = Field(description="Accuracy per class")
    confusion_matrix: List[List[int]] = Field(description="Confusion matrix")
    test_time: float = Field(description="Total testing time in seconds")
    model_used: str = Field(description="Model used for testing")


def _get_models_base_dir() -> str:
    return os.environ.get(
        "MODEL_DIR",
        os.path.join(os.path.dirname(__file__), "models"),
    )


def _get_model_dir_for_architecture(architecture: str) -> str:
    return os.path.join(_get_models_base_dir(), architecture)


def _get_device() -> torch.device:
    preferred = os.environ.get("TORCH_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    try:
        return torch.device(preferred)
    except Exception:
        return torch.device("cpu")


def _load_metadata() -> Dict[str, Dict[str, str]]:
    """Load metadata CSV into memory for fast lookups."""
    metadata_path = os.path.join(os.path.dirname(__file__), "..", "data", "metadata", "metadata.csv")
    metadata = {}
    
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Use image_name as key (without extension)
                image_name = row['image_name']
                metadata[image_name] = {
                    'image_name': row['image_name'],
                    'csv_source': row['csv_source'],
                    'diagnosis_from_csv': row['diagnosis_from_csv'],
                    'unified_diagnosis': row['unified_diagnosis'],
                    'exp1': row['exp1'],
                    'exp2': row['exp2'],
                    'exp3': row['exp3'],
                    'exp4': row['exp4'],
                    'exp5': row['exp5']
                }
        print(f"Loaded metadata for {len(metadata)} images")
    except Exception as e:
        print(f"Warning: Could not load metadata: {e}")
        metadata = {}
    
    return metadata


def _lookup_metadata(image_filename: str) -> Optional[Dict[str, str]]:
    """Look up metadata by image filename."""
    # Remove file extension to get image_name
    image_name = os.path.splitext(image_filename)[0]
    return metadata_cache.get(image_name)


def _get_available_architectures() -> List[str]:
    """Return available architectures based on folders present under models/"""
    base_dir = _get_models_base_dir()
    candidates = ["parallel", "multi"]
    available: List[str] = []
    for arch in candidates:
        if os.path.isdir(os.path.join(base_dir, arch)):
            available.append(arch)
    # If none exist yet, still expose both for UI
    return available or candidates


def _build_stats(architecture: str) -> Dict[str, Any]:
    # Mock stats; replace with real metrics if available
    return {
        "training_acc": 0.85,
        "training_f1": 0.83,
        "val_acc": 0.82,
        "val_f1": 0.80,
        "epochs": 100,
        "dataset": "exp1",
    }


def _get_model_stats_for_architecture(architecture: str) -> Dict[str, Any]:
    # Parse last lines of logs for mlp1..mlp3 for the given architecture
    base_dir = _get_model_dir_for_architecture(architecture)
    stats: Dict[str, Any] = {}
    for mlp in ["mlp1", "mlp2", "mlp3"]:
        log_path = os.path.join(base_dir, f"logs_{mlp}.txt")
        parsed = {
            "test_skin_acc": None,
            "test_lesion_acc": None,
            "test_bm_acc": None,
            "test_skin_f1_macro": None,
            "test_lesion_f1_macro": None,
            "test_bm_f1_macro": None,
        }
        try:
            if os.path.exists(log_path):
                with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()[-50:]
                # Extract scalar lines of the form: wandb: key value
                for line in lines:
                    line = line.strip()
                    if line.startswith("wandb:"):
                        parts = line.split()
                        if len(parts) >= 3:
                            key = parts[1]
                            try:
                                value = float(parts[2])
                            except Exception:
                                continue
                            if key in parsed:
                                parsed[key] = value
        except Exception as e:
            print(f"Warning: failed to parse stats for {mlp} at {log_path}: {e}")
        stats[mlp] = parsed
    return stats


app = FastAPI(title="Skin Lesion Classifier API", version="1.0.0")

# CORS for simple frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


pipeline: Optional[InferencePipeline] = None
metadata_cache: Dict[str, Dict[str, str]] = {}
current_architecture: str = os.environ.get("MODEL_ARCH", "multi")


@app.on_event("startup")
def _startup() -> None:
    global pipeline, metadata_cache
    model_dir = _get_model_dir_for_architecture(current_architecture)
    device = _get_device()
    pipeline = InferencePipeline(
        model_dir=model_dir,
        device=device,
    )
    metadata_cache = _load_metadata()


@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    assert pipeline is not None
    return {
        "status": "ok",
        "device": str(pipeline.device),
        "models": pipeline.model_status(),
        "metadata_loaded": len(metadata_cache) > 0,
    }


@app.get("/model", response_model=ModelInfo)
def get_model_info() -> ModelInfo:
    available_arch = _get_available_architectures()
    stats = _get_model_stats_for_architecture(current_architecture)
    return ModelInfo(
        current_model=current_architecture,
        available=available_arch,
        stats=stats,
    )


@app.post("/model/switch")
def switch_model(request: ModelSwitchRequest) -> Dict[str, Any]:
    """Switch to a different architecture (parallel|multi)."""
    global current_architecture, pipeline

    arch = request.architecture.lower().strip()
    if arch not in {"parallel", "multi"}:
        raise HTTPException(status_code=400, detail="architecture must be 'parallel' or 'multi'")

    model_dir = _get_model_dir_for_architecture(arch)
    if not os.path.isdir(model_dir):
        # Allow switching even if folder not present yet, as per spec
        print(f"Warning: Model directory not found for architecture '{arch}': {model_dir}")

    try:
        current_architecture = arch
        device = _get_device()
        # Recreate pipeline which clears previous models from memory
        pipeline = InferencePipeline(
            model_dir=model_dir,
            device=device,
        )
        return {
            "status": "success",
            "message": f"Switched to architecture: {arch}",
            "current_model": current_architecture,
            "stats": _get_model_stats_for_architecture(current_architecture),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to switch architecture: {str(e)}")


@app.post("/quicktest", response_model=QuickTestResult)
def quick_test() -> QuickTestResult:
    """Run quick test on images in data/testing_only/ directory."""
    import time
    from PIL import Image
    import random
    import shutil
    
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    test_dir = os.path.join(os.path.dirname(__file__), "..", "data", "testing_only")
    if not os.path.exists(test_dir):
        raise HTTPException(status_code=404, detail="Testing directory not found")
    
    try:
        # Ensure directory has up to 100 exp1 test images by sampling from metadata
        # Source dataset directory (adjust if your dataset lives elsewhere)
        dataset_dir = os.path.join(os.path.dirname(__file__), "..", "data", "cleaned_resized", "isic2018_512")
        if os.path.isdir(dataset_dir) and metadata_cache:
            # Build list of candidate filenames from exp1 test
            exp1_test_names: List[str] = [
                f"{name}.jpg" for name, m in metadata_cache.items() if m.get("exp1") == "test"
            ]
            # Existing files in testing_only
            existing = set([os.path.basename(p) for p in glob.glob(os.path.join(test_dir, "*.jpg"))])
            # Sample until we have up to 100 files
            needed = max(0, 100 - len(existing))
            if needed > 0 and exp1_test_names:
                random.shuffle(exp1_test_names)
                to_copy: List[str] = []
                for fname in exp1_test_names:
                    if len(to_copy) >= needed:
                        break
                    if fname not in existing and os.path.exists(os.path.join(dataset_dir, fname)):
                        to_copy.append(fname)
                for fname in to_copy:
                    try:
                        shutil.copy2(os.path.join(dataset_dir, fname), os.path.join(test_dir, fname))
                    except Exception as copy_err:
                        print(f"Warning: failed to copy {fname}: {copy_err}")

        # Get test images
        image_files: List[str] = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(glob.glob(os.path.join(test_dir, ext)))
        if not image_files:
            raise HTTPException(status_code=404, detail="No test images found")

        # Limit to 100 images
        image_files = image_files[:100]

        start_time = time.time()

        # Run predictions
        correct_predictions = 0
        total_predictions = 0

        # Initialize confusion matrix and helpers
        confusion_matrix = [[0] * 5 for _ in range(5)]
        class_names = ['melanoma', 'nevus', 'seborrheic_keratosis', 'basal_cell_carcinoma', 'actinic_keratosis']
        class_counts = {name: 0 for name in class_names}
        class_correct = {name: 0 for name in class_names}
        label_mapping = {
            'NV': 'nevus',
            'AKIEC': 'actinic_keratosis', 
            'DF': 'dermatofibroma',
            'MEL': 'melanoma',
            'BKL': 'seborrheic_keratosis'
        }

        for image_path in image_files:
            try:
                with open(image_path, 'rb') as f:
                    image_bytes = f.read()
                # Get prediction
                result = pipeline.predict_from_bytes(image_bytes)
                # Ground truth
                filename = os.path.basename(image_path)
                metadata = _lookup_metadata(filename)
                if metadata and result['is_skin']['label'] == 'skin' and result['lesion_type']:
                    predicted_idx = result['lesion_type']['label_index']
                    predicted_label = result['lesion_type']['labels'][predicted_idx]
                    ground_truth_metadata = metadata['unified_diagnosis']
                    ground_truth = label_mapping.get(ground_truth_metadata, ground_truth_metadata)
                    try:
                        gt_idx = class_names.index(ground_truth)
                        confusion_matrix[gt_idx][predicted_idx] += 1
                        if predicted_label == ground_truth:
                            correct_predictions += 1
                            class_correct[ground_truth] += 1
                        class_counts[ground_truth] += 1
                        total_predictions += 1
                    except ValueError:
                        print(f"Warning: Ground truth '{ground_truth}' not found in class names: {class_names}")
                        continue
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue

        end_time = time.time()
        test_time = end_time - start_time

        # Metrics
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

        def _safe_div(a: float, b: float) -> float:
            return (a / b) if b != 0 else 0.0

        per_class_f1: List[float] = []
        num_classes = len(class_names)
        for c in range(num_classes):
            tp = confusion_matrix[c][c]
            fp = sum(confusion_matrix[r][c] for r in range(num_classes) if r != c)
            fn = sum(confusion_matrix[c][k] for k in range(num_classes) if k != c)
            denom = 2 * tp + fp + fn
            f1_c = _safe_div(2 * tp, denom)
            per_class_f1.append(f1_c)

        present_classes = [i for i, name in enumerate(class_names) if class_counts[name] > 0]
        if present_classes:
            f1_score = sum(per_class_f1[i] for i in present_classes) / len(present_classes)
        else:
            f1_score = 0.0

        per_class_accuracy: Dict[str, float] = {}
        for class_name in class_names:
            if class_counts[class_name] > 0:
                per_class_accuracy[class_name] = class_correct[class_name] / class_counts[class_name]
            else:
                per_class_accuracy[class_name] = 0.0

        return QuickTestResult(
            total_images=len(image_files),
            accuracy=accuracy,
            f1_score=f1_score,
            per_class_accuracy=per_class_accuracy,
            confusion_matrix=confusion_matrix,
            test_time=test_time,
            model_used=current_architecture
        )
    except HTTPException:
        raise
    except Exception as e:
        # Surface the error to the client for easier debugging
        raise HTTPException(status_code=500, detail=f"quicktest failed: {e}")


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    assert pipeline is not None
    try:
        image_bytes = await file.read()
        result = pipeline.predict_from_bytes(image_bytes)
        
        # Add metadata lookup
        metadata = _lookup_metadata(file.filename)
        if metadata:
            result["metadata"] = metadata
        
    except HTTPException:
        raise
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    return PredictionResponse(**result)


if __name__ == "__main__":
    # For local debugging: `python -m backend.main`
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        reload=os.environ.get("RELOAD", "false").lower() == "true",
    )


