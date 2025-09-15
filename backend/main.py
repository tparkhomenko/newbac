import io
import json
import os
import csv
import glob
from typing import Any, Dict, List, Optional
from datetime import datetime

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
    current_model: str = Field(description="Currently active model choice (parallel_best|multi_best)")
    actual_model: str = Field(description="Actual model run name used for inference")
    available: List[str] = Field(description="Available model choices")
    summary: Dict[str, Any] = Field(description="Summary stats (skin/lesion/bm acc/f1) for the active model")


class ModelSwitchRequest(BaseModel):
    model_choice: Optional[str] = Field(default=None, description="Model choice: 'parallel_best' or 'multi_best'")
    architecture: Optional[str] = Field(default=None, description="Deprecated: accepts 'parallel' or 'multi' for backward compatibility")


class ModelStats(BaseModel):
    model_used: str
    skin: Dict[str, float]
    lesion: Dict[str, float]
    bm: Dict[str, float]


class QuickTestResult(BaseModel):
    total_images: int = Field(description="Total number of images tested")
    accuracy: float = Field(description="Overall accuracy")
    f1_score: float = Field(description="F1 score")
    per_class_accuracy: Dict[str, float] = Field(description="Accuracy per class")
    confusion_matrix: List[List[int]] = Field(description="Confusion matrix")
    test_time: float = Field(description="Total testing time in seconds")
    model_used: str = Field(description="Model used for testing")
    class_names: List[str] = Field(description="Names of classes aligned with confusion matrix columns/rows")


def _get_models_base_dir() -> str:
    return os.environ.get(
        "MODEL_DIR",
        os.path.join(os.path.dirname(__file__), "models"),
    )


def _get_model_dir_for_architecture(architecture: str) -> str:
    return os.path.join(_get_models_base_dir(), architecture)


# Best model choices mapping
BEST_MODEL_MAP: Dict[str, str] = {
    # choice -> relative path under models/
    "parallel_best": os.path.join("parallel", "exp1_withodin_train"),
    "multi_best": os.path.join("multi", "exp1_multi_8classes_lmf_balanced_mixup"),
}


def _get_model_dir_for_choice(choice: str) -> str:
    rel = BEST_MODEL_MAP.get(choice)
    if rel is None:
        # Fallback to base architecture dir if unknown
        base = "parallel" if choice.startswith("parallel") else "multi"
        return _get_model_dir_for_architecture(base)
    return os.path.join(_get_models_base_dir(), rel)


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


def _get_available_choices() -> List[str]:
    """Return available model choices based on folders present."""
    base_dir = _get_models_base_dir()
    available: List[str] = []
    for choice, rel in BEST_MODEL_MAP.items():
        if os.path.isdir(os.path.join(base_dir, rel)):
            available.append(choice)
    # If none exist yet, still expose both for UI
    return available or list(BEST_MODEL_MAP.keys())


def _build_stats(architecture: str) -> Dict[str, Any]:
    # Deprecated
    return {}


def _read_experiment_stats(run_id: str) -> Optional[Dict[str, Any]]:
    csv_path = os.path.join(os.path.dirname(__file__), "experiments_summary.csv")
    if not os.path.exists(csv_path):
        return None
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("Run ID") == run_id:
                    def _to_float(val: str) -> Optional[float]:
                        try:
                            return float(val) if val not in (None, "",) else None
                        except Exception:
                            return None
                    skin_acc = _to_float(row.get("Test Skin Acc", ""))
                    skin_f1 = _to_float(row.get("Test Skin F1", ""))
                    lesion_acc = _to_float(row.get("Test Lesion Acc", ""))
                    lesion_f1 = _to_float(row.get("Test Lesion F1", ""))
                    bm_acc = _to_float(row.get("Test BM Acc", ""))
                    bm_f1 = _to_float(row.get("Test BM F1", ""))
                    return {
                        "skin": {"acc": skin_acc or 0.0, "f1": skin_f1 or 0.0},
                        "lesion": {"acc": lesion_acc or 0.0, "f1": lesion_f1 or 0.0},
                        "bm": {"acc": bm_acc or 0.0, "f1": bm_f1 or 0.0},
                    }
    except Exception as e:
        print(f"Warning: failed to read experiment stats for {run_id}: {e}")
    return None


def _stats_for_choice(choice: str) -> Dict[str, Any]:
    run_id_map = {
        "parallel_best": "exp1_withodin_train",
        "multi_best": "exp1_multi_8classes_lmf_balanced_mixup",
    }
    run_id = run_id_map.get(choice, "")
    stats = _read_experiment_stats(run_id) or {
        "skin": {"acc": 0.0, "f1": 0.0},
        "lesion": {"acc": 0.0, "f1": 0.0},
        "bm": {"acc": 0.0, "f1": 0.0},
    }
    stats["actual_model"] = run_id
    return stats


def _ensure_logs_dir() -> str:
    logs_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    return logs_dir


def _log_model_selection(event: str, choice: str, stats: Dict[str, Any]) -> None:
    logs_dir = _ensure_logs_dir()
    log_path = os.path.join(logs_dir, "model_selection.log")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {event} -> {choice}: {json.dumps(stats)}\n")
    except Exception as e:
        print(f"Warning: failed to write model selection log: {e}")


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
current_model_choice: str = os.environ.get("MODEL_CHOICE", "parallel_best")
current_model_summary: Dict[str, Any] = {}


@app.on_event("startup")
def _startup() -> None:
    global pipeline, metadata_cache, current_model_summary
    model_dir = _get_model_dir_for_choice(current_model_choice)
    device = _get_device()
    pipeline = InferencePipeline(
        model_dir=model_dir,
        device=device,
    )
    current_model_summary = _stats_for_choice(current_model_choice)
    print(json.dumps({
        "model_used": current_model_choice,
        **current_model_summary,
    }))
    _log_model_selection("startup", current_model_choice, current_model_summary)
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
    available_choices = _get_available_choices()
    return ModelInfo(
        current_model=current_model_choice,
        actual_model=str(current_model_summary.get("actual_model", "")),
        available=available_choices,
        summary=current_model_summary,
    )


@app.post("/model/switch")
def switch_model(request: ModelSwitchRequest) -> Dict[str, Any]:
    """Switch to a different best model choice (parallel_best|multi_best).
    Backward compatible with 'architecture' field ('parallel'|'multi').
    """
    global current_model_choice, pipeline, current_model_summary

    choice = (request.model_choice or "").strip()
    if not choice and request.architecture:
        arch = request.architecture.lower().strip()
        if arch == "parallel":
            choice = "parallel_best"
        elif arch == "multi":
            choice = "multi_best"
    if choice not in {"parallel_best", "multi_best"}:
        raise HTTPException(status_code=400, detail="model_choice must be 'parallel_best' or 'multi_best'")

    model_dir = _get_model_dir_for_choice(choice)
    if not os.path.isdir(model_dir):
        print(f"Warning: Model directory not found for choice '{choice}': {model_dir}")

    try:
        current_model_choice = choice
        device = _get_device()
        pipeline = InferencePipeline(
            model_dir=model_dir,
            device=device,
        )
        current_model_summary = _stats_for_choice(current_model_choice)
        _log_model_selection("switch", current_model_choice, current_model_summary)
        return {
            "status": "success",
            "message": f"Switched to model: {choice}",
            "current_model": current_model_choice,
            "actual_model": str(current_model_summary.get("actual_model", "")),
            "summary": current_model_summary,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to switch model: {str(e)}")


@app.get("/model_stats", response_model=ModelStats)
def get_model_stats() -> ModelStats:
    return ModelStats(
        model_used=current_model_choice,
        skin={"acc": float(current_model_summary.get("skin", {}).get("acc", 0.0)), "f1": float(current_model_summary.get("skin", {}).get("f1", 0.0))},
        lesion={"acc": float(current_model_summary.get("lesion", {}).get("acc", 0.0)), "f1": float(current_model_summary.get("lesion", {}).get("f1", 0.0))},
        bm={"acc": float(current_model_summary.get("bm", {}).get("acc", 0.0)), "f1": float(current_model_summary.get("bm", {}).get("f1", 0.0))},
    )


@app.post("/quicktest")
def quick_test(max_images: int = 100) -> Dict[str, Any]:
    """Run quick test (skin, lesion, bm) on up to max_images in data/testing_only/."""
    import time
    from PIL import Image
    import random
    import shutil
    from collections import defaultdict
    try:
        from sklearn.metrics import accuracy_score as sk_accuracy, f1_score as sk_f1, confusion_matrix as sk_confusion
    except Exception:
        sk_accuracy = None  # type: ignore
        sk_f1 = None  # type: ignore
        sk_confusion = None  # type: ignore
    
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    test_dir = os.path.join(os.path.dirname(__file__), "..", "data", "testing_only")
    if not os.path.exists(test_dir):
        raise HTTPException(status_code=404, detail="Testing directory not found")
    
    try:
        # Load optional structured testing metadata
        testing_meta_path = os.path.join(test_dir, "metadata.csv")
        testing_meta: Dict[str, Dict[str, str]] = {}
        if os.path.exists(testing_meta_path):
            try:
                with open(testing_meta_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # keyed by filename without extension
                        filename = row.get("filename") or row.get("image_name") or row.get("image") or row.get("file")
                        if filename is None:
                            continue
                        testing_meta[os.path.splitext(filename)[0]] = row
            except Exception as e:
                print(f"Warning: Failed to load testing metadata.csv: {e}")
        
        # Get test images
        image_files: List[str] = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(glob.glob(os.path.join(test_dir, ext)))
        if not image_files:
            raise HTTPException(status_code=404, detail="No test images found")
        
        # Limit to max_images parameter (default 100, but can be up to 500+)
        if max_images > 0:
            image_files = image_files[:max_images]
        
        print(f"QuickTest: Evaluating {len(image_files)} images (max={max_images})")
        
        start_time = time.time()
        detailed_log_count = 0  # limit detailed per-image logs to first 5 images
        
        # Class spaces
        skin_classes = ["skin", "not_skin"]
        bm_classes = ["benign", "malignant"]
        
        class_names = [
            "melanoma",
            "nevus",
            "basal_cell_carcinoma",
            "squamous_cell_carcinoma",
            "seborrheic_keratosis",
            "actinic_keratosis",
            "dermatofibroma",
            "vascular",
        ]
        
        # Helpers for label mapping
        abbr_to_full = {
            'NV': 'nevus',
            'MEL': 'melanoma',
            'BCC': 'basal_cell_carcinoma',
            'SCC': 'squamous_cell_carcinoma',
            'BKL': 'seborrheic_keratosis',
            'AKIEC': 'actinic_keratosis',
            'DF': 'dermatofibroma',
            'VASC': 'vascular',
        }
        full_to_abbr = {v: k for k, v in abbr_to_full.items()}
        malignant_full = {"melanoma", "basal_cell_carcinoma", "squamous_cell_carcinoma"}
        benign_full = {"nevus", "seborrheic_keratosis", "actinic_keratosis", "dermatofibroma", "vascular"}
        
        def map_unified_to_triplet(unified: str) -> Dict[str, str]:
            u = (unified or '').strip().lower()
            if u == 'not_skin':
                return {"skin_label": "not_skin", "lesion_label": "-", "bm_label": "-"}
            if u == 'unknown' or u == '':
                return {"skin_label": "skin", "lesion_label": "-", "bm_label": "-"}
            # known 8-class
            if u in malignant_full:
                return {"skin_label": "skin", "lesion_label": full_to_abbr[u], "bm_label": "malignant"}
            if u in benign_full:
                return {"skin_label": "skin", "lesion_label": full_to_abbr[u], "bm_label": "benign"}
            # default
            return {"skin_label": "skin", "lesion_label": "-", "bm_label": "-"}
        
        # Accumulators for sklearn metrics
        y_true_skin: List[int] = []
        y_pred_skin: List[int] = []
        y_true_lesion: List[int] = []
        y_pred_lesion: List[int] = []
        y_true_bm: List[int] = []
        y_pred_bm: List[int] = []
        
        # Collect sample logs
        sample_logs: List[Dict[str, Any]] = []
        
        for image_path in image_files:
            try:
                with open(image_path, 'rb') as f:
                    image_bytes = f.read()
                # Get prediction
                result = pipeline.predict_from_bytes(image_bytes)
                # Ground truth lookup
                filename = os.path.basename(image_path)
                meta_row = testing_meta.get(os.path.splitext(filename)[0]) if testing_meta else None
                metadata = _lookup_metadata(filename)
                
                # Extra diagnostics: show raw logits shapes for each head and which slice is used
                try:
                    # Access internal helpers for diagnostics
                    features_diag = pipeline._generate_sam_features(image_bytes)  # type: ignore[attr-defined]
                    if pipeline.mlp2 is not None:
                        skin_logits_diag, lesion_logits_diag, bm_logits_diag = pipeline._forward_multihead(pipeline.mlp2, features_diag)  # type: ignore[attr-defined]
                        if detailed_log_count < 5:
                            print(f"QuickTest Diagnostics: {filename}")
                            print(f"  Raw logits shapes -> skin: {tuple(skin_logits_diag.shape)}, lesion: {tuple(lesion_logits_diag.shape)}, bm: {tuple(bm_logits_diag.shape)}")
                            print(f"  Lesion head slice used: outputs[1] -> shape {tuple(lesion_logits_diag.shape)}")
                    else:
                        if detailed_log_count < 5:
                            print(f"QuickTest Diagnostics: {filename}")
                            print("  mlp2 not loaded; using fallback path (no raw logits available)")
                except Exception as diag_err:
                    if detailed_log_count < 5:
                        print(f"Diagnostics error for {filename}: {diag_err}")
                
                # Prepare GT labels (structured testing meta overrides unified)
                unified_gt = (metadata.get('unified_diagnosis') or '').strip() if metadata else ''
                skin_gt = lesion_gt_abbr = bm_gt = None
                if meta_row:
                    skin_gt = (meta_row.get('skin_label') or '').strip() or None
                    lesion_gt_abbr = (meta_row.get('lesion_label') or '').strip() or None
                    bm_gt = (meta_row.get('bm_label') or '').strip() or None
                    unified_gt = (meta_row.get('unified_label') or unified_gt).strip()
                # Backward compatibility mapping if any missing
                if skin_gt is None or lesion_gt_abbr is None or bm_gt is None:
                    mapped = map_unified_to_triplet(unified_gt)
                    skin_gt = skin_gt or mapped['skin_label']
                    lesion_gt_abbr = lesion_gt_abbr or mapped['lesion_label']
                    bm_gt = bm_gt or mapped['bm_label']
                
                # Predictions per head
                # Skin
                if result.get('is_skin'):
                    pred_skin_label = result['is_skin'].get('label')
                    if skin_gt in skin_classes and pred_skin_label in skin_classes:
                        y_true_skin.append(skin_classes.index(skin_gt))
                        y_pred_skin.append(skin_classes.index(pred_skin_label))
                
                # Lesion (skip if no GT)
                if result.get('lesion_type') and lesion_gt_abbr and lesion_gt_abbr != '-':
                    predicted_idx = int(result['lesion_type']['label_index'])
                    predicted_label = result['lesion_type']['labels'][predicted_idx]
                    ground_truth = abbr_to_full.get(lesion_gt_abbr.upper())
                    # Log details for every image
                    try:
                        conf_val = float(result['lesion_type']['probabilities'][predicted_idx])
                        # Top-3 for debugging (first 5 images only)
                        if detailed_log_count < 5:
                            probs_arr = result['lesion_type']['probabilities']
                            probs_arr_rounded = [round(float(p), 3) for p in probs_arr]
                            top3 = sorted(list(enumerate(probs_arr)), key=lambda x: x[1], reverse=True)[:3]
                            top3_fmt = [f"(idx={i}, label=\"{result['lesion_type']['labels'][i]}\", p={p:.4f})" for i, p in top3]
                            print(f"QuickTest Diagnostics: {filename}")
                            print(f"  Lesion probs (softmax, rounded): {probs_arr_rounded}")
                            print(f"  Top-3 lesion probabilities: {', '.join(top3_fmt)}")
                    except Exception:
                        conf_val = 0.0
                    if detailed_log_count < 5:
                        print(f"QuickTest Sample: {filename}")
                        print(f"  Skin → pred={result.get('is_skin',{}).get('label')} gt={skin_gt}")
                        print(f"  Lesion → pred_idx={predicted_idx}, pred_label=\"{predicted_label}\", gt={ground_truth}, conf={conf_val:.4f}")
                    
                    # For first 5 images, also print the raw lesion logits and derived prediction from logits
                    try:
                        if detailed_log_count < 5 and pipeline.mlp2 is not None:
                            _, lesion_logits_diag2, _ = pipeline._forward_multihead(pipeline.mlp2, features_diag)  # type: ignore[attr-defined]
                            logits_list = [round(float(x), 4) for x in lesion_logits_diag2[0].tolist()]
                            pred_from_logits = int(lesion_logits_diag2.argmax(dim=1).item())
                            mapped_label_from_logits = predicted_label if pred_from_logits == predicted_idx else (
                                result['lesion_type']['labels'][pred_from_logits] if pred_from_logits < len(result['lesion_type']['labels']) else 'out_of_range')
                            print(f"  Lesion logits (rounded): {logits_list}")
                            print(f"  Lesion logits argmax → Index: {pred_from_logits}, Label: \"{mapped_label_from_logits}\"")
                            detailed_log_count += 1
                    except Exception as e_log:
                        if detailed_log_count < 5:
                            print(f"  (logits print error) {e_log}")
                    
                    try:
                        if ground_truth in class_names:
                            y_true_lesion.append(class_names.index(ground_truth))
                            y_pred_lesion.append(predicted_idx)
                            if len(sample_logs) < 5:
                                sample_logs.append({
                                    'file': filename,
                                    'skin_gt': skin_gt,
                                    'lesion_gt': ground_truth,
                                    'bm_gt': bm_gt,
                                    'lesion_pred_idx': predicted_idx,
                                    'lesion_pred_label': predicted_label,
                                })
                    except Exception as _:
                        pass
                
                # BM (skip if no GT)
                if result.get('malignancy') and bm_gt and bm_gt != '-':
                    pred_bm_label = result['malignancy'].get('label')
                    if pred_bm_label in bm_classes and bm_gt in bm_classes:
                        y_true_bm.append(bm_classes.index(bm_gt))
                        y_pred_bm.append(bm_classes.index(pred_bm_label))
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        end_time = time.time()
        test_time = end_time - start_time
        
        # Compute metrics using sklearn when available
        def compute_task_metrics(y_true: List[int], y_pred: List[int], labels: List[str]) -> Dict[str, Any]:
            if not y_true or not y_pred or len(y_true) == 0:
                return {
                    "num_samples": 0,
                    "accuracy": 0.0,
                    "f1_macro": 0.0,
                    "confusion_matrix": [[0 for _ in labels] for _ in labels],
                    "class_names": labels,
                }
            if sk_accuracy is None or sk_f1 is None or sk_confusion is None:
                # Fallback simple metrics
                correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
                total = len(y_true)
                return {
                    "num_samples": total,
                    "accuracy": (correct / total) if total else 0.0,
                    "f1_macro": 0.0,
                    "confusion_matrix": [[0 for _ in labels] for _ in labels],
                    "class_names": labels,
                }
            acc = float(sk_accuracy(y_true, y_pred))
            f1m = float(sk_f1(y_true, y_pred, average='macro'))
            cm = sk_confusion(y_true, y_pred, labels=list(range(len(labels))))
            return {
                "num_samples": len(y_true),
                "accuracy": acc,
                "f1_macro": f1m,
                "confusion_matrix": cm.tolist(),
                "class_names": labels,
            }
        
        skin_metrics = compute_task_metrics(y_true_skin, y_pred_skin, skin_classes)
        lesion_metrics = compute_task_metrics(y_true_lesion, y_pred_lesion, class_names)
        bm_metrics = compute_task_metrics(y_true_bm, y_pred_bm, bm_classes)
        
        # Print a small verification block for debug
        if sample_logs:
            print("QuickTest sample (first up to 5):")
            for s in sample_logs:
                print(f"  {s['file']}: skin_gt={s.get('skin_gt')} lesion_gt={s.get('lesion_gt')} bm_gt={s.get('bm_gt')} | lesion_pred[{s.get('lesion_pred_idx')}]={s.get('lesion_pred_label')}")
        
        # Optional: log to wandb
        try:
            import wandb  # type: ignore
            run = wandb.init(project=os.environ.get("WANDB_PROJECT", "skin_lesion_classification"), name=f"quicktest_{int(time.time())}", reinit=True)
            wandb.summary.update({
                "quicktest/total_images": len(image_files),
                "quicktest/skin/accuracy": skin_metrics["accuracy"],
                "quicktest/skin/f1_macro": skin_metrics["f1_macro"],
                "quicktest/lesion/accuracy": lesion_metrics["accuracy"],
                "quicktest/lesion/f1_macro": lesion_metrics["f1_macro"],
                "quicktest/bm/accuracy": bm_metrics["accuracy"],
                "quicktest/bm/f1_macro": bm_metrics["f1_macro"],
            })
            wandb.finish()
        except Exception as _:
            pass
        
        return {
            "total_images": len(image_files),
            "model_used": current_model_choice,
            "skin": skin_metrics,
            "lesion": lesion_metrics,
            "bm": bm_metrics,
            "test_time": test_time,
        }
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


