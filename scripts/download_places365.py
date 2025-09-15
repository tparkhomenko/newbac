#!/usr/bin/env python3
import os
import sys
import csv
import json
import time
import shutil
import zipfile
import subprocess
from pathlib import Path
from typing import List, Tuple

from PIL import Image
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / 'data' / 'raw' / 'places365'
RESIZED_DIR = ROOT / 'data' / 'cleaned_resized' / 'places365_512'
METADATA_CSV = ROOT / 'data' / 'metadata' / 'metadata.csv'
SUMMARY_CSV = ROOT / 'data' / 'metadata' / 'places365_summary.csv'

KAGGLE_DATASET = 'benjaminkz/places365'


def _ensure_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    RESIZED_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / 'data' / 'metadata').mkdir(parents=True, exist_ok=True)


def _has_kaggle() -> bool:
    try:
        subprocess.run(['kaggle', '--help'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        return True
    except FileNotFoundError:
        return False


def download_places365() -> Path:
    """Download Places365 using Kaggle CLI if available; otherwise try kagglehub fallback."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # If images already exist, skip downloading
    existing = list(RAW_DIR.rglob('*.jpg'))
    if existing:
        print(f"Found {len(existing)} JPG files under {RAW_DIR}. Skipping download.")
        return RAW_DIR

    if _has_kaggle():
        print(f"Downloading Places365 dataset via Kaggle CLI into: {RAW_DIR}")
        tmp_dir = RAW_DIR / '_downloads'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            'kaggle', 'datasets', 'download', '-d', KAGGLE_DATASET,
            '-p', str(tmp_dir), '--force'
        ]
        print('> ' + ' '.join(cmd))
        subprocess.run(cmd, check=True)
        zips = sorted(tmp_dir.glob('*.zip'), key=lambda p: p.stat().st_mtime, reverse=True)
        if not zips:
            raise RuntimeError("No zip file downloaded from Kaggle.")
        zip_path = zips[0]
        print(f"Extracting {zip_path.name} -> {RAW_DIR}")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(RAW_DIR)
        print("Extraction complete.")
        return RAW_DIR

    # Fallback: kagglehub
    try:
        import kagglehub
        print("Kaggle CLI not found; using kagglehub fallback...")
        cache_path = kagglehub.dataset_download(KAGGLE_DATASET)
        print(f"kagglehub downloaded to: {cache_path}")
        cache_path = Path(cache_path)
        # Copy with progress bar
        src_files = [p for p in cache_path.rglob('*') if p.is_file()]
        print(f"Copying {len(src_files)} files to {RAW_DIR} ...")
        for src in tqdm(src_files, desc='Copying files'):
            rel = src.relative_to(cache_path)
            dest = RAW_DIR / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
        print(f"Copied kagglehub dataset to: {RAW_DIR}")
        return RAW_DIR
    except Exception as e:
        raise RuntimeError(f"Failed to download via Kaggle and kagglehub: {e}")


def _iter_image_files(root: Path) -> List[Path]:
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    files: List[Path] = []
    for p in root.rglob('*'):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return files


def _resize_to_512_jpg(src_path: Path, dst_path: Path):
    with Image.open(src_path) as im:
        im = im.convert('RGB')
        # Keep aspect ratio with padding to 512x512 similar to common ISIC preprocessing
        w, h = im.size
        scale = 512 / max(w, h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        im_resized = im.resize((new_w, new_h), Image.BICUBIC)
        canvas = Image.new('RGB', (512, 512), (0, 0, 0))
        offset = ((512 - new_w) // 2, (512 - new_h) // 2)
        canvas.paste(im_resized, offset)
        canvas.save(dst_path, format='JPEG', quality=95)


def preprocess_and_resize() -> List[Tuple[str, Path]]:
    print(f"Scanning images under: {RAW_DIR}")
    files = _iter_image_files(RAW_DIR)
    print(f"Found {len(files)} raw images")

    outputs: List[Tuple[str, Path]] = []
    for idx, src in enumerate(tqdm(sorted(files), desc='Resizing to 512x512')):
        # Create standardized filename: places365_<id>.jpg
        stem = src.stem
        standard_name = f"places365_{stem}.jpg"
        dst = RESIZED_DIR / standard_name
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            _resize_to_512_jpg(src, dst)
            outputs.append((standard_name, dst))
        except Exception as e:
            print(f"[WARN] Failed to process {src}: {e}")
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{len(files)} images...")

    print(f"Resized {len(outputs)} images to {RESIZED_DIR}")
    return outputs


def _read_metadata_dataframe() -> pd.DataFrame:
    if METADATA_CSV.exists():
        return pd.read_csv(METADATA_CSV)
    # Create minimal header consistent with existing schema if missing
    cols = ['image_name', 'csv_source', 'diagnosis_from_csv', 'unified_diagnosis', 'exp1', 'exp2', 'exp3', 'exp4', 'exp5', 'exp1_balanced', 'exp6', 'exp1_odin']
    return pd.DataFrame(columns=cols)


def append_to_metadata(image_names: List[str]):
    print(f"Appending {len(image_names)} Places365 rows to metadata: {METADATA_CSV}")
    df = _read_metadata_dataframe()

    # Ensure required columns exist
    required_cols = ['image_name', 'csv_source', 'diagnosis_from_csv', 'unified_diagnosis', 'exp1', 'exp2', 'exp3', 'exp4', 'exp5', 'exp6', 'exp1_balanced', 'exp1_odin']
    for col in required_cols:
        if col not in df.columns:
            df[col] = '-'

    # Build new rows
    new_rows = []
    for name in image_names:
        new_rows.append({
            'image_name': name.replace('.jpg', ''),
            'csv_source': 'places365',
            'diagnosis_from_csv': 'not_skin',
            'unified_diagnosis': 'NOT_SKIN',
            # Do NOT mark as exp1; use dedicated OOD split column
            'exp1': '-',
            'exp2': '-',
            'exp3': '-',
            'exp4': '-',
            'exp5': '-',
            'exp6': '-',
            'exp1_balanced': '-',
            'exp1_odin': 'test'
        })

    df_out = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df_out.to_csv(METADATA_CSV, index=False)
    print(f"Updated metadata rows: {len(df_out)}")

    # Summary CSV
    # Create simple summary with counts
    summary = pd.DataFrame({
        'csv_source': ['places365'],
        'unified_diagnosis': ['NOT_SKIN'],
        'total_added': [len(new_rows)]
    })
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"Wrote summary: {SUMMARY_CSV}")


def main():
    _ensure_dirs()

    # Step 1: Download & extract
    try:
        download_places365()
    except Exception as e:
        print(f"[WARN] Download step failed or skipped: {e}")
        print("Continuing with any existing files under data/raw/places365 ...")

    # Step 2: Resize to 512 JPGs
    outputs = preprocess_and_resize()
    image_names = [name for name, _ in outputs]

    # Step 3: Update metadata and summary
    append_to_metadata(image_names)

    print("\nAll done.")


if __name__ == '__main__':
    main()
