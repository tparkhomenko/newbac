#!/usr/bin/env python3
import os
import pickle
from pathlib import Path
from typing import List, Dict

import pandas as pd
import torch
from tqdm import tqdm

# Project paths
ROOT = Path(__file__).resolve().parents[1]
CLEAN_DIR = ROOT / 'data' / 'cleaned_resized' / 'places365_512'
METADATA_CSV = ROOT / 'data' / 'metadata' / 'metadata.csv'
OUT_DIR = ROOT / 'data' / 'processed' / 'features'
OUT_PKL = OUT_DIR / 'sam_features_places365_test.pkl'
OUT_META = OUT_DIR / 'sam_features_places365_test_metadata.pkl'

# SAM extractor
from sam.sam_encoder import SAMFeatureExtractor


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not METADATA_CSV.exists():
        raise FileNotFoundError(f"Missing metadata: {METADATA_CSV}")

    import argparse
    parser = argparse.ArgumentParser(description='Extract SAM features for Places365 OOD set')
    parser.add_argument('--limit', type=int, default=None, help='Randomly sample this many images from Places365 rows in metadata')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling')
    args = parser.parse_args()

    df = pd.read_csv(METADATA_CSV)

    # Filter Places365 rows (independent of exp1 flag)
    mask = (df.get('csv_source', '').astype(str) == 'places365')
    df_places = df[mask].copy()
    if df_places.empty:
        print("No Places365 rows found in metadata; nothing to extract.")
        # Still write empty stores for consistency
        with open(OUT_PKL, 'wb') as f:
            pickle.dump({}, f)
        with open(OUT_META, 'wb') as f:
            pickle.dump({'feature_dim': 256, 'num_images': 0, 'image_names': []}, f)
        return

    # Optional random sampling
    if args.limit is not None and args.limit > 0 and len(df_places) > args.limit:
        df_places = df_places.sample(n=args.limit, random_state=args.seed).reset_index(drop=True)

    # Resolve paths
    image_paths: List[str] = []
    image_names: List[str] = []
    for _, row in df_places.iterrows():
        name = row['image_name'] if 'image_name' in row else row.get('image')
        if pd.isna(name):
            continue
        # filenames in cleaned set are <name>.jpg
        img_path = CLEAN_DIR / f"{name}.jpg"
        if img_path.exists():
            image_paths.append(str(img_path))
            image_names.append(str(name))

    if not image_paths:
        print("No image files found under cleaned_resized/places365_512 for metadata rows.")
        with open(OUT_PKL, 'wb') as f:
            pickle.dump({}, f)
        with open(OUT_META, 'wb') as f:
            pickle.dump({'feature_dim': 256, 'num_images': 0, 'image_names': []}, f)
        return

    device = 'cpu'
    extractor = SAMFeatureExtractor(model_type='vit_b', device=device)

    store: Dict[str, torch.Tensor] = {}
    batch = 4
    for i in tqdm(range(0, len(image_paths), batch), desc='Extracting Places365 features'):
        paths = image_paths[i:i+batch]
        names = image_names[i:i+batch]
        feats = extractor.extract_features(paths)  # [B, 256]
        for j, n in enumerate(names):
            store[n] = feats[j].detach().cpu().numpy()

    with open(OUT_PKL, 'wb') as f:
        pickle.dump(store, f)
    with open(OUT_META, 'wb') as f:
        pickle.dump({'feature_dim': 256, 'num_images': len(store), 'image_names': list(store.keys())}, f)

    print(f"Saved Places365 features: {OUT_PKL} ({len(store)} items)")


if __name__ == '__main__':
    main()
