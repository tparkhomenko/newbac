#!/usr/bin/env python3
import pandas as pd
import pickle
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
META = ROOT / 'data' / 'metadata' / 'metadata.csv'
OUT_META = ROOT / 'data' / 'metadata' / 'exp1_odin_eval.csv'
PKL_ISIC = ROOT / 'data' / 'processed' / 'features' / 'sam_features_exp1_test.pkl'
PKL_PLCS = ROOT / 'data' / 'processed' / 'features' / 'sam_features_places365_test.pkl'
OUT_PKL = ROOT / 'data' / 'processed' / 'features' / 'sam_features_exp1_odin_eval.pkl'


def main() -> None:
    df = pd.read_csv(META)

    # Select ISIC in-distribution (exp1=test, skin=1)
    # Derive skin=1 from unified_diagnosis != NOT_SKIN
    skin_isic = (df.get('exp1', '').astype(str) == 'test') & (df.get('unified_diagnosis', '').astype(str) != 'NOT_SKIN')
    isic = df[skin_isic].copy()

    # Select Places365 OOD (exp1_odin=test if present, otherwise csv_source=places365)
    places_mask = (df.get('csv_source', '').astype(str) == 'places365')
    if 'exp1_odin' in df.columns:
        places_mask = places_mask & (df.get('exp1_odin', '').astype(str) == 'test') | places_mask
    places = df[places_mask].copy()

    # Sample 1000 each
    isic_sample = isic.sample(n=min(1000, len(isic)), random_state=42) if len(isic) > 0 else isic
    places_sample = places.sample(n=min(1000, len(places)), random_state=42) if len(places) > 0 else places

    combined = pd.concat([isic_sample.assign(odin_label=1), places_sample.assign(odin_label=0)], ignore_index=True)
    combined.to_csv(OUT_META, index=False)
    print(f"Saved eval metadata to {OUT_META} ({len(combined)} rows)")

    # Load feature stores and build combined store
    store_isic = {}
    store_places = {}
    if PKL_ISIC.exists():
        with open(PKL_ISIC, 'rb') as f:
            store_isic = pickle.load(f)
    else:
        print(f"[WARN] Missing ISIC features: {PKL_ISIC}")
    if PKL_PLCS.exists():
        with open(PKL_PLCS, 'rb') as f:
            store_places = pickle.load(f)
    else:
        print(f"[WARN] Missing Places365 features: {PKL_PLCS}")

    combined_store = {}
    missing = 0
    for _, row in combined.iterrows():
        name = str(row.get('image_name') or row.get('image'))
        if name in store_isic:
            combined_store[name] = store_isic[name]
        elif name in store_places:
            combined_store[name] = store_places[name]
        else:
            missing += 1
    with open(OUT_PKL, 'wb') as f:
        pickle.dump({'features': combined_store, 'labels': {name: 1 if name in store_isic else 0 for name in combined_store.keys()}}, f)
    print(f"Saved combined features to {OUT_PKL} ({len(combined_store)} items, missing features for {missing})")


if __name__ == '__main__':
    main()
