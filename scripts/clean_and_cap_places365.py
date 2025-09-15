#!/usr/bin/env python3
import pandas as pd
import pickle
from pathlib import Path


def main() -> None:
    meta_path = Path("data/metadata/metadata.csv")
    df = pd.read_csv(meta_path)

    # 1) Replace '-' with '' (empty string) across the entire DataFrame
    before_dash = int((df == "-").sum().sum())
    df = df.replace("-", "")
    after_dash = int((df == "-").sum().sum())
    print(f"Replaced '-' occurrences: {before_dash} -> {after_dash}")

    df.to_csv(meta_path, index=False)

    # Reload to ensure clean state
    df = pd.read_csv(meta_path)

    # 2) Cap Places365 to 1000 rows
    is_places = (df.get('csv_source', '') == 'places365')
    places_df = df[is_places].copy()
    num_places = len(places_df)
    keep_n = 1000 if num_places >= 1000 else num_places
    sampled = places_df.sample(n=keep_n, random_state=42) if keep_n > 0 else places_df

    # Ensure exp1_odin column exists for summary (keep data as-is otherwise)
    if 'exp1_odin' not in sampled.columns:
        sampled['exp1_odin'] = ''

    # Keep non-places + sampled places
    df_capped = pd.concat([df[~is_places], sampled], ignore_index=True)
    df_capped.to_csv(meta_path, index=False)

    # Write places365 summary
    if keep_n > 0:
        summary = sampled.groupby(['exp1', 'exp1_odin', 'unified_diagnosis']).size().reset_index(name='count')
    else:
        summary = pd.DataFrame(columns=['exp1', 'exp1_odin', 'unified_diagnosis', 'count'])
    summary_path = Path('data/metadata/places365_summary.csv')
    summary.to_csv(summary_path, index=False)

    # 3) Cross-check counts
    df2 = pd.read_csv(meta_path)
    # ISIC split counts
    is_isic = (df2.get('csv_source', '') == 'ISIC')
    exp1_col = df2.get('exp1', '')

    isic_train = int(((is_isic) & (exp1_col == 'train')).sum())
    isic_val = int(((is_isic) & (exp1_col == 'val')).sum())
    isic_test = int(((is_isic) & (exp1_col == 'test')).sum())

    places_after = int((df2.get('csv_source', '') == 'places365').sum())

    print(f"ISIC counts -> train: {isic_train}, val: {isic_val}, test: {isic_test}")
    print(f"Places365 capped rows in metadata: {places_after} (expected {keep_n})")
    print(f"places365_summary.csv total rows counted: {int(summary['count'].sum()) if not summary.empty else 0}")

    # Feature availability check for sampled Places365
    features_path_inc = Path('data/processed/features/sam_features_all_incremental.pkl')
    features_path_all = Path('data/processed/features/sam_features_all.pkl')
    store_path = features_path_inc if features_path_inc.exists() else features_path_all
    if keep_n > 0 and store_path.exists():
        with open(store_path, 'rb') as f:
            store = pickle.load(f)
        keys = set(store.keys())
        sampled_names = set(sampled['image_name'].astype(str))
        avail = sum(1 for n in sampled_names if n in keys)
        print(f"Feature store: {store_path}")
        print(f"Places365 features present for kept rows: {avail}/{keep_n}")
    else:
        print("Feature store not found or no Places365 rows kept; skipping feature check.")


if __name__ == '__main__':
    main()


