#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def main() -> None:
    meta_path = Path('data/metadata/metadata.csv')
    df = pd.read_csv(meta_path)

    # Work on non-Places rows as ISIC family (covers ISIC_20xx, ISIC2018, plausibility_check_512, etc.)
    is_isic = (df.get('csv_source', '').astype(str) != 'places365')
    isic_df = df[is_isic].copy()
    if isic_df.empty:
        print('No ISIC-like rows found; nothing to split.')
        return

    # Labels for stratification: use unified_diagnosis (including NOT_SKIN)
    label_col = 'unified_diagnosis'
    labels = isic_df[label_col].astype(str).fillna('')

    # First split: test 15%
    train_val_df, test_df = train_test_split(
        isic_df, test_size=0.15, stratify=labels, random_state=42
    )

    # Second split: val from remaining (15% of total -> 15/85 of remaining)
    labels_train_val = train_val_df[label_col].astype(str).fillna('')
    val_rel = 0.15 / 0.85
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_rel, stratify=labels_train_val, random_state=42
    )

    # Assign exp1 values
    df['exp1'] = df.get('exp1', '').astype(str)
    df.loc[train_df.index, 'exp1'] = 'train'
    df.loc[val_df.index, 'exp1'] = 'val'
    df.loc[test_df.index, 'exp1'] = 'test'

    # Ensure no Places365 rows have exp1 set
    is_places = (df.get('csv_source', '').astype(str) == 'places365')
    df.loc[is_places, 'exp1'] = ''

    df.to_csv(meta_path, index=False)

    print(f"ISIC exp1 splits restored: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")


if __name__ == '__main__':
    main()


