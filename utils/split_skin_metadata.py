import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

def main():
    meta_path = Path('datasets/metadata/metadata_skin_not_skin.csv')
    out_path = Path('datasets/metadata/metadata_skin_not_skin_split.csv')
    df = pd.read_csv(meta_path)

    # Stratify by 'skin' (and optionally by lesion_group for skin=1)
    train_val, test = train_test_split(
        df, test_size=0.15, random_state=42, stratify=df['skin'])
    train, val = train_test_split(
        train_val, test_size=0.1765, random_state=42, stratify=train_val['skin'])  # 0.1765*0.85 ≈ 0.15

    train['split'] = 'train'
    val['split'] = 'val'
    test['split'] = 'test'

    df_split = pd.concat([train, val, test], ignore_index=True)
    df_split.to_csv(out_path, index=False)
    print(f"Saved split metadata to {out_path}")
    print(df_split['split'].value_counts())

if __name__ == '__main__':
    main() 