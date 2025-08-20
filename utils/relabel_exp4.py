import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).parent.parent
CSV_PATH = PROJECT_ROOT / 'data' / 'metadata' / 'metadata.csv'

CLASS_NAMES = ['UNKNOWN','NV','MEL','BCC','BKL','AKIEC','NOT_SKIN','SCC','VASC','DF']
TOTAL = 3782
TRAIN_N = 2554
VAL_N = 553
TEST_N = 675


def main():
    df = pd.read_csv(CSV_PATH)
    assert 'unified_diagnosis' in df.columns, 'metadata.csv missing unified_diagnosis'

    # Start clean for exp4
    df['exp4'] = ''

    # Keep only rows with valid labels
    df['label_up'] = df['unified_diagnosis'].astype(str).str.upper()
    valid_mask = df['label_up'].isin(CLASS_NAMES)
    pool = df[valid_mask].copy()

    # Global class counts and inverse-frequency weights to favor minorities
    global_counts = pool['label_up'].value_counts().reindex(CLASS_NAMES, fill_value=0)
    inv_freq = 1.0 / np.clip(global_counts.values.astype(float), a_min=1.0, a_max=None)
    class_probs = inv_freq / inv_freq.sum()
    class_to_prob = {c: p for c, p in zip(CLASS_NAMES, class_probs)}

    # Ensure minimum per-class presence
    min_per_class = {c: min(global_counts[c], 20) for c in CLASS_NAMES}
    selected_idx = []
    used = set()
    for c in CLASS_NAMES:
        c_rows = pool[pool['label_up'] == c]
        if len(c_rows) == 0:
            continue
        take = min(len(c_rows), min_per_class[c])
        sel = c_rows.sample(n=take, random_state=42, replace=False).index.tolist()
        selected_idx.extend(sel)
        used.update(sel)

    remaining = TOTAL - len(selected_idx)
    if remaining < 0:
        # If pool too small, trim to TOTAL deterministically
        selected_idx = selected_idx[:TOTAL]
        remaining = 0

    # Weighted sampling on remaining pool
    if remaining > 0:
        rem_pool = pool[~pool.index.isin(used)].copy()
        if not rem_pool.empty:
            weights = rem_pool['label_up'].map(class_to_prob).values.astype(float)
            weights = weights / weights.sum()
            add_idx = np.random.choice(rem_pool.index.values, size=min(remaining, len(rem_pool)), replace=False, p=weights)
            selected_idx.extend(add_idx.tolist())

    # Finalize selection
    selected = pool.loc[selected_idx].copy()
    # Stratified splits: train (2554) / temp, then val (553) / test (675)
    labels = selected['label_up']
    train_sel, temp_sel = train_test_split(selected, train_size=min(TRAIN_N, len(selected)), stratify=labels, random_state=42)
    temp_labels = temp_sel['label_up']
    val_size = min(VAL_N, len(temp_sel))
    test_size = min(TEST_N, max(0, len(temp_sel) - val_size))
    if val_size + test_size > len(temp_sel):
        val_size = int(round(0.45 * len(temp_sel)))
        test_size = len(temp_sel) - val_size
    val_sel, test_sel = train_test_split(temp_sel, train_size=val_size, stratify=temp_labels, random_state=42)

    # Apply back to df
    df.loc[train_sel.index, 'exp4'] = 'train'
    df.loc[val_sel.index, 'exp4'] = 'val'
    df.loc[test_sel.index, 'exp4'] = 'test'

    # Save backup and updated CSV
    backup = CSV_PATH.with_suffix('.csv.bak')
    CSV_PATH.replace(backup)
    df.drop(columns=['label_up']).to_csv(CSV_PATH, index=False)

    # Print summary
    print('exp4 value_counts:')
    print(df['exp4'].value_counts(dropna=False))
    for split in ['train','val','test']:
        subset = df[df['exp4'] == split]
        counts = subset['unified_diagnosis'].str.upper().value_counts().reindex(CLASS_NAMES, fill_value=0)
        print(f"\n{split} ({len(subset)})")
        print(counts)


if __name__ == '__main__':
    main()


