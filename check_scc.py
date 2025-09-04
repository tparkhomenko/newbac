#!/usr/bin/env python3
"""
Check SCC distribution in the balanced dataset.
"""

import pandas as pd

def main():
    print("=== Checking SCC distribution ===")
    
    # Load metadata
    df = pd.read_csv("data/metadata/metadata.csv")
    
    # Check SCC in balanced dataset
    balanced_all = df[df["exp1_balanced"].astype(str) != ""]
    print(f"SCC in balanced dataset: {len(balanced_all[balanced_all['unified_diagnosis'] == 'SCC'])}")
    
    # Check SCC in each split
    for split in ['train', 'val', 'test']:
        split_data = balanced_all[balanced_all["exp1_balanced"] == split]
        scc_count = len(split_data[split_data['unified_diagnosis'] == 'SCC'])
        print(f"SCC in {split}: {scc_count}")

if __name__ == "__main__":
    main()








