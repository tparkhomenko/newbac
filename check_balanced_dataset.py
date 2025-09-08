#!/usr/bin/env python3
"""
Check the balanced dataset class distribution.
"""

import pandas as pd

def main():
    print("=== Checking balanced dataset class distribution ===")
    
    # Load metadata
    df = pd.read_csv("data/metadata/metadata.csv")
    
    # Check balanced train data
    balanced_train = df[df["exp1_balanced"] == "train"]
    print(f"Balanced train samples: {len(balanced_train)}")
    
    print("\nLesion distribution in balanced train:")
    print(balanced_train["unified_diagnosis"].value_counts().sort_index())
    
    print("\nExpected 8 classes: MEL, NV, BCC, SCC, BKL, AKIEC, DF, VASC")
    for cls in ["MEL", "NV", "BCC", "SCC", "BKL", "AKIEC", "DF", "VASC"]:
        count = len(balanced_train[balanced_train["unified_diagnosis"] == cls])
        print(f"{cls}: {count}")
    
    # Check if any classes have 0 samples
    zero_classes = []
    for cls in ["MEL", "NV", "BCC", "SCC", "BKL", "AKIEC", "DF", "VASC"]:
        count = len(balanced_train[balanced_train["unified_diagnosis"] == cls])
        if count == 0:
            zero_classes.append(cls)
    
    if zero_classes:
        print(f"\n⚠️  Classes with 0 samples: {zero_classes}")
    else:
        print("\n✅ All expected classes have samples")

if __name__ == "__main__":
    main()










