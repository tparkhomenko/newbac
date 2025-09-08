#!/usr/bin/env python3
"""
Create balanced Exp1 dataset with a simple approach.
Ensures each class has samples and uses basic splitting.
"""

import pandas as pd
import numpy as np

def main():
    print("=== Creating balanced Exp1 dataset (simple approach) ===")
    
    # Load metadata
    df = pd.read_csv("data/metadata/metadata.csv")
    
    print(f"Original metadata: {len(df)} total images")
    
    # Check exp1 distribution
    exp1_data = df[df['exp1'].astype(str) != '']
    print(f"\nExp1 data: {len(exp1_data)} samples")
    
    # Check lesion distribution in exp1
    if 'unified_diagnosis' in df.columns:
        print(f"Lesion distribution in exp1:")
        lesion_counts = exp1_data['unified_diagnosis'].value_counts().sort_index()
        print(lesion_counts)
    
    print(f"\nCreating balanced version...")
    
    # Create balanced dataset by sampling from exp1
    balanced_data = []
    target_samples = 2000
    min_samples_per_class = 100  # Ensure each class has at least 100 samples
    
    # Get exp1 all data for sampling
    exp1_all = df[df['exp1'].astype(str) != '']
    
    print(f"Exp1 all data: {len(exp1_all)} samples")
    
    # Sample from each lesion class (excluding NOT_SKIN)
    if 'unified_diagnosis' in df.columns:
        # Exclude NOT_SKIN since it's not used for the lesion task
        lesion_classes = [cls for cls in exp1_all['unified_diagnosis'].unique() if cls != 'NOT_SKIN' and not pd.isna(cls)]
        
        for lesion_type in lesion_classes:
            class_data = exp1_all[exp1_all['unified_diagnosis'] == lesion_type]
            print(f"Class {lesion_type}: {len(class_data)} samples")
            
            if len(class_data) > target_samples:
                # Cap majority classes
                sampled = class_data.sample(n=target_samples, random_state=42)
                balanced_data.append(sampled)
                print(f"  -> Sampled {len(sampled)} samples")
            else:
                # For minority classes, ensure minimum samples
                if len(class_data) < min_samples_per_class:
                    # If too few samples, skip this class for now
                    print(f"  -> Skipping {lesion_type} (too few samples: {len(class_data)})")
                    continue
                else:
                    # Keep minority classes as is
                    balanced_data.append(class_data)
                    print(f"  -> Kept all {len(class_data)} samples")
    
    if balanced_data:
        balanced_df = pd.concat(balanced_data, ignore_index=True)
        
        print(f"\nBalanced dataset created: {len(balanced_df)} samples")
        print(f"Balanced lesion distribution:")
        print(balanced_df['unified_diagnosis'].value_counts().sort_index())
        
        # Create simple splits (70/15/15) without stratification
        print(f"\nCreating simple splits...")
        
        # Shuffle the data
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Calculate split points
        total_samples = len(balanced_df)
        train_end = int(total_samples * 0.7)
        val_end = int(total_samples * 0.85)
        
        # Split the data
        train_data = balanced_df.iloc[:train_end]
        val_data = balanced_df.iloc[train_end:val_end]
        test_data = balanced_df.iloc[val_end:]
        
        print(f"Split sizes: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        
        # Add balanced column to original dataframe
        df['exp1_balanced'] = ''
        df.loc[train_data.index, 'exp1_balanced'] = 'train'
        df.loc[val_data.index, 'exp1_balanced'] = 'val'
        df.loc[test_data.index, 'exp1_balanced'] = 'test'
        
        # Save updated metadata
        df.to_csv("data/metadata/metadata.csv", index=False)
        print(f"\nUpdated metadata.csv with exp1_balanced column")
        
        # Also save just the balanced data
        balanced_df.to_csv("data/metadata/metadata_exp1_balanced.csv", index=False)
        print(f"Saved balanced data to: data/metadata/metadata_exp1_balanced.csv")
        
        # Verify the balanced dataset
        print(f"\nVerification:")
        print(f"Original exp1 total: {len(exp1_data)}")
        print(f"Balanced total: {len(balanced_df)}")
        print(f"Balanced classes: {balanced_df['unified_diagnosis'].nunique()}")
        
        # Check the new column
        exp1_balanced_data = df[df['exp1_balanced'].astype(str) != '']
        print(f"Exp1_balanced data: {len(exp1_balanced_data)} samples")
        
        # Check per-split distribution
        for split in ['train', 'val', 'test']:
            split_data = df[df['exp1_balanced'] == split]
            print(f"\n{split.capitalize()} split ({len(split_data)} samples):")
            print(split_data['unified_diagnosis'].value_counts().sort_index())
            
            # Check if all expected classes are present
            expected_classes = ['MEL', 'NV', 'BCC', 'SCC', 'BKL', 'AKIEC', 'DF', 'VASC']
            missing_classes = []
            for cls in expected_classes:
                if len(split_data[split_data['unified_diagnosis'] == cls]) == 0:
                    missing_classes.append(cls)
            
            if missing_classes:
                print(f"  ⚠️  Missing classes: {missing_classes}")
            else:
                print(f"  ✅ All expected classes present")
    else:
        print("No balanced data created - check if unified_diagnosis column exists")

if __name__ == "__main__":
    main()










