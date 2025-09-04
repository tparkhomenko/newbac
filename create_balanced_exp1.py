#!/usr/bin/env python3
"""
Create balanced Exp1 dataset by oversampling minority classes and capping majority classes.
Excludes NOT_SKIN samples since they're not used for the lesion task.
Uses manual stratification to ensure each class has samples in each split.
"""

import pandas as pd
import numpy as np

def main():
    print("=== Creating balanced Exp1 dataset ===")
    
    # Load metadata
    df = pd.read_csv("data/metadata/metadata.csv")
    
    print(f"Original metadata: {len(df)} total images")
    print(f"Columns: {list(df.columns)}")
    
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
    
    # Get exp1 all data for sampling (not just train split)
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
                # Keep minority classes as is
                balanced_data.append(class_data)
                print(f"  -> Kept all {len(class_data)} samples")
    
    if balanced_data:
        balanced_df = pd.concat(balanced_data, ignore_index=True)
        
        print(f"\nBalanced dataset created: {len(balanced_df)} samples")
        print(f"Balanced lesion distribution:")
        print(balanced_df['unified_diagnosis'].value_counts().sort_index())
        
        # Create new balanced splits (70/15/15) with manual stratification
        print(f"\nCreating balanced splits with manual stratification...")
        
        # Ensure each class has samples in each split
        min_samples_per_split = 1  # At least 1 sample per class per split
        
        # Group by class and create balanced splits
        train_data_list = []
        val_data_list = []
        test_data_list = []
        
        for lesion_type in balanced_df['unified_diagnosis'].unique():
            if pd.isna(lesion_type):
                continue
                
            class_data = balanced_df[balanced_df['unified_diagnosis'] == lesion_type]
            class_count = len(class_data)
            
            if class_count < 3:  # Very small classes, distribute evenly
                if class_count == 1:
                    train_data_list.append(class_data)
                elif class_count == 2:
                    train_data_list.append(class_data.iloc[:1])
                    val_data_list.append(class_data.iloc[1:2])
                else:  # class_count == 3
                    train_data_list.append(class_data.iloc[:1])
                    val_data_list.append(class_data.iloc[1:2])
                    test_data_list.append(class_data.iloc[2:3])
            else:
                # Larger classes, use proper stratification
                # Calculate split sizes to maintain proportions
                train_size = max(1, int(class_count * 0.7))
                val_size = max(1, int(class_count * 0.15))
                test_size = class_count - train_size - val_size
                
                # Ensure test_size is at least 1
                if test_size < 1:
                    val_size -= 1
                    test_size = 1
                
                # Split the data
                train_class = class_data.iloc[:train_size]
                val_class = class_data.iloc[train_size:train_size + val_size]
                test_class = class_data.iloc[train_size + val_size:]
                
                train_data_list.append(train_class)
                val_data_list.append(val_class)
                test_data_list.append(test_class)
        
        # Combine all splits
        train_data = pd.concat(train_data_list, ignore_index=True)
        val_data = pd.concat(val_data_list, ignore_index=True)
        test_data = pd.concat(test_data_list, ignore_index=True)
        
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
        print(f"Original exp1 classes: {exp1_data['unified_diagnosis'].nunique() if 'unified_diagnosis' in exp1_data.columns else 'N/A'}")
        print(f"Balanced classes: {balanced_df['unified_diagnosis'].nunique() if 'unified_diagnosis' in balanced_df.columns else 'N/A'}")
        
        # Check the new column
        exp1_balanced_data = df[df['exp1_balanced'].astype(str) != '']
        print(f"Exp1_balanced data: {len(exp1_balanced_data)} samples")
        if 'unified_diagnosis' in exp1_balanced_data.columns:
            print(f"Exp1_balanced lesion distribution:")
            print(exp1_balanced_data['unified_diagnosis'].value_counts().sort_index())
            
            # Check per-split distribution
            for split in ['train', 'val', 'test']:
                split_data = df[df['exp1_balanced'] == split]
                print(f"\n{split.capitalize()} split ({len(split_data)} samples):")
                print(split_data['unified_diagnosis'].value_counts().sort_index())
    else:
        print("No balanced data created - check if unified_diagnosis column exists")

if __name__ == "__main__":
    main()
