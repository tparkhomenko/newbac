#!/usr/bin/env python3
"""
Create balanced Exp1 dataset with equal samples per class.
Ensures each class has exactly the same number of samples in each split.
"""

import pandas as pd
import numpy as np

def main():
    print("=== Creating balanced Exp1 dataset (equal samples per class) ===")
    
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
    
    # Get exp1 all data for sampling
    exp1_all = df[df['exp1'].astype(str) != '']
    
    print(f"Exp1 all data: {len(exp1_all)} samples")
    
    # Define target samples per class (use the minimum available class size)
    if 'unified_diagnosis' in exp1_all.columns:
        # Exclude NOT_SKIN since it's not used for the lesion task
        lesion_classes = [cls for cls in exp1_all['unified_diagnosis'].unique() if cls != 'NOT_SKIN' and not pd.isna(cls)]
        
        # Find the minimum class size
        min_class_size = float('inf')
        for cls in lesion_classes:
            class_size = len(exp1_all[exp1_all['unified_diagnosis'] == cls])
            min_class_size = min(min_class_size, class_size)
            print(f"Class {cls}: {class_size} samples")
        
        # Use a reasonable target that ensures all classes can contribute
        target_samples_per_class = min(500, min_class_size)  # Cap at 500 to keep it manageable
        print(f"\nTarget samples per class: {target_samples_per_class}")
        
        # Create balanced dataset
        balanced_data = []
        for lesion_type in lesion_classes:
            class_data = exp1_all[exp1_all['unified_diagnosis'] == lesion_type]
            
            if len(class_data) >= target_samples_per_class:
                # Sample from larger classes
                sampled = class_data.sample(n=target_samples_per_class, random_state=42)
                balanced_data.append(sampled)
                print(f"  -> Sampled {len(sampled)} samples from {lesion_type}")
            else:
                # Use all available samples from smaller classes
                balanced_data.append(class_data)
                print(f"  -> Used all {len(class_data)} samples from {lesion_type}")
        
        if balanced_data:
            balanced_df = pd.concat(balanced_data, ignore_index=True)
            
            print(f"\nBalanced dataset created: {len(balanced_df)} samples")
            print(f"Balanced lesion distribution:")
            print(balanced_df['unified_diagnosis'].value_counts().sort_index())
            
            # Create balanced splits ensuring each class is represented
            print(f"\nCreating balanced splits...")
            
            train_data_list = []
            val_data_list = []
            test_data_list = []
            
            # For each class, distribute samples across splits
            for lesion_type in balanced_df['unified_diagnosis'].unique():
                if pd.isna(lesion_type):
                    continue
                    
                class_data = balanced_df[balanced_df['unified_diagnosis'] == lesion_type]
                class_count = len(class_data)
                
                # Calculate samples per split (70/15/15)
                train_size = max(1, int(class_count * 0.7))
                val_size = max(1, int(class_count * 0.15))
                test_size = class_count - train_size - val_size
                
                # Ensure test_size is at least 1
                if test_size < 1:
                    val_size -= 1
                    test_size = 1
                
                # Split the class data
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
            print("No balanced data created")
    else:
        print("unified_diagnosis column not found")

if __name__ == "__main__":
    main()








