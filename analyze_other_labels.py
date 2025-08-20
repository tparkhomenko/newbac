#!/usr/bin/env python3
"""
Analyze sources of 'other' labels in unified CSV.
"""

import pandas as pd

def analyze_other_labels():
    df = pd.read_csv('data/metadata/unified_labels.csv')
    other_df = df[df['unified_label'] == 'other']
    
    print("=== ANALYSIS OF 'OTHER' LABELS ===")
    print(f"Total 'other' labels: {len(other_df)}")
    print(f"Percentage of 'other': {len(other_df)/len(df)*100:.1f}%")
    
    print("\nSource distribution for 'other' labels:")
    source_counts = other_df['source_csv'].value_counts()
    for source, count in source_counts.items():
        print(f"  {source}: {count}")
    
    print("\nSample 'other' entries from each source:")
    for source in other_df['source_csv'].unique():
        print(f"\n=== {source} ===")
        sample = other_df[other_df['source_csv'] == source].head(3)
        print(sample[['image', 'diagnosis', 'unified_label', 'source_csv']].to_string(index=False))
        
        # Check what columns are available in original file
        print(f"\nColumns in original {source}:")
        original_file = f"data/raw/csv/{source}"
        try:
            original_df = pd.read_csv(original_file)
            print(f"  Available columns: {list(original_df.columns)}")
            if 'diagnosis' in original_df.columns:
                print(f"  Sample diagnoses: {original_df['diagnosis'].value_counts().head(5).to_dict()}")
        except Exception as e:
            print(f"  Error reading original file: {e}")

if __name__ == "__main__":
    analyze_other_labels() 