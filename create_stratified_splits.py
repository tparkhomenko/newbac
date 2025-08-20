import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def create_stratified_splits(df, split_column, unified_label_col='unified_label', 
                           train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    """
    Create stratified train/val/test splits for a given split column.
    
    Args:
        df: DataFrame containing the data
        split_column: Name of the split column (e.g., 'split1')
        unified_label_col: Name of the label column for stratification
        train_size: Proportion for training set
        val_size: Proportion for validation set  
        test_size: Proportion for test set
        random_state: Random seed for reproducibility
    
    Returns:
        DataFrame with updated split column
    """
    # Get rows where the split column equals 1
    split_mask = df[split_column] == 1
    split_df = df[split_mask].copy()
    
    if len(split_df) == 0:
        print(f"No images found for {split_column}")
        return df
    
    # Get the labels for stratification
    labels = split_df[unified_label_col].values
    
    # Calculate validation size relative to what remains after test split
    val_from_train = val_size / (1 - test_size)
    
    # First split off test set
    train_val_df, test_df = train_test_split(
        split_df, 
        test_size=test_size,
        stratify=labels,
        random_state=random_state
    )
    
    # Then split validation from training
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_from_train,
        stratify=train_val_df[unified_label_col],
        random_state=random_state
    )
    
    # Assign split labels
    train_df[split_column] = 'train'
    val_df[split_column] = 'val'
    test_df[split_column] = 'test'
    
    # Combine back with the original dataframe
    result_df = df.copy()
    result_df.loc[train_df.index, split_column] = 'train'
    result_df.loc[val_df.index, split_column] = 'val'
    result_df.loc[test_df.index, split_column] = 'test'
    
    # Replace 0s with empty strings
    result_df[split_column] = result_df[split_column].replace(0, '')
    
    return result_df, train_df, val_df, test_df

def print_split_statistics(split_name, train_df, val_df, test_df, unified_label_col='unified_label'):
    """Print statistics for a split."""
    print(f"\n{split_name.upper()} Statistics:")
    print(f"{'='*50}")
    
    # Count total images in this split
    total_images = len(train_df) + len(val_df) + len(test_df)
    print(f"Total images: {total_images}")
    print(f"Train: {len(train_df)} ({len(train_df)/total_images*100:.1f}%)")
    print(f"Val: {len(val_df)} ({len(val_df)/total_images*100:.1f}%)")
    print(f"Test: {len(test_df)} ({len(test_df)/total_images*100:.1f}%)")
    
    # Show label distribution
    print(f"\nLabel distribution:")
    all_labels = sorted(train_df[unified_label_col].unique())
    
    for label in all_labels:
        train_count = len(train_df[train_df[unified_label_col] == label])
        val_count = len(val_df[val_df[unified_label_col] == label])
        test_count = len(test_df[test_df[unified_label_col] == label])
        total_count = train_count + val_count + test_count
        
        if total_count > 0:
            print(f"  {label}: Train={train_count}, Val={val_count}, Test={test_count} (Total={total_count})")
    
    # Show source dataset distribution
    print(f"\nSource dataset distribution:")
    all_sources = sorted(train_df['source_csv'].unique())
    
    for source in all_sources:
        train_count = len(train_df[train_df['source_csv'] == source])
        val_count = len(val_df[val_df['source_csv'] == source])
        test_count = len(test_df[test_df['source_csv'] == source])
        total_count = train_count + val_count + test_count
        
        if total_count > 0:
            print(f"  {source}: Train={train_count}, Val={val_count}, Test={test_count} (Total={total_count})")

def print_detailed_source_statistics(split_name, train_df, val_df, test_df):
    """Print detailed statistics about source datasets for each split."""
    print(f"\n{split_name.upper()} - DETAILED SOURCE DATASET BREAKDOWN:")
    print(f"{'='*70}")
    
    # Combine all dataframes for analysis
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    # Create a summary table
    summary_data = []
    
    for source in sorted(all_df['source_csv'].unique()):
        source_data = all_df[all_df['source_csv'] == source]
        
        # Count by split type
        train_count = len(train_df[train_df['source_csv'] == source])
        val_count = len(val_df[val_df['source_csv'] == source])
        test_count = len(test_df[test_df['source_csv'] == source])
        total_count = len(source_data)
        
        # Count by label
        label_counts = source_data['unified_label'].value_counts().to_dict()
        
        summary_data.append({
            'Source': source,
            'Total': total_count,
            'Train': train_count,
            'Val': val_count,
            'Test': test_count,
            'Labels': label_counts
        })
    
    # Print summary table
    print(f"{'Source':<50} {'Total':<8} {'Train':<8} {'Val':<8} {'Test':<8}")
    print("-" * 90)
    
    for row in summary_data:
        print(f"{row['Source']:<50} {row['Total']:<8} {row['Train']:<8} {row['Val']:<8} {row['Test']:<8}")
    
    # Print label breakdown for each source
    print(f"\nLabel breakdown by source:")
    for row in summary_data:
        if row['Labels']:
            print(f"\n{row['Source']}:")
            for label, count in sorted(row['Labels'].items()):
                print(f"  {label}: {count}")

def main():
    # Read the CSV file
    print("Loading CSV file...")
    df = pd.read_csv('data/metadata/unified_labels_with_splits.csv')
    print(f"Loaded {len(df)} total images")
    
    # Get unique labels for reference
    unique_labels = df['unified_label'].unique()
    print(f"Unique labels: {unique_labels}")
    
    # Get unique source datasets
    unique_sources = df['source_csv'].unique()
    print(f"Source datasets: {unique_sources}")
    
    # Process each split column
    split_columns = ['split1', 'split2', 'split3', 'split4', 'split5']
    result_df = df.copy()
    
    for split_col in split_columns:
        if split_col in df.columns:
            print(f"\nProcessing {split_col}...")
            
            # Create stratified splits
            result_df, train_df, val_df, test_df = create_stratified_splits(
                result_df, split_col, 'unified_label'
            )
            
            # Print basic statistics
            print_split_statistics(split_col, train_df, val_df, test_df)
            
            # Print detailed source statistics
            print_detailed_source_statistics(split_col, train_df, val_df, test_df)
        else:
            print(f"Warning: Column {split_col} not found in CSV")
    
    # Save the result
    output_path = 'data/metadata/unified_labels_with_stratified_splits.csv'
    result_df.to_csv(output_path, index=False)
    print(f"\nSaved stratified splits to: {output_path}")
    
    # Print overall summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")
    
    for split_col in split_columns:
        if split_col in result_df.columns:
            # Count non-empty values (actual splits)
            split_counts = result_df[split_col].value_counts()
            total_in_split = split_counts.sum()
            
            print(f"\n{split_col}:")
            print(f"  Total images in split: {total_in_split}")
            for split_type, count in split_counts.items():
                if split_type != '':  # Skip empty strings
                    print(f"  {split_type}: {count} ({count/total_in_split*100:.1f}%)")

if __name__ == "__main__":
    main() 