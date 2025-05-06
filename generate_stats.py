import pandas as pd
import os
import yaml
from pathlib import Path

# Load configuration
def load_config():
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()
# Get MAX_SAMPLES_PER_CLASS from config, defaulting to 2000 if not found
MAX_SAMPLES_PER_CLASS = config.get('training', {}).get('max_samples_per_class', 2000)
# For validation and test sets, we typically use a percentage of the training samples
VAL_SAMPLES_PER_CLASS = int(MAX_SAMPLES_PER_CLASS * 0.15)  # 15% for validation
TEST_SAMPLES_PER_CLASS = int(MAX_SAMPLES_PER_CLASS * 0.15)  # 15% for testing

# Load full dataset (unified augmented database)
print("Loading dataset metadata...")
df = pd.read_csv('datasets/metadata/unified_augmented.csv')
print(f"Loaded unified_augmented.csv with {len(df)} entries")

# Define classes of interest - only skin lesion classes
skin_classes = ['melanocytic', 'non-melanocytic carcinoma', 'keratosis', 'fibrous', 'vascular']

# Count by class
print("\nGenerating statistics for skin lesion classes...")

# Create a formatted table with fixed-width columns - reordered columns to train, test, val
print('\n| Class                     | Total in DB | Originals | Used Currently | Train    | Test    | Val     |')
print('|---------------------------|------------|-----------|---------------|----------|---------|---------|')

# Process skin lesion classes
for lesion_class in skin_classes:
    # Total in database
    class_total = len(df[df['lesion_group'] == lesion_class])
    
    # Original (non-augmented) images
    originals = len(df[(df['lesion_group'] == lesion_class) & 
                     (df['image'].str.contains('_original.jpg'))])
    
    # Split counts - total available
    train_count_total = len(df[(df['lesion_group'] == lesion_class) & (df['split'] == 'train')])
    val_count_total = len(df[(df['lesion_group'] == lesion_class) & (df['split'] == 'val')])
    test_count_total = len(df[(df['lesion_group'] == lesion_class) & (df['split'] == 'test')])
    
    # Currently used in training (based on pipeline_protocol.md settings)
    # Max samples per class as per configuration
    used_train = min(train_count_total, MAX_SAMPLES_PER_CLASS)
    
    # For small classes, we use all available samples
    if train_count_total <= MAX_SAMPLES_PER_CLASS:
        used_val = val_count_total
        used_test = test_count_total
    else:
        # For larger classes, we use a proportion based on MAX_SAMPLES_PER_CLASS
        used_val = min(val_count_total, VAL_SAMPLES_PER_CLASS)
        used_test = min(test_count_total, TEST_SAMPLES_PER_CLASS)
    
    # Print the row with fixed width formatting - reordered columns to train, test, val
    print(f'| {lesion_class:<25} | {class_total:<10} | {originals:<9} | {used_train:<13} | {used_train:<8} | {used_test:<7} | {used_val:<7} |')

# Calculate totals for bottom row
total_db = sum([len(df[df['lesion_group'] == c]) for c in skin_classes])
total_originals = sum([len(df[(df['lesion_group'] == c) & (df['image'].str.contains('_original.jpg'))]) for c in skin_classes])

# Calculate totals for used samples
total_train_used = 0
total_val_used = 0 
total_test_used = 0

for lesion_class in skin_classes:
    train_count_total = len(df[(df['lesion_group'] == lesion_class) & (df['split'] == 'train')])
    val_count_total = len(df[(df['lesion_group'] == lesion_class) & (df['split'] == 'val')])
    test_count_total = len(df[(df['lesion_group'] == lesion_class) & (df['split'] == 'test')])
    
    used_train = min(train_count_total, MAX_SAMPLES_PER_CLASS)
    total_train_used += used_train
    
    if train_count_total <= MAX_SAMPLES_PER_CLASS:
        total_val_used += val_count_total
        total_test_used += test_count_total
    else:
        total_val_used += min(val_count_total, VAL_SAMPLES_PER_CLASS)
        total_test_used += min(test_count_total, TEST_SAMPLES_PER_CLASS)

# Print the separator and total row - reordered columns to train, test, val
print('|---------------------------|------------|-----------|---------------|----------|---------|---------|')
print(f'| {"TOTAL":<25} | {total_db:<10} | {total_originals:<9} | {total_train_used:<13} | {total_train_used:<8} | {total_test_used:<7} | {total_val_used:<7} |')

# Print summary information
print("\nNote: 'Used Currently' shows the number of samples used in training based on")
print(f"MAX_SAMPLES_PER_CLASS={MAX_SAMPLES_PER_CLASS} from config.yaml")
print(f"Validation and test sets use up to {VAL_SAMPLES_PER_CLASS}/{TEST_SAMPLES_PER_CLASS} samples per class")
print("For classes with fewer samples than the maximum, all available samples are used.")