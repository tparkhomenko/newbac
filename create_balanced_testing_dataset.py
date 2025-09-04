#!/usr/bin/env python3
"""
Create a balanced testing dataset with 50 images per class from exp1 test split.
This will replace the current testing_only folder with a balanced dataset and structured metadata.
"""

import os
import shutil
import pandas as pd
from pathlib import Path
import random

def create_balanced_testing_dataset():
    """Create a balanced testing dataset and structured testing metadata.csv with skin/lesion/bm labels."""
    
    # Paths
    project_root = Path(__file__).parent
    metadata_path = project_root / "data" / "metadata" / "metadata.csv"
    source_dirs = [
        project_root / "data" / "cleaned_resized" / "isic2018_512",
        project_root / "data" / "cleaned_resized" / "isic2019_512", 
        project_root / "data" / "cleaned_resized" / "isic2020_512"
    ]
    testing_dir = project_root / "data" / "testing_only"
    
    print("🧪 Creating Balanced Testing Dataset from Exp1 Test Split")
    print("=" * 60)
    
    # Load metadata
    print(f"📊 Loading metadata from: {metadata_path}")
    df = pd.read_csv(metadata_path)
    print(f"   Total images: {len(df)}")
    
    # Get exp1 test split
    exp1_test = df[df["exp1"] == "test"]
    print(f"   Exp1 test images: {len(exp1_test)}")
    
    # Check which test images are actually available across all source directories
    print(f"\n🔍 Searching for available images in source directories:")
    available_images = set()
    for source_dir in source_dirs:
        if source_dir.exists():
            count = len(list(source_dir.glob("*.jpg")))
            print(f"   {source_dir.name}: {count} images")
            available_images.update([f.stem for f in source_dir.glob("*.jpg")])
        else:
            print(f"   {source_dir.name}: directory not found")
    
    print(f"   Total unique images found: {len(available_images)}")
    
    exp1_test_available = exp1_test[exp1_test["image_name"].isin(available_images)]
    print(f"   Exp1 test images available in source: {len(exp1_test_available)}")
    
    # Show available diagnosis distribution
    print(f"\n📋 Available diagnosis distribution in exp1 test:")
    diagnosis_counts = exp1_test_available["unified_diagnosis"].value_counts()
    for diagnosis, count in diagnosis_counts.items():
        print(f"   {diagnosis}: {count}")
    
    # Define the 10 unified classes we want to sample
    target_classes = {
        'melanoma': ['MEL'],
        'nevus': ['NV'],
        'basal_cell_carcinoma': ['BCC'],
        'squamous_cell_carcinoma': ['SCC'],
        'seborrheic_keratosis': ['BKL'],
        'actinic_keratosis': ['AKIEC'],
        'dermatofibroma': ['DF'],
        'vascular': ['VASC'],
        'unknown': ['unknown'],
        'not_skin': ['not_skin']
    }
    
    print(f"\n🎯 Target classes for balanced dataset:")
    for group, classes in target_classes.items():
        print(f"   {group}: {', '.join(classes)}")
    
    # Clear existing testing directory
    if testing_dir.exists():
        print(f"\n🗑️  Clearing existing testing directory: {testing_dir}")
        shutil.rmtree(testing_dir)
    
    testing_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample images per class (aim for 50, but use what's available)
    target_images_per_class = 50
    total_images = 0
    
    print(f"\n📸 Sampling up to {target_images_per_class} images per class...")
    
    for group_name, class_codes in target_classes.items():
        print(f"\n   📁 {group_name} ({', '.join(class_codes)}):")
        
        # Get all available images for this group from exp1 test
        group_images = exp1_test_available[exp1_test_available["unified_diagnosis"].isin(class_codes)]
        
        if len(group_images) == 0:
            print(f"      ⚠️  No images found for classes: {class_codes}")
            continue
        
        # Sample up to target_images_per_class images
        if len(group_images) >= target_images_per_class:
            sampled = group_images.sample(n=target_images_per_class, random_state=42)
            print(f"      📊 Sampled {target_images_per_class} from {len(group_images)} available")
        else:
            print(f"      📊 Using all {len(group_images)} available (target: {target_images_per_class})")
            sampled = group_images
        
        # Copy images to testing directory
        for _, row in sampled.iterrows():
            image_name = row["image_name"]
            source_path = None
            
            # Find the image in one of the source directories
            for source_dir in source_dirs:
                potential_path = source_dir / f"{image_name}.jpg"
                if potential_path.exists():
                    source_path = potential_path
                    break
            
            if source_path:
                dest_path = testing_dir / f"{image_name}.jpg"
                shutil.copy2(source_path, dest_path)
                print(f"      ✅ {image_name}.jpg ({row['unified_diagnosis']}) from {source_path.parent.name}")
                total_images += 1
            else:
                print(f"      ❌ {image_name}.jpg (source not found in any directory)")
    
    print(f"\n🎉 Dataset creation complete!")
    print(f"   📁 Testing directory: {testing_dir}")
    print(f"   📊 Total images: {total_images}")
    print(f"   🎯 Target: {len(target_classes) * target_images_per_class} images")
    
    # Verify the final dataset
    print(f"\n🔍 Verifying final dataset...")
    final_files = list(testing_dir.glob("*.jpg"))
    print(f"   Files in testing_only: {len(final_files)}")
    
    if final_files:
        print(f"   Sample files:")
        for i, file_path in enumerate(final_files[:5]):
            print(f"      {file_path.name}")
        if len(final_files) > 5:
            print(f"      ... and {len(final_files) - 5} more")
    
    # Build structured testing metadata.csv
    print(f"\n📝 Writing testing metadata.csv ...")
    abbr_to_full = {
        'NV': 'nevus', 'MEL': 'melanoma', 'BCC': 'basal_cell_carcinoma', 'SCC': 'squamous_cell_carcinoma',
        'BKL': 'seborrheic_keratosis', 'AKIEC': 'actinic_keratosis', 'DF': 'dermatofibroma', 'VASC': 'vascular'
    }
    malignant_full = {"melanoma", "basal_cell_carcinoma", "squamous_cell_carcinoma"}
    testing_rows = []
    
    for file_path in final_files:
        image_name = file_path.stem
        row = exp1_test_available[exp1_test_available["image_name"] == image_name]
        unified = None
        if not row.empty:
            unified = row.iloc[0]["unified_diagnosis"]
        
        # Map unified to structured labels
        if unified is None:
            skin_label = "skin"
            lesion_label = "-"
            bm_label = "-"
        else:
            full = abbr_to_full.get(unified, unified)
            if full == 'not_skin':
                skin_label = 'not_skin'
                lesion_label = '-'
                bm_label = '-'
            elif full == 'unknown':
                skin_label = 'skin'
                lesion_label = '-'
                bm_label = '-'
            else:
                skin_label = 'skin'
                # Store lesion label as the 3-6 letter abbreviation if available else '-'
                lesion_label = unified if unified in abbr_to_full else '-'
                bm_label = 'malignant' if full in malignant_full else 'benign'
        
        testing_rows.append({
            'filename': image_name,
            'skin_label': skin_label,
            'lesion_label': lesion_label,
            'bm_label': bm_label,
            'unified_label': unified if unified is not None else '',
        })
    
    meta_df = pd.DataFrame(testing_rows)
    meta_df.to_csv(testing_dir / 'metadata.csv', index=False)
    print(f"   Wrote {len(meta_df)} rows to {testing_dir / 'metadata.csv'}")
    
    # Show final class distribution
    print(f"\n📊 Final class distribution:")
    final_metadata = []
    for file_path in final_files:
        image_name = file_path.stem
        row = exp1_test_available[exp1_test_available["image_name"] == image_name]
        if not row.empty:
            final_metadata.append(row.iloc[0]["unified_diagnosis"])
    
    if final_metadata:
        final_counts = pd.Series(final_metadata).value_counts()
        for diagnosis, count in final_counts.items():
            print(f"   {diagnosis}: {count}")
    
    return total_images

if __name__ == "__main__":
    try:
        create_balanced_testing_dataset()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
