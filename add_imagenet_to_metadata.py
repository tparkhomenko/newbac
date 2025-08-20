import pandas as pd
import os
import random

# Set random seed for reproducibility
random.seed(42)

# Input and output files
input_file = 'data/metadata/metadata.csv'
output_file = 'data/metadata/metadata.csv'

# Number of ImageNet images to add
num_imagenet_images = 15000

print(f"Adding {num_imagenet_images} ImageNet images to metadata.csv...")

# Get list of all ImageNet images
imagenet_images = []
for root, dirs, files in os.walk('data/raw/imagenet'):
    for file in files:
        if file.endswith('.jpg'):
            # Get relative path from data/raw/imagenet
            rel_path = os.path.relpath(os.path.join(root, file), 'data/raw/imagenet')
            imagenet_images.append(rel_path)

print(f"Found {len(imagenet_images)} total ImageNet images")

# Randomly sample ImageNet images
if len(imagenet_images) >= num_imagenet_images:
    sampled_images = random.sample(imagenet_images, num_imagenet_images)
else:
    sampled_images = imagenet_images
    print(f"Warning: Only {len(imagenet_images)} images available, using all")

# Read existing metadata
existing_df = pd.read_csv(input_file)
print(f"Existing metadata has {len(existing_df)} rows")

# Create new ImageNet entries
imagenet_data = []
for image_path in sampled_images:
    # Extract image name without extension
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    imagenet_data.append({
        'diagnosis': 'imagenet',
        'unified_label': 'imagenet',
        'source_csv': 'ImageNet_Sampled',
        'exp1': '',
        'exp2': '',
        'exp3': '',
        'exp4': '',
        'exp5': ''
    })

# Append new data
result_df = pd.concat([existing_df, pd.DataFrame(imagenet_data)], ignore_index=True)

# Save combined result
result_df.to_csv(output_file, index=False)

print(f"\nImageNet addition complete!")
print(f"New ImageNet images added: {len(imagenet_data)}")
print(f"Total images in metadata.csv: {len(result_df)}")
print(f"\nUpdated dataset distribution:")
print(result_df['source_csv'].value_counts())
print(f"\nUpdated diagnosis distribution:")
print(result_df['diagnosis'].value_counts())
print(f"\nUpdated unified label distribution:")
print(result_df['unified_label'].value_counts()) 