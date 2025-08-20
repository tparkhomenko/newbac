import os

# Desired folder structure
folders = [
    'data/raw/isic2018',
    'data/raw/isic2019',
    'data/raw/isic2020',
    'data/raw/dtd',
    'data/raw/imagenet_subset',
    'data/cleaned_resized/isic2018_512',
    'data/cleaned_resized/isic2019_512',
    'data/cleaned_resized/isic2020_512',
    'data/cleaned_resized/plausibility_check_512',
    'data/metadata',
]

def create_folders(folder_list):
    for folder in folder_list:
        os.makedirs(folder, exist_ok=True)
        print(f"Created or already exists: {folder}")

if __name__ == "__main__":
    create_folders(folders) 