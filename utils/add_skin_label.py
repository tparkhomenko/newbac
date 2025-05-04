import pandas as pd
from pathlib import Path

def add_skin_label():
    """Add skin label column to ISIC metadata"""
    # Read the metadata
    metadata_path = 'data/metadata/metadata_isic2019.csv'
    df = pd.read_csv(metadata_path)
    
    # Add skin column (all ISIC images are skin lesions)
    df['skin'] = 1
    
    # Save updated metadata
    df.to_csv(metadata_path, index=False)
    
    print(f"Added 'skin' column to {metadata_path}")
    print(f"Total images labeled as skin: {len(df)}")
    print("\nFirst few rows of updated metadata:")
    print(df.head())

if __name__ == "__main__":
    add_skin_label() 