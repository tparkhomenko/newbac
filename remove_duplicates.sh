#!/bin/bash

# Script to remove duplicate images from raw and resized directories

CSV_FILE="data/raw/csv/ISIC_2020_Training_Duplicates.csv"
RAW_DIR="data/raw/isic2020/ISIC_2020_Training_JPEG/train"
RESIZED_DIR="data/cleaned_resized/isic2020_512"

echo "Checking for duplicate images..."

# Count total duplicates
TOTAL_PAIRS=$(tail -n +2 $CSV_FILE | wc -l)
echo "Found $TOTAL_PAIRS duplicate pairs in CSV file"

# Check how many files exist in raw directory
RAW_FOUND=0
RAW_NOT_FOUND=0
RESIZED_FOUND=0
RESIZED_NOT_FOUND=0

while IFS=',' read -r img1 img2; do
    if [[ "$img1" != "image_name_1" ]]; then
        # Check raw directory
        if [[ -f "$RAW_DIR/$img1.jpg" ]]; then
            ((RAW_FOUND++))
        else
            ((RAW_NOT_FOUND++))
        fi
        if [[ -f "$RAW_DIR/$img2.jpg" ]]; then
            ((RAW_FOUND++))
        else
            ((RAW_NOT_FOUND++))
        fi

        # Check resized directory
        if [[ -f "$RESIZED_DIR/$img1.jpg" ]]; then
            ((RESIZED_FOUND++))
        else
            ((RESIZED_NOT_FOUND++))
        fi
        if [[ -f "$RESIZED_DIR/$img2.jpg" ]]; then
            ((RESIZED_FOUND++))
        else
            ((RESIZED_NOT_FOUND++))
        fi
    fi
done < $CSV_FILE

echo "=== DUPLICATE CHECK SUMMARY ==="
echo "Total duplicate pairs: $TOTAL_PAIRS"
echo "Raw images found: $RAW_FOUND"
echo "Raw images not found: $RAW_NOT_FOUND"
echo "Resized images found: $RESIZED_FOUND"
echo "Resized images not found: $RESIZED_NOT_FOUND"
echo "================================"

# Ask for confirmation
echo ""
echo "WARNING: This will delete duplicate images from both directories!"
read -p "Do you want to proceed? (yes/no): " response

if [[ "$response" == "yes" || "$response" == "y" ]]; then
    echo "Removing duplicate images..."
    
    RAW_DELETED=0
    RESIZED_DELETED=0
    
    while IFS=',' read -r img1 img2; do
        if [[ "$img1" != "image_name_1" ]]; then
            # Remove from raw directory
            if [[ -f "$RAW_DIR/$img1.jpg" ]]; then
                rm "$RAW_DIR/$img1.jpg"
                echo "Deleted raw image: $img1.jpg"
                ((RAW_DELETED++))
            fi
            if [[ -f "$RAW_DIR/$img2.jpg" ]]; then
                rm "$RAW_DIR/$img2.jpg"
                echo "Deleted raw image: $img2.jpg"
                ((RAW_DELETED++))
            fi
            
            # Remove from resized directory
            if [[ -f "$RESIZED_DIR/$img1.jpg" ]]; then
                rm "$RESIZED_DIR/$img1.jpg"
                echo "Deleted resized image: $img1.jpg"
                ((RESIZED_DELETED++))
            fi
            if [[ -f "$RESIZED_DIR/$img2.jpg" ]]; then
                rm "$RESIZED_DIR/$img2.jpg"
                echo "Deleted resized image: $img2.jpg"
                ((RESIZED_DELETED++))
            fi
        fi
    done < $CSV_FILE
    
    echo ""
    echo "=== DELETION SUMMARY ==="
    echo "Raw images deleted: $RAW_DELETED"
    echo "Resized images deleted: $RESIZED_DELETED"
    echo "========================"
else
    echo "Operation cancelled."
fi 