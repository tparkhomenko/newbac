import os
import random
from typing import List, Dict

import pandas as pd
from PIL import Image, ImageEnhance


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def find_image_path(dataset_root: str, image_name: str) -> str:
    # Search for file by name under dataset_root; allow common extensions
    name_no_ext = os.path.splitext(image_name)[0]
    candidates = [f"{name_no_ext}.jpg", f"{name_no_ext}.jpeg", f"{name_no_ext}.png", image_name]
    for root, _, files in os.walk(dataset_root):
        files_set = set(files)
        for cand in candidates:
            if cand in files_set:
                return os.path.join(root, cand)
    return ""


def apply_augmentations(img: Image.Image) -> Image.Image:
    # Random horizontal/vertical flip
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

    # Small rotation ±15° with expand to keep content
    angle = random.uniform(-15, 15)
    img = img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(0, 0, 0))

    # Random brightness/contrast
    brightness_factor = random.uniform(0.9, 1.1)
    contrast_factor = random.uniform(0.9, 1.1)
    img = ImageEnhance.Brightness(img).enhance(brightness_factor)
    img = ImageEnhance.Contrast(img).enhance(contrast_factor)

    # Small color jitter (saturation via Color, hue not supported by PIL directly)
    color_factor = random.uniform(0.9, 1.1)
    img = ImageEnhance.Color(img).enhance(color_factor)

    return img


def main() -> None:
    project_root = "/home/parkhomenko/Documents/new_project"
    metadata_path = os.path.join(project_root, "data/metadata/metadata.csv")
    # Use cleaned/resized images as requested
    dataset_root = os.path.join(project_root, "data/cleaned_resized")

    classes_to_augment: List[str] = ["AKIEC", "DF", "VASC", "SCC", "NOT_SKIN"]
    target_per_class: int = 3000
    aug_suffix_template = "_aug{}"

    # Load metadata
    df = pd.read_csv(metadata_path)

    # Columns
    exp_cols = ["exp1", "exp2", "exp3", "exp4", "exp5"]
    copy_cols = [c for c in df.columns if c not in ["image_name"]]  # copy everything else by default

    # Only use rows that are part of exp6 (train/val/test)
    df_exp6 = df[df["exp6"].isin(["train", "val", "test"])].copy()

    summary_records: List[Dict] = []
    new_rows: List[Dict] = []

    for cls in classes_to_augment:
        cls_rows = df_exp6[df_exp6["unified_diagnosis"] == cls].copy()
        originals = len(cls_rows)
        if originals >= target_per_class:
            summary_records.append({"Class": cls, "Originals": originals, "Augmented": 0, "Total": originals})
            continue

        needed = target_per_class - originals
        # Sample with replacement from existing images of the class
        sampled = cls_rows.sample(n=needed, replace=True, random_state=42)

        # Build a per-image counter map to create unique filenames
        base_counts: Dict[str, int] = {}
        augmented_count = 0

        for _, row in sampled.iterrows():
            orig_name = str(row["image_name"]).strip()
            if not orig_name:
                continue

            src_path = find_image_path(dataset_root, orig_name)
            if not src_path or not os.path.isfile(src_path):
                # Skip if file missing
                continue

            try:
                with Image.open(src_path) as img:
                    img = img.convert("RGB")
                    aug_img = apply_augmentations(img)
            except Exception:
                continue

            base = os.path.splitext(os.path.basename(orig_name))[0]
            base_counts.setdefault(base, 0)
            base_counts[base] += 1
            idx = base_counts[base]
            aug_name = f"{base}{aug_suffix_template.format(idx)}.jpg"

            # Save alongside original
            out_dir = os.path.dirname(src_path)
            ensure_dir(out_dir)
            out_path = os.path.join(out_dir, aug_name)

            # Ensure no collision by incrementing idx until unique
            while os.path.exists(out_path):
                base_counts[base] += 1
                idx = base_counts[base]
                aug_name = f"{base}{aug_suffix_template.format(idx)}.jpg"
                out_path = os.path.join(out_dir, aug_name)

            try:
                aug_img.save(out_path, format="JPEG", quality=95)
            except Exception:
                continue

            # Construct new metadata row
            new_row = {c: row[c] if c in row else "-" for c in copy_cols}
            # image_name should be just the filename stem (consistent with metadata)
            new_row["image_name"] = os.path.splitext(aug_name)[0]

            # Assign splits and experiments fields
            for c in exp_cols:
                if c in new_row:
                    new_row[c] = "-"
            new_row["exp6"] = row.get("exp6", "-")
            new_row["unified_diagnosis"] = cls

            new_rows.append(new_row)
            augmented_count += 1

        total = originals + augmented_count
        summary_records.append({"Class": cls, "Originals": originals, "Augmented": augmented_count, "Total": total})

    # Append new rows and save
    if new_rows:
        df_aug = pd.DataFrame(new_rows)
        # Align columns to existing order; add missing cols if any
        for col in df.columns:
            if col not in df_aug.columns:
                df_aug[col] = "-"
        df_aug = df_aug[df.columns]
        df = pd.concat([df, df_aug], ignore_index=True)

    df.to_csv(metadata_path, index=False)

    # Print summary
    summary_df = pd.DataFrame(summary_records, columns=["Class", "Originals", "Augmented", "Total"])
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()


