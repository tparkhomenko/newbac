import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image, ImageEnhance
import random


def main():
    project_root = "/home/parkhomenko/Documents/new_project"
    metadata_path = os.path.join(project_root, "data/metadata/metadata.csv")
    summary_out_path = os.path.join(project_root, "data/metadata/exp6_summary.csv")

    required_columns = [
        "image_name",
        "unified_diagnosis",
        "exp1",
    ]

    classes = [
        "AKIEC",
        "BCC",
        "BKL",
        "DF",
        "MEL",
        "NV",
        "SCC",
        "VASC",
        "NOT_SKIN",
        "UNKNOWN",
    ]

    # Load metadata
    df = pd.read_csv(metadata_path)

    # Validate columns
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in metadata.csv")

    # Drop old exp6 column entirely, then re-add fresh
    if "exp6" in df.columns:
        df = df.drop(columns=["exp6"])  # start clean
    df["exp6"] = "-"

    # Start primarily from exp1 samples, but INCLUDE NOT_SKIN and UNKNOWN even if not in exp1
    df_exp1 = df[(df["exp1"].isin(["train", "val", "test"])) | (df["unified_diagnosis"].isin(["NOT_SKIN", "UNKNOWN"]))].copy()

    # Keep only target classes
    df_exp1 = df_exp1[df_exp1["unified_diagnosis"].isin(classes)].copy()

    # Smart balancing caps/targets
    CAP_5000 = {"NV", "UNKNOWN"}
    CAP_3000 = {"BCC", "MEL"}
    TARGET_1000 = {"DF", "VASC", "SCC", "AKIEC", "NOT_SKIN"}

    # Directory for augmented images
    aug_root = os.path.join(project_root, "data/cleaned_resized/augmented")
    os.makedirs(aug_root, exist_ok=True)

    def apply_aug(img: Image.Image) -> Image.Image:
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        img = img.rotate(random.uniform(-15, 15), resample=Image.BICUBIC, expand=True, fillcolor=(0, 0, 0))
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.9, 1.1))
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.9, 1.1))
        img = ImageEnhance.Color(img).enhance(random.uniform(0.9, 1.1))
        return img

    def find_image_path(image_root: str, image_name: str) -> str:
        base = image_name
        candidates = [base, f"{base}.jpg", f"{base}.jpeg", f"{base}.png"]
        for root, _, files in os.walk(image_root):
            files_set = set(files)
            for cand in candidates:
                if cand in files_set:
                    return os.path.join(root, cand)
        return ""

    # Build per-class frames with caps/oversampling
    parts = []
    augmented_rows_all = []
    rng = random.Random(42)
    for cls, part in df_exp1.groupby("unified_diagnosis", group_keys=False):
        n = len(part)
        # caps
        if cls in CAP_5000 and n > 5000:
            part = part.sample(n=5000, random_state=42)
        if cls in CAP_3000 and len(part) > 3000:
            part = part.sample(n=3000, random_state=42)

        # oversampling to 1000 for minority defined classes
        if cls in TARGET_1000 and len(part) < 1000:
            needed = 1000 - len(part)
            # oversample with augmentation
            # sample with replacement original rows
            sampled = part.sample(n=needed, replace=True, random_state=42)
            new_rows = []
            image_root = os.path.join(project_root, "data/cleaned_resized")
            base_counts = {}
            for _, row in sampled.iterrows():
                img_name = str(row["image_name"]).strip()
                src = find_image_path(image_root, img_name)
                if not src or not os.path.isfile(src):
                    continue
                try:
                    with Image.open(src) as im:
                        im = im.convert("RGB")
                        aug_im = apply_aug(im)
                except Exception:
                    continue
                base = os.path.splitext(os.path.basename(img_name))[0]
                base_counts.setdefault(base, 0)
                base_counts[base] += 1
                idx = base_counts[base]
                aug_name = f"{base}_exp6aug{idx}.jpg"
                # save under augmented dir, keep flat or mirror subdir? flat is fine
                out_path = os.path.join(aug_root, aug_name)
                while os.path.exists(out_path):
                    base_counts[base] += 1
                    idx = base_counts[base]
                    aug_name = f"{base}_exp6aug{idx}.jpg"
                    out_path = os.path.join(aug_root, aug_name)
                try:
                    aug_im.save(out_path, format="JPEG", quality=95)
                except Exception:
                    continue
                # create new metadata row
                new_row = row.copy()
                new_row["image_name"] = os.path.splitext(aug_name)[0]
                # reset experiment columns exp1..exp5 to '-'
                for col in ["exp1", "exp2", "exp3", "exp4", "exp5", "exp1_balanced"]:
                    if col in new_row:
                        new_row[col] = "-"
                new_rows.append(new_row)
            if new_rows:
                aug_df = pd.DataFrame(new_rows)
                # Track to append to main df later
                augmented_rows_all.append(aug_df.copy())
                # Include in class pool for splitting
                part = pd.concat([part, aug_df], ignore_index=True)

        parts.append(part)

    # Preserve original indices where possible; augmented rows will have duplicate indexes removed by reindexing
    df_balanced = pd.concat(parts, ignore_index=True)

    # Append augmented rows to the main metadata so they can receive exp6 labels and persist
    if augmented_rows_all:
        aug_concat = pd.concat(augmented_rows_all, ignore_index=True)
        # Ensure any missing columns exist
        for col in df.columns:
            if col not in aug_concat.columns:
                aug_concat[col] = "-"
        # Align columns
        aug_concat = aug_concat[df.columns]
        df = pd.concat([df, aug_concat], ignore_index=True)

    # Stratified split 70/15/15
    # First split train vs temp (val+test): 70% train, 30% temp
    train_df, temp_df = train_test_split(
        df_balanced,
        test_size=0.30,
        stratify=df_balanced["unified_diagnosis"],
        random_state=42,
    )

    # Split temp into val and test equally: 15/15 (i.e., 0.5 each of temp)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df["unified_diagnosis"],
        random_state=42,
    )

    # Assign labels back to main df in exp6 by image_name to include augmented rows
    def set_split(split_df, label):
        names = set(split_df["image_name"].astype(str))
        df.loc[df["image_name"].astype(str).isin(names), "exp6"] = label
    set_split(train_df, "train")
    set_split(val_df, "val")
    set_split(test_df, "test")

    # Build summary: counts per class per split
    subset = df[df["exp6"].isin(["train", "val", "test"])][["unified_diagnosis", "exp6"]]
    summary = (
        subset.groupby(["unified_diagnosis", "exp6"]).size().unstack(fill_value=0).reindex(index=classes)
    )
    # Add totals per split and per class
    summary["total"] = summary.sum(axis=1)
    summary.loc["TOTAL"] = summary.sum(axis=0)

    # Print split sizes and per-class counts
    print("Exp6 split sizes:")
    print({k: v for k, v in df["exp6"].value_counts().to_dict().items() if k in ["train", "val", "test"]})
    print("\nPer-class counts:")
    print(summary)

    # Save outputs
    # Ensure directory exists
    os.makedirs(os.path.dirname(summary_out_path), exist_ok=True)
    summary.to_csv(summary_out_path)
    df.to_csv(metadata_path, index=False)


if __name__ == "__main__":
    main()


