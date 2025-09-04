import os
from pathlib import Path

BASE = Path("/home/parkhomenko/Documents/new_project").resolve()
OUTPUT_FILE = BASE / "project_structure.txt"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp", ".svg", ".ico"}
IGNORE_DIRS = {".git", "node_modules", ".venv", "__pycache__", ".mypy_cache", ".idea", ".pytest_cache"}
IGNORE_FILES = {OUTPUT_FILE.name, "export_tree.py"}
IMAGE_DATASET_DIRS_LIMIT = 2
IMAGE_DATASET_DIR_NAMES = {"images", "train", "val", "test", "JPEGImages"}


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def listdir_safe(path: Path):
    try:
        return sorted(os.listdir(path))
    except Exception:
        return []


def build_entries_for_dir(dir_path: Path):
    entries = listdir_safe(dir_path)
    pruned = []
    image_count = 0
    for name in entries:
        full = dir_path / name
        if name in IGNORE_FILES:
            continue
        if full.is_dir() and name in IGNORE_DIRS:
            continue
        # Add the entry
        pruned.append(name)
        if full.is_file() and is_image(full):
            image_count += 1
            if image_count >= 2:
                break
    return pruned


def tree(dir_path: Path, prefix: str = ""):
    lines = []
    entries = build_entries_for_dir(dir_path)
    total = len(entries)
    # Shared traversal context across the entire tree to enforce global caps
    ctx = {"image_dataset_dirs_seen": 0}
    for idx, name in enumerate(entries):
        full = dir_path / name
        connector = "└── " if idx == total - 1 else "├── "
        lines.append(f"{prefix}{connector}{name}")
        if full.is_dir():
            extension_prefix = "    " if idx == total - 1 else "│   "
            lines.extend(_tree_recurse(full, prefix + extension_prefix, ctx))
    return lines


def _tree_recurse(dir_path: Path, prefix: str, ctx: dict | None = None):
    if ctx is None:
        ctx = {"image_dataset_dirs_seen": 0}

    lines = []
    # If this directory is a dataset-style image directory, enforce the global limit
    if _is_image_dataset_dir(dir_path):
        if ctx["image_dataset_dirs_seen"] >= IMAGE_DATASET_DIRS_LIMIT:
            return lines  # Skip recursing into further image dataset-style dirs
        ctx["image_dataset_dirs_seen"] += 1

    entries = build_entries_for_dir(dir_path)
    total = len(entries)
    for idx, name in enumerate(entries):
        full = dir_path / name
        connector = "└── " if idx == total - 1 else "├── "
        lines.append(f"{prefix}{connector}{name}")
        if full.is_dir():
            extension_prefix = "    " if idx == total - 1 else "│   "
            lines.extend(_tree_recurse(full, prefix + extension_prefix, ctx))
    return lines


def _is_image_dataset_dir(path: Path) -> bool:
    # Match directories like images/, train/, val/, test/, JPEGImages/ (name-based)
    name = path.name
    return name in IMAGE_DATASET_DIR_NAMES


if __name__ == "__main__":
    print(BASE.name)
    for line in tree(BASE):
        print(line)
