"""
Create 70/15/15 train/val/test splits stratified by crowd density category.

Usage:
    python data/make_splits.py \
        --category-info /path/to/CategoryInfo.xlsx \
        --frames-dir /path/to/Frames \
        --out splits.json

CategoryInfo.xlsx must have columns: VideoName (or similar), Category (SP/DF/DC).
Output: splits.json with keys "train", "val", "test", each a list of video stem strings.
"""

import argparse
import json
import random
from pathlib import Path

import openpyxl


def load_categories(xlsx_path: Path) -> dict[str, str]:
    """Returns {video_stem: category} from CategoryInfo.xlsx."""
    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    header = [str(c).strip().lower() if c else "" for c in rows[0]]

    # Find the video name and category columns
    name_col = next((i for i, h in enumerate(header) if "video" in h or "name" in h), 0)
    cat_col = next((i for i, h in enumerate(header) if "cat" in h or "type" in h), 1)

    categories = {}
    for row in rows[1:]:
        if row[name_col] is None:
            continue
        raw = str(row[name_col]).strip()
        # Video Number column is an integer (1, 2, ...) — zero-pad to 3 digits
        # Also handle "001.avi" / "001" style strings
        stem = Path(raw).stem  # strips extension if present
        try:
            stem = f"{int(stem):03d}"  # zero-pad integers
        except ValueError:
            pass  # already a string stem like "001"
        cat = str(row[cat_col]).strip().upper() if row[cat_col] else "UNKNOWN"
        # Skip summary rows embedded in the xlsx (category names in the video-name column)
        if cat not in ("SP", "DF", "DC"):
            continue
        categories[stem] = cat
    return categories


def split_stratified(
    videos: list[str], categories: dict[str, str], train_frac: float = 0.70, val_frac: float = 0.15, seed: int = 42
) -> tuple[list[str], list[str], list[str]]:
    rng = random.Random(seed)
    by_cat: dict[str, list[str]] = {}
    for v in videos:
        c = categories.get(v, "UNKNOWN")
        by_cat.setdefault(c, []).append(v)

    train, val, test = [], [], []
    for cat, vids in sorted(by_cat.items()):
        rng.shuffle(vids)
        n = len(vids)
        n_train = max(1, round(n * train_frac))
        n_val = max(1, round(n * val_frac))
        train += vids[:n_train]
        val += vids[n_train : n_train + n_val]
        test += vids[n_train + n_val :]
        print(f"  {cat}: {n} videos → {len(vids[:n_train])} train / {len(vids[n_train:n_train+n_val])} val / {len(vids[n_train+n_val:])} test")

    return train, val, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category-info", required=True)
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--out", default="splits.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir)
    available = sorted(p.name for p in frames_dir.iterdir() if p.is_dir())
    print(f"Found {len(available)} video directories in {frames_dir}")

    categories = load_categories(Path(args.category_info))
    print(f"Loaded {len(categories)} video categories")

    train, val, test = split_stratified(available, categories, seed=args.seed)
    print(f"\nTotal: {len(train)} train / {len(val)} val / {len(test)} test")

    # Include per-video category for density conditioning at training time
    splits = {
        "train": sorted(train),
        "val": sorted(val),
        "test": sorted(test),
        "categories": categories,  # {video_stem: "SP"|"DF"|"DC"}
    }
    with open(args.out, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"Saved splits to {args.out}")


if __name__ == "__main__":
    main()
