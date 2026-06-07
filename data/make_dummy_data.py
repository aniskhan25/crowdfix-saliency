"""
Generate minimal synthetic CrowdFix data for smoke-testing the training pipeline.

Creates the same directory structure that CrowdFixDataset expects:
  <data_dir>/Frames/<video_id>/frame_NNNNN.jpg
  <data_dir>/SaliencyMaps/<video_id>/frame_NNNNN.png
  <data_dir>/BinaryMaps/<video_id>/frame_NNNNN.png
  <data_dir>/CategoryInfo.xlsx

Usage:
    python data/make_dummy_data.py --data-dir /path/to/crowdfix-data
"""

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image
import openpyxl


CATEGORIES = {
    "SP": ["video_001", "video_002", "video_003"],
    "DF": ["video_004", "video_005", "video_006"],
    "DC": ["video_007", "video_008", "video_009"],
}
N_FRAMES = 20
H, W = 360, 640


def make_frame(rng, idx: int) -> Image.Image:
    arr = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def make_sal(rng) -> Image.Image:
    # Gaussian blob as a fake saliency map
    cy, cx = rng.integers(H // 4, 3 * H // 4), rng.integers(W // 4, 3 * W // 4)
    y, x = np.ogrid[:H, :W]
    sal = np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * (H // 8) ** 2))
    sal = (sal / sal.max() * 255).astype(np.uint8)
    return Image.fromarray(sal, mode="L")


def make_fix(rng) -> Image.Image:
    arr = np.zeros((H, W), dtype=np.uint8)
    for _ in range(5):
        r = rng.integers(0, H)
        c = rng.integers(0, W)
        arr[r, c] = 255
    return Image.fromarray(arr, mode="L")


def make_category_xlsx(data_dir: Path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["VideoName", "Category"])
    for cat, vids in CATEGORIES.items():
        for vid in vids:
            ws.append([vid, cat])
    wb.save(data_dir / "CategoryInfo.xlsx")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    rng = np.random.default_rng(args.seed)

    for cat, vids in CATEGORIES.items():
        for vid in vids:
            frame_dir = data_dir / "Frames" / vid
            sal_dir   = data_dir / "SaliencyMaps" / vid
            fix_dir   = data_dir / "BinaryMaps" / vid
            frame_dir.mkdir(parents=True, exist_ok=True)
            sal_dir.mkdir(parents=True, exist_ok=True)
            fix_dir.mkdir(parents=True, exist_ok=True)

            for i in range(1, N_FRAMES + 1):
                name = f"frame_{i:05d}"
                make_frame(rng, i).save(frame_dir / f"{name}.jpg", quality=85)
                make_sal(rng).save(sal_dir / f"{name}.png")
                make_fix(rng).save(fix_dir / f"{name}.png")

            print(f"  {cat}/{vid}: {N_FRAMES} frames")

    make_category_xlsx(data_dir)
    print(f"\nDummy data written to {data_dir}")
    print("Now run: python data/make_splits.py --category-info <data_dir>/CategoryInfo.xlsx "
          "--frames-dir <data_dir>/Frames --out splits.json")


if __name__ == "__main__":
    main()
