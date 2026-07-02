"""
Compute the mean ground-truth saliency map over the training split.
This is the centre-bias prior: the average fixation density across all
3,189 training clips, reflecting spatial biases shared by all samples
(centre tendency from VideoSwin Kinetics pretraining + dataset recording setup).

Used for:
  - Debiased NSS: subtract from model predictions before computing metrics
  - Centre-bias baseline: use as the prediction for every test sample

Usage:
    python scripts/compute_mean_gt.py \
        --data-dir /path/to/crowdfix-data \
        --splits splits.json \
        --out results/mean_gt_saliency.npy \
        --frame-size 224 384
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--splits", default="splits.json")
    p.add_argument("--out", default="results/mean_gt_saliency.npy")
    p.add_argument("--frame-size", nargs=2, type=int, default=[224, 384])
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    sal_root = data_dir / "SaliencyMaps"
    h, w = args.frame_size

    with open(args.splits) as f:
        splits = json.load(f)

    train_vids = splits["train"]
    acc = np.zeros((h, w), dtype=np.float64)
    count = 0

    for vid in train_vids:
        frame_dir = data_dir / "Frames" / vid
        if not frame_dir.is_dir():
            continue
        for fp in sorted(frame_dir.glob("frame_*.jpg")):
            frame_num = int(fp.stem.split("_")[1])
            sal_path = sal_root / f"{vid}_{frame_num:03d}.png"
            if not sal_path.exists():
                continue
            sal = Image.open(sal_path).convert("L")
            sal = TF.resize(sal, [h, w], antialias=True)
            acc += np.array(sal, dtype=np.float32) / 255.0
            count += 1

    if count == 0:
        raise RuntimeError(f"No saliency maps found under {sal_root}")

    mean_gt = (acc / count).astype(np.float32)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, mean_gt)
    print(f"Saved mean GT ({h}x{w}) from {count} training frames → {args.out}")
    print(f"  min={mean_gt.min():.4f}  max={mean_gt.max():.4f}  mean={mean_gt.mean():.4f}")


if __name__ == "__main__":
    main()
