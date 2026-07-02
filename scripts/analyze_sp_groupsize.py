"""
Zero-cost within-SP group-size analysis (Phase 1c).

Loads per-clip metrics from evaluate.py --save-clips output and bins SP clips
by an approximate group-size proxy derived from the GT saliency map: the number
of distinct saliency peaks (local maxima above a threshold), which correlates
with the number of fixation targets (individuals / small groups).

Bins:
  few    — 1–2 peaks  (individual / pair: pure tracking task)
  medium — 3–5 peaks  (small group: divided-attention regime)
  many   — 6+ peaks   (large group / scene scan)

Usage:
    python scripts/analyze_sp_groupsize.py \
        --clips results/eval_clips_oracle.json \
        --data-dir /path/to/crowdfix-data \
        --splits splits.json \
        --frame-size 224 384

Outputs a table to stdout and optionally saves a JSON summary.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from scipy.ndimage import label as nd_label, maximum_filter


def count_peaks(sal_arr: np.ndarray, threshold: float = 0.3, min_dist: int = 20) -> int:
    """
    Count distinct local maxima in a 2D saliency map.

    threshold: fraction of max value; peaks below this are ignored.
    min_dist: minimum pixel separation between peaks (approx body diameter).
    """
    sal_norm = sal_arr.astype(float)
    if sal_norm.max() < 1e-6:
        return 0
    sal_norm = sal_norm / sal_norm.max()

    # Local maximum filter: a pixel is a peak iff it equals the max in its neighbourhood
    footprint = np.ones((min_dist, min_dist), dtype=bool)
    local_max = (maximum_filter(sal_norm, footprint=footprint) == sal_norm)
    peaks_mask = local_max & (sal_norm >= threshold)

    n_peaks, _ = nd_label(peaks_mask)
    return int(n_peaks)


def bin_peaks(n: int) -> str:
    if n <= 2:
        return "few (1–2)"
    if n <= 5:
        return "medium (3–5)"
    return "many (6+)"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clips", required=True, help="JSON from evaluate.py --save-clips")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--splits", default="splits.json")
    p.add_argument("--frame-size", nargs=2, type=int, default=[224, 384])
    p.add_argument("--mode", default="oracle", help="Key inside the clips JSON to use")
    p.add_argument("--out", default=None, help="Optional JSON summary output path")
    args = p.parse_args()

    h, w = args.frame_size
    data_dir = Path(args.data_dir)
    sal_root = data_dir / "SaliencyMaps"

    with open(args.clips) as f:
        clips_data = json.load(f)
    clip_records = clips_data[args.mode]  # list of {cat, nss, cc, ...}

    with open(args.splits) as f:
        splits = json.load(f)
    categories = splits.get("categories", {})
    # Build ordered list of SP test video IDs (same iteration order as dataset)
    sp_vids = [v for v in splits["test"] if categories.get(v) == "SP"]

    # Collect one peak count per SP test frame (mid-frame of each clip)
    # We iterate SP videos in dataset order and match to clip_records by position
    sp_clip_indices = [i for i, r in enumerate(clip_records) if r.get("cat") == 0]

    # Build peak count per clip using mid-frame GT saliency
    # We need to pair each SP clip record with its video's saliency maps.
    # Since clip ordering matches dataset ordering, we rebuild the clip list.
    sp_clips_ordered: list[tuple[str, int]] = []  # (vid, start_frame_idx)
    for vid in sp_vids:
        frame_dir = data_dir / "Frames" / vid
        if not frame_dir.is_dir():
            continue
        all_paths = sorted(frame_dir.glob("frame_*.jpg"))
        valid = [
            p for p in all_paths
            if (sal_root / f"{vid}_{int(p.stem.split('_')[1]):03d}.png").exists()
        ]
        clip_len = 8
        for start in range(0, len(valid) - clip_len + 1, 1):  # stride=1 as in eval
            sp_clips_ordered.append((vid, start, valid))

    if len(sp_clips_ordered) != len(sp_clip_indices):
        print(
            f"Warning: SP clip count mismatch — "
            f"reconstructed {len(sp_clips_ordered)}, found {len(sp_clip_indices)} in records. "
            f"Results may be misaligned."
        )

    bin_metrics: dict[str, defaultdict] = {
        "few (1–2)": defaultdict(list),
        "medium (3–5)": defaultdict(list),
        "many (6+)": defaultdict(list),
    }

    for idx, sp_idx in enumerate(sp_clip_indices):
        rec = clip_records[sp_idx]
        if idx >= len(sp_clips_ordered):
            break

        vid, start, valid_paths = sp_clips_ordered[idx]
        clip_len = 8
        mid = start + clip_len // 2
        fp = valid_paths[mid] if mid < len(valid_paths) else valid_paths[-1]
        frame_num = int(fp.stem.split("_")[1])
        sal_path = sal_root / f"{vid}_{frame_num:03d}.png"

        if not sal_path.exists():
            continue

        sal = Image.open(sal_path).convert("L")
        sal = TF.resize(sal, [h, w], antialias=True)
        sal_arr = np.array(sal, dtype=np.float32) / 255.0
        n_peaks = count_peaks(sal_arr)
        bname = bin_peaks(n_peaks)

        for k in ("nss", "cc", "auc_judd"):
            v = rec.get(k)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                bin_metrics[bname][k].append(v)

    # Print results
    print(f"\nSP within-category breakdown by GT saliency peak count")
    print(f"  (proxy for group size; peaks = distinct fixation targets in mid-frame)\n")
    print(f"  {'Bin':<16} {'n':>6} {'NSS':>8} {'CC':>8} {'AUC-J':>8}")
    print(f"  {'-'*50}")
    summary = {}
    for bname in ["few (1–2)", "medium (3–5)", "many (6+)"]:
        d = bin_metrics[bname]
        n = len(d.get("nss", []))
        nss_m = float(np.mean(d["nss"])) if d["nss"] else float("nan")
        cc_m = float(np.mean(d["cc"])) if d["cc"] else float("nan")
        auc_m = float(np.mean(d["auc_judd"])) if d["auc_judd"] else float("nan")
        print(f"  {bname:<16} {n:>6} {nss_m:>8.4f} {cc_m:>8.4f} {auc_m:>8.4f}")
        summary[bname] = {"n": n, "nss": nss_m, "cc": cc_m, "auc_judd": auc_m}

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved → {args.out}")


if __name__ == "__main__":
    main()
