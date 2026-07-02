"""
Aggregate evaluate.py oracle results across multiple seed checkpoints
to produce mean ± std for the paper's Table 1.

Usage (run from repo root after copying seed checkpoints locally):
    python scripts/aggregate_seeds.py \
        --checkpoints checkpoints/best.pth \
                      checkpoints/seed_0/best.pth \
                      checkpoints/seed_1/best.pth \
                      checkpoints/seed_2/best.pth \
        --data-dir /path/to/crowdfix-data \
        --splits   splits.json

Outputs a LaTeX-formatted table row plus raw per-seed numbers.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


METRICS = ["auc_judd", "nss", "cc"]


def run_evaluate(checkpoint: str, data_dir: str, splits: str) -> dict[str, float]:
    """Run evaluate.py for one checkpoint and parse the summary line."""
    cmd = [
        sys.executable, "evaluate.py",
        "--checkpoint", checkpoint,
        "--data-dir",   data_dir,
        "--splits",     splits,
        "--model",      "density_swin",
        "--density-mode", "oracle",
        "--batch-size", "8",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    # Parse the final comparison table — look for "Ours" line
    metrics: dict[str, float] = {}
    for line in result.stdout.splitlines():
        if "Ours" in line and "oracle" in line:
            parts = line.split()
            # format: "  Ours (density_swin, oracle)   0.820   1.437   0.517"
            try:
                nums = [float(p) for p in parts if _is_float(p)]
                if len(nums) >= 3:
                    metrics["auc_judd"] = nums[0]
                    metrics["nss"]      = nums[1]
                    metrics["cc"]       = nums[2]
            except (ValueError, IndexError):
                pass
    if not metrics:
        raise RuntimeError(f"Could not parse metrics from evaluate.py output:\n{result.stdout}")
    return metrics


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="Paths to best.pth files from each seed run")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--splits", default="splits.json")
    return p.parse_args()


def main():
    args = parse_args()
    all_results: list[dict[str, float]] = []

    print(f"Running evaluation for {len(args.checkpoints)} seed(s)...\n")
    for ckpt in args.checkpoints:
        print(f"  {ckpt} ... ", end="", flush=True)
        m = run_evaluate(ckpt, args.data_dir, args.splits)
        all_results.append(m)
        print(f"AUC-J={m['auc_judd']:.3f}  NSS={m['nss']:.3f}  CC={m['cc']:.3f}")

    print()
    print(f"{'Metric':<12} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("  " + "-" * 46)
    for metric in METRICS:
        vals = np.array([r[metric] for r in all_results])
        print(f"  {metric:<10} {vals.mean():>8.4f} {vals.std():>8.4f} "
              f"{vals.min():>8.4f} {vals.max():>8.4f}")

    # LaTeX table row
    aucs = np.array([r["auc_judd"] for r in all_results])
    nsss = np.array([r["nss"]      for r in all_results])
    ccs  = np.array([r["cc"]       for r in all_results])
    n = len(all_results)
    print(f"\nLaTeX row (n={n} seeds):")
    print(r"    \textbf{\ourmodel{} (ours)}")
    print(f"      & ${aucs.mean():.3f} \\pm {aucs.std():.3f}$"
          f" & ${nsss.mean():.3f} \\pm {nsss.std():.3f}$"
          f" & ${ccs.mean():.3f} \\pm {ccs.std():.3f}$ \\\\")


if __name__ == "__main__":
    main()
