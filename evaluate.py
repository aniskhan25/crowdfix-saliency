"""
Evaluate a trained saliency model on the CrowdFix test split.

Usage:
    python evaluate.py --checkpoint checkpoints/best.pth \
                        --data-dir /path/to/crowdfix-data \
                        --splits splits.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

from data.crowdfix_dataset import CrowdFixDataset, build_eval_transforms
from models.density_swin_saliency import DensitySwinSaliency
from models.tased_net import TASEDNet
from models.video_swin_saliency import VideoSwinSaliency
from saliency_metrics import compute_all

DENSITY_NAMES = {0: "SP", 1: "DF", 2: "DC"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--splits", default="splits.json")
    p.add_argument("--model", choices=["tased", "swin", "density_swin"], default="swin")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--clip-len", type=int, default=8)
    p.add_argument("--frame-size", nargs=2, type=int, default=[224, 384])
    return p.parse_args()


@torch.no_grad()
def run_evaluation(model, loader, device, model_name):
    model.eval()
    overall = defaultdict(list)
    per_cat: dict[int, defaultdict] = {i: defaultdict(list) for i in range(3)}

    for frames, sals, fixes, density in loader:
        frames  = frames.to(device)
        density = density.to(device)
        mid = sals.shape[1] // 2
        gt_sal = sals[:, mid].cpu().float().numpy()    # (B, 1, H, W)
        gt_fix = fixes[:, mid].cpu().float().numpy()   # (B, 1, H, W)

        with autocast("cuda", dtype=torch.bfloat16):
            if model_name == "density_swin":
                pred_t, _ = model(frames.permute(0, 2, 1, 3, 4), density)
            else:
                pred_t = model(frames.permute(0, 2, 1, 3, 4))
        pred = pred_t.float().cpu().numpy()            # (B, 1, H, W)

        for b in range(pred.shape[0]):
            m = compute_all(pred[b, 0], gt_sal[b, 0], gt_fix[b, 0])
            for k, v in m.items():
                if not np.isnan(v):
                    overall[k].append(v)
            cat = int(density[b].item())
            if cat in per_cat:
                for k, v in m.items():
                    if not np.isnan(v):
                        per_cat[cat][k].append(v)

    agg_overall = {k: float(np.mean(v)) for k, v in overall.items()}
    agg_per_cat = {
        cat: {k: float(np.mean(v)) for k, v in d.items()}
        for cat, d in per_cat.items() if d
    }
    counts = {cat: len(next(iter(d.values()))) for cat, d in per_cat.items() if d}
    return agg_overall, agg_per_cat, counts


DIRECTIONS = {"auc_judd": "↑", "auc_borji": "↑", "nss": "↑", "cc": "↑", "kldiv": "↓", "sim": "↑"}
METRICS_SHORT = ["auc_judd", "nss", "cc", "kldiv", "sim"]

BASELINES_2019 = {
    "SAM (static)": {"auc_judd": 0.777, "nss": 0.894, "cc": 0.325},
    "DeepVS":        {"auc_judd": 0.788, "nss": 0.985, "cc": 0.401},
    "ACL-Net":       {"auc_judd": 0.817, "nss": 1.250, "cc": 0.450},
}


def print_metrics(label, metrics):
    print(f"\n{label}")
    print(f"  {'Metric':<12} {'Value':>8}  {'Dir':>4}")
    print(f"  {'-'*30}")
    for k, v in metrics.items():
        print(f"  {k:<12} {v:>8.4f}  {DIRECTIONS.get(k, '?'):>4}")


def print_category_table(overall, per_cat, counts, model_name):
    cols = METRICS_SHORT
    header = f"  {'':16}" + "".join(f"{c:>10}" for c in cols)
    sep = "  " + "-" * (16 + 10 * len(cols))
    print(f"\nPer-category breakdown — {model_name.upper()}")
    print(header)
    print(sep)

    def row(label, m):
        vals = "".join(f"{m.get(c, float('nan')):>10.4f}" for c in cols)
        print(f"  {label:<16}{vals}")

    row("Overall", overall)
    for cat_id in sorted(per_cat):
        label = f"  {DENSITY_NAMES[cat_id]} (n={counts.get(cat_id, 0)})"
        row(label, per_cat[cat_id])


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == "tased":
        model = TASEDNet()
    elif args.model == "density_swin":
        model = DensitySwinSaliency(pretrained=False)
    else:
        model = VideoSwinSaliency(pretrained=False)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)

    with open(args.splits) as f:
        splits = json.load(f)

    categories = splits.get("categories", {})
    frame_size = tuple(args.frame_size)
    test_ds = CrowdFixDataset(
        args.data_dir, splits["test"], clip_len=args.clip_len,
        transform=build_eval_transforms(frame_size), stride=1,
        categories=categories,
    )
    loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=4, pin_memory=True)

    overall, per_cat, counts = run_evaluation(model, loader, device, args.model)

    label = f"Test set — {args.model.upper()} ({Path(args.checkpoint).name})"
    print_metrics(label, overall)
    print_category_table(overall, per_cat, counts, args.model)

    print(f"\n{'Model':<22} {'AUC-J':>7} {'NSS':>7} {'CC':>7}")
    print(f"  {'-'*45}")
    for name, m in BASELINES_2019.items():
        print(f"  {name:<20} {m['auc_judd']:>7.3f} {m['nss']:>7.3f} {m['cc']:>7.3f}")
    our = f"Ours ({args.model})"
    print(f"  {our:<20} {overall.get('auc_judd', float('nan')):>7.3f} "
          f"{overall.get('nss', float('nan')):>7.3f} {overall.get('cc', float('nan')):>7.3f}")


if __name__ == "__main__":
    main()
