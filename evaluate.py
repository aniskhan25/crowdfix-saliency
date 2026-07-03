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
from PIL import Image
import torchvision.transforms.functional as TF

from data.crowdfix_dataset import CrowdFixDataset, build_eval_transforms
from models.density_swin_continuous import DensitySwinContinuous
from models.density_swin_multiscale import DensitySwinMultiscale
from models.density_swin_onehot import DensitySwinOneHot
from models.density_swin_saliency import DensitySwinSaliency
from models.density_swin_soft import DensitySwinSoft
from models.tased_net import TASEDNet
from models.three_branch_saliency import ThreeBranchSaliency
from models.video_swin_saliency import VideoSwinSaliency
from saliency_metrics import compute_all

DENSITY_NAMES = {0: "SP", 1: "DF", 2: "DC"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--splits", default="splits.json")
    p.add_argument("--model", choices=["tased", "swin", "density_swin", "density_swin_onehot", "density_swin_soft", "density_swin_multiscale", "density_swin_continuous", "three_branch"], default="swin")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--clip-len", type=int, default=8)
    p.add_argument("--frame-size", nargs=2, type=int, default=[224, 384])
    p.add_argument(
        "--density-mode",
        choices=["oracle", "predicted", "both"],
        default="oracle",
        help=(
            "oracle: use GT category label (upper bound). "
            "predicted: use the model's own density_head argmax. "
            "both: run and print both."
        ),
    )
    p.add_argument(
        "--debias-map",
        default=None,
        help="Path to mean GT saliency .npy (from compute_mean_gt.py). "
             "When set, subtract this map from every model prediction before "
             "computing metrics (debiased NSS).",
    )
    p.add_argument(
        "--baseline",
        choices=["centre-bias"],
        default=None,
        help="Skip the model and evaluate a trivial baseline. "
             "'centre-bias': predict the --debias-map for every sample.",
    )
    p.add_argument(
        "--save-clips",
        default=None,
        help="Path to write per-clip metric JSON (for downstream analysis).",
    )
    return p.parse_args()


@torch.no_grad()
def run_evaluation(
    model,
    loader,
    device,
    model_name,
    density_mode="oracle",
    debias_map=None,
    use_centre_bias=False,
):
    """Run evaluation loop.

    density_mode:
      "oracle"    — use GT category label from the dataloader (upper bound).
      "predicted" — two-pass: first forward with dummy label to get logits,
                    then second forward with argmax(logits) as the label.
                    Only meaningful for density_swin / three_branch.

    debias_map: np.ndarray (H, W) or None.
      When provided, subtracted from each model prediction before metrics.
      Isolates the conditioning gain from shared centre-bias exploitation.

    use_centre_bias: bool.
      When True, ignore model output and predict debias_map for every sample.
      This is the centre-bias upper-bound baseline.
    """
    if not use_centre_bias:
        model.eval()
    overall = defaultdict(list)
    per_cat: dict[int, defaultdict] = {i: defaultdict(list) for i in range(3)}
    clip_records: list[dict] = []
    uses_density = model_name in ("density_swin", "density_swin_onehot", "density_swin_soft", "density_swin_multiscale", "density_swin_continuous", "three_branch")

    for frames, sals, fixes, density in loader:
        frames  = frames.to(device)
        density = density.to(device)
        mid = sals.shape[1] // 2
        gt_sal = sals[:, mid].cpu().float().numpy()
        gt_fix = fixes[:, mid].cpu().float().numpy()

        if use_centre_bias:
            # Tile the centre-bias map to batch size; shape (B, 1, H, W)
            cb = debias_map[None, None]  # 1×1×H×W
            pred = np.broadcast_to(cb, (frames.shape[0], 1, *debias_map.shape)).copy()
        else:
            with autocast("cuda", dtype=torch.bfloat16):
                if uses_density:
                    if density_mode == "predicted":
                        dummy = torch.zeros_like(density)
                        _, logits = model(frames.permute(0, 2, 1, 3, 4), dummy)
                        pred_density = logits.argmax(dim=1)
                        pred_t, _ = model(frames.permute(0, 2, 1, 3, 4), pred_density)
                        density = pred_density
                    else:
                        pred_t, _ = model(frames.permute(0, 2, 1, 3, 4), density)
                else:
                    pred_t = model(frames.permute(0, 2, 1, 3, 4))
            pred = pred_t.float().cpu().numpy()

            if debias_map is not None:
                # Subtract centre-bias prior; normalisation in each metric handles negatives
                pred = pred - debias_map[None, None]

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
            clip_records.append({"cat": cat, **{k: v for k, v in m.items()}})

    agg_overall = {k: float(np.mean(v)) for k, v in overall.items()}
    agg_per_cat = {
        cat: {k: float(np.mean(v)) for k, v in d.items()}
        for cat, d in per_cat.items() if d
    }
    counts = {cat: len(next(iter(d.values()))) for cat, d in per_cat.items() if d}
    return agg_overall, agg_per_cat, counts, clip_records


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

    # Load centre-bias map if provided (needed for both --debias-map and --baseline)
    debias_map = None
    if args.debias_map:
        debias_map = np.load(args.debias_map).astype(np.float32)
        print(f"Loaded debias map from {args.debias_map}  shape={debias_map.shape}")

    if args.baseline == "centre-bias":
        if debias_map is None:
            raise ValueError("--baseline centre-bias requires --debias-map")
        model = None
    else:
        if args.model == "tased":
            model = TASEDNet()
        elif args.model == "density_swin":
            model = DensitySwinSaliency(pretrained=False)
        elif args.model == "density_swin_onehot":
            model = DensitySwinOneHot(pretrained=False)
        elif args.model == "density_swin_soft":
            model = DensitySwinSoft(pretrained=False)
        elif args.model == "density_swin_multiscale":
            model = DensitySwinMultiscale(pretrained=False)
        elif args.model == "density_swin_continuous":
            model = DensitySwinContinuous(pretrained=False)
        elif args.model == "three_branch":
            model = ThreeBranchSaliency(pretrained=False)
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

    if args.baseline == "centre-bias":
        modes = ["centre-bias"]
    else:
        modes = ["oracle", "predicted"] if args.density_mode == "both" else [args.density_mode]

    results: dict[str, tuple] = {}
    all_clip_records: dict[str, list] = {}

    for mode in modes:
        overall, per_cat, counts, clip_records = run_evaluation(
            model, loader, device, args.model,
            density_mode=mode if mode != "centre-bias" else "oracle",
            debias_map=debias_map,
            use_centre_bias=(mode == "centre-bias"),
        )
        results[mode] = (overall, per_cat, counts)
        all_clip_records[mode] = clip_records
        ckpt_name = Path(args.checkpoint).name if args.checkpoint else "none"
        label = f"Test set — {args.model.upper()} ({ckpt_name})  [{mode} label]"
        if args.debias_map and mode != "centre-bias":
            label += "  [debiased]"
        print_metrics(label, overall)
        print_category_table(overall, per_cat, counts, args.model)

    if args.save_clips:
        out = {mode: records for mode, records in all_clip_records.items()}
        Path(args.save_clips).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_clips, "w") as f:
            json.dump(out, f)
        print(f"\nPer-clip results saved → {args.save_clips}")

    print(f"\n{'Model':<36} {'AUC-J':>7} {'NSS':>7} {'CC':>7}")
    print(f"  {'-'*59}")
    for name, m in BASELINES_2019.items():
        print(f"  {name:<34} {m['auc_judd']:>7.3f} {m['nss']:>7.3f} {m['cc']:>7.3f}")
    for mode, (overall, _, _) in results.items():
        label = f"Ours ({args.model}, {mode})"
        print(f"  {label:<34} {overall.get('auc_judd', float('nan')):>7.3f} "
              f"{overall.get('nss', float('nan')):>7.3f} "
              f"{overall.get('cc', float('nan')):>7.3f}")


if __name__ == "__main__":
    main()
