"""
DDP training for CrowdFix video saliency on LUMI.

Launch:
    torchrun --standalone --nnodes=1 --nproc_per_node=8 train.py --data-dir $DATA_DIR --splits splits.json
"""

import os
import argparse
import json
from pathlib import Path

import psutil
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.amp import autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data.crowdfix_dataset import CrowdFixDataset, build_eval_transforms, build_train_transforms
from models.density_swin_saliency import DensitySwinSaliency
from models.tased_net import TASEDNet
from models.video_swin_saliency import VideoSwinSaliency

# Each GCD maps to its nearest CPU cores on a LUMI-G node.
# Cores 0, 8, 16, 24, 32, 40, 48, 56 are reserved for the OS.
LUMI_GPU_CPU_MAP = {
    0: [49, 50, 51, 52, 53, 54, 55], 1: [57, 58, 59, 60, 61, 62, 63],
    2: [17, 18, 19, 20, 21, 22, 23], 3: [25, 26, 27, 28, 29, 30, 31],
    4: [1, 2, 3, 4, 5, 6, 7],        5: [9, 10, 11, 12, 13, 14, 15],
    6: [33, 34, 35, 36, 37, 38, 39], 7: [41, 42, 43, 44, 45, 46, 47],
}

# Backbone sub-module names in DensitySwinSaliency v1
_BACKBONE_KEYS = {"patch_embed", "pos_drop", "encoder", "norm"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["tased", "swin", "density_swin"], default="swin")
    p.add_argument("--data-dir", required=True, help="Root of crowdfix-data/")
    p.add_argument("--splits", default="splits.json")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=2, help="Per-GCD batch size")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--clip-len", type=int, default=8)
    p.add_argument("--frame-size", nargs=2, type=int, default=[224, 384])
    p.add_argument("--resume", default=None)
    p.add_argument("--smoke-test", action="store_true", help="Run 2 steps then exit")
    p.add_argument("--freeze-epochs", type=int, default=0,
                   help="Freeze Swin backbone for this many epochs before fine-tuning")
    p.add_argument("--backbone-lr-scale", type=float, default=0.1,
                   help="Backbone LR multiplier after unfreezing (relative to --lr)")
    p.add_argument("--early-stop-patience", type=int, default=0,
                   help="Stop if val loss does not improve for this many epochs (0=disabled)")
    return p.parse_args()


def kl_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    B = pred.shape[0]
    p = pred.view(B, -1) + 1e-8
    g = gt.view(B, -1) + 1e-8
    p = p / p.sum(1, keepdim=True)
    g = g / g.sum(1, keepdim=True)
    return (g * (g / p).log()).sum(1).mean()


def cc_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    B = pred.shape[0]
    p = pred.view(B, -1)
    g = gt.view(B, -1)
    p = p - p.mean(1, keepdim=True)
    g = g - g.mean(1, keepdim=True)
    num = (p * g).sum(1)
    denom = (p.pow(2).sum(1) * g.pow(2).sum(1)).sqrt() + 1e-8
    return (1.0 - num / denom).mean()


def saliency_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return kl_loss(pred, gt) + 0.2 * cc_loss(pred, gt)


AUX_LOSS_WEIGHT = 0.1


def train_epoch(model, loader, optimizer, device, model_name, max_steps=None):
    model.train()
    total, count = 0.0, 0
    for step, (frames, sals, _, density) in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        frames  = frames.to(device)
        gt      = sals[:, sals.shape[1] // 2].to(device)
        density = density.to(device)
        optimizer.zero_grad()
        with autocast("cuda", dtype=torch.bfloat16):
            if model_name == "density_swin":
                pred, logits = model(frames.permute(0, 2, 1, 3, 4), density)
                loss = saliency_loss(pred, gt) + AUX_LOSS_WEIGHT * F.cross_entropy(
                    logits, density, ignore_index=-1
                )
            else:
                pred = model(frames.permute(0, 2, 1, 3, 4))
                loss = saliency_loss(pred, gt)
        loss.backward()
        optimizer.step()
        total += loss.item()
        count += 1
    return total / max(count, 1)


@torch.no_grad()
def validate(model, loader, device, model_name):
    model.eval()
    total, count = 0.0, 0
    for frames, sals, _, density in loader:
        frames  = frames.to(device)
        gt      = sals[:, sals.shape[1] // 2].to(device)
        density = density.to(device)
        with autocast("cuda", dtype=torch.bfloat16):
            if model_name == "density_swin":
                pred, logits = model(frames.permute(0, 2, 1, 3, 4), density)
                loss = saliency_loss(pred, gt) + AUX_LOSS_WEIGHT * F.cross_entropy(
                    logits, density, ignore_index=-1
                )
            else:
                pred = model(frames.permute(0, 2, 1, 3, 4))
                loss = saliency_loss(pred, gt)
            total += loss.item()
        count += 1
    t = torch.tensor([total, float(count)], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t[0] / t[1].clamp(min=1)).item()


def main():
    args = parse_args()

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])

    # Give each rank its own MIOpen cache dir on /tmp to avoid SQLite contention.
    # Must be set before the first CUDA op so MIOpen picks it up on init.
    for _var in ("MIOPEN_USER_DB_PATH", "MIOPEN_CUSTOM_CACHE_DIR"):
        base = os.environ.get(_var, f"/tmp/miopen_{os.getenv('SLURM_JOB_ID', 'local')}")
        per_rank = f"{base}_rank{local_rank}"
        os.makedirs(per_rank, exist_ok=True)
        os.environ[_var] = per_rank

    torch.cuda.set_device(local_rank)
    psutil.Process().cpu_affinity(LUMI_GPU_CPU_MAP[local_rank])
    device = torch.device("cuda", local_rank)

    with open(args.splits) as f:
        splits = json.load(f)

    categories = splits.get("categories", {})
    frame_size = tuple(args.frame_size)
    train_ds = CrowdFixDataset(
        args.data_dir, splits["train"], clip_len=args.clip_len,
        transform=build_train_transforms(frame_size), categories=categories,
    )
    val_ds = CrowdFixDataset(
        args.data_dir, splits["val"], clip_len=args.clip_len,
        transform=build_eval_transforms(frame_size), categories=categories,
    )
    train_loader = DataLoader(
        train_ds, sampler=DistributedSampler(train_ds),
        batch_size=args.batch_size, num_workers=7, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, sampler=DistributedSampler(val_ds, shuffle=False),
        batch_size=args.batch_size, num_workers=7, pin_memory=True,
    )

    if args.model == "tased":
        model = TASEDNet()
    elif args.model == "density_swin":
        model = DensitySwinSaliency()
    else:
        model = VideoSwinSaliency()
    model = model.to(device)

    # Split backbone vs head for differential LR
    backbone_params = [p for n, p in model.named_parameters()
                       if n.split(".")[0] in _BACKBONE_KEYS]
    head_params     = [p for n, p in model.named_parameters()
                       if n.split(".")[0] not in _BACKBONE_KEYS]

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * args.backbone_lr_scale},
        {"params": head_params,     "lr": args.lr},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    if args.freeze_epochs > 0:
        for p in backbone_params:
            p.requires_grad_(False)

    model = DistributedDataParallel(model, device_ids=[local_rank])

    start_epoch, best_val = 0, float("inf")
    no_improve = 0  # epochs since last val improvement
    ckpt_dir = Path(args.checkpoint_dir)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.module.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt.get("best_val", float("inf"))

    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(f"Model: {args.model}  Epochs: {args.epochs}  "
              f"Batch/GCD: {args.batch_size}  LR: {args.lr}")
        print(f"Dataset: {len(train_ds)} train clips / {len(val_ds)} val clips")

    max_steps = 2 if args.smoke_test else None

    for epoch in range(start_epoch, args.epochs):
        if args.freeze_epochs > 0 and epoch == args.freeze_epochs:
            for p in backbone_params:
                p.requires_grad_(True)
            if rank == 0:
                print(f"Epoch {epoch+1}: backbone unfrozen (lr={args.lr * args.backbone_lr_scale:.2e})")

        train_loader.sampler.set_epoch(epoch)
        train_loss = train_epoch(model, train_loader, optimizer, device, args.model, max_steps)
        val_loss   = validate(model, val_loader, device, args.model)
        scheduler.step()

        improved = val_loss < best_val
        if improved:
            best_val  = val_loss
            no_improve = 0
        else:
            no_improve += 1

        if rank == 0:
            print(f"Epoch {epoch+1}/{args.epochs}  train={train_loss:.4f}  val={val_loss:.4f}"
                  + ("  *" if improved else ""))
            state = {
                "epoch": epoch,
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val": best_val,
            }
            torch.save(state, ckpt_dir / "latest.pth")
            if improved:
                torch.save(state, ckpt_dir / "best.pth")

        # Broadcast early-stop decision from rank 0 to all ranks
        stop = torch.tensor(
            int(args.early_stop_patience > 0 and no_improve >= args.early_stop_patience),
            device=device,
        )
        dist.broadcast(stop, src=0)
        if stop.item():
            if rank == 0:
                print(f"Early stop: no improvement for {args.early_stop_patience} epochs.")
            break

        if args.smoke_test:
            break

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
