"""
Visualize saliency predictions overlaid on a CrowdFix video.

Produces:
  results/<video_id>/frame_NNNNN.png  —  3-panel: original | GT saliency | predicted
  results/<video_id>.mp4              —  same panels as video at 30fps

Usage:
    python visualize.py --checkpoint checkpoints/best.pth \
                        --data-dir /path/to/crowdfix-data \
                        --video-id video_001 --out results/
"""

import argparse
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
from PIL import Image

from data.crowdfix_dataset import build_eval_transforms
from models.density_swin_saliency import DensitySwinSaliency
from models.tased_net import TASEDNet
from models.three_branch_saliency import ThreeBranchSaliency
from models.video_swin_saliency import VideoSwinSaliency

_DENSITY_MODELS = {"density_swin", "three_branch"}


def to_heatmap_overlay(frame_rgb: np.ndarray, sal_map: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend a saliency map as a jet heatmap onto an RGB frame."""
    sal_u8 = ((sal_map - sal_map.min()) / (sal_map.max() - sal_map.min() + 1e-8) * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(sal_u8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    return np.clip(frame_rgb * (1 - alpha) + heatmap_rgb * alpha, 0, 255).astype(np.uint8)


@torch.no_grad()
def predict_all(model, model_name, frame_paths, clip_len, frame_size, device, density_label=0):
    """Run inference for every frame using a clip centred on that frame."""
    transform = build_eval_transforms(frame_size)
    pil_frames = [Image.open(p).convert("RGB") for p in frame_paths]
    blank = Image.new("L", pil_frames[0].size)
    half = clip_len // 2
    n = len(pil_frames)
    is_density_model = model_name in _DENSITY_MODELS
    density_t = torch.tensor([density_label], dtype=torch.long, device=device)
    preds = []
    for i in range(n):
        indices = [max(0, min(n - 1, i + j - half)) for j in range(clip_len)]
        clip = [pil_frames[j] for j in indices]
        frames_t, _, _ = transform(clip, [blank] * clip_len, [blank] * clip_len)
        x = frames_t.unsqueeze(0).to(device)               # (1, T, 3, H, W)
        if is_density_model:
            pred, _ = model(x.permute(0, 2, 1, 3, 4), density_t)
        else:
            pred = model(x.permute(0, 2, 1, 3, 4))        # (1, 1, H, W)
        preds.append(pred[0, 0].cpu().float().numpy())
    return preds


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--video-id", required=True)
    p.add_argument("--model", choices=["tased", "swin", "density_swin", "three_branch"], default="density_swin")
    p.add_argument("--density", type=int, default=0, choices=[0, 1, 2],
                   help="Density label for density_swin: 0=SP, 1=DF, 2=DC")
    p.add_argument("--out", default="results")
    p.add_argument("--clip-len", type=int, default=8)
    p.add_argument("--frame-size", nargs=2, type=int, default=[224, 384])
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--alpha", type=float, default=0.5)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    frame_size = tuple(args.frame_size)
    out_dir = Path(args.out) / args.video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.model == "tased":
        model = TASEDNet()
    elif args.model == "density_swin":
        model = DensitySwinSaliency(pretrained=False)
    elif args.model == "three_branch":
        model = ThreeBranchSaliency(pretrained=False)
    else:
        model = VideoSwinSaliency(pretrained=False)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()

    frame_dir = data_dir / "Frames" / args.video_id
    sal_dir = data_dir / "SaliencyMaps" / args.video_id
    frame_paths = sorted(frame_dir.glob("frame_*.jpg"))
    if not frame_paths:
        raise SystemExit(f"No frames found in {frame_dir}")

    print(f"Running inference on {len(frame_paths)} frames of '{args.video_id}'...")
    preds = predict_all(model, args.model, frame_paths, args.clip_len, frame_size, device, args.density)

    h, w = frame_size
    panels = []
    for fp, pred in zip(frame_paths, preds):
        frame_rgb = np.array(Image.open(fp).convert("RGB").resize((w, h)))
        sal_path = sal_dir / fp.name.replace(".jpg", ".png")
        gt = np.array(Image.open(sal_path).convert("L").resize((w, h))) if sal_path.exists() \
             else np.zeros((h, w), dtype=np.uint8)

        col_orig = frame_rgb
        col_gt = to_heatmap_overlay(frame_rgb, gt.astype(float), args.alpha)
        col_pred = to_heatmap_overlay(frame_rgb, pred, args.alpha)
        panel = np.concatenate([col_orig, col_gt, col_pred], axis=1)
        panels.append(panel)
        Image.fromarray(panel).save(out_dir / fp.name.replace(".jpg", ".png"))

    video_path = Path(args.out) / f"{args.video_id}.mp4"
    writer = imageio.get_writer(str(video_path), fps=args.fps, format="FFMPEG", codec="libx264")
    for panel in panels:
        writer.append_data(panel)
    writer.close()

    print(f"Saved {len(panels)} frame images to {out_dir}/")
    print(f"Saved video to {video_path}")


if __name__ == "__main__":
    main()
