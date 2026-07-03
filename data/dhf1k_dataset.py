"""
DHF1K PyTorch Dataset for pretraining the VideoSwin saliency backbone.

Expected directory layout after download + extraction:
  <data_dir>/
    training/
      0001/
        images/  — JPEG frames  (0001.jpg, 0002.jpg, ...)
        maps/    — PNG saliency  (0001.png, 0002.png, ...)
        fixation/ — PNG binary   (0001.png, 0002.png, ...)
      0002/ ...
    validation/
      0601/ ... 0700/

Reference: "A Deep Spatial Contextual Long-term Recurrent Convolutional Network
for Saliency Detection", Wang et al., IEEE TNNLS 2019. Dataset: 600 train /
100 val clips; no density labels (returns density=-1 for compatibility).

Returns clips identical in format to CrowdFixDataset:
  frames:   FloatTensor [T, 3, H, W]  ImageNet-normalized
  sal_maps: FloatTensor [T, 1, H, W]  in [0, 1]
  fix_maps: FloatTensor [T, 1, H, W]  binary {0, 1}
  density:  LongTensor  []            always -1
"""

import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from data.crowdfix_dataset import build_train_transforms, build_eval_transforms

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _find_frames(video_dir: Path) -> list[Path]:
    imgs = sorted((video_dir / "images").glob("*.jpg"))
    if not imgs:
        imgs = sorted((video_dir / "images").glob("*.png"))
    return imgs


class DHF1KDataset(Dataset):
    """Clips from DHF1K training or validation split."""

    def __init__(
        self,
        data_dir: str,
        split: str = "training",       # "training" | "validation"
        clip_len: int = 8,
        stride: int = 4,
        transform=None,
    ):
        super().__init__()
        self.data_dir  = Path(data_dir)
        self.clip_len  = clip_len
        self.transform = transform

        split_dir = self.data_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"DHF1K split directory not found: {split_dir}")

        self.clips: list[tuple[Path, int]] = []  # (video_dir, start_frame_idx)
        for vid_dir in sorted(split_dir.iterdir()):
            if not vid_dir.is_dir():
                continue
            frames = _find_frames(vid_dir)
            if len(frames) < clip_len:
                continue
            for start in range(0, len(frames) - clip_len + 1, stride):
                self.clips.append((vid_dir, start, frames))

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int):
        vid_dir, start, frame_paths = self.clips[idx]
        clip_paths = frame_paths[start : start + self.clip_len]

        frames, sal_maps, fix_maps = [], [], []
        for fp in clip_paths:
            stem = fp.stem
            frames.append(Image.open(fp).convert("RGB"))

            sal_path = vid_dir / "maps"     / f"{stem}.png"
            fix_path = vid_dir / "fixation" / f"{stem}.png"

            sal_maps.append(
                Image.open(sal_path).convert("L") if sal_path.exists()
                else Image.fromarray(np.zeros((frames[-1].height, frames[-1].width), dtype=np.uint8))
            )
            fix_maps.append(
                Image.open(fix_path).convert("L") if fix_path.exists()
                else Image.fromarray(np.zeros((frames[-1].height, frames[-1].width), dtype=np.uint8))
            )

        if self.transform is not None:
            frames, sal_maps, fix_maps = self.transform(frames, sal_maps, fix_maps)
        else:
            h, w = frames[0].height, frames[0].width
            frames   = [TF.to_tensor(f) for f in frames]
            sal_maps = [TF.to_tensor(s) for s in sal_maps]
            fix_maps = [TF.to_tensor(x) for x in fix_maps]

        frames   = torch.stack(frames)                  # (T, 3, H, W)
        sal_maps = torch.stack(sal_maps)                # (T, 1, H, W)
        fix_maps = torch.stack(fix_maps).float()        # (T, 1, H, W)

        frames = (frames - _IMAGENET_MEAN) / _IMAGENET_STD
        sal_maps = sal_maps / (sal_maps.amax() + 1e-8)

        density = torch.tensor(-1, dtype=torch.long)    # unknown
        return frames, sal_maps, fix_maps, density


def build_dhf1k_splits(data_dir: str) -> dict:
    """Return a minimal splits-like dict with train/val lists of video ids."""
    root = Path(data_dir)
    train_ids = sorted(d.name for d in (root / "training").iterdir() if d.is_dir())
    val_ids   = sorted(d.name for d in (root / "validation").iterdir() if d.is_dir())
    return {"train": train_ids, "val": val_ids}
