"""
CrowdFix PyTorch Dataset.

Expects this directory structure:
  <data_dir>/Frames/<video_id>/frame_NNNNN.jpg   (created by extract_frames.py)
  <data_dir>/SaliencyMaps/<video_id>_NNN.png     (CrowdFix native flat layout)
  <data_dir>/BinaryMaps/<video_id>_NNN.png       (CrowdFix native flat layout)

Saliency/fixation maps use 3-digit 1-based frame indices; extracted frames use
5-digit 1-based indices (frame_00001.jpg ↔ 001_001.png).

Returns clips of (clip_len) consecutive frames per sample:
  frames:        FloatTensor  [T, 3, H, W]  ImageNet-normalized
  sal_maps:      FloatTensor  [T, 1, H, W]  in [0, 1]
  fix_maps:      FloatTensor  [T, 1, H, W]  binary {0, 1}
  density_label: LongTensor   []            SP=0, DF=1, DC=2; -1 if unknown
"""

import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _load_frame(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _load_map(path: Path) -> Image.Image:
    return Image.open(path).convert("L")


def build_train_transforms(frame_size: tuple[int, int]):
    """Returns a function that applies random augmentations to a clip."""
    h, w = frame_size

    def transform(frames, sal_maps, fix_maps):
        # Sample augmentation params once per clip so temporal consistency is preserved
        do_flip  = random.random() > 0.5
        bright   = 1.0 + random.uniform(-0.2, 0.2)
        contrast = 1.0 + random.uniform(-0.2, 0.2)

        out_frames, out_sal, out_fix = [], [], []
        for frame, sal, fix in zip(frames, sal_maps, fix_maps):
            frame = TF.resize(frame, [h, w], antialias=True)
            sal   = TF.resize(sal,   [h, w], antialias=True)
            fix   = TF.resize(fix,   [h, w], antialias=True)
            if do_flip:
                frame, sal, fix = TF.hflip(frame), TF.hflip(sal), TF.hflip(fix)
            frame = TF.adjust_brightness(frame, bright)
            frame = TF.adjust_contrast(frame, contrast)
            out_frames.append(TF.to_tensor(frame))
            out_sal.append(TF.to_tensor(sal))
            out_fix.append(TF.to_tensor(fix))

        frames_t = torch.stack(out_frames)  # T×3×H×W
        frames_t = (frames_t - _IMAGENET_MEAN) / _IMAGENET_STD
        return frames_t, torch.stack(out_sal), torch.stack(out_fix)

    return transform


def build_eval_transforms(frame_size: tuple[int, int]):
    h, w = frame_size

    def transform(frames, sal_maps, fix_maps):
        out_frames, out_sal, out_fix = [], [], []
        for frame, sal, fix in zip(frames, sal_maps, fix_maps):
            frame = TF.resize(frame, [h, w], antialias=True)
            sal = TF.resize(sal, [h, w], antialias=True)
            fix = TF.resize(fix, [h, w], antialias=True)
            out_frames.append(TF.to_tensor(frame))
            out_sal.append(TF.to_tensor(sal))
            out_fix.append(TF.to_tensor(fix))

        frames_t = torch.stack(out_frames)
        frames_t = (frames_t - _IMAGENET_MEAN) / _IMAGENET_STD
        return frames_t, torch.stack(out_sal), torch.stack(out_fix)

    return transform


_DENSITY_MAP: dict[str, int] = {"SP": 0, "DF": 1, "DC": 2}


class CrowdFixDataset(Dataset):
    """
    Each item is a clip of `clip_len` consecutive frames from one video.
    Clips are non-overlapping by default (stride = clip_len).
    Set stride=1 for dense evaluation.

    Pass categories={video_id: "SP"|"DF"|"DC"} (from splits.json["categories"])
    to enable density conditioning.  Videos absent from the dict return label -1.
    """

    def __init__(
        self,
        data_dir: str | Path,
        video_ids: list[str],
        clip_len: int = 8,
        stride: int | None = None,
        transform=None,
        categories: dict[str, str] | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.clip_len = clip_len
        self.stride = stride if stride is not None else clip_len
        self.transform = transform
        self.categories = categories or {}
        self.clips: list[tuple[str, int]] = []      # (video_id, start_frame_idx)
        self._frame_paths: dict[str, list[Path]] = {}  # cached per-video frame lists

        sal_root = self.data_dir / "SaliencyMaps"  # flat: sal_root/{vid}_{N:03d}.png

        for vid in video_ids:
            frame_dir = self.data_dir / "Frames" / vid
            if not frame_dir.is_dir():
                continue
            all_paths = sorted(frame_dir.glob("frame_*.jpg"))
            # Flat saliency layout: SaliencyMaps/001_001.png  (1-based, 3-digit)
            valid = [
                p for p in all_paths
                if (sal_root / f"{vid}_{int(p.stem.split('_')[1]):03d}.png").exists()
            ]
            if not valid:
                continue
            self._frame_paths[vid] = valid
            n = len(valid)
            for start in range(0, n - clip_len + 1, self.stride):
                self.clips.append((vid, start))

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int):
        vid, start = self.clips[idx]
        sal_root = self.data_dir / "SaliencyMaps"  # flat
        fix_root = self.data_dir / "BinaryMaps"    # flat

        frame_paths = self._frame_paths[vid][start : start + self.clip_len]

        frames, sals, fixes = [], [], []
        for fp in frame_paths:
            frame_num = int(fp.stem.split("_")[1])  # frame_00042 → 42
            sal_name = f"{vid}_{frame_num:03d}.png"
            frames.append(_load_frame(fp))
            sals.append(_load_map(sal_root / sal_name))
            fix_path = fix_root / sal_name
            fixes.append(_load_map(fix_path) if fix_path.exists() else Image.new("L", frames[-1].size))

        if self.transform:
            frames_t, sals_t, fixes_t = self.transform(frames, sals, fixes)
        else:
            frames_t = torch.stack([TF.to_tensor(f) for f in frames])
            sals_t = torch.stack([TF.to_tensor(s) for s in sals])
            fixes_t = torch.stack([TF.to_tensor(f) for f in fixes])

        cat_str = self.categories.get(vid, "")
        density_label = torch.tensor(_DENSITY_MAP.get(cat_str, -1), dtype=torch.long)

        return frames_t, sals_t, fixes_t, density_label
