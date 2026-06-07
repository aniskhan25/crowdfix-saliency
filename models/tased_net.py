"""
TASED-Net: Temporally-Aggregating Spatial Encoder-Decoder Network.
Simplified implementation inspired by Min et al., ICCV 2019.

Input:  (B, 3, T, H, W)  — video clip, ImageNet-normalized frames
Output: (B, 1, H, W)     — saliency map in [0, 1]

The encoder uses separable 3D convolutions (spatial 1×k×k + temporal k×1×1)
to progressively reduce spatial resolution while maintaining temporal context.
Temporal dimension is pooled via learned aggregation before the 2D decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SepConv3d(nn.Module):
    """Separable 3D conv: spatial (1,k,k) then temporal (k,1,1), each with BN+ReLU."""

    def __init__(self, in_ch: int, out_ch: int, k: int, stride=(1, 1, 1), pad=(0, 0, 0)):
        super().__init__()
        st, sh, sw = stride
        pt, ph, pw = pad
        self.spatial = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, (1, k, k), stride=(1, sh, sw), padding=(0, ph, pw), bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.temporal = nn.Sequential(
            nn.Conv3d(out_ch, out_ch, (k, 1, 1), stride=(st, 1, 1), padding=(pt, 0, 0), bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.temporal(self.spatial(x))


class UpBlock2d(nn.Module):
    """2× spatial upsample, optional skip concatenation, then 3×3 conv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class TASEDNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder: 4 stages of increasing channels with spatial 2× downsampling each
        # Stage 1: H → H/2 (stem: conv stride 2 + maxpool stride 2) → H/4 total
        self.stem = nn.Sequential(
            nn.Conv3d(3, 64, 7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        # Stage 2: H/4 → H/8 (spatial stride 2, no temporal downsampling yet)
        self.stage2 = SepConv3d(64, 128, k=3, stride=(1, 2, 2), pad=(1, 1, 1))
        # Stage 3: H/8 → H/16 (spatial + temporal stride 2)
        self.stage3 = SepConv3d(128, 256, k=3, stride=(2, 2, 2), pad=(1, 1, 1))
        # Stage 4: H/16 → H/32 (spatial + temporal stride 2)
        self.stage4 = SepConv3d(256, 512, k=3, stride=(2, 2, 2), pad=(1, 1, 1))

        # Pool temporal dimension from each stage (used as skip connections for decoder)
        self.tpool = nn.AdaptiveAvgPool3d((1, None, None))

        # Decoder: skip connections from stages 1–3
        # f4 (B, 512, H/32, W/32) + f3 skip (B, 256, H/16, W/16) → (B, 256, H/16, W/16)
        self.up4 = UpBlock2d(512, 256, 256)
        # (B, 256, H/16, W/16) + f2 skip (B, 128, H/8, W/8) → (B, 128, H/8, W/8)
        self.up3 = UpBlock2d(256, 128, 128)
        # (B, 128, H/8, W/8) + f1 skip (B, 64, H/4, W/4) → (B, 64, H/4, W/4)
        self.up2 = UpBlock2d(128, 64, 64)
        # (B, 64, H/4, W/4) → (B, 32, H/2, W/2)
        self.up1 = UpBlock2d(64, 0, 32)
        # (B, 32, H/2, W/2) → (B, 16, H, W)
        self.up0 = UpBlock2d(32, 0, 16)

        self.head = nn.Sequential(nn.Conv2d(16, 1, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, T, H, W)
        H, W = x.shape[-2], x.shape[-1]

        s1 = self.stem(x)    # (B, 64,  T,   H/4,  W/4)
        s2 = self.stage2(s1) # (B, 128, T,   H/8,  W/8)
        s3 = self.stage3(s2) # (B, 256, T/2, H/16, W/16)
        s4 = self.stage4(s3) # (B, 512, T/4, H/32, W/32)

        # Temporally pool each encoder feature map
        f4 = self.tpool(s4).squeeze(2)  # (B, 512, H/32, W/32)
        f3 = self.tpool(s3).squeeze(2)  # (B, 256, H/16, W/16)
        f2 = self.tpool(s2).squeeze(2)  # (B, 128, H/8,  W/8)
        f1 = self.tpool(s1).squeeze(2)  # (B, 64,  H/4,  W/4)

        x = self.up4(f4, f3)  # (B, 256, H/16, W/16)
        x = self.up3(x, f2)   # (B, 128, H/8,  W/8)
        x = self.up2(x, f1)   # (B, 64,  H/4,  W/4)
        x = self.up1(x)       # (B, 32,  H/2,  W/2)
        x = self.up0(x)       # (B, 16,  H,    W)

        out = self.head(x)    # (B, 1, H, W)
        # Ensure exact spatial match in case of odd dimensions
        if out.shape[-2:] != (H, W):
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        return out
