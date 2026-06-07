"""
Video Swin Transformer saliency model.

Uses torchvision's swin3d_s backbone (pretrained on Kinetics-400) as encoder.
Temporal features are mean-pooled before a 5-stage 2D FPN decoder.

Input:  (B, 3, T, H, W)  — video clip, ImageNet-normalized frames
Output: (B, 1, H, W)     — saliency map in [0, 1]

Requires torchvision >= 0.15 (for Swin3D).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UpBlock2d(nn.Module):
    """2× spatial upsample then 3×3 conv."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.conv(x)


class VideoSwinEncoder(nn.Module):
    """
    Wraps torchvision swin3d_s, returning the final spatial feature map
    after temporal mean pooling: (B, 768, H/32, W/32).
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        from torchvision.models.video import swin3d_s, Swin3D_S_Weights

        weights = Swin3D_S_Weights.KINETICS400_V1 if pretrained else None
        swin = swin3d_s(weights=weights)

        # Extract components before the classification head
        self.patch_embed = swin.patch_embed
        self.pos_drop = swin.pos_drop
        self.features = swin.features   # torchvision >= 0.16 renamed layers → features
        self.norm = swin.norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, T, H, W)
        # torchvision >= 0.16: outputs channels-last (B, T', H', W', C)
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        x = self.features(x)
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3)  # → (B, C, T', H', W')
        return x.mean(dim=2)           # temporal mean pool → (B, 768, H/32, W/32)


class VideoSwinSaliency(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.encoder = VideoSwinEncoder(pretrained=pretrained)

        # Decoder: 5 × 2× upsample to go from H/32 back to H
        self.decoder = nn.Sequential(
            UpBlock2d(768, 256),
            UpBlock2d(256, 128),
            UpBlock2d(128, 64),
            UpBlock2d(64, 32),
            UpBlock2d(32, 16),
        )
        self.head = nn.Sequential(nn.Conv2d(16, 1, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2], x.shape[-1]
        feat = self.encoder(x)        # (B, 768, H/32, W/32)
        feat = self.decoder(feat)     # (B, 16, H, W) — approx
        out = self.head(feat)         # (B, 1, H, W)
        # Exact resize to input spatial dims to handle rounding
        if out.shape[-2:] != (H, W):
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        return out
