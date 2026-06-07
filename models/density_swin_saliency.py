"""
Density-Aware FiLM Saliency Model — v1.

Architecture:
  Input:   x       (B, 3, T, H, W)  — video clip, ImageNet-normalised
           density (B,) int64        — SP=0, DF=1, DC=2; -1=unknown
  Output:  saliency (B, 1, H, W)    — in [0, 1]
           logits   (B, 3)           — raw density class scores
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UpBlock2d(nn.Module):
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


class FiLMLayer(nn.Module):
    """Projects a density embedding to per-channel scale γ and shift β."""

    def __init__(self, embed_dim: int, n_channels: int):
        super().__init__()
        self.proj = nn.Linear(embed_dim, 2 * n_channels)

    def forward(self, feat: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.proj(emb).chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta  = beta.unsqueeze(-1).unsqueeze(-1)
        return feat * (1.0 + gamma) + beta


class DensitySwinSaliency(nn.Module):
    """Video Swin encoder + bottleneck FiLM conditioning + 5-stage decoder."""

    NUM_DENSITY = 3
    EMBED_DIM   = 64
    BOTTLENECK  = 768  # swin3d_s final channel count

    def __init__(self, pretrained: bool = True, dropout: float = 0.5):
        super().__init__()

        from torchvision.models.video import swin3d_s, Swin3D_S_Weights
        weights = Swin3D_S_Weights.KINETICS400_V1 if pretrained else None
        swin = swin3d_s(weights=weights)

        self.patch_embed = swin.patch_embed
        self.pos_drop    = swin.pos_drop
        self.encoder     = swin.features
        self.norm        = swin.norm

        # Density conditioning
        self.density_emb    = nn.Embedding(self.NUM_DENSITY, self.EMBED_DIM)
        self.film           = FiLMLayer(self.EMBED_DIM, self.BOTTLENECK)
        self.density_head   = nn.Linear(self.BOTTLENECK, self.NUM_DENSITY)
        self.bottleneck_drop = nn.Dropout2d(p=dropout)

        # 5-stage decoder
        self.dec1 = UpBlock2d(768, 256)
        self.dec2 = UpBlock2d(256, 128)
        self.dec3 = UpBlock2d(128, 64)
        self.dec4 = UpBlock2d(64, 32)
        self.dec5 = UpBlock2d(32, 16)

        self.head = nn.Sequential(nn.Conv2d(16, 1, 1), nn.Sigmoid())

    def forward(
        self, x: torch.Tensor, density: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        H, W = x.shape[-2], x.shape[-1]

        f = self.patch_embed(x)
        f = self.pos_drop(f)
        f = self.encoder(f)
        f = self.norm(f)                               # (B, T', H/32, W/32, 768)

        # channels-last → channels-first + temporal mean pool
        f = f.permute(0, 4, 1, 2, 3).mean(dim=2)     # (B, 768, H/32, W/32)

        logits = self.density_head(f.mean(dim=(-2, -1)))  # (B, 3)

        f   = self.bottleneck_drop(f)
        emb = self.density_emb(density.clamp(min=0))
        f   = self.film(f, emb)

        f = self.dec1(f)
        f = self.dec2(f)
        f = self.dec3(f)
        f = self.dec4(f)
        f = self.dec5(f)

        out = self.head(f)
        if out.shape[-2:] != (H, W):
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        return out, logits
