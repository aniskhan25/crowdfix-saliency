"""
Density-Aware One-Hot Baseline — ablation counterpart to DensitySwinSaliency.

Instead of a learned 64-dim embedding + FiLM (γ/β scale-shift), this model
injects density information as a direct one-hot-coded additive bias:
  bias = Linear(3, C)(one_hot(density))   # shape (B, C)
  f    = f + bias.unsqueeze(-1).unsqueeze(-1)

This is equivalent to FiLM with γ=0 (no channel scaling) and the β source
being a 3×C weight matrix with no intermediate representation.  If DensitySwin
outperforms this baseline, the gain from FiLM is attributable to the learned
64-dim embedding geometry and the multiplicative γ pathway, not merely the
presence of the density conditioning signal.

Architecture and decoder are otherwise identical to DensitySwinSaliency v1.
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


class DensitySwinOneHot(nn.Module):
    """Video Swin encoder + one-hot density bias injection + 5-stage decoder."""

    NUM_DENSITY = 3
    BOTTLENECK  = 768

    def __init__(self, pretrained: bool = True, dropout: float = 0.5):
        super().__init__()

        from torchvision.models.video import swin3d_s, Swin3D_S_Weights
        weights = Swin3D_S_Weights.KINETICS400_V1 if pretrained else None
        swin = swin3d_s(weights=weights)

        self.patch_embed = swin.patch_embed
        self.pos_drop    = swin.pos_drop
        self.encoder     = swin.features
        self.norm        = swin.norm

        # One-hot density bias: 3-class input → 768 additive bias per channel
        self.density_bias   = nn.Linear(self.NUM_DENSITY, self.BOTTLENECK, bias=False)
        self.density_head   = nn.Linear(self.BOTTLENECK, self.NUM_DENSITY)
        self.bottleneck_drop = nn.Dropout2d(p=dropout)

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

        f = f.permute(0, 4, 1, 2, 3).mean(dim=2)      # (B, 768, H/32, W/32)

        logits = self.density_head(f.mean(dim=(-2, -1)))  # (B, 3)

        f = self.bottleneck_drop(f)

        # One-hot encode and project to a spatial bias
        one_hot = F.one_hot(density.clamp(min=0), num_classes=self.NUM_DENSITY).float()
        bias = self.density_bias(one_hot)              # (B, 768)
        f = f + bias.unsqueeze(-1).unsqueeze(-1)       # broadcast over H', W'

        f = self.dec1(f)
        f = self.dec2(f)
        f = self.dec3(f)
        f = self.dec4(f)
        f = self.dec5(f)

        out = self.head(f)
        if out.shape[-2:] != (H, W):
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        return out, logits
