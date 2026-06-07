"""
Density-Aware FiLM Saliency Model — v2.

Improvements over v1:
  - Encoder split into individual stage sub-modules to expose skip features
  - Skip connections from encoder stages 2 (192ch) and 4 (384ch) into decoder
  - Multi-scale FiLM: density conditioning applied after every decoder stage,
    not only at the bottleneck
  - Bottleneck Dropout2d retained for regularisation

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
    """Video Swin encoder + skip connections + multi-scale FiLM decoder."""

    NUM_DENSITY = 3
    EMBED_DIM   = 64
    BOTTLENECK  = 768  # swin3d_s final channel count

    def __init__(self, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()

        from torchvision.models.video import swin3d_s, Swin3D_S_Weights
        weights = Swin3D_S_Weights.KINETICS400_V1 if pretrained else None
        swin = swin3d_s(weights=weights)

        # Split swin.features into individual stage sub-modules so the forward
        # pass can capture intermediate skip features at H/8 and H/16.
        # Structure confirmed for torchvision 0.24.1 swin3d_s:
        #   [0] SwinBlocks → 96ch  H/4
        #   [1] PatchMerging       H/8   192ch
        #   [2] SwinBlocks → 192ch H/8          ← skip2
        #   [3] PatchMerging       H/16  384ch
        #   [4] SwinBlocks → 384ch H/16         ← skip3
        #   [5] PatchMerging       H/32  768ch
        #   [6] SwinBlocks → 768ch H/32
        assert len(swin.features) == 7, (
            f"Expected 7 entries in swin3d_s.features, got {len(swin.features)}."
        )
        self.patch_embed = swin.patch_embed
        self.pos_drop    = swin.pos_drop
        self.stage0      = swin.features[0]   # →  96ch  H/4
        self.merge1      = swin.features[1]   # → 192ch  H/8
        self.stage1      = swin.features[2]   # → 192ch  H/8   (skip2)
        self.merge2      = swin.features[3]   # → 384ch  H/16
        self.stage2      = swin.features[4]   # → 384ch  H/16  (skip3)
        self.merge3      = swin.features[5]   # → 768ch  H/32
        self.stage3      = swin.features[6]   # → 768ch  H/32  (bottleneck)
        self.norm        = swin.norm

        # Density conditioning
        self.density_emb   = nn.Embedding(self.NUM_DENSITY, self.EMBED_DIM)
        self.film           = FiLMLayer(self.EMBED_DIM, self.BOTTLENECK)  # bottleneck FiLM
        self.density_head   = nn.Linear(self.BOTTLENECK, self.NUM_DENSITY)
        self.bottleneck_drop = nn.Dropout2d(p=dropout)

        # Lateral 1×1 convs: project encoder skip features to decoder widths
        self.lat3 = nn.Conv2d(384, 256, 1, bias=False)  # stage2 → after dec1
        self.lat2 = nn.Conv2d(192, 128, 1, bias=False)  # stage1 → after dec2

        # Decoder — explicit stages so FiLM can be applied after each
        self.dec1 = UpBlock2d(768, 256)
        self.dec2 = UpBlock2d(256, 128)
        self.dec3 = UpBlock2d(128, 64)
        self.dec4 = UpBlock2d(64, 32)
        self.dec5 = UpBlock2d(32, 16)

        # Multi-scale FiLM — one per decoder stage output channel width
        self.film_layers = nn.ModuleList([
            FiLMLayer(self.EMBED_DIM, ch) for ch in (256, 128, 64, 32, 16)
        ])

        self.head = nn.Sequential(nn.Conv2d(16, 1, 1), nn.Sigmoid())

    def forward(
        self, x: torch.Tensor, density: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        H, W = x.shape[-2], x.shape[-1]

        # Encode — torchvision 0.16+ output is channels-last (B, T', H', W', C)
        f = self.patch_embed(x)
        f = self.pos_drop(f)

        f = self.stage0(f)
        f = self.merge1(f)
        f = self.stage1(f)
        skip2 = f                       # (B, T', H/8,  W/8,  192)
        f = self.merge2(f)
        f = self.stage2(f)
        skip3 = f                       # (B, T', H/16, W/16, 384)
        f = self.merge3(f)
        f = self.stage3(f)
        f = self.norm(f)                # (B, T', H/32, W/32, 768)

        # channels-last → channels-first + temporal mean pool
        f     = f.permute(0, 4, 1, 2, 3).mean(dim=2)      # (B, 768, H/32, W/32)
        skip3 = skip3.permute(0, 4, 1, 2, 3).mean(dim=2)  # (B, 384, H/16, W/16)
        skip2 = skip2.permute(0, 4, 1, 2, 3).mean(dim=2)  # (B, 192, H/8,  W/8)

        # Auxiliary density head from global pooled bottleneck
        logits = self.density_head(f.mean(dim=(-2, -1)))   # (B, 3)

        # Bottleneck dropout + FiLM
        f   = self.bottleneck_drop(f)
        emb = self.density_emb(density.clamp(min=0))       # (B, 64)
        f   = self.film(f, emb)

        # Decode: upsample → skip → multi-scale FiLM
        f = self.dec1(f)                      # (B, 256, H/16, W/16)
        f = f + self.lat3(skip3)              # skip from stage2
        f = self.film_layers[0](f, emb)

        f = self.dec2(f)                      # (B, 128, H/8, W/8)
        f = f + self.lat2(skip2)              # skip from stage1
        f = self.film_layers[1](f, emb)

        f = self.dec3(f)                      # (B, 64, H/4, W/4)
        f = self.film_layers[2](f, emb)

        f = self.dec4(f)                      # (B, 32, H/2, W/2)
        f = self.film_layers[3](f, emb)

        f = self.dec5(f)                      # (B, 16, H, W)
        f = self.film_layers[4](f, emb)

        out = self.head(f)                    # (B, 1, H, W)
        if out.shape[-2:] != (H, W):
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        return out, logits
