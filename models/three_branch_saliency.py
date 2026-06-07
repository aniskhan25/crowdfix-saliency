"""
Three-Branch Density-Conditioned Saliency Model — Idea 3 (Tier 1).

Combines three literature-motivated improvements over run 4 (density_swin):

  1. Density-conditioned FiLM at the bottleneck (run 4, Idea 1).
  2. SalFoM-inspired dual decoder branches:
       Branch A (static):   temporal mean-pool → density FiLM → 5-stage 2D decoder.
       Branch B (temporal): keep 3D volume → 2 × UpBlock3d (THTD-Net-style gradual
                            temporal reduction) → 3-stage 2D decoder tail.
     The two 16-channel feature maps are fused by a 1×1 conv before the final head.
  3. GASP-inspired learned social prior: 3 spatial bias maps (one per density
     category), initialised to zero, applied as a logit addition before sigmoid.
     The gate scalar is also learned, initialised to 0 so the prior starts inactive.

Input:
    x       (B, 3, T, H, W)   video clip, ImageNet-normalised
    density (B,)  int64        SP=0, DF=1, DC=2; -1=unknown

Output:
    saliency (B, 1, H, W)     in [0, 1]
    logits   (B, 3)            density class scores
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UpBlock2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))


class UpBlock3d(nn.Module):
    """
    Spatial 2× bilinear upsample + temporal halving via avg_pool3d.
    If T == 1 the temporal dimension is left unchanged.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        # Spatial 2× upsample (fuse T into batch dim for 2D interpolate)
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = x.reshape(B, T, C, H * 2, W * 2).permute(0, 2, 1, 3, 4)  # (B, C, T, 2H, 2W)
        # Temporal halving
        if T > 1:
            x = F.avg_pool3d(x, kernel_size=(2, 1, 1), stride=(2, 1, 1))
        return self.conv(x)


class FiLMLayer(nn.Module):
    def __init__(self, embed_dim: int, n_channels: int):
        super().__init__()
        self.proj = nn.Linear(embed_dim, 2 * n_channels)

    def forward(self, feat: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.proj(emb).chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta  = beta.unsqueeze(-1).unsqueeze(-1)
        return feat * (1.0 + gamma) + beta


class ThreeBranchSaliency(nn.Module):

    NUM_DENSITY = 3
    EMBED_DIM   = 64
    BOTTLENECK  = 768

    # Social prior at H/16 × W/16 = 14 × 24 for 224×384 input.
    # Small enough to be a global shape prior; large enough to express
    # meaningful spatial gaze patterns per density category.
    PRIOR_H = 14
    PRIOR_W = 24

    def __init__(self, pretrained: bool = True, dropout: float = 0.5):
        super().__init__()

        from torchvision.models.video import swin3d_s, Swin3D_S_Weights
        weights = Swin3D_S_Weights.KINETICS400_V1 if pretrained else None
        swin = swin3d_s(weights=weights)

        self.patch_embed = swin.patch_embed
        self.pos_drop    = swin.pos_drop
        self.encoder     = swin.features
        self.norm        = swin.norm

        # Shared density conditioning
        self.density_emb     = nn.Embedding(self.NUM_DENSITY, self.EMBED_DIM)
        self.film            = FiLMLayer(self.EMBED_DIM, self.BOTTLENECK)
        self.density_head    = nn.Linear(self.BOTTLENECK, self.NUM_DENSITY)
        self.bottleneck_drop = nn.Dropout2d(p=dropout)

        # ── Branch A: static 2D decoder ──────────────────────────────────
        self.a_dec1 = UpBlock2d(768, 256)
        self.a_dec2 = UpBlock2d(256, 128)
        self.a_dec3 = UpBlock2d(128, 64)
        self.a_dec4 = UpBlock2d(64, 32)
        self.a_dec5 = UpBlock2d(32, 16)

        # ── Branch B: temporal-dynamic 3D decoder ────────────────────────
        # 3D stages: (B, 768, T', H', W') → (B, 128, ≈1, 4H', 4W')
        self.b_3d_1 = UpBlock3d(768, 256)   # T'/2, 2H', 2W'
        self.b_3d_2 = UpBlock3d(256, 128)   # T'/4, 4H', 4W'
        # 2D tail (picks up after the spatial equivalent of a_dec3)
        self.b_dec3 = UpBlock2d(128, 64)
        self.b_dec4 = UpBlock2d(64, 32)
        self.b_dec5 = UpBlock2d(32, 16)

        # ── Fusion ───────────────────────────────────────────────────────
        self.fusion = nn.Sequential(
            nn.Conv2d(32, 16, 1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
        )

        # ── Social prior (logit-space additive bias per density class) ────
        self.social_prior = nn.Parameter(
            torch.zeros(self.NUM_DENSITY, 1, self.PRIOR_H, self.PRIOR_W)
        )
        # Scalar gate initialised to 0 → sigmoid(0)=0.5 initial scale.
        # Starting from 0 lets training decide whether to use the prior.
        self.social_gate = nn.Parameter(torch.zeros(1))

        # Final head (logit → sigmoid, prior inserted before sigmoid)
        self.head_conv = nn.Conv2d(16, 1, 1)

    def forward(
        self, x: torch.Tensor, density: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, C, T, H, W = x.shape

        # ── Shared encoder ───────────────────────────────────────────────
        f = self.patch_embed(x)
        f = self.pos_drop(f)
        f = self.encoder(f)
        f = self.norm(f)                              # (B, T', H', W', 768)
        f3d = f.permute(0, 4, 1, 2, 3)              # (B, 768, T', H', W')

        logits = self.density_head(f3d.mean(dim=(-3, -2, -1)))  # (B, 3)

        # ── Branch A: static ─────────────────────────────────────────────
        fa = f3d.mean(dim=2)                          # (B, 768, H', W')
        fa = self.bottleneck_drop(fa)
        emb = self.density_emb(density.clamp(min=0))
        fa  = self.film(fa, emb)
        fa  = self.a_dec1(fa)
        fa  = self.a_dec2(fa)
        fa  = self.a_dec3(fa)
        fa  = self.a_dec4(fa)
        fa  = self.a_dec5(fa)                         # (B, 16, H, W)

        # ── Branch B: temporal-dynamic ───────────────────────────────────
        fb = self.b_3d_1(f3d)                         # (B, 256, T'/2, 2H', 2W')
        fb = self.b_3d_2(fb)                          # (B, 128, ≈1,   4H', 4W')
        fb = fb.mean(dim=2)                           # collapse residual T → (B, 128, 4H', 4W')
        fb = self.b_dec3(fb)
        fb = self.b_dec4(fb)
        fb = self.b_dec5(fb)                          # (B, 16, H, W)

        # ── Fusion ───────────────────────────────────────────────────────
        fused = self.fusion(torch.cat([fa, fb], dim=1))  # (B, 16, H, W)

        # ── Head + social prior ──────────────────────────────────────────
        logit = self.head_conv(fused)                 # (B, 1, H_out, W_out)

        d     = density.clamp(min=0)
        prior = self.social_prior[d]                  # (B, 1, 14, 24)
        prior = F.interpolate(prior, size=logit.shape[-2:],
                              mode="bilinear", align_corners=False)
        logit = logit + torch.sigmoid(self.social_gate) * prior

        out = torch.sigmoid(logit)                    # (B, 1, H_out, W_out)

        if out.shape[-2:] != (H, W):
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        return out, logits
