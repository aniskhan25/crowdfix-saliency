"""
Multi-scale FiLM variant of DensitySwinSaliency.

Applies FiLM at four spatial scales using the same shared density embedding
with independent projection weights per scale:
  stride 32 — bottleneck      (768 channels)
  stride 16 — after dec1      (256 channels)
  stride  8 — after dec2      (128 channels)
  stride  4 — after dec3      ( 64 channels)

Dec4 and dec5 are left unmodulated (16→8px resolution; density is a
global scene property and spatial-detail stages are less sensitive to it).

The three decoder-scale FiLM projections are zero-initialized so each
starts as an identity transform, preserving the stability of the
freeze→fine-tune training protocol.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.density_swin_saliency import FiLMLayer, UpBlock2d


class DensitySwinMultiscale(nn.Module):
    """Video Swin + multi-scale FiLM conditioning (strides 32/16/8/4)."""

    NUM_DENSITY = 3
    EMBED_DIM   = 64
    BOTTLENECK  = 768

    def __init__(self, pretrained: bool = True, dropout: float = 0.5):
        super().__init__()

        from torchvision.models.video import swin3d_s, Swin3D_S_Weights
        weights = Swin3D_S_Weights.KINETICS400_V1 if pretrained else None
        swin = swin3d_s(weights=weights)

        self.patch_embed     = swin.patch_embed
        self.pos_drop        = swin.pos_drop
        self.encoder         = swin.features
        self.norm            = swin.norm

        self.density_emb     = nn.Embedding(self.NUM_DENSITY, self.EMBED_DIM)
        self.density_head    = nn.Linear(self.BOTTLENECK, self.NUM_DENSITY)
        self.bottleneck_drop = nn.Dropout2d(p=dropout)

        # FiLM at each scale; decoder-scale projections zero-initialized
        self.film_s32 = FiLMLayer(self.EMBED_DIM, 768)
        self.film_s16 = FiLMLayer(self.EMBED_DIM, 256)
        self.film_s8  = FiLMLayer(self.EMBED_DIM, 128)
        self.film_s4  = FiLMLayer(self.EMBED_DIM,  64)
        for film in (self.film_s16, self.film_s8, self.film_s4):
            nn.init.zeros_(film.proj.weight)
            nn.init.zeros_(film.proj.bias)

        self.dec1 = UpBlock2d(768, 256)
        self.dec2 = UpBlock2d(256, 128)
        self.dec3 = UpBlock2d(128,  64)
        self.dec4 = UpBlock2d( 64,  32)
        self.dec5 = UpBlock2d( 32,  16)
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

        f   = self.bottleneck_drop(f)
        emb = self.density_emb(density.clamp(min=0))   # (B, 64)

        f = self.film_s32(f, emb)   # stride 32 — bottleneck

        f = self.dec1(f)
        f = self.film_s16(f, emb)   # stride 16

        f = self.dec2(f)
        f = self.film_s8(f, emb)    # stride  8

        f = self.dec3(f)
        f = self.film_s4(f, emb)    # stride  4

        f = self.dec4(f)
        f = self.dec5(f)

        out = self.head(f)
        if out.shape[-2:] != (H, W):
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        return out, logits
