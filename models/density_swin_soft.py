"""
Soft (continuous) density conditioning variant of DensitySwinSaliency.

The bottleneck FiLM conditioning uses softmax(density_head_logits) as a
convex weighting of the three class embeddings, rather than a hard GT class
lookup. This enables continuous interpolation between density regimes and
removes the GT label requirement at inference.

At training time, GT density labels are still used for the auxiliary
cross-entropy loss; the conditioning signal itself is always self-derived.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.density_swin_saliency import FiLMLayer, UpBlock2d


class DensitySwinSoft(nn.Module):
    """Video Swin + soft (continuous) bottleneck FiLM conditioning."""

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
        self.film            = FiLMLayer(self.EMBED_DIM, self.BOTTLENECK)
        self.density_head    = nn.Linear(self.BOTTLENECK, self.NUM_DENSITY)
        self.bottleneck_drop = nn.Dropout2d(p=dropout)

        self.dec1 = UpBlock2d(768, 256)
        self.dec2 = UpBlock2d(256, 128)
        self.dec3 = UpBlock2d(128,  64)
        self.dec4 = UpBlock2d( 64,  32)
        self.dec5 = UpBlock2d( 32,  16)
        self.head = nn.Sequential(nn.Conv2d(16, 1, 1), nn.Sigmoid())

    def forward(
        self, x: torch.Tensor, density: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        H, W = x.shape[-2], x.shape[-1]

        f = self.patch_embed(x)
        f = self.pos_drop(f)
        f = self.encoder(f)
        f = self.norm(f)                               # (B, T', H/32, W/32, 768)

        f = f.permute(0, 4, 1, 2, 3).mean(dim=2)      # (B, 768, H/32, W/32)
        logits = self.density_head(f.mean(dim=(-2, -1)))  # (B, 3)

        f    = self.bottleneck_drop(f)
        # Soft conditioning: weighted sum of class embeddings by predicted probs.
        # Stop-gradient so the density head trains from the aux loss, not from
        # FiLM's downstream saliency signal (avoids circular gradient).
        soft = F.softmax(logits.detach(), dim=-1)      # (B, 3)
        emb  = soft @ self.density_emb.weight          # (B, 64)
        f    = self.film(f, emb)

        f = self.dec1(f)
        f = self.dec2(f)
        f = self.dec3(f)
        f = self.dec4(f)
        f = self.dec5(f)

        out = self.head(f)
        if out.shape[-2:] != (H, W):
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        return out, logits
