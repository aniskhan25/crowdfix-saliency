"""
Continuous crowd density conditioning via frozen VGG16 features.

Replaces the discrete 3-class label lookup in DensitySwinSaliency with a
continuous embedding derived from the middle frame's visual content:

  middle_frame → VGG16 frontend (frozen, ImageNet) → dilated-conv backend
               → GAP → 64-dim embedding → FiLM bottleneck

An auxiliary classification head on the 64-dim embedding provides density
supervision during training (same cross-entropy loss as the other variants).
At inference, no GT density label is needed — the embedding is driven entirely
by what the model sees in the frame.

This addresses the reviewer concern that a 3-class categorical label is too
coarse: the conditioning signal is now a continuous, high-capacity descriptor
derived from VGG16 crowd-scene features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights

from models.density_swin_saliency import FiLMLayer, UpBlock2d


class CrowdDensityEncoder(nn.Module):
    """VGG16 frontend (conv1-conv4_3, frozen) + dilated-conv backend → 64-dim."""

    def __init__(self, freeze_vgg: bool = True):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        # CSRNet frontend: conv1-conv4_3, no pool4 → 512 ch at H/8
        self.frontend = nn.Sequential(*list(vgg.features[:23]))
        if freeze_vgg:
            for p in self.frontend.parameters():
                p.requires_grad_(False)

        # Dilated-conv backend (CSRNet-style); maintains spatial size
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(128,  64, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
        )
        self.proj = nn.Linear(64, 64)

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        """frame: (B, 3, H, W) → (B, 64) density embedding."""
        f = self.frontend(frame)          # (B, 512, H/8, W/8)
        f = self.backend(f)               # (B,  64, H/8, W/8)
        f = f.mean(dim=(-2, -1))          # (B, 64)  global average pool
        return self.proj(f)               # (B, 64)


class DensitySwinContinuous(nn.Module):
    """Video Swin + continuous density conditioning from frozen VGG16 features."""

    NUM_DENSITY = 3
    BOTTLENECK  = 768
    EMBED_DIM   = 64

    def __init__(self, pretrained: bool = True, dropout: float = 0.5,
                 freeze_vgg: bool = True):
        super().__init__()

        from torchvision.models.video import swin3d_s, Swin3D_S_Weights
        weights = Swin3D_S_Weights.KINETICS400_V1 if pretrained else None
        swin = swin3d_s(weights=weights)
        self.patch_embed     = swin.patch_embed
        self.pos_drop        = swin.pos_drop
        self.encoder         = swin.features
        self.norm            = swin.norm

        self.density_enc     = CrowdDensityEncoder(freeze_vgg=freeze_vgg)
        self.density_head    = nn.Linear(self.EMBED_DIM, self.NUM_DENSITY)
        self.film            = FiLMLayer(self.EMBED_DIM, self.BOTTLENECK)
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
        """
        x:       (B, C, T, H, W) video clip (channels-first temporal)
        density: ignored for conditioning; accepted for API compatibility
                 (the aux cross-entropy loss in train.py uses it)
        Returns: (saliency (B,1,H,W), logits (B,3))
        """
        H, W = x.shape[-2], x.shape[-1]
        T    = x.shape[2]

        # Middle frame drives density conditioning
        mid = x[:, :, T // 2]             # (B, C, H, W)

        f = self.patch_embed(x)
        f = self.pos_drop(f)
        f = self.encoder(f)
        f = self.norm(f)                   # (B, T', H/32, W/32, 768)
        f = f.permute(0, 4, 1, 2, 3).mean(dim=2)  # (B, 768, H/32, W/32)

        emb    = self.density_enc(mid)     # (B, 64)
        logits = self.density_head(emb)    # (B, 3)

        f = self.bottleneck_drop(f)
        f = self.film(f, emb)

        f = self.dec1(f)
        f = self.dec2(f)
        f = self.dec3(f)
        f = self.dec4(f)
        f = self.dec5(f)

        out = self.head(f)
        if out.shape[-2:] != (H, W):
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        return out, logits
