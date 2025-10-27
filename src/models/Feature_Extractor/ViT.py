"""
Vision Transformer feature extractor with SimCLR projection head.

Uses torchvision's ViT-B/16 backbone to produce 768-D embeddings which are fed
through the standard two-layer projection MLP.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ViT_B_16_Weights, vit_b_16


class ProjectionHead(nn.Module):
    """Two-layer projection head used in SimCLR."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimCLRModel(nn.Module):
    """ViT-B/16 backbone with projection head for contrastive learning."""

    def __init__(
        self,
        pretrained: bool,
        proj_hidden_dim: int,
        proj_out_dim: int,
    ) -> None:
        super().__init__()
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = vit_b_16(weights=weights)
        feature_dim = backbone.heads.head.in_features
        backbone.heads = nn.Identity()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.projector = ProjectionHead(self.feature_dim, proj_hidden_dim, proj_out_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.backbone.forward_features(x)
        feats = feats.flatten(1)
        proj = self.projector(feats)
        proj = F.normalize(proj, dim=1)
        return feats, proj
