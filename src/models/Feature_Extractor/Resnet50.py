"""
ResNet-based feature extractor with SimCLR-style projection head.

The extractor consists of a ResNet-50 backbone (optionally ImageNet-pretrained)
followed by a small MLP projection head used during contrastive training.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50


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
    """ResNet backbone with projection head for contrastive learning."""

    def __init__(
        self,
        pretrained: bool,
        proj_hidden_dim: int,
        proj_out_dim: int,
    ) -> None:
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)
        self.backbone = nn.Sequential(*(list(backbone.children())[:-1]))
        self.feature_dim = backbone.fc.in_features  # 2048 for ResNet-50
        self.projector = ProjectionHead(self.feature_dim, proj_hidden_dim, proj_out_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.backbone(x).flatten(1)  # (batch, 2048)
        proj = self.projector(feats)
        proj = F.normalize(proj, dim=1)
        return feats, proj
