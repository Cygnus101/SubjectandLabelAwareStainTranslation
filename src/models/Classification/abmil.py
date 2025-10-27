"""
Attention-based MIL classifier components.

Provides the standard gated attention pooling (Ilse et al., 2018) and an MLP
classifier head that operates on slide-level aggregated embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedAttentionPool(nn.Module):
    """
    Gated attention pooling for multiple-instance learning.

    Given bag embeddings ``H`` with shape (N, D), produces attention weights
    over instances and returns the aggregated representation ``z`` (D,).
    """

    def __init__(self, in_dim: int, attn_dim: int) -> None:
        super().__init__()
        self.V = nn.Linear(in_dim, attn_dim)
        self.U = nn.Linear(in_dim, attn_dim)
        self.w = nn.Linear(attn_dim, 1)

    def forward(self, H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        A = torch.tanh(self.V(H)) * torch.sigmoid(self.U(H))
        A = self.w(A)                    # (N, 1)
        A = torch.softmax(A, dim=0)      # attention distribution over instances
        z = torch.sum(A * H, dim=0)      # aggregated slide embedding
        return z, A.squeeze(1)


class SlideClassifier(nn.Module):
    """Simple MLP classifier operating on aggregated slide embeddings."""

    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.25) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.mlp(z)


class ABMIL(nn.Module):
    """
    Complete attention-based MIL classifier.

    For each bag of embeddings, attention pooling produces a slide representation
    that is passed through the classifier head.
    """

    def __init__(self, in_dim: int, attn_dim: int, classifier_hidden: int, num_classes: int, dropout: float = 0.25) -> None:
        super().__init__()
        self.pool = GatedAttentionPool(in_dim, attn_dim)
        self.classifier = SlideClassifier(in_dim, classifier_hidden, num_classes, dropout)

    def forward_one(self, bag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z, attn = self.pool(bag)
        logits = self.classifier(z.unsqueeze(0))
        return logits, attn, z

    def forward(self, bags: list[torch.Tensor]) -> Tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        logits, attn_weights, slide_vecs = [], [], []
        for bag in bags:
            lg, attn, z = self.forward_one(bag)
            logits.append(lg)
            attn_weights.append(attn)
            slide_vecs.append(z)
        return torch.cat(logits), attn_weights, slide_vecs
