"""
2.4 Global Aggregation and Affinity Prediction.

This module performs condition-aware global aggregation over pair-level tokens
and predicts the final drug-target affinity value.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalAggregationPrediction(nn.Module):
    def __init__(self, hidden_dim: int = 256, regression_hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.weight_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim * 2, regression_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(regression_hidden_dim, regression_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(regression_hidden_dim // 2, 1),
        )

    def forward(self, pair_features: torch.Tensor, condition: torch.Tensor, pair_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, N, D = pair_features.shape
        condition_expand = condition.unsqueeze(1).expand(B, N, D)
        weight_input = torch.cat([pair_features, condition_expand], dim=-1)
        scores = self.weight_net(weight_input).squeeze(-1)

        if pair_mask is not None:
            scores = scores.masked_fill(pair_mask <= 0, -1e9)

        weights = F.softmax(scores, dim=-1)
        global_interaction = torch.sum(pair_features * weights.unsqueeze(-1), dim=1)
        pred_input = torch.cat([global_interaction, condition], dim=-1)
        affinity = self.regressor(pred_input).squeeze(-1)
        return affinity
