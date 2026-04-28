"""
2.3 Pair-Level Interaction Modeling.

This module treats pair-level interaction tokens as graph nodes and performs
condition-aware message passing through attention.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionedGraphAttentionLayer(nn.Module):
    def __init__(self, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.condition_bias = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, condition: torch.Tensor, pair_mask: torch.Tensor | None = None) -> torch.Tensor:
        residual = x
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        cond = self.condition_bias(condition).unsqueeze(1)
        cond_scores = torch.matmul(cond, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        scores = scores + cond_scores

        if pair_mask is not None:
            key_mask = pair_mask.unsqueeze(1).bool()
            scores = scores.masked_fill(~key_mask, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = self.out_proj(out)
        out = self.norm(residual + self.dropout(out))

        if pair_mask is not None:
            out = out * pair_mask.unsqueeze(-1)
        return out


class PairLevelInteractionModeling(nn.Module):
    def __init__(self, hidden_dim: int = 256, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            ConditionedGraphAttentionLayer(hidden_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, pair_tokens: torch.Tensor, condition: torch.Tensor, pair_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = pair_tokens
        for layer in self.layers:
            x = layer(x, condition, pair_mask)
        return x
