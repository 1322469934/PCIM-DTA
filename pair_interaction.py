"""
2.2 Pair-Level Interaction Construction.

This module constructs atom-residue pair tokens and applies global conditional
modulation, corresponding to Section 2.2 of the manuscript.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PairLevelInteractionConstruction(nn.Module):
    def __init__(self, hidden_dim: int = 256, pair_dim: int = 256, condition_dim: int = 256, dropout: float = 0.1, max_pair_tokens: int | None = None):
        super().__init__()
        self.relevance = nn.Bilinear(hidden_dim, hidden_dim, 1, bias=False)
        self.pair_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, pair_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(pair_dim, pair_dim),
        )
        self.condition_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, condition_dim),
            nn.ReLU(),
            nn.Linear(condition_dim, condition_dim),
        )
        self.condition_to_gate = nn.Linear(condition_dim, pair_dim)
        self.norm = nn.LayerNorm(pair_dim)
        self.max_pair_tokens = max_pair_tokens

    def forward(self, rep_outputs: dict) -> dict:
        drug_local = rep_outputs["drug_local"]
        protein_local = rep_outputs["protein_local"]
        drug_global = rep_outputs["drug_global"]
        protein_global = rep_outputs["protein_global"]
        atom_mask = rep_outputs["atom_mask"]
        seq_mask = rep_outputs["seq_mask"]

        B, Na, D = drug_local.shape
        Nt = protein_local.shape[1]

        drug_expand = drug_local.unsqueeze(2).expand(B, Na, Nt, D)
        protein_expand = protein_local.unsqueeze(1).expand(B, Na, Nt, D)

        relevance = self.relevance(
            drug_expand.reshape(B * Na * Nt, D),
            protein_expand.reshape(B * Na * Nt, D),
        ).reshape(B, Na, Nt, 1)

        pair_input = torch.cat([drug_expand, protein_expand, relevance], dim=-1)
        pair_tokens = self.pair_mlp(pair_input).reshape(B, Na * Nt, -1)

        condition = self.condition_mlp(torch.cat([drug_global, protein_global], dim=-1))
        gate = torch.sigmoid(self.condition_to_gate(condition)).unsqueeze(1)
        pair_tokens = self.norm(pair_tokens * gate)

        pair_mask = (atom_mask.unsqueeze(2) * seq_mask.unsqueeze(1)).reshape(B, Na * Nt)
        pair_tokens = pair_tokens * pair_mask.unsqueeze(-1)

        # Dense atom-residue pairing can be very large. For practical inference,
        # keep the most relevant valid pair tokens when max_pair_tokens is set.
        if self.max_pair_tokens is not None and pair_tokens.size(1) > self.max_pair_tokens:
            flat_rel = relevance.reshape(B, Na * Nt)
            flat_rel = flat_rel.masked_fill(pair_mask <= 0, -1e9)
            topk = torch.topk(flat_rel, k=self.max_pair_tokens, dim=1).indices
            gather_idx = topk.unsqueeze(-1).expand(-1, -1, pair_tokens.size(-1))
            pair_tokens = torch.gather(pair_tokens, dim=1, index=gather_idx)
            pair_mask = torch.gather(pair_mask, dim=1, index=topk)

        return {
            "pair_tokens": pair_tokens,
            "pair_mask": pair_mask,
            "condition": condition,
            "relevance": relevance.squeeze(-1),
        }
