"""
2.1 Pre-trained Representation Learning for Drugs and Proteins.

This module corresponds to Section 2.1 of the manuscript. It produces:
    - drug local atom-level representations
    - protein local residue-level representations
    - drug global representation
    - protein global representation

For a lightweight public release, the protein encoder is implemented as an
embedding + CNN encoder. In formal reproduction, this class can be replaced or
extended with ESM-2 features while keeping the output interface unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DrugGraphEncoder(nn.Module):
    """Simple GIN-like molecular graph encoder."""

    def __init__(self, atom_feature_dim: int, hidden_dim: int, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(atom_feature_dim, hidden_dim)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

    def forward(self, atom_features: torch.Tensor, adjacency: torch.Tensor, atom_mask: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(atom_features)
        deg = adjacency.sum(dim=-1, keepdim=True).clamp(min=1.0)
        adj_norm = adjacency / deg

        for layer, norm in zip(self.layers, self.norms):
            message = torch.bmm(adj_norm, x)
            update = layer(message)
            x = norm(x + update)

        x = x * atom_mask.unsqueeze(-1)
        return x


class ProteinSequenceEncoder(nn.Module):
    """Residue-level protein encoder with an ESM-compatible output interface."""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.proj = nn.Linear(embed_dim, hidden_dim)
        self.conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, sequence: torch.Tensor, seq_mask: torch.Tensor) -> torch.Tensor:
        x = self.embedding(sequence)
        x = self.proj(x)
        residual = x
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.norm(x + residual)
        x = x * seq_mask.unsqueeze(-1)
        return x


class PCIMRepresentationLearning(nn.Module):
    """Representation learning module for PCIM-DTA."""

    def __init__(self, config):
        super().__init__()
        self.drug_encoder = DrugGraphEncoder(
            atom_feature_dim=config.atom_feature_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.gnn_layers,
            dropout=config.dropout,
        )
        self.protein_encoder = ProteinSequenceEncoder(
            vocab_size=config.protein_vocab_size,
            embed_dim=config.protein_embed_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )
        self.drug_projection = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.protein_projection = nn.Linear(config.hidden_dim, config.hidden_dim)

    @staticmethod
    def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        return (x * mask.unsqueeze(-1)).sum(dim=1) / denom

    def forward(self, drug_inputs: dict, target_inputs: dict) -> dict:
        drug_local = self.drug_encoder(
            atom_features=drug_inputs["atom_features"],
            adjacency=drug_inputs["adjacency"],
            atom_mask=drug_inputs["atom_mask"],
        )
        protein_local = self.protein_encoder(
            sequence=target_inputs["sequence"],
            seq_mask=target_inputs["seq_mask"],
        )

        drug_local = F.normalize(self.drug_projection(drug_local), dim=-1)
        protein_local = F.normalize(self.protein_projection(protein_local), dim=-1)

        drug_global = self.masked_mean(drug_local, drug_inputs["atom_mask"])
        protein_global = self.masked_mean(protein_local, target_inputs["seq_mask"])

        return {
            "drug_local": drug_local,
            "protein_local": protein_local,
            "drug_global": drug_global,
            "protein_global": protein_global,
            "atom_mask": drug_inputs["atom_mask"],
            "seq_mask": target_inputs["seq_mask"],
        }
