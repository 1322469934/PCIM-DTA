"""Complete PCIM-DTA model assembled from Sections 2.1 to 2.4."""

from __future__ import annotations

import torch.nn as nn

from .representation_learning import PCIMRepresentationLearning
from .pair_interaction import PairLevelInteractionConstruction
from .interaction_modeling import PairLevelInteractionModeling
from .prediction_head import GlobalAggregationPrediction


class PCIMDTA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.representation = PCIMRepresentationLearning(config)
        self.pair_construction = PairLevelInteractionConstruction(
            hidden_dim=config.hidden_dim,
            pair_dim=config.pair_dim,
            condition_dim=config.condition_dim,
            dropout=config.dropout,
            max_pair_tokens=config.max_pair_tokens,
        )
        self.interaction_modeling = PairLevelInteractionModeling(
            hidden_dim=config.pair_dim,
            num_layers=config.interaction_layers,
            dropout=config.dropout,
        )
        self.prediction_head = GlobalAggregationPrediction(
            hidden_dim=config.pair_dim,
            regression_hidden_dim=config.regression_hidden_dim,
            dropout=config.dropout,
        )

    def forward(self, drug_inputs: dict, target_inputs: dict):
        rep_outputs = self.representation(drug_inputs, target_inputs)
        pair_outputs = self.pair_construction(rep_outputs)
        pair_features = self.interaction_modeling(
            pair_tokens=pair_outputs["pair_tokens"],
            condition=pair_outputs["condition"],
            pair_mask=pair_outputs["pair_mask"],
        )
        affinity = self.prediction_head(
            pair_features=pair_features,
            condition=pair_outputs["condition"],
            pair_mask=pair_outputs["pair_mask"],
        )
        return affinity
