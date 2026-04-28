# PCIM-DTA
PCIM-DTA: Pairwise Conditional Interaction Modeling for Drug–Target Affinity Prediction

This repository provides a modular PyTorch implementation of **PCIM-DTA: Pairwise Conditional Interaction Modeling for Drug–Target Affinity Prediction**.

The code is organized according to the method section of the manuscript:

- `2.1 Pre-trained Representation Learning for Drugs and Proteins`
- `2.2 Pair-Level Interaction Construction`
- `2.3 Pair-Level Interaction Modeling`
- `2.4 Global Aggregation and Affinity Prediction`


## Module correspondence

### 2.1 Representation learning

File:

```text
pcim_dta/models/representation_learning.py
```

Outputs local and global representations for drugs and proteins.

### 2.2 Pair-level interaction construction

File:

```text
pcim_dta/models/pair_interaction.py
```

Constructs atom-residue pair tokens and global condition vectors.

### 2.3 Pair-level interaction modeling

File:

```text
pcim_dta/models/interaction_modeling.py
```

Performs condition-aware attention-based message passing over pair-level tokens.

### 2.4 Global aggregation and prediction

File:

```text
pcim_dta/models/prediction_head.py
```

Performs condition-aware weighted aggregation and affinity regression.
