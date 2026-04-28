"""
PCIM-DTA inference example.

This script demonstrates how to call the complete PCIM-DTA model assembled from:
    2.1 Pre-trained Representation Learning for Drugs and Proteins
    2.2 Pair-Level Interaction Construction
    2.3 Pair-Level Interaction Modeling
    2.4 Global Aggregation and Affinity Prediction

Note:
    If no checkpoint is found, the model will run with random initialization.
    The output is then only a code-flow demonstration, not a valid affinity result.
"""

from __future__ import annotations

import os
import torch
import numpy as np

from config import Config
from pcim_dta.data.preprocess import smiles_to_graph, seq_to_indices
from pcim_dta.models.pcim_dta import PCIMDTA


def build_inputs(smiles: str, sequence: str, config: Config, device: str):
    graph = smiles_to_graph(smiles, config.max_drug_atoms, config.atom_feature_dim)
    seq_idx, seq_mask = seq_to_indices(sequence, config.max_seq_len)

    drug_inputs = {
        "atom_features": torch.tensor(graph["atom_features"], dtype=torch.float32).unsqueeze(0).to(device),
        "adjacency": torch.tensor(graph["adjacency"], dtype=torch.float32).unsqueeze(0).to(device),
        "atom_mask": torch.tensor(graph["atom_mask"], dtype=torch.float32).unsqueeze(0).to(device),
    }
    target_inputs = {
        "sequence": torch.tensor(seq_idx, dtype=torch.long).unsqueeze(0).to(device),
        "seq_mask": torch.tensor(seq_mask, dtype=torch.float32).unsqueeze(0).to(device),
    }
    return drug_inputs, target_inputs


def load_checkpoint_if_available(model: PCIMDTA, checkpoint_path: str, device: str):
    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print("Warning: checkpoint not found. Running with random initialization.")
        print("The prediction value is only for code-flow demonstration.")


def predict_affinity(model: PCIMDTA, smiles: str, sequence: str, config: Config, device: str = "cpu") -> float:
    model.eval()
    drug_inputs, target_inputs = build_inputs(smiles, sequence, config, device)
    with torch.no_grad():
        pred = model(drug_inputs, target_inputs)
    return float(pred.item())


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    set_seed(42)
    config = Config()
    device = "cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu"

    model = PCIMDTA(config).to(device)
    load_checkpoint_if_available(model, config.checkpoint_path, device)

    examples = [
        {
            "name": "Gefitinib-EGFR",
            "smiles": "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4",
            "sequence": "MRPSGTAGAALLALLAALCPASRALEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQRNYDLSFLKTIQEVAGYVLIALNTVERIPLENLQIIRGNMYYENSYALAVLSNYDANKTGLKELPMRNLQEILHGAVRFSNNPALCNVESIQWRDIVSSDFLSNMSMDFQNHLGSCQKCDPSCPNGSCWGAGEENCQKLTKIICAQQCSGRCRGKSPSDCCHNQCAAGCTGPRCLICRKFRDEATCKDTCPPLMLYNPTTYQMDVNPEGKYSFGATCVKKCPRNYVVTDHGSCVRACGADSYEMEEDGVRKCKKCEGPCRKVCNGIGIGEFKDSLSINATNIKHFKNCTSISGDLHILPVAFRGDSFTHTPPLDPQELDILKTVKEITGFLLIQAWPENRTDLHAFENLEIIRGRTKQHGQFSLAVVSLNITSLGLRSLKEISDGDVIISGNKNLCYANTINWKKLFGTSGQKTKIISNRGENSCKATGQVCHALCSPEGCWGPEPRDCVSCRNVSRGRECVDKCNLLLEGEPREFVENSECIQCHPECLPQAMNITCTGRGPDNCIQCAHYIDGPHCVKTCPAGVMGENNTLVWKYADAGHVCHLCHPNCTYGCTGPGLEGCPTNGPKIPSIATGMVGALLLLLVVALGIGLFM",
        },
        {
            "name": "Imatinib-ABL1",
            "smiles": "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",
            "sequence": "MLEICLKLVGCKSKKGLSSSSSCYLEEALQRPVASDFEPQGLSEAARWNSKENLLAGPSENDPNLFVALYDFVASGDNTLSITKGEKLRVLGYNHNGEWCEAQTKNGQGWVPSNYITPVNSLEKHSWYHGPVSRNAAEYLLSSGINGSFLVRESESSPGQLSISLRYEGRVYHYRINTASDGKLYVSSESRFNTLAELVHHHSTVADGLITTLHYPAPKRNKPTVYGVSPNYDKWEMERTDITMKHKLGGGQYGEVYEGVWKKYSLTVAVKTLKEDTMEVEEFLKEAAVMKEIKHPNLVQLLGVCTREPPFYIITEFMTYGNLLDYLRECNRQEVNAVVLLYMATQISSAMEYLEKKNFIHRDLAARNCLVGENHLVKVADFGLSRLMTGDTYTAHAGAKFPIKWTAPESLAYNKFSIKSDVWAFGVLLWEIATYGMSPYPGIDLSQVYELLEKDYRMERPEGCPEKVYELMRACWQWNPSDRPSFAEIHQAFETMFQESSISDEVEKELGKQGVRGAVSTLLQAPELPTKTRTSRRAAEHRDTTDVPEMPHSKGQGESDPLDHEPAVSPLLPRKERGPPEGGLNEDERLLPKDKKTNLFSALIKKKKKTAPTPPKRSSSFREMDGQPERGQ",
        },
    ]

    print("=" * 70)
    print("PCIM-DTA inference demo")
    print("=" * 70)

    for item in examples:
        pred = predict_affinity(model, item["smiles"], item["sequence"], config, device)
        print(f"{item['name']}: predicted affinity = {pred:.4f}")


if __name__ == "__main__":
    main()
