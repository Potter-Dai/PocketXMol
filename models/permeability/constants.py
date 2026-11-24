"""Constants and vocabulary mappings for 3D chameleonic permeability models."""

from typing import Dict, List, Tuple

# PocketXMol pretrained atom & bond vocabularies (from pocketxmol.ckpt featurizer)
PXM_ATOMIC_NUMBERS: Tuple[int, ...] = (6, 7, 8, 9, 15, 16, 17, 5, 35, 53, 34)
PXM_MASK_NODE_IDX: int = 11
NUM_PXM_NODE_TYPES: int = 12  # 0..10 heavy elements + 11 <MASK>

ATOMIC_NUM_TO_PXM_IDX: Dict[int, int] = {
    atomic_num: idx for idx, atomic_num in enumerate(PXM_ATOMIC_NUMBERS)
}
# Index 1 -> N (7), Index 2 -> O (8)
PXM_POLAR_NODE_INDICES: Tuple[int, int] = (1, 2)

# Edge types: 0=non-bonded, 1=single, 2=double, 3=triple, 4=aromatic, 5=<MASK>
PXM_BOND_TYPES: Tuple[int, ...] = (1, 2, 3, 4)
PXM_MASK_EDGE_IDX: int = 5
NUM_PXM_EDGE_TYPES: int = 6
POCKET_IN_DIM: int = 25

# Assay IDs (0 reserved strictly for UNKNOWN -> zero embedding for zero-shot assays)
ASSAY_TO_ID: Dict[str, int] = {
    "UNKNOWN": 0,
    "PAMPA": 1,
    "CACO2": 2,
    "RRCK": 3,
    "MDCK": 4,
    "SOLUBILITY": 5,
}
ID_TO_ASSAY: Dict[int, str] = {v: k for k, v in ASSAY_TO_ID.items()}
PERM_ASSAY_NAMES: Tuple[str, ...] = ("PAMPA", "CACO2", "RRCK", "MDCK")
ALL_ASSAY_NAMES: Tuple[str, ...] = (
    "UNKNOWN",
    "PAMPA",
    "CACO2",
    "RRCK",
    "MDCK",
    "SOLUBILITY",
)
NUM_ASSAYS: int = len(ASSAY_TO_ID)

# Standardized 5-dim Chameleonic Auxiliary targets
CHAM_AUX_NAMES: Tuple[str, ...] = (
    "psa_3d_min",
    "delta_psa_3d",
    "imhb_max",
    "rg_min",
    "closure",
)
NUM_CHAM_TARGETS: int = len(CHAM_AUX_NAMES)

# Full 8-dim cham_targets potentially stored in LMDB
DEFAULT_CHAM8_ORDER: Tuple[str, ...] = (
    "psa_3d_min",
    "psa_3d_max",
    "delta_psa_3d",
    "psa_3d_boltz",
    "imhb_max",
    "imhb_mean",
    "rg_min",
    "delta_rg",
)

# Default pocketxmol.ckpt model hyperparameters (mirrored from train.yml;
# overwritten by ckpt['hyper_parameters']['config'].model when ckpt is loaded)
DEFAULT_PXM_MODEL_CFG: Dict[str, object] = {
    "name": "pm_asym_denoiser",
    "pocket_dim": 128,
    "node_dim": 320,
    "addition_node_features": ["is_peptide"],
    "add_output": ["confidence"],
    "edge_dim": 96,
    "pocket": {
        "edge_dim": 32,
        "hidden_dim": 128,
        "num_blocks": 4,
        "dist_cfg": {"num_gaussians": 32},
    },
    "denoiser": {
        "hidden_dim": 320,
        "num_blocks": 6,
        "gate_dim": 2,
        "dist_cfg": {"stop": 15, "num_gaussians": 64},
        "context_cfg": {
            "edge_dim": 128,
            "knn": 32,
            "dist_cfg": {"stop": 20, "num_gaussians": 64, "type_": "linear"},
        },
    },
}


def canonicalize_assay_name(assay: str) -> str:
    """Normalize assay string names across CycDelta/EnsembleCycPerm conventions."""
    key = str(assay).strip().upper().replace("-", "").replace("_", "").replace(" ", "")
    if key in ("", "NONE", "UNKNOWN", "UNK", "UNSEEN", "NIELSEN", "CAPA"):
        return "UNKNOWN"
    if "PAMPA" in key:
        return "PAMPA"
    if "CACO" in key:
        return "CACO2"
    if "RRCK" in key:
        return "RRCK"
    if "MDCK" in key:
        return "MDCK"
    if "SOL" in key:
        return "SOLUBILITY"
    return "UNKNOWN"


def atomic_numbers_to_pxm_idx(elements):
    """Map atomic numbers tensor or sequence [N] to PXM node_type indices [0..11]."""
    import torch

    if not torch.is_tensor(elements):
        elements = torch.as_tensor(elements, dtype=torch.long)
    out = torch.full_like(elements, PXM_MASK_NODE_IDX, dtype=torch.long)
    for z_val, idx in ATOMIC_NUM_TO_PXM_IDX.items():
        out = torch.where(elements == z_val, torch.as_tensor(idx, device=elements.device, dtype=torch.long), out)
    return out
