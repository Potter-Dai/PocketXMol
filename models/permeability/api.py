"""Differentiable Phase 3 guidance API and RDKit scoring wrapper for PermCycXMol."""

import copy
import multiprocessing
import os
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from models.permeability.constants import (
    ASSAY_TO_ID,
    CHAM_AUX_NAMES,
    NUM_PXM_EDGE_TYPES,
    NUM_PXM_NODE_TYPES,
    atomic_numbers_to_pxm_idx,
    canonicalize_assay_name,
)
from models.permeability.encoder import ChameleonicPXMEncoder, build_complete_halfedges
from models.permeability.heads import PermCycXMolModel


def _resolve_assay_tensor(
    assay: Union[str, int, torch.Tensor],
    num_mols: int,
    device: torch.device,
) -> torch.Tensor:
    if torch.is_tensor(assay):
        t = assay.long().to(device)
        if t.numel() == 1 and num_mols > 1:
            return t.view(1).expand(num_mols)
        return t.view(num_mols)
    if isinstance(assay, str):
        a_id = ASSAY_TO_ID[canonicalize_assay_name(assay)]
    else:
        a_id = int(assay)
    return torch.full((num_mols,), a_id, dtype=torch.long, device=device)


def compute_rdkit_atom_phys_feat(mol_no_h: Any) -> np.ndarray:
    """Compute exact 14-dim per-heavy-atom physical features matching perm_v1.lmdb schema.

    Columns [N, 14]:
      0: is_hbd (N/O with total Hs > 0)
      1: is_hba (N/O with formal_charge <= 0)
      2: total_Hs
      3: heavy_degree
      4: formal_charge
      5: is_aromatic
      6: is_in_ring
      7: is_in_macrocycle (ring size >= 12)
      8: hybridization_id(0..5) / 5.0
      9: crippen_logp_contrib
     10: crippen_mr_contrib
     11: tpsa_contrib
     12: labute_asa_contrib
     13: gasteiger_charge
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, Crippen, rdMolDescriptors

    ri = mol_no_h.GetRingInfo()
    macro_atoms = set()
    for ring in ri.AtomRings():
        if len(ring) >= 12:
            macro_atoms.update(ring)

    hyb_map = {
        Chem.HybridizationType.UNSPECIFIED: 0.0,
        Chem.HybridizationType.S: 1.0,
        Chem.HybridizationType.SP: 2.0,
        Chem.HybridizationType.SP2: 3.0,
        Chem.HybridizationType.SP3: 4.0,
        Chem.HybridizationType.SP3D: 5.0,
        Chem.HybridizationType.SP3D2: 5.0,
    }

    crippen_contribs = Crippen._GetAtomContribs(mol_no_h)
    tpsa_contribs = rdMolDescriptors._CalcTPSAContribs(mol_no_h)
    mol_tmp = Chem.Mol(mol_no_h)
    labute_contribs = rdMolDescriptors._CalcLabuteASAContribs(mol_tmp)[0]
    try:
        AllChem.ComputeGasteigerCharges(mol_tmp)
    except Exception:
        pass

    n = mol_no_h.GetNumAtoms()
    feats = np.zeros((n, 14), dtype=np.float32)
    for idx in range(n):
        atom = mol_no_h.GetAtomWithIdx(idx)
        z = atom.GetAtomicNum()
        fc = float(atom.GetFormalCharge())
        total_hs = float(atom.GetTotalNumHs())
        is_no = z in (7, 8)
        g_q = 0.0
        try:
            a_tmp = mol_tmp.GetAtomWithIdx(idx)
            if a_tmp.HasProp("_GasteigerCharge"):
                val = float(a_tmp.GetProp("_GasteigerCharge"))
                if np.isfinite(val):
                    g_q = val
        except Exception:
            g_q = 0.0

        c_logp, c_mr = crippen_contribs[idx]
        feats[idx, 0] = 1.0 if (is_no and total_hs > 0.0) else 0.0
        feats[idx, 1] = 1.0 if (is_no and fc <= 0.0) else 0.0
        feats[idx, 2] = total_hs
        feats[idx, 3] = float(sum(1 for nb in atom.GetNeighbors() if nb.GetAtomicNum() > 1))
        feats[idx, 4] = fc
        feats[idx, 5] = 1.0 if atom.GetIsAromatic() else 0.0
        feats[idx, 6] = 1.0 if atom.IsInRing() else 0.0
        feats[idx, 7] = 1.0 if idx in macro_atoms else 0.0
        feats[idx, 8] = hyb_map.get(atom.GetHybridization(), 0.0) / 5.0
        feats[idx, 9] = float(c_logp)
        feats[idx, 10] = float(c_mr)
        feats[idx, 11] = float(tpsa_contribs[idx])
        feats[idx, 12] = float(labute_contribs[idx])
        feats[idx, 13] = g_q

    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _rdkit_bond_to_pxm_type(bond: Any) -> int:
    from rdkit import Chem

    bt = bond.GetBondType()
    if bt == Chem.BondType.SINGLE:
        return 1
    if bt == Chem.BondType.DOUBLE:
        return 2
    if bt == Chem.BondType.TRIPLE:
        return 3
    if bt == Chem.BondType.AROMATIC:
        return 4
    return 1


def _mol_to_graph_record(
    mol_or_smiles: Any,
    k_confs: int = 8,
    embed_if_needed: bool = True,
    seed: int = 2026,
    center_coords: bool = True,
    use_atom_phys: bool = True,
) -> Dict[str, Any]:
    """Convert an RDKit Mol or SMILES into heavy-atom multi-conformer tensors."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    if isinstance(mol_or_smiles, str):
        mol = Chem.MolFromSmiles(mol_or_smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {mol_or_smiles}")
    else:
        mol = Chem.Mol(mol_or_smiles)

    # Check if valid 3D conformers exist before altering Hs
    has_3d = False
    if mol.GetNumConformers() > 0:
        conf0 = mol.GetConformer(0)
        if conf0.Is3D():
            has_3d = True
        else:
            pos0 = np.asarray(conf0.GetPositions(), dtype=np.float32)
            has_3d = bool(np.std(pos0[:, 2]) > 1e-4)

    mol_no_h = Chem.RemoveHs(mol)
    n_heavy = mol_no_h.GetNumAtoms()
    if n_heavy == 0:
        raise ValueError("Molecule has zero heavy atoms.")

    confs_pos: List[np.ndarray] = []
    if has_3d:
        # Read heavy-atom coordinates from existing conformers
        heavy_map = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
        if len(heavy_map) != n_heavy:
            # Fallback to mol_no_h conformers directly
            n_c = min(mol_no_h.GetNumConformers(), k_confs)
            for cid in range(n_c):
                p = np.asarray(mol_no_h.GetConformer(cid).GetPositions(), dtype=np.float32)
                if center_coords:
                    p = p - p.mean(axis=0, keepdims=True)
                confs_pos.append(p)
        else:
            n_c = min(mol.GetNumConformers(), k_confs)
            for cid in range(n_c):
                full_p = np.asarray(mol.GetConformer(cid).GetPositions(), dtype=np.float32)
                p = full_p[heavy_map]
                if center_coords:
                    p = p - p.mean(axis=0, keepdims=True)
                confs_pos.append(p)
    elif embed_if_needed:
        mol_h = Chem.AddHs(mol_no_h)
        params = AllChem.ETKDGv3()
        params.randomSeed = int(seed)
        params.useMacrocycleTorsions = True
        params.pruneRmsThresh = 0.5
        n_embed = max(k_confs * 2, 16)
        cids = list(AllChem.EmbedMultipleConfs(mol_h, numConfs=n_embed, params=params))
        if len(cids) == 0:
            params.useRandomCoords = True
            cids = list(AllChem.EmbedMultipleConfs(mol_h, numConfs=n_embed, params=params))
        if len(cids) == 0:
            raise RuntimeError("Failed to embed 3D conformers with RDKit ETKDGv3.")

        energies: List[Tuple[float, int]] = []
        try:
            opt_res = AllChem.MMFFOptimizeMoleculeConfs(mol_h, mmffVariant="MMFF94s", maxIters=400)
            for cid, (not_conv, e_val) in zip(cids, opt_res):
                energies.append((float(e_val) if np.isfinite(e_val) else 1e9, int(cid)))
        except Exception:
            try:
                opt_res = AllChem.UFFOptimizeMoleculeConfs(mol_h, maxIters=400)
                for cid, (not_conv, e_val) in zip(cids, opt_res):
                    energies.append((float(e_val) if np.isfinite(e_val) else 1e9, int(cid)))
            except Exception:
                energies = [(float(i), int(cid)) for i, cid in enumerate(cids)]

        energies.sort(key=lambda x: x[0])
        selected = [cid for _, cid in energies[:k_confs]]
        heavy_map = [a.GetIdx() for a in mol_h.GetAtoms() if a.GetAtomicNum() > 1]
        for cid in selected:
            full_p = np.asarray(mol_h.GetConformer(cid).GetPositions(), dtype=np.float32)
            p = full_p[heavy_map]
            if center_coords:
                p = p - p.mean(axis=0, keepdims=True)
            confs_pos.append(p)
    else:
        raise ValueError("Molecule has no 3D conformer and embed_if_needed=False.")

    elements = np.asarray([a.GetAtomicNum() for a in mol_no_h.GetAtoms()], dtype=np.int64)
    src_list: List[int] = []
    dst_list: List[int] = []
    bt_list: List[int] = []
    for bond in mol_no_h.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = _rdkit_bond_to_pxm_type(bond)
        src_list.extend([i, j])
        dst_list.extend([j, i])
        bt_list.extend([bt, bt])

    bond_index = np.asarray([src_list, dst_list], dtype=np.int64) if src_list else np.zeros((2, 0), dtype=np.int64)
    bond_type = np.asarray(bt_list, dtype=np.int64)
    atom_phys = compute_rdkit_atom_phys_feat(mol_no_h) if use_atom_phys else None

    return {
        "pos_all_confs": np.stack(confs_pos, axis=0),  # [K, N, 3]
        "element": elements,                           # [N]
        "bond_index": bond_index,                      # [2, 2B]
        "bond_type": bond_type,                        # [2B]
        "atom_phys_feat": atom_phys,                   # [N, 14] or None
    }


def _load_pocket_tensors(pocket_pdb_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Featurize a PDB pocket using existing PocketXMol FeaturizePocket utility."""
    from easydict import EasyDict
    from utils.data import PocketMolData, torchify_dict
    from utils.parser import PDBProtein
    from utils.transforms import FeaturizePocket

    pocket_dict = PDBProtein(pocket_pdb_path).to_dict_atom()
    data = PocketMolData.from_pocket_mol_dicts(
        pocket_dict=torchify_dict(pocket_dict),
        mol_dict=None,
    )
    data.pdbid = os.path.basename(pocket_pdb_path)
    data.data_id = str(pocket_pdb_path)
    featurizer = FeaturizePocket(EasyDict({"knn": 32}))
    data = featurizer(data)
    # Restore PDB coordinate frame so holo ligand poses match pocket_pos directly
    raw_pos = data["pocket_pos"] + data["pocket_center"]
    pocket_pos = np.asarray(raw_pos.cpu().numpy(), dtype=np.float32)
    pocket_x = np.asarray(data["pocket_atom_feature"].cpu().numpy(), dtype=np.float32)
    return pocket_pos, pocket_x


def _init_embed_worker() -> None:
    """Keep CPU threads bounded inside spawned RDKit embedding workers."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)


def _embed_graph_record_worker(
    task: Tuple[Any, int, bool, int, bool, bool],
) -> Dict[str, Any]:
    """Top-level picklable worker for parallel conformer embedding in score_rdkit_mols."""
    mol_or_smiles, k_confs, embed_if_needed, seed, center_coords, use_atom_phys = task
    return _mol_to_graph_record(
        mol_or_smiles,
        k_confs=k_confs,
        embed_if_needed=embed_if_needed,
        seed=seed,
        center_coords=center_coords,
        use_atom_phys=use_atom_phys,
    )


class PermeabilityGuidanceScorer:
    """Clean Phase 3 differentiable guidance & RDKit scoring API for PermCycXMol."""

    def __init__(
        self,
        model: PermCycXMolModel,
        device: Union[str, torch.device] = "cuda",
        cham_names: Sequence[str] = CHAM_AUX_NAMES,
        use_cuda_graph: bool = True,
        max_cuda_graph_cache: int = 8,
    ) -> None:
        self.device = torch.device(device if (str(device) != "cuda" or torch.cuda.is_available()) else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.cham_names = tuple(cham_names)
        self.use_cuda_graph = bool(use_cuda_graph)
        self._max_cuda_graph_cache = max(1, int(max_cuda_graph_cache))
        self._pocket_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._assay_tensor_cache: Dict[Tuple[int, int], torch.Tensor] = {}
        self._cuda_graph_cache: OrderedDict[Tuple[Any, ...], Dict[str, Any]] = OrderedDict()
        self._cuda_graph_seen: set = set()
        self._cuda_graph_failed: set = set()
        self._cuda_graph_errors: Dict[Tuple[Any, ...], str] = {}

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: str,
        device: Union[str, torch.device] = "cuda",
        pxm_ckpt_path: Optional[str] = None,
        use_cuda_graph: bool = True,
    ) -> "PermeabilityGuidanceScorer":
        """Instantiate PermCycXMolModel + PermeabilityGuidanceScorer from a Phase 1 checkpoint."""
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model_cfg = ckpt.get("model_config", {})
        pxm_model_cfg = ckpt.get("pxm_model_cfg", None)

        encoder = ChameleonicPXMEncoder(
            pxm_ckpt_path=pxm_ckpt_path,
            load_pretrained=False,
            atom_phys_dim=int(model_cfg.get("atom_phys_dim", 14)),
            conf_dim=int(model_cfg.get("conf_dim", 256)),
            shield_hidden_dim=int(model_cfg.get("shield_hidden_dim", 128)),
            phys_dropout_prob=0.0,
            use_grad_checkpoint=False,
            max_graphs_per_chunk=int(model_cfg.get("max_graphs_per_chunk", 32)),
            max_edges_per_chunk=int(model_cfg.get("max_edges_per_chunk", 65536)),
            pxm_model_cfg=pxm_model_cfg,
        )
        model = PermCycXMolModel(
            encoder=encoder,
            trunk_dim=int(model_cfg.get("trunk_dim", 256)),
            g_dim=int(model_cfg.get("g_dim", 256)),
            cham_hidden_dim=int(model_cfg.get("cham_hidden_dim", 128)),
            assay_emb_dim=int(model_cfg.get("assay_emb_dim", 32)),
            max_sources=int(model_cfg.get("max_sources", 256)),
            use_cham_in_trunk=bool(model_cfg.get("use_cham_in_trunk", True)),
        )
        state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        model.load_state_dict(state, strict=True)
        return cls(model=model, device=device, use_cuda_graph=use_cuda_graph)

    def _get_pocket_on_device(self, pocket_pdb_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if pocket_pdb_path not in self._pocket_cache:
            pos_np, x_np = _load_pocket_tensors(pocket_pdb_path)
            self._pocket_cache[pocket_pdb_path] = (
                torch.from_numpy(pos_np).to(self.device),
                torch.from_numpy(x_np).to(self.device),
            )
        return self._pocket_cache[pocket_pdb_path]

    def unstandardize_cham(self, cham_norm: torch.Tensor) -> torch.Tensor:
        return cham_norm * self.model.cham_std.unsqueeze(0) + self.model.cham_mean.unsqueeze(0)

    def _cham_tensor_to_dict(self, cham_phys: np.ndarray) -> Dict[str, np.ndarray]:
        return {name: cham_phys[:, i] for i, name in enumerate(self.cham_names)}

    @torch.no_grad()
    def score_batch(
        self,
        batch: Dict[str, Any],
        assay: Optional[Union[str, int, torch.Tensor]] = None,
        k_mode: str = "ensemble",
    ) -> Dict[str, Any]:
        """Score a collated conformer batch without gradients."""
        self.model.eval()
        pos = batch["pos"].to(self.device)
        halfedge_index = batch["halfedge_index"].to(self.device)
        batch_node = batch["batch_node"].to(self.device)
        conf_to_mol = batch.get("conf_to_mol")
        if conf_to_mol is not None:
            conf_to_mol = conf_to_mol.to(self.device)

        num_mols = int(batch.get("num_mols", int(conf_to_mol.max().item()) + 1 if conf_to_mol is not None else int(batch_node.max().item()) + 1))

        if k_mode in ("first", "single", "k1") and conf_to_mol is not None:
            # Select first conformer graph per molecule cleanly
            keep_confs: List[int] = []
            for m_idx in range(num_mols):
                cands = torch.nonzero(conf_to_mol == m_idx, as_tuple=False).view(-1)
                if cands.numel() > 0:
                    keep_confs.append(int(cands[0].item()))
            keep_set = set(keep_confs)
            keep_mask_node = torch.as_tensor(
                [int(g.item()) in keep_set for g in batch_node], dtype=torch.bool, device=self.device
            )
            old_to_new_g = {old_g: new_g for new_g, old_g in enumerate(keep_confs)}
            new_node_idx = torch.full((pos.shape[0],), -1, dtype=torch.long, device=self.device)
            kept_nodes = torch.nonzero(keep_mask_node, as_tuple=False).view(-1)
            new_node_idx[kept_nodes] = torch.arange(kept_nodes.numel(), device=self.device)

            edge_keep = keep_mask_node[halfedge_index[0]] & keep_mask_node[halfedge_index[1]]
            halfedge_index = new_node_idx[halfedge_index[:, edge_keep]]
            pos = pos[kept_nodes]
            batch_node = torch.as_tensor(
                [old_to_new_g[int(g.item())] for g in batch_node[kept_nodes]],
                dtype=torch.long,
                device=self.device,
            )
            conf_to_mol = torch.arange(num_mols, dtype=torch.long, device=self.device)
            node_type = batch["node_type"].to(self.device)[kept_nodes] if batch.get("node_type") is not None else None
            element = batch["element"].to(self.device)[kept_nodes] if batch.get("element") is not None else None
            halfedge_type = batch["halfedge_type"].to(self.device)[edge_keep] if batch.get("halfedge_type") is not None else None
            atom_phys = batch["atom_phys_feat"].to(self.device)[kept_nodes] if batch.get("atom_phys_feat") is not None else None
            halfedge_batch = None
        else:
            node_type = batch.get("node_type")
            node_type = node_type.to(self.device) if node_type is not None else None
            element = batch.get("element")
            element = element.to(self.device) if element is not None else None
            halfedge_type = batch.get("halfedge_type")
            halfedge_type = halfedge_type.to(self.device) if halfedge_type is not None else None
            atom_phys = batch.get("atom_phys_feat")
            atom_phys = atom_phys.to(self.device) if atom_phys is not None else None
            halfedge_batch = batch.get("halfedge_batch")
            halfedge_batch = halfedge_batch.to(self.device) if halfedge_batch is not None else None

        if assay is not None:
            assay_id = _resolve_assay_tensor(assay, num_mols, self.device)
        elif batch.get("assay_id") is not None:
            assay_id = batch["assay_id"].to(self.device)
        else:
            assay_id = torch.ones((num_mols,), dtype=torch.long, device=self.device)

        pocket_pos = batch.get("pocket_pos")
        pocket_x = batch.get("pocket_x")
        batch_pocket = batch.get("batch_pocket")
        if pocket_pos is not None:
            pocket_pos = pocket_pos.to(self.device)
        if pocket_x is not None:
            pocket_x = pocket_x.to(self.device)
        if batch_pocket is not None:
            batch_pocket = batch_pocket.to(self.device)

        out = self.model(
            pos=pos,
            halfedge_index=halfedge_index,
            batch_node=batch_node,
            conf_to_mol=conf_to_mol,
            num_mols=num_mols,
            assay_id=assay_id,
            apply_source_offset=False,
            return_k1_view=False,
            node_type=node_type,
            element=element,
            halfedge_type=halfedge_type,
            halfedge_batch=halfedge_batch,
            atom_phys_feat=atom_phys,
            pocket_pos=pocket_pos,
            pocket_x=pocket_x,
            batch_pocket=batch_pocket,
        )
        cham_phys = self.unstandardize_cham(out["cham_pred"]).cpu().numpy()
        shield_np = {k: v.cpu().numpy() for k, v in out["shield_metrics"].items()}
        return {
            "perm_pred": out["y_abs_infer"].cpu().numpy(),
            "perm_base": out["y_base"].cpu().numpy(),
            "cham_pred_norm": out["cham_pred"].cpu().numpy(),
            "cham_pred_phys": cham_phys,
            "cham_metrics": self._cham_tensor_to_dict(cham_phys),
            "conformer_weights": out["weights"].cpu().numpy(),
            "g_perm": out["g"].cpu().numpy(),
            "z_boltz": out["z_boltz"].cpu().numpy(),
            "shield_metrics": shield_np,
        }

    @torch.no_grad()
    def score_rdkit_mols(
        self,
        mols_or_smiles: Sequence[Any],
        k_confs: int = 8,
        assay: Union[str, int] = "PAMPA",
        batch_size: int = 16,
        embed_if_needed: bool = True,
        seed: int = 2026,
        pocket_pdb_path: Optional[str] = None,
        use_atom_phys: bool = True,
        num_workers: int = 1,
    ) -> Dict[str, Any]:
        """Embed/read RDKit molecules and score permeability + chameleonic metrics."""
        center_coords = pocket_pdb_path is None
        tasks = [
            (m, k_confs, embed_if_needed, seed + i, center_coords, use_atom_phys)
            for i, m in enumerate(mols_or_smiles)
        ]
        if int(num_workers) > 1 and embed_if_needed and len(tasks) > 1:
            workers = min(int(num_workers), len(tasks))
            ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=ctx,
                initializer=_init_embed_worker,
            ) as executor:
                records = list(executor.map(_embed_graph_record_worker, tasks, chunksize=1))
        else:
            records = [_embed_graph_record_worker(t) for t in tasks]

        all_perm: List[np.ndarray] = []
        all_base: List[np.ndarray] = []
        all_cham_norm: List[np.ndarray] = []
        all_cham_phys: List[np.ndarray] = []
        all_g: List[np.ndarray] = []
        all_imhb: List[np.ndarray] = []
        all_shield: List[np.ndarray] = []
        all_rg: List[np.ndarray] = []
        all_force: List[np.ndarray] = []

        pocket_t = self._get_pocket_on_device(pocket_pdb_path) if pocket_pdb_path is not None else None

        for start in range(0, len(records), batch_size):
            chunk = records[start : start + batch_size]
            pos_list: List[torch.Tensor] = []
            node_type_list: List[torch.Tensor] = []
            batch_node_list: List[torch.Tensor] = []
            conf_to_mol_list: List[torch.Tensor] = []
            num_nodes_per_graph: List[int] = []
            bond_index_list: List[torch.Tensor] = []
            bond_type_list: List[torch.Tensor] = []
            phys_list: List[torch.Tensor] = []

            g_idx = 0
            node_offset = 0
            for m_idx, rec in enumerate(chunk):
                pos_k = rec["pos_all_confs"]  # [K_m, N_m, 3]
                k_m, n_m, _ = pos_k.shape
                nt = atomic_numbers_to_pxm_idx(torch.from_numpy(rec["element"]))
                bi = torch.from_numpy(rec["bond_index"])
                bt = torch.from_numpy(rec["bond_type"])
                ap = torch.from_numpy(rec["atom_phys_feat"]) if rec["atom_phys_feat"] is not None else None
                if ap is not None:
                    ap = (ap - self.model.atom_phys_mean.detach().cpu()) / torch.clamp(
                        self.model.atom_phys_std.detach().cpu(), min=1e-6
                    )

                for c_idx in range(k_m):
                    pos_list.append(torch.from_numpy(pos_k[c_idx]))
                    node_type_list.append(nt)
                    batch_node_list.append(torch.full((n_m,), g_idx, dtype=torch.long))
                    conf_to_mol_list.append(torch.tensor(m_idx, dtype=torch.long))
                    num_nodes_per_graph.append(n_m)
                    if bi.numel() > 0:
                        bond_index_list.append(bi + node_offset)
                        bond_type_list.append(bt)
                    if ap is not None:
                        phys_list.append(ap)
                    node_offset += n_m
                    g_idx += 1

            pos_b = torch.cat(pos_list, dim=0)
            nt_b = torch.cat(node_type_list, dim=0)
            bn_b = torch.cat(batch_node_list, dim=0)
            c2m_b = torch.stack(conf_to_mol_list, dim=0)
            bi_b = torch.cat(bond_index_list, dim=1) if bond_index_list else torch.zeros((2, 0), dtype=torch.long)
            bt_b = torch.cat(bond_type_list, dim=0) if bond_type_list else torch.zeros((0,), dtype=torch.long)
            he_idx, he_type, he_batch = build_complete_halfedges(
                num_nodes_per_graph=num_nodes_per_graph,
                bond_index=bi_b,
                bond_type=bt_b,
            )

            batch_dict: Dict[str, Any] = {
                "pos": pos_b,
                "node_type": nt_b,
                "halfedge_index": he_idx,
                "halfedge_type": he_type,
                "halfedge_batch": he_batch,
                "batch_node": bn_b,
                "conf_to_mol": c2m_b,
                "num_mols": len(chunk),
                "atom_phys_feat": torch.cat(phys_list, dim=0) if phys_list else None,
            }

            if pocket_t is not None:
                p_pos_1, p_x_1 = pocket_t
                n_p = int(p_pos_1.shape[0])
                if n_p > 0:
                    p_center = p_pos_1.mean(dim=0, keepdim=True)
                    p_pos_1 = p_pos_1 - p_center
                    batch_dict["pos"] = batch_dict["pos"] - p_center.to(batch_dict["pos"].device)
                batch_dict["pocket_pos"] = p_pos_1.unsqueeze(0).expand(g_idx, n_p, 3).reshape(-1, 3)
                batch_dict["pocket_x"] = p_x_1.unsqueeze(0).expand(g_idx, n_p, p_x_1.shape[1]).reshape(-1, p_x_1.shape[1])
                batch_dict["batch_pocket"] = torch.arange(g_idx, device=self.device).repeat_interleave(n_p)

            res = self.score_batch(batch_dict, assay=assay, k_mode="ensemble")
            all_perm.append(res["perm_pred"])
            all_base.append(res["perm_base"])
            all_cham_norm.append(res["cham_pred_norm"])
            all_cham_phys.append(res["cham_pred_phys"])
            all_g.append(res["g_perm"])

            # Boltzmann-average conformer-level smooth metrics per molecule
            w_t = torch.from_numpy(res["conformer_weights"])
            for lst, key in (
                (all_imhb, "smooth_imhb_trans"),
                (all_shield, "smooth_polar_shield"),
                (all_rg, "smooth_rg"),
                (all_force, "pxm_force_norm"),
            ):
                val_t = torch.from_numpy(res["shield_metrics"][key])
                agg = torch.zeros((len(chunk),), dtype=torch.float32)
                agg.index_add_(0, c2m_b, w_t * val_t)
                lst.append(agg.numpy())

        cham_phys_cat = np.concatenate(all_cham_phys, axis=0)
        return {
            "perm_pred": np.concatenate(all_perm, axis=0),
            "perm_base": np.concatenate(all_base, axis=0),
            "cham_pred_norm": np.concatenate(all_cham_norm, axis=0),
            "cham_pred_phys": cham_phys_cat,
            "cham_metrics": self._cham_tensor_to_dict(cham_phys_cat),
            "g_perm": np.concatenate(all_g, axis=0),
            "smooth_imhb_trans": np.concatenate(all_imhb, axis=0),
            "smooth_polar_shield": np.concatenate(all_shield, axis=0),
            "smooth_rg": np.concatenate(all_rg, axis=0),
            "pxm_force_norm": np.concatenate(all_force, axis=0),
        }

    def _get_cached_assay_tensor(
        self,
        assay: Union[str, int, torch.Tensor],
        num_mols: int,
    ) -> Tuple[torch.Tensor, Union[int, str]]:
        if not torch.is_tensor(assay):
            a_id = int(ASSAY_TO_ID[canonicalize_assay_name(assay)]) if isinstance(assay, str) else int(assay)
            key = (a_id, int(num_mols))
            cached = self._assay_tensor_cache.get(key)
            if cached is None:
                if len(self._assay_tensor_cache) >= 64:
                    self._assay_tensor_cache.clear()
                cached = torch.full((num_mols,), a_id, dtype=torch.long, device=self.device)
                self._assay_tensor_cache[key] = cached
            return cached, a_id
        return _resolve_assay_tensor(assay, num_mols, self.device), "tensor"

    def _eval_guidance_core(
        self,
        pos_in: torch.Tensor,
        nl_in: Optional[torch.Tensor],
        hl_in: Optional[torch.Tensor],
        halfedge_index: torch.Tensor,
        batch_node: torch.Tensor,
        conf_to_mol: Optional[torch.Tensor],
        assay_id: torch.Tensor,
        num_graphs: int,
        num_mols: int,
        max_nodes_per_graph: Optional[int],
        objective: str,
        node_type: Optional[torch.Tensor] = None,
        halfedge_type: Optional[torch.Tensor] = None,
        atom_phys_feat: Optional[torch.Tensor] = None,
        is_peptide: Optional[torch.Tensor] = None,
        fixed_node: Optional[torch.Tensor] = None,
        fixed_pos: Optional[torch.Tensor] = None,
        pocket_pos: Optional[torch.Tensor] = None,
        pocket_x: Optional[torch.Tensor] = None,
        batch_pocket: Optional[torch.Tensor] = None,
        ref_embedding: Optional[torch.Tensor] = None,
        ref_perm: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        grad_targets: List[torch.Tensor] = [pos_in]
        if nl_in is not None:
            grad_targets.append(nl_in)
        if hl_in is not None:
            grad_targets.append(hl_in)

        halfedge_batch = (
            batch_node[halfedge_index[0]] if (num_graphs > 1 and halfedge_index.numel() > 0) else None
        )

        with torch.enable_grad():
            out = self.model(
                pos=pos_in,
                halfedge_index=halfedge_index,
                batch_node=batch_node,
                conf_to_mol=conf_to_mol,
                num_mols=num_mols,
                assay_id=assay_id,
                apply_source_offset=False,
                return_k1_view=False,
                num_graphs=num_graphs,
                max_nodes_per_graph=max_nodes_per_graph,
                node_type=node_type,
                node_logits=nl_in,
                halfedge_type=halfedge_type,
                halfedge_logits=hl_in,
                halfedge_batch=halfedge_batch,
                atom_phys_feat=atom_phys_feat,
                is_peptide=is_peptide,
                fixed_node=fixed_node,
                fixed_pos=fixed_pos,
                pocket_pos=pocket_pos,
                pocket_x=pocket_x,
                batch_pocket=batch_pocket,
            )

            if objective == "perm":
                score = out["y_abs_infer"]
            elif objective == "perm_base":
                score = out["y_base"]
            elif objective == "closure":
                score = out["cham_pred"][:, 4]
            elif objective == "delta":
                if ref_embedding is None:
                    raise ValueError("ref_embedding is required for objective='delta'.")
                g_r = ref_embedding.to(self.device)
                if g_r.dim() == 1:
                    g_r = g_r.unsqueeze(0).expand(num_mols, -1)
                if ref_perm is None:
                    y_r = self.model.predict_abs_for_assay(g_r, int(assay_id[0].item()))
                else:
                    y_r = ref_perm.to(self.device).view(-1)
                    if y_r.numel() == 1 and num_mols > 1:
                        y_r = y_r.expand(num_mols)
                score = self.model.predict_delta(
                    g_q=out["g"],
                    g_r=g_r,
                    y_q=out["y_abs_infer"],
                    y_r=y_r,
                    assay_id=assay_id,
                )
            else:
                raise ValueError(f"Unsupported guidance objective: {objective}")

            grads = torch.autograd.grad(score.sum(), grad_targets, allow_unused=False)

        grad_pos = grads[0].detach().contiguous()
        ptr = 1
        grad_nl = None
        grad_hl = None
        if nl_in is not None:
            grad_nl = grads[ptr].detach().contiguous()
            ptr += 1
        if hl_in is not None:
            grad_hl = grads[ptr].detach().contiguous()

        cham_norm = out["cham_pred"].detach().contiguous()
        cham_phys = self.unstandardize_cham(cham_norm).contiguous()
        return (
            score.detach().contiguous(),
            out["y_abs_infer"].detach().contiguous(),
            out["y_base"].detach().contiguous(),
            cham_norm,
            cham_phys,
            out["g"].detach().contiguous(),
            grad_pos,
            grad_nl,
            grad_hl,
        )

    def _run_guidance_cuda_graph(
        self,
        cg_key: Tuple[Any, ...],
        pos_dev: torch.Tensor,
        nl_dev: Optional[torch.Tensor],
        hl_dev: Optional[torch.Tensor],
        halfedge_index_dev: torch.Tensor,
        batch_node_dev: torch.Tensor,
        conf_to_mol_dev: Optional[torch.Tensor],
        assay_id_dev: torch.Tensor,
        assay_key: Union[int, str],
        num_graphs: int,
        num_mols: int,
        max_nodes_per_graph: Optional[int],
        objective: str,
    ) -> Optional[Tuple[Optional[torch.Tensor], ...]]:
        entry = self._cuda_graph_cache.get(cg_key)
        if entry is not None:
            self._cuda_graph_cache.move_to_end(cg_key)
            with torch.no_grad():
                entry["static_pos"].copy_(pos_dev.detach())
                if entry["static_nl"] is not None:
                    entry["static_nl"].copy_(nl_dev.detach())
                if entry["static_hl"] is not None:
                    entry["static_hl"].copy_(hl_dev.detach())
                entry["static_he_idx"].copy_(halfedge_index_dev)
                entry["static_batch_node"].copy_(batch_node_dev)
                if entry["static_conf_to_mol"] is not None:
                    entry["static_conf_to_mol"].copy_(conf_to_mol_dev)
                if assay_key == "tensor":
                    entry["static_assay_id"].copy_(assay_id_dev)
            entry["graph"].replay()
            return tuple(t.detach().clone() if t is not None else None for t in entry["out_tensors"])

        if cg_key in self._cuda_graph_failed:
            return None
        if cg_key not in self._cuda_graph_seen:
            self._cuda_graph_seen.add(cg_key)
            return None

        try:
            static_pos = pos_dev.detach().clone().requires_grad_(True)
            static_nl = nl_dev.detach().clone().requires_grad_(True) if nl_dev is not None else None
            static_hl = hl_dev.detach().clone().requires_grad_(True) if hl_dev is not None else None
            static_he_idx = halfedge_index_dev.detach().clone()
            static_batch_node = batch_node_dev.detach().clone()
            static_conf_to_mol = conf_to_mol_dev.detach().clone() if conf_to_mol_dev is not None else None
            static_assay_id = assay_id_dev.detach().clone()

            cap_stream = torch.cuda.Stream(device=self.device)
            cap_stream.wait_stream(torch.cuda.current_stream(device=self.device))
            with torch.cuda.stream(cap_stream):
                # Warm up forward + autograd backward on capture stream before recording graph
                _ = self._eval_guidance_core(
                    pos_in=static_pos,
                    nl_in=static_nl,
                    hl_in=static_hl,
                    halfedge_index=static_he_idx,
                    batch_node=static_batch_node,
                    conf_to_mol=static_conf_to_mol,
                    assay_id=static_assay_id,
                    num_graphs=num_graphs,
                    num_mols=num_mols,
                    max_nodes_per_graph=max_nodes_per_graph,
                    objective=objective,
                )
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=cap_stream):
                    out_tensors = self._eval_guidance_core(
                        pos_in=static_pos,
                        nl_in=static_nl,
                        hl_in=static_hl,
                        halfedge_index=static_he_idx,
                        batch_node=static_batch_node,
                        conf_to_mol=static_conf_to_mol,
                        assay_id=static_assay_id,
                        num_graphs=num_graphs,
                        num_mols=num_mols,
                        max_nodes_per_graph=max_nodes_per_graph,
                        objective=objective,
                    )
            torch.cuda.current_stream(device=self.device).wait_stream(cap_stream)
            graph.replay()

            while len(self._cuda_graph_cache) >= self._max_cuda_graph_cache:
                self._cuda_graph_cache.popitem(last=False)
            self._cuda_graph_cache[cg_key] = {
                "graph": graph,
                "static_pos": static_pos,
                "static_nl": static_nl,
                "static_hl": static_hl,
                "static_he_idx": static_he_idx,
                "static_batch_node": static_batch_node,
                "static_conf_to_mol": static_conf_to_mol,
                "static_assay_id": static_assay_id,
                "out_tensors": out_tensors,
            }
            return tuple(t.detach().clone() if t is not None else None for t in out_tensors)
        except Exception as exc:
            self._cuda_graph_failed.add(cg_key)
            self._cuda_graph_errors[cg_key] = repr(exc)
            return None

    def guidance_gradients(
        self,
        pos: torch.Tensor,
        node_logits: Optional[torch.Tensor] = None,
        halfedge_logits: Optional[torch.Tensor] = None,
        halfedge_index: Optional[torch.Tensor] = None,
        batch_node: Optional[torch.Tensor] = None,
        conf_to_mol: Optional[torch.Tensor] = None,
        node_type: Optional[torch.Tensor] = None,
        halfedge_type: Optional[torch.Tensor] = None,
        assay: Union[str, int, torch.Tensor] = "PAMPA",
        objective: str = "perm",
        ref_embedding: Optional[torch.Tensor] = None,
        ref_perm: Optional[torch.Tensor] = None,
        pocket_pos: Optional[torch.Tensor] = None,
        pocket_x: Optional[torch.Tensor] = None,
        batch_pocket: Optional[torch.Tensor] = None,
        is_peptide: Optional[torch.Tensor] = None,
        fixed_node: Optional[torch.Tensor] = None,
        fixed_pos: Optional[torch.Tensor] = None,
        atom_phys_feat: Optional[torch.Tensor] = None,
        num_graphs: Optional[int] = None,
        num_mols: Optional[int] = None,
        sync_timing: bool = False,
        use_cuda_graph: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Compute exact Phase 3 guidance gradients wrt `pos`, `node_logits`, and `halfedge_logits`."""
        self.model.eval()
        if sync_timing and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()

        pos_dev = pos.to(self.device)
        n_total = int(pos_dev.shape[0])

        if batch_node is None:
            batch_node_dev = torch.zeros((n_total,), dtype=torch.long, device=self.device)
            ng_resolved = 1 if n_total > 0 else 0
            max_nodes_g: Optional[int] = n_total
        else:
            batch_node_dev = batch_node.to(self.device)
            if num_graphs is not None:
                ng_resolved = int(num_graphs)
            else:
                ng_resolved = int(batch_node_dev.max().item()) + 1 if batch_node_dev.numel() > 0 else 0
            max_nodes_g = n_total if ng_resolved <= 1 else None

        if halfedge_index is None:
            if ng_resolved <= 1:
                counts_list = [n_total] if n_total > 0 else []
            else:
                counts_t = torch.bincount(batch_node_dev, minlength=ng_resolved)
                counts_list = counts_t.tolist()
                max_nodes_g = max(counts_list) if counts_list else 0
            halfedge_index_dev, _, _ = build_complete_halfedges(counts_list, device=self.device)
        else:
            halfedge_index_dev = halfedge_index.to(self.device)

        nl_dev = node_logits.to(self.device) if node_logits is not None else None
        hl_dev = halfedge_logits.to(self.device) if halfedge_logits is not None else None

        if conf_to_mol is not None:
            conf_to_mol_dev: Optional[torch.Tensor] = conf_to_mol.to(self.device)
            if num_mols is not None:
                num_mols_resolved = int(num_mols)
            else:
                num_mols_resolved = int(conf_to_mol_dev.max().item()) + 1 if conf_to_mol_dev.numel() > 0 else 0
        else:
            conf_to_mol_dev = None
            num_mols_resolved = ng_resolved

        assay_id_dev, assay_key = self._get_cached_assay_tensor(assay, num_mols_resolved)

        use_cg = self.use_cuda_graph if use_cuda_graph is None else bool(use_cuda_graph)
        chunk_g = int(self.model.encoder.max_graphs_per_chunk)
        chunk_e = int(getattr(self.model.encoder, "max_edges_per_chunk", 0))
        e_half = int(halfedge_index_dev.shape[1])
        no_chunk_split = (
            ng_resolved <= 1
            or ((chunk_g <= 0 or ng_resolved <= chunk_g) and (chunk_e <= 0 or (2 * e_half) <= chunk_e))
        )
        cg_eligible = (
            use_cg
            and self.device.type == "cuda"
            and ng_resolved >= 1
            and n_total >= 1
            and no_chunk_split
            and pocket_pos is None
            and pocket_x is None
            and batch_pocket is None
            and ref_embedding is None
            and ref_perm is None
            and objective != "delta"
            and atom_phys_feat is None
            and is_peptide is None
            and fixed_node is None
            and fixed_pos is None
            and node_type is None
            and halfedge_type is None
            and (nl_dev is not None or hl_dev is not None)
        )

        results_tuple: Optional[Tuple[Optional[torch.Tensor], ...]] = None
        if cg_eligible:
            if max_nodes_g is None:
                counts_t = torch.bincount(batch_node_dev, minlength=ng_resolved)
                max_nodes_g = int(counts_t.max().item()) if counts_t.numel() > 0 else 0
            cg_key = (
                n_total,
                e_half,
                ng_resolved,
                num_mols_resolved,
                max_nodes_g,
                nl_dev is not None,
                hl_dev is not None,
                conf_to_mol_dev is not None,
                assay_key,
                objective,
                str(pos_dev.dtype),
                str(nl_dev.dtype) if nl_dev is not None else None,
                str(hl_dev.dtype) if hl_dev is not None else None,
                self.device.index,
            )
            results_tuple = self._run_guidance_cuda_graph(
                cg_key=cg_key,
                pos_dev=pos_dev,
                nl_dev=nl_dev,
                hl_dev=hl_dev,
                halfedge_index_dev=halfedge_index_dev,
                batch_node_dev=batch_node_dev,
                conf_to_mol_dev=conf_to_mol_dev,
                assay_id_dev=assay_id_dev,
                assay_key=assay_key,
                num_graphs=ng_resolved,
                num_mols=num_mols_resolved,
                max_nodes_per_graph=max_nodes_g,
                objective=objective,
            )

        if results_tuple is None:
            pos_in = pos_dev if pos_dev.requires_grad else pos_dev.detach().requires_grad_(True)
            nl_in = (
                (nl_dev if nl_dev.requires_grad else nl_dev.detach().requires_grad_(True))
                if nl_dev is not None
                else None
            )
            hl_in = (
                (hl_dev if hl_dev.requires_grad else hl_dev.detach().requires_grad_(True))
                if hl_dev is not None
                else None
            )
            results_tuple = self._eval_guidance_core(
                pos_in=pos_in,
                nl_in=nl_in,
                hl_in=hl_in,
                halfedge_index=halfedge_index_dev,
                batch_node=batch_node_dev,
                conf_to_mol=conf_to_mol_dev,
                assay_id=assay_id_dev,
                num_graphs=ng_resolved,
                num_mols=num_mols_resolved,
                max_nodes_per_graph=max_nodes_g,
                objective=objective,
                node_type=node_type.to(self.device) if node_type is not None else None,
                halfedge_type=halfedge_type.to(self.device) if halfedge_type is not None else None,
                atom_phys_feat=atom_phys_feat.to(self.device) if atom_phys_feat is not None else None,
                is_peptide=is_peptide.to(self.device) if is_peptide is not None else None,
                fixed_node=fixed_node.to(self.device) if fixed_node is not None else None,
                fixed_pos=fixed_pos.to(self.device) if fixed_pos is not None else None,
                pocket_pos=pocket_pos.to(self.device) if pocket_pos is not None else None,
                pocket_x=pocket_x.to(self.device) if pocket_x is not None else None,
                batch_pocket=batch_pocket.to(self.device) if batch_pocket is not None else None,
                ref_embedding=ref_embedding,
                ref_perm=ref_perm,
            )

        if sync_timing and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        (
            score_t,
            perm_pred_t,
            perm_base_t,
            cham_norm_t,
            cham_phys_t,
            g_perm_t,
            grad_pos_t,
            grad_nl_t,
            grad_hl_t,
        ) = results_tuple
        return {
            "score": score_t,
            "perm_pred": perm_pred_t,
            "perm_base": perm_base_t,
            "cham_pred_norm": cham_norm_t,
            "cham_pred_phys": cham_phys_t,
            "g_perm": g_perm_t,
            "grad_pos": grad_pos_t,
            "grad_node_logits": grad_nl_t,
            "grad_halfedge_logits": grad_hl_t,
            "latency_ms": float(latency_ms),
        }

    def benchmark_guidance_latency(
        self,
        num_atoms: int = 64,
        batch_sizes: Sequence[int] = (1, 4, 8),
        warmup: int = 5,
        iters: int = 20,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Benchmark guidance_gradients wall time (ms/batch and ms/mol) on synthetic 3D peptides."""
        gen = torch.Generator(device=self.device)
        gen.manual_seed(seed)
        results: Dict[str, Any] = {"device": str(self.device), "num_atoms": int(num_atoms), "batches": {}}
        is_cuda = self.device.type == "cuda"

        for bsz in batch_sizes:
            counts = [num_atoms] * bsz
            he_idx, _, _ = build_complete_halfedges(counts, device=self.device)
            batch_node = torch.arange(bsz, device=self.device).repeat_interleave(num_atoms)
            pos = torch.randn((bsz * num_atoms, 3), device=self.device, generator=gen) * 4.0
            node_logits = torch.randn((bsz * num_atoms, NUM_PXM_NODE_TYPES), device=self.device, generator=gen)
            halfedge_logits = torch.randn((he_idx.shape[1], NUM_PXM_EDGE_TYPES), device=self.device, generator=gen)

            for _ in range(warmup):
                _ = self.guidance_gradients(
                    pos=pos,
                    node_logits=node_logits,
                    halfedge_logits=halfedge_logits,
                    halfedge_index=he_idx,
                    batch_node=batch_node,
                    assay="PAMPA",
                    sync_timing=False,
                )

            if is_cuda:
                torch.cuda.synchronize(self.device)
                events = [
                    (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                    for _ in range(iters)
                ]
            else:
                events = []

            cpu_latencies: List[float] = []
            t_loop0 = time.perf_counter()
            for i in range(iters):
                if is_cuda:
                    ev0, ev1 = events[i]
                    ev0.record()
                else:
                    t_iter0 = time.perf_counter()
                _ = self.guidance_gradients(
                    pos=pos,
                    node_logits=node_logits,
                    halfedge_logits=halfedge_logits,
                    halfedge_index=he_idx,
                    batch_node=batch_node,
                    assay="PAMPA",
                    sync_timing=False,
                )
                if is_cuda:
                    ev1.record()
                else:
                    cpu_latencies.append((time.perf_counter() - t_iter0) * 1000.0)

            if is_cuda:
                torch.cuda.synchronize(self.device)
                loop_wall_ms = (time.perf_counter() - t_loop0) * 1000.0
                mean_ms = float(loop_wall_ms / max(1, iters))
                arr = np.asarray([ev0.elapsed_time(ev1) for ev0, ev1 in events], dtype=np.float64)
            else:
                arr = np.asarray(cpu_latencies, dtype=np.float64)
                mean_ms = float(arr.mean())

            results["batches"][str(bsz)] = {
                "ms_per_batch_mean": mean_ms,
                "ms_per_batch_p50": float(np.percentile(arr, 50)),
                "ms_per_batch_p95": float(np.percentile(arr, 95)),
                "ms_per_mol_mean": float(mean_ms / bsz),
            }
        return results
