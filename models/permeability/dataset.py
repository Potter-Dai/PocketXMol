"""LMDB dataset, source-grouped pair sampler, and collation for Phase 1 PermCycXMol."""

import json
import os
import pickle
from typing import Any, Dict, List, Optional, Sequence, Tuple

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from models.permeability.constants import (
    ASSAY_TO_ID,
    CHAM_AUX_NAMES,
    DEFAULT_CHAM8_ORDER,
    atomic_numbers_to_pxm_idx,
    canonicalize_assay_name,
)
from models.permeability.encoder import build_complete_halfedges


def _pick_first_present(d: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def load_lmdb_metadata(lmdb_path: str) -> Dict[str, Any]:
    """Read b'__metadata__' pickle or json from perm_v1.lmdb if present."""
    env = lmdb.open(
        lmdb_path,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        subdir=os.path.isdir(lmdb_path),
    )
    try:
        with env.begin(write=False) as txn:
            raw = txn.get(b"__metadata__")
            if raw is None:
                raw = txn.get(b"metadata")
            if raw is None:
                return {}
            try:
                return pickle.loads(raw)
            except Exception:
                return json.loads(raw.decode("utf-8"))
    finally:
        env.close()


def extract_cham_5_raw(
    cham_raw: Any,
    cham8_names: Sequence[str] = DEFAULT_CHAM8_ORDER,
) -> np.ndarray:
    """Extract [psa_3d_min, delta_psa_3d, imhb_max, rg_min, closure] from LMDB entry."""
    if isinstance(cham_raw, dict):
        psa_min = float(cham_raw.get("psa_3d_min", np.nan))
        delta_psa = float(cham_raw.get("delta_psa_3d", np.nan))
        imhb_max = float(cham_raw.get("imhb_max", np.nan))
        rg_min = float(cham_raw.get("rg_min", np.nan))
        closure = float(cham_raw.get("closure", cham_raw.get("closure_mol", np.nan)))
        return np.asarray([psa_min, delta_psa, imhb_max, rg_min, closure], dtype=np.float32)

    arr = np.asarray(cham_raw, dtype=np.float32).reshape(-1)
    name_to_idx = {str(n): i for i, n in enumerate(cham8_names)}
    psa_min = float(arr[name_to_idx.get("psa_3d_min", 0)]) if len(arr) > 0 else np.nan
    delta_psa = float(arr[name_to_idx.get("delta_psa_3d", 2)]) if len(arr) > 2 else np.nan
    imhb_max = float(arr[name_to_idx.get("imhb_max", 4)]) if len(arr) > 4 else np.nan
    rg_min = float(arr[name_to_idx.get("rg_min", 6)]) if len(arr) > 6 else np.nan
    closure = float(arr[name_to_idx["closure"]]) if "closure" in name_to_idx and name_to_idx["closure"] < len(arr) else np.nan
    return np.asarray([psa_min, delta_psa, imhb_max, rg_min, closure], dtype=np.float32)


def _find_norm_block(
    meta: Dict[str, Any],
    splits_meta: Dict[str, Any],
    split_family: str,
) -> Optional[Dict[str, Any]]:
    stat_key_candidates = [
        f"{split_family}_train",
        f"{split_family}_train_stats",
        split_family,
    ]
    for src in (meta, splits_meta):
        if not isinstance(src, dict):
            continue
        for sub_root in (src.get("norm_stats"), src.get("stats"), src):
            if not isinstance(sub_root, dict):
                continue
            for k in stat_key_candidates:
                if k in sub_root and isinstance(sub_root[k], dict):
                    return sub_root[k]
    return None


def build_normalizers(
    meta: Dict[str, Any],
    splits_meta: Dict[str, Any],
    split_family: str = "od",
    cham8_names: Sequence[str] = DEFAULT_CHAM8_ORDER,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resolve train-only (mean, std) for cham[5], cham8[8], and atom_phys_feat[14].

    Supports both array-based `cham_targets: {"mean": [...], "std": [...]}` + `closure_mol`
    and named-dict formats.
    """
    cham5_mean = np.zeros((5,), dtype=np.float32)
    cham5_std = np.ones((5,), dtype=np.float32)
    cham8_mean = np.zeros((len(cham8_names),), dtype=np.float32)
    cham8_std = np.ones((len(cham8_names),), dtype=np.float32)
    phys_mean = np.zeros((14,), dtype=np.float32)
    phys_std = np.ones((14,), dtype=np.float32)

    block = _find_norm_block(meta, splits_meta, split_family)
    if block is None:
        return cham5_mean, cham5_std, cham8_mean, cham8_std, phys_mean, phys_std

    # 1. Atom phys stats [14]
    ap = block.get("atom_phys_feat")
    if isinstance(ap, dict) and "mean" in ap and "std" in ap:
        pm = np.asarray(ap["mean"], dtype=np.float32).reshape(-1)
        ps = np.asarray(ap["std"], dtype=np.float32).reshape(-1)
        if len(pm) == 14 and len(ps) == 14:
            phys_mean = pm
            phys_std = np.clip(ps, 1e-6, None)

    # 2. Chameleonic stats
    name_to_idx8 = {str(n): i for i, n in enumerate(cham8_names)}
    if "cham_mean" in block and "cham_std" in block:
        cm = np.asarray(block["cham_mean"], dtype=np.float32).reshape(-1)
        cs = np.clip(np.asarray(block["cham_std"], dtype=np.float32).reshape(-1), 1e-6, None)
        if len(cm) >= 5:
            cham5_mean = cm[:5]
            cham5_std = cs[:5]

    ct = block.get("cham_targets", block.get("cham"))
    if isinstance(ct, dict):
        if "mean" in ct and "std" in ct and isinstance(ct["mean"], (list, tuple, np.ndarray)):
            m8 = np.asarray(ct["mean"], dtype=np.float32).reshape(-1)
            s8 = np.clip(np.asarray(ct["std"], dtype=np.float32).reshape(-1), 1e-6, None)
            if len(m8) >= 8:
                cham8_mean[:8] = m8[:8]
                cham8_std[:8] = s8[:8]
                i_psa = name_to_idx8.get("psa_3d_min", 0)
                i_dpsa = name_to_idx8.get("delta_psa_3d", 2)
                i_imhb = name_to_idx8.get("imhb_max", 4)
                i_rg = name_to_idx8.get("rg_min", 6)
                cham5_mean[:4] = np.asarray([m8[i_psa], m8[i_dpsa], m8[i_imhb], m8[i_rg]], dtype=np.float32)
                cham5_std[:4] = np.asarray([s8[i_psa], s8[i_dpsa], s8[i_imhb], s8[i_rg]], dtype=np.float32)
        elif all(k in ct for k in ("psa_3d_min", "delta_psa_3d", "imhb_max", "rg_min")):
            for j, k in enumerate(cham8_names):
                if k in ct and isinstance(ct[k], dict):
                    cham8_mean[j] = float(ct[k]["mean"])
                    cham8_std[j] = max(float(ct[k]["std"]), 1e-6)
            for j, k in enumerate(CHAM_AUX_NAMES[:4]):
                cham5_mean[j] = float(ct[k]["mean"])
                cham5_std[j] = max(float(ct[k]["std"]), 1e-6)

    cl = block.get("closure_mol", block.get("closure"))
    if isinstance(cl, dict) and "mean" in cl and "std" in cl:
        cham5_mean[4] = float(cl["mean"])
        cham5_std[4] = max(float(cl["std"]), 1e-6)
    elif isinstance(ct, dict) and "closure" in ct and isinstance(ct["closure"], dict):
        cham5_mean[4] = float(ct["closure"]["mean"])
        cham5_std[4] = max(float(ct["closure"]["std"]), 1e-6)

    return cham5_mean, cham5_std, cham8_mean, cham8_std, phys_mean, phys_std


def resolve_ref_pairs_from_splits(
    splits_data: Dict[str, Any],
    suite_name: str,
    records: List[Dict[str, Any]],
) -> Optional[List[Tuple[np.ndarray, np.ndarray]]]:
    """Parse CycDeltaBench fixed 10-seed ref_pairs from splits_v1.json if available."""
    ref_container = splits_data.get("ref_pairs")
    if not isinstance(ref_container, dict) or suite_name not in ref_container:
        return None
    suite_rp = ref_container[suite_name]
    if not isinstance(suite_rp, dict):
        return None

    rec_id_to_local = {str(r.get("record_id", "")): i for i, r in enumerate(records)}
    seed_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    for seed_idx in range(10):
        s_key = f"seed_{seed_idx}"
        if s_key not in suite_rp:
            s_key = str(seed_idx)
        if s_key not in suite_rp:
            return None
        entry = suite_rp[s_key]

        q_list: List[int] = []
        r_list: List[int] = []
        # Format 1: list of [query_id, ref_id] pairs or dicts {"query": ..., "ref": ...}
        if isinstance(entry, list):
            for pair in entry:
                if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                    qv, rv = pair[0], pair[1]
                elif isinstance(pair, dict):
                    qv = _pick_first_present(pair, ("query_id", "query_record_id", "q_idx", "query", "q"))
                    rv = _pick_first_present(pair, ("ref_id", "ref_record_id", "r_idx", "ref", "r"))
                else:
                    continue
                qi = rec_id_to_local.get(str(qv), qv if isinstance(qv, int) else None)
                ri = rec_id_to_local.get(str(rv), rv if isinstance(rv, int) else None)
                if isinstance(qi, int) and isinstance(ri, int) and 0 <= qi < len(records) and 0 <= ri < len(records):
                    q_list.append(qi)
                    r_list.append(ri)
        # Format 2: dict with {"pairs": [...]} or {"query_ids": [...], "ref_ids": [...]}
        elif isinstance(entry, dict):
            if "pairs" in entry:
                for pair in entry["pairs"]:
                    qv, rv = pair[0], pair[1]
                    qi = rec_id_to_local.get(str(qv), qv if isinstance(qv, int) else None)
                    ri = rec_id_to_local.get(str(rv), rv if isinstance(rv, int) else None)
                    if isinstance(qi, int) and isinstance(ri, int) and 0 <= qi < len(records) and 0 <= ri < len(records):
                        q_list.append(qi)
                        r_list.append(ri)
            elif "query_ids" in entry and "ref_ids" in entry:
                for qv, rv in zip(entry["query_ids"], entry["ref_ids"]):
                    qi = rec_id_to_local.get(str(qv), qv if isinstance(qv, int) else None)
                    ri = rec_id_to_local.get(str(rv), rv if isinstance(rv, int) else None)
                    if isinstance(qi, int) and isinstance(ri, int) and 0 <= qi < len(records) and 0 <= ri < len(records):
                        q_list.append(qi)
                        r_list.append(ri)

        if not q_list:
            return None
        seed_pairs.append((np.asarray(q_list, dtype=np.int64), np.asarray(r_list, dtype=np.int64)))

    return seed_pairs if len(seed_pairs) > 0 else None


class PermLMDBDataset(Dataset):
    """Dataset reading multi-conformer cyclic peptides from perm_v1.lmdb + splits_v1.json."""

    def __init__(
        self,
        lmdb_path: str,
        splits_json_path: str,
        split_name: str,
        split_family: Optional[str] = None,
        source_vocab: Optional[Dict[str, int]] = None,
        k_confs: int = 8,
        training: bool = False,
        standardize_atom_phys: bool = True,
    ) -> None:
        super().__init__()
        self.lmdb_path = lmdb_path
        self.splits_json_path = splits_json_path
        self.split_name = split_name
        self.k_confs = int(k_confs)
        self.training = bool(training)
        self.standardize_atom_phys = bool(standardize_atom_phys)

        if split_family is None:
            split_family = "id" if split_name.startswith("id_") else "od"
        self.split_family = split_family

        self.meta = load_lmdb_metadata(lmdb_path)
        with open(splits_json_path, "r") as f:
            self.splits_data = json.load(f)

        records_raw = self._resolve_split_records(self.splits_data, split_name)
        self.records: List[Dict[str, Any]] = []
        self._env: Optional[lmdb.Environment] = None

        self.cham8_names = tuple(self.meta.get("cham_target_names", DEFAULT_CHAM8_ORDER))
        (
            self.cham_mean,
            self.cham_std,
            self.cham8_mean,
            self.cham8_std,
            self.atom_phys_mean,
            self.atom_phys_std,
        ) = build_normalizers(
            self.meta,
            self.splits_data.get("metadata", self.splits_data),
            split_family=split_family,
            cham8_names=self.cham8_names,
        )

        # Resolve source vocabulary strictly from training split if not provided
        if source_vocab is None:
            train_key = f"{split_family}_train"
            try:
                train_recs = self._resolve_split_records(self.splits_data, train_key)
            except KeyError:
                train_recs = records_raw
            sources = sorted({str(self._extract_source(r)) for r in train_recs if self._extract_source(r)})
            self.source_vocab: Dict[str, int] = {s: i for i, s in enumerate(sources)}
        else:
            self.source_vocab = dict(source_vocab)

        # Optional EnsembleCycPerm auxiliary enrichment (applied ONLY during train splits!)
        enrichment_map = self.splits_data.get("enrichment", {}) if self.training else {}
        if not isinstance(enrichment_map, dict):
            enrichment_map = {}

        for item in records_raw:
            if isinstance(item, dict):
                mol_key = _pick_first_present(
                    item,
                    ("mol_key", "lmdb_key", "key", "smiles_hash", "mol_id", "canonical_smiles", "smiles"),
                )
                assay_str = str(_pick_first_present(item, ("assay", "assay_type", "endpoint"), "PAMPA"))
                source_str = str(_pick_first_present(item, ("source", "dataset_source", "publication", "lab"), ""))
                y_val = _pick_first_present(item, ("y", "y_perm", "log_p", "LogP_exp", "target", "value"), np.nan)
                aux_dict = dict(item.get("aux_assays", item.get("aux_labels", {})) or {})
                if self.training and str(mol_key) in enrichment_map and isinstance(enrichment_map[str(mol_key)], dict):
                    for ak, av in enrichment_map[str(mol_key)].items():
                        aux_dict.setdefault(ak, av)

                rec = {
                    "lmdb_key": str(mol_key),
                    "assay_str": assay_str,
                    "assay_id": ASSAY_TO_ID[canonicalize_assay_name(assay_str)],
                    "source_str": source_str,
                    "source_id": self.source_vocab.get(source_str, -1),
                    "y_perm": float(y_val) if y_val is not None else float("nan"),
                    "aux_dict": aux_dict,
                    "record_id": str(item.get("record_id", mol_key)),
                    "smiles": str(item.get("canonical_smiles", item.get("smiles", ""))),
                }
            else:
                rec = {"lmdb_key": str(item), "record_id": str(item), "deferred_labels": True}
            self.records.append(rec)

        if self.records and self.records[0].get("deferred_labels"):
            self._hydrate_deferred_records(train_vocab=(source_vocab is None))

        self.y_perm_array = np.asarray([r["y_perm"] for r in self.records], dtype=np.float32)
        self.assay_id_array = np.asarray([r["assay_id"] for r in self.records], dtype=np.int64)
        self.source_id_array = np.asarray([r["source_id"] for r in self.records], dtype=np.int64)
        self.source_str_list = [r["source_str"] for r in self.records]

    @staticmethod
    def _extract_source(item: Any) -> str:
        if isinstance(item, dict):
            return str(_pick_first_present(item, ("source", "dataset_source", "publication", "lab"), ""))
        return ""

    def _resolve_split_records(self, splits_data: Dict[str, Any], split_name: str) -> List[Any]:
        # 1. Check flat records list with `od_split` / `id_split` fields (contract #8)
        if "records" in splits_data and isinstance(splits_data["records"], list):
            if split_name in ("od_train", "od_val", "od_test"):
                sub = split_name.split("_", 1)[1]
                filtered = [r for r in splits_data["records"] if str(r.get("od_split", "")).lower() == sub]
                if filtered:
                    return filtered
            elif split_name in ("id_train", "id_val", "id_test"):
                sub = split_name.split("_", 1)[1]
                filtered = [r for r in splits_data["records"] if str(r.get("id_split", "")).lower() == sub]
                if filtered:
                    return filtered

        # 2. Check external or nested dicts
        containers = [splits_data]
        for subkey in ("external", "splits"):
            if subkey in splits_data and isinstance(splits_data[subkey], dict):
                containers.append(splits_data[subkey])
        for c in containers:
            if split_name in c:
                val = c[split_name]
                if isinstance(val, list):
                    return val
                if isinstance(val, dict) and "records" in val:
                    return val["records"]
        raise KeyError(f"Split '{split_name}' not found in {self.splits_json_path}")

    def _get_env(self) -> lmdb.Environment:
        if self._env is None:
            self._env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                subdir=os.path.isdir(self.lmdb_path),
            )
        return self._env

    def _read_entry(self, key_str: str) -> Dict[str, Any]:
        env = self._get_env()
        with env.begin(write=False) as txn:
            raw = txn.get(key_str.encode("utf-8"))
            if raw is None:
                raise KeyError(f"LMDB key not found: {key_str}")
            return pickle.loads(raw)

    def _hydrate_deferred_records(self, train_vocab: bool) -> None:
        sources_seen = set()
        for rec in self.records:
            entry = self._read_entry(rec["lmdb_key"])
            assay_str = str(_pick_first_present(entry, ("assay", "assay_type", "endpoint"), "PAMPA"))
            source_str = str(_pick_first_present(entry, ("source", "dataset_source", "publication", "lab"), ""))
            y_val = _pick_first_present(entry, ("y", "y_perm", "log_p", "LogP_exp", "target"), np.nan)
            rec["assay_str"] = assay_str
            rec["assay_id"] = ASSAY_TO_ID[canonicalize_assay_name(assay_str)]
            rec["source_str"] = source_str
            rec["y_perm"] = float(y_val) if y_val is not None else float("nan")
            rec["aux_dict"] = entry.get("aux_assays", entry.get("aux_labels", {}))
            rec["smiles"] = str(_pick_first_present(entry, ("canonical_smiles", "smiles"), ""))
            if source_str:
                sources_seen.add(source_str)
        if train_vocab and not self.source_vocab:
            self.source_vocab = {s: i for i, s in enumerate(sorted(sources_seen))}
        for rec in self.records:
            rec["source_id"] = self.source_vocab.get(rec["source_str"], -1)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        entry = self._read_entry(rec["lmdb_key"])

        pos_all = np.asarray(
            _pick_first_present(entry, ("pos_all_confs", "pos", "coords")),
            dtype=np.float32,
        )
        if pos_all.ndim == 2:
            pos_all = pos_all[None, :, :]
        k_avail = int(pos_all.shape[0])
        k_take = min(self.k_confs, k_avail)
        if self.training and k_avail > k_take and k_take > 1:
            rest = np.sort(np.random.choice(np.arange(1, k_avail), size=k_take - 1, replace=False))
            sel_idx = np.concatenate([np.array([0], dtype=np.int64), rest])
            pos_sel = pos_all[sel_idx]
        else:
            pos_sel = pos_all[:k_take]

        if "node_type" in entry and entry["node_type"] is not None:
            node_type = torch.from_numpy(np.asarray(entry["node_type"], dtype=np.int64))
        else:
            elem_raw = np.asarray(_pick_first_present(entry, ("element", "atomic_nums")), dtype=np.int64)
            node_type = atomic_numbers_to_pxm_idx(torch.from_numpy(elem_raw))

        bond_index = np.asarray(_pick_first_present(entry, ("bond_index", "edge_index")), dtype=np.int64)
        bond_type = np.asarray(_pick_first_present(entry, ("bond_type", "edge_type")), dtype=np.int64)

        atom_phys = _pick_first_present(entry, ("atom_phys_feat", "atom_feat"))
        if atom_phys is not None:
            ap_np = np.nan_to_num(np.asarray(atom_phys, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            if self.standardize_atom_phys:
                ap_np = (ap_np - self.atom_phys_mean[None, :]) / self.atom_phys_std[None, :]
            atom_phys_t: Optional[torch.Tensor] = torch.from_numpy(ap_np)
        else:
            atom_phys_t = None

        cham_raw = _pick_first_present(entry, ("cham_targets", "cham"))
        if cham_raw is not None:
            c5 = extract_cham_5_raw(cham_raw, self.cham8_names)
            # Compute closure = z(imhb_max) - z(psa_3d_min) - z(rg_min) using train-only stats
            if not np.isfinite(c5[4]) and np.all(np.isfinite(c5[:4])):
                z_psa = (c5[0] - self.cham_mean[0]) / self.cham_std[0]
                z_imhb = (c5[2] - self.cham_mean[2]) / self.cham_std[2]
                z_rg = (c5[3] - self.cham_mean[3]) / self.cham_std[3]
                c5[4] = float(z_imhb - z_psa - z_rg)
            cham_norm = (c5 - self.cham_mean) / self.cham_std
            cham_mask = np.isfinite(cham_norm)
            cham_norm = np.nan_to_num(cham_norm, nan=0.0).astype(np.float32)
        else:
            cham_norm = np.zeros((5,), dtype=np.float32)
            cham_mask = np.zeros((5,), dtype=bool)

        aux_y = np.full((6,), np.nan, dtype=np.float32)
        aux_mask = np.zeros((6,), dtype=bool)
        aux_dict = rec.get("aux_dict", {})
        if isinstance(aux_dict, dict):
            for k_a, v_a in aux_dict.items():
                if v_a is None:
                    continue
                fv = float(v_a)
                if not np.isfinite(fv):
                    continue
                aid = ASSAY_TO_ID[canonicalize_assay_name(str(k_a))]
                if 1 <= aid <= 5:
                    aux_y[aid] = fv
                    aux_mask[aid] = True

        morgan_fp = _pick_first_present(entry, ("morgan_fp",))
        desc_2d = _pick_first_present(entry, ("desc_2d",))

        return {
            "idx": int(idx),
            "pos_k": torch.from_numpy(pos_sel),
            "node_type": node_type,
            "bond_index": torch.from_numpy(bond_index),
            "bond_type": torch.from_numpy(bond_type),
            "atom_phys_feat": atom_phys_t,
            "y_perm": float(rec["y_perm"]),
            "assay_id": int(rec["assay_id"]),
            "source_id": int(rec["source_id"]),
            "cham_targets_norm": torch.from_numpy(cham_norm),
            "cham_mask": torch.from_numpy(cham_mask),
            "aux_y": torch.from_numpy(aux_y),
            "aux_mask": torch.from_numpy(aux_mask),
            "morgan_fp": np.asarray(morgan_fp, dtype=np.float32) if morgan_fp is not None else None,
            "desc_2d": np.asarray(desc_2d, dtype=np.float32) if desc_2d is not None else None,
        }


def collate_perm_batch(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate variable-atom multi-conformer molecules into flat conformer graphs."""
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
    has_phys = all(it["atom_phys_feat"] is not None for it in items)

    for m_idx, it in enumerate(items):
        pos_k = it["pos_k"]
        k_m, n_m, _ = pos_k.shape
        nt = it["node_type"]
        bi = it["bond_index"]
        bt = it["bond_type"]
        ap = it["atom_phys_feat"]

        for c_idx in range(k_m):
            pos_list.append(pos_k[c_idx])
            node_type_list.append(nt)
            batch_node_list.append(torch.full((n_m,), g_idx, dtype=torch.long))
            conf_to_mol_list.append(torch.tensor(m_idx, dtype=torch.long))
            num_nodes_per_graph.append(n_m)
            if bi.numel() > 0:
                bond_index_list.append(bi + node_offset)
                bond_type_list.append(bt)
            if has_phys and ap is not None:
                phys_list.append(ap)
            node_offset += n_m
            g_idx += 1

    bi_all = torch.cat(bond_index_list, dim=1) if bond_index_list else torch.zeros((2, 0), dtype=torch.long)
    bt_all = torch.cat(bond_type_list, dim=0) if bond_type_list else torch.zeros((0,), dtype=torch.long)
    he_idx, he_type, he_batch = build_complete_halfedges(
        num_nodes_per_graph=num_nodes_per_graph,
        bond_index=bi_all,
        bond_type=bt_all,
    )

    y_perm = torch.tensor([it["y_perm"] for it in items], dtype=torch.float32)
    return {
        "indices": torch.tensor([it["idx"] for it in items], dtype=torch.long),
        "num_mols": len(items),
        "pos": torch.cat(pos_list, dim=0),
        "node_type": torch.cat(node_type_list, dim=0),
        "halfedge_index": he_idx,
        "halfedge_type": he_type,
        "halfedge_batch": he_batch,
        "batch_node": torch.cat(batch_node_list, dim=0),
        "conf_to_mol": torch.stack(conf_to_mol_list, dim=0),
        "atom_phys_feat": torch.cat(phys_list, dim=0) if has_phys and phys_list else None,
        "y_perm": y_perm,
        "y_mask": torch.isfinite(y_perm),
        "assay_id": torch.tensor([it["assay_id"] for it in items], dtype=torch.long),
        "source_id": torch.tensor([it["source_id"] for it in items], dtype=torch.long),
        "cham_targets_norm": torch.stack([it["cham_targets_norm"] for it in items], dim=0),
        "cham_mask": torch.stack([it["cham_mask"] for it in items], dim=0),
        "aux_y": torch.stack([it["aux_y"] for it in items], dim=0),
        "aux_mask": torch.stack([it["aux_mask"] for it in items], dim=0),
    }


class SourceAssayEpochBatchSampler(Sampler):
    """Groups training molecules into same-(source, assay) chunks per epoch for rich delta pairs."""

    def __init__(
        self,
        assay_ids: np.ndarray,
        source_ids: np.ndarray,
        batch_size: int = 12,
        chunk_size: int = 4,
        seed: int = 42,
    ) -> None:
        self.assay_ids = np.asarray(assay_ids, dtype=np.int64)
        self.source_ids = np.asarray(source_ids, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.chunk_size = int(chunk_size)
        self.seed = int(seed)
        self.epoch = 0

        self.groups: Dict[Tuple[int, int], List[int]] = {}
        for idx in range(len(self.assay_ids)):
            key = (int(self.source_ids[idx]), int(self.assay_ids[idx]))
            self.groups.setdefault(key, []).append(idx)

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        self.epoch += 1

        chunks: List[List[int]] = []
        for _, idxs in self.groups.items():
            perm = list(idxs)
            rng.shuffle(perm)
            for s in range(0, len(perm), self.chunk_size):
                chunks.append(perm[s : s + self.chunk_size])

        rng.shuffle(chunks)
        batches: List[List[int]] = []
        cur: List[int] = []
        for ch in chunks:
            if len(cur) + len(ch) > self.batch_size and cur:
                batches.append(cur)
                cur = list(ch)
            else:
                cur.extend(ch)
        if cur:
            batches.append(cur)

        for b in batches:
            yield b

    def __len__(self) -> int:
        total_chunks = sum((len(v) + self.chunk_size - 1) // self.chunk_size for v in self.groups.values())
        approx_per_batch = max(1, self.batch_size // self.chunk_size)
        return max(1, (total_chunks + approx_per_batch - 1) // approx_per_batch)


def extract_tabular_arrays(dataset: PermLMDBDataset) -> Dict[str, np.ndarray]:
    """Load morgan_fp, desc_2d, y_perm, assay_id, source_id arrays from dataset for baselines."""
    fps: List[np.ndarray] = []
    descs: List[np.ndarray] = []
    for i in range(len(dataset)):
        item = dataset[i]
        fp = item["morgan_fp"]
        d2 = item["desc_2d"]
        fps.append(np.zeros((2048,), dtype=np.float32) if fp is None else fp.astype(np.float32))
        descs.append(np.zeros((14,), dtype=np.float32) if d2 is None else d2.astype(np.float32))
    return {
        "morgan_fp": np.stack(fps, axis=0),
        "desc_2d": np.stack(descs, axis=0),
        "y_perm": dataset.y_perm_array.copy(),
        "assay_id": dataset.assay_id_array.copy(),
        "source_id": dataset.source_id_array.copy(),
    }
