"""Retroactively score Phase 0 3BIK cyclic peptides in K=1 pocket pose and K=8 RDKit ensemble modes."""

import argparse
import csv
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.permeability.api import PermeabilityGuidanceScorer


def _discover_phase0_entries(phase0_root: str) -> List[Dict[str, Any]]:
    """Collect valid Phase 0 molecules with SDF path, pocket PDB path, group tag, and existing metrics."""
    entries: List[Dict[str, Any]] = []
    if not os.path.isdir(phase0_root):
        raise FileNotFoundError(f"Phase 0 directory not found: {phase0_root}")

    exp_dirs = sorted(
        d for d in glob.glob(os.path.join(phase0_root, "*")) if os.path.isdir(d)
    )
    for exp_dir in exp_dirs:
        exp_name = os.path.basename(exp_dir)
        lower = exp_name.lower()
        if "ncaa" in lower or "nme" in lower:
            group = "ncaa_nme"
        elif "thio" in lower:
            group = "thioether"
        elif "canon" in lower:
            group = "canonical"
        else:
            group = exp_name

        # Locate pocket_block.pdb if present inside *_SDF/0_inputs/
        pocket_cands = glob.glob(os.path.join(exp_dir, "*_SDF", "0_inputs", "pocket_block.pdb"))
        if not pocket_cands:
            pocket_cands = glob.glob(os.path.join(exp_dir, "**", "pocket_block.pdb"), recursive=True)
        pocket_pdb = pocket_cands[0] if pocket_cands else None

        # Read phase0_mol_metrics.csv and/or gen_info.csv if present
        metrics_by_idx: Dict[str, Dict[str, Any]] = {}
        metrics_csv = os.path.join(exp_dir, "phase0_mol_metrics.csv")
        if os.path.isfile(metrics_csv):
            with open(metrics_csv, "r") as f:
                for row in csv.DictReader(f):
                    key = str(row.get("sdf_path") or row.get("mol_id") or row.get("idx") or row.get("smiles") or len(metrics_by_idx))
                    metrics_by_idx[key] = row

        # Collect generated SDF files (excluding 0_inputs reference SDFs)
        all_sdfs = sorted(glob.glob(os.path.join(exp_dir, "*_SDF", "*.sdf")))
        if not all_sdfs:
            all_sdfs = sorted(
                p for p in glob.glob(os.path.join(exp_dir, "**", "*.sdf"), recursive=True)
                if "0_inputs" not in p
            )

        for sdf_path in all_sdfs:
            suppl = Chem.SDMolSupplier(sdf_path, removeHs=True, sanitize=True)
            for mol_idx, mol in enumerate(suppl):
                if mol is None or mol.GetNumAtoms() == 0:
                    continue
                smi = Chem.MolToSmiles(mol, isomericSmiles=True)
                row_meta = metrics_by_idx.get(sdf_path) or metrics_by_idx.get(os.path.basename(sdf_path)) or metrics_by_idx.get(smi) or {}
                entries.append(
                    {
                        "exp_dir": exp_name,
                        "group": group,
                        "sdf_path": sdf_path,
                        "sdf_mol_idx": mol_idx,
                        "pocket_pdb": pocket_pdb,
                        "smiles": smi,
                        "mol": mol,
                        "phase0_meta": row_meta,
                    }
                )
    return entries


def _safe_float(v: Any) -> float:
    try:
        val = float(v)
        return val if np.isfinite(val) else float("nan")
    except Exception:
        return float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score Phase 0 3BIK peptides using Phase 1 PermCycXMol.")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--phase0-root", type=str, default="/flashfs1/work/btdai/PermCycXMol/runs/phase0_baseline")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--assay", type=str, default="PAMPA")
    parser.add_argument("--k-confs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, min(16, int(os.environ.get("SLURM_CPUS_PER_TASK") or (os.cpu_count() or 1)))),
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    scorer = PermeabilityGuidanceScorer.from_checkpoint(args.ckpt, device=args.device)
    entries = _discover_phase0_entries(args.phase0_root)
    if not entries:
        raise RuntimeError(f"No valid SDF molecules discovered under {args.phase0_root}")

    print(f"[Phase 0 Scoring] Discovered {len(entries)} valid SDF molecules across experiments.")

    # 1. Mode A: K=1 Holo Pocket Pose (grouped by pocket_pdb so coordinates match PDB frame)
    by_pocket: Dict[Optional[str], List[int]] = {}
    for idx, ent in enumerate(entries):
        by_pocket.setdefault(ent["pocket_pdb"], []).append(idx)

    k1_perm = np.full((len(entries),), np.nan, dtype=np.float32)
    k1_base = np.full((len(entries),), np.nan, dtype=np.float32)
    k1_cham = np.full((len(entries), 5), np.nan, dtype=np.float32)
    k1_imhb = np.full((len(entries),), np.nan, dtype=np.float32)
    k1_shield = np.full((len(entries),), np.nan, dtype=np.float32)
    k1_rg = np.full((len(entries),), np.nan, dtype=np.float32)

    for p_pdb, idxs in by_pocket.items():
        mols = [entries[i]["mol"] for i in idxs]
        res_k1 = scorer.score_rdkit_mols(
            mols_or_smiles=mols,
            k_confs=1,
            assay=args.assay,
            batch_size=args.batch_size,
            embed_if_needed=False,
            pocket_pdb_path=p_pdb,
        )
        k1_perm[idxs] = res_k1["perm_pred"]
        k1_base[idxs] = res_k1["perm_base"]
        k1_cham[idxs] = res_k1["cham_pred_phys"]
        k1_imhb[idxs] = res_k1["smooth_imhb_trans"]
        k1_shield[idxs] = res_k1["smooth_polar_shield"]
        k1_rg[idxs] = res_k1["smooth_rg"]

    # 2. Mode B: K=8 RDKit Conformer Ensemble (h_ctx=None)
    all_smiles = [e["smiles"] for e in entries]
    print(
        f"[phase0_score] Mode B: embedding & scoring {len(all_smiles)} peptides "
        f"(k_confs={args.k_confs}, num_workers={args.num_workers})...",
        flush=True,
    )
    res_k8 = scorer.score_rdkit_mols(
        mols_or_smiles=all_smiles,
        k_confs=args.k_confs,
        assay=args.assay,
        batch_size=args.batch_size,
        embed_if_needed=True,
        pocket_pdb_path=None,
        num_workers=args.num_workers,
    )
    print("[phase0_score] Mode B complete.", flush=True)

    rows: List[Dict[str, Any]] = []
    for i, ent in enumerate(entries):
        meta = ent["phase0_meta"]
        mol_i = ent["mol"]
        hbd_val = _safe_float(meta.get("hbd", meta.get("HBD")))
        tpsa_val = _safe_float(meta.get("tpsa", meta.get("TPSA")))
        clogp_val = _safe_float(meta.get("clogp", meta.get("cLogP")))
        mw_val = _safe_float(meta.get("mw", meta.get("MW")))
        if not np.isfinite(hbd_val):
            hbd_val = float(Lipinski.NumHDonors(mol_i))
        if not np.isfinite(tpsa_val):
            tpsa_val = float(rdMolDescriptors.CalcTPSA(mol_i))
        if not np.isfinite(clogp_val):
            clogp_val = float(Descriptors.MolLogP(mol_i))
        if not np.isfinite(mw_val):
            mw_val = float(Descriptors.MolWt(mol_i))
        rows.append(
            {
                "group": ent["group"],
                "exp_dir": ent["exp_dir"],
                "sdf_path": ent["sdf_path"],
                "sdf_mol_idx": ent["sdf_mol_idx"],
                "smiles": ent["smiles"],
                "n_heavy": mol_i.GetNumAtoms(),
                "p0_hbd": hbd_val,
                "p0_tpsa": tpsa_val,
                "p0_clogp": clogp_val,
                "p0_mw": mw_val,
                "perm_k1_holo": float(k1_perm[i]),
                "perm_base_k1_holo": float(k1_base[i]),
                "cham_k1_psa3d_min": float(k1_cham[i, 0]),
                "cham_k1_delta_psa3d": float(k1_cham[i, 1]),
                "cham_k1_imhb_max": float(k1_cham[i, 2]),
                "cham_k1_rg_min": float(k1_cham[i, 3]),
                "cham_k1_closure": float(k1_cham[i, 4]),
                "geom_k1_imhb_trans": float(k1_imhb[i]),
                "geom_k1_polar_shield": float(k1_shield[i]),
                "geom_k1_rg": float(k1_rg[i]),
                "perm_k8_ens": float(res_k8["perm_pred"][i]),
                "perm_base_k8_ens": float(res_k8["perm_base"][i]),
                "cham_k8_psa3d_min": float(res_k8["cham_pred_phys"][i, 0]),
                "cham_k8_delta_psa3d": float(res_k8["cham_pred_phys"][i, 1]),
                "cham_k8_imhb_max": float(res_k8["cham_pred_phys"][i, 2]),
                "cham_k8_rg_min": float(res_k8["cham_pred_phys"][i, 3]),
                "cham_k8_closure": float(res_k8["cham_pred_phys"][i, 4]),
                "geom_k8_imhb_trans": float(res_k8["smooth_imhb_trans"][i]),
                "geom_k8_polar_shield": float(res_k8["smooth_polar_shield"][i]),
                "geom_k8_rg": float(res_k8["smooth_rg"][i]),
            }
        )

    scores_csv = os.path.join(args.output_dir, "phase0_perm_scores.csv")
    with open(scores_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Compute per-group distributions & correlations
    groups = sorted({r["group"] for r in rows})
    summary_rows: List[Dict[str, Any]] = []
    for g_name in groups:
        sub = [r for r in rows if r["group"] == g_name]
        pk1 = np.asarray([r["perm_k1_holo"] for r in sub], dtype=np.float64)
        pk8 = np.asarray([r["perm_k8_ens"] for r in sub], dtype=np.float64)
        ck8 = np.asarray([r["cham_k8_closure"] for r in sub], dtype=np.float64)
        summary_rows.append(
            {
                "group": g_name,
                "n_valid": len(sub),
                "perm_k1_holo_mean": float(np.nanmean(pk1)),
                "perm_k1_holo_std": float(np.nanstd(pk1)),
                "perm_k1_holo_median": float(np.nanmedian(pk1)),
                "perm_k1_ge_minus6_frac": float(np.mean(pk1 >= -6.0)),
                "perm_k8_ens_mean": float(np.nanmean(pk8)),
                "perm_k8_ens_std": float(np.nanstd(pk8)),
                "perm_k8_ens_median": float(np.nanmedian(pk8)),
                "perm_k8_ge_minus6_frac": float(np.mean(pk8 >= -6.0)),
                "cham_k8_closure_mean": float(np.nanmean(ck8)),
                "cham_k8_closure_std": float(np.nanstd(ck8)),
            }
        )

    summary_csv = os.path.join(args.output_dir, "phase0_perm_group_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"[Phase 0 Scoring] Saved {len(rows)} rows -> {scores_csv}")
    for sr in summary_rows:
        print(
            f"  {sr['group']} (N={sr['n_valid']}): "
            f"K1_Holo={sr['perm_k1_holo_mean']:.3f}+-{sr['perm_k1_holo_std']:.3f} (>= -6: {sr['perm_k1_ge_minus6_frac']:.1%}) | "
            f"K8_Ens={sr['perm_k8_ens_mean']:.3f}+-{sr['perm_k8_ens_std']:.3f} (>= -6: {sr['perm_k8_ge_minus6_frac']:.1%})"
        )


if __name__ == "__main__":
    main()
