"""Evaluate Phase 1 PermCycXMol checkpoints on CycDeltaBench OD/ID, External, Zero-Shot CAPA, and Phase 3 guidance."""

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List, Optional

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.permeability.api import PermeabilityGuidanceScorer
from models.permeability.constants import ASSAY_TO_ID, NUM_PXM_EDGE_TYPES, NUM_PXM_NODE_TYPES
from models.permeability.dataset import PermLMDBDataset
from models.permeability.determinism import enable_deterministic_eval, restore_eval_state
from models.permeability.encoder import build_complete_halfedges
from scripts.train_perm_phase1 import evaluate_3d_dataset


def _flatten_summary_rows(tag: str, split_name: str, k_mode: int, res: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    d = res["delta"]
    ws = d["within_source_delta"]
    ra = d["reconstructed_abs"]
    dir_all = res["direct"]["overall"]
    rows.append(
        {
            "model_tag": tag,
            "dataset": split_name,
            "k_confs": k_mode,
            "n_samples": dir_all["n"],
            "n_pairs_per_seed": d["n_pairs_per_seed"],
            "delta_pearson_mean": ws["pearson"]["mean"],
            "delta_pearson_std": ws["pearson"]["std"],
            "delta_spearman_mean": ws["spearman"]["mean"],
            "delta_spearman_std": ws["spearman"]["std"],
            "delta_mae_mean": ws["mae"]["mean"],
            "delta_mae_std": ws["mae"]["std"],
            "rec_abs_pearson_mean": ra["pearson"]["mean"],
            "rec_abs_pearson_std": ra["pearson"]["std"],
            "rec_abs_spearman_mean": ra["spearman"]["mean"],
            "rec_abs_spearman_std": ra["spearman"]["std"],
            "rec_abs_mae_mean": ra["mae"]["mean"],
            "rec_abs_mae_std": ra["mae"]["std"],
            "direct_abs_pearson": dir_all["pearson"],
            "direct_abs_spearman": dir_all["spearman"],
            "direct_abs_mae": dir_all["mae"],
            "direct_abs_rmse": dir_all["rmse"],
        }
    )
    return rows


def check_guidance_and_antisymmetry(
    scorer: PermeabilityGuidanceScorer,
    model_eval_res: Dict[str, Any],
) -> Dict[str, Any]:
    """Verify exact dy(q,r) == -dy(r,q), finite/non-zero Phase 3 gradients, and GPU latency."""
    device = scorer.device
    model = scorer.model
    model.eval()

    # 1. Exact antisymmetry check across held-out embeddings
    preds = model_eval_res["predictions"]
    g_np = preds["g"]
    y_np = preds["y_pred"]
    a_np = preds["assay_id"]
    m = min(128, len(g_np))
    g_t = torch.from_numpy(g_np[:m]).to(device)
    y_t = torch.from_numpy(y_np[:m]).to(device)
    a_t = torch.from_numpy(a_np[:m]).to(device)
    rng = np.random.RandomState(2026)
    iq = torch.as_tensor(rng.randint(0, m, size=64), dtype=torch.long, device=device)
    ir = torch.as_tensor(rng.randint(0, m, size=64), dtype=torch.long, device=device)
    with torch.no_grad():
        dy_qr = model.predict_delta(g_t[iq], g_t[ir], y_t[iq], y_t[ir], a_t[iq])
        dy_rq = model.predict_delta(g_t[ir], g_t[iq], y_t[ir], y_t[iq], a_t[iq])
        dy_qq = model.predict_delta(g_t[iq], g_t[iq], y_t[iq], y_t[iq], a_t[iq])
    max_antisym_err = float(torch.max(torch.abs(dy_qr + dy_rq)).item())
    max_self_err = float(torch.max(torch.abs(dy_qq)).item())

    # 2. Gradient finiteness & non-zero norm check wrt pos, node_logits, halfedge_logits
    gen = torch.Generator(device=device)
    gen.manual_seed(2026)
    n_atoms = 48
    he_idx, _, _ = build_complete_halfedges([n_atoms], device=device)
    pos = torch.randn((n_atoms, 3), device=device, generator=gen) * 3.5
    node_logits = torch.randn((n_atoms, NUM_PXM_NODE_TYPES), device=device, generator=gen)
    halfedge_logits = torch.randn((he_idx.shape[1], NUM_PXM_EDGE_TYPES), device=device, generator=gen)
    batch_node = torch.zeros((n_atoms,), dtype=torch.long, device=device)

    g_out = scorer.guidance_gradients(
        pos=pos,
        node_logits=node_logits,
        halfedge_logits=halfedge_logits,
        halfedge_index=he_idx,
        batch_node=batch_node,
        assay="PAMPA",
        objective="perm",
    )
    gp = g_out["grad_pos"]
    gn = g_out["grad_node_logits"]
    ge = g_out["grad_halfedge_logits"]

    grad_checks = {
        "grad_pos_finite": bool(torch.isfinite(gp).all().item()),
        "grad_pos_norm": float(gp.norm().item()),
        "grad_node_logits_finite": bool(torch.isfinite(gn).all().item()),
        "grad_node_logits_norm": float(gn.norm().item()),
        "grad_halfedge_logits_finite": bool(torch.isfinite(ge).all().item()),
        "grad_halfedge_logits_norm": float(ge.norm().item()),
        "max_antisym_err": max_antisym_err,
        "max_self_zero_err": max_self_err,
    }
    latency = scorer.benchmark_guidance_latency(num_atoms=64, batch_sizes=(1, 4, 8), warmup=5, iters=20)
    return {"grad_checks": grad_checks, "latency": latency}


def _find_available_split(splits_json: str, candidates: List[str]) -> Optional[str]:
    with open(splits_json, "r") as f:
        data = json.load(f)
    containers = [data]
    if "splits" in data and isinstance(data["splits"], dict):
        containers.append(data["splits"])
    if "external" in data and isinstance(data["external"], dict):
        containers.append(data["external"])
    for cand in candidates:
        for c in containers:
            if cand in c:
                return cand
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Phase 1 checkpoint comprehensively.")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--lmdb-path", type=str, required=True)
    parser.add_argument("--splits-json", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--tag", type=str, default="PermCycXMol")
    parser.add_argument("--split-family", type=str, default=None, choices=["od", "id"])
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    enable_deterministic_eval()
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    split_family = args.split_family or ckpt.get("split_family", "od")
    source_vocab = ckpt.get("source_vocab", None)

    scorer = PermeabilityGuidanceScorer.from_checkpoint(args.ckpt, device=args.device)
    model = scorer.model
    device = scorer.device

    summary_rows: List[Dict[str, Any]] = []
    per_assay_rows: List[Dict[str, Any]] = []
    full_results: Dict[str, Any] = {
        "tag": args.tag,
        "ckpt": args.ckpt,
        "split_family": split_family,
        "benchmarks": {},
    }

    internal_test = f"{split_family}_test"
    ds_test_k8 = PermLMDBDataset(
        args.lmdb_path, args.splits_json, internal_test, split_family=split_family, source_vocab=source_vocab, k_confs=8
    )
    res_k8 = evaluate_3d_dataset(model, ds_test_k8, device, batch_size=args.batch_size, num_workers=args.num_workers, use_amp=False)
    summary_rows.extend(_flatten_summary_rows(args.tag, internal_test, 8, res_k8))
    full_results["benchmarks"][f"{internal_test}_K8"] = {
        "delta": res_k8["delta"],
        "direct": res_k8["direct"],
        "cham_eval": res_k8["cham_eval"],
    }

    for assay_name, metrics in res_k8["direct"].items():
        per_assay_rows.append(
            {"model_tag": args.tag, "dataset": internal_test, "k_confs": 8, "assay": assay_name, **metrics}
        )

    # Save exact test predictions for independent audit reproducibility check
    pred_csv = os.path.join(args.output_dir, f"{args.tag}_{internal_test}_K8_predictions.csv")
    preds = res_k8["predictions"]
    with open(pred_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "smiles", "assay_id", "source_id", "y_true", "y_pred", "y_base"])
        for i in range(len(ds_test_k8)):
            w.writerow([
                i,
                ds_test_k8.records[i].get("smiles", ""),
                int(preds["assay_id"][i]),
                int(preds["source_id"][i]),
                float(preds["y_true"][i]),
                float(preds["y_pred"][i]),
                float(preds["y_base"][i]),
            ])

    # K=1 Single-Pose internal evaluation
    ds_test_k1 = PermLMDBDataset(
        args.lmdb_path, args.splits_json, internal_test, split_family=split_family, source_vocab=source_vocab, k_confs=1
    )
    res_k1 = evaluate_3d_dataset(model, ds_test_k1, device, batch_size=args.batch_size, num_workers=args.num_workers, use_amp=False)
    summary_rows.extend(_flatten_summary_rows(args.tag, internal_test, 1, res_k1))
    full_results["benchmarks"][f"{internal_test}_K1"] = {
        "delta": res_k1["delta"],
        "direct": res_k1["direct"],
        "cham_eval": res_k1["cham_eval"],
    }

    # External benchmarks + Zero-Shot Unseen Assay Nielsen CAPA (assay_id = 0 UNKNOWN!)
    ext_specs = [
        ("merz", ["merz", "ext_merz", "external_merz"], None),
        ("faris", ["faris", "ext_faris", "external_faris"], None),
        ("nielsen", ["nielsen", "nielsen_capa", "capa", "ext_nielsen", "zero_shot_nielsen"], ASSAY_TO_ID["UNKNOWN"]),
    ]
    for ext_label, candidates, override_a in ext_specs:
        found = _find_available_split(args.splits_json, candidates)
        if found is None:
            print(f"[Warn] External split for '{ext_label}' not found among {candidates}; skipping.")
            continue
        ds_ext = PermLMDBDataset(
            args.lmdb_path, args.splits_json, found, split_family=split_family, source_vocab=source_vocab, k_confs=8
        )
        res_ext = evaluate_3d_dataset(
            model,
            ds_ext,
            device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            override_assay=override_a,
            use_amp=False,
        )
        summary_rows.extend(_flatten_summary_rows(args.tag, ext_label, 8, res_ext))
        full_results["benchmarks"][ext_label] = {
            "split_key": found,
            "override_assay": override_a,
            "delta": res_ext["delta"],
            "direct": res_ext["direct"],
        }

        ext_csv = os.path.join(args.output_dir, f"{args.tag}_{ext_label}_predictions.csv")
        ep = res_ext["predictions"]
        with open(ext_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["idx", "smiles", "assay_id", "y_true", "y_pred", "y_base"])
            for i in range(len(ds_ext)):
                w.writerow([
                    i,
                    ds_ext.records[i].get("smiles", ""),
                    int(ep["assay_id"][i]),
                    float(ep["y_true"][i]),
                    float(ep["y_pred"][i]),
                    float(ep["y_base"][i]),
                ])

    full_results["eval_precision"] = "fp32"
    print(f"[Eval {args.tag}] Precision: fp32 (use_amp=False, deterministic)")

    # Restore default scatter ops before Phase 3 CUDA Graph latency & guidance check
    restore_eval_state()
    full_results["phase3_guidance_check"] = check_guidance_and_antisymmetry(scorer, res_k8)

    sum_csv = os.path.join(args.output_dir, f"{args.tag}_eval_summary.csv")
    if summary_rows:
        with open(sum_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    assay_csv = os.path.join(args.output_dir, f"{args.tag}_per_assay.csv")
    if per_assay_rows:
        with open(assay_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_assay_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_assay_rows)

    json_path = os.path.join(args.output_dir, f"{args.tag}_eval_results.json")
    with open(json_path, "w") as f:
        json.dump(full_results, f, indent=2)

    print(f"[Eval {args.tag}] Complete -> {json_path}")
    for row in summary_rows:
        print(
            f"  {row['dataset']} (K={row['k_confs']}): "
            f"Delta_P={row['delta_pearson_mean']:.4f}+-{row['delta_pearson_std']:.4f} "
            f"Delta_Sp={row['delta_spearman_mean']:.4f} "
            f"Direct_P={row['direct_abs_pearson']:.4f} "
            f"Direct_Sp={row['direct_abs_spearman']:.4f} "
            f"Direct_MAE={row['direct_abs_mae']:.4f}"
        )


if __name__ == "__main__":
    main()
