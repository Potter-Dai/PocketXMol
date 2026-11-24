"""Train Phase 1 PermCycXMol models, ablations, and 2D baselines on szbl-1."""

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.permeability.baselines import (
    ABLATION_PRESETS,
    RidgeBaselineModel,
    build_cycdelta_seed_pairs,
    evaluate_delta_over_seeds,
    evaluate_direct_per_assay,
    regression_metrics,
    train_tabular_mlp_baseline,
)
from models.permeability.dataset import (
    PermLMDBDataset,
    SourceAssayEpochBatchSampler,
    collate_perm_batch,
    extract_tabular_arrays,
    resolve_ref_pairs_from_splits,
)
from models.permeability.encoder import ChameleonicPXMEncoder
from models.permeability.heads import PermCycXMolModel, compute_multitask_loss


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _cuda_autocast(enabled: bool, dtype: torch.dtype):
    import contextlib

    if not enabled:
        return contextlib.nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        try:
            return torch.amp.autocast(device_type="cuda", dtype=dtype, enabled=True)
        except TypeError:
            pass
    return torch.cuda.amp.autocast(enabled=True, dtype=dtype)


def _make_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            pass
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


@torch.no_grad()
def evaluate_3d_dataset(
    model: PermCycXMolModel,
    dataset: PermLMDBDataset,
    device: torch.device,
    batch_size: int = 12,
    num_workers: int = 4,
    override_assay: Optional[int] = None,
    seeds: Tuple[int, ...] = tuple(range(10)),
    use_amp: bool = False,
) -> Dict[str, Any]:
    """Evaluate 3D PermCycXMolModel on a dataset split (direct abs + 10-seed delta)."""
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_perm_batch,
        pin_memory=(device.type == "cuda"),
    )

    n = len(dataset)
    g_dim = model.g_dim
    all_g = np.zeros((n, g_dim), dtype=np.float32)
    all_y_pred = np.zeros((n,), dtype=np.float32)
    all_y_base = np.zeros((n,), dtype=np.float32)
    all_cham_pred = np.zeros((n, 5), dtype=np.float32)
    all_cham_true = np.zeros((n, 5), dtype=np.float32)
    all_cham_mask = np.zeros((n, 5), dtype=bool)

    has_native_bf16 = (
        device.type == "cuda"
        and torch.cuda.get_device_capability(device)[0] >= 8
        and torch.cuda.is_bf16_supported()
    )
    use_eval_amp = bool(use_amp) and has_native_bf16
    eval_amp_dtype = torch.bfloat16 if use_eval_amp else torch.float32

    for raw_b in loader:
        b = _move_batch_to_device(raw_b, device)
        a_in = b["assay_id"]
        if override_assay is not None:
            a_in = torch.full_like(a_in, int(override_assay))

        with _cuda_autocast(use_eval_amp, eval_amp_dtype):
            out = model(
                pos=b["pos"],
                halfedge_index=b["halfedge_index"],
                batch_node=b["batch_node"],
                conf_to_mol=b["conf_to_mol"],
                num_mols=b["num_mols"],
                assay_id=a_in,
                apply_source_offset=False,
                return_k1_view=False,
                node_type=b["node_type"],
                halfedge_type=b["halfedge_type"],
                halfedge_batch=b["halfedge_batch"],
                atom_phys_feat=b.get("atom_phys_feat"),
            )
        idxs = raw_b["indices"].numpy()
        all_g[idxs] = out["g"].cpu().numpy()
        all_y_pred[idxs] = out["y_abs_infer"].cpu().numpy()
        all_y_base[idxs] = out["y_base"].cpu().numpy()
        all_cham_pred[idxs] = out["cham_pred"].cpu().numpy()
        all_cham_true[idxs] = raw_b["cham_targets_norm"].numpy()
        all_cham_mask[idxs] = raw_b["cham_mask"].numpy()

    y_true = dataset.y_perm_array
    assay_ids = dataset.assay_id_array
    if override_assay is not None:
        assay_for_delta = np.full_like(assay_ids, int(override_assay))
    else:
        assay_for_delta = assay_ids
    source_ids = dataset.source_id_array

    # Note: for single-source external sets without explicit positive source_ids, assign 0
    if (source_ids < 0).all():
        source_for_delta = np.zeros_like(source_ids)
    else:
        source_for_delta = source_ids

    g_t = torch.from_numpy(all_g).to(device)
    y_t = torch.from_numpy(all_y_pred).to(device)

    def _predict_delta(qi: np.ndarray, ri: np.ndarray, pa: np.ndarray) -> np.ndarray:
        qt = torch.as_tensor(qi, dtype=torch.long, device=device)
        rt = torch.as_tensor(ri, dtype=torch.long, device=device)
        at = torch.as_tensor(pa, dtype=torch.long, device=device)
        with torch.no_grad():
            return model.predict_delta(g_t[qt], g_t[rt], y_t[qt], y_t[rt], at).cpu().numpy()

    seed_pairs = resolve_ref_pairs_from_splits(dataset.splits_data, dataset.split_name, dataset.records)
    if seed_pairs is None:
        seed_pairs = build_cycdelta_seed_pairs(y_true, assay_for_delta, source_for_delta, seeds=seeds)
    elif len(seeds) < len(seed_pairs):
        seed_pairs = [seed_pairs[s] for s in seeds if s < len(seed_pairs)]
    delta_res = evaluate_delta_over_seeds(
        y_true, assay_for_delta, source_for_delta, _predict_delta, seed_pairs=seed_pairs, seeds=seeds
    )
    direct_res = evaluate_direct_per_assay(y_true, all_y_pred, assay_ids)

    cham_eval: Dict[str, Any] = {}
    cham_names = ("psa_3d_min", "delta_psa_3d", "imhb_max", "rg_min", "closure")
    for i, cname in enumerate(cham_names):
        m = all_cham_mask[:, i]
        if m.any():
            cham_eval[cname] = regression_metrics(all_cham_true[m, i], all_cham_pred[m, i])

    return {
        "delta": delta_res,
        "direct": direct_res,
        "cham_eval": cham_eval,
        "predictions": {
            "y_true": y_true,
            "y_pred": all_y_pred,
            "y_base": all_y_base,
            "assay_id": assay_ids,
            "source_id": source_ids,
            "g": all_g,
            "cham_pred_norm": all_cham_pred,
        },
    }


def run_2d_baselines_for_split(
    lmdb_path: str,
    splits_json: str,
    split_family: str,
    out_dir: str,
    seed: int = 2026,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Run MorganFP-Ridge/MLP and RDKit2D-MLP baselines on exact same split."""
    os.makedirs(out_dir, exist_ok=True)
    ds_tr = PermLMDBDataset(lmdb_path, splits_json, f"{split_family}_train", split_family=split_family, k_confs=1)
    ds_va = PermLMDBDataset(
        lmdb_path, splits_json, f"{split_family}_val", split_family=split_family, source_vocab=ds_tr.source_vocab, k_confs=1
    )
    ds_te = PermLMDBDataset(
        lmdb_path, splits_json, f"{split_family}_test", split_family=split_family, source_vocab=ds_tr.source_vocab, k_confs=1
    )

    tr = extract_tabular_arrays(ds_tr)
    va = extract_tabular_arrays(ds_va)
    te = extract_tabular_arrays(ds_te)

    # Standardize desc_2d strictly on training split
    d2_mean = np.nanmean(tr["desc_2d"], axis=0, keepdims=True)
    d2_std = np.clip(np.nanstd(tr["desc_2d"], axis=0, keepdims=True), 1e-6, None)
    tr_d2 = np.nan_to_num((tr["desc_2d"] - d2_mean) / d2_std, nan=0.0)
    va_d2 = np.nan_to_num((va["desc_2d"] - d2_mean) / d2_std, nan=0.0)
    te_d2 = np.nan_to_num((te["desc_2d"] - d2_mean) / d2_std, nan=0.0)

    results: Dict[str, Any] = {"split_family": split_family, "models": {}}

    # 1. MorganFP Ridge (Direct & Pairwise Delta)
    ridge = RidgeBaselineModel()
    ridge.fit_direct(
        tr["morgan_fp"], tr["assay_id"], tr["source_id"], tr["y_perm"],
        va["morgan_fp"], va["assay_id"], va["y_perm"],
    )
    ridge.fit_pairwise_delta(
        tr["morgan_fp"], tr["assay_id"], tr["source_id"], tr["y_perm"],
        va["morgan_fp"], va["assay_id"], va["source_id"], va["y_perm"],
        seed=seed,
    )
    te_seed_pairs = resolve_ref_pairs_from_splits(ds_te.splits_data, ds_te.split_name, ds_te.records)
    if te_seed_pairs is None:
        te_seed_pairs = build_cycdelta_seed_pairs(te["y_perm"], te["assay_id"], te["source_id"])

    te_pred_rd = ridge.predict_direct(te["morgan_fp"], te["assay_id"])
    results["models"]["MorganFP_Ridge_Direct"] = {
        "direct": evaluate_direct_per_assay(te["y_perm"], te_pred_rd, te["assay_id"]),
        "delta": evaluate_delta_over_seeds(
            te["y_perm"], te["assay_id"], te["source_id"],
            lambda q, r, _: te_pred_rd[q] - te_pred_rd[r],
            seed_pairs=te_seed_pairs,
        ),
        "best_alpha": ridge.best_alpha_direct,
    }
    results["models"]["MorganFP_Ridge_Delta"] = {
        "direct": evaluate_direct_per_assay(te["y_perm"], te_pred_rd, te["assay_id"]),
        "delta": evaluate_delta_over_seeds(
            te["y_perm"], te["assay_id"], te["source_id"],
            lambda q, r, _: ridge.predict_delta(te["morgan_fp"][q], te["morgan_fp"][r]),
            seed_pairs=te_seed_pairs,
        ),
        "best_alpha": ridge.best_alpha_delta,
    }

    # 2. Neural MLPs (MorganFP & RDKit2D; direct vs joint delta)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tabular_specs = [
        ("MorganFP_MLP_Direct", tr["morgan_fp"], va["morgan_fp"], te["morgan_fp"], "direct"),
        ("MorganFP_MLP_Delta", tr["morgan_fp"], va["morgan_fp"], te["morgan_fp"], "joint"),
        ("RDKit2D_MLP_Direct", tr_d2, va_d2, te_d2, "direct"),
        ("RDKit2D_MLP_Delta", tr_d2, va_d2, te_d2, "joint"),
    ]
    for name, x_tr, x_va, x_te, mode in tabular_specs:
        mlp, val_info = train_tabular_mlp_baseline(
            x_train=x_tr, assay_train=tr["assay_id"], source_train=tr["source_id"], y_train=tr["y_perm"],
            x_val=x_va, assay_val=va["assay_id"], source_val=va["source_id"], y_val=va["y_perm"],
            mode=mode, seed=seed, device=device,
        )
        with torch.no_grad():
            x_te_t = torch.from_numpy(x_te.astype(np.float32)).to(device)
            a_te_t = torch.from_numpy(te["assay_id"].astype(np.int64)).to(device)
            g_te, y_te_pred_t, _ = mlp.encode_and_abs(x_te_t, a_te_t, None, apply_source_offset=False)
            y_te_pred = y_te_pred_t.cpu().numpy()

        def _mlp_delta_fn(qi: np.ndarray, ri: np.ndarray, pa: np.ndarray, _m=mode, _g=g_te, _y=y_te_pred_t, _mlp=mlp) -> np.ndarray:
            if _m == "direct":
                return (_y[torch.as_tensor(qi, device=device)] - _y[torch.as_tensor(ri, device=device)]).cpu().numpy()
            qt = torch.as_tensor(qi, dtype=torch.long, device=device)
            rt = torch.as_tensor(ri, dtype=torch.long, device=device)
            at = torch.as_tensor(pa, dtype=torch.long, device=device)
            with torch.no_grad():
                return _mlp.predict_delta(_g[qt], _g[rt], _y[qt], _y[rt], at).cpu().numpy()

        results["models"][name] = {
            "val_info": val_info,
            "direct": evaluate_direct_per_assay(te["y_perm"], y_te_pred, te["assay_id"]),
            "delta": evaluate_delta_over_seeds(
                te["y_perm"], te["assay_id"], te["source_id"], _mlp_delta_fn, seed_pairs=te_seed_pairs
            ),
            "external": {},
        }

        # Evaluate MLP baseline on External Merz / Faris / Zero-Shot Nielsen CAPA
        for ext_name, ov_assay in (("merz", None), ("faris", None), ("nielsen_capa", 0)):
            try:
                ds_ext = PermLMDBDataset(
                    lmdb_path, splits_json, ext_name, split_family=split_family, source_vocab=ds_tr.source_vocab, k_confs=1
                )
            except KeyError:
                continue
            ex = extract_tabular_arrays(ds_ext)
            x_ex = ex["morgan_fp"] if "MorganFP" in name else np.nan_to_num((ex["desc_2d"] - d2_mean) / d2_std, nan=0.0)
            a_ex = np.full_like(ex["assay_id"], ov_assay) if ov_assay is not None else ex["assay_id"]
            src_ex = np.zeros_like(ex["source_id"]) if (ex["source_id"] < 0).all() else ex["source_id"]
            ex_seed_pairs = resolve_ref_pairs_from_splits(ds_ext.splits_data, ext_name, ds_ext.records)
            if ex_seed_pairs is None:
                ex_seed_pairs = build_cycdelta_seed_pairs(ex["y_perm"], a_ex, src_ex)
            with torch.no_grad():
                x_ex_t = torch.from_numpy(x_ex.astype(np.float32)).to(device)
                a_ex_t = torch.from_numpy(a_ex.astype(np.int64)).to(device)
                g_ex, y_ex_pred_t, _ = mlp.encode_and_abs(x_ex_t, a_ex_t, None, apply_source_offset=False)
                y_ex_pred = y_ex_pred_t.cpu().numpy()

            def _ex_delta_fn(qi: np.ndarray, ri: np.ndarray, pa: np.ndarray, _m=mode, _g=g_ex, _y=y_ex_pred_t, _mlp=mlp) -> np.ndarray:
                if _m == "direct":
                    return (_y[torch.as_tensor(qi, device=device)] - _y[torch.as_tensor(ri, device=device)]).cpu().numpy()
                qt = torch.as_tensor(qi, dtype=torch.long, device=device)
                rt = torch.as_tensor(ri, dtype=torch.long, device=device)
                at = torch.as_tensor(pa, dtype=torch.long, device=device)
                with torch.no_grad():
                    return _mlp.predict_delta(_g[qt], _g[rt], _y[qt], _y[rt], at).cpu().numpy()

            results["models"][name]["external"][ext_name] = {
                "direct": regression_metrics(ex["y_perm"], y_ex_pred),
                "delta": evaluate_delta_over_seeds(ex["y_perm"], a_ex, src_ex, _ex_delta_fn, seed_pairs=ex_seed_pairs),
            }

    out_path = os.path.join(out_dir, f"baselines_{split_family}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Baselines {split_family}] saved -> {out_path}")
    return results


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)


def train_3d_model(cfg: Dict[str, Any], args: argparse.Namespace) -> str:
    set_seed(int(cfg["train"]["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    split_family = args.split_family
    run_name = args.run_name or f"{split_family}_{args.ablation}"
    out_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    lmdb_path = args.lmdb_path or cfg["lmdb_path"]
    splits_json = args.splits_json or cfg["splits_json"]
    pxm_ckpt = args.pxm_ckpt or cfg["pxm_ckpt"]

    preset = ABLATION_PRESETS[args.ablation]
    mcfg = dict(cfg["model"])
    tcfg = dict(cfg["train"])
    for k, v in preset.items():
        if k in mcfg:
            mcfg[k] = v
        if k in tcfg:
            tcfg[k] = v
    if args.epochs is not None:
        tcfg["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        tcfg["batch_size"] = int(args.batch_size)
    if args.k_confs_train is not None:
        tcfg["k_confs_train"] = int(args.k_confs_train)
    if args.amp is not None:
        tcfg["use_amp"] = bool(args.amp)
    if args.use_grad_checkpoint is not None:
        mcfg["use_grad_checkpoint"] = bool(args.use_grad_checkpoint)

    ds_train = PermLMDBDataset(
        lmdb_path=lmdb_path,
        splits_json_path=splits_json,
        split_name=f"{split_family}_train",
        split_family=split_family,
        k_confs=int(tcfg["k_confs_train"]),
        training=True,
    )
    ds_val = PermLMDBDataset(
        lmdb_path=lmdb_path,
        splits_json_path=splits_json,
        split_name=f"{split_family}_val",
        split_family=split_family,
        source_vocab=ds_train.source_vocab,
        k_confs=int(tcfg["k_confs_eval"]),
        training=False,
    )

    encoder = ChameleonicPXMEncoder(
        pxm_ckpt_path=pxm_ckpt,
        load_pretrained=bool(mcfg["load_pretrained"]),
        atom_phys_dim=int(mcfg["atom_phys_dim"]),
        conf_dim=int(mcfg["conf_dim"]),
        shield_hidden_dim=int(mcfg["shield_hidden_dim"]),
        phys_dropout_prob=float(mcfg["phys_dropout_prob"]),
        use_grad_checkpoint=bool(mcfg["use_grad_checkpoint"]),
        max_graphs_per_chunk=int(mcfg.get("max_graphs_per_chunk", 8)),
        max_edges_per_chunk=int(mcfg.get("max_edges_per_chunk", 16000)),
    )
    model = PermCycXMolModel(
        encoder=encoder,
        trunk_dim=int(mcfg["trunk_dim"]),
        g_dim=int(mcfg["g_dim"]),
        cham_hidden_dim=int(mcfg["cham_hidden_dim"]),
        assay_emb_dim=int(mcfg["assay_emb_dim"]),
        max_sources=max(int(mcfg["max_sources"]), len(ds_train.source_vocab) + 8),
        use_cham_in_trunk=bool(mcfg["use_cham_in_trunk"]),
    ).to(device)
    model.set_cham_stats(
        torch.from_numpy(ds_train.cham_mean),
        torch.from_numpy(ds_train.cham_std),
    )
    model.set_atom_phys_stats(
        torch.from_numpy(ds_train.atom_phys_mean),
        torch.from_numpy(ds_train.atom_phys_std),
    )

    optimizer = torch.optim.AdamW(
        [
            {"params": model.pxm_trunk_parameters(), "lr": float(tcfg["trunk_lr"])},
            {"params": model.new_head_parameters(), "lr": float(tcfg["head_lr"])},
        ],
        weight_decay=float(tcfg["weight_decay"]),
    )

    sampler = SourceAssayEpochBatchSampler(
        assay_ids=ds_train.assay_id_array,
        source_ids=ds_train.source_id_array,
        batch_size=int(tcfg["batch_size"]),
        chunk_size=int(tcfg["chunk_size"]),
        seed=int(tcfg["seed"]),
    )
    grad_accum = max(1, int(tcfg["grad_accum"]))
    epochs = int(tcfg["epochs"])
    steps_per_epoch = max(1, (len(sampler) + grad_accum - 1) // grad_accum)
    scheduler = build_lr_scheduler(
        optimizer,
        total_steps=epochs * steps_per_epoch,
        warmup_steps=int(tcfg["warmup_epochs"]) * steps_per_epoch,
    )

    ckpt_save_payload_base = {
        "model_config": mcfg,
        "train_config": tcfg,
        "ablation": args.ablation,
        "split_family": split_family,
        "source_vocab": ds_train.source_vocab,
        "pxm_model_cfg": dict(encoder.pxm_model_cfg),
    }
    with open(os.path.join(out_dir, "resolved_config.json"), "w") as f:
        json.dump(ckpt_save_payload_base, f, indent=2, default=str)

    best_score = -1e9
    best_epoch = 0
    no_improve = 0
    history: List[Dict[str, Any]] = []
    best_ckpt_path = os.path.join(out_dir, "best.pt")

    use_amp = bool(tcfg.get("use_amp", False)) and (device.type == "cuda")
    amp_cfg = str(tcfg.get("amp_dtype", "auto")).lower()
    amp_dtype = torch.bfloat16
    if use_amp:
        has_native_bf16 = torch.cuda.get_device_capability(device)[0] >= 8 and torch.cuda.is_bf16_supported()
        if amp_cfg == "bf16" or (amp_cfg == "auto" and has_native_bf16):
            amp_dtype = torch.bfloat16
        elif amp_cfg in ("fp16", "float16"):
            amp_dtype = torch.float16
        else:
            print(
                f"[{run_name}] Disabling AMP on {torch.cuda.get_device_name(device)} "
                "(no native bf16; pretrained PXM denoiser overflows fp16) -> running FP32.",
                flush=True,
            )
            use_amp = False
    else:
        amp_dtype = torch.float32
    use_scaler = use_amp and (amp_dtype == torch.float16)
    scaler = _make_grad_scaler(enabled=use_scaler)
    print(f"[{run_name}] Device={device}, use_amp={use_amp} ({amp_dtype}), use_scaler={use_scaler}, k_train={tcfg['k_confs_train']}")

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        loader = DataLoader(
            ds_train,
            batch_sampler=sampler,
            num_workers=int(tcfg["num_workers"]),
            collate_fn=collate_perm_batch,
            pin_memory=(device.type == "cuda"),
        )
        optimizer.zero_grad(set_to_none=True)
        running: Dict[str, float] = {}
        n_batches = len(loader)

        for step_idx, raw_b in enumerate(loader):
            b = _move_batch_to_device(raw_b, device)
            return_k1 = (int(tcfg["k_confs_train"]) > 1) and (float(tcfg["lambda_cons"]) > 0.0)
            with _cuda_autocast(use_amp, amp_dtype):
                out = model(
                    pos=b["pos"],
                    halfedge_index=b["halfedge_index"],
                    batch_node=b["batch_node"],
                    conf_to_mol=b["conf_to_mol"],
                    num_mols=b["num_mols"],
                    assay_id=b["assay_id"],
                    source_id=b["source_id"],
                    apply_source_offset=True,
                    return_k1_view=return_k1,
                    node_type=b["node_type"],
                    halfedge_type=b["halfedge_type"],
                    halfedge_batch=b["halfedge_batch"],
                    atom_phys_feat=b.get("atom_phys_feat"),
                )
            loss, stats = compute_multitask_loss(
                model=model,
                outputs=out,
                batch=b,
                lambda_abs=float(tcfg["lambda_abs"]),
                lambda_delta=float(tcfg["lambda_delta"]),
                lambda_cham=float(tcfg["lambda_cham"]),
                lambda_cons=float(tcfg["lambda_cons"]),
                lambda_aux_assay=float(tcfg["lambda_aux_assay"]),
                lambda_source_l2=float(tcfg["lambda_source_l2"]),
                smooth_l1_beta=float(tcfg["smooth_l1_beta"]),
            )
            scaler.scale(loss / float(grad_accum)).backward()

            for sk, sv in stats.items():
                running[sk] = running.get(sk, 0.0) + float(sv)

            if ((step_idx + 1) % grad_accum == 0) or ((step_idx + 1) == n_batches):
                if use_scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(tcfg["clip_grad_norm"]))
                if use_scaler:
                    prev_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    if scaler.get_scale() >= prev_scale:
                        scheduler.step()
                else:
                    optimizer.step()
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        train_stats = {k: v / max(1, n_batches) for k, v in running.items()}

        # Validation evaluation (use 3 seeds during epoch loop for speed; 10 seeds on final eval)
        val_res = evaluate_3d_dataset(
            model=model,
            dataset=ds_val,
            device=device,
            batch_size=int(tcfg["batch_size"]),
            num_workers=int(tcfg["num_workers"]),
            seeds=(0, 1, 2),
            use_amp=use_amp,
        )
        val_abs_p = val_res["direct"]["overall"]["pearson"]
        val_abs_mae = val_res["direct"]["overall"]["mae"]
        val_delta_p = val_res["delta"]["within_source_delta"]["pearson"]["mean"]
        val_delta_mae = val_res["delta"]["within_source_delta"]["mae"]["mean"]

        if args.ablation == "NoDeltaLoss":
            comp_score = val_abs_p if np.isfinite(val_abs_p) else -1e9
        elif args.ablation == "DeltaOnly":
            comp_score = val_delta_p if np.isfinite(val_delta_p) else -1e9
        else:
            dp = val_delta_p if np.isfinite(val_delta_p) else 0.0
            ap = val_abs_p if np.isfinite(val_abs_p) else 0.0
            comp_score = dp + 0.5 * ap

        dt = time.perf_counter() - t0
        ep_log = {
            "epoch": epoch,
            "time_s": round(dt, 2),
            "train": train_stats,
            "val_abs_pearson": val_abs_p,
            "val_abs_mae": val_abs_mae,
            "val_delta_pearson": val_delta_p,
            "val_delta_mae": val_delta_mae,
            "composite_score": comp_score,
        }
        history.append(ep_log)
        print(
            f"[{run_name}] Ep {epoch:02d}/{epochs} ({dt:.1f}s) "
            f"loss={train_stats.get('loss_total', 0.0):.4f} "
            f"abs_p={val_abs_p:.4f} abs_mae={val_abs_mae:.4f} "
            f"delta_p={val_delta_p:.4f} delta_mae={val_delta_mae:.4f}"
        )

        payload = dict(ckpt_save_payload_base)
        payload["epoch"] = epoch
        payload["val_metrics"] = ep_log
        payload["model_state_dict"] = model.state_dict()
        torch.save(payload, os.path.join(out_dir, "last.pt"))

        if comp_score > best_score:
            best_score = comp_score
            best_epoch = epoch
            no_improve = 0
            torch.save(payload, best_ckpt_path)
        else:
            no_improve += 1
            if no_improve >= int(tcfg["patience"]):
                print(f"[{run_name}] Early stopping at epoch {epoch} (best epoch {best_epoch}).")
                break

    with open(os.path.join(out_dir, "train_history.json"), "w") as f:
        json.dump({"best_epoch": best_epoch, "best_score": best_score, "history": history}, f, indent=2)
    print(f"[{run_name}] Done. Best checkpoint: {best_ckpt_path}")
    return best_ckpt_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Phase 1 PermCycXMol & baselines.")
    parser.add_argument("--config", type=str, default=os.path.join(REPO_ROOT, "configs/permeability/phase1_default.yml"))
    parser.add_argument("--lmdb-path", type=str, default=None)
    parser.add_argument("--splits-json", type=str, default=None)
    parser.add_argument("--pxm-ckpt", type=str, default=None)
    parser.add_argument("--split-family", type=str, default="od", choices=["od", "id"])
    parser.add_argument("--ablation", type=str, default="full", choices=list(ABLATION_PRESETS.keys()))
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--k-confs-train", type=int, default=None)
    parser.add_argument("--amp", type=int, default=None)
    parser.add_argument("--use-grad-checkpoint", type=int, default=None)
    parser.add_argument("--run-baselines", action="store_true")
    parser.add_argument("--baselines-only", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    lmdb_path = args.lmdb_path or cfg["lmdb_path"]
    splits_json = args.splits_json or cfg["splits_json"]

    if args.run_baselines or args.baselines_only:
        run_2d_baselines_for_split(
            lmdb_path=lmdb_path,
            splits_json=splits_json,
            split_family=args.split_family,
            out_dir=args.output_dir,
            seed=int(cfg["train"]["seed"]),
        )
    if not args.baselines_only:
        train_3d_model(cfg, args)


if __name__ == "__main__":
    main()
