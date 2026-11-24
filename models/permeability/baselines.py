"""Baseline models (MorganFP-Ridge/MLP, RDKit2D-MLP), ablation presets, and CycDeltaBench metrics."""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.permeability.constants import ID_TO_ASSAY, NUM_ASSAYS, PERM_ASSAY_NAMES


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return float("nan")
    yt = y_true[mask].astype(np.float64)
    yp = y_pred[mask].astype(np.float64)
    yt = yt - yt.mean()
    yp = yp - yp.mean()
    denom = float(np.sqrt(np.sum(yt ** 2) * np.sum(yp ** 2)))
    if denom < 1e-12:
        return 0.0
    return float(np.sum(yt * yp) / denom)


def spearman_rho(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return float("nan")
    from scipy.stats import spearmanr

    res = spearmanr(y_true[mask], y_pred[mask])
    val = float(res.correlation if hasattr(res, "correlation") else res[0])
    return 0.0 if not np.isfinite(val) else val


def mae_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def rmse_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return {
        "n": int(mask.sum()),
        "pearson": pearson_r(y_true, y_pred),
        "spearman": spearman_rho(y_true, y_pred),
        "mae": mae_score(y_true, y_pred),
        "rmse": rmse_score(y_true, y_pred),
    }


def build_cycdelta_seed_pairs(
    y_true: np.ndarray,
    assay_ids: np.ndarray,
    source_ids: np.ndarray,
    seeds: Sequence[int] = tuple(range(10)),
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Build 10 reference-selection seeds matching CycDeltaBench protocol.

    Per (source_id, assay_id) group with >= 2 records, chooses 1 reference record per seed
    and pairs all remaining records in that group as queries against it.
    """
    n = len(y_true)
    groups: Dict[Tuple[int, int], List[int]] = {}
    for i in range(n):
        if not np.isfinite(y_true[i]):
            continue
        key = (int(source_ids[i]), int(assay_ids[i]))
        groups.setdefault(key, []).append(i)

    seed_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    for seed in seeds:
        rng = np.random.RandomState(int(seed))
        q_list: List[int] = []
        r_list: List[int] = []
        for _, idxs in sorted(groups.items()):
            if len(idxs) < 2:
                continue
            ref_pos = int(rng.randint(0, len(idxs)))
            r_idx = idxs[ref_pos]
            for pos, q_idx in enumerate(idxs):
                if pos != ref_pos:
                    q_list.append(q_idx)
                    r_list.append(r_idx)
        seed_pairs.append((np.asarray(q_list, dtype=np.int64), np.asarray(r_list, dtype=np.int64)))
    return seed_pairs


def evaluate_delta_over_seeds(
    y_true: np.ndarray,
    assay_ids: np.ndarray,
    source_ids: np.ndarray,
    predict_delta_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    seed_pairs: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    seeds: Sequence[int] = tuple(range(10)),
) -> Dict[str, Any]:
    """Evaluate Within-Source Delta and Reconstructed Absolute metrics across 10 ref seeds."""
    if seed_pairs is None:
        seed_pairs = build_cycdelta_seed_pairs(y_true, assay_ids, source_ids, seeds=seeds)

    delta_pearson: List[float] = []
    delta_spearman: List[float] = []
    delta_mae: List[float] = []
    abs_pearson: List[float] = []
    abs_spearman: List[float] = []
    abs_mae: List[float] = []
    n_pairs_list: List[int] = []

    for q_idx, r_idx in seed_pairs:
        if len(q_idx) == 0:
            continue
        p_assay = assay_ids[q_idx]
        dy_true = y_true[q_idx] - y_true[r_idx]
        dy_pred = predict_delta_fn(q_idx, r_idx, p_assay)
        y_rec = y_true[r_idx] + dy_pred

        dm = regression_metrics(dy_true, dy_pred)
        am = regression_metrics(y_true[q_idx], y_rec)

        delta_pearson.append(dm["pearson"])
        delta_spearman.append(dm["spearman"])
        delta_mae.append(dm["mae"])
        abs_pearson.append(am["pearson"])
        abs_spearman.append(am["spearman"])
        abs_mae.append(am["mae"])
        n_pairs_list.append(len(q_idx))

    def _mean_std(vals: List[float]) -> Dict[str, float]:
        arr = np.asarray(vals, dtype=np.float64)
        valid = arr[np.isfinite(arr)]
        if len(valid) == 0:
            return {"mean": float("nan"), "std": float("nan")}
        return {"mean": float(valid.mean()), "std": float(valid.std())}

    return {
        "num_seeds": len(delta_pearson),
        "n_pairs_per_seed": int(np.mean(n_pairs_list)) if n_pairs_list else 0,
        "within_source_delta": {
            "pearson": _mean_std(delta_pearson),
            "spearman": _mean_std(delta_spearman),
            "mae": _mean_std(delta_mae),
        },
        "reconstructed_abs": {
            "pearson": _mean_std(abs_pearson),
            "spearman": _mean_std(abs_spearman),
            "mae": _mean_std(abs_mae),
        },
    }


def evaluate_direct_per_assay(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    assay_ids: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Compute direct absolute prediction metrics overall and per permeability assay."""
    out: Dict[str, Dict[str, float]] = {"overall": regression_metrics(y_true, y_pred)}
    for a_id, a_name in ID_TO_ASSAY.items():
        if a_name not in PERM_ASSAY_NAMES and a_name != "UNKNOWN":
            continue
        mask = assay_ids == a_id
        if mask.any():
            out[a_name] = regression_metrics(y_true[mask], y_pred[mask])
    return out


class RidgeBaselineModel:
    """Fast exact closed-form Ridge regression baseline (direct + antisymmetric delta)."""

    def __init__(self, alphas: Sequence[float] = (0.1, 1.0, 10.0, 50.0, 200.0, 1000.0)) -> None:
        self.alphas = tuple(float(a) for a in alphas)
        self.w_direct: Optional[np.ndarray] = None
        self.b_direct: float = 0.0
        self.n_feat_infer: int = 0
        self.w_delta: Optional[np.ndarray] = None
        self.best_alpha_direct: float = 10.0
        self.best_alpha_delta: float = 10.0

    @staticmethod
    def _assay_onehot(assay_ids: np.ndarray) -> np.ndarray:
        oh = np.zeros((len(assay_ids), NUM_ASSAYS), dtype=np.float32)
        for i, a in enumerate(assay_ids):
            if 0 <= int(a) < NUM_ASSAYS:
                oh[i, int(a)] = 1.0
        return oh

    def fit_direct(
        self,
        x_train: np.ndarray,
        assay_train: np.ndarray,
        source_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        assay_val: np.ndarray,
        y_val: np.ndarray,
    ) -> float:
        x_tr = np.concatenate([x_train.astype(np.float64), self._assay_onehot(assay_train)], axis=1)
        x_va = np.concatenate([x_val.astype(np.float64), self._assay_onehot(assay_val)], axis=1)
        self.n_feat_infer = x_tr.shape[1]

        # Add source one-hot strictly for training nuisance absorption (zeroed at inference)
        max_src = int(max(source_train.max(), 0)) + 1 if len(source_train) > 0 else 1
        src_oh = np.zeros((len(source_train), max_src), dtype=np.float64)
        for i, s in enumerate(source_train):
            if 0 <= int(s) < max_src:
                src_oh[i, int(s)] = 1.0
        x_tr_full = np.concatenate([x_tr, src_oh], axis=1)

        x_mean = x_tr_full.mean(axis=0)
        y_mean = float(y_train.mean())
        xc = x_tr_full - x_mean
        yc = y_train - y_mean
        xtx = xc.T @ xc
        xty = xc.T @ yc
        d = xtx.shape[0]
        eye = np.eye(d, dtype=np.float64)

        best_val_mae = float("inf")
        best_w = None
        best_b = y_mean
        best_alpha = self.alphas[0]

        for alpha in self.alphas:
            w = np.linalg.solve(xtx + alpha * eye, xty)
            w_infer = w[: self.n_feat_infer]
            b_infer = y_mean - float(x_mean[: self.n_feat_infer] @ w_infer)
            pred_val = x_va @ w_infer + b_infer
            val_mae = mae_score(y_val, pred_val)
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_w = w_infer
                best_b = b_infer
                best_alpha = alpha

        self.w_direct = best_w
        self.b_direct = float(best_b)
        self.best_alpha_direct = float(best_alpha)
        return best_val_mae

    def predict_direct(self, x: np.ndarray, assay_ids: np.ndarray) -> np.ndarray:
        assert self.w_direct is not None
        xa = np.concatenate([x.astype(np.float64), self._assay_onehot(assay_ids)], axis=1)
        return (xa @ self.w_direct + self.b_direct).astype(np.float32)

    def fit_pairwise_delta(
        self,
        x_train: np.ndarray,
        assay_train: np.ndarray,
        source_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        assay_val: np.ndarray,
        source_val: np.ndarray,
        y_val: np.ndarray,
        max_train_pairs: int = 80000,
        seed: int = 42,
    ) -> float:
        """Fit exact antisymmetric ridge model on within-source difference vectors (x_q - x_r)."""
        rng = np.random.RandomState(seed)
        q_idx, r_idx = _sample_within_source_pairs_np(
            assay_train, source_train, y_train, max_pairs=max_train_pairs, rng=rng
        )
        # Note: purely linear model without intercept on (x_q - x_r) is guaranteed antisymmetric
        dx = x_train[q_idx].astype(np.float64) - x_train[r_idx].astype(np.float64)
        dy = (y_train[q_idx] - y_train[r_idx]).astype(np.float64)

        val_pairs = build_cycdelta_seed_pairs(y_val, assay_val, source_val, seeds=(0,))
        vq, vr = val_pairs[0]
        dx_val = x_val[vq].astype(np.float64) - x_val[vr].astype(np.float64)
        dy_val = (y_val[vq] - y_val[vr]).astype(np.float64)

        xtx = dx.T @ dx
        xty = dx.T @ dy
        eye = np.eye(dx.shape[1], dtype=np.float64)

        best_mae = float("inf")
        best_w = np.zeros((dx.shape[1],), dtype=np.float64)
        best_alpha = self.alphas[0]
        for alpha in self.alphas:
            w = np.linalg.solve(xtx + alpha * eye, xty)
            val_pred = dx_val @ w
            m = mae_score(dy_val, val_pred)
            if m < best_mae:
                best_mae = m
                best_w = w
                best_alpha = alpha

        self.w_delta = best_w
        self.best_alpha_delta = float(best_alpha)
        return best_mae

    def predict_delta(self, x_q: np.ndarray, x_r: np.ndarray) -> np.ndarray:
        assert self.w_delta is not None
        dx = x_q.astype(np.float64) - x_r.astype(np.float64)
        return (dx @ self.w_delta).astype(np.float32)


def _sample_within_source_pairs_np(
    assay_ids: np.ndarray,
    source_ids: np.ndarray,
    y_true: np.ndarray,
    max_pairs: int,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray]:
    groups: Dict[Tuple[int, int], List[int]] = {}
    for i in range(len(y_true)):
        if not np.isfinite(y_true[i]) or int(source_ids[i]) < 0 or int(assay_ids[i]) < 1:
            continue
        groups.setdefault((int(source_ids[i]), int(assay_ids[i])), []).append(i)

    q_all: List[int] = []
    r_all: List[int] = []
    for _, idxs in groups.items():
        k = len(idxs)
        if k < 2:
            continue
        if k * (k - 1) <= 2000:
            for a in range(k):
                for b in range(k):
                    if a != b:
                        q_all.append(idxs[a])
                        r_all.append(idxs[b])
        else:
            # Cap quadratic blowup in huge sources while preserving random symmetric coverage
            arr = np.asarray(idxs, dtype=np.int64)
            n_draw = min(2000, k * (k - 1))
            qa = rng.choice(arr, size=n_draw, replace=True)
            rb = rng.choice(arr, size=n_draw, replace=True)
            keep = qa != rb
            q_all.extend(qa[keep].tolist())
            r_all.extend(rb[keep].tolist())

    q_np = np.asarray(q_all, dtype=np.int64)
    r_np = np.asarray(r_all, dtype=np.int64)
    if len(q_np) > max_pairs:
        sel = rng.choice(len(q_np), size=max_pairs, replace=False)
        q_np = q_np[sel]
        r_np = r_np[sel]
    return q_np, r_np


class TabularPermMLP(nn.Module):
    """Multi-assay & exact antisymmetric pairwise DeltaPerm MLP baseline (MorganFP / RDKit2D)."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        assay_emb_dim: int = 32,
        max_sources: int = 256,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.assay_embed = nn.Embedding(NUM_ASSAYS, assay_emb_dim, padding_idx=0)
        self.base_abs_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.assay_res_head = nn.Sequential(
            nn.Linear(hidden_dim + assay_emb_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.source_offset = nn.Embedding(max_sources, 1)
        nn.init.zeros_(self.source_offset.weight)
        self.delta_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2 + assay_emb_dim),
            nn.Linear(hidden_dim * 2 + assay_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode_and_abs(
        self,
        x: torch.Tensor,
        assay_id: torch.Tensor,
        source_id: Optional[torch.Tensor] = None,
        apply_source_offset: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        g = self.encoder(x)
        e_a = self.assay_embed(assay_id.long())
        y_base = self.base_abs_head(g).squeeze(-1)
        is_known = ((assay_id >= 1) & (assay_id <= 4)).to(y_base.dtype)
        y_infer = y_base + is_known * self.assay_res_head(torch.cat([g, e_a], dim=-1)).squeeze(-1)
        if apply_source_offset and self.training and source_id is not None:
            valid = source_id >= 0
            safe_s = torch.where(valid, source_id.long().clamp(max=self.source_offset.num_embeddings - 1), torch.zeros_like(source_id))
            y_train = y_infer + self.source_offset(safe_s).squeeze(-1) * valid.to(y_infer.dtype)
        else:
            y_train = y_infer
        return g, y_infer, y_train

    def predict_delta(
        self,
        g_q: torch.Tensor,
        g_r: torch.Tensor,
        y_q: torch.Tensor,
        y_r: torch.Tensor,
        assay_id: torch.Tensor,
    ) -> torch.Tensor:
        e_a = self.assay_embed(assay_id.long())
        h_qr = self.delta_mlp(torch.cat([g_q - g_r, g_q * g_r, e_a], dim=-1)).squeeze(-1)
        h_rq = self.delta_mlp(torch.cat([g_r - g_q, g_r * g_q, e_a], dim=-1)).squeeze(-1)
        return (y_q - y_r) + (h_qr - h_rq)


def train_tabular_mlp_baseline(
    x_train: np.ndarray,
    assay_train: np.ndarray,
    source_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    assay_val: np.ndarray,
    source_val: np.ndarray,
    y_val: np.ndarray,
    mode: str = "joint",  # "direct" or "joint" (direct + pairwise delta)
    hidden_dim: int = 256,
    epochs: int = 60,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> Tuple[TabularPermMLP, Dict[str, float]]:
    """Train MorganFP-MLP or RDKit2D-MLP cleanly on exact split and return best checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)

    model = TabularPermMLP(in_dim=x_train.shape[1], hidden_dim=hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))

    xt = torch.from_numpy(x_train.astype(np.float32)).to(device)
    at = torch.from_numpy(assay_train.astype(np.int64)).to(device)
    st = torch.from_numpy(source_train.astype(np.int64)).to(device)
    yt = torch.from_numpy(y_train.astype(np.float32)).to(device)

    xv = torch.from_numpy(x_val.astype(np.float32)).to(device)
    av = torch.from_numpy(assay_val.astype(np.int64)).to(device)

    val_seed_pairs = build_cycdelta_seed_pairs(y_val, assay_val, source_val, seeds=(0, 1, 2))

    # Build source-grouped chunks per epoch so within-source pairs exist naturally in batches
    src_groups: Dict[Tuple[int, int], List[int]] = {}
    for idx in range(len(y_train)):
        src_groups.setdefault((int(source_train[idx]), int(assay_train[idx])), []).append(idx)

    best_score = -1e9
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val_info: Dict[str, float] = {}

    for epoch in range(epochs):
        model.train()
        chunks: List[List[int]] = []
        for _, idxs in src_groups.items():
            perm = list(idxs)
            rng.shuffle(perm)
            for s in range(0, len(perm), 16):
                chunks.append(perm[s : s + 16])
        rng.shuffle(chunks)

        batches: List[List[int]] = []
        cur: List[int] = []
        for ch in chunks:
            if len(cur) + len(ch) > batch_size and cur:
                batches.append(cur)
                cur = list(ch)
            else:
                cur.extend(ch)
        if cur:
            batches.append(cur)

        for b_idx in batches:
            bi = torch.as_tensor(b_idx, dtype=torch.long, device=device)
            opt.zero_grad(set_to_none=True)
            g, y_inf, y_tr = model.encode_and_abs(xt[bi], at[bi], st[bi], apply_source_offset=True)
            loss = F.smooth_l1_loss(y_tr, yt[bi], beta=0.5) + 1e-3 * (model.source_offset.weight ** 2).mean()

            if mode == "joint":
                m = len(b_idx)
                same_a = at[bi].unsqueeze(1) == at[bi].unsqueeze(0)
                same_s = st[bi].unsqueeze(1) == st[bi].unsqueeze(0)
                not_self = ~torch.eye(m, dtype=torch.bool, device=device)
                valid_p = same_a & same_s & not_self & (st[bi] >= 0).unsqueeze(1)
                iq, ir = torch.nonzero(valid_p, as_tuple=True)
                if iq.numel() > 0:
                    dy_hat = model.predict_delta(g[iq], g[ir], y_inf[iq], y_inf[ir], at[bi][iq])
                    dy_true = yt[bi][iq] - yt[bi][ir]
                    loss = loss + F.smooth_l1_loss(dy_hat, dy_true, beta=0.5)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        sched.step()

        model.eval()
        with torch.no_grad():
            gv, y_val_pred_t, _ = model.encode_and_abs(xv, av, None, apply_source_offset=False)
            y_val_pred = y_val_pred_t.cpu().numpy()
            abs_p = pearson_r(y_val, y_val_pred)

            def _pred_fn(qi: np.ndarray, ri: np.ndarray, pa: np.ndarray) -> np.ndarray:
                if mode == "direct":
                    return (y_val_pred_t[torch.as_tensor(qi, device=device)] - y_val_pred_t[torch.as_tensor(ri, device=device)]).cpu().numpy()
                q_t = torch.as_tensor(qi, dtype=torch.long, device=device)
                r_t = torch.as_tensor(ri, dtype=torch.long, device=device)
                a_t = torch.as_tensor(pa, dtype=torch.long, device=device)
                return model.predict_delta(gv[q_t], gv[r_t], y_val_pred_t[q_t], y_val_pred_t[r_t], a_t).cpu().numpy()

            d_eval = evaluate_delta_over_seeds(y_val, assay_val, source_val, _pred_fn, seed_pairs=val_seed_pairs)
            delta_p = d_eval["within_source_delta"]["pearson"]["mean"]
            comb = (delta_p if np.isfinite(delta_p) else 0.0) + 0.5 * (abs_p if np.isfinite(abs_p) else 0.0)
            if comb > best_score:
                best_score = comb
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_val_info = {
                    "val_abs_pearson": float(abs_p),
                    "val_delta_pearson": float(delta_p),
                    "best_epoch": int(epoch + 1),
                }

    model.load_state_dict(best_state)
    model.eval()
    return model, best_val_info


ABLATION_PRESETS: Dict[str, Dict[str, Any]] = {
    "full": {
        "load_pretrained": True,
        "lambda_abs": 1.0,
        "lambda_delta": 1.0,
        "lambda_cham": 0.3,
        "lambda_cons": 0.25,
        "lambda_aux_assay": 0.2,
        "k_confs_train": 8,
        "k_confs_eval": 8,
        "trunk_lr": 2e-5,
        "head_lr": 3e-4,
    },
    "NoPXMPretrain": {
        "load_pretrained": False,
        "lambda_abs": 1.0,
        "lambda_delta": 1.0,
        "lambda_cham": 0.3,
        "lambda_cons": 0.25,
        "lambda_aux_assay": 0.2,
        "k_confs_train": 8,
        "k_confs_eval": 8,
        "trunk_lr": 3e-4,
        "head_lr": 3e-4,
    },
    "NoChamAux": {
        "load_pretrained": True,
        "lambda_abs": 1.0,
        "lambda_delta": 1.0,
        "lambda_cham": 0.0,
        "lambda_cons": 0.25,
        "lambda_aux_assay": 0.2,
        "k_confs_train": 8,
        "k_confs_eval": 8,
        "trunk_lr": 2e-5,
        "head_lr": 3e-4,
    },
    "NoDeltaLoss": {
        "load_pretrained": True,
        "lambda_abs": 1.0,
        "lambda_delta": 0.0,
        "lambda_cham": 0.3,
        "lambda_cons": 0.25,
        "lambda_aux_assay": 0.2,
        "k_confs_train": 8,
        "k_confs_eval": 8,
        "trunk_lr": 2e-5,
        "head_lr": 3e-4,
    },
    "DeltaOnly": {
        "load_pretrained": True,
        "lambda_abs": 0.0,
        "lambda_delta": 1.0,
        "lambda_cham": 0.3,
        "lambda_cons": 0.25,
        "lambda_aux_assay": 0.0,
        "lambda_source_l2": 0.0,
        "k_confs_train": 8,
        "k_confs_eval": 8,
        "trunk_lr": 2e-5,
        "head_lr": 3e-4,
    },
    "K1SinglePose": {
        "load_pretrained": True,
        "lambda_abs": 1.0,
        "lambda_delta": 1.0,
        "lambda_cham": 0.3,
        "lambda_cons": 0.0,
        "lambda_aux_assay": 0.2,
        "k_confs_train": 1,
        "k_confs_eval": 1,
        "trunk_lr": 2e-5,
        "head_lr": 3e-4,
    },
}
