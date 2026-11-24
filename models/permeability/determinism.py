"""Deterministic FP32 scatter utilities for exact cross-run reproducibility."""

import os
import sys
from typing import Any, Dict, Optional, Tuple

import torch

_SAVED_STATE: Optional[Dict[str, Any]] = None


def det_scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    """Deterministic drop-in replacement for torch_scatter.scatter_sum/scatter_add."""
    d = dim if dim >= 0 else src.ndim + dim
    idx = index.long()
    if out is None:
        if dim_size is None:
            dim_size = int(idx.max().item()) + 1 if idx.numel() > 0 else 0
        shape = list(src.shape)
        shape[d] = int(dim_size)
        out = torch.zeros(shape, dtype=src.dtype, device=src.device)
    if src.numel() == 0 or idx.numel() == 0:
        return out
    if idx.dim() == 1:
        return out.index_add_(d, idx, src)
    view_shape = list(idx.shape) + [1] * (src.ndim - idx.dim())
    idx_exp = idx.view(view_shape).expand_as(src)
    return out.scatter_add_(d, idx_exp, src)


def det_scatter_mean(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    """Deterministic drop-in replacement for torch_scatter.scatter_mean."""
    d = dim if dim >= 0 else src.ndim + dim
    idx = index.long()
    sums = det_scatter_sum(src, idx, dim=d, out=out, dim_size=dim_size)
    if idx.dim() == 1:
        ones = torch.ones((idx.numel(),), dtype=src.dtype, device=src.device)
        counts = torch.zeros((sums.shape[d],), dtype=src.dtype, device=src.device)
        if idx.numel() > 0:
            counts.index_add_(0, idx, ones)
        view_shape = [1] * sums.ndim
        view_shape[d] = sums.shape[d]
        counts_v = counts.view(view_shape)
    else:
        ones = torch.ones_like(src)
        counts_v = det_scatter_sum(ones, idx, dim=d, dim_size=sums.shape[d])
    return torch.where(counts_v > 0, sums / counts_v.clamp_min(1.0), sums)


def _patch_targets() -> Tuple[Any, ...]:
    import torch_scatter

    targets = [torch_scatter]
    try:
        import torch_scatter.scatter as ts_scatter

        targets.append(ts_scatter)
    except Exception:
        pass
    for mod_name in (
        "models.permeability.encoder",
        "models.graph_context",
        "models.common",
        "models.graph",
        "models.ipa",
        "models.corrector",
    ):
        mod = sys.modules.get(mod_name)
        if mod is not None:
            targets.append(mod)
    return tuple(targets)


def enable_deterministic_eval() -> None:
    """Enable deterministic FP32 execution and non-atomic scatter ops for eval."""
    global _SAVED_STATE
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    saved_attrs: Dict[Tuple[int, str], Tuple[Any, str, Any]] = {}
    replacements = {
        "scatter_sum": det_scatter_sum,
        "scatter_add": det_scatter_sum,
        "scatter_mean": det_scatter_mean,
    }
    for obj in _patch_targets():
        for attr, fn in replacements.items():
            if hasattr(obj, attr):
                key = (id(obj), attr)
                if key not in saved_attrs:
                    saved_attrs[key] = (obj, attr, getattr(obj, attr))
                setattr(obj, attr, fn)

    if _SAVED_STATE is None:
        _SAVED_STATE = {
            "attrs": saved_attrs,
            "matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
            "cudnn_det": bool(torch.backends.cudnn.deterministic),
            "cudnn_bench": bool(torch.backends.cudnn.benchmark),
        }

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def restore_eval_state() -> None:
    """Restore default scatter kernels and CUDA graph friendly runtime flags."""
    global _SAVED_STATE
    torch.use_deterministic_algorithms(False)
    if _SAVED_STATE is None:
        return
    for obj, attr, orig_fn in _SAVED_STATE["attrs"].values():
        setattr(obj, attr, orig_fn)
    torch.backends.cuda.matmul.allow_tf32 = _SAVED_STATE["matmul_tf32"]
    torch.backends.cudnn.allow_tf32 = _SAVED_STATE["cudnn_tf32"]
    torch.backends.cudnn.deterministic = _SAVED_STATE["cudnn_det"]
    torch.backends.cudnn.benchmark = _SAVED_STATE["cudnn_bench"]
    _SAVED_STATE = None
