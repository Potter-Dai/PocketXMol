"""Multi-task Chameleonic Permeability & Exact Antisymmetric DeltaPerm heads."""

import contextlib
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.permeability.constants import NUM_ASSAYS, NUM_CHAM_TARGETS
from models.permeability.encoder import ChameleonicPXMEncoder


def _fp32_autocast_disable(is_cuda: bool):
    if not is_cuda:
        return contextlib.nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        try:
            return torch.amp.autocast("cuda", enabled=False)
        except TypeError:
            pass
    return torch.cuda.amp.autocast(enabled=False)


class PermCycXMolModel(nn.Module):
    """Unified Phase 1 3D Chameleonic Permeability & DeltaPerm model."""

    def __init__(
        self,
        encoder: ChameleonicPXMEncoder,
        trunk_dim: int = 256,
        g_dim: int = 256,
        cham_hidden_dim: int = 128,
        assay_emb_dim: int = 32,
        max_sources: int = 256,
        use_cham_in_trunk: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.conf_dim = int(encoder.conf_dim)
        self.trunk_dim = int(trunk_dim)
        self.g_dim = int(g_dim)
        self.cham_hidden_dim = int(cham_hidden_dim)
        self.assay_emb_dim = int(assay_emb_dim)
        self.use_cham_in_trunk = bool(use_cham_in_trunk)

        # Molecule projection over [z_boltz, z_disp]
        self.mol_proj = nn.Sequential(
            nn.LayerNorm(self.conf_dim * 2),
            nn.Linear(self.conf_dim * 2, self.trunk_dim),
            nn.SiLU(),
            nn.Linear(self.trunk_dim, self.trunk_dim),
            nn.LayerNorm(self.trunk_dim),
        )

        # (a) Chameleonic Auxiliary Distillation Head: [psa_3d_min, delta_psa_3d, imhb_max, rg_min, closure]
        self.cham_trunk = nn.Sequential(
            nn.Linear(self.trunk_dim, self.cham_hidden_dim),
            nn.LayerNorm(self.cham_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.cham_hidden_dim, self.cham_hidden_dim),
            nn.SiLU(),
        )
        self.cham_out = nn.Linear(self.cham_hidden_dim, NUM_CHAM_TARGETS)

        perm_in_dim = (
            self.trunk_dim + self.cham_hidden_dim + NUM_CHAM_TARGETS
            if self.use_cham_in_trunk
            else self.trunk_dim
        )
        self.perm_trunk = nn.Sequential(
            nn.LayerNorm(perm_in_dim),
            nn.Linear(perm_in_dim, self.g_dim),
            nn.SiLU(),
            nn.Linear(self.g_dim, self.g_dim),
            nn.LayerNorm(self.g_dim),
        )

        # (b) Multi-Assay Absolute Heads (index 0 UNKNOWN -> fixed zero vector via padding_idx=0)
        self.assay_embed = nn.Embedding(NUM_ASSAYS, self.assay_emb_dim, padding_idx=0)
        self.base_abs_head = nn.Sequential(
            nn.Linear(self.g_dim, self.g_dim // 2),
            nn.SiLU(),
            nn.Linear(self.g_dim // 2, 1),
        )
        self.assay_res_head = nn.Sequential(
            nn.Linear(self.g_dim + self.assay_emb_dim, self.g_dim // 2),
            nn.SiLU(),
            nn.Linear(self.g_dim // 2, 1),
        )
        # Separate SOLUBILITY auxiliary head (assay_id == 5)
        self.solubility_head = nn.Sequential(
            nn.Linear(self.g_dim, self.g_dim // 2),
            nn.SiLU(),
            nn.Linear(self.g_dim // 2, 1),
        )
        # Training-only L2-penalized inter-source baseline calibration offset (zeroed at inference)
        self.source_offset = nn.Embedding(max_sources, 1)
        nn.init.zeros_(self.source_offset.weight)

        # (c) Exact Antisymmetric Pairwise DeltaPerm Head h_delta(g_q - g_r, g_q * g_r, e_a)
        delta_in_dim = self.g_dim * 2 + self.assay_emb_dim
        self.delta_mlp = nn.Sequential(
            nn.LayerNorm(delta_in_dim),
            nn.Linear(delta_in_dim, self.g_dim),
            nn.SiLU(),
            nn.Linear(self.g_dim, self.g_dim // 2),
            nn.SiLU(),
            nn.Linear(self.g_dim // 2, 1),
        )

        # Optional normalization stats saved with checkpoint for Phase 3 physical-unit reporting
        self.register_buffer("cham_mean", torch.zeros((NUM_CHAM_TARGETS,), dtype=torch.float32))
        self.register_buffer("cham_std", torch.ones((NUM_CHAM_TARGETS,), dtype=torch.float32))
        self.register_buffer("atom_phys_mean", torch.zeros((14,), dtype=torch.float32))
        self.register_buffer("atom_phys_std", torch.ones((14,), dtype=torch.float32))

    def set_cham_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.cham_mean.copy_(mean.detach().to(self.cham_mean.device, dtype=torch.float32))
        self.cham_std.copy_(torch.clamp(std.detach().to(self.cham_std.device, dtype=torch.float32), min=1e-6))

    def set_atom_phys_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.atom_phys_mean.copy_(mean.detach().to(self.atom_phys_mean.device, dtype=torch.float32))
        self.atom_phys_std.copy_(torch.clamp(std.detach().to(self.atom_phys_std.device, dtype=torch.float32), min=1e-6))

    def pxm_trunk_parameters(self) -> List[nn.Parameter]:
        return self.encoder.pxm_trunk_parameters()

    def new_head_parameters(self) -> List[nn.Parameter]:
        params = list(self.encoder.new_head_parameters())
        for mod in (
            self.mol_proj,
            self.cham_trunk,
            self.cham_out,
            self.perm_trunk,
            self.assay_embed,
            self.base_abs_head,
            self.assay_res_head,
            self.solubility_head,
            self.source_offset,
            self.delta_mlp,
        ):
            params.extend(list(mod.parameters()))
        return params

    def forward_from_pooled(
        self,
        z_boltz: torch.Tensor,
        z_disp: torch.Tensor,
        assay_id: Optional[torch.Tensor] = None,
        source_id: Optional[torch.Tensor] = None,
        apply_source_offset: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Run molecule-level chameleonic and multi-assay permeability heads from pooled latents."""
        num_mols = int(z_boltz.shape[0])
        device = z_boltz.device
        with _fp32_autocast_disable(z_boltz.is_cuda):
            z_boltz = z_boltz.float()
            z_disp = z_disp.float()
            if assay_id is None:
                assay_id = torch.ones((num_mols,), dtype=torch.long, device=device)  # default PAMPA

            z_base = self.mol_proj(torch.cat([z_boltz, z_disp], dim=-1))
            cham_hidden = self.cham_trunk(z_base)
            cham_pred = self.cham_out(cham_hidden)

            if self.use_cham_in_trunk:
                g_in = torch.cat([z_base, cham_hidden, cham_pred], dim=-1)
            else:
                g_in = z_base
            g = self.perm_trunk(g_in)

            e_a = self.assay_embed(assay_id.long())
            y_base = self.base_abs_head(g).squeeze(-1)
            y_res = self.assay_res_head(torch.cat([g, e_a], dim=-1)).squeeze(-1)
            y_sol = self.solubility_head(g).squeeze(-1)

            is_known_perm = ((assay_id >= 1) & (assay_id <= 4)).to(y_base.dtype)
            is_sol = (assay_id == 5).to(y_base.dtype)
            # If assay_id == 0 (UNKNOWN), y_abs_infer == y_base cleanly
            y_abs_infer = (1.0 - is_sol) * (y_base + is_known_perm * y_res) + is_sol * y_sol

            if apply_source_offset and self.training and source_id is not None:
                valid_src = source_id >= 0
                safe_src = torch.where(
                    valid_src,
                    source_id.long().clamp(max=self.source_offset.num_embeddings - 1),
                    torch.zeros_like(source_id),
                )
                offset = self.source_offset(safe_src).squeeze(-1) * valid_src.to(y_abs_infer.dtype)
                y_abs_train = y_abs_infer + offset
            else:
                offset = y_abs_infer.new_zeros((num_mols,))
                y_abs_train = y_abs_infer

        return {
            "z_base": z_base,
            "cham_pred": cham_pred,
            "g": g,
            "y_base": y_base,
            "y_abs_infer": y_abs_infer,
            "y_abs_train": y_abs_train,
            "y_sol": y_sol,
            "source_offset": offset,
        }

    def predict_delta(
        self,
        g_q: torch.Tensor,
        g_r: torch.Tensor,
        y_q: torch.Tensor,
        y_r: torch.Tensor,
        assay_id: torch.Tensor,
    ) -> torch.Tensor:
        """Exact antisymmetric pairwise DeltaPerm prediction: dy(q, r) == -dy(r, q)."""
        with _fp32_autocast_disable(g_q.is_cuda):
            g_q = g_q.float()
            g_r = g_r.float()
            y_q = y_q.float()
            y_r = y_r.float()
            e_a = self.assay_embed(assay_id.long())
            feat_qr = torch.cat([g_q - g_r, g_q * g_r, e_a], dim=-1)
            feat_rq = torch.cat([g_r - g_q, g_r * g_q, e_a], dim=-1)
            h_qr = self.delta_mlp(feat_qr).squeeze(-1)
            h_rq = self.delta_mlp(feat_rq).squeeze(-1)
            return (y_q - y_r) + (h_qr - h_rq)

    def predict_abs_for_assay(self, g: torch.Tensor, assay_id_scalar: int) -> torch.Tensor:
        """Predict inference absolute value for a specific scalar assay id across a batch g."""
        with _fp32_autocast_disable(g.is_cuda):
            g = g.float()
            m = int(g.shape[0])
            a_t = torch.full((m,), int(assay_id_scalar), dtype=torch.long, device=g.device)
            y_base = self.base_abs_head(g).squeeze(-1)
            if assay_id_scalar == 0:
                return y_base
            if assay_id_scalar == 5:
                return self.solubility_head(g).squeeze(-1)
            e_a = self.assay_embed(a_t)
            return y_base + self.assay_res_head(torch.cat([g, e_a], dim=-1)).squeeze(-1)

    def forward(
        self,
        pos: torch.Tensor,
        halfedge_index: torch.Tensor,
        batch_node: torch.Tensor,
        conf_to_mol: Optional[torch.Tensor] = None,
        num_mols: Optional[int] = None,
        assay_id: Optional[torch.Tensor] = None,
        source_id: Optional[torch.Tensor] = None,
        apply_source_offset: bool = False,
        return_k1_view: bool = False,
        **encoder_kwargs: Any,
    ) -> Dict[str, Any]:
        """Full 3D encode -> Boltzmann ensemble pool -> chameleonic & permeability heads."""
        if conf_to_mol is None and num_mols is not None and "num_graphs" not in encoder_kwargs:
            encoder_kwargs["num_graphs"] = int(num_mols)
        enc_out = self.encoder(
            pos=pos,
            halfedge_index=halfedge_index,
            batch_node=batch_node,
            conf_to_mol=conf_to_mol,
            num_mols=num_mols,
            **encoder_kwargs,
        )
        head_out = self.forward_from_pooled(
            z_boltz=enc_out["z_boltz"],
            z_disp=enc_out["z_disp"],
            assay_id=assay_id,
            source_id=source_id,
            apply_source_offset=apply_source_offset,
        )

        out: Dict[str, Any] = {}
        out.update(enc_out)
        out.update(head_out)

        # Optional cheap K=1 subsampling consistency pass reusing z_conf (zero extra GNN cost!)
        if return_k1_view:
            z_conf = enc_out["z_conf"]
            g_count = int(z_conf.shape[0])
            if conf_to_mol is None:
                z_k1 = z_conf
            else:
                m_count = int(head_out["g"].shape[0])
                # Randomly pick 1 conformer per molecule when training, or first conformer in eval
                picked_indices = []
                for m_idx in range(m_count):
                    cand = torch.nonzero(conf_to_mol == m_idx, as_tuple=False).view(-1)
                    if cand.numel() == 1:
                        picked_indices.append(cand[0])
                    elif self.training:
                        r_pos = int(torch.randint(0, int(cand.numel()), (1,), device=z_conf.device).item())
                        picked_indices.append(cand[r_pos])
                    else:
                        picked_indices.append(cand[0])
                pick_t = torch.stack(picked_indices)
                z_k1 = z_conf[pick_t]
            z_disp_zero = torch.zeros_like(z_k1)
            k1_out = self.forward_from_pooled(
                z_boltz=z_k1,
                z_disp=z_disp_zero,
                assay_id=assay_id,
                source_id=None,
                apply_source_offset=False,
            )
            out["k1_cham_pred"] = k1_out["cham_pred"]
            out["k1_g"] = k1_out["g"]
            out["k1_y_abs_infer"] = k1_out["y_abs_infer"]

        return out


def extract_within_source_pairs(
    assay_id: torch.Tensor,
    source_id: torch.Tensor,
    y_true: torch.Tensor,
    y_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract all valid ordered within-(source, permeability_assay) pairs in a batch."""
    m = int(y_true.shape[0])
    device = y_true.device
    if m < 2:
        empty = torch.zeros((0,), dtype=torch.long, device=device)
        return empty, empty, y_true.new_zeros((0,)), empty

    valid = torch.isfinite(y_true) & (assay_id >= 1) & (assay_id <= 4) & (source_id >= 0)
    if y_mask is not None:
        valid = valid & y_mask.bool()

    same_assay = assay_id.unsqueeze(1) == assay_id.unsqueeze(0)
    same_src = source_id.unsqueeze(1) == source_id.unsqueeze(0)
    both_valid = valid.unsqueeze(1) & valid.unsqueeze(0)
    not_self = ~torch.eye(m, dtype=torch.bool, device=device)

    pair_mask = same_assay & same_src & both_valid & not_self
    idx_q, idx_r = torch.nonzero(pair_mask, as_tuple=True)
    if idx_q.numel() == 0:
        empty = torch.zeros((0,), dtype=torch.long, device=device)
        return empty, empty, y_true.new_zeros((0,)), empty

    dy_true = y_true[idx_q] - y_true[idx_r]
    pair_assay = assay_id[idx_q]
    return idx_q, idx_r, dy_true, pair_assay


def compute_multitask_loss(
    model: PermCycXMolModel,
    outputs: Dict[str, Any],
    batch: Dict[str, Any],
    lambda_abs: float = 1.0,
    lambda_delta: float = 1.0,
    lambda_cham: float = 0.3,
    lambda_cons: float = 0.25,
    lambda_aux_assay: float = 0.2,
    lambda_source_l2: float = 1e-3,
    smooth_l1_beta: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute multi-task permeability, pairwise delta, chameleonic aux, and K=1 consistency loss."""
    y_true = batch["y_perm"]
    y_mask = batch.get("y_mask", torch.isfinite(y_true))
    assay_id = batch["assay_id"]
    source_id = batch["source_id"]

    total_loss = y_true.new_zeros(())
    stats: Dict[str, float] = {}

    # 1. Absolute primary permeability loss (SmoothL1 beta=0.5, training-only source offset)
    if lambda_abs > 0.0 and y_mask.any():
        l_abs = F.smooth_l1_loss(
            outputs["y_abs_train"][y_mask],
            y_true[y_mask],
            beta=smooth_l1_beta,
            reduction="mean",
        )
        total_loss = total_loss + lambda_abs * l_abs
        stats["loss_abs"] = float(l_abs.detach().item())
        if lambda_source_l2 > 0.0:
            l_src = (model.source_offset.weight ** 2).mean()
            total_loss = total_loss + lambda_source_l2 * l_src
            stats["loss_src_l2"] = float(l_src.detach().item())
    else:
        stats["loss_abs"] = 0.0

    # 2. Exact antisymmetric pairwise within-source DeltaPerm loss
    if lambda_delta > 0.0:
        idx_q, idx_r, dy_true, pair_assay = extract_within_source_pairs(
            assay_id=assay_id,
            source_id=source_id,
            y_true=y_true,
            y_mask=y_mask,
        )
        if idx_q.numel() > 0:
            g = outputs["g"]
            y_inf = outputs["y_abs_infer"]
            dy_hat = model.predict_delta(
                g_q=g[idx_q],
                g_r=g[idx_r],
                y_q=y_inf[idx_q],
                y_r=y_inf[idx_r],
                assay_id=pair_assay,
            )
            l_delta = F.smooth_l1_loss(dy_hat, dy_true, beta=smooth_l1_beta, reduction="mean")
            total_loss = total_loss + lambda_delta * l_delta
            stats["loss_delta"] = float(l_delta.detach().item())
            stats["num_pairs"] = float(idx_q.numel())
        else:
            stats["loss_delta"] = 0.0
            stats["num_pairs"] = 0.0
    else:
        stats["loss_delta"] = 0.0
        stats["num_pairs"] = 0.0

    # 3. Chameleonic 3D auxiliary distillation loss
    if lambda_cham > 0.0 and "cham_targets_norm" in batch:
        cham_t = batch["cham_targets_norm"]
        cham_m = batch.get("cham_mask", torch.isfinite(cham_t))
        if cham_m.any():
            cham_err = F.smooth_l1_loss(outputs["cham_pred"], torch.nan_to_num(cham_t, 0.0), beta=0.5, reduction="none")
            l_cham = (cham_err * cham_m.to(cham_err.dtype)).sum() / torch.clamp(cham_m.sum().to(cham_err.dtype), min=1.0)
            total_loss = total_loss + lambda_cham * l_cham
            stats["loss_cham"] = float(l_cham.detach().item())
        else:
            stats["loss_cham"] = 0.0
    else:
        stats["loss_cham"] = 0.0

    # 4. Auxiliary multi-assay enrichment labels (from EnsembleCycPerm cross-ref if present)
    if lambda_aux_assay > 0.0 and "aux_y" in batch and "aux_mask" in batch:
        aux_y = batch["aux_y"]      # [M, 6]
        aux_mask = batch["aux_mask"]  # [M, 6]
        aux_losses: List[torch.Tensor] = []
        g = outputs["g"]
        for a_id in (1, 2, 3, 4, 5):
            m_a = aux_mask[:, a_id] & torch.isfinite(aux_y[:, a_id])
            if m_a.any():
                pred_a = model.predict_abs_for_assay(g[m_a], a_id)
                aux_losses.append(F.smooth_l1_loss(pred_a, aux_y[m_a, a_id], beta=smooth_l1_beta, reduction="mean"))
        if aux_losses:
            l_aux = torch.stack(aux_losses).mean()
            total_loss = total_loss + lambda_aux_assay * l_aux
            stats["loss_aux_assay"] = float(l_aux.detach().item())
        else:
            stats["loss_aux_assay"] = 0.0
    else:
        stats["loss_aux_assay"] = 0.0

    # 5. K=1 vs K>=2 ensemble consistency loss (teacher stop-grad on full ensemble)
    if lambda_cons > 0.0 and "k1_y_abs_infer" in outputs:
        l_cons_y = F.smooth_l1_loss(outputs["k1_y_abs_infer"], outputs["y_abs_infer"].detach(), beta=0.25)
        l_cons_g = F.mse_loss(outputs["k1_g"], outputs["g"].detach())
        l_cons_c = F.mse_loss(outputs["k1_cham_pred"], outputs["cham_pred"].detach())
        l_cons = l_cons_y + 0.5 * l_cons_g + 0.5 * l_cons_c
        total_loss = total_loss + lambda_cons * l_cons
        stats["loss_cons"] = float(l_cons.detach().item())
    else:
        stats["loss_cons"] = 0.0

    stats["loss_total"] = float(total_loss.detach().item())
    return total_loss, stats
