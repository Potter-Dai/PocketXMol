"""3D Chameleonic PocketXMol encoder with Polar Shielding & Boltzmann conformer pooling."""

import copy
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from torch_geometric.utils import softmax as pyg_softmax
from torch_scatter import scatter_max, scatter_mean, scatter_min, scatter_sum

from models.maskfill import PMAsymDenoiser
from models.permeability.constants import (
    DEFAULT_PXM_MODEL_CFG,
    NUM_PXM_EDGE_TYPES,
    NUM_PXM_NODE_TYPES,
    POCKET_IN_DIM,
    atomic_numbers_to_pxm_idx,
)


def _to_easydict(obj: Any) -> Any:
    if isinstance(obj, EasyDict):
        return EasyDict({k: _to_easydict(v) for k, v in obj.items()})
    if isinstance(obj, dict):
        return EasyDict({k: _to_easydict(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_easydict(v) for v in obj]
    return obj


def build_complete_halfedges(
    num_nodes_per_graph: List[int],
    bond_index: Optional[torch.Tensor] = None,
    bond_type: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build PXM-exact upper-triangular halfedges (u < v) and map covalent bond types.

    Args:
        num_nodes_per_graph: Heavy atom count for each conformer graph in the batch.
        bond_index: Optional [2, E_bond] global atom index pairs (symmetric or directed).
        bond_type: Optional [E_bond] bond orders in {1, 2, 3, 4} (or 0..4).
        device: Target tensor device.

    Returns:
        halfedge_index: LongTensor [2, E_half] with halfedge_index[0] < halfedge_index[1].
        halfedge_type: LongTensor [E_half] in {0=non-bonded, 1, 2, 3, 4}.
        halfedge_batch: LongTensor [E_half] graph index for each halfedge.
    """
    if device is None:
        if bond_index is not None:
            device = bond_index.device
        else:
            device = torch.device("cpu")

    halfedge_indices: List[torch.Tensor] = []
    halfedge_batches: List[torch.Tensor] = []
    graph_offsets: List[int] = []
    halfedge_offsets: List[int] = []

    node_offset = 0
    edge_offset = 0
    for g_idx, n_nodes in enumerate(num_nodes_per_graph):
        graph_offsets.append(node_offset)
        halfedge_offsets.append(edge_offset)
        if n_nodes >= 2:
            triu = torch.triu_indices(n_nodes, n_nodes, offset=1, device=device)
            halfedge_indices.append(triu + node_offset)
            num_pairs = int(triu.shape[1])
            halfedge_batches.append(torch.full((num_pairs,), g_idx, dtype=torch.long, device=device))
            edge_offset += num_pairs
        node_offset += n_nodes

    if len(halfedge_indices) == 0:
        empty_idx = torch.zeros((2, 0), dtype=torch.long, device=device)
        empty_val = torch.zeros((0,), dtype=torch.long, device=device)
        return empty_idx, empty_val, empty_val

    halfedge_index = torch.cat(halfedge_indices, dim=1)
    halfedge_batch = torch.cat(halfedge_batches, dim=0)
    halfedge_type = torch.zeros((halfedge_index.shape[1],), dtype=torch.long, device=device)

    if bond_index is not None and bond_type is not None and bond_index.numel() > 0:
        # Map global node index -> graph id & local offset in O(N_total)
        total_nodes = node_offset
        node_to_graph = torch.empty((total_nodes,), dtype=torch.long, device=device)
        cur = 0
        for g_idx, n_nodes in enumerate(num_nodes_per_graph):
            if n_nodes > 0:
                node_to_graph[cur : cur + n_nodes] = g_idx
            cur += n_nodes

        graph_offsets_t = torch.as_tensor(graph_offsets, dtype=torch.long, device=device)
        halfedge_offsets_t = torch.as_tensor(halfedge_offsets, dtype=torch.long, device=device)
        n_per_graph_t = torch.as_tensor(num_nodes_per_graph, dtype=torch.long, device=device)

        src = bond_index[0].long()
        dst = bond_index[1].long()
        keep = src < dst
        if keep.any():
            u = src[keep]
            v = dst[keep]
            bt = bond_type[keep].long()
            g_id = node_to_graph[u]
            n_g = n_per_graph_t[g_id]
            lu = u - graph_offsets_t[g_id]
            lv = v - graph_offsets_t[g_id]
            local_pair_idx = lu * n_g - (lu * (lu + 1)) // 2 + (lv - lu - 1)
            global_pair_idx = halfedge_offsets_t[g_id] + local_pair_idx
            halfedge_type[global_pair_idx] = bt

    return halfedge_index, halfedge_type, halfedge_batch


def batched_knn_graph(
    pos: torch.Tensor,
    k: int,
    batch: torch.Tensor,
    flow: str = "target_to_source",
) -> torch.Tensor:
    """Compute kNN graph edges per batch element purely in PyTorch without torch_cluster."""
    device = pos.device
    num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    src_list: List[torch.Tensor] = []
    dst_list: List[torch.Tensor] = []

    for g in range(num_graphs):
        idx = torch.nonzero(batch == g, as_tuple=False).view(-1)
        n = int(idx.numel())
        if n <= 1:
            continue
        p = pos[idx]
        dist2 = torch.cdist(p, p, p=2.0).pow(2)
        dist2.fill_diagonal_(float("inf"))
        k_eff = min(k, n - 1)
        _, nn_local = torch.topk(dist2, k=k_eff, dim=-1, largest=False)
        centers = idx.unsqueeze(1).expand(-1, k_eff).reshape(-1)
        neighbors = idx[nn_local].reshape(-1)
        if flow == "target_to_source":
            # Edge (j -> i): target i receives from source j
            src_list.append(neighbors)
            dst_list.append(centers)
        else:
            src_list.append(centers)
            dst_list.append(neighbors)

    if not src_list:
        return torch.zeros((2, 0), dtype=torch.long, device=device)
    return torch.stack([torch.cat(src_list), torch.cat(dst_list)], dim=0)


class PolarShieldingTransannularLayer(nn.Module):
    """Differentiable 3D polar shielding and transannular IMHB contact layer.

    Operates cleanly on both discrete masks and soft probabilities (`polar_probs`,
    `covalent_probs`) so gradients flow cleanly to `pos`, `node_logits`, and
    `halfedge_logits`.
    """

    def __init__(
        self,
        node_dim: int,
        hidden_dim: int = 128,
        imhb_cutoff: float = 3.6,
        imhb_temp: float = 0.25,
        crowd_cutoff: float = 4.5,
        crowd_temp: float = 0.4,
        max_dense_nodes: int = 384,
    ) -> None:
        super().__init__()
        self.imhb_cutoff = float(imhb_cutoff)
        self.imhb_temp = float(imhb_temp)
        self.crowd_cutoff = float(crowd_cutoff)
        self.crowd_temp = float(crowd_temp)
        self.max_dense_nodes = int(max_dense_nodes)

        self.atom_feat_dim = 8
        self.global_geom_dim = 6
        self.node_proj = nn.Sequential(
            nn.Linear(self.atom_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim),
        )

    def _compute_transannular_mask(
        self,
        halfedge_index: torch.Tensor,
        covalent_probs: torch.Tensor,
        halfedge_batch: torch.Tensor,
        batch_node: torch.Tensor,
        num_graphs: int,
        max_nodes_per_graph: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute differentiable transannular (>=4 covalent bond) mask per halfedge.

        Excludes 1-2, 1-3, and 1-4 covalently separated pairs via soft adjacency powers
        A + A^2 + A^3 per graph when graph size <= max_dense_nodes.
        """
        non_bonded = torch.clamp(1.0 - covalent_probs, 0.0, 1.0)
        if halfedge_index.numel() == 0 or num_graphs == 0:
            return non_bonded

        if num_graphs == 1:
            n_max = int(batch_node.shape[0])
            if n_max > self.max_dense_nodes or n_max < 2:
                return non_bonded
            u_loc = halfedge_index[0]
            v_loc = halfedge_index[1]
            adj = torch.zeros(
                (1, n_max, n_max),
                dtype=covalent_probs.dtype,
                device=covalent_probs.device,
            )
            cov = torch.clamp(covalent_probs, 0.0, 1.0)
            adj[0, u_loc, v_loc] = cov
            adj[0, v_loc, u_loc] = cov
            adj2 = torch.bmm(adj, adj)
            adj3 = torch.bmm(adj2, adj)
            within_3bonds = torch.clamp(adj + adj2 + adj3, 0.0, 1.0)
            mask_ge4 = 1.0 - within_3bonds[0, u_loc, v_loc]
            return torch.clamp(mask_ge4, 0.0, 1.0) * non_bonded

        counts = scatter_sum(
            torch.ones_like(batch_node), batch_node, dim=0, dim_size=num_graphs
        )
        n_max = (
            int(max_nodes_per_graph)
            if max_nodes_per_graph is not None
            else (int(counts.max().item()) if counts.numel() > 0 else 0)
        )
        if n_max > self.max_dense_nodes or n_max < 2:
            return non_bonded

        # Compute local atom indices 0..n_g-1 inside each graph
        starts = torch.cumsum(counts, dim=0) - counts
        global_idx = torch.arange(batch_node.shape[0], device=batch_node.device)
        local_idx = global_idx - starts[batch_node]

        u_loc = local_idx[halfedge_index[0]]
        v_loc = local_idx[halfedge_index[1]]

        adj = torch.zeros(
            (num_graphs, n_max, n_max),
            dtype=covalent_probs.dtype,
            device=covalent_probs.device,
        )
        cov = torch.clamp(covalent_probs, 0.0, 1.0)
        adj[halfedge_batch, u_loc, v_loc] = cov
        adj[halfedge_batch, v_loc, u_loc] = cov

        adj2 = torch.bmm(adj, adj)
        adj3 = torch.bmm(adj2, adj)
        within_3bonds = torch.clamp(adj + adj2 + adj3, 0.0, 1.0)
        mask_ge4 = 1.0 - within_3bonds[halfedge_batch, u_loc, v_loc]
        return torch.clamp(mask_ge4, 0.0, 1.0) * non_bonded

    def forward(
        self,
        pos: torch.Tensor,
        polar_probs: torch.Tensor,
        covalent_probs: torch.Tensor,
        halfedge_index: torch.Tensor,
        batch_node: torch.Tensor,
        halfedge_batch: Optional[torch.Tensor] = None,
        num_graphs: Optional[int] = None,
        max_nodes_per_graph: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute per-atom polar shielding embeddings and differentiable conformer geometry.

        Returns:
            atom_shield_emb: [N_total, node_dim] projected shielding/IMHB embedding.
            global_geom: [G, 6] smooth physical 3D descriptors per conformer graph.
            metrics: dict of scalar per-conformer tensors (detached-friendly for logging).
        """
        n_total = int(pos.shape[0])
        if num_graphs is None:
            num_graphs = int(batch_node.max().item()) + 1 if batch_node.numel() > 0 else 0
        if halfedge_batch is None and num_graphs > 1:
            halfedge_batch = batch_node[halfedge_index[0]] if halfedge_index.numel() > 0 else torch.zeros(
                (0,), dtype=torch.long, device=pos.device
            )

        u = halfedge_index[0]
        v = halfedge_index[1]
        if halfedge_index.numel() > 0:
            diff = pos[u] - pos[v]
            dist = torch.sqrt((diff * diff).sum(dim=-1) + 1e-8)
            non_bonded = torch.clamp(1.0 - covalent_probs, 0.0, 1.0)
            trans_mask = self._compute_transannular_mask(
                halfedge_index=halfedge_index,
                covalent_probs=covalent_probs,
                halfedge_batch=halfedge_batch,
                batch_node=batch_node,
                num_graphs=num_graphs,
                max_nodes_per_graph=max_nodes_per_graph,
            )

            imhb_kern = torch.sigmoid((self.imhb_cutoff - dist) / self.imhb_temp)
            crowd_kern = torch.sigmoid((self.crowd_cutoff - dist) / self.crowd_temp)

            p_u = polar_probs[u]
            p_v = polar_probs[v]
            a_u = 1.0 - p_u
            a_v = 1.0 - p_v

            pair_imhb_nb = p_u * p_v * non_bonded * imhb_kern
            pair_imhb_trans = p_u * p_v * trans_mask * imhb_kern
            pair_hydro_trans = a_u * a_v * trans_mask * crowd_kern

            # Directed steric crowding weights onto atom u (from v) and onto v (from u)
            crowd_on_u_apolar = p_u * a_v * non_bonded * crowd_kern
            crowd_on_v_apolar = p_v * a_u * non_bonded * crowd_kern
            crowd_nb_all = non_bonded * crowd_kern

            pair_u_feats = torch.stack(
                [pair_imhb_nb, pair_imhb_trans, crowd_on_u_apolar, crowd_nb_all, pair_hydro_trans],
                dim=-1,
            )
            pair_v_feats = torch.stack(
                [pair_imhb_nb, pair_imhb_trans, crowd_on_v_apolar, crowd_nb_all, pair_hydro_trans],
                dim=-1,
            )
            atom_edge_stats = scatter_sum(pair_u_feats, u, dim=0, dim_size=n_total) + scatter_sum(
                pair_v_feats, v, dim=0, dim_size=n_total
            )
            (
                imhb_nb_atom,
                imhb_trans_atom,
                apolar_crowd_atom,
                all_crowd_atom,
                hydro_trans_atom,
            ) = atom_edge_stats.unbind(dim=-1)

            if num_graphs == 1:
                conf_imhb_nb = pair_imhb_nb.sum().unsqueeze(0)
                conf_imhb_trans = pair_imhb_trans.sum().unsqueeze(0)
            else:
                conf_imhb_stats = scatter_sum(
                    torch.stack([pair_imhb_nb, pair_imhb_trans], dim=-1),
                    halfedge_batch,
                    dim=0,
                    dim_size=num_graphs,
                )
                conf_imhb_nb, conf_imhb_trans = conf_imhb_stats.unbind(dim=-1)
        else:
            zeros_n = pos.new_zeros((n_total,))
            zeros_g = pos.new_zeros((num_graphs,))
            imhb_nb_atom = zeros_n
            imhb_trans_atom = zeros_n
            apolar_crowd_atom = zeros_n
            all_crowd_atom = zeros_n
            hydro_trans_atom = zeros_n
            conf_imhb_nb = zeros_g
            conf_imhb_trans = zeros_g

        polar_exposure_atom = polar_probs / (1.0 + apolar_crowd_atom)

        # Centroid-relative burial & smooth Rg
        if num_graphs == 1 and n_total > 0:
            centroid = pos.mean(dim=0, keepdim=True)
            rel = pos - centroid
            r2 = (rel * rel).sum(dim=-1)
            dist_centroid = torch.sqrt(r2 + 1e-8)

            rg = torch.sqrt(r2.mean().unsqueeze(0) + 1e-8)
            norm_radial = dist_centroid / (rg + 1e-4)

            graph_sums = torch.stack(
                [polar_probs, polar_probs * r2, apolar_crowd_atom, polar_exposure_atom],
                dim=-1,
            ).sum(dim=0, keepdim=True)
            polar_sum, polar_r2_sum, apolar_crowd_sum, polar_exposure_sum = graph_sums.unbind(dim=-1)
        else:
            centroid = scatter_mean(pos, batch_node, dim=0, dim_size=num_graphs)
            rel = pos - centroid[batch_node]
            r2 = (rel * rel).sum(dim=-1)
            dist_centroid = torch.sqrt(r2 + 1e-8)

            rg = torch.sqrt(scatter_mean(r2, batch_node, dim=0, dim_size=num_graphs) + 1e-8)
            norm_radial = dist_centroid / (rg[batch_node] + 1e-4)

            graph_sums = scatter_sum(
                torch.stack([polar_probs, polar_probs * r2, apolar_crowd_atom, polar_exposure_atom], dim=-1),
                batch_node,
                dim=0,
                dim_size=num_graphs,
            )
            polar_sum, polar_r2_sum, apolar_crowd_sum, polar_exposure_sum = graph_sums.unbind(dim=-1)

        polar_rg = torch.sqrt(polar_r2_sum / (polar_sum + 1e-4) + 1e-8)
        polar_rg_ratio = polar_rg / (rg + 1e-4)
        mean_polar_shield = apolar_crowd_sum / (polar_sum + 1e-4)
        mean_polar_exposure = polar_exposure_sum / (polar_sum + 1e-4)

        atom_shield_feats = torch.stack(
            [
                imhb_nb_atom,
                imhb_trans_atom,
                polar_exposure_atom,
                apolar_crowd_atom * 0.1,
                all_crowd_atom * 0.1,
                hydro_trans_atom * 0.1,
                polar_probs,
                polar_probs * norm_radial,
            ],
            dim=-1,
        )
        atom_shield_emb = self.node_proj(atom_shield_feats)

        global_geom = torch.stack(
            [
                conf_imhb_nb,
                conf_imhb_trans,
                mean_polar_shield,
                mean_polar_exposure,
                rg * 0.1,
                polar_rg_ratio,
            ],
            dim=-1,
        )
        metrics = {
            "smooth_imhb_nb": conf_imhb_nb,
            "smooth_imhb_trans": conf_imhb_trans,
            "smooth_polar_shield": mean_polar_shield,
            "smooth_polar_exposure": mean_polar_exposure,
            "smooth_rg": rg,
            "smooth_polar_rg_ratio": polar_rg_ratio,
        }
        return atom_shield_emb, global_geom, metrics


class ChameleonicPXMEncoder(nn.Module):
    """3D conformer and ensemble encoder initialized from PocketXMol (`pocketxmol.ckpt`).

    Supports:
      - Pretrained PXM backbone (`nodetype_embedder`, `edgetype_embedder`,
        `pocket_embedder`, `pocket_encoder`, `denoiser`) or random-init ablation.
      - Empty pocket (`h_ctx=None`) for standalone 3D cyclic peptide conformers AND
        holo receptor pockets (`pocket_pos`, `pocket_x`, `batch_pocket`).
      - Both discrete node/edge indices and differentiable soft logits (`node_logits`,
        `halfedge_logits`) for Phase 3 guidance gradients.
      - Single-conformer `K=1` and multi-conformer `K>=2` Boltzmann pooling +
        conformational plasticity dispersion (`max_k z_k - min_k z_k`).
    """

    def __init__(
        self,
        pxm_ckpt_path: Optional[str] = None,
        load_pretrained: bool = True,
        atom_phys_dim: int = 14,
        conf_dim: int = 256,
        shield_hidden_dim: int = 128,
        phys_dropout_prob: float = 0.25,
        use_grad_checkpoint: bool = False,
        max_graphs_per_chunk: int = 32,
        max_edges_per_chunk: int = 65536,
        pxm_model_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.atom_phys_dim = int(atom_phys_dim)
        self.conf_dim = int(conf_dim)
        self.phys_dropout_prob = float(phys_dropout_prob)
        self.use_grad_checkpoint = bool(use_grad_checkpoint)
        self.max_graphs_per_chunk = int(max_graphs_per_chunk)
        self.max_edges_per_chunk = int(max_edges_per_chunk)
        self.load_pretrained = bool(load_pretrained)

        ckpt_data = None
        if pxm_ckpt_path is not None:
            ckpt_data = torch.load(pxm_ckpt_path, map_location="cpu", weights_only=False)
            if pxm_model_cfg is None and "hyper_parameters" in ckpt_data:
                hp = ckpt_data["hyper_parameters"]
                if "config" in hp and hasattr(hp["config"], "model"):
                    pxm_model_cfg = copy.deepcopy(hp["config"].model)

        if pxm_model_cfg is None:
            pxm_model_cfg = copy.deepcopy(DEFAULT_PXM_MODEL_CFG)
        self.pxm_model_cfg = _to_easydict(pxm_model_cfg)

        # Build exact PocketXMol denoiser architecture and extract backbone modules
        # (decoders & confidence heads are discarded so no unused params exist)
        pxm_full = PMAsymDenoiser(
            self.pxm_model_cfg,
            num_node_types=NUM_PXM_NODE_TYPES,
            num_edge_types=NUM_PXM_EDGE_TYPES,
            pocket_in_dim=POCKET_IN_DIM,
        )
        self.pocket_embedder = pxm_full.pocket_embedder
        self.pocket_encoder = pxm_full.pocket_encoder
        self.nodetype_embedder = pxm_full.nodetype_embedder
        self.edgetype_embedder = pxm_full.edgetype_embedder
        self.denoiser = pxm_full.denoiser
        self.pocket_knn = int(
            getattr(getattr(self.pxm_model_cfg.denoiser, "context_cfg", {}), "knn", 32)
        )
        self.pocket_edge_dim = int(self.pxm_model_cfg.pocket.edge_dim)
        self.node_dim = int(self.pxm_model_cfg.node_dim)  # 320
        del pxm_full

        if load_pretrained and ckpt_data is not None:
            self._load_pxm_state_dict(ckpt_data)

        # Differentiable 3D polar shielding & transannular contact layer
        self.polar_shield = PolarShieldingTransannularLayer(
            node_dim=self.node_dim,
            hidden_dim=shield_hidden_dim,
        )

        # Atom-level physical feature fusion + differentiable fallback for soft guidance
        self.atom_phys_proj = nn.Sequential(
            nn.LayerNorm(self.atom_phys_dim),
            nn.Linear(self.atom_phys_dim, self.node_dim),
            nn.SiLU(),
        )
        # Fallback surrogate computes [node_probs(12), soft_degree(1), soft_arom_deg(1)] = 14 dims
        self.atom_phys_fallback_proj = nn.Sequential(
            nn.LayerNorm(NUM_PXM_NODE_TYPES + 2),
            nn.Linear(NUM_PXM_NODE_TYPES + 2, self.node_dim),
            nn.SiLU(),
        )
        self.node_fusion = nn.Sequential(
            nn.LayerNorm(self.node_dim),
            nn.Linear(self.node_dim, self.node_dim),
            nn.SiLU(),
            nn.Linear(self.node_dim, self.node_dim),
        )

        # Attention + Mean + Max graph pooling over heavy atoms -> conformer latent z_conf
        self.attn_gate = nn.Sequential(
            nn.Linear(self.node_dim, self.node_dim // 2),
            nn.SiLU(),
            nn.Linear(self.node_dim // 2, 1),
        )
        pool_in_dim = self.node_dim * 3 + self.polar_shield.global_geom_dim
        self.conf_proj = nn.Sequential(
            nn.LayerNorm(pool_in_dim),
            nn.Linear(pool_in_dim, self.conf_dim * 2),
            nn.SiLU(),
            nn.Linear(self.conf_dim * 2, self.conf_dim),
            nn.LayerNorm(self.conf_dim),
        )

        # Pseudo-energy head for Boltzmann conformer weighting: w_k = softmax_k(-E(z_k))
        self.energy_head = nn.Sequential(
            nn.Linear(self.conf_dim, self.conf_dim // 2),
            nn.SiLU(),
            nn.Linear(self.conf_dim // 2, 1),
        )

    def _load_pxm_state_dict(self, ckpt_data: Dict[str, Any]) -> None:
        raw_sd = ckpt_data.get("state_dict", ckpt_data)
        target_prefixes = (
            "pocket_embedder.",
            "pocket_encoder.",
            "nodetype_embedder.",
            "edgetype_embedder.",
            "denoiser.",
        )
        filtered: Dict[str, torch.Tensor] = {}
        for k, v in raw_sd.items():
            clean_k = k[6:] if k.startswith("model.") else k
            if clean_k.startswith(target_prefixes):
                filtered[clean_k] = v

        missing, unexpected = self.load_state_dict(filtered, strict=False)
        backbone_missing = [m for m in missing if m.startswith(target_prefixes)]
        if backbone_missing or unexpected:
            raise RuntimeError(
                f"Failed strict PXM backbone weight load: missing={backbone_missing[:10]}, unexpected={unexpected[:10]}"
            )

    def pxm_trunk_parameters(self) -> List[nn.Parameter]:
        """Parameters initialized from pocketxmol.ckpt (for differential LR)."""
        params: List[nn.Parameter] = []
        for mod in (
            self.pocket_embedder,
            self.pocket_encoder,
            self.nodetype_embedder,
            self.edgetype_embedder,
            self.denoiser,
        ):
            params.extend(list(mod.parameters()))
        return params

    def new_head_parameters(self) -> List[nn.Parameter]:
        """Newly added chameleonic encoder/pooling parameters."""
        params: List[nn.Parameter] = []
        for mod in (
            self.polar_shield,
            self.atom_phys_proj,
            self.atom_phys_fallback_proj,
            self.node_fusion,
            self.attn_gate,
            self.conf_proj,
            self.energy_head,
        ):
            params.extend(list(mod.parameters()))
        return params

    def encode_pocket(
        self,
        pocket_pos: torch.Tensor,
        pocket_x: torch.Tensor,
        batch_pocket: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode holo receptor pocket atoms via pretrained pocket_embedder + pocket_encoder."""
        h_ctx_in = self.pocket_embedder(pocket_x.float())
        edge_index_ctx = batched_knn_graph(
            pocket_pos, k=self.pocket_knn, batch=batch_pocket, flow="target_to_source"
        )
        n_ctx = int(pocket_pos.shape[0])
        e_ctx = int(edge_index_ctx.shape[1])
        h_edge_ctx = h_ctx_in.new_zeros((e_ctx, self.pocket_edge_dim))
        node_extra_ctx = h_ctx_in.new_zeros((n_ctx, 0))
        edge_extra_ctx = h_ctx_in.new_zeros((e_ctx, 0))

        out = self.pocket_encoder(
            h_node=h_ctx_in,
            pos_node=pocket_pos,
            h_edge=h_edge_ctx,
            edge_index=edge_index_ctx,
            node_extra=node_extra_ctx,
            edge_extra=edge_extra_ctx,
            batch_node=batch_pocket,
            h_ctx=None,
            pos_ctx=None,
            batch_ctx=None,
        )
        h_ctx = out[0] if isinstance(out, (tuple, list)) else out
        return h_ctx, pocket_pos, batch_pocket

    def _run_denoiser(
        self,
        h_node_in: torch.Tensor,
        pos: torch.Tensor,
        h_edge_in: torch.Tensor,
        edge_index: torch.Tensor,
        node_extra: torch.Tensor,
        edge_extra: torch.Tensor,
        batch_node: torch.Tensor,
        h_ctx: Optional[torch.Tensor],
        pos_ctx: Optional[torch.Tensor],
        batch_ctx: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.use_grad_checkpoint and self.training and torch.is_grad_enabled():
            # checkpoint handles None kwargs smoothly with use_reentrant=False
            def _fn(hn, pn, he, ei, ne, ee, bn, hc, pc, bc):
                return self.denoiser(
                    h_node=hn,
                    pos_node=pn,
                    h_edge=he,
                    edge_index=ei,
                    node_extra=ne,
                    edge_extra=ee,
                    batch_node=bn,
                    h_ctx=hc,
                    pos_ctx=pc,
                    batch_ctx=bc,
                )

            return grad_checkpoint(
                _fn,
                h_node_in,
                pos,
                h_edge_in,
                edge_index,
                node_extra,
                edge_extra,
                batch_node,
                h_ctx,
                pos_ctx,
                batch_ctx,
                use_reentrant=False,
            )

        return self.denoiser(
            h_node=h_node_in,
            pos_node=pos,
            h_edge=h_edge_in,
            edge_index=edge_index,
            node_extra=node_extra,
            edge_extra=edge_extra,
            batch_node=batch_node,
            h_ctx=h_ctx,
            pos_ctx=pos_ctx,
            batch_ctx=batch_ctx,
        )

    def _run_denoiser_chunked(
        self,
        h_node_in: torch.Tensor,
        pos: torch.Tensor,
        halfedge_index: torch.Tensor,
        h_halfedge_in: torch.Tensor,
        node_extra: torch.Tensor,
        halfedge_extra: torch.Tensor,
        batch_node: torch.Tensor,
        halfedge_batch: Optional[torch.Tensor],
        num_graphs: int,
        h_ctx: Optional[torch.Tensor],
        pos_ctx: Optional[torch.Tensor],
        batch_ctx: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Run denoiser in edge-budgeted conformer-graph chunks so 6-block dense edge activations fit in VRAM."""
        chunk_g = int(self.max_graphs_per_chunk)
        chunk_e = int(getattr(self, "max_edges_per_chunk", 0))
        total_dir_edges = int(halfedge_index.shape[1]) * 2
        no_graph_limit = chunk_g <= 0 or num_graphs <= chunk_g
        no_edge_limit = chunk_e <= 0 or total_dir_edges <= chunk_e
        if num_graphs <= 1 or halfedge_batch is None or (no_graph_limit and no_edge_limit):
            edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)
        elif batch_node.numel() > 1 and not bool((batch_node[1:] >= batch_node[:-1]).all().item()):
            edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)
        else:
            edge_index = None
        if edge_index is not None:
            h_edge_in = torch.cat([h_halfedge_in, h_halfedge_in], dim=0)
            edge_extra = torch.cat([halfedge_extra, halfedge_extra], dim=0)
            return self._run_denoiser(
                h_node_in=h_node_in,
                pos=pos,
                h_edge_in=h_edge_in,
                edge_index=edge_index,
                node_extra=node_extra,
                edge_extra=edge_extra,
                batch_node=batch_node,
                h_ctx=h_ctx,
                pos_ctx=pos_ctx,
                batch_ctx=batch_ctx,
            )

        counts = torch.bincount(batch_node, minlength=num_graphs)
        node_starts = torch.zeros((num_graphs + 1,), dtype=torch.long, device=pos.device)
        node_starts[1:] = torch.cumsum(counts, dim=0)
        he_counts = torch.bincount(halfedge_batch, minlength=num_graphs).tolist()

        h_out_chunks: List[torch.Tensor] = []
        p_out_chunks: List[torch.Tensor] = []

        g0 = 0
        while g0 < num_graphs:
            g1 = g0 + 1
            acc_dir_edges = 2 * int(he_counts[g0])
            while g1 < num_graphs:
                if chunk_g > 0 and (g1 - g0) >= chunk_g:
                    break
                next_dir_edges = 2 * int(he_counts[g1])
                if chunk_e > 0 and (acc_dir_edges + next_dir_edges) > chunk_e:
                    break
                acc_dir_edges += next_dir_edges
                g1 += 1

            n0 = int(node_starts[g0].item())
            n1 = int(node_starts[g1].item())
            if n1 <= n0:
                g0 = g1
                continue

            he_mask = (halfedge_batch >= g0) & (halfedge_batch < g1)
            he_c = halfedge_index[:, he_mask] - n0
            edge_index_c = torch.cat([he_c, he_c.flip(0)], dim=1)
            h_he_c = h_halfedge_in[he_mask]
            h_edge_c = torch.cat([h_he_c, h_he_c], dim=0)
            ee_c = halfedge_extra[he_mask]
            edge_extra_c = torch.cat([ee_c, ee_c], dim=0)
            bn_c = batch_node[n0:n1] - g0

            if h_ctx is not None and pos_ctx is not None and batch_ctx is not None:
                ctx_mask = (batch_ctx >= g0) & (batch_ctx < g1)
                h_ctx_c: Optional[torch.Tensor] = h_ctx[ctx_mask]
                pos_ctx_c: Optional[torch.Tensor] = pos_ctx[ctx_mask]
                batch_ctx_c: Optional[torch.Tensor] = batch_ctx[ctx_mask] - g0
            else:
                h_ctx_c, pos_ctx_c, batch_ctx_c = None, None, None

            hn_c, po_c, _ = self._run_denoiser(
                h_node_in=h_node_in[n0:n1],
                pos=pos[n0:n1],
                h_edge_in=h_edge_c,
                edge_index=edge_index_c,
                node_extra=node_extra[n0:n1],
                edge_extra=edge_extra_c,
                batch_node=bn_c,
                h_ctx=h_ctx_c,
                pos_ctx=pos_ctx_c,
                batch_ctx=batch_ctx_c,
            )
            h_out_chunks.append(hn_c)
            p_out_chunks.append(po_c)
            g0 = g1

        return torch.cat(h_out_chunks, dim=0), torch.cat(p_out_chunks, dim=0), None

    def _compute_fallback_atom_phys(
        self,
        node_probs: torch.Tensor,
        edge_probs: torch.Tensor,
        halfedge_index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """Differentiable topological surrogate [node_probs(12), deg(1), arom_deg(1)] = 14 dims."""
        cov_w = edge_probs[:, 1:5].sum(dim=-1)
        arom_w = edge_probs[:, 4]
        if halfedge_index.numel() > 0:
            u, v = halfedge_index[0], halfedge_index[1]
            edge_pair = torch.stack([cov_w, arom_w], dim=-1)
            deg_arom = scatter_sum(edge_pair, u, dim=0, dim_size=num_nodes) + scatter_sum(
                edge_pair, v, dim=0, dim_size=num_nodes
            )
            deg_arom = deg_arom * 0.25
        else:
            deg_arom = node_probs.new_zeros((num_nodes, 2))
        return torch.cat([node_probs, deg_arom], dim=-1)

    def encode_conformers(
        self,
        pos: torch.Tensor,
        halfedge_index: torch.Tensor,
        batch_node: torch.Tensor,
        node_type: Optional[torch.Tensor] = None,
        element: Optional[torch.Tensor] = None,
        node_logits: Optional[torch.Tensor] = None,
        halfedge_type: Optional[torch.Tensor] = None,
        halfedge_logits: Optional[torch.Tensor] = None,
        halfedge_batch: Optional[torch.Tensor] = None,
        atom_phys_feat: Optional[torch.Tensor] = None,
        is_peptide: Optional[torch.Tensor] = None,
        fixed_node: Optional[torch.Tensor] = None,
        fixed_pos: Optional[torch.Tensor] = None,
        fixed_halfedge: Optional[torch.Tensor] = None,
        fixed_halfdist: Optional[torch.Tensor] = None,
        pocket_pos: Optional[torch.Tensor] = None,
        pocket_x: Optional[torch.Tensor] = None,
        batch_pocket: Optional[torch.Tensor] = None,
        h_ctx: Optional[torch.Tensor] = None,
        pos_ctx: Optional[torch.Tensor] = None,
        batch_ctx: Optional[torch.Tensor] = None,
        num_graphs: Optional[int] = None,
        max_nodes_per_graph: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Encode G conformer graphs into conformer embeddings z_conf [G, conf_dim]."""
        device = pos.device
        dtype = pos.dtype
        n_total = int(pos.shape[0])
        if num_graphs is None:
            num_graphs = int(batch_node.max().item()) + 1 if batch_node.numel() > 0 else 0
        if halfedge_batch is None and num_graphs > 1 and halfedge_index.numel() > 0:
            halfedge_batch = batch_node[halfedge_index[0]]

        # 1. Node embedding & differentiable polar probabilities (N, O -> indices 1, 2)
        if node_logits is not None:
            node_probs = F.softmax(node_logits, dim=-1)
            emb_node = node_probs @ self.nodetype_embedder.weight
            polar_probs = node_probs[:, 1] + node_probs[:, 2]
        else:
            if node_type is None:
                if element is None:
                    raise ValueError("Must supply node_logits, node_type, or element.")
                node_type = atomic_numbers_to_pxm_idx(element)
            emb_node = self.nodetype_embedder(node_type.long())
            node_probs = F.one_hot(node_type.long(), num_classes=NUM_PXM_NODE_TYPES).to(dtype)
            polar_probs = ((node_type == 1) | (node_type == 2)).to(dtype)

        # 2. Halfedge embedding & differentiable covalent probabilities
        e_half = int(halfedge_index.shape[1])
        if halfedge_logits is not None:
            edge_probs = F.softmax(halfedge_logits, dim=-1)
            emb_halfedge = edge_probs @ self.edgetype_embedder.weight
            covalent_probs = edge_probs[:, 1:5].sum(dim=-1)
        else:
            if halfedge_type is None:
                raise ValueError("Must supply halfedge_logits or halfedge_type.")
            emb_halfedge = self.edgetype_embedder(halfedge_type.long())
            edge_probs = F.one_hot(halfedge_type.long(), num_classes=NUM_PXM_EDGE_TYPES).to(dtype)
            covalent_probs = ((halfedge_type >= 1) & (halfedge_type <= 4)).to(dtype)

        # 3. Construct exact PMAsymDenoiser node/edge inputs
        if is_peptide is None:
            is_pep_col = torch.ones((n_total, 1), dtype=dtype, device=device)
        else:
            is_pep_col = is_peptide.to(dtype=dtype, device=device).view(n_total, 1)

        fn = torch.zeros((n_total,), dtype=dtype, device=device) if fixed_node is None else fixed_node.to(dtype)
        fp = torch.zeros((n_total,), dtype=dtype, device=device) if fixed_pos is None else fixed_pos.to(dtype)
        node_extra = torch.stack([fn, fp], dim=1)  # [N, 2]
        h_node_in = torch.cat([emb_node, fn.unsqueeze(1), fp.unsqueeze(1), is_pep_col], dim=-1)  # [N, 320]

        fhe = torch.zeros((e_half,), dtype=dtype, device=device) if fixed_halfedge is None else fixed_halfedge.to(dtype)
        fhd = torch.zeros((e_half,), dtype=dtype, device=device) if fixed_halfdist is None else fixed_halfdist.to(dtype)
        halfedge_extra = torch.stack([fhe, fhd], dim=1)  # [E_half, 2]
        h_halfedge_in = torch.cat([emb_halfedge, halfedge_extra], dim=-1)  # [E_half, 96]

        # 4. Optional holo receptor pocket context (h_ctx=None skips context blocks safely)
        if h_ctx is None and pocket_pos is not None and pocket_x is not None:
            if batch_pocket is None:
                batch_pocket = torch.zeros((pocket_pos.shape[0],), dtype=torch.long, device=device)
            h_ctx, pos_ctx, batch_ctx = self.encode_pocket(pocket_pos, pocket_x, batch_pocket)

        # 5. Run pretrained/initialized PocketXMol denoiser (chunked over conformer graphs to bound VRAM)
        h_node_out, pos_out, _ = self._run_denoiser_chunked(
            h_node_in=h_node_in,
            pos=pos,
            halfedge_index=halfedge_index,
            h_halfedge_in=h_halfedge_in,
            node_extra=node_extra,
            halfedge_extra=halfedge_extra,
            batch_node=batch_node,
            halfedge_batch=halfedge_batch,
            num_graphs=num_graphs,
            h_ctx=h_ctx,
            pos_ctx=pos_ctx,
            batch_ctx=batch_ctx,
        )

        # 6. Differentiable 3D Polar Shielding & Transannular Contact Layer
        shield_emb, global_geom, shield_metrics = self.polar_shield(
            pos=pos,
            polar_probs=polar_probs,
            covalent_probs=covalent_probs,
            halfedge_index=halfedge_index,
            batch_node=batch_node,
            halfedge_batch=halfedge_batch,
            num_graphs=num_graphs,
            max_nodes_per_graph=max_nodes_per_graph,
        )

        # 7. Atom physical features (with dropout to fallback surrogate for Phase 3 readiness)
        if atom_phys_feat is not None:
            phys_emb = self.atom_phys_proj(atom_phys_feat.to(dtype=dtype))
            if self.training and self.phys_dropout_prob > 0.0:
                fallback_feats = self._compute_fallback_atom_phys(
                    node_probs=node_probs,
                    edge_probs=edge_probs,
                    halfedge_index=halfedge_index,
                    num_nodes=n_total,
                )
                fallback_emb = self.atom_phys_fallback_proj(fallback_feats)
                drop_graph = (torch.rand((num_graphs, 1), device=device) < self.phys_dropout_prob).to(dtype)
                drop_atom = drop_graph[batch_node]
                phys_emb = phys_emb * (1.0 - drop_atom) + fallback_emb * drop_atom
        else:
            fallback_feats = self._compute_fallback_atom_phys(
                node_probs=node_probs,
                edge_probs=edge_probs,
                halfedge_index=halfedge_index,
                num_nodes=n_total,
            )
            phys_emb = self.atom_phys_fallback_proj(fallback_feats)

        h_fused = h_node_out + self.node_fusion(h_node_out + shield_emb + phys_emb)

        # 8. Attention + Mean + Max atom pooling -> conformer latent z_conf [G, conf_dim]
        attn_logits = self.attn_gate(h_fused).squeeze(-1)
        delta_pos = pos_out - pos
        atom_force_norm = torch.sqrt((delta_pos * delta_pos).sum(dim=-1) + 1e-8)
        if num_graphs == 1 and n_total > 0:
            attn_weights = F.softmax(attn_logits, dim=0).unsqueeze(-1)
            z_attn = (h_fused * attn_weights).sum(dim=0, keepdim=True)
            z_mean = h_fused.mean(dim=0, keepdim=True)
            z_max = h_fused.amax(dim=0, keepdim=True)
            mean_force_norm = atom_force_norm.mean().unsqueeze(0)
        else:
            attn_weights = pyg_softmax(attn_logits, batch_node, num_nodes=num_graphs).unsqueeze(-1)
            z_attn = scatter_sum(h_fused * attn_weights, batch_node, dim=0, dim_size=num_graphs)
            z_mean = scatter_mean(h_fused, batch_node, dim=0, dim_size=num_graphs)
            z_max = scatter_max(h_fused, batch_node, dim=0, dim_size=num_graphs)[0]
            mean_force_norm = scatter_mean(atom_force_norm, batch_node, dim=0, dim_size=num_graphs)

        z_conf_in = torch.cat([z_attn, z_mean, z_max, global_geom], dim=-1)
        z_conf = self.conf_proj(z_conf_in)

        # Measure denoiser local strain magnitude as auxiliary diagnostics
        shield_metrics["pxm_force_norm"] = mean_force_norm

        return z_conf, shield_metrics

    def pool_ensemble(
        self,
        z_conf: torch.Tensor,
        conf_to_mol: Optional[torch.Tensor] = None,
        num_mols: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pool conformer embeddings z_conf [G, D] -> molecule embeddings via Boltzmann + plasticity.

        Args:
            z_conf: [G, D] conformer representations.
            conf_to_mol: [G] mapping each conformer to molecule 0..M-1. If None, assumes K=1.
            num_mols: M.

        Returns:
            z_boltz: [M, D] Boltzmann weighted average across conformers.
            z_disp: [M, D] conformational plasticity dispersion (max_k z_k - min_k z_k).
            weights: [G] Boltzmann weights w_k = softmax_k(-E(z_k)).
            energies: [G] learned pseudo-energies E(z_k).
        """
        g_count = int(z_conf.shape[0])
        if conf_to_mol is None:
            energies = self.energy_head(z_conf).squeeze(-1)  # [G]
            weights = z_conf.new_ones((g_count,))
            z_boltz = z_conf
            z_disp = torch.zeros_like(z_conf)
            return z_boltz, z_disp, weights, energies
        if num_mols is None:
            num_mols = int(conf_to_mol.max().item()) + 1 if conf_to_mol.numel() > 0 else 0

        energies = self.energy_head(z_conf).squeeze(-1)  # [G]
        if g_count == 1 and num_mols == 1:
            weights = z_conf.new_ones((1,))
            z_boltz = z_conf
            z_disp = torch.zeros_like(z_conf)
            return z_boltz, z_disp, weights, energies

        weights = pyg_softmax(-energies, conf_to_mol, num_nodes=num_mols)  # [G]

        z_boltz = scatter_sum(z_conf * weights.unsqueeze(-1), conf_to_mol, dim=0, dim_size=num_mols)
        z_max = scatter_max(z_conf, conf_to_mol, dim=0, dim_size=num_mols)[0]
        z_min = scatter_min(z_conf, conf_to_mol, dim=0, dim_size=num_mols)[0]
        z_disp = z_max - z_min  # Zero identically when K=1

        return z_boltz, z_disp, weights, energies

    def forward(
        self,
        pos: torch.Tensor,
        halfedge_index: torch.Tensor,
        batch_node: torch.Tensor,
        conf_to_mol: Optional[torch.Tensor] = None,
        num_mols: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        z_conf, shield_metrics = self.encode_conformers(
            pos=pos,
            halfedge_index=halfedge_index,
            batch_node=batch_node,
            **kwargs,
        )
        z_boltz, z_disp, weights, energies = self.pool_ensemble(
            z_conf=z_conf,
            conf_to_mol=conf_to_mol,
            num_mols=num_mols,
        )
        return {
            "z_conf": z_conf,
            "z_boltz": z_boltz,
            "z_disp": z_disp,
            "weights": weights,
            "energies": energies,
            "shield_metrics": shield_metrics,
        }
