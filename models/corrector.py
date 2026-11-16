import math

import torch
import torch.nn as nn
import torch.linalg as la
from torch.nn import functional as F
from torch_scatter import scatter_mean, scatter_sum

try:
    from utils.motion import apply_torsional_rotation_multiple_domains
except:
    pass


def grad_len_to_pos(pos, edge_index, config):
    pos_left = pos[edge_index[0]]
    pos_right = pos[edge_index[1]]
    value_min, value_max = config['min'], config['max']
    value_std = config['std']
    lens = torch.linalg.norm(pos_left - pos_right, dim=-1).unsqueeze(-1)
    grad_min = 2 * (lens - value_min) / value_std**2 / lens * (pos_left - pos_right)
    grad_max = 2 * (lens - value_max) / value_std**2 / lens * (pos_left - pos_right)
    grad = torch.where(lens < value_min, grad_min, 
                       torch.where(lens > value_max, grad_max, 0))
    grad_pos = scatter_mean(grad, edge_index[0], dim=0, dim_size=pos.shape[0])
    return grad_pos
    
    
def correct_pos_by_fixed_dist_batch(batch, outputs, use_pos='in', config=None):
    if use_pos == 'in':
        pos_in = batch['pos_in'].detach().clone()
        fixed_halfdist = batch['fixed_halfdist'].bool()
    elif use_pos == 'gt':
        pos_in = batch['gt_node_pos'].detach().clone()
        fixed_halfdist = batch['fixed_halfdist_flex'].bool()
    pred_pos = outputs['pred_pos'].detach().clone()
    
    # get parameter
    if config is None:
        iters = 10
        lr = 0.5
        # lamb = 0
    else:
        iters = config['iters']
        lr = config['lr']

    # get bond index
    halfedge_index = batch['halfedge_index']
    edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=-1)
    fixed_dist = torch.cat([fixed_halfdist, fixed_halfdist], dim=-1)
    fixed_edge_index = edge_index[:, fixed_dist]
    
    # fetch relevant pos
    fixed_pos = batch['fixed_pos'].bool()
    
    # spring the pos
    with torch.enable_grad():
        dist_in = torch.linalg.norm(pos_in[fixed_edge_index[0]] - pos_in[fixed_edge_index[1]], dim=-1)
        for i in range(iters):
            dist_pred = torch.linalg.norm(pred_pos[fixed_edge_index[0]] - pred_pos[fixed_edge_index[1]], dim=-1)
            grad = 2 * ((dist_pred - dist_in) / dist_in).unsqueeze(-1) * (pred_pos[fixed_edge_index[0]] - pred_pos[fixed_edge_index[1]])
            grad_pos = scatter_mean(grad, fixed_edge_index[0], dim=0, dim_size=pos_in.shape[0])
            # grad = torch.autograd.grad(loss, in_pos)[0]
            pred_pos[~fixed_pos] = pred_pos[~fixed_pos] - grad_pos[~fixed_pos] * lr

            loss = F.mse_loss(torch.linalg.norm(pred_pos[fixed_edge_index[0]] - pred_pos[fixed_edge_index[1]], dim=-1), dist_in)
            if loss.isnan():
                pred_pos = outputs['pred_pos'].detach().clone()
                break
    # print(loss)
    return pred_pos


@torch.no_grad()
def correct_pos_batch_no_tor(batch, outputs):
    pos_in = batch['pos_in']
    pred_pos = outputs['pred_pos']
    
    pred_pos_corr = correct_pos(
        no_tor=True,
        pos_in=pos_in.clone(), pos_out=pred_pos.clone(),
        sin_in=None, cos_in=None,
        sin_out=None, cos_out=None,
        domain_node_index=batch['domain_node_index'],
        # domain_center_nodes=batch['domain_center_nodes'],
        tor_bonds_anno=None,
        twisted_nodes_anno=None,
        dihedral_pairs_anno=None,
    )
    return pred_pos_corr

@torch.no_grad()
def correct_pos_batch(batch, outputs, use_pos='in'):
    if use_pos == 'in':
        pos_in = batch['pos_in']
    elif use_pos == 'gt':
        pos_in = batch['gt_node_pos']
    pred_pos = outputs['pred_pos']
    
    if 'dih_sin' not in outputs:
        sin_out, cos_out = get_dihedral_batch(pred_pos, batch['tor_bonds_anno'], batch['dihedral_pairs_anno'])
    else:
        sin_out, cos_out = outputs['dih_sin'], outputs['dih_cos']
    sin_in, cos_in = get_dihedral_batch(pos_in, batch['tor_bonds_anno'], batch['dihedral_pairs_anno'])
    pred_pos_corr = correct_pos(
        pos_in=pos_in.clone(), pos_out=pred_pos.clone(),
        sin_in=sin_in, cos_in=cos_in,
        sin_out=sin_out,
        cos_out=cos_out,
        domain_node_index=batch['domain_node_index'],
        # domain_center_nodes=batch['domain_center_nodes'],
        tor_bonds_anno=batch['tor_bonds_anno'],
        twisted_nodes_anno=batch['twisted_nodes_anno'],
        dihedral_pairs_anno=batch['dihedral_pairs_anno'],
    )
    return pred_pos_corr


@torch.no_grad()
def correct_pos(pos_in, pos_out, 
                sin_in, cos_in,
                sin_out, cos_out,
                
                domain_node_index,
                tor_bonds_anno,
                twisted_nodes_anno, dihedral_pairs_anno,
                no_tor=False):
    """
    Input:
        pos_in/out: (N, 3)
        domain_node_index: (2, N)
        tor_bonds_anno: (n_tor, 3)
        twisted_nodes_anno: (n_twisted, 2)
        dihedral_pairs_anno: (n_dih, 3)
    """
    
    if not no_tor:
        # # get dihedral torsion
        index_tor = dihedral_pairs_anno[:, 0]
        # tor = out - in, (n_dih, 2)
        cos_tor = cos_out * cos_in + sin_out * sin_in
        sin_tor = sin_out * cos_in - cos_out * sin_in
        angles_tri = torch.cat([sin_tor, cos_tor], dim=1)  # (n_dih, 2)
        angles_tri = scatter_mean(angles_tri, index_tor, dim=0)
        angles_tri = F.normalize(angles_tri, dim=-1)
        
        # # apply torsion to pos_in
        tor_order = tor_bonds_anno[:, 0]
        tor_bonds = tor_bonds_anno[:, 1:]
        index_tor_twisted = twisted_nodes_anno[:, 0]
        twisted_nodes = twisted_nodes_anno[:, 1]
        pos_tor = apply_torsional_rotation_multiple_domains(pos_in, 
                                tor_order, tor_bonds, angles_tri,
                                twisted_nodes, index_tor_twisted)
    else:
        pos_tor = pos_in.clone()
        
    # # global rotation and translation (use center_nodes). not around center
    domain_index, node_index = domain_node_index
    # pos_center_tor = pos_tor[domain_center_nodes]
    # pos_center_out = pos_out[domain_center_nodes]
    # global_rot, global_trans = kabsch_batch(pos_center_tor, pos_center_out)
    global_rot, global_trans = kabsch_flatten(pos_tor[node_index], pos_out[node_index], domain_index)
    
    # # apply global rotation and translation
    pos_corrected_expand = torch.matmul(
        pos_tor[node_index, None, :],
        global_rot.transpose(1, 2)[domain_index]
    ) + global_trans[domain_index]

    pos_corrected = pos_out.clone()
    pos_corrected[node_index] = pos_corrected_expand.squeeze(1)
    return pos_corrected
    



def get_dihedral_batch(pos, tor_bonds_anno, dihedral_pairs_anno):
    index_tor = dihedral_pairs_anno[:, 0]
    dihedral_ends = dihedral_pairs_anno[:, 1:]  # (n_dih, 2)
    dihedral_tor_nodes = tor_bonds_anno[:, 1:][index_tor]  # (n_dih, 2)

    sin, cos = get_dihedral(pos[dihedral_ends[:, 0]], pos[dihedral_tor_nodes[:, 0]],
                    pos[dihedral_tor_nodes[:, 1]], pos[dihedral_ends[:, 1]])
    return sin, cos


def get_dihedral(p0, p1, p2, p3):
    """from https://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python
    Praxeolitic formula
    1 sqrt, 1 cross product"""
    b0 = p0 - p1
    # b1 = p2 - p1
    b1 = p1 - p2
    b2 = p3 - p2

    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 = F.normalize(b1, dim=-1)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - (b0 * b1).sum(-1, keepdim=True) * b1
    w = b2 - (b2 * b1).sum(-1, keepdim=True) * b1
    v = F.normalize(v, dim=-1)
    w = F.normalize(w, dim=-1)

    # angle between v and w in a plane is the torsion angle
    # v and w is normalized
    cos = (v * w).sum(-1, keepdim=True)
    # sin = (torch.cross(v, w) * b1).sum()
    sin = (torch.cross(w, v) * b1).sum(-1, keepdim=True)
    return sin, cos


def kabsch_flatten(X, Y, domain_index):
    """
    Align X to Y using rigid transformation
    see https://en.wikipedia.org/wiki/Kabsch_algorithm
    X, Y: (N, 3)
    domain_index: (N,)
    """
    # Normalize the data by centering at the origin
    n_domain = domain_index.max() + 1
    X_mean = scatter_mean(X, domain_index, dim=0) # (n_domain, 3)
    Y_mean = scatter_mean(Y, domain_index, dim=0) # (n_domain, 3)
    X_centered = (X - X_mean[domain_index])  # (N, 3)
    Y_centered = (Y - Y_mean[domain_index])  # (N, 3)
    
    # Compute the covariance matrix
    covariance_matrix = torch.zeros([n_domain, 3, 3], device=X.device, dtype=X.dtype)  # (n_domain, 3, 3)
    for i_domain in range(n_domain):
        this_domain = (domain_index == i_domain)
        covariance_matrix[i_domain] = torch.matmul(X_centered[this_domain].transpose(0, 1),
                                                   Y_centered[this_domain])  # (3, 3)
    
    # Perform Singular Value Decomposition, use float64 to avoid numerical issue
    U, _, Vt = torch.linalg.svd(covariance_matrix.to(torch.float32)) # (n_domain, 3, 3)
    d = torch.sign(torch.det(torch.matmul(U, Vt).to(torch.float32)))  # (n_domain,)

    # Compute the rotation matrix
    diag_mat = torch.eye(3, device=X.device, dtype=d.dtype).unsqueeze(0).repeat(n_domain, 1, 1)
    diag_mat[:, -1, -1] = d # (n_domain, 3, 3)
    R = torch.matmul(torch.matmul(Vt.transpose(1, 2), diag_mat), U.transpose(1, 2)).to(X.dtype) # (n_domain, 3, 3)
    
    
    # Compute the translation vector
    # translation = Y_mean - X_mean
    # translation = Y_mean - torch.matmul(R, X_mean)  # (N, 1, 3)
    translation = Y_mean.unsqueeze(1) - torch.matmul(X_mean.unsqueeze(1), R.transpose(1, 2))  # (N, 1, 3)
    
    return R, translation


def kabsch_batch(X, Y):
    """
    Align X to Y using rigid transformation
    see https://en.wikipedia.org/wiki/Kabsch_algorithm
    X, Y: (B, N, 3)
    """
    # Normalize the data by centering at the origin
    X_mean = torch.mean(X, dim=1, keepdim=True) # (B, 1, 3)
    Y_mean = torch.mean(Y, dim=1, keepdim=True) # (B, 1, 3)
    X_centered = (X - X_mean)  # (B, N, 3)
    Y_centered = (Y - Y_mean)  # (B, 3, N)
    
    # Compute the covariance matrix
    covariance_matrix = torch.matmul(X_centered.transpose(1, 2), Y_centered)  # (B, 3, 3)
    
    # Perform Singular Value Decomposition, use float64 to avoid numerical issue
    U, _, Vt = torch.linalg.svd(covariance_matrix.to(torch.float32)) # (B, 3, 3)
    d = torch.sign(torch.det(torch.matmul(U, Vt).to(torch.float32)))

    # Compute the rotation matrix
    diag_mat = torch.eye(3, device=X.device, dtype=d.dtype).unsqueeze(0).repeat(X.shape[0], 1, 1)
    diag_mat[:, -1, -1] = d
    R = torch.matmul(torch.matmul(Vt.transpose(1, 2), diag_mat), U.transpose(1, 2)).to(X.dtype) # (B, 3, 3)
    
    
    # Compute the translation vector
    # translation = Y_mean - X_mean
    # translation = Y_mean - torch.matmul(R, X_mean)  # (N, 1, 3)
    translation = Y_mean - torch.matmul(X_mean, R.transpose(1, 2))  # (N, 1, 3)
    
    return R, translation


def procrustes_analysis_batch(X, Y):
    """
    Align X to Y using rigid transformation
    see https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
    and https://en.wikipedia.org/wiki/Procrustes_analysis
    X, Y: (B, N, 3)
    """
    # Normalize the data by centering at the origin
    X_mean = torch.mean(X, dim=1, keepdim=True) # (B, 1, 3)
    Y_mean = torch.mean(Y, dim=1, keepdim=True) # (B, 1, 3)
    X_centered_data_T = (X - X_mean)  # (B, N, 3)
    Y_centered_data = (Y - Y_mean).transpose(1, 2)  # (B, 3, N)
    
    # Compute the covariance matrix
    covariance_matrix = torch.matmul(Y_centered_data, X_centered_data_T)  # (B, 3, 3)
    
    # Perform Singular Value Decomposition, use float64 to avoid numerical issue
    U, _, Vt = torch.linalg.svd(covariance_matrix.to(torch.float64)) # (B, 3, 3)
    
    # Compute the rotation matrix
    R = torch.matmul(U, Vt).to(X.dtype) # (B, 3, 3)
    
    # Compute the translation vector
    # translation = Y_mean - X_mean
    # translation = Y_mean - torch.matmul(R, X_mean)  # (N, 1, 3)
    translation = Y_mean - torch.matmul(X_mean, R.transpose(1, 2))  # (N, 1, 3)
    
    return R, translation


def procrustes_analysis_one_sample(X, Y):
    """
    Align X to Y using rigid transformation
    see https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
    and https://en.wikipedia.org/wiki/Procrustes_analysis
    X, Y: (N, 3)
    """
    # Normalize the data by centering at the origin
    X_mean = torch.mean(X, dim=0) # (3,)
    Y_mean = torch.mean(Y, dim=0) # (3,)
    X_centered_data = (X - X_mean).T  # (3, N)
    Y_centered_data = (Y - Y_mean).T  # (3, N)
    
    # Compute the covariance matrix
    covariance_matrix = torch.matmul(Y_centered_data, X_centered_data.T)  # (3, 3)
    
    # Perform Singular Value Decomposition
    U, _, Vt = la.svd(covariance_matrix) # (3, 3)
    
    # Compute the rotation matrix
    R = torch.matmul(U, Vt) # (3, 3)
    
    # Compute the translation vector
    # translation = Y_mean - X_mean
    translation = Y_mean - torch.matmul(R, X_mean)  # (3,)
    
    return R, translation


def _safe_unit_vec(vec, eps=1e-4):
    norm = torch.sqrt((vec * vec).sum(dim=-1, keepdim=True) + eps * eps)
    return vec / norm


def _safe_dihedral_sin_cos(p0, p1, p2, p3, eps=1e-4):
    b0 = p0 - p1
    b1 = _safe_unit_vec(p1 - p2, eps=eps)
    b2 = p3 - p2
    v = b0 - (b0 * b1).sum(dim=-1, keepdim=True) * b1
    w = b2 - (b2 * b1).sum(dim=-1, keepdim=True) * b1
    v = _safe_unit_vec(v, eps=eps)
    w = _safe_unit_vec(w, eps=eps)
    cos = (v * w).sum(dim=-1, keepdim=True)
    sin = (torch.linalg.cross(w, v, dim=-1) * b1).sum(dim=-1, keepdim=True)
    return sin, cos


# Node-type mapping matching FeaturizeMol: 0:C, 1:N, 2:O, 3:F, 4:P, 5:S, 6:Cl, 7:B, 8:Br, 9:I, 10:Se
_SINGLE_BOND_LEN = {
    (0, 0): 1.52, (0, 1): 1.46, (0, 2): 1.43, (0, 3): 1.35,
    (0, 4): 1.84, (0, 5): 1.81, (0, 6): 1.74, (0, 7): 1.56,
    (0, 8): 1.90, (0, 9): 2.09, (0, 10): 1.95,
    (1, 1): 1.45, (1, 2): 1.40, (1, 5): 1.65,
    (2, 4): 1.60, (2, 5): 1.58, (5, 5): 2.04,
}
_DOUBLE_BOND_LEN = {
    (0, 0): 1.34, (0, 1): 1.29, (0, 2): 1.23, (0, 5): 1.62,
    (2, 4): 1.48, (2, 5): 1.44,
}


def _ideal_bond_length(zu, zv, bt, is_amide_carbonyl_c=False):
    pair = (zu, zv) if zu <= zv else (zv, zu)
    if bt == 4:
        return 1.39
    if bt == 3:
        return 1.20
    if bt == 2:
        return _DOUBLE_BOND_LEN.get(pair, 1.34)
    if pair == (0, 1) and is_amide_carbonyl_c:
        return 1.34
    return _SINGLE_BOND_LEN.get(pair, 1.50)


def _extract_node_index_set(batch, keys, n_nodes):
    nodes = set()
    for key in keys:
        if key not in batch or batch[key] is None:
            continue
        t = batch[key]
        if t.dtype == torch.bool:
            vals = torch.nonzero(t, as_tuple=False).view(-1).detach().cpu().tolist()
        else:
            vals = t.detach().cpu().tolist()
        for idx in vals:
            idx_i = int(idx)
            if 0 <= idx_i < n_nodes:
                nodes.add(idx_i)
    return nodes


def _build_covalent_graph(batch, n_nodes):
    adj = [set() for _ in range(n_nodes)]
    bond_bt = {}
    he_idx = batch.get('halfedge_index', None)
    he_type = batch.get('halfedge_type', None)
    if he_idx is not None and he_type is not None and he_idx.numel() > 0:
        cov_mask = (he_type > 0) & (he_type <= 4)
        if cov_mask.any():
            u_list = he_idx[0, cov_mask].detach().cpu().tolist()
            v_list = he_idx[1, cov_mask].detach().cpu().tolist()
            bt_list = he_type[cov_mask].detach().cpu().tolist()
            for u, v, bt in zip(u_list, v_list, bt_list):
                if 0 <= u < n_nodes and 0 <= v < n_nodes and u != v:
                    adj[u].add(v)
                    adj[v].add(u)
                    bond_bt[(min(u, v), max(u, v))] = int(bt)
    elif 'bond_index' in batch and batch['bond_index'] is not None and batch['bond_index'].numel() > 0:
        bi = batch['bond_index']
        bt_arr = batch.get('bond_type', None)
        u_list = bi[0].detach().cpu().tolist()
        v_list = bi[1].detach().cpu().tolist()
        bt_list = bt_arr.detach().cpu().tolist() if bt_arr is not None else [1] * len(u_list)
        for u, v, bt in zip(u_list, v_list, bt_list):
            if 0 <= u < n_nodes and 0 <= v < n_nodes and u != v:
                adj[u].add(v)
                adj[v].add(u)
                bond_bt[(min(u, v), max(u, v))] = int(bt)

    close_bonds = batch.get('ring_closure_bonds', None)
    close_lens = batch.get('ring_closure_target_len', None)
    closure_pairs = []
    if close_bonds is not None and close_bonds.numel() > 0:
        cb_cpu = close_bonds.detach().cpu().tolist()
        cl_cpu = (
            close_lens.detach().cpu().tolist()
            if (close_lens is not None and len(close_lens) == len(cb_cpu))
            else [1.45] * len(cb_cpu)
        )
        for (cu, cv), tl in zip(cb_cpu, cl_cpu):
            cu_i, cv_i = int(cu), int(cv)
            if 0 <= cu_i < n_nodes and 0 <= cv_i < n_nodes and cu_i != cv_i:
                adj[cu_i].add(cv_i)
                adj[cv_i].add(cu_i)
                closure_pairs.append((cu_i, cv_i, float(tl)))
    return [sorted(s) for s in adj], bond_bt, closure_pairs


def _reflect_wrong_chiral_sidechains(pos_work, batch, adj, bb_nodes, fixed_cpu):
    ca_triples = batch.get('ca_anchor_triples', None)
    chiral_prompt = batch.get('chiral_prompt', None)
    if ca_triples is None or ca_triples.numel() == 0 or chiral_prompt is None or chiral_prompt.numel() == 0:
        return
    triples_cpu = ca_triples.detach().cpu().tolist()
    prompt_cpu = chiral_prompt.detach().cpu().numpy()
    n_nodes = pos_work.shape[0]
    device = pos_work.device

    for ca_i, n_i, c_i in triples_cpu:
        if not (0 <= ca_i < n_nodes and 0 <= n_i < n_nodes and 0 <= c_i < n_nodes):
            continue
        t_sign = float(prompt_cpu[ca_i, 0])
        t_mask = float(prompt_cpu[ca_i, 1])
        if t_mask < 0.5 or abs(t_sign) < 0.5:
            continue
        sc_nbs = [nb for nb in adj[ca_i] if nb not in (n_i, c_i) and not fixed_cpu[nb]]
        if len(sc_nbs) != 1:
            continue
        cb_i = sc_nbs[0]
        if cb_i in bb_nodes:
            continue
        vn = pos_work[n_i] - pos_work[ca_i]
        vc = pos_work[c_i] - pos_work[ca_i]
        vcb = pos_work[cb_i] - pos_work[ca_i]
        normal = torch.linalg.cross(vn, vc, dim=-1)
        n_norm = normal.norm()
        if n_norm < 1e-4:
            continue
        vol = float((normal * vcb).sum().item())
        if vol * t_sign < 0.0:
            a_unit = normal / n_norm
            visited = {cb_i}
            q = [cb_i]
            while q:
                cur = q.pop(0)
                for nb in adj[cur]:
                    if nb not in visited and nb not in bb_nodes and not fixed_cpu[nb]:
                        visited.add(nb)
                        q.append(nb)
            idx_t = torch.tensor(sorted(visited), dtype=torch.long, device=device)
            rel = pos_work[idx_t] - pos_work[ca_i].unsqueeze(0)
            proj = (rel * a_unit.unsqueeze(0)).sum(dim=-1, keepdim=True)
            pos_work[idx_t] = pos_work[idx_t] - 2.0 * proj * a_unit.unsqueeze(0)


@torch.no_grad()
def correct_macrocycle_geometry(batch, pos=None, config=None):
    """Optional post-denoising 3D macrocycle & stereochemical geometry projector (float32).

    Step 1: Local sidechain reflection across (N, CA, C) when explicit `chiral_prompt` disagrees.
    Step 2: 40-step differentiable float32 Adam refinement (`lr=0.04`) optimizing:
      - Element-aware covalent + closure bond lengths
      - 1-3 geminal distance envelope (`[1.95, 2.85] A`)
      - All covalent `C(=O)-N` + `amide_omega_quads` omega planarity (`<= 16 deg` from cis/trans)
      - Carbonyl `sp2` improper planarity `(prev, C, O, N)`
      - CA chirality hinge preservation/enforcement
      - Intra-molecular (`>= 3` topological hops, `< 1.45 A`) & ligand-pocket (`< 1.75 A`) clash barriers
      with per-graph NaN/divergence fallback to `pos_orig`.
    """
    cfg = config if isinstance(config, dict) else {}
    fix_chirality = bool(cfg.get('fix_chirality', True))
    lr = float(cfg.get('lr', 0.04))
    if 'opt_iters' in cfg:
        opt_iters = int(cfg['opt_iters'])
    elif ('bond_iters' in cfg or 'omega_iters' in cfg) and (
        int(cfg.get('bond_iters', 1)) == 0 and int(cfg.get('omega_iters', 1)) == 0
    ):
        opt_iters = 0
    else:
        opt_iters = 40

    pos_orig = (batch['node_pos'] if pos is None else pos).detach().clone()
    n_nodes = pos_orig.shape[0]
    if n_nodes == 0:
        return pos_orig.clone()

    pos_work = pos_orig.float().clone()
    device = pos_work.device
    fixed_pos = (
        batch['fixed_pos'].bool().to(device)
        if 'fixed_pos' in batch
        else torch.zeros(n_nodes, dtype=torch.bool, device=device)
    )
    fixed_cpu = fixed_pos.detach().cpu().numpy()
    batch_node = (
        batch['node_type_batch'].to(device)
        if 'node_type_batch' in batch
        else torch.zeros(n_nodes, dtype=torch.long, device=device)
    )
    n_graphs = int(batch_node.max().item()) + 1

    adj, bond_bt, closure_pairs = _build_covalent_graph(batch, n_nodes)
    bb_nodes = _extract_node_index_set(
        batch, ('node_bb', 'node_part_bbanchor', 'node_part_bbbody', 'peptide_is_backbone'), n_nodes
    )
    ca_triples = batch.get('ca_anchor_triples', None)
    ca_set = set()
    if ca_triples is not None and ca_triples.numel() > 0:
        for t in ca_triples.detach().cpu().tolist():
            ca_set.add(int(t[0]))
            if not bb_nodes:
                bb_nodes.update(int(x) for x in t)

    # Step 1: exact sidechain reflection for inverted non-bridgehead stereocenters
    if fix_chirality and any(len(nbs) > 0 for nbs in adj):
        _reflect_wrong_chiral_sidechains(pos_work, batch, adj, bb_nodes, fixed_cpu)

    pos_ref = pos_work.detach().clone()
    unfixed_mask = ~fixed_pos
    if opt_iters <= 0 or (not unfixed_mask.any()) or (not torch.isfinite(pos_ref).all()):
        for g in range(n_graphs):
            gm = batch_node == g
            if not torch.isfinite(pos_ref[gm]).all():
                pos_ref[gm] = pos_orig[gm].float()
        return pos_ref.to(pos_orig.dtype)

    # Build topology tensors once on CPU, then vectorize Adam on target device
    node_types = (
        [int(x) for x in batch['node_type'].detach().cpu().tolist()]
        if 'node_type' in batch
        else [0] * n_nodes
    )
    carbonyl_os = [[] for _ in range(n_nodes)]
    for (u, v), bt in bond_bt.items():
        if bt != 2:
            continue
        if node_types[u] == 0 and node_types[v] == 2:
            carbonyl_os[u].append(v)
        elif node_types[v] == 0 and node_types[u] == 2:
            carbonyl_os[v].append(u)

    bond_target_map = {}
    for (u, v), bt in bond_bt.items():
        zu, zv = node_types[u], node_types[v]
        c_has_o = (zu == 0 and zv == 1 and bool(carbonyl_os[u])) or (zv == 0 and zu == 1 and bool(carbonyl_os[v]))
        bond_target_map[(u, v)] = _ideal_bond_length(zu, zv, bt, is_amide_carbonyl_c=c_has_o)
    for cu, cv, tl in closure_pairs:
        key = (min(cu, cv), max(cu, cv))
        bond_target_map.setdefault(key, tl)

    bond_pairs, bond_targets = [], []
    for (u, v), tl in bond_target_map.items():
        if not (fixed_cpu[u] and fixed_cpu[v]):
            bond_pairs.append([u, v])
            bond_targets.append(tl)

    # Detect all covalent C(=O)-N amide quads + carbonyl sp2 impropers + union batch['amide_omega_quads']
    omega_map = {}
    improper_by_c = {}
    batch_omegas = batch.get('amide_omega_quads', None)
    if batch_omegas is not None and batch_omegas.numel() > 0:
        for q in batch_omegas.detach().cpu().tolist():
            q0, q1, q2, q3 = int(q[0]), int(q[1]), int(q[2]), int(q[3])
            if len({q0, q1, q2, q3}) == 4 and all(0 <= x < n_nodes for x in (q0, q1, q2, q3)):
                if not all(fixed_cpu[x] for x in (q0, q1, q2, q3)):
                    omega_map[(min(q1, q2), max(q1, q2))] = [q0, q1, q2, q3]
                if carbonyl_os[q1]:
                    o_i = carbonyl_os[q1][0]
                    if len({q0, q1, o_i, q2}) == 4 and not all(fixed_cpu[x] for x in (q0, q1, o_i, q2)):
                        improper_by_c.setdefault(q1, [q0, q1, o_i, q2])

    for (u, v) in bond_bt.keys():
        zu, zv = node_types[u], node_types[v]
        if not ((zu == 0 and zv == 1) or (zu == 1 and zv == 0)):
            continue
        c_i, n_i = (u, v) if zu == 0 else (v, u)
        if not carbonyl_os[c_i] or len(adj[n_i]) < 2:
            continue
        o_set = set(carbonyl_os[c_i])
        prev_cands = [nb for nb in adj[c_i] if nb != n_i and nb not in o_set]
        next_cands = [nb for nb in adj[n_i] if nb != c_i]
        if not prev_cands or not next_cands:
            continue
        prev_cands.sort(key=lambda nb: (nb in bb_nodes, nb in ca_set, len(adj[nb]) >= 2), reverse=True)
        next_cands.sort(key=lambda nb: (nb in ca_set, nb in bb_nodes, len(adj[nb]) >= 2), reverse=True)
        prev_i, next_i = prev_cands[0], next_cands[0]
        o_i = carbonyl_os[c_i][0]
        if len({prev_i, c_i, o_i, n_i}) == 4 and not all(fixed_cpu[x] for x in (prev_i, c_i, o_i, n_i)):
            improper_by_c.setdefault(c_i, [prev_i, c_i, o_i, n_i])
        if len({prev_i, c_i, n_i, next_i}) == 4 and not all(fixed_cpu[x] for x in (prev_i, c_i, n_i, next_i)):
            omega_map.setdefault((min(c_i, n_i), max(c_i, n_i)), [prev_i, c_i, n_i, next_i])

    omega_quads = list(omega_map.values())
    improper_quads = list(improper_by_c.values())

    # 1-3 geminal pairs via shared bonded neighbor b
    gem_set = set()
    for b in range(n_nodes):
        nbs = adj[b]
        for i in range(len(nbs)):
            a = nbs[i]
            for j in range(i + 1, len(nbs)):
                c = nbs[j]
                key = (min(a, c), max(a, c))
                if key not in bond_target_map and not (fixed_cpu[key[0]] and fixed_cpu[key[1]]):
                    gem_set.add(key)
    gem_pairs = sorted(gem_set)

    # Intra non-bonded pairs (>= 3 topological hops in the same graph)
    batch_node_cpu = batch_node.detach().cpu().tolist()
    graph_nodes = [[] for _ in range(n_graphs)]
    for idx_n, g_id in enumerate(batch_node_cpu):
        graph_nodes[g_id].append(idx_n)

    intra_pairs = []
    for g_nodes in graph_nodes:
        for u in g_nodes:
            hop02 = {u}
            for nb1 in adj[u]:
                hop02.add(nb1)
                hop02.update(adj[nb1])
            for v in g_nodes:
                if v > u and v not in hop02 and not (fixed_cpu[u] and fixed_cpu[v]):
                    intra_pairs.append([u, v])

    # Ligand-pocket pairs (initial distance <= 5.0 A in the same graph)
    lp_lig_idx, lp_poc_idx = [], []
    poc_pos_raw = batch.get('pocket_pos', None)
    poc_pos_f32 = None
    if poc_pos_raw is not None and poc_pos_raw.numel() > 0:
        poc_pos_f32 = poc_pos_raw.detach().float().to(device)
        poc_batch = (
            batch['pocket_pos_batch'].to(device)
            if 'pocket_pos_batch' in batch
            else torch.zeros(poc_pos_f32.shape[0], dtype=torch.long, device=device)
        )
        for g in range(n_graphs):
            lig_g = torch.nonzero((batch_node == g) & unfixed_mask, as_tuple=False).view(-1)
            poc_g = torch.nonzero(poc_batch == g, as_tuple=False).view(-1)
            if lig_g.numel() == 0 or poc_g.numel() == 0:
                continue
            dmat = torch.cdist(pos_ref[lig_g], poc_pos_f32[poc_g])
            li, pi = torch.nonzero(dmat <= 5.0, as_tuple=True)
            if li.numel() > 0:
                lp_lig_idx.append(lig_g[li])
                lp_poc_idx.append(poc_g[pi])
    lp_lig_t = torch.cat(lp_lig_idx) if lp_lig_idx else None
    lp_poc_t = torch.cat(lp_poc_idx) if lp_poc_idx else None

    # Chiral hinge quads preserving/enforcing L/D orientation
    chiral_quads, chiral_signs = [], []
    chiral_prompt = batch.get('chiral_prompt', None)
    prompt_cpu = chiral_prompt.detach().cpu().numpy() if (chiral_prompt is not None and chiral_prompt.numel() > 0) else None
    if ca_triples is not None and ca_triples.numel() > 0:
        for ca_i, n_i, c_i in ca_triples.detach().cpu().tolist():
            if not (0 <= ca_i < n_nodes and 0 <= n_i < n_nodes and 0 <= c_i < n_nodes):
                continue
            if n_i not in adj[ca_i] or c_i not in adj[ca_i]:
                continue
            sc_nbs = [nb for nb in adj[ca_i] if nb not in (n_i, c_i)]
            if len(sc_nbs) != 1:
                continue
            cb_i = sc_nbs[0]
            if all(fixed_cpu[x] for x in (ca_i, n_i, c_i, cb_i)):
                continue
            u_r = _safe_unit_vec(pos_ref[n_i] - pos_ref[ca_i])
            v_r = _safe_unit_vec(pos_ref[c_i] - pos_ref[ca_i])
            w_r = _safe_unit_vec(pos_ref[cb_i] - pos_ref[ca_i])
            vol_r = float((torch.linalg.cross(u_r, v_r, dim=-1) * w_r).sum().item())
            t_sign = None
            if (cb_i not in bb_nodes) and (prompt_cpu is not None):
                pm, ps = float(prompt_cpu[ca_i, 1]), float(prompt_cpu[ca_i, 0])
                if pm >= 0.5 and abs(ps) >= 0.5:
                    t_sign = 1.0 if ps > 0.0 else -1.0
            if t_sign is None and abs(vol_r) > 0.10:
                t_sign = 1.0 if vol_r > 0.0 else -1.0
            if t_sign is not None:
                chiral_quads.append([ca_i, n_i, c_i, cb_i])
                chiral_signs.append(t_sign)

    # Convert active index lists to device tensors
    bond_t = torch.tensor(bond_pairs, dtype=torch.long, device=device) if bond_pairs else None
    bond_len_t = torch.tensor(bond_targets, dtype=torch.float32, device=device) if bond_targets else None
    gem_t = torch.tensor(gem_pairs, dtype=torch.long, device=device) if gem_pairs else None
    omega_t = torch.tensor(omega_quads, dtype=torch.long, device=device) if omega_quads else None
    imp_t = torch.tensor(improper_quads, dtype=torch.long, device=device) if improper_quads else None
    intra_t = torch.tensor(intra_pairs, dtype=torch.long, device=device) if intra_pairs else None
    chiral_t = torch.tensor(chiral_quads, dtype=torch.long, device=device) if chiral_quads else None
    chiral_s_t = torch.tensor(chiral_signs, dtype=torch.float32, device=device) if chiral_signs else None

    unfixed_scale = unfixed_mask.float().unsqueeze(-1)
    with torch.enable_grad():
        delta = torch.zeros_like(pos_ref, requires_grad=True)
        opt = torch.optim.Adam([delta], lr=lr)
        for _ in range(opt_iters):
            opt.zero_grad()
            pos_cur = pos_ref + delta * unfixed_scale
            tether_mse = F.mse_loss(pos_cur[unfixed_mask], pos_ref[unfixed_mask])
            total_loss = 0.5 * tether_mse

            if bond_t is not None:
                diff_b = pos_cur[bond_t[:, 0]] - pos_cur[bond_t[:, 1]]
                d_b = torch.sqrt((diff_b * diff_b).sum(dim=-1) + 1e-8)
                total_loss = total_loss + 15.0 * ((d_b - bond_len_t) ** 2).mean()

            if gem_t is not None:
                diff_g = pos_cur[gem_t[:, 0]] - pos_cur[gem_t[:, 1]]
                d_13 = torch.sqrt((diff_g * diff_g).sum(dim=-1) + 1e-8)
                gem_loss = (F.relu(1.95 - d_13).pow(2) + F.relu(d_13 - 2.85).pow(2)).mean()
                total_loss = total_loss + 3.0 * gem_loss

            if omega_t is not None:
                sin_o, cos_o = _safe_dihedral_sin_cos(
                    pos_cur[omega_t[:, 0]], pos_cur[omega_t[:, 1]], pos_cur[omega_t[:, 2]], pos_cur[omega_t[:, 3]]
                )
                s_1d, c_1d = sin_o.squeeze(-1), cos_o.squeeze(-1)
                dev_rad = torch.atan2(torch.sqrt(s_1d * s_1d + 1e-10), c_1d.abs() + 1e-8)
                omega_loss = (s_1d * s_1d).mean() + 4.0 * F.relu(dev_rad - 0.28).pow(2).mean()
                total_loss = total_loss + 10.0 * omega_loss

            if imp_t is not None:
                c_p = pos_cur[imp_t[:, 1]]
                u_i = _safe_unit_vec(pos_cur[imp_t[:, 0]] - c_p)
                v_i = _safe_unit_vec(pos_cur[imp_t[:, 2]] - c_p)
                w_i = _safe_unit_vec(pos_cur[imp_t[:, 3]] - c_p)
                vol_imp = (torch.linalg.cross(u_i, v_i, dim=-1) * w_i).sum(dim=-1)
                total_loss = total_loss + 4.0 * (vol_imp * vol_imp).mean()

            if chiral_t is not None:
                ca_p = pos_cur[chiral_t[:, 0]]
                u_c = _safe_unit_vec(pos_cur[chiral_t[:, 1]] - ca_p)
                v_c = _safe_unit_vec(pos_cur[chiral_t[:, 2]] - ca_p)
                w_c = _safe_unit_vec(pos_cur[chiral_t[:, 3]] - ca_p)
                vol_c = (torch.linalg.cross(u_c, v_c, dim=-1) * w_c).sum(dim=-1)
                chiral_loss = F.relu(0.45 - chiral_s_t * vol_c).pow(2).mean()
                total_loss = total_loss + 6.0 * chiral_loss

            if intra_t is not None:
                diff_i = pos_cur[intra_t[:, 0]] - pos_cur[intra_t[:, 1]]
                d_ij = torch.sqrt((diff_i * diff_i).sum(dim=-1) + 1e-8)
                total_loss = total_loss + 12.0 * F.relu(1.45 - d_ij).pow(2).mean()

            if lp_lig_t is not None and poc_pos_f32 is not None:
                diff_p = pos_cur[lp_lig_t] - poc_pos_f32[lp_poc_t]
                d_ip = torch.sqrt((diff_p * diff_p).sum(dim=-1) + 1e-8)
                total_loss = total_loss + 12.0 * F.relu(1.75 - d_ip).pow(2).mean()

            if not torch.isfinite(total_loss):
                break
            total_loss.backward()
            opt.step()

    pos_out = (pos_ref + delta.detach() * unfixed_scale).float()
    for g in range(n_graphs):
        gm = batch_node == g
        pos_g = pos_out[gm]
        if (not torch.isfinite(pos_g).all()) or float((pos_g - pos_ref[gm]).norm(dim=-1).max().item()) > 8.0:
            pos_out[gm] = pos_orig[gm].float()
    return pos_out.to(pos_orig.dtype)


if __name__ == '__main__':
    
    # Original points
    # X = torch.tensor([[1.0, 1.0, 1.0],
    #                 [2.0, 2.0, 2.0],
    #                 [3.0, 3.0, 3.0]])
    X = torch.tensor([[[1., 0, 0],
                      [1, 0, 2],
                      [0, 0, 2],
                      [0, -5, 0]],
                      [[1., 0, 0],
                      [1, 0, 2],
                      [0, 0, 2],
                      [0, -5, 0]]])

    # Transformed points
    Y = torch.tensor([[[0., 1, 0],
                      [0, 1, 2],
                      [0, 0, 2],
                      [5, 0, 0]],
                      [[0., 1, 0],
                      [0, 1, 2],
                      [0, 0, 2],
                      [5, 0, 0]]])
    # Y = torch.tensor([[2.0, 3.0, 0.0],
    #                   [3.0, 3.0, 1.0],
    #                   [4.0, 4.0, 2.0]])
    # Y = torch.tensor([[ 1.8453,  2.7560, -0.1547],
    #         [ 3.0000,  3.3333,  1.0000],
    #         [ 4.1547,  3.9107,  2.1547]])

    # Perform Procrustes analysis
    # R, translation = procrustes_analysis_batch(X, Y)

    # Perform Kabsch algorithm
    R, translation = kabsch_batch(X, Y)


    print("Rotation Matrix:")
    print(R)
    print("\nTranslation Vector:")
    print(translation)

    print('Apply transformation to X:')
    # print(torch.matmul(R, X.T).T + translation)
    Y_app = torch.matmul(X, R.transpose(1, 2)) + translation
    print(Y_app)
    
    print('Delta:')
    print((Y - Y_app).abs().sum())

            