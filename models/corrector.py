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


@torch.no_grad()
def correct_macrocycle_geometry(batch, pos=None, config=None):
    """Optional light post-denoising 3D macrocycle geometry corrector (float32).

    Refines:
      1. Macrocycle closure bond lengths (`ring_closure_bonds` -> `ring_closure_target_len`)
         and extreme covalent bond stretching (`d > 1.95 A` or `d < 1.15 A`).
      2. Strained amide omega planarity (`35 deg < |omega| < 145 deg`).
      3. Optional chiral sign alignment against explicit `chiral_prompt` via exact reflection
         of attached non-backbone sidechain subtree across the `(N, CA, C)` plane.

    Respects `batch['fixed_pos']` and leaves `batch` unchanged unless caller assigns output.
    """
    cfg = config if isinstance(config, dict) else {}
    bond_iters = int(cfg.get('bond_iters', 15))
    bond_lr = float(cfg.get('bond_lr', 0.25))
    omega_iters = int(cfg.get('omega_iters', 5))
    omega_lr = float(cfg.get('omega_lr', 0.1))
    fix_chirality = bool(cfg.get('fix_chirality', True))

    pos_orig = batch['node_pos'] if pos is None else pos
    pos_work = pos_orig.detach().clone().float()
    n_nodes = pos_work.shape[0]
    if n_nodes == 0:
        return pos_orig.clone()

    device = pos_work.device
    fixed_pos = (
        batch['fixed_pos'].bool()
        if 'fixed_pos' in batch
        else torch.zeros(n_nodes, dtype=torch.bool, device=device)
    )

    # 1. Ring closure & covalent bond length spring relaxation
    close_bonds = batch.get('ring_closure_bonds', None)
    close_lens = batch.get('ring_closure_target_len', None)
    has_close = close_bonds is not None and close_bonds.numel() > 0

    he_idx = batch.get('halfedge_index', None)
    he_type = batch.get('halfedge_type', None)
    cov_u = cov_v = None
    if he_idx is not None and he_type is not None and he_idx.numel() > 0:
        cov_mask = (he_type > 0) & (he_type <= 4)
        if cov_mask.any():
            cov_u = he_idx[0, cov_mask]
            cov_v = he_idx[1, cov_mask]

    for _ in range(bond_iters):
        disp = torch.zeros_like(pos_work)
        if has_close:
            cu, cv = close_bonds[:, 0].long(), close_bonds[:, 1].long()
            target_l = (
                close_lens.float().to(device)
                if (close_lens is not None and close_lens.shape[0] == cu.shape[0])
                else torch.full((cu.shape[0],), 1.45, dtype=torch.float32, device=device)
            )
            diff = pos_work[cu] - pos_work[cv]
            dist = torch.sqrt((diff * diff).sum(dim=-1) + 1e-8)
            err = dist - target_l
            vec = bond_lr * (err / dist).unsqueeze(-1) * diff
            disp = disp - scatter_sum(0.5 * vec, cu, dim=0, dim_size=n_nodes)
            disp = disp + scatter_sum(0.5 * vec, cv, dim=0, dim_size=n_nodes)

        if cov_u is not None:
            diff_c = pos_work[cov_u] - pos_work[cov_v]
            dist_c = torch.sqrt((diff_c * diff_c).sum(dim=-1) + 1e-8)
            too_long = dist_c > 1.95
            too_short = dist_c < 1.15
            if too_long.any() or too_short.any():
                target_c = torch.where(too_long, 1.54, torch.where(too_short, 1.35, dist_c))
                err_c = torch.where(too_long | too_short, dist_c - target_c, torch.zeros_like(dist_c))
                vec_c = (0.5 * bond_lr) * (err_c / dist_c).unsqueeze(-1) * diff_c
                disp = disp - scatter_sum(0.5 * vec_c, cov_u, dim=0, dim_size=n_nodes)
                disp = disp + scatter_sum(0.5 * vec_c, cov_v, dim=0, dim_size=n_nodes)

        disp_norm = torch.sqrt((disp * disp).sum(dim=-1, keepdim=True) + 1e-12)
        disp = disp * (disp_norm.clamp(max=0.35) / disp_norm)
        pos_work[~fixed_pos] = pos_work[~fixed_pos] + disp[~fixed_pos]

    # 2. Gentle omega amide planarity nudge for strained bonds
    omega_quads = batch.get('amide_omega_quads', None)
    if omega_quads is not None and omega_quads.numel() > 0 and omega_iters > 0:
        q0, q1, q2, q3 = omega_quads[:, 0], omega_quads[:, 1], omega_quads[:, 2], omega_quads[:, 3]
        for _ in range(omega_iters):
            sin_o, cos_o = get_dihedral(pos_work[q0], pos_work[q1], pos_work[q2], pos_work[q3])
            ang = torch.atan2(sin_o.squeeze(-1), cos_o.squeeze(-1))
            abs_ang = ang.abs()
            strained = (abs_ang > 0.61) & (abs_ang < 2.53)
            if not strained.any():
                break
            target_ang = torch.where(abs_ang >= (0.5 * torch.pi), torch.sign(ang) * torch.pi, torch.zeros_like(ang))
            delta_ang = target_ang - ang
            axis = F.normalize(pos_work[q2] - pos_work[q1], dim=-1, eps=1e-4)
            r3 = pos_work[q3] - pos_work[q2]
            perp3 = r3 - (r3 * axis).sum(dim=-1, keepdim=True) * axis
            tang3 = torch.linalg.cross(axis, perp3, dim=-1)
            step3 = torch.where(strained.unsqueeze(-1), omega_lr * delta_ang.unsqueeze(-1) * tang3, torch.zeros_like(tang3))
            n3 = torch.sqrt((step3 * step3).sum(dim=-1, keepdim=True) + 1e-12)
            step3 = step3 * (n3.clamp(max=0.15) / n3)
            disp_o = scatter_sum(step3, q3, dim=0, dim_size=n_nodes)
            pos_work[~fixed_pos] = pos_work[~fixed_pos] + disp_o[~fixed_pos]

    # 3. Optional local sidechain reflection across (N, CA, C) plane if chiral_prompt disagrees
    ca_triples = batch.get('ca_anchor_triples', None)
    chiral_prompt = batch.get('chiral_prompt', None)
    if (
        fix_chirality
        and ca_triples is not None
        and ca_triples.numel() > 0
        and chiral_prompt is not None
        and chiral_prompt.numel() > 0
        and cov_u is not None
    ):
        # Build covalent adjacency list once
        adj = [[] for _ in range(n_nodes)]
        u_list = cov_u.detach().cpu().tolist()
        v_list = cov_v.detach().cpu().tolist()
        for uu, vv in zip(u_list, v_list):
            adj[uu].append(vv)
            adj[vv].append(uu)

        def _extract_bb_indices(t):
            if t.dtype == torch.bool:
                return torch.nonzero(t, as_tuple=False).view(-1).detach().cpu().tolist()
            return t.detach().cpu().tolist()

        bb_nodes = set()
        if 'node_bb' in batch:
            bb_nodes.update(_extract_bb_indices(batch['node_bb']))
        if 'node_part_bbanchor' in batch:
            bb_nodes.update(_extract_bb_indices(batch['node_part_bbanchor']))
        if 'node_part_bbbody' in batch:
            bb_nodes.update(_extract_bb_indices(batch['node_part_bbbody']))
        if 'peptide_is_backbone' in batch:
            bb_nodes.update(_extract_bb_indices(batch['peptide_is_backbone'].bool()))
        if not bb_nodes:
            for t in ca_triples.detach().cpu().tolist():
                bb_nodes.update(t)

        triples_cpu = ca_triples.detach().cpu().tolist()
        prompt_cpu = chiral_prompt.detach().cpu().numpy()
        fixed_cpu = fixed_pos.detach().cpu().numpy()

        for ca_i, n_i, c_i in triples_cpu:
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
                # Collect attached non-backbone sidechain subtree starting from cb_i
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

    return pos_work.to(pos_orig.dtype)


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

            