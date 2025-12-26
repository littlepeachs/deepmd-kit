# SPDX-License-Identifier: LGPL-3.0-or-later

import torch

from deepmd.pt.utils.preprocess import (
    compute_envelope,
    compute_new_weight,
    compute_smooth_weight,
)

def prod_env_from_edges(
    edge_index: torch.LongTensor,     # [2, E]
    edge_vector: torch.Tensor,        # [E, 3]
    distance: torch.Tensor,           # [E] or [E, 1]
    atype: torch.LongTensor,          # [N]
    mean: torch.Tensor,               # [n_type, D]
    stddev: torch.Tensor,             # [n_type, D]
    rcut: float,
    rcut_smth: float,
    radial_only: bool = False,
    protection: float = 0.0,
    use_env_envelope: bool = False,
    use_new_sw: bool = False,
):
    """
    基于已有 edge_index / edge_vector / distance 构造 env_mat，并按原子类型做标准化。
    形状完全由 edge 决定：每一条边对应一行特征。

    返回
    ----
    env_mat:      [E, D]
    weight:       [E, 1]   （也就是 switch）
    env_mat_se_a: [E, D]
    switch:       [E, 1]
    """
    # 1. 基本形状
    if distance.dim() == 1:
        length = distance.unsqueeze(-1)    # [E, 1]
    else:
        length = distance                  # [E, 1]

    # 2. 基础几何项
    # t0 = 1/r, t1 = r_vec / r^2
    t0 = 1.0 / (length + protection)                     # [E, 1]
    t1 = edge_vector / (length + protection) ** 2        # [E, 3]

    # 3. 平滑权重
    if use_new_sw:
        weight = compute_new_weight(length, rcut_smth, rcut)   # [E, 1]
    elif use_env_envelope:
        weight = compute_envelope(length, rcut_smth, rcut)     # [E, 1]
    else:
        weight = compute_smooth_weight(length, rcut_smth, rcut)  # [E, 1]

    # 4. 组装 env_mat
    if radial_only:
        # D = 1
        env_mat = t0 * weight                     # [E, 1]
        D = 1
    else:
        # D = 4  ->  [1/r, x/r^2, y/r^2, z/r^2]
        env_mat = torch.cat([t0, t1], dim=-1) * weight   # [E, 4]
        D = 4

    # 5. 按原子类型标准化 —— 用边的源节点类型
    src = edge_index[0]                 # [E]
    # mean/stddev: [n_type, D] → 按 src 的类型取出来 → [E, D]
    t_avg = mean[atype[src]]            # [E, D]
    t_std = stddev[atype[src]]          # [E, D]
    
    env_mat_se_a = (env_mat - t_avg) / t_std

    # 6. 返回
    return env_mat_se_a, weight



def _make_env_mat(
    nlist,
    coord,
    rcut: float,
    ruct_smth: float,
    radial_only: bool = False,
    protection: float = 0.0,
    use_env_envelope: bool = False,
    use_new_sw: bool = False,
):
    """Make smooth environment matrix."""
    bsz, natoms, nnei = nlist.shape
    coord = coord.view(bsz, -1, 3)
    nall = coord.shape[1]
    mask = nlist >= 0
    # nlist = nlist * mask  ## this impl will contribute nans in Hessian calculation.
    nlist = torch.where(mask, nlist, nall - 1)
    coord_l = coord[:, :natoms].view(bsz, -1, 1, 3)
    index = nlist.view(bsz, -1).unsqueeze(-1).expand(-1, -1, 3)
    coord_r = torch.gather(coord, 1, index)
    coord_r = coord_r.view(bsz, natoms, nnei, 3)
    diff = coord_r - coord_l
    length = torch.linalg.norm(diff, dim=-1, keepdim=True)
    # for index 0 nloc atom
    length = length + ~mask.unsqueeze(-1)
    t0 = 1 / (length + protection)
    t1 = diff / (length + protection) ** 2
    if use_new_sw:
        weight = compute_new_weight(length, ruct_smth, rcut)
    elif use_env_envelope:
        weight = compute_envelope(length, ruct_smth, rcut)
    else:
        weight = compute_smooth_weight(length, ruct_smth, rcut)
    weight = weight * mask.unsqueeze(-1)
    if radial_only:
        env_mat = t0 * weight
    else:
        env_mat = torch.cat([t0, t1], dim=-1) * weight
    return env_mat, diff * mask.unsqueeze(-1), weight


def prod_env_mat(
    extended_coord,
    nlist,
    atype,
    mean,
    stddev,
    rcut: float,
    rcut_smth: float,
    radial_only: bool = False,
    protection: float = 0.0,
    use_env_envelope: bool = False,
    use_new_sw: bool = False,
):
    """Generate smooth environment matrix from atom coordinates and other context.

    Args:
    - extended_coord: Copied atom coordinates with shape [nframes, nall*3].
    - atype: Atom types with shape [nframes, nloc].
    - mean: Average value of descriptor per element type with shape [len(sec), nnei, 4 or 1].
    - stddev: Standard deviation of descriptor per element type with shape [len(sec), nnei, 4 or 1].
    - rcut: Cut-off radius.
    - rcut_smth: Smooth hyper-parameter for pair force & energy.
    - radial_only: Whether to return a full description or a radial-only descriptor.
    - protection: Protection parameter to prevent division by zero errors during calculations.

    Returns
    -------
    - env_mat: Shape is [nframes, natoms[1]*nnei*4].
    """
    _env_mat_se_a, diff, switch = _make_env_mat(
        nlist,
        extended_coord,
        rcut,
        rcut_smth,
        radial_only,
        protection=protection,
        use_env_envelope=use_env_envelope,
        use_new_sw=use_new_sw,
    )  # shape [n_atom, dim, 4 or 1]
    t_avg = mean[atype]  # [n_atom, dim, 4 or 1]
    t_std = stddev[atype]  # [n_atom, dim, 4 or 1]
    env_mat_se_a = (_env_mat_se_a - t_avg) / t_std
    return env_mat_se_a, diff, switch
