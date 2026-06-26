# SPDX-License-Identifier: LGPL-3.0-or-later

import torch


def get_graph_index_flat(
    nlist_flat: torch.Tensor,
    a_nlist_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get edge and angle graph indices for flat local neighbor lists.

    ``nlist_flat`` has shape ``[total_atoms, nnei]`` and stores flattened local
    atom indices, with ``-1`` for padding. ``a_nlist_mask`` marks valid angle
    neighbors in the first ``a_sel`` columns.
    """
    total_atoms = nlist_flat.shape[0]
    nnei = nlist_flat.shape[1]
    device = nlist_flat.device
    dtype = nlist_flat.dtype
    a_sel = a_nlist_mask.shape[1]

    nlist_mask = nlist_flat >= 0
    n_edge = nlist_mask.sum().item()

    a_nlist_mask_3d = a_nlist_mask[:, :, None] & a_nlist_mask[:, None, :]

    atom_indices = torch.arange(total_atoms, dtype=dtype, device=device)
    n2e_index = atom_indices[:, None].expand(-1, nnei)[nlist_mask]
    n_ext2e_index = nlist_flat[nlist_mask]
    edge_index = torch.stack([n2e_index, n_ext2e_index], dim=0)

    n2a_index = atom_indices[:, None, None].expand(-1, a_sel, a_sel)[a_nlist_mask_3d]

    edge_id = torch.arange(n_edge, dtype=dtype, device=device)
    edge_lookup = torch.full((total_atoms, nnei), -1, dtype=dtype, device=device)
    edge_lookup[nlist_mask] = edge_id
    edge_lookup_a = edge_lookup[:, :a_sel]

    edge_lookup_ij = edge_lookup_a[:, :, None].expand(-1, -1, a_sel)
    eij2a_index = edge_lookup_ij[a_nlist_mask_3d]

    edge_lookup_ik = edge_lookup_a[:, None, :].expand(-1, a_sel, -1)
    eik2a_index = edge_lookup_ik[a_nlist_mask_3d]

    angle_index = torch.stack([n2a_index, eij2a_index, eik2a_index], dim=0)
    return edge_index, angle_index
