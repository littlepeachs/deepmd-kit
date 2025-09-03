import argparse
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
from torch_geometric.data import Data, Batch
from torch_scatter import scatter, segment_coo, segment_csr
import numpy as np
import torch
from torch_scatter import scatter
import torch.nn as nn
from e3nn import o3
from e3nn.o3 import Irreps
from e3nn.o3 import FullyConnectedTensorProduct, TensorProduct, Linear
from e3nn.nn import FullyConnectedNet, Gate
import vesin
from torch_geometric.nn import radius_graph

from deepmd.pt.model.descriptor.env_mat import (
    prod_env_mat,
)
from deepmd.dpmodel.descriptor.eqnorm import EqnormArgs
from .base_descriptor import (
    BaseDescriptor,
)
from deepmd.pt.model.network.network import (
    TypeEmbedNet,
    TypeEmbedNetConsistent,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    PRECISION_DICT,
)
from deepmd.pt.utils.update_sel import (
    UpdateSel,
)
from deepmd.pt.utils.utils import (
    to_numpy_array,
)
from deepmd.utils.data_system import (
    DeepmdDataSystem,
)
from deepmd.utils.finetune import (
    get_index_between_two_maps,
    map_pair_exclude_types,
)
from deepmd.utils.path import (
    DPPath,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from .base_descriptor import (
    BaseDescriptor,
)
from .descriptor import (
    extend_descrpt_stat,
)
from .repflow_layer import (
    RepFlowLayer,
)
from .repflows import (
    DescrptBlockRepflows,
)
from deepmd.pt.model.network.mlp import (
    MLPLayer,
)
from deepmd.dpmodel.utils import EnvMat as DPEnvMat
from deepmd.pt.utils.env_mat_stat import (
    EnvMatStatSe,
)

element_dict = {
    'H':1, 'He':2, 'Li':3, 'Be':4, 'B':5, 'C':6, 'N':7, 'O':8, 'F':9, 'Ne':10, 
    'Na':11, 'Mg':12, 'Al':13, 'Si':14, 'P':15, 'S':16, 'Cl':17, 'Ar':18, 
    'K':19, 'Ca':20, 'Sc':21, 'Ti':22, 'V':23, 'Cr':24, 'Mn':25, 'Fe':26, 'Co':27, 'Ni':28, 
    'Cu':29, 'Zn':30, 'Ga':31, 'Ge':32, 'As':33, 'Se':34, 'Br':35, 'Kr':36, 
    'Rb':37, 'Sr':38, 'Y':39, 'Zr':40, 'Nb':41, 'Mo':42, 'Tc':43, 'Ru':44, 'Rh':45, 'Pd':46, 
    'Ag':47, 'Cd':48, 'In':49, 'Sn':50, 'Sb':51, 'Te':52, 'I':53, 'Xe':54, 
    'Cs':55, 'Ba':56, 'La':57, 'Ce':58, 'Pr':59, 'Nd':60, 'Pm':61, 'Sm':62, 'Eu':63, 'Gd':64, 'Tb':65, 'Dy':66, 'Ho':67, 'Er':68, 'Tm':69, 'Yb':70, 'Lu':71, 'Hf':72, 'Ta':73, 'W':74, 'Re':75, 'Os':76, 'Ir':77, 'Pt':78, 'Au':79, 'Hg':80, 'Tl':81, 'Pb':82, 'Bi':83, 'Po':84, 'At':85, 'Rn':86,
    'Fr':87, 'Ra':88, 'Ac':89, 'Th':90, 'Pa':91, 'U':92, 'Np':93, 'Pu':94, 'Am':95, 'Cm':96, 'Bk':97, 'Cf':98, 'Es':99, 'Fm':100, 'Md':101, 'No':102, 'Lr':103, 'Rf':104, 'Db':105, 'Sg':106, 'Bh':107, 'Hs':108, 'Mt':109, 'Ds':110, 'Rg':111, 'Cn':112, 'Nh':113, 'Fl':114, 'Mc':115, 'Lv':116, 'Ts':117, 'Og':118,
}

def get_l_to_all_m_expand_index(lmax: int):
    expand_index = torch.zeros([(lmax + 1) ** 2]).long()
    for lval in range(lmax + 1):
        start_idx = lval**2
        length = 2 * lval + 1
        expand_index[start_idx : (start_idx + length)] = lval
    return expand_index

def radius_graph_pbc(
    data,
    radius,
    max_num_neighbors_threshold,
    enforce_max_neighbors_strictly: bool = False,
    pbc=[True, True, True],
):
    device = data.pos.device
    data_type = data.pos.dtype
    batch_size = len(data.natoms)

    if hasattr(data, "pbc"):
        data.pbc = torch.atleast_2d(data.pbc)
        for i in range(3):
            if not torch.any(data.pbc[:, i]).item():
                pbc[i] = False
            elif torch.all(data.pbc[:, i]).item():
                pbc[i] = True
            else:
                raise RuntimeError(
                    "Different structures in the batch have different PBC configurations. This is not currently supported."
                )

    # position of the atoms
    atom_pos = data.pos

    # Before computing the pairwise distances between atoms, first create a list of atom indices to compare for the entire batch
    num_atoms_per_image = data.natoms
    num_atoms_per_image_sqr = (num_atoms_per_image**2).long()

    # index offset between images
    index_offset = (
        torch.cumsum(num_atoms_per_image, dim=0) - num_atoms_per_image
    )

    index_offset_expand = torch.repeat_interleave(
        index_offset, num_atoms_per_image_sqr
    )
    num_atoms_per_image_expand = torch.repeat_interleave(
        num_atoms_per_image, num_atoms_per_image_sqr
    )

    # Compute a tensor containing sequences of numbers that range from 0 to num_atoms_per_image_sqr for each image
    # that is used to compute indices for the pairs of atoms. This is a very convoluted way to implement
    # the following (but 10x faster since it removes the for loop)
    # for batch_idx in range(batch_size):
    #    batch_count = torch.cat([batch_count, torch.arange(num_atoms_per_image_sqr[batch_idx], device=device)], dim=0)
    num_atom_pairs = torch.sum(num_atoms_per_image_sqr)
    index_sqr_offset = (
        torch.cumsum(num_atoms_per_image_sqr, dim=0) - num_atoms_per_image_sqr
    )
    index_sqr_offset = torch.repeat_interleave(
        index_sqr_offset, num_atoms_per_image_sqr
    )
    atom_count_sqr = (
        torch.arange(num_atom_pairs, device=device) - index_sqr_offset
    )

    # Compute the indices for the pairs of atoms (using division and mod)
    # If the systems get too large this apporach could run into numerical precision issues
    index1 = (
        torch.div(
            atom_count_sqr, num_atoms_per_image_expand, rounding_mode="floor"
        )
    ) + index_offset_expand
    index2 = (
        atom_count_sqr % num_atoms_per_image_expand
    ) + index_offset_expand
    # Get the positions for each atom
    pos1 = torch.index_select(atom_pos, 0, index1)
    pos2 = torch.index_select(atom_pos, 0, index2)

    # Calculate required number of unit cells in each direction.
    # Smallest distance between planes separated by a1 is
    # 1 / ||(a2 x a3) / V||_2, since a2 x a3 is the area of the plane.
    # Note that the unit cell volume V = a1 * (a2 x a3) and that
    # (a2 x a3) / V is also the reciprocal primitive vector
    # (crystallographer's definition).
    cross_a2a3 = torch.cross(data.cell[:, 1], data.cell[:, 2], dim=-1)
    cell_vol = torch.sum(data.cell[:, 0] * cross_a2a3, dim=-1, keepdim=True)

    if pbc[0]:
        inv_min_dist_a1 = torch.norm(cross_a2a3 / cell_vol, p=2, dim=-1)
        rep_a1 = torch.ceil(radius * inv_min_dist_a1)
    else:
        rep_a1 = data.cell.new_zeros(1)

    if pbc[1]:
        cross_a3a1 = torch.cross(data.cell[:, 2], data.cell[:, 0], dim=-1)
        inv_min_dist_a2 = torch.norm(cross_a3a1 / cell_vol, p=2, dim=-1)
        rep_a2 = torch.ceil(radius * inv_min_dist_a2)
    else:
        rep_a2 = data.cell.new_zeros(1)

    if pbc[2]:
        cross_a1a2 = torch.cross(data.cell[:, 0], data.cell[:, 1], dim=-1)
        inv_min_dist_a3 = torch.norm(cross_a1a2 / cell_vol, p=2, dim=-1)
        rep_a3 = torch.ceil(radius * inv_min_dist_a3)
    else:
        rep_a3 = data.cell.new_zeros(1)

    # Take the max over all images for uniformity. This is essentially padding.
    # Note that this can significantly increase the number of computed distances
    # if the required repetitions are very different between images
    # (which they usually are). Changing this to sparse (scatter) operations
    # might be worth the effort if this function becomes a bottleneck.
    max_rep = [rep_a1.max(), rep_a2.max(), rep_a3.max()]

    # Tensor of unit cells
    cells_per_dim = [
        torch.arange(-rep, rep + 1, device=device, dtype=torch.float)
        for rep in max_rep
    ]
    unit_cell = torch.cartesian_prod(*cells_per_dim).to(data_type)
    num_cells = len(unit_cell)
    unit_cell_per_atom = unit_cell.view(1, num_cells, 3).repeat(
        len(index2), 1, 1
    )
    unit_cell = torch.transpose(unit_cell, 0, 1)
    unit_cell_batch = unit_cell.view(1, 3, num_cells).expand(
        batch_size, -1, -1
    )

    # Compute the x, y, z positional offsets for each cell in each image
    data_cell = torch.transpose(data.cell, 1, 2)
    
    pbc_offsets = torch.bmm(data_cell, unit_cell_batch)
    pbc_offsets_per_atom = torch.repeat_interleave(
        pbc_offsets, num_atoms_per_image_sqr, dim=0
    )

    # Expand the positions and indices for the 9 cells
    pos1 = pos1.view(-1, 3, 1).expand(-1, -1, num_cells)
    pos2 = pos2.view(-1, 3, 1).expand(-1, -1, num_cells)
    index1 = index1.view(-1, 1).repeat(1, num_cells).view(-1)
    index2 = index2.view(-1, 1).repeat(1, num_cells).view(-1)
    # Add the PBC offsets for the second atom
    pos2 = pos2 + pbc_offsets_per_atom

    # Compute the squared distance between atoms
    edge_vector = pos2 - pos1
    atom_distance_sqr = torch.sum((pos1 - pos2) ** 2, dim=1)
    atom_distance_sqr = atom_distance_sqr.view(-1)
    import pdb; pdb.set_trace()
    # Remove pairs that are too far apart
    mask_within_radius = torch.le(atom_distance_sqr, radius * radius)
    # Remove pairs with the same atoms (distance = 0.0)
    mask_not_same = torch.gt(atom_distance_sqr, 0.0001)
    mask = torch.logical_and(mask_within_radius, mask_not_same)
    index1 = torch.masked_select(index1, mask)
    index2 = torch.masked_select(index2, mask)
    unit_cell = torch.masked_select(
        unit_cell_per_atom.view(-1, 3), mask.view(-1, 1).expand(-1, 3)
    )
    unit_cell = unit_cell.view(-1, 3)
    atom_distance_sqr = torch.masked_select(atom_distance_sqr, mask)
    
    import pdb; pdb.set_trace()
    mask_num_neighbors, num_neighbors_image = get_max_neighbors_mask(
        natoms=data.natoms,
        index=index1,
        atom_distance=atom_distance_sqr,
        max_num_neighbors_threshold=max_num_neighbors_threshold,
        enforce_max_strictly=enforce_max_neighbors_strictly,
    )

    if not torch.all(mask_num_neighbors):
        # Mask out the atoms to ensure each atom has at most max_num_neighbors_threshold neighbors
        index1 = torch.masked_select(index1, mask_num_neighbors)
        index2 = torch.masked_select(index2, mask_num_neighbors)
        unit_cell = torch.masked_select(
            unit_cell.view(-1, 3), mask_num_neighbors.view(-1, 1).expand(-1, 3)
        )
        unit_cell = unit_cell.view(-1, 3)

    edge_index = torch.stack((index2, index1))

    return edge_index, edge_vector

def get_max_neighbors_mask(
    natoms,
    index,
    atom_distance,
    max_num_neighbors_threshold,
    degeneracy_tolerance: float = 0.01,
    enforce_max_strictly: bool = False,
):
    """
    Give a mask that filters out edges so that each atom has at most
    `max_num_neighbors_threshold` neighbors.
    Assumes that `index` is sorted.

    Enforcing the max strictly can force the arbitrary choice between
    degenerate edges. This can lead to undesired behaviors; for
    example, bulk formation energies which are not invariant to
    unit cell choice.

    A degeneracy tolerance can help prevent sudden changes in edge
    existence from small changes in atom position, for example,
    rounding errors, slab relaxation, temperature, etc.
    """

    device = natoms.device
    num_atoms = natoms.sum()

    # Get number of neighbors
    # segment_coo assumes sorted index
    ones = index.new_ones(1).expand_as(index)
    num_neighbors = segment_coo(ones, index, dim_size=num_atoms)
    max_num_neighbors = num_neighbors.max()
    num_neighbors_thresholded = num_neighbors.clamp(
        max=max_num_neighbors_threshold
    )

    # Get number of (thresholded) neighbors per image
    image_indptr = torch.zeros(
        natoms.shape[0] + 1, device=device, dtype=torch.long
    )
    image_indptr[1:] = torch.cumsum(natoms, dim=0)
    num_neighbors_image = segment_csr(num_neighbors_thresholded, image_indptr)

    # If max_num_neighbors is below the threshold, return early
    if (
        max_num_neighbors <= max_num_neighbors_threshold
        or max_num_neighbors_threshold <= 0
    ):
        mask_num_neighbors = torch.tensor(
            [True], dtype=bool, device=device
        ).expand_as(index)
        return mask_num_neighbors, num_neighbors_image

    # Create a tensor of size [num_atoms, max_num_neighbors] to sort the distances of the neighbors.
    # Fill with infinity so we can easily remove unused distances later.
    distance_sort = torch.full(
        [num_atoms * max_num_neighbors], np.inf, device=device
    )

    # Create an index map to map distances from atom_distance to distance_sort
    # index_sort_map assumes index to be sorted
    index_neighbor_offset = torch.cumsum(num_neighbors, dim=0) - num_neighbors
    index_neighbor_offset_expand = torch.repeat_interleave(
        index_neighbor_offset, num_neighbors
    )
    index_sort_map = (
        index * max_num_neighbors
        + torch.arange(len(index), device=device)
        - index_neighbor_offset_expand
    )
    distance_sort.index_copy_(0, index_sort_map, atom_distance)
    distance_sort = distance_sort.view(num_atoms, max_num_neighbors)

    # Sort neighboring atoms based on distance
    distance_sort, index_sort = torch.sort(distance_sort, dim=1)

    # Select the max_num_neighbors_threshold neighbors that are closest
    if enforce_max_strictly:
        distance_sort = distance_sort[:, :max_num_neighbors_threshold]
        index_sort = index_sort[:, :max_num_neighbors_threshold]
        max_num_included = max_num_neighbors_threshold

    else:
        effective_cutoff = (
            distance_sort[:, max_num_neighbors_threshold]
            + degeneracy_tolerance
        )
        is_included = torch.le(distance_sort.T, effective_cutoff)

        # Set all undesired edges to infinite length to be removed later
        distance_sort[~is_included.T] = np.inf

        # Subselect tensors for efficiency
        num_included_per_atom = torch.sum(is_included, dim=0)
        max_num_included = torch.max(num_included_per_atom)
        distance_sort = distance_sort[:, :max_num_included]
        index_sort = index_sort[:, :max_num_included]

        # Recompute the number of neighbors
        num_neighbors_thresholded = num_neighbors.clamp(
            max=num_included_per_atom
        )

        num_neighbors_image = segment_csr(
            num_neighbors_thresholded, image_indptr
        )

    # Offset index_sort so that it indexes into index
    index_sort = index_sort + index_neighbor_offset.view(-1, 1).expand(
        -1, max_num_included
    )
    # Remove "unused pairs" with infinite distances
    mask_finite = torch.isfinite(distance_sort)
    index_sort = torch.masked_select(index_sort, mask_finite)

    # At this point index_sort contains the index into index of the
    # closest max_num_neighbors_threshold neighbors per atom
    # Create a mask to remove all pairs not in index_sort
    mask_num_neighbors = torch.zeros(len(index), device=device, dtype=bool)
    mask_num_neighbors.index_fill_(0, index_sort, True)

    return mask_num_neighbors, num_neighbors_image

@torch.jit.export
def get_graph_index(
    nlist: torch.Tensor,
    nlist_mask: torch.Tensor,
    nall: int,
):
    """
    Get the index mapping for edge graph and angle graph, ready in `aggregate` or `index_select`.

    Parameters
    ----------
    nlist : nf x nloc x nnei
        Neighbor list. (padded neis are set to 0)
    nlist_mask : nf x nloc x nnei
        Masks of the neighbor list. real nei 1 otherwise 0
    a_nlist_mask : nf x nloc x a_nnei
        Masks of the neighbor list for angle. real nei 1 otherwise 0
    nall
        The number of extended atoms.

    Returns
    -------
    edge_index : n_edge x 2
        n2e_index : n_edge
            Broadcast indices from node(i) to edge(ij), or reduction indices from edge(ij) to node(i).
        n_ext2e_index : n_edge
            Broadcast indices from extended node(j) to edge(ij).
    angle_index : n_angle x 3
        n2a_index : n_angle
            Broadcast indices from extended node(j) to angle(ijk).
        eij2a_index : n_angle
            Broadcast indices from edge(ij) to angle(ijk), or reduction indices from angle(ijk) to edge(ij).
        eik2a_index : n_angle
            Broadcast indices from edge(ik) to angle(ijk).
    dihedral_index : n_dihedral x 2
        aijk2d_index : n_dihedral
            Broadcast indices from angle(ijk) to dihedral(ijkl), or reduction indices from dihedral(ijkl) to angle(ijk).
        aijl2d_index : n_dihedral
            Broadcast indices from angle(ijl) to dihedral(ijkl).
    """
    nf, nloc, nnei = nlist.shape

    # following: get n2e_index, n_ext2e_index, n2a_index, eij2a_index, eik2a_index

    # 1. atom graph
    # node(i) to edge(ij) index_select; edge(ij) to node aggregate
    nlist_loc_index = torch.arange(0, nf * nloc, dtype=nlist.dtype, device=nlist.device)
    # nf x nloc x nnei
    n2e_index = nlist_loc_index.reshape(nf, nloc, 1).expand(-1, -1, nnei)
    # n_edge
    n2e_index = n2e_index[nlist_mask]  # graph node index, atom_graph[:, 0]

    # node_ext(j) to edge(ij) index_select
    frame_shift = torch.arange(0, nf, dtype=nlist.dtype, device=nlist.device) * nall
    shifted_nlist = nlist + frame_shift[:, None, None]
    # n_edge
    n_ext2e_index = shifted_nlist[nlist_mask]  # graph neighbor index, atom_graph[:, 1]

    return torch.cat([n2e_index.unsqueeze(-1), n_ext2e_index.unsqueeze(-1)], dim=-1)

class EquiformerRMSLayerNorm(nn.Module):
    '''
        Irreps should have same multiplicity, e.g., "16x0e + 16x1o + 16x2e".
        https://github.com/facebookresearch/fairchem/blob/main/src/fairchem/core/models/uma/nn/layer_norm.py
    '''
    def __init__(
        self,
        irreps: Irreps,
        eps: float = 1e-12,
        affine: bool = True,
        normalization: str = "component",
        centering: bool = True,
        std_balance_degrees: bool = True,
    ):
        super().__init__()

        self.irreps = irreps
        self.lmax = irreps.lmax
        self.num_channels = irreps[0].mul
        self.eps = eps
        self.affine = affine
        self.centering = centering
        self.std_balance_degrees = std_balance_degrees

        # for L >= 0
        if self.affine:
            self.affine_weight = nn.Parameter(
                torch.ones((self.lmax + 1), self.num_channels)
            )
            if self.centering:
                self.affine_bias = nn.Parameter(torch.zeros(self.num_channels))
            else:
                self.register_parameter("affine_bias", nn.Parameter(torch.zeros(1)))
        else:
            self.register_parameter("affine_weight", nn.Parameter(torch.zeros(1)))
            self.register_parameter("affine_bias", nn.Parameter(torch.zeros(1)))

        assert normalization in ["norm", "component"]
        self.normalization = normalization

        expand_index = get_l_to_all_m_expand_index(self.lmax)
        self.register_buffer("expand_index", expand_index)

        if self.std_balance_degrees:
            balance_degree_weight = torch.zeros((self.lmax + 1) ** 2, 1)
            for lval in range(self.lmax + 1):
                start_idx = lval**2
                length = 2 * lval + 1
                balance_degree_weight[start_idx : (start_idx + length), :] = (
                    1.0 / length
                )
            balance_degree_weight = balance_degree_weight / (self.lmax + 1)
            self.register_buffer("balance_degree_weight", balance_degree_weight)
        else:
            self.balance_degree_weight = None

    def __repr__(self):
        return f"{self.__class__.__name__}(lmax={self.lmax}, num_channels={self.num_channels}, eps={self.eps}, centering={self.centering}, std_balance_degrees={self.std_balance_degrees})"
    
    def forward(self, node_input: torch.Tensor) -> torch.Tensor:
        '''
            Assume input is of shape [N, sphere_basis, C]
        '''
        node_input = self.reshape_irreps(node_input)

        feature = node_input

        if self.centering:
            feature_l0 = feature.narrow(1, 0, 1)
            feature_l0_mean = feature_l0.mean(dim=2, keepdim=True)  # [N, 1, 1]
            feature_l0 = feature_l0 - feature_l0_mean
            feature = torch.cat(
                (feature_l0, feature.narrow(1, 1, feature.shape[1] - 1)), dim=1
            )

        # for L >= 0
        if self.normalization == "norm":
            assert not self.std_balance_degrees
            feature_norm = feature.pow(2).sum(dim=1, keepdim=True)  # [N, 1, C]
        elif self.normalization == "component":
            if self.std_balance_degrees:
                feature_norm = feature.pow(2)  # [N, (L_max + 1)**2, C]
                feature_norm = torch.einsum(
                    "nic, ia -> nac", feature_norm, self.balance_degree_weight
                )  # [N, 1, C]
            else:
                feature_norm = feature.pow(2).mean(dim=1, keepdim=True)  # [N, 1, C]
        else:
            feature_norm = feature.pow(2).sum(dim=1, keepdim=True)  # [N, 1, C]

        feature_norm = torch.mean(feature_norm, dim=2, keepdim=True)  # [N, 1, 1]
        feature_norm = (feature_norm + self.eps).pow(-0.5)

        if self.affine:
            weight = self.affine_weight.view(
                1, (self.lmax + 1), self.num_channels
            )  # [1, L_max + 1, C]
            weight = torch.index_select(
                weight, dim=1, index=self.expand_index
            )  # [1, (L_max + 1)**2, C]
            feature_norm = feature_norm * weight  # [N, (L_max + 1)**2, C]

        out = feature * feature_norm

        if self.affine and self.centering:
            out[:, 0:1, :] = out.narrow(1, 0, 1) + self.affine_bias.view(
                1, 1, self.num_channels
            )

        out = self.recover_irreps(out)
        return out

    def reshape_irreps(self, input: torch.Tensor) -> torch.Tensor:
        '''
            shape [N, feature] -> [N, sphere_basis, C]
        '''
        out = []
        for l in range(self.lmax + 1):
            feature = input[:, (l ** 2) * self.num_channels : ((l + 1) ** 2) * self.num_channels]
            feature = feature.reshape(-1, self.num_channels, 2 * l + 1).transpose(1, 2)
            out.append(feature)
        out = torch.cat(out, dim=1)
        return out
    
    def recover_irreps(self, input: torch.Tensor) -> torch.Tensor:
        '''
            shape [N, sphere_basis, C] -> [N, feature]
        '''
        out = []
        for l in range(self.lmax + 1):
            feature = input[:, l ** 2 : (l + 1) ** 2]
            feature = feature.transpose(1, 2).flatten(1)
            out.append(feature)
        out = torch.cat(out, dim=1)
        return out


class RMSLayerNorm(nn.Module):
    '''
        Irreps can have different multiplicity, e.g., "16x0e + 8x1o + 4x2e".
    '''
    def __init__(
            self, 
            irreps: Irreps, 
            eps: float = 1e-12, 
            affine: bool = True, 
            centering: bool = True, 
            std_balance_degrees: bool = True
            ) -> None:
        super().__init__()

        self.irreps = irreps
        self.lmax: int = self.irreps.lmax
        self.max_dim: int = max([self.irreps[l].mul for l in range(self.lmax + 1)])
        # self.slices: List[slice] = self.irreps.slices()
        self.slices: List[int] = [0] + [i.stop for i in self.irreps.slices()]
        self.channels: List[int] = [self.irreps[l].mul for l in range(self.lmax + 1)]
        self.eps = eps
        self.affine = affine
        self.centering = centering
        self.std_balance_degrees = std_balance_degrees
        
        self.affine_weight = torch.jit.annotate(Optional[nn.ParameterList], None)
        self.affine_bias = torch.jit.annotate(Optional[nn.Parameter], None)
        if self.affine:
            self.affine_weight = nn.ParameterList()
            for l in range(self.lmax + 1):
                self.affine_weight.append(nn.Parameter(torch.ones(self.irreps[l].mul)))  # [C_L]
            if self.centering:
                self.affine_bias = nn.Parameter(torch.zeros(self.irreps[0].mul))  # [C_0]

        if self.std_balance_degrees:
            balance_degree_weight = torch.zeros((self.lmax + 1) ** 2, 1)
            balance_channel_weight = torch.zeros(self.max_dim, 1)
            for lval in range(self.lmax + 1):
                start_idx = lval**2
                length = 2 * lval + 1
                balance_degree_weight[start_idx : (start_idx + length), :] = 1.0 / length
                balance_channel_weight[:self.irreps[lval].mul, :] += 1
            balance_weight = torch.einsum("ai, bi -> ab", balance_degree_weight, 1 / balance_channel_weight)  # [(L + 1) ** 2, max_dim]
            self.register_buffer("balance_weight", balance_weight)
        else:
            self.balance_weight = None

    def __repr__(self):
        num_params = sum(p.numel() for p in self.parameters())
        return f"{self.__class__.__name__}(irreps={self.irreps}, eps={self.eps}, affine={self.affine}, centering={self.centering}, std_balance_degrees={self.std_balance_degrees}) | params: {num_params}"
    
    def forward(self, node_input: torch.Tensor) -> torch.Tensor:
        '''
            Assume input is of shape [N, sum(2L + 1 * C)]
        '''
        # for L = 0
        if self.centering:
            feature_l0 = node_input[:, self.slices[0] : self.slices[1]]  # [N, C_0]
            feature_l0_mean = feature_l0.mean(dim=1, keepdim=True)  # [N, 1]
            feature_l0 = feature_l0 - feature_l0_mean
            if self.lmax == 0:
                feature = feature_l0
            else:
                feature = torch.cat((feature_l0, node_input[:, self.slices[1]:]), dim=1)  # [N, sum(2L + 1 * C)]
        else:
            feature = node_input

        weights = torch.jit.annotate(Optional[torch.Tensor], None)
        if self.affine:
            weights_list = []
            for l, weight in enumerate(self.affine_weight):
                weight = weight.view(-1, 1).repeat(1, 2 * l + 1).view(1, -1)  # [1, (2L + 1) * C_L]
                weights_list.append(weight)
            weights = torch.cat(weights_list, dim=1)  # [1, sum((2L + 1) * C)]

        # for L >= 0
        feature_list = []
        # num_paddings = []
        for l in range(self.lmax + 1):
            feature_l = feature[:, self.slices[l] : self.slices[l+1]].view(-1, self.channels[l], 2 * l + 1).transpose(1, 2)  # [N, 2L + 1, C_L]
            feature_l = torch.nn.functional.pad(feature_l, (0, self.max_dim - feature_l.shape[2]))  # [N, 2L + 1, max_dim]
            # num_paddings.append(self.max_dim - feature_l.shape[2])
            feature_list.append(feature_l)
        feature_list = torch.cat(feature_list, dim=1)  # [N, (L + 1) ** 2, max_dim]

        if self.std_balance_degrees:
            feature_norm = feature_list.pow(2)  # [N, (L_max + 1)**2, max_dim]
            feature_norm = (feature_norm * self.balance_weight.unsqueeze(0)).sum(dim=1, keepdim=True)  # [N, 1, max_dim]
        else:
            raise NotImplementedError(f"set std_balance_degrees as True.")

        feature_norm = torch.mean(feature_norm, dim=2)  # [N, 1]
        feature_norm = (feature_norm + self.eps).pow(-0.5)

        if weights is not None:
            feature = feature * feature_norm * weights  # [N, sum(2L + 1 * C)]

        if self.affine_bias is not None:
            feature[:, self.slices[0] : self.slices[1]] = feature[:, self.slices[0] : self.slices[1]] + self.affine_bias.view(1, -1)

        return feature
    

class NodewiseGrad(torch.nn.Module):
    def __init__(self, calc_stress: bool = False):
        super().__init__()
        self.calc_stress = calc_stress

    def forward(
        self,
        energy: torch.Tensor, 
        data: dict[str, torch.Tensor], 
        training: bool, 
        ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        forces = torch.jit.annotate(Optional[torch.Tensor], None)
        stress = torch.jit.annotate(Optional[torch.Tensor], None)
        if self.calc_stress:
            grad = torch.autograd.grad(
                [energy.sum()],
                [data['pos'], data['displacement']],
                create_graph=training,
                retain_graph=training,
                )
            forces = grad[0]
            if forces is not None:
                forces = torch.neg(forces)
            stress = grad[1]
            if stress is not None:
                volume = torch.linalg.det(data['cell']).abs().unsqueeze(-1)
                stress = stress / volume.view(len(data['cell']), 1, 1)
                stress = stress.flatten(1, 2)[:, [0, 4, 8, 5, 2, 1]]  # voigt notation
        else:
            grad = torch.autograd.grad(
                [energy.sum()],
                [data['pos']],
                create_graph=training,
                retain_graph=training,
                )
            forces = grad[0]
            if forces is not None:
                forces = torch.neg(forces)

        return forces, stress


class EdgewiseGrad(torch.nn.Module):
    """
    https://github.com/MDIL-SNU/SevenNet/blob/main/sevenn/nn/force_output.py
    """
    def __init__(self, calc_stress: bool = False) -> None:
        super().__init__()
        self.calc_stress = calc_stress

    def forward(
        self,
        energy: torch.Tensor,
        data: dict[str, torch.Tensor],
        training: bool,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        grad = torch.autograd.grad(
            [energy.sum()],
            [data['edge_vec']],
            create_graph=training,
            retain_graph=training,
        )
        fij = grad[0]

        forces = torch.jit.annotate(Optional[torch.Tensor], None)
        stress = torch.jit.annotate(Optional[torch.Tensor], None)
        if fij is not None:
            # pf = torch.zeros(len(data['pos']), 3, dtype=fij.dtype, device=fij.device)
            # nf = torch.zeros(len(data['pos']), 3, dtype=fij.dtype, device=fij.device)
            # pf.index_add_(0, data['edge_index'][0], fij)
            # nf.index_add_(0, data['edge_index'][1], fij)
            pf = scatter(fij, data['edge_index'][0], dim=0, dim_size=len(data['pos']))
            nf = scatter(fij, data['edge_index'][1], dim=0, dim_size=len(data['pos']))
            forces = pf - nf

            # compute stress
            if self.calc_stress:
                diag = data['edge_vec'] * fij
                s12 = data['edge_vec'][..., 0] * fij[..., 1]
                s23 = data['edge_vec'][..., 1] * fij[..., 2]
                s31 = data['edge_vec'][..., 2] * fij[..., 0]
                # cat last dimension
                _virial = torch.cat([
                    diag,
                    s23.unsqueeze(-1),
                    s31.unsqueeze(-1),
                    s12.unsqueeze(-1),
                ], dim=-1)  # voigt notation

                # _s = torch.zeros(len(data['pos']), 6, dtype=fij.dtype, device=fij.device)
                # _s.index_add_(0, data['edge_index'][1], _virial)
                _s = scatter(_virial, data['edge_index'][1], dim=0, dim_size=len(data['pos']))

                # sout = torch.zeros(data['batch'][-1] + 1, 6, dtype=_virial.dtype, device=_virial.device)
                # sout.index_add_(0, data['batch'], _s)
                sout = scatter(_s, data['batch'], dim=0, dim_size=data['batch'][-1] + 1)

                volume = torch.linalg.det(data['cell']).abs().unsqueeze(-1)
                stress = sout / volume

        return forces, stress

def tp_path_exists(
        irreps_in1: Union[str, o3.Irreps], 
        irreps_in2: Union[str, o3.Irreps], 
        ir_out: Union[str, o3.Irreps], 
        ) -> bool:
    irreps_in1 = o3.Irreps(irreps_in1).simplify()
    irreps_in2 = o3.Irreps(irreps_in2).simplify()
    ir_out = o3.Irrep(ir_out)

    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False


def get_path(
        irreps_in1: o3.Irreps, 
        irreps_in2: o3.Irreps, 
        irreps_out: o3.Irreps
        ) -> Tuple[o3.Irreps, list[Tuple[int, int, int, str, bool]]]:
    irreps_mid = []
    instructions = []
    for i, (mul, ir_in) in enumerate(irreps_in1):
        for j, (_, ir_sh) in enumerate(irreps_in2):
            for ir_out in ir_in * ir_sh:
                if ir_out in irreps_out:
                    k = len(irreps_mid)
                    irreps_mid.append((mul, ir_out))
                    instructions.append((i, j, k, "uvu", True))
    irreps_mid = o3.Irreps(irreps_mid)
    irreps_mid, p, _ = irreps_mid.sort()

    # Permute the output indexes of the instructions to match the sorted irreps:
    instructions = [
        (i_in1, i_in2, p[i_out], mode, train)
        for i_in1, i_in2, i_out, mode, train in instructions
    ]
    return irreps_mid, instructions


class E3NN(torch.nn.Module):
    def __init__(
            self,
            irreps_hidden: Union[str, o3.Irreps],  # node hidden representation
            irreps_sh: Union[str, o3.Irreps],  # edge spherical harmonics
            num_conv_layers: int = 5,  # number of convolution layers
            num_types: int = 4,  # number of node atom types
            num_features: int = 128,  # number of features for node embedding
            max_radius: int = 6.0,  # cutoff radius
            num_basis: int = 8,   # number of Bessel basis functions
            invariant_layers: int = 2,  # number of radial layers
            invariant_neurons: int = 64,  # number of hidden neurons in radial function
            poly_p: int = 6,  # polynomial cutoff
            nonlinearity_type: str = "gate",  # nonlinearity type for invariant and equivariant layers
            avg_num_neighbors: Optional[float] = None,  # average number of neighbors
            ) -> None:
        super().__init__()
        self.irreps_hidden = o3.Irreps(irreps_hidden)
        self.irreps_sh = o3.Irreps(irreps_sh)

        self.num_types = num_types
        self.num_features = num_features
        self.scalar_features = o3.Irreps(f"{self.num_features}x0e")

        self.max_radius = max_radius
        self.num_basis = num_basis
        self.invariant_layers = invariant_layers
        self.invariant_neurons = invariant_neurons
        self.poly_p = poly_p
        
        self.embedding = Embedding_layer(num_types=self.num_types+1, num_features=self.num_features)

        self.sph = o3.SphericalHarmonics(self.irreps_sh, normalize=True, normalization='component')

        self.besssel_basis = BesselBasis(r_max=self.max_radius, num_basis=self.num_basis)
        self.poly_cutoff = PolynomialCutoff(r_max=self.max_radius, p=self.poly_p)

        self.nonlinearity_scalars = {
            1: torch.nn.functional.silu,
            -1: torch.tanh,
        }
        self.nonlinearity_gates = {
            1: torch.nn.functional.silu,
            -1: torch.tanh,
        }

        self.num_conv_layers = num_conv_layers
        self.layers = torch.nn.ModuleList()
        self.irreps_in = self.scalar_features
        for num_conv_layer in range(self.num_conv_layers):
            irreps_scalars = o3.Irreps()
            irreps_gate_scalars = o3.Irreps()
            irreps_nonscalars = o3.Irreps()

            # get scalar target irreps
            for multiplicity, irrep in self.irreps_hidden:
                if o3.Irrep(irrep).l == 0 and tp_path_exists(self.irreps_in, self.irreps_sh, irrep):
                    irreps_scalars += [(multiplicity, irrep)]
                irreps_scalars = o3.Irreps(irreps_scalars)

            # get non-scalar target irreps
            for multiplicity, irrep in self.irreps_hidden:
                if o3.Irrep(irrep).l > 0 and tp_path_exists(
                    self.irreps_in, self.irreps_sh, irrep
                ):
                    irreps_nonscalars += [(multiplicity, irrep)]
                irreps_nonscalars = o3.Irreps(irreps_nonscalars)

            # get gate scalar irreps
            if tp_path_exists(self.irreps_in, self.irreps_sh, '0e'):
                gate_scalar_irreps_type = '0e'
            else:
                gate_scalar_irreps_type = '0o'

            for multiplicity, _ in irreps_nonscalars:
                irreps_gate_scalars += [(multiplicity, gate_scalar_irreps_type)]
            irreps_gate_scalars = o3.Irreps(irreps_gate_scalars).simplify()

            # final layer output irreps are all three
            self.irreps_out = irreps_scalars + irreps_gate_scalars + irreps_nonscalars
            self.irreps_out = self.irreps_out.sort().irreps.simplify()

            self.layers.append(E3Conv(
                self.irreps_in, 
                self.irreps_sh, 
                self.irreps_out, 
                num_basis=self.num_basis,
                invariant_layers=self.invariant_layers,
                invariant_neurons=self.invariant_neurons,
                avg_num_neighbors=avg_num_neighbors,
                mode="nonscalar",
                ))
            self.irreps_in = irreps_scalars + irreps_nonscalars

            if nonlinearity_type == "gate":
                self.layers.append(EquivariantGate(
                    irreps_scalars=irreps_scalars,
                    act_scalars=[self.nonlinearity_scalars[ir.p] for _, ir in irreps_scalars],
                    irreps_gates=irreps_gate_scalars,
                    act_gates=[self.nonlinearity_gates[ir.p] for _, ir in irreps_gate_scalars],
                    irreps_gated=irreps_nonscalars,
                ))
            else:
                raise NotImplementedError(f"nonlinearity_type {nonlinearity_type} not implemented.")

            self.layers.append(E3Conv(
                self.irreps_in, 
                self.irreps_sh, 
                self.scalar_features, 
                num_basis=self.num_basis,
                invariant_layers=self.invariant_layers,
                invariant_neurons=self.invariant_neurons,
                avg_num_neighbors=avg_num_neighbors,
                mode="scalar",
                ))
            
            self.layers.append(NonlinearAndAdd())
           
        self.output_block = FullyConnectedNet(
            [self.num_features]
            + [self.num_features // 2]
            + [1],
            self.nonlinearity_scalars[1],
        )

    def forward(
            self, 
            data: dict[str, torch.Tensor],
            ) -> torch.Tensor:
        data['node_hiddens'] = self.embedding(data['atomic_numbers'])
        data['output'] = data['node_hiddens']
        
        distance = torch.norm(data['edge_vec'], p=2, dim=-1)
        # data['edge_sh'] = o3.spherical_harmonics(self.irreps_sh, data['edge_vec'], normalize=True, normalization='component')
        data['edge_sh'] = self.sph(data['edge_vec'])

        data['edge_attr'] = self.besssel_basis(distance)
        cutoff = self.poly_cutoff(distance).unsqueeze(-1)
        data['edge_attr'] = data['edge_attr'] * cutoff
        
        # print("otuput", -1, data['output'][:, :128].pow(2).mean(dim=-1, keepdim=True).pow(0.5).mean())
        for layer_idx, layer in enumerate(self.layers):
            layer(data)
            # print("otuput", layer_idx, data['output'][:, :128].pow(2).mean(dim=-1, keepdim=True).pow(0.5).mean())
            # print("hidden", layer_idx, data['node_hiddens'][:, :128].pow(2).mean(dim=-1, keepdim=True).pow(0.5).mean())
            # print("hidden", layer_idx, data['node_hiddens'][:, 128:].pow(2).mean(dim=-1, keepdim=True).pow(0.5).mean())
        
        # energy = self.output_block(data['output'])

        # print("otuput", 100, data['output'][:, :128].pow(2).mean(dim=-1, keepdim=True).pow(0.5).mean())

        return data['output']


class EquivariantGate(torch.nn.Module):
    def __init__(
            self, 
            irreps_scalars: Union[str, o3.Irreps],
            act_scalars: List[Callable],
            irreps_gates: Union[str, o3.Irreps],
            act_gates: List[Callable],
            irreps_gated: Union[str, o3.Irreps],
            ) -> None:
        super().__init__()
        self.gate = Gate(
            irreps_scalars=irreps_scalars,
            act_scalars=act_scalars,
            irreps_gates=irreps_gates,
            act_gates=act_gates,
            irreps_gated=irreps_gated,
        )

    def forward(
            self, 
            data: dict[str, torch.Tensor], 
            ) -> None:
        data['node_hiddens'] = self.gate(data['node_hiddens'])
        return None


class NonlinearAndAdd(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.nonlinear = torch.nn.SiLU()

    def forward(
            self, 
            data: dict[str, torch.Tensor], 
            ) -> None:
        data['output'] = data['output'] + self.nonlinear(data['node_scalars'])
        return None


class Embedding_layer(torch.nn.Module):
    def __init__(
            self, 
            num_types: int, 
            num_features: int, 
            ) -> None:
        super().__init__()
        self.num_types = num_types
        self.num_features = num_features
        self.linear = Linear(o3.Irreps(f"{self.num_types}x0e"), o3.Irreps(f"{self.num_features}x0e"))
    
    def forward(
            self, 
            atomic_numbers: torch.Tensor, 
            ) -> torch.Tensor:
        
        one_hot = torch.nn.functional.one_hot(atomic_numbers, num_classes=self.num_types).float()
        return self.linear(one_hot)


class BesselBasis(torch.nn.Module):
    def __init__(
            self, 
            r_max: float, 
            num_basis: int = 8, 
            trainable: bool = True,
            ) -> None:
        super().__init__()

        self.trainable = trainable
        self.num_basis = num_basis
        self.r_max = float(r_max)
        self.prefactor = 2.0 / self.r_max

        bessel_weights = torch.linspace(start=1.0, end=num_basis, steps=num_basis) * torch.math.pi
        if self.trainable:
            self.bessel_weights = torch.nn.Parameter(bessel_weights)
        else:
            self.register_buffer("bessel_weights", bessel_weights)

    def forward(
            self, 
            x: torch.Tensor
            ) -> torch.Tensor:
        numerator = torch.sin(self.bessel_weights * x.unsqueeze(-1) / self.r_max)
        return self.prefactor * (numerator / x.unsqueeze(-1))


class PolynomialCutoff(torch.nn.Module):
    def __init__(
            self, 
            r_max: float, 
            p: float = 6,
            ) -> None:
        super().__init__()
        assert p >= 2.0
        self.p = float(p)
        self._factor = 1.0 / float(r_max)

    def poly_cutoff(
            self, 
            x: torch.Tensor, 
            factor: float, 
            p: float = 6.0
            ) -> torch.Tensor:
        x = x * factor
        out = 1.0
        out = out - (((p + 1.0) * (p + 2.0) / 2.0) * torch.pow(x, p))
        out = out + (p * (p + 2.0) * torch.pow(x, p + 1.0))
        out = out - ((p * (p + 1.0) / 2) * torch.pow(x, p + 2.0))
        return out * (x < 1.0)

    def forward(self, x):
        return self.poly_cutoff(x, self._factor, p=self.p)


class E3Conv(torch.nn.Module):
    def __init__(
            self,
            irreps_hidden: Union[str, o3.Irreps],
            irreps_sh: Union[str, o3.Irreps],
            irreps_out: Union[str, o3.Irreps],
            num_basis: int = 8,
            invariant_layers: int = 2,
            invariant_neurons: int = 64,
            avg_num_neighbors: Optional[float] = None,
            use_sc: bool = True,
            nonlinearity_scalars: Dict[str, Callable] = {"e": torch.nn.functional.silu}, 
            mode: Literal["nonscalar", "scalar"] = "nonscalar",
            ) -> None:
        super().__init__()

        self.irreps_hidden = o3.Irreps(irreps_hidden)
        self.irreps_sh = o3.Irreps(irreps_sh)
        self.irreps_out = o3.Irreps(irreps_out)

        self.avg_num_neighbors = avg_num_neighbors
        self.use_sc = use_sc
        self.mode = mode

        self.linear_1 = Linear(
            irreps_in=self.irreps_hidden,
            irreps_out=self.irreps_hidden,
        )

        irreps_mid, instructions = get_path(
            self.irreps_hidden, 
            self.irreps_sh, 
            self.irreps_out
            )
        self.tp = TensorProduct(
            self.irreps_hidden,
            self.irreps_sh,
            irreps_mid,
            instructions,
            internal_weights=False,
            shared_weights=False,
        )

        self.fc = FullyConnectedNet(
            [num_basis]
            + invariant_layers * [invariant_neurons]
            + [self.tp.weight_numel],
            nonlinearity_scalars['e'],
        )

        self.linear_2 = Linear(
            irreps_in=irreps_mid.simplify(),
            irreps_out=self.irreps_out,
        )

        self.sc = None
        if self.use_sc:
            self.sc = Linear(
                self.irreps_hidden,
                self.irreps_out,
            )

        self.ln = RMSLayerNorm(self.irreps_out, centering=False)

    def forward(
            self,
            data: dict[str, torch.Tensor],
            ) -> None:
        weight = self.fc(data['edge_attr'])
        edge_src = data['edge_index'][0]
        edge_dst = data['edge_index'][1]

        if self.sc is not None:
            sc = self.sc(data['node_hiddens'])

        node_hiddens = self.linear_1(data['node_hiddens'])
        
        edge_features = self.tp(
            node_hiddens[edge_dst], data['edge_sh'], weight
        )
        if self.avg_num_neighbors is not None:
            edge_features = edge_features.div(self.avg_num_neighbors ** 0.5)
        # node_hiddens = torch.zeros(len(node_hiddens), edge_features.shape[-1], device=edge_features.device, dtype=edge_features.dtype)
        # node_hiddens.index_add_(0, edge_src, edge_features)
        node_hiddens = scatter(edge_features, edge_src, dim=0, dim_size=len(node_hiddens))

        node_hiddens = self.linear_2(node_hiddens)

        if self.sc is not None:
            node_hiddens = node_hiddens + sc
            node_hiddens = self.ln(node_hiddens)

        if self.mode == "nonscalar":
            data['node_hiddens'] = node_hiddens
        elif self.mode == "scalar":
            data['node_scalars'] = node_hiddens
        else:
            raise NotImplementedError(f"mode {self.mode} not implemented.")
        
        return None

@BaseDescriptor.register("eqnorm")
class Eqnorm(BaseDescriptor, torch.nn.Module):
    def __init__(
            self, 
            ntypes: int,
            eqnorm: Union[EqnormArgs, dict], 
            shift: Optional[torch.Tensor] = None, 
            scale: Optional[torch.Tensor] = None, 
            type_map: Optional[list[str]] = None,
            ) -> None:
        super().__init__()
        self.ntypes = ntypes
        self.type_map = type_map
        self.type_map_index = torch.tensor([element_dict[type] for type in type_map])
        self.skip_stat = True
        self.set_davg_zero = True
        self.hidden_dim = eqnorm["num_features"]
        self.r_cutoff = eqnorm["r_cutoff"]  # A
        self.num_types = 94
        self.shift = shift
        self.scale = scale
        self.grad_mode = eqnorm["grad_mode"]

        if eqnorm["shift_trainable"] and self.shift is not None:
            self.shift = torch.nn.Parameter(self.shift)
        if eqnorm["scale_trainable"] and self.scale is not None:
            self.scale = torch.nn.Parameter(self.scale)

        self.calc_stress = eqnorm["STRESS"]
        self.calc_dipole = eqnorm["DIPOLE"]
        self.calc_polar = eqnorm["POLAR"]

        self.e3nn_layer = E3NN(
            irreps_hidden=eqnorm["irreps_hidden"], 
            irreps_sh=eqnorm["irreps_sh"],
            num_conv_layers=eqnorm["num_convs"], 
            num_types=self.num_types, 
            num_features=eqnorm["num_features"],
            max_radius=self.r_cutoff, 
            num_basis=eqnorm["num_basis"], 
            invariant_layers=eqnorm["invariant_layers"],
            invariant_neurons=eqnorm["invariant_neurons"],
            poly_p=eqnorm["poly_p"],
            avg_num_neighbors=eqnorm["avg_nbr"],
            )
        
        if self.grad_mode == 'edge':
            self.force_stress_output = EdgewiseGrad(calc_stress=self.calc_stress)
        elif self.grad_mode == 'node':
            self.force_stress_output = NodewiseGrad(calc_stress=self.calc_stress)
        else:
            raise NotImplementedError(f"grad_mode {self.grad_mode} not implemented.")
        
    def __setitem__(self, key, value) -> None:
        if key in ("avg", "data_avg", "davg"):
            self.mean = value
        elif key in ("std", "data_std", "dstd"):
            self.stddev = value
        else:
            raise KeyError(key)

    def __getitem__(self, key):
        if key in ("avg", "data_avg", "davg"):
            return self.mean
        elif key in ("std", "data_std", "dstd"):
            return self.stddev
        else:
            raise KeyError(key)


    def forward(
            self, 
            coord: torch.Tensor,
            extended_coord: torch.Tensor,
            extended_atype: torch.Tensor,
            nlist: torch.Tensor,
            box: Optional[torch.Tensor] = None,
            mapping: Optional[torch.Tensor] = None,
            comm_dict: Optional[dict[str, torch.Tensor]] = None,
            data: dict[str, torch.Tensor] = None, 
            training: bool = True,  # determine whether to create graph for autograd
            ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], None, None]:
        device = extended_coord.device
        extended_coord = extended_coord.to(torch.float32)
        nframes, nloc, nnei = nlist.shape
        nall = extended_coord.view(nframes, -1).shape[1] // 3
        e_sel = 1200
        coord = extended_coord[:,:nloc,:]
        atype = extended_atype[:, :nloc]
        # nb x nloc x nnei x 4, nb x nloc x nnei x 3, nb x nloc x nnei x 1
        
        dmatrix, diff, sw = prod_env_mat(
            extended_coord,
            nlist,
            atype,
            mean=None,
            stddev=None,
            rcut=6.0,
            rcut_smth=5.3,
            normalize=False,
        )
        edge_input, h2 = torch.split(dmatrix, [1, 3], dim=-1)
        
        nlist_mask = nlist != -1
        
        edge_index = get_graph_index(
            nlist,
            nlist_mask,
            nall,
        )
        
        edge_diff = diff[nlist_mask]
        edge_input = edge_input[nlist_mask]
        # n_edge x 3
        h2 = h2[nlist_mask]
        # n_edge
        sw = sw[nlist_mask]
        # n_edge x 4
        dmatrix = dmatrix[nlist_mask]

        edge_index = edge_index.transpose(0, 1)
        edge_index[1] = edge_index[1] % nloc
        batch = torch.repeat_interleave(torch.arange(nframes, device=device), nloc)
        data = {
            'pos': coord.reshape(-1, 3),
            'edge_index': edge_index,
            'edge_vec': edge_diff,
            'atomic_numbers': self.type_map_index[atype.cpu().view(-1)].to(device),
            'batch': batch,
        }
        
        energy = self.e3nn_layer(data).reshape(nframes, nloc, -1)
        

        return energy, None, None, None, None


    @classmethod
    def update_sel(
        cls,
        train_data: DeepmdDataSystem,
        type_map: Optional[list[str]],
        local_jdata: dict,
    ) -> tuple[dict, Optional[float]]:
        """Update the selection and perform neighbor statistics.

        Parameters
        ----------
        train_data : DeepmdDataSystem
            data used to do neighbor statistics
        type_map : list[str], optional
            The name of each type of atoms
        local_jdata : dict
            The local data refer to the current class

        Returns
        -------
        dict
            The updated local data
        float
            The minimum distance between two atoms
        """
        local_jdata_cpy = local_jdata.copy()
        update_sel = UpdateSel()
        min_nbor_dist, repflow_e_sel = update_sel.update_one_sel(
            train_data,
            type_map,
            local_jdata_cpy["repflow"]["e_rcut"],
            local_jdata_cpy["repflow"]["e_sel"],
            True,
        )
        local_jdata_cpy["repflow"]["e_sel"] = repflow_e_sel[0]

        min_nbor_dist, repflow_a_sel = update_sel.update_one_sel(
            train_data,
            type_map,
            local_jdata_cpy["repflow"]["a_rcut"],
            local_jdata_cpy["repflow"]["a_sel"],
            True,
        )
        local_jdata_cpy["repflow"]["a_sel"] = repflow_a_sel[0]

        return local_jdata_cpy, min_nbor_dist

    def enable_compression(
        self,
        min_nbor_dist: float,
        table_extrapolate: float = 5,
        table_stride_1: float = 0.01,
        table_stride_2: float = 0.1,
        check_frequency: int = -1,
    ) -> None:
        """Receive the statistics (distance, max_nbor_size and env_mat_range) of the training data.

        Parameters
        ----------
        min_nbor_dist
            The nearest distance between atoms
        table_extrapolate
            The scale of model extrapolation
        table_stride_1
            The uniform stride of the first table
        table_stride_2
            The uniform stride of the second table
        check_frequency
            The overflow check frequency
        """
        raise NotImplementedError("Compression is unsupported for DPA3.")

    def get_rcut(self) -> float:
        """Returns the cut-off radius."""
        return self.r_cutoff

    def get_rcut_smth(self) -> float:
        """Returns the radius where the neighbor information starts to smoothly decay to 0."""
        return self.rcut_smth

    def get_nsel(self) -> int:
        """Returns the number of selected atoms in the cut-off radius."""
        return sum(self.sel)

    def get_sel(self) -> list[int]:
        """Returns the number of selected atoms for each type."""
        return [1200]

    def get_ntypes(self) -> int:
        """Returns the number of element types."""
        return self.ntypes

    def get_type_map(self) -> list[str]:
        """Get the name to each type of atoms."""
        return self.type_map

    def get_dim_out(self) -> int:
        """Returns the output dimension of this descriptor."""
        return self.hidden_dim

    def get_dim_emb(self) -> int:
        """Returns the embedding dimension of this descriptor."""
        return self.hidden_dim


    def mixed_types(self) -> bool:
        """If true, the descriptor
        1. assumes total number of atoms aligned across frames;
        2. requires a neighbor list that does not distinguish different atomic types.

        If false, the descriptor
        1. assumes total number of atoms of each atom type aligned across frames;
        2. requires a neighbor list that distinguishes different atomic types.

        """
        return True

    def has_message_passing(self) -> bool:
        """Returns whether the descriptor has message passing."""
        return self.repflows.has_message_passing()

    def need_sorted_nlist_for_lower(self) -> bool:
        """Returns whether the descriptor needs sorted nlist when using `forward_lower`."""
        return True

    def get_env_protection(self) -> float:
        """Returns the protection of building environment matrix."""
        return self.repflows.get_env_protection()

    def share_params(self, base_class, shared_level, resume=False) -> None:
        """
        Share the parameters of self to the base_class with shared_level during multitask training.
        If not start from checkpoint (resume is False),
        some separated parameters (e.g. mean and stddev) will be re-calculated across different classes.
        """
        assert (
            self.__class__ == base_class.__class__
        ), "Only descriptors of the same type can share params!"
        # For DPA3 descriptors, the user-defined share-level
        # shared_level: 0
        # share all parameters in type_embedding, repflow
        if shared_level == 0:
            self._modules["type_embedding"] = base_class._modules["type_embedding"]
            self.repflows.share_params(base_class.repflows, 0, resume=resume)
        # shared_level: 1
        # share all parameters in type_embedding
        elif shared_level == 1:
            self._modules["type_embedding"] = base_class._modules["type_embedding"]
        # Other shared levels
        else:
            raise NotImplementedError

    def change_type_map(
        self, type_map: list[str], model_with_new_type_stat=None
    ) -> None:
        """Change the type related params to new ones, according to `type_map` and the original one in the model.
        If there are new types in `type_map`, statistics will be updated accordingly to `model_with_new_type_stat` for these new types.
        """
        assert (
            self.type_map is not None
        ), "'type_map' must be defined when performing type changing!"
        remap_index, has_new_type = get_index_between_two_maps(self.type_map, type_map)
        self.type_map = type_map
        self.type_embedding.change_type_map(type_map=type_map)
        self.exclude_types = map_pair_exclude_types(self.exclude_types, remap_index)
        self.ntypes = len(type_map)
        repflow = self.repflows
        if has_new_type:
            # the avg and std of new types need to be updated
            extend_descrpt_stat(
                repflow,
                type_map,
                des_with_stat=model_with_new_type_stat.repflows
                if model_with_new_type_stat is not None
                else None,
            )
        repflow.ntypes = self.ntypes
        repflow.reinit_exclude(self.exclude_types)
        repflow["davg"] = repflow["davg"][remap_index]
        repflow["dstd"] = repflow["dstd"][remap_index]

    @property
    def dim_out(self):
        return self.get_dim_out()

    @property
    def dim_emb(self):
        """Returns the embedding dimension g2."""
        return self.get_dim_emb()

    def compute_input_stats(
        self,
        merged: Union[Callable[[], list[dict]], list[dict]],
        path: Optional[DPPath] = None,
    ) -> None:
        """
        Compute the input statistics (e.g. mean and stddev) for the descriptors from packed data.

        Parameters
        ----------
        merged : Union[Callable[[], list[dict]], list[dict]]
            - list[dict]: A list of data samples from various data systems.
                Each element, `merged[i]`, is a data dictionary containing `keys`: `torch.Tensor`
                originating from the `i`-th data system.
            - Callable[[], list[dict]]: A lazy function that returns data samples in the above format
                only when needed. Since the sampling process can be slow and memory-intensive,
                the lazy function helps by only sampling once.
        path : Optional[DPPath]
            The path to the stat file.

        """
        if self.skip_stat and self.set_davg_zero:
            return
        env_mat_stat = EnvMatStatSe(self)
        if path is not None:
            path = path / env_mat_stat.get_hash()
        if path is None or not path.is_dir():
            if callable(merged):
                # only get data for once
                sampled = merged()
            else:
                sampled = merged
        else:
            sampled = []
        env_mat_stat.load_or_compute_stats(sampled, path)
        self.stats = env_mat_stat.stats
        mean, stddev = env_mat_stat()
        if not self.set_davg_zero:
            self.mean.copy_(
                torch.tensor(mean, device=env.DEVICE, dtype=self.mean.dtype)
            )
        self.stddev.copy_(
            torch.tensor(stddev, device=env.DEVICE, dtype=self.stddev.dtype)
        )

    def set_stat_mean_and_stddev(
        self,
        mean: list[torch.Tensor],
        stddev: list[torch.Tensor],
    ) -> None:
        pass

    def get_stat_mean_and_stddev(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Get mean and stddev for descriptor."""
        mean_list = [self.repflows.mean]
        stddev_list = [self.repflows.stddev]
        return mean_list, stddev_list

    def serialize(self) -> dict:
        repflows = self.repflows
        data = {
            "@class": "Descriptor",
            "type": "dpa3",
            "@version": 1,
            "ntypes": self.ntypes,
            "repflow_args": self.repflow_args.serialize(),
            "concat_output_tebd": self.concat_output_tebd,
            "activation_function": self.activation_function,
            "precision": self.precision,
            "exclude_types": self.exclude_types,
            "env_protection": self.env_protection,
            "trainable": self.trainable,
            "use_econf_tebd": self.use_econf_tebd,
            "use_tebd_bias": self.use_tebd_bias,
            "type_map": self.type_map,
            "type_embedding": self.type_embedding.embedding.serialize(),
        }
        repflow_variable = {
            "edge_embd": repflows.edge_embd.serialize(),
            "angle_embd": repflows.angle_embd.serialize(),
            "repflow_layers": [layer.serialize() for layer in repflows.layers],
            "env_mat": DPEnvMat(repflows.rcut, repflows.rcut_smth).serialize(),
            "@variables": {
                "davg": to_numpy_array(repflows["davg"]),
                "dstd": to_numpy_array(repflows["dstd"]),
            },
        }
        data.update(
            {
                "repflow_variable": repflow_variable,
            }
        )
        return data

    @classmethod
    def deserialize(cls, data: dict) -> "Eqnorm":
        data = data.copy()
        version = data.pop("@version")
        check_version_compatibility(version, 1, 1)
        data.pop("@class")
        data.pop("type")
        repflow_variable = data.pop("repflow_variable").copy()
        type_embedding = data.pop("type_embedding")
        data["repflow"] = EqnormArgs(**data.pop("repflow_args"))
        obj = cls(**data)
        obj.type_embedding.embedding = TypeEmbedNetConsistent.deserialize(
            type_embedding
        )

        def t_cvt(xx):
            return torch.tensor(xx, dtype=obj.repflows.prec, device=env.DEVICE)

        # deserialize repflow
        statistic_repflows = repflow_variable.pop("@variables")
        env_mat = repflow_variable.pop("env_mat")
        repflow_layers = repflow_variable.pop("repflow_layers")
        obj.repflows.edge_embd = MLPLayer.deserialize(repflow_variable.pop("edge_embd"))
        obj.repflows.angle_embd = MLPLayer.deserialize(
            repflow_variable.pop("angle_embd")
        )
        obj.repflows["davg"] = t_cvt(statistic_repflows["davg"])
        obj.repflows["dstd"] = t_cvt(statistic_repflows["dstd"])
        obj.repflows.layers = torch.nn.ModuleList(
            [RepFlowLayer.deserialize(layer) for layer in repflow_layers]
        )
        return obj