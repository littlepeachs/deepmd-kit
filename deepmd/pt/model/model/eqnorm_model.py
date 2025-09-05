
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
# SPDX-License-Identifier: LGPL-3.0-or-later
import vesin
import functools

from deepmd.pt.model.atomic_model import (
    BaseAtomicModel,
)
from deepmd.pt.model.model.model import (
    BaseModel,
)
import numpy as np
from .dp_model import (
    DPModelCommon,
)
from .make_model import (
    make_model,
)
from deepmd.pt.utils.stat import (
    compute_output_stats,
)
import copy
from deepmd.utils.path import (
    DPPath,
)
from deepmd.dpmodel.output_def import (
    FittingOutputDef,
    ModelOutputDef,
    OutputVariableDef,
)
from deepmd.pt.utils import (
    env,
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

energy_shift = torch.tensor([[ -3.6671],
        [ -1.3228],
        [ -3.4821],
        [ -4.7372],
        [ -7.7249],
        [ -8.4056],
        [ -7.3601],
        [ -7.2847],
        [ -4.8965],
        [ -0.0296],
        [ -2.7592],
        [ -2.8129],
        [ -4.8468],
        [ -7.6949],
        [ -6.9631],
        [ -4.6726],
        [ -2.8117],
        [ -0.0626],
        [ -2.6177],
        [ -5.3902],
        [ -7.8857],
        [-10.2688],
        [ -8.6651],
        [ -9.2331],
        [ -8.3050],
        [ -7.0489],
        [ -5.5771],
        [ -5.1727],
        [ -3.2520],
        [ -1.2902],
        [ -3.5271],
        [ -4.7087],
        [ -3.9764],
        [ -3.8863],
        [ -2.5185],
        [  6.7582],
        [ -2.5634],
        [ -4.9376],
        [-10.1497],
        [-11.8468],
        [-12.1389],
        [ -8.7916],
        [ -8.7871],
        [ -7.7809],
        [ -6.8499],
        [ -4.8909],
        [ -2.0635],
        [ -0.6396],
        [ -2.7887],
        [ -3.8187],
        [ -3.5871],
        [ -2.8804],
        [ -1.6356],
        [  9.8438],
        [ -2.7655],
        [ -4.9909],
        [ -8.9338],
        [ -8.7354],
        [ -8.0189],
        [ -8.2511],
        [ -7.5917],
        [ -8.1698],
        [-13.5947],
        [-18.5173],
        [ -7.6474],
        [ -8.1226],
        [ -7.6076],
        [ -6.8502],
        [ -7.8269],
        [ -3.5847],
        [ -7.4553],
        [-12.7963],
        [-14.1081],
        [ -9.3548],
        [-11.3875],
        [ -9.6218],
        [ -7.3245],
        [ -5.3047],
        [ -2.3802],
        [  0.2495],
        [ -2.3242],
        [ -3.7299],
        [ -3.4388],
        [ -5.0627],
        [-11.0234],
        [-12.2584],
        [-13.8556],
        [-14.9203],
        [-15.2824],
        [-15.2824],
        [-15.2824],
        [-15.2824],
        [-15.2824],
        [-15.2824],
        [-15.2824]])

energy_scale = torch.tensor(0.8080)

dtype = env.GLOBAL_PT_FLOAT_PRECISION
device = env.DEVICE


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
        
        energy = self.output_block(data['output'])

        # print("otuput", 100, data['output'][:, :128].pow(2).mean(dim=-1, keepdim=True).pow(0.5).mean())

        return energy


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

@BaseModel.register("eqnorm")
class Eqnorm(BaseModel):
    def __init__(
            self, 
            model_params,
            **kwargs,
            ) -> None:
        
        super().__init__(**kwargs)
        self.type_map = model_params["type_map"]
        self.type_map_index = torch.tensor([element_dict[type] for type in self.type_map])
        # self.num_types = self.type_map_index.max() + 1
        self.num_types = 93
        self.numb_fparam = 0

        ntypes = self.get_ntypes()
        
        model_config_copy = copy.deepcopy(model_params["eqnorm"])
        self.default_fparam = model_config_copy.pop("default_fparam", [0.0, 1.0])
        
        # self.max_out_size = 1
        # self.bias_keys: list[str] = ["energy"]
        # self.n_out = len(self.bias_keys)
        # out_bias_data = torch.zeros(
        #     [self.n_out, ntypes, self.max_out_size], dtype=dtype, device=device
        # )
        # out_std_data = torch.ones(
        #     [self.n_out, ntypes, self.max_out_size], dtype=dtype, device=device
        # )
        # self.register_buffer("out_bias", out_bias_data)
        # self.register_buffer("out_std", out_std_data)
        # self.register_buffer(
        #     "default_fparam_tensor",
        #     torch.tensor(
        #         np.array(self.default_fparam), dtype=dtype, device=device
        #     ),
        # )

        self.hidden_dim = model_params["eqnorm"]["num_features"]
        self.r_cutoff = model_params["eqnorm"]["r_cutoff"]  # A
        
        self.shift = energy_shift
        self.scale = energy_scale

        self.grad_mode = model_params["eqnorm"]["grad_mode"]
        
        if model_params["eqnorm"]["shift_trainable"] and self.shift is not None:
            self.shift = torch.nn.Parameter(self.shift)
        if model_params["eqnorm"]["scale_trainable"] and self.scale is not None:
            self.scale = torch.nn.Parameter(self.scale)

        self.calc_stress = model_params["eqnorm"]["STRESS"]
        self.calc_dipole = model_params["eqnorm"]["DIPOLE"]
        self.calc_polar = model_params["eqnorm"]["POLAR"]

        self.e3nn_layer = E3NN(
            irreps_hidden=model_params["eqnorm"]["irreps_hidden"], 
            irreps_sh=model_params["eqnorm"]["irreps_sh"],
            num_conv_layers=model_params["eqnorm"]["num_convs"], 
            num_types=self.num_types, 
            num_features=model_params["eqnorm"]["num_features"],
            max_radius=self.r_cutoff, 
            num_basis=model_params["eqnorm"]["num_basis"], 
            invariant_layers=model_params["eqnorm"]["invariant_layers"],
            invariant_neurons=model_params["eqnorm"]["invariant_neurons"],
            poly_p=model_params["eqnorm"]["poly_p"],
            avg_num_neighbors=model_params["eqnorm"]["avg_nbr"],
            )
        
        if self.grad_mode == 'edge':
            self.force_stress_output = EdgewiseGrad(calc_stress=self.calc_stress)
        elif self.grad_mode == 'node':
            self.force_stress_output = NodewiseGrad(calc_stress=self.calc_stress)
        else:
            raise NotImplementedError(f"grad_mode {self.grad_mode} not implemented.")

        self.calculator = vesin.NeighborList(cutoff=self.r_cutoff, full_list=True, sorted=False)
        
    def get_type_map(self) -> list[str]:
        """Get the type map."""
        return self.type_map

    def compute_or_load_stat(
        self,
        sampled_func,  # noqa: ANN001
        stat_file_path: Optional[DPPath] = None,
    ) -> None:
        """Compute or load the statistics parameters of the model.

        For example, mean and standard deviation of descriptors or the energy bias of
        the fitting net. When `sampled` is provided, all the statistics parameters will
        be calculated (or re-calculated for update), and saved in the
        `stat_file_path`(s). When `sampled` is not provided, it will check the existence
        of `stat_file_path`(s) and load the calculated statistics parameters.

        Parameters
        ----------
        sampled_func
            The sampled data frames from different data systems.
        stat_file_path
            The path to the statistics files.
        """
        if stat_file_path is not None and self.type_map is not None:
            # descriptors and fitting net with different type_map
            # should not share the same parameters
            stat_file_path /= " ".join(self.type_map)

        @functools.lru_cache
        def wrapped_sampler():
            sampled = sampled_func()
            return sampled

        self.compute_or_load_out_stat(wrapped_sampler, stat_file_path)

    def compute_or_load_out_stat(
        self,
        merged: Union[Callable[[], list[dict]], list[dict]],
        stat_file_path: Optional[DPPath] = None,
    ) -> None:
        """
        Compute the output statistics (e.g. energy bias) for the fitting net from packed data.

        Parameters
        ----------
        merged : Union[Callable[[], list[dict]], list[dict]]
            - list[dict]: A list of data samples from various data systems.
                Each element, `merged[i]`, is a data dictionary containing `keys`: `torch.Tensor`
                originating from the `i`-th data system.
            - Callable[[], list[dict]]: A lazy function that returns data samples in the above format
                only when needed. Since the sampling process can be slow and memory-intensive,
                the lazy function helps by only sampling once.
        stat_file_path : Optional[DPPath]
            The path to the stat file.

        """
        self.change_out_bias(
            merged,
            stat_file_path=stat_file_path,
            bias_adjust_mode="set-by-statistic",
        )

    def change_out_bias(
        self,
        sample_merged,
        stat_file_path: Optional[DPPath] = None,
        bias_adjust_mode="change-by-statistic",
    ) -> None:
        """Change the output bias according to the input data and the pretrained model.

        Parameters
        ----------
        sample_merged : Union[Callable[[], list[dict]], list[dict]]
            - list[dict]: A list of data samples from various data systems.
                Each element, `merged[i]`, is a data dictionary containing `keys`: `torch.Tensor`
                originating from the `i`-th data system.
            - Callable[[], list[dict]]: A lazy function that returns data samples in the above format
                only when needed. Since the sampling process can be slow and memory-intensive,
                the lazy function helps by only sampling once.
        bias_adjust_mode : str
            The mode for changing output bias : ['change-by-statistic', 'set-by-statistic']
            'change-by-statistic' : perform predictions on labels of target dataset,
                    and do least square on the errors to obtain the target shift as bias.
            'set-by-statistic' : directly use the statistic output bias in the target dataset.
        stat_file_path : Optional[DPPath]
            The path to the stat file.
        """
        if bias_adjust_mode == "set-by-statistic":
            bias_out, std_out = compute_output_stats(
                sample_merged,
                len(self.type_map),
                keys=['energy'],
                stat_file_path=stat_file_path,
                stats_distinguish_types=True,
                intensive=False,
            )
            self._store_out_stat(bias_out, std_out)
        else:
            raise RuntimeError("Unknown bias_adjust_mode mode: " + bias_adjust_mode)

    def _store_out_stat(
        self,
        out_bias: dict[str, torch.Tensor],
        out_std: dict[str, torch.Tensor],
        add: bool = False,
    ) -> None:
        pass

    def _varsize(
        self,
        shape: list[int],
    ) -> int:
        output_size = 1
        len_shape = len(shape)
        for i in range(len_shape):
            output_size *= shape[i]
        return output_size

    def _get_bias_index(
        self,
        kk: str,
    ) -> int:
        res: list[int] = []
        for i, e in enumerate(self.bias_keys):
            if e == kk:
                res.append(i)
        assert len(res) == 1
        return res[0]


    @torch.jit.export
    def fitting_output_def(self) -> FittingOutputDef:
        """Get the output def of developer implemented atomic models."""
        return FittingOutputDef(
            [
                OutputVariableDef(
                    name="energy",
                    shape=[1],
                    reducible=True,
                    r_differentiable=True,
                    c_differentiable=True,
                ),
            ],
        )

    @torch.jit.export
    def get_rcut(self) -> float:
        """Get the cut-off radius."""
        return self.r_cutoff

    @torch.jit.export
    def get_type_map(self) -> list[str]:
        """Get the type map."""
        return self.type_map

    @torch.jit.export
    def get_sel(self) -> list[int]:
        """Return the number of selected atoms for each type."""
        return [self.sel]

    @torch.jit.export
    def get_dim_fparam(self) -> int:
        """Get the number (dimension) of frame parameters of this atomic model."""
        return self.numb_fparam

    @torch.jit.export
    def get_dim_aparam(self) -> int:
        """Get the number (dimension) of atomic parameters of this atomic model."""
        return 0

    @torch.jit.export
    def get_sel_type(self) -> list[int]:
        """Get the selected atom types of this model.

        Only atoms with selected atom types have atomic contribution
        to the result of the model.
        If returning an empty list, all atom types are selected.
        """
        return []

    @torch.jit.export
    def is_aparam_nall(self) -> bool:
        """Check whether the shape of atomic parameters is (nframes, nall, ndim).

        If False, the shape is (nframes, nloc, ndim).
        """
        return False

    @torch.jit.export
    def mixed_types(self) -> bool:
        """Return whether the model is in mixed-types mode.

        If true, the model
        1. assumes total number of atoms aligned across frames;
        2. uses a neighbor list that does not distinguish different atomic types.
        If false, the model
        1. assumes total number of atoms of each atom type aligned across frames;
        2. uses a neighbor list that distinguishes different atomic types.
        """
        return True

    @torch.jit.export
    def has_message_passing(self) -> bool:
        """Return whether the descriptor has message passing."""
        return False

    def has_default_fparam(self) -> bool:
        return True

    def get_default_fparam(self) -> Optional[torch.Tensor]:
        self.default_fparam_tensor = torch.tensor([0.0,1.0])
        return self.default_fparam_tensor

    @torch.jit.export
    def forward(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        box: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        do_atomic_virial: bool = False,
    ) -> dict[str, torch.Tensor]:
        nframes, nloc, ndim = coord.shape
        device = coord.device
        data_batch = []
        
        box = box.reshape(-1, 3, 3)

        for i in range(nframes):
            idx_i, idx_j, shifts = self.calculator.compute(
                points=coord[i].cpu().numpy(), 
                box=box[i].cpu().numpy(), 
                periodic=True, 
                quantities="ijS"
                )
            idx_i, idx_j = idx_i.astype(np.int64), idx_j.astype(np.int64)
        
            idx_i, idx_j, shifts = torch.tensor(idx_i).long(), torch.tensor(idx_j).long(), torch.tensor(shifts).long()

            data = Data(
                x=None, y=None, pos=torch.tensor(coord[i]).float(),  # A
                edge_index=torch.vstack([idx_i, idx_j]).long(),
                atomic_numbers=self.type_map_index[atype[i].cpu()]-1,
                shifts=shifts,  # for pbc, D = pos_j - pos_i + shifts @ cell or D = pos_i - pos_j - shifts @ cell
                cell=torch.tensor(box[i].unsqueeze(0)).float(),  # for pbc, A
            )
            
            data_batch.append(data)
        data = Batch.from_data_list(data_batch).to(device).to_dict()
        
        data['pos'] = data['pos'].requires_grad_(True)

        if self.calc_stress and self.grad_mode == 'node':
            data['displacement'] = torch.zeros((3, 3), dtype=data['pos'].dtype, device=data['pos'].device)
            data['displacement'] = data['displacement'].view(-1, 3, 3).expand(data['batch'][-1] + 1, 3, 3)  # (N_batch, 3, 3)
            data['displacement'] = data['displacement'].requires_grad_(True)
            data['symmetric_displacement'] = 0.5 * (data['displacement'] + data['displacement'].transpose(-1, -2))
            data['pos'] = data['pos'] + torch.bmm(
                data['pos'].unsqueeze(-2), data['symmetric_displacement'][data['batch']]
            ).squeeze(-2)
            data['cell'] = data['cell'] + torch.bmm(data['cell'], data['symmetric_displacement'])

        # calc edge vectors depending on periodicity
        if 'shifts' in data and 'cell' in data:
            pbc_shift = torch.einsum("ni,nij->nj", data['shifts'].float(), data['cell'][data['batch']][data['edge_index'][0]])
            data['edge_vec'] = data['pos'][data['edge_index'][1]] - data['pos'][data['edge_index'][0]] + pbc_shift
        else:
            data['edge_vec'] = data['pos'][data['edge_index'][1]] - data['pos'][data['edge_index'][0]]
        

        energy = self.e3nn_layer(data)
        
        if self.scale is not None:
            self.scale = self.scale.to(device)
            if self.scale.dim() == 0:
                energy = energy * self.scale
            else:
                energy = energy * self.scale[data['atomic_numbers']]
        if self.shift is not None:
            self.shift = self.shift.to(device)
            if self.shift.dim() == 0:
                energy = energy + self.shift
            else:
                energy = energy + self.shift[data['atomic_numbers']]
        # reduced_energy = torch.zeros(data['batch'][-1] + 1, 1, device=energy.device, dtype=energy.dtype)
        # reduced_energy.index_add_(0, data['batch'], energy)
        # energy = reduced_energy
        energy = scatter(energy, data['batch'], dim=0, dim_size=data['batch'][-1] + 1).squeeze(-1)
        
        forces, stress = self.force_stress_output(energy, data, training=True)

        if self.calc_dipole:
            dipole = None
        else:
            dipole = None
        
        if self.calc_polar:
            polar = None
        else:
            polar = None

        virial = torch.zeros((nframes, 3,3), dtype=energy.dtype, device=energy.device)
        
        for i in range(nframes):
            s_xx, s_yy, s_zz, s_yz, s_zx, s_xy = stress[i].unbind(dim=-1)
            virial[i] = torch.stack([
                    torch.stack([s_xx, s_xy, s_zx], dim=-1),
                    torch.stack([s_xy, s_yy, s_yz], dim=-1),
                    torch.stack([s_zx, s_yz, s_zz], dim=-1)], dim=-2)
        
        model_predict = {}
        model_predict["energy"] = energy
        model_predict["force"] = forces.view(nframes, nloc, 3)
        model_predict["virial"] = virial.view(nframes, 9)
        return model_predict

    @torch.jit.export
    def forward_lower(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        do_atomic_virial: bool = False,
        comm_dict: Optional[dict[str, torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def forward_lower_common(
        self,
        nloc: int,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        do_atomic_virial: bool = False,  # noqa: ARG002
        comm_dict: Optional[dict[str, torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def serialize(self) -> dict:
        raise NotImplementedError

    @classmethod
    def deserialize(cls, data: dict):
        raise NotImplementedError

    @torch.jit.export
    def get_nnei(self) -> int:
        """Return the total number of selected neighboring atoms in cut-off radius."""
        raise NotImplementedError

    @torch.jit.export
    def get_nsel(self) -> int:
        """Return the total number of selected neighboring atoms in cut-off radius."""
        raise NotImplementedError

    @classmethod
    def update_sel(
        cls,
        train_data,
        type_map: Optional[list[str]],
        local_jdata: dict,
    ) -> tuple[dict, Optional[float]]:
        """Update the selection and perform neighbor statistics.

        Parameters
        ----------
        train_data : DeepmdDataSystem
            data used to do neighbor statictics
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
        raise NotImplementedError

    @torch.jit.export
    def model_output_type(self) -> list[str]:
        """Get the output type for the model."""
        return ["energy"]

    def translated_output_def(self):
        """Get the translated output def for the model."""
        raise NotImplementedError

    def model_output_def(self):
        """Get the output def for the model."""
        raise NotImplementedError
