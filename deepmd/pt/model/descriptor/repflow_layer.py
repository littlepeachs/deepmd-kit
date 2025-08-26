# SPDX-License-Identifier: LGPL-3.0-or-later
import time
from typing import (
    Optional,
    Union,
)
import torch.nn.functional as F
import torch
import torch.nn as nn

from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.pt.model.descriptor.repformer_layer import (
    Atten2Map,
    Atten2MultiHeadApply,
    _apply_nlist_mask,
    _apply_switch,
    _make_nei_g1,
    get_residual,
)
from deepmd.pt.model.network.layernorm import (
    LayerNorm,
)
from .rmsnorm import RMSLayerNorm
from deepmd.pt.model.network.mlp import (
    MLPLayer,
)
from deepmd.pt.model.network.utils import (
    aggregate,
)
from deepmd.pt.utils.env import (
    PRECISION_DICT,
)
from deepmd.pt.utils.utils import (
    ActivationFn,
    to_numpy_array,
    to_torch_tensor,
)
from deepmd.utils.version import (
    check_version_compatibility,
)
from .atomic_moment import AtomicMoment
from .hyper_moment import HyperMoment
from .p3m_longrange import FNO3d
from .bessel_layer import BesselBasisLayer
from torch_scatter import scatter

class RepFlowLayer(torch.nn.Module):
    def __init__(
        self,
        e_rcut: float,
        e_rcut_smth: float,
        e_sel: int,
        a_rcut: float,
        a_rcut_smth: float,
        a_sel: int,
        ntypes: int,
        n_dim: int = 128,
        e_dim: int = 16,
        a_dim: int = 64,
        a_compress_rate: int = 0,
        a_compress_use_split: bool = False,
        a_compress_e_rate: int = 1,
        n_multi_edge_message: int = 1,
        axis_neuron: int = 4,
        update_angle: bool = True,  # angle
        optim_update: bool = True,
        use_dynamic_sel: bool = False,
        sel_reduce_factor: float = 10.0,
        smooth_edge_update: bool = False,
        use_rbf: bool = False,
        use_torsion: bool = False,
        use_atomic_moment: bool = False,
        update_dihedral: bool = False,
        d_dim: int = 32,
        d_sel: int = 10,
        d_rcut: float = 2.8,
        d_rcut_smth: float = 2.0,
        use_ffn_node_edge_message: bool = False,
        use_ffn_edge_edge_message: bool = False,
        use_ffn_edge_angle_message: bool = False,
        use_ffn_angle_angle_message: bool = False,
        ffn_hidden_dim: int = 1024,
        edge_use_attn: bool = False,
        edge_attn_hidden: int = 32,
        edge_attn_head: int = 4,
        edge_attn_use_ln: bool = True,
        edge_rbf_dot_self: bool = False,
        edge_rbf_dot_message: bool = False,
        rbf_dim: int = 16,
        residual_pref: list = [],
        message_use_self_concat: bool = False,
        use_slim_message: bool = False,
        activation_function: str = "silu",
        update_style: str = "res_residual",
        update_residual: float = 0.1,
        update_residual_init: str = "const",
        precision: str = "float64",
        seed: Optional[Union[int, list[int]]] = None,
        layer_idx: int = 0,
        max_layer_num: int = 0,
        use_p3m: bool = False,
        use_angle_weight: bool = False,
        use_les: bool = False,
    ) -> None:
        super().__init__()
        self.epsilon = 1e-4  # protection of 1./nnei
        self.e_rcut = float(e_rcut)
        self.e_rcut_smth = float(e_rcut_smth)
        self.ntypes = ntypes
        e_sel = [e_sel] if isinstance(e_sel, int) else e_sel
        self.nnei = sum(e_sel)
        assert len(e_sel) == 1
        self.e_sel = e_sel
        self.sec = self.e_sel
        self.a_rcut = a_rcut
        self.a_rcut_smth = a_rcut_smth
        self.a_sel = a_sel
        self.n_dim = n_dim
        self.e_dim = e_dim
        self.a_dim = a_dim
        self.a_compress_rate = a_compress_rate
        self.use_rbf = use_rbf
        self.use_torsion = use_torsion
        self.use_atomic_moment = use_atomic_moment
        self.use_p3m = use_p3m
        self.use_les = use_les
        self.use_angle_weight = use_angle_weight
        self.layer_idx = layer_idx
        self.max_layer_num = max_layer_num
        if a_compress_rate != 0:
            assert a_dim % (2 * a_compress_rate) == 0, (
                f"For a_compress_rate of {a_compress_rate}, a_dim must be divisible by {2 * a_compress_rate}. "
                f"Currently, a_dim={a_dim} is not valid."
            )
        self.n_multi_edge_message = n_multi_edge_message
        assert self.n_multi_edge_message >= 1, "n_multi_edge_message must >= 1!"
        self.axis_neuron = axis_neuron
        self.update_angle = update_angle
        self.activation_function = activation_function
        self.act = ActivationFn(activation_function)
        self.update_style = update_style
        self.update_residual = update_residual
        self.update_residual_init = update_residual_init
        self.a_compress_e_rate = a_compress_e_rate
        self.a_compress_use_split = a_compress_use_split
        self.precision = precision
        self.seed = seed
        self.prec = PRECISION_DICT[precision]
        self.optim_update = optim_update
        self.smooth_edge_update = smooth_edge_update
        self.use_dynamic_sel = use_dynamic_sel
        self.sel_reduce_factor = sel_reduce_factor
        self.dynamic_e_sel = self.nnei / self.sel_reduce_factor
        self.dynamic_a_sel = self.a_sel / self.sel_reduce_factor

        self.update_dihedral = update_dihedral
        self.d_dim = d_dim
        self.d_sel = d_sel
        self.d_rcut = d_rcut
        self.d_rcut_smth = d_rcut_smth
        self.dynamic_d_sel = (self.d_sel * 4) / self.sel_reduce_factor
        self.use_ffn_node_edge_message = use_ffn_node_edge_message
        self.use_ffn_edge_edge_message = use_ffn_edge_edge_message
        self.use_ffn_edge_angle_message = use_ffn_edge_angle_message
        self.use_ffn_angle_angle_message = use_ffn_angle_angle_message
        self.ffn_hidden_dim = ffn_hidden_dim
        if (
            self.use_ffn_node_edge_message
            or self.use_ffn_edge_edge_message
            or self.use_ffn_edge_angle_message
            or self.use_ffn_angle_angle_message
        ):
            assert not self.optim_update, "FFN does not support optim update!"

        self.edge_use_attn = edge_use_attn
        self.edge_attn_hidden = edge_attn_hidden
        self.edge_attn_head = edge_attn_head
        self.edge_attn_use_ln = edge_attn_use_ln
        self.edge_rbf_dot_self = edge_rbf_dot_self
        self.edge_rbf_dot_message = edge_rbf_dot_message
        self.rbf_dim = rbf_dim
        self.residual_pref = residual_pref
        self.residual_pref += [1.0] * 10
        residual_idx = 0
        self.message_use_self_concat = message_use_self_concat
        self.use_slim_message = use_slim_message
        if self.message_use_self_concat:
            assert (
                self.n_multi_edge_message == 1
            ), "Only one message head is supported for self concatenation!"

        if self.edge_rbf_dot_self or self.edge_rbf_dot_message:
            self.rbf_mlp = MLPLayer(
                self.rbf_dim,
                self.e_dim,
                precision=precision,
                seed=child_seed(seed, 30),
            )
        else:
            self.rbf_mlp = None
        
        if self.use_rbf:
            self.rbf_linear = MLPLayer(
                self.rbf_dim, self.e_dim, precision=precision, seed=child_seed(seed, 4)
            )
        else:
            self.rbf_linear = None
        
        if self.use_atomic_moment:
            if self.layer_idx == 0:
                self.max_atom_feats_rank = 0
                self.mix_atom_feats_radial_channel = False
            else:
                self.max_atom_feats_rank = None
                self.mix_atom_feats_radial_channel = True

            # for the last layer, we are only interested in the scalar output, from
            # which we can compute the total energy
            if self.layer_idx == self.max_layer_num - 1:
                self.max_out_rank = 0
            else:
                self.max_out_rank = None

            max_v = 2
            self.max_atom_feats_rank = (
                max_v if self.max_atom_feats_rank is None else self.max_atom_feats_rank
            )
            self.max_out_rank = max_v if self.max_out_rank is None else self.max_out_rank

            self.atomic_moment_layer = AtomicMoment(
                max_u=self.rbf_dim,
                max_v1=self.max_atom_feats_rank,
                max_v2=max_v,
                n_dim=self.n_dim,
                e_dim=self.e_dim,
                rbf_dim=self.rbf_dim,
            )
            self.hyper_moment_layer = HyperMoment(
                max_u=self.rbf_dim,
                max_v=max_v,
                n_dim=self.n_dim,
                e_dim=self.e_dim,
                max_out_rank=self.max_out_rank,
                rbf_dim=self.rbf_dim,
            )
            self.atom_feats_in_transform = MLPLayer(num_in=self.n_dim, 
                                        num_out=self.rbf_dim, 
                                        bias=False,
                                        precision=precision, 
                                        seed=child_seed(seed, 4))
            self.atom_feats_out_transform = MLPLayer(num_in=self.rbf_dim, 
                                        num_out=self.n_dim, 
                                        bias=False,
                                        precision=precision, 
                                        seed=child_seed(seed, 4))
            self.linear_channel_hyper = nn.ModuleDict(
                {str(rank): MLPLayer(num_in=self.rbf_dim, 
                                    num_out=self.rbf_dim, 
                                    bias=False,
                                    precision=precision, 
                                    seed=child_seed(seed, 4)) 
                                    for rank in range(self.max_out_rank + 1)}
            )
            if self.mix_atom_feats_radial_channel:
                self.linear_channel_feats = nn.ModuleDict(
                    {
                        str(rank): MLPLayer(num_in=self.rbf_dim, 
                                            num_out=self.rbf_dim, 
                                            bias=False,
                                            precision=precision, 
                                            seed=child_seed(seed, 4))
                        for rank in range(self.max_out_rank + 1)
                    }
                )
            else:
                self.linear_channel_feats = nn.ModuleDict({})
            
            self.moment_layer_norm = nn.ModuleDict({
                str(rank): RMSLayerNorm(self.rbf_dim,rank, eps=1e-5)
                for rank in range(max_v + 1)
            })
        else:
            self.atomic_moment_layer = None

        if self.use_p3m:
            self.long_mp = FNO3d(
                    *[3,3,3],
                    hidden_channels=self.n_dim // 2, 
                    in_channels=self.n_dim, 
                    out_channels=self.n_dim, 
                    n_layers=1,
                    lifting_channels=self.n_dim // 2,
                    projection_channels=self.n_dim // 2,
                    non_linearity=ActivationFn("custom_silu:10.0"),
                )
            self.long_mp_2 = MLPLayer(
                self.n_dim,
                self.n_dim,
                precision=precision,
                seed=child_seed(seed, 31),
            )

        if self.use_angle_weight:
            self.edge_weight_proj_1 = MLPLayer(
                self.a_dim+self.a_dim,
                1,
                precision=precision,
                seed=child_seed(seed, 31),
                activation_function="sigmoid",
            )
            self.edge_weight_proj_2 = MLPLayer(
                self.a_dim+self.a_dim,
                1,
                precision=precision,
                seed=child_seed(seed, 31),
            )
            self.node_j_mlp = MLPLayer(
                self.n_dim,
                self.n_dim,
                precision=precision,
                seed=child_seed(seed, 31),
            )
            self.node_k_mlp = MLPLayer(
                self.n_dim,
                self.n_dim,
                precision=precision,
                seed=child_seed(seed, 31),
            )
            
            
            

        if self.edge_rbf_dot_message:
            self.rbf_mlp_message = MLPLayer(
                self.rbf_dim,
                self.n_dim,
                precision=precision,
                seed=child_seed(seed, 31),
            )
        else:
            self.rbf_mlp_message = None

        if self.edge_use_attn:
            assert (
                not self.use_dynamic_sel
            ), "Attention does not support dynamic selection!"

        assert update_residual_init in [
            "norm",
            "const",
        ], "'update_residual_init' only support 'norm' or 'const'!"

        self.update_residual = update_residual
        self.update_residual_init = update_residual_init
        self.n_residual = []
        self.e_residual = []
        self.a_residual = []
        self.d_residual = []
        self.t_residual = []
        self.l_residual = []
        self.edge_info_dim = self.n_dim * 2 + self.e_dim

        # node self mlp
        self.node_self_mlp = MLPLayer(
            n_dim,
            n_dim,
            precision=precision,
            seed=child_seed(seed, 0),
        )
        if self.update_style == "res_residual":
            self.n_residual.append(
                get_residual(
                    n_dim,
                    self.update_residual * self.residual_pref[residual_idx],
                    self.update_residual_init,
                    precision=precision,
                    seed=child_seed(seed, 1),
                )
            )
            residual_idx += 1

        # node sym (grrg + drrd)
        self.n_sym_dim = n_dim * self.axis_neuron + e_dim * self.axis_neuron
        self.node_sym_linear = MLPLayer(
            self.n_sym_dim,
            n_dim,
            precision=precision,
            seed=child_seed(seed, 2),
        )
        if self.update_style == "res_residual":
            self.n_residual.append(
                get_residual(
                    n_dim,
                    self.update_residual * self.residual_pref[residual_idx],
                    self.update_residual_init,
                    precision=precision,
                    seed=child_seed(seed, 3),
                )
            )
            residual_idx += 1

        # node edge message
        self.node_edge_linear = MLPLayer(
            self.edge_info_dim
            if not self.use_ffn_node_edge_message
            else self.ffn_hidden_dim,
            self.n_multi_edge_message * n_dim,
            precision=precision,
            seed=child_seed(seed, 4),
        )
        if self.message_use_self_concat:
            self.node_edge_linear_2 = MLPLayer(
                self.n_dim + self.n_dim,
                self.n_dim,
                precision=precision,
                seed=child_seed(seed, 5),
            )
        else:
            self.node_edge_linear_2 = None

        if self.update_style == "res_residual":
            for head_index in range(self.n_multi_edge_message):
                self.n_residual.append(
                    get_residual(
                        n_dim,
                        self.update_residual * self.residual_pref[residual_idx],
                        self.update_residual_init,
                        precision=precision,
                        seed=child_seed(child_seed(seed, 5), head_index),
                    )
                )
                residual_idx += 1

        # edge self message
        self.edge_self_linear = MLPLayer(
            self.edge_info_dim
            if not self.use_ffn_edge_edge_message
            else self.ffn_hidden_dim,
            e_dim,
            precision=precision,
            seed=child_seed(seed, 6),
        )
        if self.update_style == "res_residual":
            self.e_residual.append(
                get_residual(
                    e_dim,
                    self.update_residual * self.residual_pref[residual_idx],
                    self.update_residual_init,
                    precision=precision,
                    seed=child_seed(seed, 7),
                )
            )
            residual_idx += 1

        # edge attention
        if self.edge_use_attn:
            self.edge_attn_map = Atten2Map(
                e_dim,
                self.edge_attn_hidden,
                self.edge_attn_head,
                has_gate=True,
                smooth=True,
                precision=precision,
                seed=child_seed(seed, 21),
            )
            self.edge_mh_apply = Atten2MultiHeadApply(
                e_dim,
                self.edge_attn_head,
                precision=precision,
                seed=child_seed(seed, 22),
            )
            self.edge_lm = LayerNorm(
                e_dim,
                trainable=True,
                precision=precision,
                seed=child_seed(seed, 23),
            )
            if self.update_style == "res_residual":
                self.e_residual.append(
                    get_residual(
                        e_dim,
                        self.update_residual,
                        self.update_residual_init,
                        precision=precision,
                        seed=child_seed(seed, 24),
                    )
                )
        else:
            self.edge_attn_map = None
            self.edge_mh_apply = None
            self.edge_lm = None

        if self.update_angle:
            self.angle_dim = self.a_dim
            if self.a_compress_rate == 0:
                # angle + node + edge * 2
                self.angle_dim += self.n_dim + 2 * self.e_dim
                self.a_compress_n_linear = None
                self.a_compress_e_linear = None
                self.e_a_compress_dim = e_dim
                self.n_a_compress_dim = n_dim
            else:
                # angle + a_dim/c + a_dim/2c * 2 * e_rate
                self.angle_dim += (1 + self.a_compress_e_rate) * (
                    self.a_dim // self.a_compress_rate
                )
                self.e_a_compress_dim = (
                    self.a_dim // (2 * self.a_compress_rate) * self.a_compress_e_rate
                )
                self.n_a_compress_dim = self.a_dim // self.a_compress_rate
                if not self.a_compress_use_split:
                    self.a_compress_n_linear = MLPLayer(
                        self.n_dim,
                        self.n_a_compress_dim,
                        precision=precision,
                        bias=False,
                        seed=child_seed(seed, 8),
                    )
                    self.a_compress_e_linear = MLPLayer(
                        self.e_dim,
                        self.e_a_compress_dim,
                        precision=precision,
                        bias=False,
                        seed=child_seed(seed, 9),
                    )
                else:
                    self.a_compress_n_linear = None
                    self.a_compress_e_linear = None

            # edge angle message
            self.edge_angle_linear1 = MLPLayer(
                self.angle_dim
                if not self.use_ffn_edge_angle_message
                else self.ffn_hidden_dim,
                self.e_dim,
                precision=precision,
                seed=child_seed(seed, 10),
            )
            if not self.use_slim_message:
                self.edge_angle_linear2 = MLPLayer(
                    self.e_dim
                    if not self.message_use_self_concat
                    else self.e_dim + self.e_dim,
                    self.e_dim,
                    precision=precision,
                    seed=child_seed(seed, 11),
                )
            else:
                self.edge_angle_linear2 = None
            if self.update_style == "res_residual":
                self.e_residual.append(
                    get_residual(
                        self.e_dim,
                        self.update_residual * self.residual_pref[residual_idx],
                        self.update_residual_init,
                        precision=precision,
                        seed=child_seed(seed, 12),
                    )
                )
                residual_idx += 1

            # angle self message
            self.angle_self_linear = MLPLayer(
                self.angle_dim
                if not self.use_ffn_angle_angle_message
                else self.ffn_hidden_dim,
                self.a_dim,
                precision=precision,
                seed=child_seed(seed, 13),
            )
            if self.update_style == "res_residual":
                self.a_residual.append(
                    get_residual(
                        self.a_dim,
                        self.update_residual,
                        self.update_residual_init,
                        precision=precision,
                        seed=child_seed(seed, 14),
                    )
                )

            if self.update_dihedral:
                self.dihedral_dim = self.d_dim + 2 * self.a_dim
                # angle dihedral message
                self.angle_dihedral_linear = MLPLayer(
                    self.dihedral_dim,
                    self.a_dim,
                    precision=precision,
                    seed=child_seed(seed, 15),
                )
                if self.update_style == "res_residual":
                    self.a_residual.append(
                        get_residual(
                            self.a_dim,
                            self.update_residual,
                            self.update_residual_init,
                            precision=precision,
                            seed=child_seed(seed, 16),
                        )
                    )

                # dihedral self message
                self.dihedral_self_linear = MLPLayer(
                    self.dihedral_dim,
                    self.d_dim,
                    precision=precision,
                    seed=child_seed(seed, 17),
                )
                if self.update_style == "res_residual":
                    self.d_residual.append(
                        get_residual(
                            self.d_dim,
                            self.update_residual,
                            self.update_residual_init,
                            precision=precision,
                            seed=child_seed(seed, 18),
                        )
                    )
            else:
                self.angle_dihedral_linear = None
                self.dihedral_self_linear = None
        else:
            self.angle_self_linear = None
            self.edge_angle_linear1 = None
            self.edge_angle_linear2 = None
            self.a_compress_n_linear = None
            self.a_compress_e_linear = None
            self.angle_dim = 0
            self.dihedral_dim = 0
            self.angle_dihedral_linear = None
            self.dihedral_self_linear = None

        if self.use_torsion:
            self.torsion_dim = self.a_dim
            if self.a_compress_rate == 0:
                # torsion + node + edge * 2
                self.torsion_dim += self.n_dim + 2 * self.e_dim
                self.t_compress_n_linear = None
                self.t_compress_e_linear = None
                self.e_t_compress_dim = self.e_dim
                self.n_t_compress_dim = self.n_dim
            else:
                # torsion + t_dim/c + t_dim/2c * 2 * e_rate
                self.torsion_dim += 5*self.a_dim
                self.e_t_compress_dim = self.a_dim
                self.n_t_compress_dim = self.a_dim
                self.t_compress_n_linear = MLPLayer(
                    self.n_dim,
                    self.n_t_compress_dim,
                    precision=precision,
                    bias=False,
                    seed=child_seed(seed, 15),
                )
                self.t_compress_e_linear = MLPLayer(
                    self.e_dim,
                    self.e_t_compress_dim,
                    precision=precision,
                    bias=False,
                    seed=child_seed(seed, 16),
                )
            self.edge_torsion_linear1 = MLPLayer(
                self.torsion_dim,
                self.e_dim,
                precision=precision,
                seed=child_seed(seed, 17),
            )
            
            if self.update_style == "res_residual":
                self.e_residual.append(
                    get_residual(
                        self.e_dim,
                        self.update_residual,
                        self.update_residual_init,
                        precision=precision,
                        seed=child_seed(seed, 18),
                    )
                )
            # torsion self message
            self.torsion_self_linear = MLPLayer(
                self.torsion_dim,
                self.a_dim,
                precision=precision,
                seed=child_seed(seed, 17),
            )
            if self.update_style == "res_residual":
                self.t_residual.append(
                    get_residual(
                        self.a_dim,
                        self.update_residual,
                        self.update_residual_init,
                        precision=precision,
                        seed=child_seed(seed, 18),
                    )
                )
        else:
            self.torsion_self_linear = None
            self.edge_torsion_linear1 = None
            self.t_compress_n_linear = None
            self.t_compress_e_linear = None
            self.torsion_dim = 0

        if self.use_ffn_node_edge_message or self.use_ffn_edge_edge_message:
            self.edge_message_ffn1 = MLPLayer(
                self.edge_info_dim,
                self.ffn_hidden_dim,
                precision=precision,
                bias=False,
                seed=child_seed(seed, 19),
            )
        else:
            self.edge_message_ffn1 = None

        if self.use_ffn_edge_angle_message or self.use_ffn_angle_angle_message:
            self.angle_message_ffn1 = MLPLayer(
                self.angle_dim,
                self.ffn_hidden_dim,
                precision=precision,
                bias=False,
                seed=child_seed(seed, 20),
            )
        else:
            self.angle_message_ffn1 = None

        # if self.use_atomic_moment:
        #     if self.update_style == "res_residual":
        #         self.n_residual.append(
        #             get_residual(
        #                 n_dim,
        #                 self.update_residual * self.residual_pref[residual_idx],
        #                 self.update_residual_init,
        #                 precision=precision,
        #                 seed=child_seed(seed, 3),
        #             )
        #         )
        #         residual_idx += 1
        
        if self.use_p3m:
            if self.update_style == "res_residual":
                self.n_residual.append(
                    get_residual(
                        self.n_dim,
                        self.update_residual,
                        precision=precision,
                        seed=child_seed(seed, 3),
                    )
                )
                self.l_residual.append(
                    get_residual(
                        self.n_dim,
                        self.update_residual,
                        precision=precision,
                        seed=child_seed(seed, 3),
                    )
                )
                self.l_residual.append(
                    get_residual(
                        self.n_dim,
                        self.update_residual,
                        precision=precision,
                        seed=child_seed(seed, 4),
                    )
                )
            self.a2m_linear = MLPLayer(
                self.n_dim,
                self.n_dim,
                precision=precision,
                seed=child_seed(seed, 5),
            )
            self.m2a_linear = MLPLayer(
                self.n_dim,
                self.n_dim,
                precision=precision,
                seed=child_seed(seed, 6),
            )
            self.atom_mesh_rbf = BesselBasisLayer(
                num_radial=self.rbf_dim,
                cutoff=self.e_rcut,
                envelope_exponent=5,
            )
            self.atom_mesh_distance_linear = MLPLayer(
                self.rbf_dim,
                self.n_dim,
                precision=precision,
                seed=child_seed(seed, 7),
            )
            
            
        self.n_residual = nn.ParameterList(self.n_residual)
        self.e_residual = nn.ParameterList(self.e_residual)
        self.a_residual = nn.ParameterList(self.a_residual)
        self.d_residual = nn.ParameterList(self.d_residual)
        self.t_residual = nn.ParameterList(self.t_residual)
    @staticmethod
    def _cal_hg(
        edge_ebd: torch.Tensor,
        h2: torch.Tensor,
        nlist_mask: torch.Tensor,
        sw: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate the transposed rotation matrix.

        Parameters
        ----------
        edge_ebd
            Neighbor-wise/Pair-wise edge embeddings, with shape nb x nloc x nnei x e_dim.
        h2
            Neighbor-wise/Pair-wise equivariant rep tensors, with shape nb x nloc x nnei x 3.
        nlist_mask
            Neighbor list mask, where zero means no neighbor, with shape nb x nloc x nnei.
        sw
            The switch function, which equals 1 within the rcut_smth range, smoothly decays from 1 to 0 between rcut_smth and rcut,
            and remains 0 beyond rcut, with shape nb x nloc x nnei.

        Returns
        -------
        hg
            The transposed rotation matrix, with shape nb x nloc x 3 x e_dim.
        """
        # edge_ebd:  nb x nloc x nnei x e_dim
        # h2:  nb x nloc x nnei x 3
        # msk: nb x nloc x nnei
        nb, nloc, nnei, _ = edge_ebd.shape
        e_dim = edge_ebd.shape[-1]
        # nb x nloc x nnei x e_dim
        edge_ebd = _apply_nlist_mask(edge_ebd, nlist_mask)
        edge_ebd = _apply_switch(edge_ebd, sw)
        invnnei = torch.rsqrt(
            float(nnei)
            * torch.ones((nb, nloc, 1, 1), dtype=edge_ebd.dtype, device=edge_ebd.device)
        )
        # nb x nloc x 3 x e_dim
        h2g2 = torch.matmul(torch.transpose(h2, -1, -2), edge_ebd) * invnnei
        return h2g2

    @staticmethod
    def _cal_hg_dynamic(
        flat_edge_ebd: torch.Tensor,
        flat_h2: torch.Tensor,
        flat_sw: torch.Tensor,
        owner: torch.Tensor,
        num_owner: int,
        nloc: int,
        scale_factor: float,
    ) -> torch.Tensor:
        """
        Calculate the transposed rotation matrix.

        Parameters
        ----------
        flat_edge_ebd
            Flatted neighbor-wise/pair-wise invariant rep tensors, with shape n_edge x e_dim.
        flat_h2
            Flatted neighbor-wise/pair-wise equivariant rep tensors, with shape n_edge x 3.
        flat_sw
            Flatted switch function, which equals 1 within the rcut_smth range, smoothly decays from 1 to 0 between rcut_smth and rcut,
            and remains 0 beyond rcut, with shape n_edge.
        owner
            The owner index of the neighbor to reduce on.
        num_owner : int
            The total number of the owner.
        nloc : int
            The number of local atoms.
        scale_factor : float
            The scale factor to apply after reduce.

        Returns
        -------
        hg
            The transposed rotation matrix, with shape nf x nloc x 3 x e_dim.
        """
        n_edge, e_dim = flat_edge_ebd.shape
        # n_edge x e_dim
        flat_edge_ebd = flat_edge_ebd * flat_sw.unsqueeze(-1)
        # n_edge x 3 x e_dim
        flat_h2g2 = (flat_h2[:, :, None] * flat_edge_ebd[:, None, :]).reshape(
            -1, 3 * e_dim
        )
        # nf x nloc x 3 x e_dim
        h2g2 = (
            aggregate(flat_h2g2, owner, average=False, num_owner=num_owner).reshape(
                -1, nloc, 3, e_dim
            )
            * scale_factor
        )
        return h2g2

    @staticmethod
    def _cal_grrg(h2g2: torch.Tensor, axis_neuron: int) -> torch.Tensor:
        """
        Calculate the atomic invariant rep.

        Parameters
        ----------
        h2g2
            The transposed rotation matrix, with shape nb x nloc x 3 x e_dim.
        axis_neuron
            Size of the submatrix.

        Returns
        -------
        grrg
            Atomic invariant rep, with shape nb x nloc x (axis_neuron x e_dim)
        """
        # nb x nloc x 3 x e_dim
        nb, nloc, _, e_dim = h2g2.shape
        # nb x nloc x 3 x axis
        h2g2m = h2g2[..., :axis_neuron]
        # nb x nloc x axis x e_dim
        g1_13 = torch.matmul(torch.transpose(h2g2m, -1, -2), h2g2) / (3.0**1)
        # nb x nloc x (axisxng2)
        g1_13 = g1_13.view(nb, nloc, axis_neuron * e_dim)
        return g1_13

    def symmetrization_op(
        self,
        edge_ebd: torch.Tensor,
        h2: torch.Tensor,
        nlist_mask: torch.Tensor,
        sw: torch.Tensor,
        axis_neuron: int,
    ) -> torch.Tensor:
        """
        Symmetrization operator to obtain atomic invariant rep.

        Parameters
        ----------
        edge_ebd
            Neighbor-wise/Pair-wise invariant rep tensors, with shape nb x nloc x nnei x e_dim.
        h2
            Neighbor-wise/Pair-wise equivariant rep tensors, with shape nb x nloc x nnei x 3.
        nlist_mask
            Neighbor list mask, where zero means no neighbor, with shape nb x nloc x nnei.
        sw
            The switch function, which equals 1 within the rcut_smth range, smoothly decays from 1 to 0 between rcut_smth and rcut,
            and remains 0 beyond rcut, with shape nb x nloc x nnei.
        axis_neuron
            Size of the submatrix.

        Returns
        -------
        grrg
            Atomic invariant rep, with shape nb x nloc x (axis_neuron x e_dim)
        """
        # edge_ebd:  nb x nloc x nnei x e_dim
        # h2:  nb x nloc x nnei x 3
        # msk: nb x nloc x nnei
        nb, nloc, nnei, _ = edge_ebd.shape
        # nb x nloc x 3 x e_dim
        h2g2 = self._cal_hg(
            edge_ebd,
            h2,
            nlist_mask,
            sw,
        )
        # nb x nloc x (axisxng2)
        g1_13 = self._cal_grrg(h2g2, axis_neuron)
        return g1_13

    def symmetrization_op_dynamic(
        self,
        flat_edge_ebd: torch.Tensor,
        flat_h2: torch.Tensor,
        flat_sw: torch.Tensor,
        owner: torch.Tensor,
        num_owner: int,
        nloc: int,
        scale_factor: float,
        axis_neuron: int,
    ) -> torch.Tensor:
        """
        Symmetrization operator to obtain atomic invariant rep.

        Parameters
        ----------
        flat_edge_ebd
            Flatted neighbor-wise/pair-wise invariant rep tensors, with shape n_edge x e_dim.
        flat_h2
            Flatted neighbor-wise/pair-wise equivariant rep tensors, with shape n_edge x 3.
        flat_sw
            Flatted switch function, which equals 1 within the rcut_smth range, smoothly decays from 1 to 0 between rcut_smth and rcut,
            and remains 0 beyond rcut, with shape n_edge.
        owner
            The owner index of the neighbor to reduce on.
        num_owner : int
            The total number of the owner.
        nloc : int
            The number of local atoms.
        scale_factor : float
            The scale factor to apply after reduce.
        axis_neuron
            Size of the submatrix.

        Returns
        -------
        grrg
            Atomic invariant rep, with shape nb x nloc x (axis_neuron x e_dim)
        """
        # nb x nloc x 3 x e_dim
        h2g2 = self._cal_hg_dynamic(
            flat_edge_ebd,
            flat_h2,
            flat_sw,
            owner,
            num_owner,
            nloc,
            scale_factor,
        )
        # nb x nloc x (axis x e_dim)
        grrg = self._cal_grrg(h2g2, axis_neuron)
        return grrg
    
    def optim_dihedral_update(
        self,
        dihedral_ebd: torch.Tensor,
        angle_ebd: torch.Tensor,
        feat: str = "angle",
    ) -> torch.Tensor:
        angle_dim = angle_ebd.shape[-1]
        dihedral_dim = dihedral_ebd.shape[-1]
        sub_dihedral_idx = (0, angle_dim)
        sub_angle_idx_ijk = (angle_dim, angle_dim + angle_dim)
        sub_edge_idx_ijl = (angle_dim + angle_dim, angle_dim + angle_dim + angle_dim)
        
        if feat == "angle":
            matrix, bias = self.angle_dihedral_linear.matrix, self.angle_dihedral_linear.bias
        elif feat == "dihedral":
            matrix, bias = self.dihedral_self_linear.matrix, self.dihedral_self_linear.bias
        else:
            raise NotImplementedError
        assert dihedral_dim + 2 * angle_dim == matrix.size()[0]
        
        sub_dihedral_update = torch.matmul(
            dihedral_ebd, matrix[sub_dihedral_idx[0] : sub_dihedral_idx[1]]
        )
        
        sub_angle_update_ijk = torch.matmul(
            angle_ebd, matrix[sub_angle_idx_ijk[0] : sub_angle_idx_ijk[1]]
        )
        
        sub_angle_update_ijl = torch.matmul(
            angle_ebd, matrix[sub_edge_idx_ijl[0] : sub_edge_idx_ijl[1]]
        )
        result_update = (
            sub_dihedral_update
            + sub_angle_update_ijk[:, :, :, :, None, :]
            + sub_angle_update_ijl[:, :, :, None, :, :]
        ) + bias
        return result_update
        

        
    def optim_angle_update(
        self,
        angle_ebd: torch.Tensor,
        node_ebd: torch.Tensor,
        edge_ebd: torch.Tensor,
        feat: str = "edge",
    ) -> torch.Tensor:
        angle_dim = angle_ebd.shape[-1]
        node_dim = node_ebd.shape[-1]
        edge_dim = edge_ebd.shape[-1]
        sub_angle_idx = (0, angle_dim)
        sub_node_idx = (angle_dim, angle_dim + node_dim)
        sub_edge_idx_ik = (angle_dim + node_dim, angle_dim + node_dim + edge_dim)
        sub_edge_idx_ij = (
            angle_dim + node_dim + edge_dim,
            angle_dim + node_dim + 2 * edge_dim,
        )

        if feat == "edge":
            matrix, bias = self.edge_angle_linear1.matrix, self.edge_angle_linear1.bias
        elif feat == "angle":
            matrix, bias = self.angle_self_linear.matrix, self.angle_self_linear.bias
        else:
            raise NotImplementedError
        assert angle_dim + node_dim + 2 * edge_dim == matrix.size()[0]

        # nf * nloc * a_sel * a_sel * angle_dim
        sub_angle_update = torch.matmul(
            angle_ebd, matrix[sub_angle_idx[0] : sub_angle_idx[1]]
        )

        # nf * nloc * angle_dim
        sub_node_update = torch.matmul(
            node_ebd, matrix[sub_node_idx[0] : sub_node_idx[1]]
        )
        # nf * nloc * a_nnei * angle_dim
        sub_edge_update_ik = torch.matmul(
            edge_ebd, matrix[sub_edge_idx_ik[0] : sub_edge_idx_ik[1]]
        )
        sub_edge_update_ij = torch.matmul(
            edge_ebd, matrix[sub_edge_idx_ij[0] : sub_edge_idx_ij[1]]
        )

        result_update = (
            sub_angle_update
            + sub_node_update[:, :, None, None, :]
            + sub_edge_update_ik[:, :, None, :, :]
            + sub_edge_update_ij[:, :, :, None, :]
        ) + bias
        return result_update

    def optim_angle_update_dynamic(
        self,
        flat_angle_ebd: torch.Tensor,
        node_ebd: torch.Tensor,
        flat_edge_ebd: torch.Tensor,
        n2a_index: torch.Tensor,
        eij2a_index: torch.Tensor,
        eik2a_index: torch.Tensor,
        feat: str = "edge",
    ) -> torch.Tensor:
        nf, nloc, node_dim = node_ebd.shape
        angle_dim = flat_angle_ebd.shape[-1]
        edge_dim = flat_edge_ebd.shape[-1]
        sub_angle_idx = (0, angle_dim)
        sub_node_idx = (angle_dim, angle_dim + node_dim)
        sub_edge_idx_ik = (angle_dim + node_dim, angle_dim + node_dim + edge_dim)
        sub_edge_idx_ij = (
            angle_dim + node_dim + edge_dim,
            angle_dim + node_dim + 2 * edge_dim,
        )

        if feat == "edge":
            matrix, bias = self.edge_angle_linear1.matrix, self.edge_angle_linear1.bias
        elif feat == "angle":
            matrix, bias = self.angle_self_linear.matrix, self.angle_self_linear.bias
        else:
            raise NotImplementedError
        assert angle_dim + node_dim + 2 * edge_dim == matrix.size()[0]

        # n_angle * angle_dim
        sub_angle_update = torch.matmul(
            flat_angle_ebd, matrix[sub_angle_idx[0] : sub_angle_idx[1]]
        )

        # nf * nloc * angle_dim
        sub_node_update = torch.matmul(
            node_ebd, matrix[sub_node_idx[0] : sub_node_idx[1]]
        )
        # n_angle * angle_dim
        sub_node_update = torch.index_select(
            sub_node_update.reshape(nf * nloc, -1), 0, n2a_index
        )

        # n_edge * angle_dim
        sub_edge_update_ik = torch.matmul(
            flat_edge_ebd, matrix[sub_edge_idx_ik[0] : sub_edge_idx_ik[1]]
        )
        sub_edge_update_ij = torch.matmul(
            flat_edge_ebd, matrix[sub_edge_idx_ij[0] : sub_edge_idx_ij[1]]
        )
        # n_angle * angle_dim
        sub_edge_update_ik = torch.index_select(sub_edge_update_ik, 0, eik2a_index)
        sub_edge_update_ij = torch.index_select(sub_edge_update_ij, 0, eij2a_index)

        result_update = (
            sub_angle_update + sub_node_update + sub_edge_update_ik + sub_edge_update_ij
        ) + bias
        return result_update

    def optim_angle_update_weight(
        self,
        flat_angle_ebd: torch.Tensor,
        node_ebd: torch.Tensor,
        node_ebd_ext: torch.Tensor,
        flat_edge_ebd: torch.Tensor,
        n2a_index: torch.Tensor,
        eij2a_index: torch.Tensor,
        eik2a_index: torch.Tensor,
        j2a_index: torch.Tensor,
        k2a_index: torch.Tensor,
        feat: str = "edge",
    ) -> torch.Tensor:
        nf, nloc, node_dim = node_ebd.shape
        nall = node_ebd_ext.shape[1]
        angle_dim = flat_angle_ebd.shape[-1]
        edge_dim = flat_edge_ebd.shape[-1]
        sub_angle_idx = (0, angle_dim)
        sub_node_idx = (angle_dim, angle_dim + node_dim)
        sub_edge_idx_ik = (angle_dim + node_dim, angle_dim + node_dim + edge_dim)
        sub_edge_idx_ij = (
            angle_dim + node_dim + edge_dim,
            angle_dim + node_dim + 2 * edge_dim,
        )

        if feat == "edge":
            matrix, bias = self.edge_angle_linear1.matrix, self.edge_angle_linear1.bias
        elif feat == "angle":
            matrix, bias = self.angle_self_linear.matrix, self.angle_self_linear.bias
        else:
            raise NotImplementedError
        assert angle_dim + node_dim + 2 * edge_dim == matrix.size()[0]

        # calculate the edge weight
        

        # n_angle * angle_dim
        sub_angle_update = torch.matmul(
            flat_angle_ebd, matrix[sub_angle_idx[0] : sub_angle_idx[1]]
        )

        # nf * nloc * angle_dim
        sub_node_update = torch.matmul(
            node_ebd, matrix[sub_node_idx[0] : sub_node_idx[1]]
        )
        sub_extended_node_update = torch.matmul(
            node_ebd_ext, matrix[sub_node_idx[0] : sub_node_idx[1]]
        )
        # n_angle * angle_dim
        sub_node_update = torch.index_select(
            sub_node_update.reshape(nf * nloc, -1), 0, n2a_index
        )
        sub_node_j_update = torch.index_select(
            sub_extended_node_update.reshape(nf * nall, -1), 0, j2a_index
        )
        sub_node_k_update = torch.index_select(
            sub_extended_node_update.reshape(nf * nall, -1), 0, k2a_index
        )

        # n_edge * angle_dim
        sub_edge_update_ik = torch.matmul(
            flat_edge_ebd, matrix[sub_edge_idx_ik[0] : sub_edge_idx_ik[1]]
        )
        sub_edge_update_ij = torch.matmul(
            flat_edge_ebd, matrix[sub_edge_idx_ij[0] : sub_edge_idx_ij[1]]
        )
        # n_angle * angle_dim
        sub_edge_update_ik = torch.index_select(sub_edge_update_ik, 0, eik2a_index)
        sub_edge_update_ij = torch.index_select(sub_edge_update_ij, 0, eij2a_index)

        edge_feat_1 = torch.cat([sub_node_j_update, sub_edge_update_ij], dim=-1)
        edge_feat_2 = torch.cat([sub_node_k_update, sub_edge_update_ik], dim=-1)
        edge_weight_1 = self.edge_weight_proj_1(edge_feat_1)
        edge_weight_2 = self.edge_weight_proj_2(edge_feat_2)
        edge_weight = F.softmax(torch.cat([edge_weight_1, edge_weight_2], dim=-1), dim=-1)
        edge_weight_1, edge_weight_2 = edge_weight.chunk(2, dim=-1)
        
        result_update = (
            sub_angle_update + sub_node_update + edge_weight_1 * sub_edge_update_ik + edge_weight_2 * sub_edge_update_ij
        ) + bias
        return result_update


    def optim_torsion_update(
        self,
        torsion_ebd: torch.Tensor,
        node_ebd: torch.Tensor,
        edge_ebd: torch.Tensor,
        feat: str = "angle",
    ) -> torch.Tensor:
        pass

    def optim_torsion_update_dynamic(
        self,
        torsion_ebd: torch.Tensor,
        node_ebd: torch.Tensor,
        flat_edge_ebd: torch.Tensor,
        torsion_index: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        torsion_mask: Optional[torch.Tensor] = None,
        feat: str = "edge",
    ) -> torch.Tensor:
        assert torsion_index is not None
        assert torsion_mask is not None
        
        nf, nloc, node_dim = node_ebd.shape
        angle_dim = torsion_ebd.shape[-1]
        edge_dim = flat_edge_ebd.shape[-1]
        sub_angle_idx = (0, angle_dim)
        sub_node_idx_i = (angle_dim, angle_dim + node_dim)
        sub_node_idx_j = (angle_dim + node_dim, angle_dim + 2 * node_dim)
        sub_node_idx_ij = (angle_dim + 2 * node_dim, angle_dim + 2 * node_dim+edge_dim)
        sub_edge_idx_iref_i = (angle_dim + 2 * node_dim+edge_dim, angle_dim + 2 * node_dim + 2*edge_dim)
        sub_edge_idx_jref_i = (
            angle_dim + 2 * node_dim + 2*edge_dim,
            angle_dim + 2 * node_dim + 3*edge_dim,
        )
        
        i2t_index, j2t_index, ij2t_index, i_iref2t_index, j_jref2t_index = torsion_index
        if feat == "edge":
            matrix, bias = self.edge_torsion_linear1.matrix, self.edge_torsion_linear1.bias
        elif feat == "torsion":
            matrix, bias = self.torsion_self_linear.matrix, self.torsion_self_linear.bias
        else:
            raise NotImplementedError
        
        assert angle_dim + 2 * node_dim + 3*edge_dim == matrix.size()[0]
        
        # n_angle * angle_dim
        torsion_update = torch.matmul(
            torsion_ebd, matrix[sub_angle_idx[0] : sub_angle_idx[1]]
        )
        node_update_i = torch.zeros_like(torsion_update)
        node_update_j = torch.zeros_like(torsion_update)
        edge_update_ij = torch.zeros_like(torsion_update)
        edge_update_iref_i = torch.zeros_like(torsion_update)
        edge_update_jref_j = torch.zeros_like(torsion_update)

        # nf * nloc * angle_dim
        sub_node_update_i = torch.matmul(
            node_ebd, matrix[sub_node_idx_i[0] : sub_node_idx_i[1]]
        )
        sub_node_update_j = torch.matmul(
            node_ebd, matrix[sub_node_idx_j[0] : sub_node_idx_j[1]]
        )
        sub_edge_update_ij = torch.matmul(
            flat_edge_ebd, matrix[sub_node_idx_ij[0] : sub_node_idx_ij[1]]
        )
        sub_edge_update_iref_i = torch.matmul(
            flat_edge_ebd, matrix[sub_edge_idx_iref_i[0] : sub_edge_idx_iref_i[1]]
        )
        sub_edge_update_jref_j = torch.matmul(
            flat_edge_ebd, matrix[sub_edge_idx_jref_i[0] : sub_edge_idx_jref_i[1]]
        )
        # n_angle * angle_dim
        node_update_i[torsion_mask] = torch.index_select(
            sub_node_update_i.reshape(nf * nloc, -1), 0, i2t_index
        )
        node_update_j[torsion_mask] = torch.index_select(
            sub_node_update_j.reshape(nf * nloc, -1), 0, j2t_index
        )
        edge_update_ij[torsion_mask] = torch.index_select(sub_edge_update_ij, 0, ij2t_index)
        edge_update_iref_i[torsion_mask] = torch.index_select(sub_edge_update_iref_i, 0, i_iref2t_index)
        edge_update_jref_j[torsion_mask] = torch.index_select(sub_edge_update_jref_j, 0, j_jref2t_index)
        
        result_update = (
            torsion_update + node_update_i + node_update_j + edge_update_ij + edge_update_iref_i + edge_update_jref_j
        ) + bias
        
        return result_update

    def optim_edge_update(
        self,
        node_ebd: torch.Tensor,
        node_ebd_ext: torch.Tensor,
        edge_ebd: torch.Tensor,
        nlist: torch.Tensor,
        feat: str = "node",
    ) -> torch.Tensor:
        node_dim = node_ebd.shape[-1]
        edge_dim = edge_ebd.shape[-1]
        sub_node_idx = (0, node_dim)
        sub_node_ext_idx = (node_dim, 2 * node_dim)
        sub_edge_idx = (2 * node_dim, 2 * node_dim + edge_dim)

        if feat == "node":
            matrix, bias = self.node_edge_linear.matrix, self.node_edge_linear.bias
        elif feat == "edge":
            matrix, bias = self.edge_self_linear.matrix, self.edge_self_linear.bias
        else:
            raise NotImplementedError
        assert 2 * node_dim + edge_dim == matrix.size()[0]

        # nf * nloc * node/edge_dim
        sub_node_update = torch.matmul(
            node_ebd, matrix[sub_node_idx[0] : sub_node_idx[1]]
        )

        # nf * nall * node/edge_dim
        sub_node_ext_update = torch.matmul(
            node_ebd_ext, matrix[sub_node_ext_idx[0] : sub_node_ext_idx[1]]
        )
        # nf * nloc * nnei * node/edge_dim
        sub_node_ext_update = _make_nei_g1(sub_node_ext_update, nlist)

        # nf * nloc * nnei * node/edge_dim
        sub_edge_update = torch.matmul(
            edge_ebd, matrix[sub_edge_idx[0] : sub_edge_idx[1]]
        )

        result_update = (
            sub_edge_update + sub_node_ext_update + sub_node_update[:, :, None, :]
        ) + bias
        return result_update

    def optim_edge_update_dynamic(
        self,
        node_ebd: torch.Tensor,
        node_ebd_ext: torch.Tensor,
        flat_edge_ebd: torch.Tensor,
        n2e_index: torch.Tensor,
        n_ext2e_index: torch.Tensor,
        feat: str = "node",
    ) -> torch.Tensor:
        nf, nall, node_dim = node_ebd_ext.shape
        _, nloc, _ = node_ebd.shape
        edge_dim = flat_edge_ebd.shape[-1]
        sub_node_idx = (0, node_dim)
        sub_node_ext_idx = (node_dim, 2 * node_dim)
        sub_edge_idx = (2 * node_dim, 2 * node_dim + edge_dim)

        if feat == "node":
            matrix, bias = self.node_edge_linear.matrix, self.node_edge_linear.bias
        elif feat == "edge":
            matrix, bias = self.edge_self_linear.matrix, self.edge_self_linear.bias
        else:
            raise NotImplementedError
        assert 2 * node_dim + edge_dim == matrix.size()[0]

        # nf * nloc * node/edge_dim
        sub_node_update = torch.matmul(
            node_ebd, matrix[sub_node_idx[0] : sub_node_idx[1]]
        )

        # n_edge * node/edge_dim
        sub_node_update = torch.index_select(
            sub_node_update.reshape(nf * nloc, -1), 0, n2e_index
        )

        # nf * nall * node/edge_dim
        sub_node_ext_update = torch.matmul(
            node_ebd_ext, matrix[sub_node_ext_idx[0] : sub_node_ext_idx[1]]
        )
        # n_edge * node/edge_dim
        sub_node_ext_update = torch.index_select(
            sub_node_ext_update.reshape(nf * nall, -1), 0, n_ext2e_index
        )

        # n_edge * node/edge_dim
        sub_edge_update = torch.matmul(
            flat_edge_ebd, matrix[sub_edge_idx[0] : sub_edge_idx[1]]
        )

        result_update = (sub_edge_update + sub_node_ext_update + sub_node_update) + bias
        return result_update

    def layer_norm_dim0(self,tensor,fn=None):
        # 获取张量的形状
        shape = tensor.shape
        # 调整维度顺序，将第 0 维移到最后
        permuted_tensor = tensor.permute(1, *range(2, len(shape)), 0)
        # 对调整后的张量应用 LayerNorm
        normalized_tensor = fn(permuted_tensor)
        # normalized_tensor = fn(permuted_tensor, permuted_tensor.shape[-1:])
        # 恢复原始的维度顺序
        return normalized_tensor.permute(-1, *range(len(shape) - 1))

    def forward(
        self,
        node_ebd_ext: torch.Tensor,  # nf x nall x n_dim
        edge_ebd: torch.Tensor,  # nf x nloc x nnei x e_dim
        h2: torch.Tensor,  # nf x nloc x nnei x 3
        angle_ebd: torch.Tensor,  # nf x nloc x a_nnei x a_nnei x a_dim
        nlist: torch.Tensor,  # nf x nloc x nnei
        nlist_mask: torch.Tensor,  # nf x nloc x nnei
        sw: torch.Tensor,  # switch func, nf x nloc x nnei
        a_nlist: torch.Tensor,  # nf x nloc x a_nnei
        a_nlist_mask: torch.Tensor,  # nf x nloc x a_nnei
        a_sw: torch.Tensor,  # switch func, nf x nloc x a_nnei
        edge_index: torch.Tensor,  # n_edge x 2
        angle_index: torch.Tensor,  # n_angle x 3
        d_nlist: Optional[torch.Tensor] = None,  # nf x nloc x d_nnei
        d_nlist_mask: Optional[torch.Tensor] = None,  # nf x nloc x d_nnei
        dihedral_index: Optional[torch.Tensor] = None,  # n_dihedral x 2
        dihedral_ebd: Optional[torch.Tensor] = None,  # n_dihedral x d_dim
        d_sw: Optional[torch.Tensor] = None,  # n_dihedral
        rbf_ebd: Optional[torch.Tensor] = None,  # n_edge x num_b
        torsion_ebd: Optional[torch.Tensor] = None,
        torsion_mask: Optional[torch.Tensor] = None,
        torsion_index: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        atom_feats_in: Optional[torch.Tensor] = None,
        edge_diff: Optional[torch.Tensor] = None,
        p3m_info: Optional[dict[str, torch.Tensor]] = None,
    ):
        """
        Parameters
        ----------
        node_ebd_ext : nf x nall x n_dim
            Extended node embedding.
        edge_ebd : nf x nloc x nnei x e_dim
            Edge embedding.
        h2 : nf x nloc x nnei x 3
            Pair-atom channel, equivariant.
        angle_ebd : nf x nloc x a_nnei x a_nnei x a_dim
            Angle embedding.
        nlist : nf x nloc x nnei
            Neighbor list. (padded neis are set to 0)
        nlist_mask : nf x nloc x nnei
            Masks of the neighbor list. real nei 1 otherwise 0
        sw : nf x nloc x nnei
            Switch function.
        a_nlist : nf x nloc x a_nnei
            Neighbor list for angle. (padded neis are set to 0)
        a_nlist_mask : nf x nloc x a_nnei
            Masks of the neighbor list for angle. real nei 1 otherwise 0
        a_sw : nf x nloc x a_nnei
            Switch function for angle.
        edge_index : Optional for dynamic sel, n_edge x 2
            n2e_index : n_edge
                Broadcast indices from node(i) to edge(ij), or reduction indices from edge(ij) to node(i).
            n_ext2e_index : n_edge
                Broadcast indices from extended node(j) to edge(ij).
        angle_index : Optional for dynamic sel, n_angle x 3
            n2a_index : n_angle
                Broadcast indices from extended node(j) to angle(ijk).
            eij2a_index : n_angle
                Broadcast indices from extended edge(ij) to angle(ijk), or reduction indices from angle(ijk) to edge(ij).
            eik2a_index : n_angle
                Broadcast indices from extended edge(ik) to angle(ijk).

        Returns
        -------
        n_updated:     nf x nloc x n_dim
            Updated node embedding.
        e_updated:     nf x nloc x nnei x e_dim
            Updated edge embedding.
        a_updated : nf x nloc x a_nnei x a_nnei x a_dim
            Updated angle embedding.
        """
        nb, nloc, nnei = nlist.shape
        nall = node_ebd_ext.shape[1]
        node_ebd, _ = torch.split(node_ebd_ext, [nloc, nall - nloc], dim=1)
        n_edge = int(nlist_mask.sum().item())
        assert (nb, nloc) == node_ebd.shape[:2]
        if not self.use_dynamic_sel:
            assert (nb, nloc, nnei, 3) == h2.shape
        else:
            assert (n_edge, 3) == h2.shape
        del a_nlist  # may be used in the future

        n2e_index, n_ext2e_index = edge_index[:, 0], edge_index[:, 1]
        n2a_index, eij2a_index, eik2a_index, j2a_index, k2a_index = (
            angle_index[:, 0],
            angle_index[:, 1],
            angle_index[:, 2],
            angle_index[:, 3],
            angle_index[:, 4],
        )

        # nb x nloc x nnei x n_dim [OR] n_edge x n_dim
        nei_node_ebd = (
            _make_nei_g1(node_ebd_ext, nlist)
            if not self.use_dynamic_sel
            else torch.index_select(
                node_ebd_ext.reshape(-1, self.n_dim), 0, n_ext2e_index
            )
        )

        if self.use_rbf:
            assert self.rbf_linear is not None
            assert rbf_ebd is not None
            edge_rbf_ebd = self.rbf_linear(rbf_ebd)
            edge_ebd = edge_ebd * edge_rbf_ebd
        else:
            edge_ebd = edge_ebd
        # handle edge rbf
        if self.edge_rbf_dot_self or self.edge_rbf_dot_message:
            assert rbf_ebd is not None
            assert self.rbf_mlp is not None
            edge_rbf = self.rbf_mlp(rbf_ebd)
        else:
            edge_rbf = None

        if self.edge_rbf_dot_message:
            assert rbf_ebd is not None
            assert self.rbf_mlp_message is not None
            edge_rbf_node = self.rbf_mlp_message(rbf_ebd)
        else:
            edge_rbf_node = None

        if self.edge_rbf_dot_self:
            assert edge_rbf is not None
            edge_ebd = edge_ebd * edge_rbf

        n_update_list: list[torch.Tensor] = [node_ebd]
        e_update_list: list[torch.Tensor] = [edge_ebd]
        a_update_list: list[torch.Tensor] = [angle_ebd]
        if torsion_ebd is not None:
            t_update_list = [torsion_ebd]
        else:
            t_update_list = None
        # node self mlp
        node_self_mlp = self.act(self.node_self_mlp(node_ebd))
        n_update_list.append(node_self_mlp)

        # node sym (grrg + drrd)
        node_sym_list: list[torch.Tensor] = []
        node_sym_list.append(
            self.symmetrization_op(
                edge_ebd,
                h2,
                nlist_mask,
                sw,
                self.axis_neuron,
            )
            if not self.use_dynamic_sel
            else self.symmetrization_op_dynamic(
                edge_ebd,
                h2,
                sw,
                owner=n2e_index,
                num_owner=nb * nloc,
                nloc=nloc,
                scale_factor=self.dynamic_e_sel ** (-0.5),
                axis_neuron=self.axis_neuron,
            )
        )
        node_sym_list.append(
            self.symmetrization_op(
                nei_node_ebd,
                h2,
                nlist_mask,
                sw,
                self.axis_neuron,
            )
            if not self.use_dynamic_sel
            else self.symmetrization_op_dynamic(
                nei_node_ebd,
                h2,
                sw,
                owner=n2e_index,
                num_owner=nb * nloc,
                nloc=nloc,
                scale_factor=self.dynamic_e_sel ** (-0.5),
                axis_neuron=self.axis_neuron,
            )
        )
        node_sym = self.act(self.node_sym_linear(torch.cat(node_sym_list, dim=-1)))
        n_update_list.append(node_sym)

        if not self.optim_update:
            if not self.use_dynamic_sel:
                # nb x nloc x nnei x (n_dim * 2 + e_dim)
                edge_info = torch.cat(
                    [
                        torch.tile(node_ebd.unsqueeze(-2), [1, 1, self.nnei, 1]),
                        nei_node_ebd,
                        edge_ebd,
                    ],
                    dim=-1,
                )
            else:
                # n_edge x (n_dim * 2 + e_dim)
                edge_info = torch.cat(
                    [
                        torch.index_select(
                            node_ebd.reshape(-1, self.n_dim), 0, n2e_index
                        ),
                        nei_node_ebd,
                        edge_ebd,
                    ],
                    dim=-1,
                )
            if self.use_ffn_node_edge_message or self.use_ffn_edge_edge_message:
                assert self.edge_message_ffn1 is not None
                edge_info_ffn = self.act(self.edge_message_ffn1(edge_info))
            else:
                edge_info_ffn = None
        else:
            edge_info = None
            edge_info_ffn = None

        # node edge message
        # nb x nloc x nnei x (h * n_dim)
        if not self.optim_update:
            assert edge_info is not None
            if not self.use_ffn_node_edge_message:
                node_edge_update = self.act(
                    self.node_edge_linear(edge_info)
                ) * sw.unsqueeze(-1)
            else:
                assert edge_info_ffn is not None
                node_edge_update = self.act(
                    self.node_edge_linear(edge_info_ffn)
                ) * sw.unsqueeze(-1)
        else:
            node_edge_update = self.act(
                self.optim_edge_update(
                    node_ebd,
                    node_ebd_ext,
                    edge_ebd,
                    nlist,
                    "node",
                )
                if not self.use_dynamic_sel
                else self.optim_edge_update_dynamic(
                    node_ebd,
                    node_ebd_ext,
                    edge_ebd,
                    n2e_index,
                    n_ext2e_index,
                    "node",
                )
            ) * sw.unsqueeze(-1)
        if self.edge_rbf_dot_message:
            assert edge_rbf_node is not None
            node_edge_update = node_edge_update * edge_rbf_node
        node_edge_update = (
            (torch.sum(node_edge_update, dim=-2) / self.nnei)
            if not self.use_dynamic_sel
            
            else (
                aggregate(
                    node_edge_update,
                    n2e_index,
                    average=False,
                    num_owner=nb * nloc,
                ).reshape(nb, nloc, -1)
                / self.dynamic_e_sel
            )
        )

        if self.n_multi_edge_message > 1:
            # nb x nloc x h x n_dim
            node_edge_update_mul_head = node_edge_update.view(
                nb, nloc, self.n_multi_edge_message, self.n_dim
            )
            for head_index in range(self.n_multi_edge_message):
                n_update_list.append(node_edge_update_mul_head[:, :, head_index, :])
        else:
            if self.message_use_self_concat:
                assert self.node_edge_linear_2 is not None
                node_edge_update = self.act(
                    self.node_edge_linear_2(
                        torch.cat([node_ebd, node_edge_update], dim=-1)
                    )
                )
            n_update_list.append(node_edge_update)
        # update node_ebd
        # TODO: add atomic moment
        # neighbor number
        num_neighbors_per_atom = nlist_mask.sum(dim=-1).float()  # shape: (nb, nloc)
        avg_num_neighbors = num_neighbors_per_atom.mean(dim=-1)  # shape: (nb,)
        num_average_neigh = avg_num_neighbors.mean().item()  # 标量值
        
        edge_index_real = torch.zeros_like(edge_index)
        for i in range(nb):
            start_idx = i * nloc
            end_idx = (i + 1) * nloc
            edge_index_mask = (edge_index[:, 0] >= start_idx) & (edge_index[:, 0] < end_idx)
            edge_index_real[edge_index_mask] = edge_index[edge_index_mask] % nloc + i * nloc
        
        if self.use_atomic_moment:
            if edge_index.numel() > 0 and num_average_neigh >= 4 and edge_index_real.max()==nb*nloc-1:
                assert self.atomic_moment_layer is not None
                assert self.hyper_moment_layer is not None

                
                if atom_feats_in is None:
                    temp_node_ebd = self.atom_feats_in_transform(node_ebd)
                    atom_feats_in = {0: temp_node_ebd.clone().view(-1, temp_node_ebd.shape[-1]).permute(1, 0)} 
                
                am = self.atomic_moment_layer(atom_feats_in,edge_index_real,rbf_ebd,edge_diff,num_average_neigh,sw)
                # 检查am字典中是否存在NaN值
                
                hm = self.hyper_moment_layer(am)

                for rank, m in hm.items():
                    fn = self.linear_channel_hyper[str(rank)]
                    hm[rank] = fn.forward(m,dims=0)
                out = hm
                if self.mix_atom_feats_radial_channel:
                    max_rank = min(self.max_atom_feats_rank, self.max_out_rank)
                    for rank in range(max_rank + 1):
                        fn = self.linear_channel_feats[str(rank)]
                        out[rank] = out[rank] + fn.forward(atom_feats_in[rank],dims=0)
                
                
                for key,value in out.items():
                    out[key] = self.layer_norm_dim0(value,self.moment_layer_norm[str(key)])

                moment_ebd = out[0].permute(1,0).view(nb,nloc,-1)

                moment_ebd = self.atom_feats_out_transform(moment_ebd)
                
                n_update_list.append(moment_ebd)
            else:
                out = None
                n_update_list.append(torch.zeros_like(node_ebd))
        # edge self message
        else:
            out = None
     
        if not self.optim_update:
            assert edge_info is not None
            if not self.use_ffn_edge_edge_message:
                edge_self_update = self.act(self.edge_self_linear(edge_info))
            else:
                assert edge_info_ffn is not None
                edge_self_update = self.act(self.edge_self_linear(edge_info_ffn))
        else:
            edge_self_update = self.act(
                self.optim_edge_update(
                    node_ebd,
                    node_ebd_ext,
                    edge_ebd,
                    nlist,
                    "edge",
                )
                if not self.use_dynamic_sel
                else self.optim_edge_update_dynamic(
                    node_ebd,
                    node_ebd_ext,
                    edge_ebd,
                    n2e_index,
                    n_ext2e_index,
                    "edge",
                )
            )
        if self.edge_rbf_dot_message:
            assert edge_rbf is not None
            edge_self_update = edge_self_update * edge_rbf
        e_update_list.append(edge_self_update)

        # edge attention message
        if self.edge_use_attn:
            # gated_attention(g2, h2)
            assert self.edge_attn_map is not None
            assert self.edge_mh_apply is not None
            assert self.edge_lm is not None
            # nf x nloc x nnei x nnei x nh
            attention_weights = self.edge_attn_map(edge_ebd, h2, nlist_mask, sw)
            # nf x nloc x nnei x e_dim
            edge_attention_update = self.edge_mh_apply(attention_weights, edge_ebd)
            if self.edge_attn_use_ln:
                edge_attention_update = self.edge_lm(edge_attention_update)
            e_update_list.append(edge_attention_update)
        
        if self.update_angle:
            assert self.angle_self_linear is not None
            assert self.edge_angle_linear1 is not None
            # assert self.edge_angle_linear2 is not None
            # get angle info
            if self.a_compress_rate != 0:
                if not self.a_compress_use_split:
                    assert self.a_compress_n_linear is not None
                    assert self.a_compress_e_linear is not None
                    node_ebd_for_angle = self.a_compress_n_linear(node_ebd)
                    node_ebd_for_angle_ext = self.a_compress_n_linear(node_ebd_ext)
                    edge_ebd_for_angle = self.a_compress_e_linear(edge_ebd)
                else:
                    # use the first a_compress_dim dim for node and edge
                    node_ebd_for_angle = node_ebd[:, :, : self.n_a_compress_dim]
                    node_ebd_for_angle_ext = node_ebd_ext[:, :, : self.n_a_compress_dim]
                    edge_ebd_for_angle = edge_ebd[..., : self.e_a_compress_dim]
            else:
                node_ebd_for_angle = node_ebd
                node_ebd_for_angle_ext = node_ebd_ext
                edge_ebd_for_angle = edge_ebd

            if not self.use_dynamic_sel:
                # nb x nloc x a_nnei x e_dim
                edge_ebd_for_angle = edge_ebd_for_angle[:, :, : self.a_sel, :]
                # nb x nloc x a_nnei x e_dim
                edge_ebd_for_angle = torch.where(
                    a_nlist_mask.unsqueeze(-1), edge_ebd_for_angle, 0.0
                )
            if not self.optim_update:
                # nb x nloc x a_nnei x a_nnei x n_dim [OR] n_angle x n_dim
                node_for_angle_info = (
                    torch.tile(
                        node_ebd_for_angle.unsqueeze(2).unsqueeze(2),
                        (1, 1, self.a_sel, self.a_sel, 1),
                    )
                    if not self.use_dynamic_sel
                    else torch.index_select(
                        node_ebd_for_angle.reshape(-1, self.n_a_compress_dim),
                        0,
                        n2a_index,
                    )
                )

                # nb x nloc x (a_nnei) x a_nnei x e_dim [OR] n_angle x e_dim
                edge_for_angle_k = (
                    torch.tile(
                        edge_ebd_for_angle.unsqueeze(2), (1, 1, self.a_sel, 1, 1)
                    )
                    if not self.use_dynamic_sel
                    else torch.index_select(edge_ebd_for_angle, 0, eik2a_index)
                )
                # nb x nloc x a_nnei x (a_nnei) x e_dim [OR] n_angle x e_dim
                edge_for_angle_j = (
                    torch.tile(
                        edge_ebd_for_angle.unsqueeze(3), (1, 1, 1, self.a_sel, 1)
                    )
                    if not self.use_dynamic_sel
                    else torch.index_select(edge_ebd_for_angle, 0, eij2a_index)
                )
                # nb x nloc x a_nnei x a_nnei x (e_dim + e_dim) [OR] n_angle x (e_dim + e_dim)
                edge_for_angle_info = torch.cat(
                    [edge_for_angle_k, edge_for_angle_j], dim=-1
                )
                angle_info_list = [angle_ebd]
                angle_info_list.append(node_for_angle_info)
                angle_info_list.append(edge_for_angle_info)
                # nb x nloc x a_nnei x a_nnei x (a + n_dim + e_dim*2) or (a + a/c + a/c)
                # [OR]
                # n_angle x (a + n_dim + e_dim*2) or (a + a/c + a/c)
                angle_info = torch.cat(angle_info_list, dim=-1)
                if self.use_ffn_edge_angle_message or self.use_ffn_angle_angle_message:
                    assert self.angle_message_ffn1 is not None
                    angle_info_ffn = self.act(self.angle_message_ffn1(angle_info))
                else:
                    angle_info_ffn = None
            else:
                angle_info = None
                angle_info_ffn = None

            # edge angle message
            # nb x nloc x a_nnei x a_nnei x e_dim [OR] n_angle x e_dim
            if not self.optim_update:
                assert angle_info is not None
                if not self.use_ffn_edge_angle_message:
                    edge_angle_update = self.act(self.edge_angle_linear1(angle_info))
                else:
                    assert angle_info_ffn is not None
                    edge_angle_update = self.act(
                        self.edge_angle_linear1(angle_info_ffn)
                    )
            else:
                edge_angle_update = self.act(
                    self.optim_angle_update(
                        angle_ebd,
                        node_ebd_for_angle,
                        edge_ebd_for_angle,
                        "edge",
                    )
                    if not self.use_dynamic_sel
                    else self.optim_angle_update_dynamic(
                        angle_ebd,
                        node_ebd_for_angle,
                        edge_ebd_for_angle,
                        n2a_index,
                        eij2a_index,
                        eik2a_index,
                        "edge",
                    )
                )

            if not self.use_dynamic_sel:
                # nb x nloc x a_nnei x a_nnei x e_dim
                weighted_edge_angle_update = (
                    edge_angle_update
                    * a_sw[:, :, :, None, None]
                    * a_sw[:, :, None, :, None]
                )
                # nb x nloc x a_nnei x e_dim
                reduced_edge_angle_update = torch.sum(
                    weighted_edge_angle_update, dim=-2
                ) / (self.a_sel**0.5)
                # nb x nloc x nnei x e_dim
                padding_edge_angle_update = torch.concat(
                    [
                        reduced_edge_angle_update,
                        torch.zeros(
                            [nb, nloc, self.nnei - self.a_sel, self.e_dim],
                            dtype=edge_ebd.dtype,
                            device=edge_ebd.device,
                        ),
                    ],
                    dim=2,
                )
            else:
                # n_angle x e_dim
                weighted_edge_angle_update = edge_angle_update * a_sw.unsqueeze(-1)
                # n_edge x e_dim
                padding_edge_angle_update = aggregate(
                    weighted_edge_angle_update,
                    eij2a_index,
                    average=False,
                    num_owner=n_edge,
                ) / (self.dynamic_a_sel**0.5)
            if not self.smooth_edge_update:
                # will be deprecated in the future
                # not support dynamic index, will pass anyway
                if self.use_dynamic_sel:
                    raise NotImplementedError(
                        "smooth_edge_update must be True when use_dynamic_sel is True!"
                    )
                full_mask = torch.concat(
                    [
                        a_nlist_mask,
                        torch.zeros(
                            [nb, nloc, self.nnei - self.a_sel],
                            dtype=a_nlist_mask.dtype,
                            device=a_nlist_mask.device,
                        ),
                    ],
                    dim=-1,
                )
                padding_edge_angle_update = torch.where(
                    full_mask.unsqueeze(-1), padding_edge_angle_update, edge_ebd
                )
            if self.message_use_self_concat:
                padding_edge_angle_update = torch.cat(
                    [edge_ebd, padding_edge_angle_update], dim=-1
                )
            if not self.use_slim_message:
                assert self.edge_angle_linear2 is not None
                e_update_list.append(
                    self.act(self.edge_angle_linear2(padding_edge_angle_update))
                )
            else:
                e_update_list.append(padding_edge_angle_update)
            # update edge_ebd
            

            # angle self message
            # nb x nloc x a_nnei x a_nnei x dim_a
            if not self.optim_update:
                assert angle_info is not None
                if not self.use_ffn_angle_angle_message:
                    angle_self_update = self.act(self.angle_self_linear(angle_info))
                else:
                    assert angle_info_ffn is not None
                    angle_self_update = self.act(self.angle_self_linear(angle_info_ffn))
            else:
                if self.use_angle_weight:
                    angle_self_update = self.act(
                        self.optim_angle_update_weight(
                            angle_ebd,
                            node_ebd_for_angle,
                            node_ebd_for_angle_ext,
                            edge_ebd_for_angle,
                            n2a_index,
                            eij2a_index,
                            eik2a_index,
                            j2a_index,
                            k2a_index,
                            "angle",
                        )
                    )
                else:
                    angle_self_update = self.act(
                        self.optim_angle_update(
                            angle_ebd,
                            node_ebd_for_angle,
                            edge_ebd_for_angle,
                            "angle",
                        )
                        if not self.use_dynamic_sel
                        else self.optim_angle_update_dynamic(
                            angle_ebd,
                            node_ebd_for_angle,
                            edge_ebd_for_angle,
                            n2a_index,
                            eij2a_index,
                            eik2a_index,
                            "angle",
                        )
                    )
            a_update_list.append(angle_self_update)



            # dihedral update with fixed sel
            if self.update_dihedral and not self.use_dynamic_sel:
                assert d_nlist is not None
                assert d_nlist_mask is not None
                assert dihedral_ebd is not None
                assert d_sw is not None
                assert self.angle_dihedral_linear is not None

                # nb x nloc x d_sel x d_sel x e_dim
                angle_ebd_for_dihedral = angle_ebd[:, :, :self.d_sel, :self.d_sel, :]
                # nb x nloc x d_sel x d_sel x e_dim
                d_nlist_mask = d_nlist_mask[:,:,:,None] * d_nlist_mask[:,:,None,:]
                angle_ebd_for_dihedral = torch.where(
                    d_nlist_mask.unsqueeze(-1), angle_ebd_for_dihedral, 0.0
                )

                # nb x nloc x d_sel x d_sel x d_sel x a_dim
                angle_dihedral_update = self.act(
                    self.optim_dihedral_update(
                        dihedral_ebd,
                        angle_ebd_for_dihedral,
                        "angle",
                    )
                )              
                # nb x nloc x d_sel x d_sel x d_sel x a_dim
                weighted_angle_dihedral_update = (
                    angle_dihedral_update
                    * d_sw[:, :, :, None, None, None]
                    * d_sw[:, :, None, :, None, None]
                    * d_sw[:, :, None, None, :, None]
                )
                # nb x nloc x d_sel x d_sel x a_dim
                reduced_angle_dihedral_update = torch.sum(
                    weighted_angle_dihedral_update, dim=-2
                ) / (self.d_sel**0.5)

                # Need two dimensional padding
                # nb x nloc x a_sel x a_sel x a_dim
                padding_angle_dihedral_update = torch.concat(
                    [
                        reduced_angle_dihedral_update,
                        torch.zeros(
                            [nb, nloc, self.d_sel, self.a_sel - self.d_sel, self.a_dim],
                            dtype=edge_ebd.dtype,
                            device=edge_ebd.device,
                        ),
                    ],
                    dim=-2,
                )
                padding_angle_dihedral_update = torch.concat(
                    [
                        padding_angle_dihedral_update,
                        torch.zeros(
                            [nb, nloc, self.a_sel-self.d_sel, self.a_sel, self.a_dim],
                            dtype=edge_ebd.dtype,
                            device=edge_ebd.device,
                        ),
                    ],
                    dim=-3,
                )
                a_update_list.append(padding_angle_dihedral_update)

                dihedral_self_update = self.act(
                    self.optim_dihedral_update(
                        dihedral_ebd,
                        angle_ebd_for_dihedral,
                        "dihedral",
                    )
                )

                d_update_list: list[torch.Tensor] = [dihedral_ebd, dihedral_self_update]
                d_updated = self.list_update(d_update_list, "dihedral")

            # dihedral update with dynamic sel
            elif self.update_dihedral and self.use_dynamic_sel:
                n_angle = int(a_nlist_mask.sum().item())
                assert dihedral_ebd is not None
                assert d_sw is not None
                assert dihedral_index is not None
                assert self.angle_dihedral_linear is not None
                assert self.dihedral_self_linear is not None
                aijk2d_index, aijl2d_index = dihedral_index[:, 0], dihedral_index[:, 1]

                # n_dihedral x a_dim
                angle_for_dihedral_k = torch.index_select(angle_ebd, 0, aijk2d_index)
                angle_for_dihedral_l = torch.index_select(angle_ebd, 0, aijl2d_index)

                # n_dihedral x (d + a + a)
                dihedral_info = torch.cat(
                    [dihedral_ebd, angle_for_dihedral_k, angle_for_dihedral_l], dim=-1
                )

                # angle dihedral message
                # n_dihedral x a_dim
                angle_dihedral_update = self.act(
                    self.angle_dihedral_linear(dihedral_info)
                ) * d_sw.unsqueeze(-1)
                # n_angle x a_dim
                padding_angle_dihedral_update = aggregate(
                    angle_dihedral_update,
                    aijk2d_index,
                    average=False,
                    num_owner=n_angle,
                ) / (self.dynamic_d_sel**0.5)
                a_update_list.append(padding_angle_dihedral_update)

                # dihedral self update
                # n_dihedral x d_dim
                dihedral_self_update = self.act(
                    self.dihedral_self_linear(dihedral_info)
                )
                d_update_list: list[torch.Tensor] = [dihedral_ebd, dihedral_self_update]
                d_updated = self.list_update(d_update_list, "dihedral")
            else:
                d_updated = dihedral_ebd
        else:
            # update edge_ebd
            e_updated = self.list_update(e_update_list, "edge")
            d_updated = dihedral_ebd
        
        if torsion_ebd is not None and self.use_torsion:
            assert self.t_compress_e_linear is not None
            assert t_update_list is not None
            assert torsion_index is not None
            assert torsion_mask is not None
            node_ebd_for_torsion = self.t_compress_n_linear(node_ebd)
            edge_ebd_for_torsion = self.t_compress_e_linear(edge_ebd)
            torsion_self_update = self.act(
                self.optim_angle_update(
                    torsion_ebd,
                    node_ebd_for_torsion,
                    edge_ebd_for_torsion,
                    "angle",
                )
                if not self.use_dynamic_sel
                else self.optim_torsion_update_dynamic(
                    torsion_ebd,
                    node_ebd_for_torsion,
                    edge_ebd_for_torsion,
                    torsion_index,
                    torsion_mask,
                    "torsion",
                )
            )
            t_update_list.append(torsion_self_update)

            edge_torsion_update = self.act(
                self.optim_angle_update(
                    torsion_ebd,
                    node_ebd_for_torsion,
                    edge_ebd_for_torsion,
                    "edge",
                )
                if not self.use_dynamic_sel
                else self.optim_torsion_update_dynamic(
                    torsion_ebd,
                    node_ebd_for_torsion,
                    edge_ebd_for_torsion,
                    torsion_index,
                    torsion_mask,
                    "edge",
                )
            )
            e_update_list.append(edge_torsion_update)
            

        else:
            pass

        if p3m_info is not None:
            mesh_ebd = p3m_info["m_x"]
            a2m_edge_index = p3m_info["a2m_edge_index"]
            m2a_edge_index = p3m_info["m2a_edge_index"]
            atom_mesh_distance = p3m_info["atom_mesh_distance"].unsqueeze(-1)
            mesh_sw = p3m_info["mesh_sw"]
            n_am_edge = atom_mesh_distance.shape[0]
            am_diff_ebd = self.atom_mesh_rbf(atom_mesh_distance)
            am_diff_ebd = self.atom_mesh_distance_linear(am_diff_ebd).view(n_am_edge, -1)
            
            new_mesh_ebd = self.long_mp(mesh_ebd,nb)
            
            new_mesh_ebd = self.m2a_linear(new_mesh_ebd)
            a2m_edge_ebd = torch.index_select(self.a2m_linear(node_ebd).view(nb*nloc, -1), 0, a2m_edge_index[0])
            m2a_edge_ebd = torch.index_select(new_mesh_ebd, 0, m2a_edge_index[0])
            
            a2m_node_ebd = scatter(a2m_edge_ebd * am_diff_ebd * mesh_sw.unsqueeze(-1),m2a_edge_index[0],dim=0,reduce='sum',dim_size=nb * 3**3).reshape(nb, 3**3, -1) / self.dynamic_e_sel
            m2a_node_ebd = scatter(m2a_edge_ebd * am_diff_ebd * mesh_sw.unsqueeze(-1),a2m_edge_index[0],dim=0,reduce='sum',dim_size=nb * nloc).reshape(nb, nloc, -1) / self.dynamic_e_sel
            
            # a2m_node_ebd = scatter(a2m_edge_ebd,m2a_edge_index[0],dim=0,reduce='sum',dim_size=nb * 1**3).reshape(nb, 1**3, -1) / self.dynamic_e_sel
            # m2a_node_ebd = scatter(m2a_edge_ebd,a2m_edge_index[0],dim=0,reduce='sum',dim_size=nb * nloc).reshape(nb, nloc, -1) / self.dynamic_e_sel
            
            l_update_list: list[torch.Tensor] = [mesh_ebd.reshape(nb,-1, self.n_dim), a2m_node_ebd,new_mesh_ebd.reshape(nb,-1, self.n_dim)]
            for i in range(len(l_update_list)):
                l_update_list[i] = l_update_list[i].to(dtype=node_ebd.dtype)
            n_update_list.append(m2a_node_ebd.to(dtype=node_ebd.dtype))
        else:
            l_update_list = None
        
        # update angle_ebd
        n_updated = self.list_update(n_update_list, "node")
        e_updated = self.list_update(e_update_list, "edge")
        a_updated = self.list_update(a_update_list, "angle")
        if t_update_list is not None:
            t_updated = self.list_update(t_update_list, "torsion")
            # t_updated = t_update_list[0]
        else:
            t_updated = None
        if l_update_list is not None:
            l_updated = self.list_update(l_update_list, "mesh")
        else:
            l_updated = None
        # print(f"layer time: {end_layer_time - start_layer_time}")
        # import pdb;pdb.set_trace()
        return n_updated, e_updated, a_updated, d_updated, t_updated,l_updated,out

    @torch.jit.export
    def list_update_res_avg(
        self,
        update_list: list[torch.Tensor],
    ) -> torch.Tensor:
        nitem = len(update_list)
        uu = update_list[0]
        for ii in range(1, nitem):
            uu = uu + update_list[ii]
        return uu / (float(nitem) ** 0.5)

    @torch.jit.export
    def list_update_res_incr(self, update_list: list[torch.Tensor]) -> torch.Tensor:
        nitem = len(update_list)
        uu = update_list[0]
        scale = 1.0 / (float(nitem - 1) ** 0.5) if nitem > 1 else 0.0
        for ii in range(1, nitem):
            uu = uu + scale * update_list[ii]
        return uu

    @torch.jit.export
    def list_update_res_residual(
        self, update_list: list[torch.Tensor], update_name: str = "node"
    ) -> torch.Tensor:
        nitem = len(update_list)
        uu = update_list[0]
        # make jit happy
        if update_name == "node":
            for ii, vv in enumerate(self.n_residual):
                uu = uu + vv * update_list[ii + 1]
        elif update_name == "edge":
            for ii, vv in enumerate(self.e_residual):
                uu = uu + vv * update_list[ii + 1]
        elif update_name == "angle":
            for ii, vv in enumerate(self.a_residual):
                uu = uu + vv * update_list[ii + 1]
        elif update_name == "dihedral":
            for ii, vv in enumerate(self.d_residual):
                uu = uu + vv * update_list[ii + 1]
        elif update_name == "torsion":
            for idx, residual in enumerate(self.t_residual):
                if idx < nitem:
                    uu = uu + residual * update_list[idx + 1]
        elif update_name == "mesh":
            for ii, vv in enumerate(self.l_residual):
                uu = uu + vv * update_list[ii + 1]
        else:
            raise NotImplementedError
        return uu

    @torch.jit.export
    def list_update(
        self, update_list: list[torch.Tensor], update_name: str = "node"
    ) -> torch.Tensor:
        if self.update_style == "res_avg":
            return self.list_update_res_avg(update_list)
        elif self.update_style == "res_incr":
            return self.list_update_res_incr(update_list)
        elif self.update_style == "res_residual":
            return self.list_update_res_residual(update_list, update_name=update_name)
        else:
            raise RuntimeError(f"unknown update style {self.update_style}")

    def serialize(self) -> dict:
        """Serialize the networks to a dict.

        Returns
        -------
        dict
            The serialized networks.
        """
        data = {
            "@class": "RepformerLayer",
            "@version": 1,
            "e_rcut": self.e_rcut,
            "e_rcut_smth": self.e_rcut_smth,
            "e_sel": self.e_sel,
            "a_rcut": self.a_rcut,
            "a_rcut_smth": self.a_rcut_smth,
            "a_sel": self.a_sel,
            "ntypes": self.ntypes,
            "n_dim": self.n_dim,
            "e_dim": self.e_dim,
            "a_dim": self.a_dim,
            "a_compress_rate": self.a_compress_rate,
            "a_compress_e_rate": self.a_compress_e_rate,
            "a_compress_use_split": self.a_compress_use_split,
            "n_multi_edge_message": self.n_multi_edge_message,
            "axis_neuron": self.axis_neuron,
            "activation_function": self.activation_function,
            "update_angle": self.update_angle,
            "update_style": self.update_style,
            "update_residual": self.update_residual,
            "update_residual_init": self.update_residual_init,
            "precision": self.precision,
            "node_self_mlp": self.node_self_mlp.serialize(),
            "node_sym_linear": self.node_sym_linear.serialize(),
            "node_edge_linear": self.node_edge_linear.serialize(),
            "edge_self_linear": self.edge_self_linear.serialize(),
        }
        if self.update_angle:
            data.update(
                {
                    "edge_angle_linear1": self.edge_angle_linear1.serialize(),
                    "edge_angle_linear2": self.edge_angle_linear2.serialize(),
                    "angle_self_linear": self.angle_self_linear.serialize(),
                }
            )
            if self.a_compress_rate != 0 and not self.a_compress_use_split:
                data.update(
                    {
                        "a_compress_n_linear": self.a_compress_n_linear.serialize(),
                        "a_compress_e_linear": self.a_compress_e_linear.serialize(),
                    }
                )
        if self.update_style == "res_residual":
            data.update(
                {
                    "@variables": {
                        "n_residual": [to_numpy_array(t) for t in self.n_residual],
                        "e_residual": [to_numpy_array(t) for t in self.e_residual],
                        "a_residual": [to_numpy_array(t) for t in self.a_residual],
                    }
                }
            )
        return data

    @classmethod
    def deserialize(cls, data: dict) -> "RepFlowLayer":
        """Deserialize the networks from a dict.

        Parameters
        ----------
        data : dict
            The dict to deserialize from.
        """
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 1, 1)
        data.pop("@class")
        update_angle = data["update_angle"]
        a_compress_rate = data["a_compress_rate"]
        a_compress_use_split = data["a_compress_use_split"]
        node_self_mlp = data.pop("node_self_mlp")
        node_sym_linear = data.pop("node_sym_linear")
        node_edge_linear = data.pop("node_edge_linear")
        edge_self_linear = data.pop("edge_self_linear")
        edge_angle_linear1 = data.pop("edge_angle_linear1", None)
        edge_angle_linear2 = data.pop("edge_angle_linear2", None)
        angle_self_linear = data.pop("angle_self_linear", None)
        a_compress_n_linear = data.pop("a_compress_n_linear", None)
        a_compress_e_linear = data.pop("a_compress_e_linear", None)
        update_style = data["update_style"]
        variables = data.pop("@variables", {})
        n_residual = variables.get("n_residual", data.pop("n_residual", []))
        e_residual = variables.get("e_residual", data.pop("e_residual", []))
        a_residual = variables.get("a_residual", data.pop("a_residual", []))

        obj = cls(**data)
        obj.node_self_mlp = MLPLayer.deserialize(node_self_mlp)
        obj.node_sym_linear = MLPLayer.deserialize(node_sym_linear)
        obj.node_edge_linear = MLPLayer.deserialize(node_edge_linear)
        obj.edge_self_linear = MLPLayer.deserialize(edge_self_linear)

        if update_angle:
            assert isinstance(edge_angle_linear1, dict)
            assert isinstance(edge_angle_linear2, dict)
            assert isinstance(angle_self_linear, dict)
            obj.edge_angle_linear1 = MLPLayer.deserialize(edge_angle_linear1)
            obj.edge_angle_linear2 = MLPLayer.deserialize(edge_angle_linear2)
            obj.angle_self_linear = MLPLayer.deserialize(angle_self_linear)
            if a_compress_rate != 0 and not a_compress_use_split:
                assert isinstance(a_compress_n_linear, dict)
                assert isinstance(a_compress_e_linear, dict)
                obj.a_compress_n_linear = MLPLayer.deserialize(a_compress_n_linear)
                obj.a_compress_e_linear = MLPLayer.deserialize(a_compress_e_linear)

        if update_style == "res_residual":
            for ii, t in enumerate(obj.n_residual):
                t.data = to_torch_tensor(n_residual[ii])
            for ii, t in enumerate(obj.e_residual):
                t.data = to_torch_tensor(e_residual[ii])
            for ii, t in enumerate(obj.a_residual):
                t.data = to_torch_tensor(a_residual[ii])
        return obj
