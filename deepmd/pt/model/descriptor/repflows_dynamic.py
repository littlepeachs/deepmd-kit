# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Callable,
    Optional,
    Union,
)

import torch

from e3nn.o3 import Irreps, Linear
from e3nn.o3 import FullyConnectedTensorProduct as FCTP

import time
from deepmd.pt.utils import env
from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.pt.model.descriptor.descriptor import (
    DescriptorBlock,
)
from deepmd.pt.model.descriptor.env_mat import (
    prod_env_mat,
    prod_env_from_edges,
)
from deepmd.pt.model.network.mlp import (
    MLPLayer,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    PRECISION_DICT,
)
from deepmd.pt.utils.env_mat_stat import (
    EnvMatStatSe,
)
from deepmd.pt.utils.exclude_mask import (
    PairExcludeMask,
)
from deepmd.pt.utils.spin import (
    concat_switch_virtual,
)
from deepmd.pt.utils.utils import (
    ActivationFn,
)
from deepmd.utils.env_mat_stat import (
    StatItem,
)
from deepmd.utils.path import (
    DPPath,
)

from .repflows_layer_dynamic import (
    RepFlowLayerDynamic,
)

if not hasattr(torch.ops.deepmd, "border_op"):

    def border_op(
        argument0,
        argument1,
        argument2,
        argument3,
        argument4,
        argument5,
        argument6,
        argument7,
        argument8,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "border_op is not available since customized PyTorch OP library is not built when freezing the model. "
            "See documentation for DPA-3 for details."
        )

    # Note: this hack cannot actually save a model that can be run using LAMMPS.
    torch.ops.deepmd.border_op = border_op


@DescriptorBlock.register("se_repflow_dynamic")
class DescrptBlockRepflowsDynamic(DescriptorBlock):
    def __init__(
        self,
        e_rcut,
        e_rcut_smth,
        e_sel: int,
        a_rcut,
        a_rcut_smth,
        a_sel: int,
        ntypes: int,
        nlayers: int = 6,
        n_dim: int = 128,
        e_dim: int = 64,
        a_dim: int = 32,
        a_compress_rate: int = 1,
        a_compress_e_rate: int = 2,
        a_compress_use_split: bool = True,
        axis_neuron: int = 4,
        update_angle: bool = True,
        update_style: str = "res_residual",
        update_residual: float = 0.1,
        update_residual_init: str = "const",
        activation_function: str = "silu",
        skip_stat: bool = True,
        smooth_edge_update: bool = True,
        use_dynamic_sel: bool = True,
        sel_reduce_factor: float = 10.0,
        use_e3nn_conv: bool = True,
        e3nn_conv_l_max: int = 2,
        use_e3nn_denominator: bool = False,
        seed: int = 42,
    ) -> None:
        r"""
        The repflow descriptor block.

        Parameters
        ----------
        n_dim : int, optional
            The dimension of node representation.
        e_dim : int, optional
            The dimension of edge representation.
        a_dim : int, optional
            The dimension of angle representation.
        nlayers : int, optional
            Number of repflow layers.
        e_rcut : float, optional
            The edge cut-off radius.
        e_rcut_smth : float, optional
            Where to start smoothing for edge. For example the 1/r term is smoothed from rcut to rcut_smth.
        e_sel : int, optional
            Maximally possible number of selected edge neighbors.
        a_rcut : float, optional
            The angle cut-off radius.
        a_rcut_smth : float, optional
            Where to start smoothing for angle. For example the 1/r term is smoothed from rcut to rcut_smth.
        a_sel : int, optional
            Maximally possible number of selected angle neighbors.
        a_compress_rate : int, optional
            The compression rate for angular messages. The default value is 0, indicating no compression.
            If a non-zero integer c is provided, the node and edge dimensions will be compressed
            to a_dim/c and a_dim/2c, respectively, within the angular message.
        a_compress_e_rate : int, optional
            The extra compression rate for edge in angular message compression. The default value is 1.
            When using angular message compression with a_compress_rate c and a_compress_e_rate c_e,
            the edge dimension will be compressed to (c_e * a_dim / 2c) within the angular message.
        a_compress_use_split : bool, optional
            Whether to split first sub-vectors instead of linear mapping during angular message compression.
            The default value is False.
        n_multi_edge_message : int, optional
            The head number of multiple edge messages to update node feature.
            Default is 1, indicating one head edge message.
        axis_neuron : int, optional
            The number of dimension of submatrix in the symmetrization ops.
        update_angle : bool, optional
            Where to update the angle rep. If not, only node and edge rep will be used.
        update_style : str, optional
            Style to update a representation.
            Supported options are:
            -'res_avg': Updates a rep `u` with: u = 1/\\sqrt{n+1} (u + u_1 + u_2 + ... + u_n)
            -'res_incr': Updates a rep `u` with: u = u + 1/\\sqrt{n} (u_1 + u_2 + ... + u_n)
            -'res_residual': Updates a rep `u` with: u = u + (r1*u_1 + r2*u_2 + ... + r3*u_n)
            where `r1`, `r2` ... `r3` are residual weights defined by `update_residual`
            and `update_residual_init`.
        update_residual : float, optional
            When update using residual mode, the initial std of residual vector weights.
        update_residual_init : str, optional
            When update using residual mode, the initialization mode of residual vector weights.
        ntypes : int
            Number of element types
        activation_function : str, optional
            The activation function in the embedding net.
        set_davg_zero : bool, optional
            Set the normalization average to zero.
        precision : str, optional
            The precision of the embedding net parameters.
        exclude_types : list[list[int]], optional
            The excluded pairs of types which have no interaction with each other.
            For example, `[[0, 1]]` means no interaction between type 0 and type 1.
        env_protection : float, optional
            Protection parameter to prevent division by zero errors during environment matrix calculations.
            For example, when using paddings, there may be zero distances of neighbors, which may make division by zero error during environment matrix calculations without protection.
        seed : int, optional
            Random seed for parameter initialization.
        """
        super().__init__()
        self.e_rcut = float(e_rcut)
        self.e_rcut_smth = float(e_rcut_smth)
        self.e_sel = e_sel
        self.a_rcut = float(a_rcut)
        self.a_rcut_smth = float(a_rcut_smth)
        self.a_sel = a_sel
        self.ntypes = ntypes
        self.nlayers = nlayers
        self.n_dim = n_dim
        self.e_dim = e_dim
        self.a_dim = a_dim
        # for other common desciptor method
        sel = [e_sel] if isinstance(e_sel, int) else e_sel
        self.nnei = sum(sel)
        self.ndescrpt = self.nnei * 4  # use full descriptor.
        assert len(sel) == 1
        self.sel = sel
        self.rcut = e_rcut
        self.rcut_smth = e_rcut_smth
        self.sec = self.sel
        self.split_sel = self.sel
        self.a_compress_rate = a_compress_rate
        self.a_compress_e_rate = a_compress_e_rate
        self.axis_neuron = axis_neuron
        self.update_angle = update_angle
        self.update_style = update_style
        self.update_residual = update_residual
        self.update_residual_init = update_residual_init
        self.activation_function = activation_function
        self.skip_stat = skip_stat
        self.a_compress_use_split = a_compress_use_split
        self.optim_update = True
        self.smooth_edge_update = smooth_edge_update
        self.use_dynamic_sel = use_dynamic_sel
        self.sel_reduce_factor = sel_reduce_factor
        self.dynamic_e_sel = self.nnei / self.sel_reduce_factor
        self.dynamic_a_sel = self.a_sel / self.sel_reduce_factor

        layers = []

        self.edge_embd = MLPLayer(
            1,
            self.e_dim,
            precision="float32",
            seed=child_seed(seed, 0),
            bias=False,
        )

        self.angle_embd = MLPLayer(
            1,
            self.a_dim,
            precision="float32",
            bias=False,
            seed=child_seed(seed, 1),
        )

        self.act = ActivationFn(activation_function)

        for ii in range(nlayers):
            # for node edge e3nn conv
            layers.append(
                RepFlowLayerDynamic(
                    e_rcut=self.e_rcut,
                    e_rcut_smth=self.e_rcut_smth,
                    e_sel=self.sel,
                    a_rcut=self.a_rcut,
                    a_rcut_smth=self.a_rcut_smth,
                    a_sel=self.a_sel,
                    ntypes=self.ntypes,
                    n_dim=self.n_dim,
                    e_dim=self.e_dim,
                    a_dim=self.a_dim,
                    a_compress_rate=self.a_compress_rate,
                    a_compress_use_split=self.a_compress_use_split,
                    a_compress_e_rate=self.a_compress_e_rate,
                    axis_neuron=self.axis_neuron,
                    update_angle=self.update_angle,
                    optim_update=self.optim_update,
                    use_dynamic_sel=self.use_dynamic_sel,
                    sel_reduce_factor=self.sel_reduce_factor,
                    smooth_edge_update=self.smooth_edge_update,
                    activation_function=self.activation_function,
                    update_style=self.update_style,
                    update_residual=self.update_residual,
                    update_residual_init=self.update_residual_init,
                    precision="float32",
                    seed=child_seed(child_seed(seed, 1), ii),
                )
            )
        self.layers = torch.nn.ModuleList(layers)
        self.additional_output_for_fitting: dict[str, Optional[torch.Tensor]] = {}

        wanted_shape = (self.ntypes, 4)
        self.prec = PRECISION_DICT["float32"]
        mean = torch.zeros(wanted_shape, dtype=self.prec, device=env.DEVICE)
        stddev = torch.ones(wanted_shape, dtype=self.prec, device=env.DEVICE)
        if self.skip_stat:
            stddev = stddev * 0.3
        self.register_buffer("mean", mean)
        self.register_buffer("stddev", stddev)
        self.stats = None

    def get_rcut(self) -> float:
        """Returns the cut-off radius."""
        return self.e_rcut

    additional_output_for_fitting: dict[str, Optional[torch.Tensor]]

    def get_rcut_smth(self) -> float:
        """Returns the radius where the neighbor information starts to smoothly decay to 0."""
        return self.e_rcut_smth

    def get_norm_fact(self) -> list[float]:
        """Returns the norm factor."""
        return [
            float(self.dynamic_e_sel if self.use_dynamic_sel else self.nnei),
            float(self.dynamic_a_sel if self.use_dynamic_sel else self.a_sel),
        ]

    def get_nsel(self) -> int:
        """Returns the number of selected atoms in the cut-off radius."""
        return sum(self.sel)

    def get_sel(self) -> list[int]:
        """Returns the number of selected atoms for each type."""
        return self.sel

    def get_ntypes(self) -> int:
        """Returns the number of element types."""
        return self.ntypes

    def get_dim_out(self) -> int:
        """Returns the output dimension."""
        return self.dim_out

    def get_dim_in(self) -> int:
        """Returns the input dimension."""
        return self.dim_in

    def get_dim_emb(self) -> int:
        """Returns the embedding dimension e_dim."""
        return self.e_dim

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

    def mixed_types(self) -> bool:
        """If true, the descriptor
        1. assumes total number of atoms aligned across frames;
        2. requires a neighbor list that does not distinguish different atomic types.

        If false, the descriptor
        1. assumes total number of atoms of each atom type aligned across frames;
        2. requires a neighbor list that distinguishes different atomic types.

        """
        return True

    def get_env_protection(self) -> float:
        """Returns the protection of building environment matrix."""
        return self.env_protection

    @property
    def dim_out(self):
        """Returns the output dimension of this descriptor."""
        out_dim = self.n_dim
        
        return out_dim

    @property
    def dim_in(self):
        """Returns the atomic input dimension of this descriptor."""
        return self.n_dim

    @property
    def dim_emb(self):
        """Returns the embedding dimension e_dim."""
        return self.get_dim_emb()

    def reinit_exclude(
        self,
        exclude_types: list[tuple[int, int]] = [],
    ) -> None:
        self.exclude_types = exclude_types
        self.emask = PairExcludeMask(self.ntypes, exclude_types=exclude_types)

    def get_additional_output_for_fitting(self):
        return self.additional_output_for_fitting

    def forward(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        atype_embedding: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vectors: torch.Tensor,
        distances: torch.Tensor,
        angle_index: torch.Tensor,
        batch: torch.Tensor,
    ):
        nall = len(batch)

        
        dmatrix, sw = prod_env_from_edges(
            edge_index,
            edge_vectors,
            distances,
            atype,
            self.mean,
            self.stddev,
            self.e_rcut,
            self.e_rcut_smth,
            protection=0.0,
            use_env_envelope=False,
            use_new_sw=False,
        )
        
        angle_edge_index_i = edge_index[:,angle_index[1,:]]
        angle_vector_i = edge_vectors[angle_index[1,:]]
        angle_distance_i = distances[angle_index[1,:]]

        angle_edge_index_j = edge_index[:,angle_index[2,:]]
        angle_vector_j = edge_vectors[angle_index[2,:]]
        angle_distance_j = distances[angle_index[2,:]]

        _, angle_sw_1 = prod_env_from_edges(
            angle_edge_index_i,
            angle_vector_i,
            angle_distance_i,
            atype,
            self.mean,
            self.stddev,
            self.a_rcut,
            self.a_rcut_smth,
            protection=0.0,
            use_env_envelope=False,
            use_new_sw=False,
        )

        _, angle_sw_2 = prod_env_from_edges(
            angle_edge_index_j,
            angle_vector_j,
            angle_distance_j,
            atype,
            self.mean,
            self.stddev,
            self.a_rcut,
            self.a_rcut_smth,
            protection=0.0,
            use_env_envelope=False,
            use_new_sw=False,
        ) 
        # get edge and angle embedding input
        # nb x nloc x nnei x 1,  nb x nloc x nnei x 3
        edge_input, h2 = torch.split(dmatrix, [1, 3], dim=-1)

        a_sw = angle_sw_1 * angle_sw_2
        
        
        normalized_diff_i = angle_vector_i / (
            torch.linalg.norm(angle_vector_i, dim=-1, keepdim=True) + 1e-6
        )
        normalized_diff_j = angle_vector_j / (
            torch.linalg.norm(angle_vector_j, dim=-1, keepdim=True) + 1e-6
        )
        # 1 - 1e-6 for torch.acos stability
        cosine_ij = torch.sum(normalized_diff_i * normalized_diff_j, dim=-1) * (1 - 1e-6)
        sine_ij = torch.sqrt(1 - cosine_ij**2)
        
        theta_list = None

        # nf x nloc x a_nnei x a_nnei x 1,  nf x nloc x a_nnei x a_nnei x n_freq
        angle_input_list = [cosine_ij.unsqueeze(-1)] + (
            [torch.cos(theta_list)] if theta_list is not None else []
        )
        angle_input = torch.cat(angle_input_list, dim=-1) / (torch.pi**0.5)

        
        edge_ebd = self.act(self.edge_embd(edge_input))
        # nf x nloc x a_nnei x a_nnei x a_dim [OR] n_angle x a_dim
        angle_ebd = self.angle_embd(angle_input)

        for idx, ll in enumerate(self.layers):
            # node_ebd:     nb x nloc x n_dim
            # node_ebd_ext: nb x nall x n_dim [OR] nb x nloc x n_dim when not parrallel_mode
            
            node_ebd, edge_ebd, angle_ebd= ll.forward(
                atype_embedding,
                edge_ebd,
                h2,
                angle_ebd,
                sw,
                a_sw,
                edge_index=edge_index,
                angle_index=angle_index,
                batch=batch,
            )


        # nb x nloc x 3 x e_dim
        h2g2 = (
            RepFlowLayerDynamic._cal_hg_dynamic(
                edge_ebd,
                h2,
                sw,
                owner=edge_index[0, :],
                num_owner=nall,
                scale_factor=(self.nnei / self.sel_reduce_factor) ** (-0.5),
            )
        )
        
        # (nb x nloc) x e_dim x 3
        rot_mat = torch.permute(h2g2, (0, 2, 1))

        self.additional_output_for_fitting["angle_embd"] = angle_ebd

        return node_ebd, edge_ebd, h2, rot_mat.view(nall, self.dim_emb, 3), sw

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

    def get_stats(self) -> dict[str, StatItem]:
        """Get the statistics of the descriptor."""
        if self.stats is None:
            raise RuntimeError(
                "The statistics of the descriptor has not been computed."
            )
        return self.stats

    def has_message_passing(self) -> bool:
        """Returns whether the descriptor block has message passing."""
        return True

    def need_sorted_nlist_for_lower(self) -> bool:
        """Returns whether the descriptor block needs sorted nlist when using `forward_lower`."""
        return True
