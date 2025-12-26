
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
from torch_geometric.data import Data, Batch
import numpy as np
import torch
import pickle
import vesin
import functools

from deepmd.pt.model.atomic_model import (
    BaseAtomicModel,
)
from deepmd.pt.model.model.model import (
    BaseModel,
)
from deepmd.pt.model.task import (
    BaseFitting,
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

from deepmd.pt.model.network.network import (
    TypeEmbedNet,
    TypeEmbedNetConsistent,
)
from deepmd.dpmodel.output_def import (
    FittingOutputDef,
    ModelOutputDef,
    OutputVariableDef,
)
from deepmd.pt.utils import (
    AtomExcludeMask,
    PairExcludeMask,
    env,
)
from deepmd.pt.model.descriptor.repflows_dynamic import (
    DescrptBlockRepflowsDynamic,
)
from deepmd.pt.model.model.transform_output import (
    communicate_extended_output,
    fit_output_to_model_output,
)
from deepmd.pt.utils.nl import compute_neighborlist_
from deepmd.pt.utils.AtomicDataDict import batched_from_list
dtype = env.GLOBAL_PT_FLOAT_PRECISION
device = env.DEVICE

@BaseModel.register("dpa3_dynamic")
class DPA3DynamicModel(BaseModel):
    def __init__(
            self, 
            model_params,
            **kwargs,
            ) -> None:
        
        super().__init__(**kwargs)
        
        # descriptor parameters
        self.n_dim = model_params["descriptor"].get("n_dim", 128)
        self.e_dim = model_params["descriptor"].get("e_dim", 64)
        self.a_dim = model_params["descriptor"].get("a_dim", 32)
        self.nlayers = model_params["descriptor"].get("nlayers", 6)
        self.e_rcut = model_params["descriptor"].get("e_rcut", 6.0)
        self.e_rcut_smth = model_params["descriptor"].get("e_rcut_smth", 3.0)
        self.e_sel = model_params["descriptor"].get("e_sel", 1200)
        self.a_rcut = model_params["descriptor"].get("a_rcut", 4.0)
        self.a_rcut_smth = model_params["descriptor"].get("a_rcut_smth", 3.5)
        self.a_sel = model_params["descriptor"].get("a_sel", 300)
        self.axis_neuron = model_params["descriptor"].get("axis_neuron", 4)
        self.skip_stat = model_params["descriptor"].get("skip_stat", True)
        self.a_compress_rate = model_params["descriptor"].get("a_compress_rate", 1)
        self.a_compress_e_rate = model_params["descriptor"].get("a_compress_e_rate", 2)
        self.a_compress_use_split = model_params["descriptor"].get("a_compress_use_split", True)
        self.update_angle = model_params["descriptor"].get("update_angle", True)
        self.update_style = model_params["descriptor"].get("update_style", "res_residual")
        self.update_residual = model_params["descriptor"].get("update_residual", 0.1)
        self.update_residual_init = model_params["descriptor"].get("update_residual_init", "const")
        self.smooth_edge_update = model_params["descriptor"].get("smooth_edge_update", True)
        self.use_dynamic_sel = model_params["descriptor"].get("use_dynamic_sel", True)
        self.sel_reduce_factor = model_params["descriptor"].get("sel_reduce_factor", 10.0)
        self.e3nn_conv_l_max = model_params["descriptor"].get("e3nn_conv_l_max", 2)
        self.use_e3nn_conv = model_params["descriptor"].get("use_e3nn_conv", True)
        self.use_e3nn_denominator = model_params["descriptor"].get("use_e3nn_denominator", False)
        self.ntypes = len(model_params["type_map"])

        # descriptor activation and precision parameters
        self.activation_function = model_params["descriptor"].get("activation_function", "silu")
        self.use_tebd_bias = model_params["descriptor"].get("use_tebd_bias", False)
        self.precision = model_params["descriptor"].get("precision", "float32")
        self.concat_output_tebd = model_params["descriptor"].get("concat_output_tebd", False)

        # global parameters
        self.type_map = model_params.get("type_map", None)

        # fitting net parameters
        self.fitting_neuron = model_params["fitting_net"].get("neuron", [240, 240, 240])
        self.fitting_resnet_dt = model_params["fitting_net"].get("resnet_dt", True)
        self.fitting_seed = model_params["fitting_net"].get("seed", 1)
        self.fitting_precision = model_params["fitting_net"].get("precision", "float32")
        self.fitting_activation_function = model_params["fitting_net"].get("activation_function", "silu")
        self.activation_function = model_params["fitting_net"].get("activation_function", "silu")

        # self.out_bias = torch.zeros([1,118,1],dtype=dtype,device=device)
        # self.out_std = torch.ones([1,118,1],dtype=dtype,device=device)

        fitting_net_config = {
            "neuron": self.fitting_neuron,
            "resnet_dt":True,
            "seed": self.fitting_seed,
            "precision": self.fitting_precision,
            "activation_function": self.fitting_activation_function,
            "type": "ener",
            "ntypes": self.ntypes,
            "dim_descrpt": self.n_dim,
        }
        
        self.fitting_net = BaseFitting(
            **fitting_net_config
        )

        self.type_embedding = TypeEmbedNet(
                self.ntypes,
                self.n_dim,
                precision=self.precision,
                seed=self.fitting_seed,
                use_econf_tebd=False,
                use_tebd_bias=False,
                type_map=self.type_map,
            )

        self.repflows = DescrptBlockRepflowsDynamic(
            n_dim=self.n_dim,
            e_dim=self.e_dim,
            a_dim=self.a_dim,
            nlayers=self.nlayers,
            e_rcut=self.e_rcut,
            e_rcut_smth=self.e_rcut_smth,
            e_sel=self.e_sel,
            a_rcut=self.a_rcut,
            a_rcut_smth=self.a_rcut_smth,
            a_sel=self.a_sel,
            ntypes=self.ntypes,
            a_compress_rate=self.a_compress_rate,
            a_compress_e_rate=self.a_compress_e_rate,
            a_compress_use_split=self.a_compress_use_split,
            axis_neuron=self.axis_neuron,
            update_angle=self.update_angle,
            update_style=self.update_style,
            update_residual=self.update_residual,
            update_residual_init=self.update_residual_init,
            skip_stat=self.skip_stat,
            smooth_edge_update=self.smooth_edge_update,
            activation_function=self.activation_function,
            use_dynamic_sel=self.use_dynamic_sel,
            sel_reduce_factor=self.sel_reduce_factor,
            use_e3nn_conv=self.use_e3nn_conv,
            e3nn_conv_l_max=self.e3nn_conv_l_max,
            use_e3nn_denominator=self.use_e3nn_denominator,
            seed=self.fitting_seed,
        )

        self.init_out_stat()
        

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

    def apply_out_stat(
        self,
        ret: dict[str, torch.Tensor],
        atype: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Apply the stat to each atomic output.
        The developer may override the method to define how the bias is applied
        to the atomic output of the model.

        Parameters
        ----------
        ret
            The returned dict by the forward_atomic method
        atype
            The atom types. nf x nloc

        """
        out_bias, out_std = self._fetch_out_stat(self.bias_keys)
        for kk in self.bias_keys:
            # nf x nloc x odims, out_bias: ntypes x odims
            ret[kk] = ret[kk] + out_bias[kk][atype]
        return ret

    def make_atom_mask(
        self,
        atype: torch.Tensor,
    ) -> torch.Tensor:
        """The atoms with type < 0 are treated as virtual atoms,
        which serves as place-holders for multi-frame calculations
        with different number of atoms in different frames.

        Parameters
        ----------
        atype
            Atom types. >= 0 for real atoms <0 for virtual atoms.

        Returns
        -------
        mask
            True for real atoms and False for virtual atoms.

        """
        # supposed to be supported by all backends
        return atype >= 0

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

    def atomic_output_def(self) -> FittingOutputDef:
        old_def = self.fitting_output_def()
        old_list = list(old_def.get_data().values())
        return FittingOutputDef(
            old_list  # noqa:RUF005
            + [
                OutputVariableDef(
                    name="mask",
                    shape=[1],
                    reducible=False,
                    r_differentiable=False,
                    c_differentiable=False,
                )
            ]
        )

    def _store_out_stat(
        self,
        out_bias: dict[str, torch.Tensor],
        out_std: dict[str, torch.Tensor],
        add: bool = False,
    ) -> None:
        ntypes = self.get_ntypes()
        out_bias_data = torch.clone(self.out_bias)
        out_std_data = torch.clone(self.out_std)
        for kk in out_bias.keys():
            assert kk in out_std.keys()
            idx = self._get_bias_index(kk)
            size = self._varsize(self.atomic_output_def()[kk].shape)
            if not add:
                out_bias_data[idx, :, :size] = out_bias[kk].view(ntypes, size)
            else:
                out_bias_data[idx, :, :size] += out_bias[kk].view(ntypes, size)
            out_std_data[idx, :, :size] = out_std[kk].view(ntypes, size)
        self.out_bias.copy_(out_bias_data)
        self.out_std.copy_(out_std_data)

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
        return 0

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
        batch: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_cell_shift: Optional[torch.Tensor] = None,
        angle_index: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        do_atomic_virial: bool = False,
    ) -> dict[str, torch.Tensor]:
        device = coord.device
        debug_test = False
        if debug_test:
            mptraj_data1 = pickle.load(open("/aisi/mnt/data_nas/liwentao/devel_workspace/deepmd-kit-arch/test_mptraj/mptraj_test_data.pkl","rb"))
            import pdb; pdb.set_trace()
            data1 = compute_neighborlist_(mptraj_data1,self.e_rcut,use_angle=True)
            mptraj_data2 = pickle.load(open("/aisi/mnt/data_nas/liwentao/devel_workspace/deepmd-kit-arch/test_mptraj/mptraj_test_data2.pkl","rb"))
            data2 = compute_neighborlist_(mptraj_data2,self.e_rcut,use_angle=True)
            
            batch_data = batched_from_list([data1,data2])
            coord = batch_data['pos'].to(device)
            coord.requires_grad_(True)
            atype = batch_data['atomic_numbers'].squeeze(-1).to(device)
            edge_index = batch_data['edge_index'].to(device)
            edge_cell_shift = batch_data['edge_cell_shift'].to(device)
            angle_index = batch_data['angle_index'].to(device)
            batch = batch_data['batch'].to(device)
            box = box.to(device)

            energy_gt = batch_data['total_energy'].to(device)
            forces_gt = batch_data['forces'].to(device)
            stress_gt = batch_data['stress'].to(device)
        else:
            nf = coord.shape[0]
            batch_data_list = []
            for ii in range(nf):
                if torch.max(torch.abs(box[ii])).max() > 1e-6:
                    pbc = torch.tensor([[True,True,True]],device=device)
                mptraj_data = {
                    'pos': coord[ii],
                    'atomic_numbers': atype[ii],
                    'cell': box[ii].view(3,3).unsqueeze(0),
                    'pbc': pbc,
                }
                data = compute_neighborlist_(mptraj_data,self.e_rcut,use_angle=True)
                batch_data_list.append(data)
            batch_data = batched_from_list(batch_data_list)
            coord = batch_data['pos'].to(device)
            coord.requires_grad_(True)
            atype = batch_data['atomic_numbers'].squeeze(-1).to(device)
            edge_index = batch_data['edge_index'].to(device)
            edge_cell_shift = batch_data['edge_cell_shift'].to(device)
            angle_index = batch_data['angle_index'].to(device)
            batch = batch_data['batch'].to(device)
            box = box.to(device)
        
        if self.precision == "float32":
            coord = coord.to(dtype=torch.float32)
            edge_cell_shift = edge_cell_shift.to(dtype=torch.float32)
            box = box.to(dtype=torch.float32)
        elif self.precision == "float64":
            coord = coord.to(dtype=torch.float64)
            edge_cell_shift = edge_cell_shift.to(dtype=torch.float64)
            box = box.to(dtype=torch.float64)

        atype_embedding = self.type_embedding(atype)
        # calculate edge vectors

        src,dst = edge_index[0],edge_index[1]
        r_i = coord[src]
        r_j = coord[dst]
        edge_batch = batch[src]                                # [E]
        box_edge = box[edge_batch].reshape(-1, 3, 3)                                  # [E,3,3]
        s = edge_cell_shift.to(coord.device).unsqueeze(1)                 # [E,1,3]
        # (s @ H): [E,1,3] bmm [E,3,3] -> [E,1,3] -> [E,3]
        
        shift_real = torch.bmm(s, box_edge.to(coord.device)).squeeze(1)   # [E,3]

        edge_vectors = (r_j - r_i) + shift_real                         # [E,3]
        distances = torch.linalg.norm(edge_vectors, dim=-1)             # [E]

        descriptor, rot_mat, g2, h2, sw = self.repflows(
            coord,
            atype,
            atype_embedding,
            edge_index,
            edge_vectors,
            distances,
            angle_index,
            batch,
        )
        
        ret_dict = self.fitting_net(
            descriptor,
            atype,
            gr=rot_mat,
            g2=g2,
            h2=h2,
            fparam=fparam,
            aparam=aparam,
        )
        ret_dict = self.apply_out_stat(ret_dict, atype)

        # nf x nloc
        ext_atom_mask = self.make_atom_mask(atype).unsqueeze(0)
        atom_mask = ext_atom_mask.to(torch.int32)
        for kk in ret_dict.keys():
            out_shape = ret_dict[kk].shape
            out_shape2 = 1
            for ss in out_shape[2:]:
                out_shape2 *= ss
            ret_dict[kk] = (
                ret_dict[kk].reshape([out_shape[0], out_shape[1], out_shape2])
                * atom_mask[:, :, None]
            ).view(out_shape)
        ret_dict["mask"] = atom_mask
        ret_dict['batch'] = batch
        model_ret = fit_output_to_model_output(
            ret_dict,
            self.atomic_output_def(),
            coord,
            do_atomic_virial=do_atomic_virial,
            create_graph=self.training,
            mask=ret_dict["mask"] if "mask" in ret_dict else None,
        )
        model_predict: dict[str, torch.Tensor] = {}
        model_predict["atom_energy"] = model_ret["energy"]
        model_predict["energy"] = model_ret["energy_redu"]
        
        model_predict["force"] = model_ret["energy_derv_r"].squeeze(-2)
        
        model_predict["virial"] = model_ret["energy_derv_c_redu"].squeeze(-2)
        
        return model_predict
        # return model_predict,(energy_gt,forces_gt,stress_gt)

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
    
    def _default_bias(self) -> torch.Tensor:
        ntypes = self.get_ntypes()
        return torch.zeros(
            [self.n_out, ntypes, self.max_out_size], dtype=dtype, device=device
        )

    def _default_std(self) -> torch.Tensor:
        ntypes = self.get_ntypes()
        return torch.ones(
            [self.n_out, ntypes, self.max_out_size], dtype=dtype, device=device
        )
    
    def init_out_stat(self) -> None:
        """Initialize the output bias."""
        ntypes = self.get_ntypes()
        self.bias_keys: list[str] = list(self.fitting_output_def().keys())
        self.max_out_size = max(
            [self.atomic_output_def()[kk].size for kk in self.bias_keys]
        )
        self.n_out = len(self.bias_keys)
        out_bias_data = self._default_bias()
        out_std_data = self._default_std()
        self.register_buffer("out_bias", out_bias_data)
        self.register_buffer("out_std", out_std_data)

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

    def _store_out_stat(
        self,
        out_bias: dict[str, torch.Tensor],
        out_std: dict[str, torch.Tensor],
        add: bool = False,
    ) -> None:
        ntypes = self.get_ntypes()
        out_bias_data = torch.clone(self.out_bias)
        out_std_data = torch.clone(self.out_std)
        for kk in out_bias.keys():
            assert kk in out_std.keys()
            idx = self._get_bias_index(kk)
            size = self._varsize(self.atomic_output_def()[kk].shape)
            if not add:
                out_bias_data[idx, :, :size] = out_bias[kk].view(ntypes, size)
            else:
                out_bias_data[idx, :, :size] += out_bias[kk].view(ntypes, size)
            out_std_data[idx, :, :size] = out_std[kk].view(ntypes, size)
        self.out_bias.copy_(out_bias_data)
        self.out_std.copy_(out_std_data)

    def _fetch_out_stat(
        self,
        keys: list[str],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        ret_bias = {}
        ret_std = {}
        ntypes = self.get_ntypes()
        for kk in keys:
            idx = self._get_bias_index(kk)
            isize = self._varsize(self.atomic_output_def()[kk].shape)
            ret_bias[kk] = self.out_bias[idx, :, :isize].view(
                [ntypes] + list(self.atomic_output_def()[kk].shape)  # noqa: RUF005
            )
            ret_std[kk] = self.out_std[idx, :, :isize].view(
                [ntypes] + list(self.atomic_output_def()[kk].shape)  # noqa: RUF005
            )
        return ret_bias, ret_std
