# SPDX-License-Identifier: LGPL-3.0-or-later
import os
from typing import (
    Any,
    Optional,
)
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

from deepmd.pt.loss.loss import (
    TaskLoss,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    GLOBAL_PT_FLOAT_PRECISION,
)
from deepmd.utils.data import (
    DataRequirementItem,
)
from deepmd.utils.version import (
    check_version_compatibility,
)
from deepmd.pt.model.model.transform_output import (
    fit_output_to_model_output,
)

from deepmd.pt.utils.nlist import (
    extend_input_and_build_neighbor_list,
    nlist_distinguish_types,
)

from typing import Tuple, List, Union

def custom_huber_loss(
    predictions: torch.Tensor, targets: torch.Tensor, delta: float = 1.0
) -> torch.Tensor:
    error = targets - predictions
    abs_error = torch.abs(error)
    quadratic_loss = 0.5 * torch.pow(error, 2)
    linear_loss = delta * (abs_error - 0.5 * delta)
    loss = torch.where(abs_error <= delta, quadratic_loss, linear_loss)
    return torch.mean(loss)

def communicate_extended_output_simple(
    model_ret: dict[str, torch.Tensor],
    mapping: torch.Tensor,  # shape: (n_frames, n_all_atoms)
    do_atomic_virial: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Transform the output of the model network defined on local and ghost (extended)
    atoms to local atoms without using ModelOutputDef.
    """
    new_ret = {}
    
    # 获取本地原子数 (根据mapping的最大索引推断，mapping索引范围是 0 到 nloc-1)
    # 注意：这里假设mapping中包含所有本地原子的索引
    n_loc = int(mapping.max().item()) + 1

    for key, tensor in model_ret.items():
        # 1. 基础量直接拷贝 (如 energy, mask, energy_redu)
        # 忽略掉已经存在的 global virial redu (稍后会重新计算)
        if not key.endswith(('_derv_r', '_derv_c', '_derv_c_redu')):
            new_ret[key] = tensor
            continue

        # 2. 处理力/坐标导数 (Force: _derv_r) -> shape usually (B, N, 3)
        if key.endswith('_derv_r'):
            # 准备输出张量 (B, n_loc, 3)
            out_shape = list(tensor.shape)
            out_shape[1] = n_loc 
            
            # 扩展 mapping 维度以匹配 tensor: (B, N_all) -> (B, N_all, 3)
            # 相当于原代码的 view + expand/tile
            map_expanded = mapping.unsqueeze(-1).expand_as(tensor)
            
            out_tensor = torch.zeros(out_shape, dtype=tensor.dtype, device=tensor.device)
            
            # 执行 scatter_reduce (求和: 将 ghost 原子的力累加到 local 原子)
            new_ret[key] = torch.scatter_reduce(
                out_tensor,
                dim=1,
                index=map_expanded,
                src=tensor,
                reduce="sum"
            )

        # 3. 处理维里/晶胞导数 (Virial: _derv_c) -> shape usually (B, N, 9)
        elif key.endswith('_derv_c') and not key.endswith('_derv_c_redu'):
            out_shape = list(tensor.shape)
            out_shape[1] = n_loc
            
            # 扩展 mapping: (B, N_all) -> (B, N_all, 9)
            map_expanded = mapping.unsqueeze(-1).expand_as(tensor)
            
            out_tensor = torch.zeros(out_shape, dtype=tensor.dtype, device=tensor.device)
            
            # 累加 atomic virial
            atomic_virial = torch.scatter_reduce(
                out_tensor,
                dim=1,
                index=map_expanded,
                src=tensor,
                reduce="sum"
            )
            
            if do_atomic_virial:
                new_ret[key] = atomic_virial
            
            # 重新计算 Global Virial (sum over atoms)
            # 原代码逻辑：new_ret[kk_derv_c + "_redu"] = torch.sum(...)
            new_ret[key + '_redu'] = torch.sum(atomic_virial, dim=1)

    return new_ret

class EnergyStdLoss(TaskLoss):
    def __init__(
        self,
        starter_learning_rate: float = 1.0,
        start_pref_e: float = 0.0,
        limit_pref_e: float = 0.0,
        start_pref_f: float = 0.0,
        limit_pref_f: float = 0.0,
        start_pref_v: float = 0.0,
        limit_pref_v: float = 0.0,
        start_pref_ae: float = 0.0,
        limit_pref_ae: float = 0.0,
        start_pref_pf: float = 0.0,
        limit_pref_pf: float = 0.0,
        relative_f: Optional[float] = None,
        enable_atom_ener_coeff: bool = False,
        start_pref_gf: float = 0.0,
        limit_pref_gf: float = 0.0,
        numb_generalized_coord: int = 0,
        use_l1_all: bool = False,
        inference: bool = False,
        use_huber: bool = False,
        huber_delta: float = 0.01,
        **kwargs: Any,
    ) -> None:
        r"""Construct a layer to compute loss on energy, force and virial.

        Parameters
        ----------
        starter_learning_rate : float
            The learning rate at the start of the training.
        start_pref_e : float
            The prefactor of energy loss at the start of the training.
        limit_pref_e : float
            The prefactor of energy loss at the end of the training.
        start_pref_f : float
            The prefactor of force loss at the start of the training.
        limit_pref_f : float
            The prefactor of force loss at the end of the training.
        start_pref_v : float
            The prefactor of virial loss at the start of the training.
        limit_pref_v : float
            The prefactor of virial loss at the end of the training.
        start_pref_ae : float
            The prefactor of atomic energy loss at the start of the training.
        limit_pref_ae : float
            The prefactor of atomic energy loss at the end of the training.
        start_pref_pf : float
            The prefactor of atomic prefactor force loss at the start of the training.
        limit_pref_pf : float
            The prefactor of atomic prefactor force loss at the end of the training.
        relative_f : float
            If provided, relative force error will be used in the loss. The difference
            of force will be normalized by the magnitude of the force in the label with
            a shift given by relative_f
        enable_atom_ener_coeff : bool
            if true, the energy will be computed as \sum_i c_i E_i
        start_pref_gf : float
            The prefactor of generalized force loss at the start of the training.
        limit_pref_gf : float
            The prefactor of generalized force loss at the end of the training.
        numb_generalized_coord : int
            The dimension of generalized coordinates.
        use_l1_all : bool
            Whether to use L1 loss, if False (default), it will use L2 loss.
        inference : bool
            If true, it will output all losses found in output, ignoring the pre-factors.
        use_huber : bool
            Enables Huber loss calculation for energy/force/virial terms with user-defined threshold delta (D).
            The loss function smoothly transitions between L2 and L1 loss:
            - For absolute prediction errors within D: quadratic loss (0.5 * (error**2))
            - For absolute errors exceeding D: linear loss (D * |error| - 0.5 * D)
            Formula: loss = 0.5 * (error**2) if |error| <= D else D * (|error| - 0.5 * D).
        huber_delta : float
            The threshold delta (D) used for Huber loss, controlling transition between L2 and L1 loss.
        **kwargs
            Other keyword arguments.
        """
        super().__init__()
        self.starter_learning_rate = starter_learning_rate
        self.has_e = (start_pref_e != 0.0 and limit_pref_e != 0.0) or inference
        self.has_f = (start_pref_f != 0.0 and limit_pref_f != 0.0) or inference
        self.has_v = (start_pref_v != 0.0 and limit_pref_v != 0.0) or inference
        self.has_ae = (start_pref_ae != 0.0 and limit_pref_ae != 0.0) or inference
        self.has_pf = (start_pref_pf != 0.0 and limit_pref_pf != 0.0) or inference
        self.has_gf = start_pref_gf != 0.0 and limit_pref_gf != 0.0

        self.start_pref_e = start_pref_e
        self.limit_pref_e = limit_pref_e
        self.start_pref_f = start_pref_f
        self.limit_pref_f = limit_pref_f
        self.start_pref_v = start_pref_v
        self.limit_pref_v = limit_pref_v
        self.start_pref_ae = start_pref_ae
        self.limit_pref_ae = limit_pref_ae
        self.start_pref_pf = start_pref_pf
        self.limit_pref_pf = limit_pref_pf
        self.start_pref_gf = start_pref_gf
        self.limit_pref_gf = limit_pref_gf
        self.relative_f = relative_f
        self.enable_atom_ener_coeff = enable_atom_ener_coeff
        self.numb_generalized_coord = numb_generalized_coord
        if self.has_gf and self.numb_generalized_coord < 1:
            raise RuntimeError(
                "When generalized force loss is used, the dimension of generalized coordinates should be larger than 0"
            )
        self.use_l1_all = use_l1_all
        self.inference = inference
        self.use_huber = use_huber
        self.huber_delta = huber_delta
        if self.use_huber and (
            self.has_pf or self.has_gf or self.relative_f is not None
        ):
            raise RuntimeError(
                "Huber loss is not implemented for force with atom_pref, generalized force and relative force. "
            )

        self.pipe_buffers = {
            'inputs': [],  # [feas, graph_bg_ag]
            'outputs': [],  # [feas, graph_bg_ag] or preadiction : dict[str, Tensor]
            'forces':[], # in first stage， computing the force，
            'stress':[], # in first stage， computing the force，
            'energy':[], # in first stage, computing the energy
            'dE_dinputs': [None,None,None,None], # tuple[dE_dfeas], FW1 output   FW1 send
            'dL_doutputs': [], # BW0 recv
            'dL_dinputs': [], # BW0 send
            'dE_doutputs': [None,None,None,None], # FW1 recv 
            'dL_dE_dinputs': [], # BW1 recv
            'dL_dE_doutputs': [], # BW1 send
            'loss_f': [],  # loss from batch input
            'loss_e': [],  # loss from batch input
            'loss_s': [],  # loss from batch input
            'loss_m': [],  # loss from batch input
            'data':[],
            'batched_graph': [], 
            'targets': [],
        }

        self.global_atom_num = 0
        self.stage_id = 0
        self.num_stages = 4
        self.pref_e = 0

    def filter_tensors_with_grad(self, tensors: Tuple[torch.Tensor, ...]) -> Tuple[Tuple[torch.Tensor, ...], List[bool]]:
        """
            过滤不支持计算梯度的中间激活值和为None的Tensor
        Args:
            Tuple[Tensor], 输入中间的激活值
        Returns:
            Tuple[Tensor], 输出中间激活值可以计算梯度的部分，

        """
        
        mask = [(t is not None) and (t.requires_grad) for t in tensors]
        filtered = tuple(t for t, m in zip(tensors, mask) if m)
        return filtered, mask

    def filter_tensor_with_mask(self, grad_outputs: Optional[Union[torch.Tensor, Tuple[Optional[torch.Tensor], ...]]],
                                mask: List[bool]) -> Optional[Tuple[Optional[torch.Tensor], ...]]:
        """

        提取与filtered_inputs对应的grad_outputs子集
        提取mask[i]==True的tensor[i]
        
        Args:
            grad_outputs：需要过滤的Tuple[Tensor]
            mask： 根据mask来进行选择，如果为True则选择
        """
        if grad_outputs is None:
            return None
        if isinstance(grad_outputs, torch.Tensor):
            return grad_outputs  # 单个Tensor不需要mask
        filtered = tuple(go for go, m in zip(grad_outputs, mask) if m)
        return filtered

    def restore_tensor_with_none(self, grads: Tuple[torch.Tensor, ...], mask: List[bool]) -> Tuple[Optional[torch.Tensor], ...]:
            """
            将mask != True的使用None填充。
            输入 tensor(len=5)，mask(len=7， 5个True)
            输出tuple[tensor] (len=7， 使用None填充了mask=False的相应位置。)
            """
            grad_iter = iter(grads)
            return tuple(next(grad_iter) if m else None for m in mask)
    
    
    def safe_autograd_grad(self,
            outputs: torch.Tensor,
            inputs: Tuple[torch.Tensor, ...],
            grad_outputs: Optional[Union[torch.Tensor, Tuple[Optional[torch.Tensor], ...]]] = None,
            retain_graph: bool = False,
            create_graph: bool = False,
            allow_unused: bool = False
        ) -> Tuple[Optional[torch.Tensor], ...]:
            """
            安全地计算梯度，输入的output，input，grad_output，必须是同样的len，然后根据input的情况
            限制：必须要输入的三个tuple tensor len是一样的，只有FW1 的model intermediate 使用
            Args:
                outputs (torch.Tensor): _description_ 待着梯度为none
                inputs (Tuple[torch.Tensor, ...]): _description_
                grad_outputs (Optional[Union[torch.Tensor, Tuple[Optional[torch.Tensor], ...]]], optional): _description_. Defaults to None.
                retain_graph (bool, optional): _description_. Defaults to False.
                create_graph (bool, optional): _description_. Defaults to False.
                allow_unused (bool, optional): _description_. Defaults to False.

            Returns:
                Tuple[Optional[torch.Tensor], ...]: _description_
            """
            filtered_inputs, mask = self.filter_tensors_with_grad(inputs)
            if not filtered_inputs:
                return tuple(None for _ in inputs)
            
            filtered_grad_outputs = self.filter_tensor_with_mask(grad_outputs, mask)
            grads = torch.autograd.grad(
                outputs, filtered_inputs,
                grad_outputs=filtered_grad_outputs,
                retain_graph=retain_graph,
                create_graph=create_graph,
                allow_unused=allow_unused
            )
            return self.restore_tensor_with_none(grads, mask)

    def _forward_pass_0(self, model,buffer_id, micro_batch_id):
        if self.stage_id == 0:
            data = self.pipe_buffers['data'][buffer_id] 
            coord = data['coord']
            atype = data['atype']
            box = data['box']
            do_atomic_virial = data['do_atomic_virial']
            fparam = data['fparam']
            aparam = data['aparam']
            cc, bb, fp, ap, input_prec = model.input_type_cast(
                coord, box=box, fparam=fparam, aparam=aparam
            )         
            
            (
                extended_coord,
                extended_atype,
                mapping,
                nlist,
            ) = extend_input_and_build_neighbor_list(
                cc,
                atype,
                model.get_rcut(),
                model.get_sel(),
                # types will be distinguished in the lower interface,
                # so it doesn't need to be distinguished here
                mixed_types=True,
                box=bb,
            )
            nframes, nall = extended_atype.shape[:2]
            extended_coord = extended_coord.view(nframes, -1, 3)
            nlist = model.format_nlist(
                extended_coord, extended_atype, nlist, extra_nlist_sort=False
            )
            extended_coord, _, fp, ap, input_prec = model.input_type_cast(
                extended_coord, fparam=fparam, aparam=aparam
            )
            
            _, nloc, _ = nlist.shape
            atype = extended_atype[:, :nloc]

            if model.atomic_model.pair_excl is not None:
                pair_mask = model.atomic_model.pair_excl(nlist, extended_atype)
                # exclude neighbors in the nlist
                nlist = torch.where(pair_mask == 1, nlist, -1)

            ext_atom_mask = model.atomic_model.make_atom_mask(extended_atype)

            nframes, nloc, nnei = nlist.shape
            atype = extended_atype[:, :nloc]
            if model.atomic_model.do_grad_r() or model.atomic_model.do_grad_c():
                extended_coord.requires_grad_(True)

            comm_dict = None
            parallel_mode = comm_dict is not None
            # cast the input to internal precsion
            extended_coord = extended_coord.to(dtype=model.atomic_model.descriptor.prec)
            nframes, nloc, nnei = nlist.shape
            nall = extended_coord.view(nframes, -1).shape[1] // 3

            if not parallel_mode and model.atomic_model.descriptor.use_loc_mapping:
                node_ebd_ext = model.atomic_model.descriptor.type_embedding(extended_atype[:, :nloc])
            else:
                node_ebd_ext = model.atomic_model.descriptor.type_embedding(extended_atype)
            node_ebd_inp = node_ebd_ext[:, :nloc, :]

            
            middle_output, batch_graph = model.atomic_model.descriptor.repflows(
                nlist,
                extended_coord,
                extended_atype,
                node_ebd_ext,
                mapping,
                comm_dict=comm_dict,
                stage_size = 3,
                stage_index = self.stage_id,
            )
            
            self.pipe_buffers['inputs'].append(data)
            
            batch_graph['mapping'] = mapping
            self.pipe_buffers['batched_graph'].append(batch_graph)
            
            self.pipe_buffers['outputs'].append(middle_output)

            self.pipe_buffers['inputs'].append(middle_output)
            

        elif self.stage_id != self.num_stages - 1:
            
            inputs = self.pipe_buffers['inputs'][buffer_id]
            batch_graph = self.pipe_buffers['batched_graph'][0]
            # logging.info(f"inputs {inputs}")
            middle_output, batch_graph = model.atomic_model.descriptor.repflows(
                nlist=batch_graph['nlist'],
                extended_coord=batch_graph['extended_coord'],
                extended_atype=batch_graph['extended_atype'],
                stage_size = 3,
                stage_index = self.stage_id,
                middle_output = inputs,
                )
            
            # self.pipe_buffers['batched_graph'][buffer_id] = batch_graph
            self.pipe_buffers['outputs'].append(middle_output)
            self.pipe_buffers['inputs'].append(middle_output)
            
            # logging.info(f"outputs {outputs}")

        elif self.stage_id == self.num_stages - 1:
            
            inputs = self.pipe_buffers['inputs'][buffer_id]
            batch_graph = self.pipe_buffers['batched_graph'][0]
            extended_atype = batch_graph['extended_atype']
            node_ebd, edge_ebd, h2, rot_mat,_ = model.atomic_model.descriptor.repflows(
                nlist=batch_graph['nlist'],
                extended_coord=batch_graph['extended_coord'],
                extended_atype=batch_graph['extended_atype'],
                stage_size = 3,
                stage_index = self.stage_id,
                middle_output = inputs,
                )
            
            _,nloc, _ = inputs['nlist'].shape
            atype = extended_atype[:, :nloc]
            fit_ret = model.atomic_model.fitting_net(
                node_ebd,
                atype,
                gr=rot_mat,
                g2=edge_ebd,
                h2=h2,
                fparam=None,
                aparam=None,
            )   
            

            ret_dict = model.atomic_model.apply_out_stat(fit_ret, atype)

            # nf x nloc
            ext_atom_mask = model.atomic_model.make_atom_mask(extended_atype)
            atom_mask = ext_atom_mask[:, :nloc].to(torch.int32)
            if model.atomic_model.atom_excl is not None:
                atom_mask *= model.atomic_model.atom_excl(atype)

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

            ret_dict['energy_redu'] = ret_dict['energy'].squeeze(-1).sum(dim=-1, keepdim=True)

            # model_predict = fit_output_to_model_output(
            #     ret_dict,
            #     model.atomic_output_def(),
            #     batch_graph['extended_coord'],
            #     do_atomic_virial=False,
            #     create_graph=model.training,
            #     mask=ret_dict["mask"] if "mask" in ret_dict else None,
            # )
            # model_predict = model.output_type_cast(model_predict, 'float32')
            
            self.pipe_buffers['outputs'].append(ret_dict)
            E = ret_dict['energy_redu']
            self.pipe_buffers['energy'].append(E)

            target_energy = self.pipe_buffers['targets'][0]['energy']
       
            loss_e = torch.mean(torch.square(E - target_energy))
            loss_e = self.pref_e * loss_e/ nloc
            
            # logging.info(f'loss_e {loss_e}')
            self.pipe_buffers['loss_e'].append(loss_e)
            
     
    def _forward_pass_1(self, buffer_id, micro_batch_id):
        if self.stage_id == self.num_stages - 1:
            E = self.pipe_buffers['energy'][0]
            # logging.info(f"============= E {E}")
            
            # Convert dict of tensors to tuple of tensors
            input_dict = self.pipe_buffers['inputs'][buffer_id]
            if isinstance(input_dict, dict):
                input_tensors = tuple(input_dict.values())
            else:
                input_tensors = input_dict
            
            filtered_inputs, mask = self.filter_tensors_with_grad(input_tensors)
            # logging.info(f"============= mask {mask}")
            # logging.info(f"============= self.pipe_buffers['inputs'][buffer_id] {self.pipe_buffers['inputs'][buffer_id]}")
            # logging.info(f"============= filtered_inputs {filtered_inputs}")

            grad_outputs = torch.ones_like(E)
            import pdb; pdb.set_trace()
            dE_dfeas = self.safe_autograd_grad(E, filtered_inputs, grad_outputs=grad_outputs, create_graph=True, retain_graph=True)
            
            self.pipe_buffers['dE_dinputs'][buffer_id] = self.restore_tensor_with_none(dE_dfeas, mask)
            
            self.pipe_buffers['dE_doutputs'][buffer_id-1] = self.pipe_buffers['dE_dinputs'][buffer_id]
            

        elif self.stage_id != 0:
            
            filtered_dE_doutputs, mask = self.filter_tensors_with_grad(self.pipe_buffers['dE_doutputs'][buffer_id])

            output_dict = self.pipe_buffers['outputs'][buffer_id]
            if isinstance(output_dict, dict):
                output_tensors = tuple(output_dict.values())
            else:
                output_tensors = output_dict
            
            outputs = self.filter_tensor_with_mask(output_tensors, mask)

                        
            input_dict = self.pipe_buffers['inputs'][buffer_id]
            if isinstance(input_dict, dict):
                input_tensors = tuple(input_dict.values())
            else:
                input_tensors = input_dict

            filtered_input_feas = self.filter_tensor_with_mask(input_tensors, mask) # x messsage
            
            dE_dfeas = self.safe_autograd_grad(outputs, filtered_input_feas, grad_outputs=filtered_dE_doutputs, create_graph=True, retain_graph=True)
            

            self.pipe_buffers['dE_dinputs'][buffer_id] = self.restore_tensor_with_none(dE_dfeas, mask)
            self.pipe_buffers['dE_doutputs'][buffer_id-1] = self.pipe_buffers['dE_dinputs'][buffer_id]
            

        elif self.stage_id == 0:
            
            filtered_dE_doutputs, mask = self.filter_tensors_with_grad(self.pipe_buffers['dE_doutputs'][buffer_id])
            
            output_dict = self.pipe_buffers['outputs'][buffer_id]
            if isinstance(output_dict, dict):
                output_tensors = tuple(output_dict.values())
            else:
                output_tensors = output_dict
            outputs = self.filter_tensor_with_mask(output_tensors, mask)
            
            # logging.info(f"mask {mask} filtered_dE_doutputs { filtered_dE_doutputs}, ")
            # logging.info(f"outputs {outputs} ")
            
            batched_graph = self.pipe_buffers['batched_graph'][buffer_id]
            # atoms_per_graph = torch.bincount(batched_graph.atom_owners)
            # atoms_per_graph_list = atoms_per_graph.tolist() 
            
            grad = (
                torch.autograd.grad(outputs, batched_graph['extended_coord'], grad_outputs=filtered_dE_doutputs, create_graph=True, retain_graph=True)
            )
            extended_force = -1 * grad[0]

            ret_dict = {
                'energy_derv_r': extended_force,
            }
            mapping = batched_graph['mapping']
            
            ret_dict = communicate_extended_output_simple(ret_dict, batched_graph['mapping'], do_atomic_virial=False)
            force = ret_dict['energy_derv_r']
            
            
            # TODO: 这里其实是f+s
            loss = torch.mean(torch.square(force - self.pipe_buffers['targets'][buffer_id]['force']))
            loss_f = self.pref_f * loss
            self.pipe_buffers['loss_f'].append(loss_f)


    # mask 记录了中间的有效的feature，方便进行梯度的传递。因为传递有的是图的信息，标量，不涉及导数
    # mask = [True, True, True, False, True, True, False]  
    def _backward_pass_1(self, buffer_id, micro_batch_id):
        if self.stage_id == 0:
            import pdb; pdb.set_trace()
            loss_f = self.pipe_buffers['loss_f'][buffer_id]
            torch.autograd.backward(loss_f / self.global_atom_num, retain_graph=True)
            dE_doutputs = self.pipe_buffers['dE_doutputs'][buffer_id] #FW1 的 inputs，求一下梯度
            dL_dE_doutputs = tuple([t.grad.clone().detach() if (t is not None and t.grad is not None ) else None for t in dE_doutputs]) # 有的dE_doutputs是none，none type没有梯度
            self.pipe_buffers['dL_dE_doutputs'][buffer_id] = dL_dE_doutputs
            
            self.pipe_buffers['dE_doutputs'][buffer_id] = None
            self.pipe_buffers['dL_dE_dinputs'][buffer_id] = None
            self.pipe_buffers['dE_dinputs'][buffer_id] = None
            
            
        elif self.stage_id != self.num_stages - 1:
            dL_dE_dinputs, mask = self.filter_tensors_with_grad(self.pipe_buffers['dL_dE_dinputs'][buffer_id]) #  output grad
            dE_dinputs = self.filter_tensor_with_mask(self.pipe_buffers['dE_dinputs'][buffer_id], mask) # output
            torch.autograd.backward(dE_dinputs, grad_tensors=dL_dE_dinputs, retain_graph=True)

            
            dE_doutputs = self.pipe_buffers['dE_doutputs'][buffer_id]
            self.pipe_buffers['dE_doutputs'][buffer_id] = None
            self.pipe_buffers['dL_dE_dinputs'][buffer_id] = None
            self.pipe_buffers['dE_dinputs'][buffer_id] = None
            dL_dE_doutputs = tuple(t.grad.clone().detach() if (t is not None and t.grad is not None ) else None for t in dE_doutputs)
            self.pipe_buffers['dL_dE_doutputs'][buffer_id] = dL_dE_doutputs
            
        elif self.stage_id == self.num_stages - 1:
            loss_e = self.pipe_buffers['loss_e'][buffer_id]
            dL_dE_dinputs, mask = self.filter_tensors_with_grad(self.pipe_buffers['dL_dE_dinputs'][buffer_id]) #  output grad
            dE_dinputs = self.filter_tensor_with_mask(self.pipe_buffers['dE_dinputs'][buffer_id], mask) # output
            torch.autograd.backward(dE_dinputs, grad_tensors=dL_dE_dinputs, retain_graph=True)
            torch.autograd.backward(loss_e/self.global_energy_size, retain_graph=False)
            # print(f"BW1 loss_e{loss_e}, self.global_energy_size {self.global_energy_size},loss_e/self.global_energy_size {loss_e/self.global_energy_size}")
            dL_dinputs = tuple(t.grad.clone().detach() if (t is not None and t.grad is not None ) else None for t in self.pipe_buffers['inputs'][buffer_id])
            self.pipe_buffers['dL_dinputs'][buffer_id] = dL_dinputs
            
            self.pipe_buffers['inputs'][buffer_id] = None
            self.pipe_buffers['dE_doutputs'][buffer_id] = None
            self.pipe_buffers['dL_dE_dinputs'][buffer_id] = None
            self.pipe_buffers['dE_dinputs'][buffer_id] = None
            
    
    def _backward_pass_0(self, buffer_id, micro_batch_id):
        
        # assert self.stage_id != self.num_stages - 1
        if self.stage_id == self.num_stages - 1:
            # TODO 注意 there is the loss_e.backward() but now is in the BW1 (_backward_pass_1)
            pass
        else:
            dL_doutputs, mask = self.filter_tensors_with_grad(self.pipe_buffers['dL_doutputs'][buffer_id])
            outputs = self.filter_tensor_with_mask(self.pipe_buffers['outputs'][buffer_id], mask) # output
            torch.autograd.backward(outputs, grad_tensors=dL_doutputs, retain_graph=False)
            if self.stage_id != 0:
                dL_dinputs = tuple([t.grad.clone().detach() if (t is not None and t.grad is not None ) else None for t in self.pipe_buffers['inputs'][buffer_id]])
                self.pipe_buffers['dL_dinputs'][buffer_id] = dL_dinputs
            elif self.stage_id == 0:
                self._clear_pipe_buffers(buffer_id, micro_batch_id)

    def forward_pp(self,
                input_dict: dict[str, torch.Tensor],
                model: torch.nn.Module,
                label: dict[str, torch.Tensor],
                natoms: int,
                learning_rate: float,
                mae: bool = False,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
        coef = learning_rate / self.starter_learning_rate
        self.pref_e = self.limit_pref_e + (self.start_pref_e - self.limit_pref_e) * coef
        self.pref_f = self.limit_pref_f + (self.start_pref_f - self.limit_pref_f) * coef

        if os.path.exists('/aisi/mnt/data_nas/liwentao/devel_workspace/deepmd-kit-pp/temp_input/input_dict.pkl'):
            with open('/aisi/mnt/data_nas/liwentao/devel_workspace/deepmd-kit-pp/temp_input/input_dict.pkl', 'rb') as f:
                input_dict = pickle.load(f)
        self.pipe_buffers['data'].append(input_dict)
        self.global_atom_num = input_dict['atype'].shape[1]
        
        self.pipe_buffers['targets'].append(label)
        self.stage_id = 0
        self._forward_pass_0(model, 0, 0)
        self.stage_id = 1
        self._forward_pass_0(model,1, 0)
        self.stage_id = 2
        self._forward_pass_0(model,2, 0)
        self.stage_id = 3
        self._forward_pass_0(model,3, 0)
        self._forward_pass_1(3,0)
        self.stage_id = 2
        self._forward_pass_1(2,0)
        self.stage_id = 1
        self._forward_pass_1(1,0)
        self.stage_id = 0
        self._forward_pass_1(0,0)
        self._backward_pass_1(0,0)

    def forward(
        self,
        input_dict: dict[str, torch.Tensor],
        model: torch.nn.Module,
        label: dict[str, torch.Tensor],
        natoms: int,
        learning_rate: float,
        mae: bool = False,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
        """Return loss on energy and force.

        Parameters
        ----------
        input_dict : dict[str, torch.Tensor]
            Model inputs.
        model : torch.nn.Module
            Model to be used to output the predictions.
        label : dict[str, torch.Tensor]
            Labels.
        natoms : int
            The local atom number.

        Returns
        -------
        model_pred: dict[str, torch.Tensor]
            Model predictions.
        loss: torch.Tensor
            Loss for model to minimize.
        more_loss: dict[str, torch.Tensor]
            Other losses for display.
        """
        if os.path.exists('/aisi/mnt/data_nas/liwentao/devel_workspace/deepmd-kit-pp/temp_input/input_dict.pkl'):
            with open('/aisi/mnt/data_nas/liwentao/devel_workspace/deepmd-kit-pp/temp_input/input_dict.pkl', 'rb') as f:
                input_dict = pickle.load(f)
        model_pred = model(**input_dict)
        import pdb; pdb.set_trace()
        coef = learning_rate / self.starter_learning_rate
        pref_e = self.limit_pref_e + (self.start_pref_e - self.limit_pref_e) * coef
        pref_f = self.limit_pref_f + (self.start_pref_f - self.limit_pref_f) * coef
        pref_v = self.limit_pref_v + (self.start_pref_v - self.limit_pref_v) * coef
        pref_ae = self.limit_pref_ae + (self.start_pref_ae - self.limit_pref_ae) * coef
        pref_pf = self.limit_pref_pf + (self.start_pref_pf - self.limit_pref_pf) * coef
        pref_gf = self.limit_pref_gf + (self.start_pref_gf - self.limit_pref_gf) * coef

        loss = torch.zeros(1, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE)[0]
        more_loss = {}
        # more_loss['log_keys'] = []  # showed when validation on the fly
        # more_loss['test_keys'] = []  # showed when doing dp test
        atom_norm = 1.0 / natoms
        if self.has_e and "energy" in model_pred and "energy" in label:
            energy_pred = model_pred["energy"]
            energy_label = label["energy"]
            if self.enable_atom_ener_coeff and "atom_energy" in model_pred:
                atom_ener_pred = model_pred["atom_energy"]
                # when ener_coeff (\nu) is defined, the energy is defined as
                # E = \sum_i \nu_i E_i
                # instead of the sum of atomic energies.
                #
                # A case is that we want to train reaction energy
                # A + B -> C + D
                # E = - E(A) - E(B) + E(C) + E(D)
                # A, B, C, D could be put far away from each other
                atom_ener_coeff = label["atom_ener_coeff"]
                atom_ener_coeff = atom_ener_coeff.reshape(atom_ener_pred.shape)
                energy_pred = torch.sum(atom_ener_coeff * atom_ener_pred, dim=1)
            find_energy = label.get("find_energy", 0.0)
            pref_e = pref_e * find_energy
            
            if not self.use_l1_all:
                l2_ener_loss = torch.mean(torch.square(energy_pred - energy_label))
                if not self.inference:
                    more_loss["l2_ener_loss"] = self.display_if_exist(
                        l2_ener_loss.detach(), find_energy
                    )
                if not self.use_huber:
                    
                    loss += atom_norm * (pref_e * l2_ener_loss)
                else:
                    l_huber_loss = custom_huber_loss(
                        atom_norm * model_pred["energy"],
                        atom_norm * label["energy"],
                        delta=self.huber_delta,
                    )
                    loss += pref_e * l_huber_loss
                rmse_e = l2_ener_loss.sqrt() * atom_norm
                more_loss["rmse_e"] = self.display_if_exist(
                    rmse_e.detach(), find_energy
                )
                # more_loss['log_keys'].append('rmse_e')
            else:  # use l1 and for all atoms
                l1_ener_loss = F.l1_loss(
                    energy_pred.reshape(-1),
                    energy_label.reshape(-1),
                    reduction="sum",
                )
                loss += pref_e * l1_ener_loss
                more_loss["mae_e"] = self.display_if_exist(
                    F.l1_loss(
                        energy_pred.reshape(-1),
                        energy_label.reshape(-1),
                        reduction="mean",
                    ).detach(),
                    find_energy,
                )
                # more_loss['log_keys'].append('rmse_e')
            if mae:
                mae_e = torch.mean(torch.abs(energy_pred - energy_label)) * atom_norm
                more_loss["mae_e"] = self.display_if_exist(mae_e.detach(), find_energy)
                mae_e_all = torch.mean(torch.abs(energy_pred - energy_label))
                more_loss["mae_e_all"] = self.display_if_exist(
                    mae_e_all.detach(), find_energy
                )

        if (
            (self.has_f or self.has_pf or self.relative_f or self.has_gf)
            and "force" in model_pred
            and "force" in label
        ):
            find_force = label.get("find_force", 0.0)
            pref_f = pref_f * find_force
            force_pred = model_pred["force"]
            force_label = label["force"]
            diff_f = (force_label - force_pred).reshape(-1)

            if self.relative_f is not None:
                force_label_3 = force_label.reshape(-1, 3)
                norm_f = force_label_3.norm(dim=1, keepdim=True) + self.relative_f
                diff_f_3 = diff_f.reshape(-1, 3)
                diff_f_3 = diff_f_3 / norm_f
                diff_f = diff_f_3.reshape(-1)

            if self.has_f:
                if not self.use_l1_all:
                    l2_force_loss = torch.mean(torch.square(diff_f))
                    if not self.inference:
                        more_loss["l2_force_loss"] = self.display_if_exist(
                            l2_force_loss.detach(), find_force
                        )
                    if not self.use_huber:
                        loss += (pref_f * l2_force_loss).to(GLOBAL_PT_FLOAT_PRECISION)
                    else:
                        l_huber_loss = custom_huber_loss(
                            force_pred.reshape(-1),
                            force_label.reshape(-1),
                            delta=self.huber_delta,
                        )
                        loss += pref_f * l_huber_loss
                    rmse_f = l2_force_loss.sqrt()
                    more_loss["rmse_f"] = self.display_if_exist(
                        rmse_f.detach(), find_force
                    )
                else:
                    l1_force_loss = F.l1_loss(force_label, force_pred, reduction="none")
                    more_loss["mae_f"] = self.display_if_exist(
                        l1_force_loss.mean().detach(), find_force
                    )
                    l1_force_loss = l1_force_loss.sum(-1).mean(-1).sum()
                    loss += (pref_f * l1_force_loss).to(GLOBAL_PT_FLOAT_PRECISION)
                if mae:
                    mae_f = torch.mean(torch.abs(diff_f))
                    more_loss["mae_f"] = self.display_if_exist(
                        mae_f.detach(), find_force
                    )

            if self.has_pf and "atom_pref" in label:
                atom_pref = label["atom_pref"]
                find_atom_pref = label.get("find_atom_pref", 0.0)
                pref_pf = pref_pf * find_atom_pref
                atom_pref_reshape = atom_pref.reshape(-1)
                l2_pref_force_loss = (torch.square(diff_f) * atom_pref_reshape).mean()
                if not self.inference:
                    more_loss["l2_pref_force_loss"] = self.display_if_exist(
                        l2_pref_force_loss.detach(), find_atom_pref
                    )
                loss += (pref_pf * l2_pref_force_loss).to(GLOBAL_PT_FLOAT_PRECISION)
                rmse_pf = l2_pref_force_loss.sqrt()
                more_loss["rmse_pf"] = self.display_if_exist(
                    rmse_pf.detach(), find_atom_pref
                )

            if self.has_gf and "drdq" in label:
                drdq = label["drdq"]
                find_drdq = label.get("find_drdq", 0.0)
                pref_gf = pref_gf * find_drdq
                force_reshape_nframes = force_pred.reshape(-1, natoms * 3)
                force_label_reshape_nframes = force_label.reshape(-1, natoms * 3)
                drdq_reshape = drdq.reshape(-1, natoms * 3, self.numb_generalized_coord)
                gen_force_label = torch.einsum(
                    "bij,bi->bj", drdq_reshape, force_label_reshape_nframes
                )
                gen_force = torch.einsum(
                    "bij,bi->bj", drdq_reshape, force_reshape_nframes
                )
                diff_gen_force = gen_force_label - gen_force
                l2_gen_force_loss = torch.square(diff_gen_force).mean()
                if not self.inference:
                    more_loss["l2_gen_force_loss"] = self.display_if_exist(
                        l2_gen_force_loss.detach(), find_drdq
                    )
                loss += (pref_gf * l2_gen_force_loss).to(GLOBAL_PT_FLOAT_PRECISION)
                rmse_gf = l2_gen_force_loss.sqrt()
                more_loss["rmse_gf"] = self.display_if_exist(
                    rmse_gf.detach(), find_drdq
                )
        import pdb; pdb.set_trace()
        if self.has_v and "virial" in model_pred and "virial" in label:
            find_virial = label.get("find_virial", 0.0)
            pref_v = pref_v * find_virial
            diff_v = label["virial"] - model_pred["virial"].reshape(-1, 9)
            l2_virial_loss = torch.mean(torch.square(diff_v))
            if not self.inference:
                more_loss["l2_virial_loss"] = self.display_if_exist(
                    l2_virial_loss.detach(), find_virial
                )
            if not self.use_huber:
                loss += atom_norm * (pref_v * l2_virial_loss)
            else:
                l_huber_loss = custom_huber_loss(
                    atom_norm * model_pred["virial"].reshape(-1),
                    atom_norm * label["virial"].reshape(-1),
                    delta=self.huber_delta,
                )
                loss += pref_v * l_huber_loss
            rmse_v = l2_virial_loss.sqrt() * atom_norm
            more_loss["rmse_v"] = self.display_if_exist(rmse_v.detach(), find_virial)
            if mae:
                mae_v = torch.mean(torch.abs(diff_v)) * atom_norm
                more_loss["mae_v"] = self.display_if_exist(mae_v.detach(), find_virial)

        if self.has_ae and "atom_energy" in model_pred and "atom_ener" in label:
            atom_ener = model_pred["atom_energy"]
            atom_ener_label = label["atom_ener"]
            find_atom_ener = label.get("find_atom_ener", 0.0)
            pref_ae = pref_ae * find_atom_ener
            atom_ener_reshape = atom_ener.reshape(-1)
            atom_ener_label_reshape = atom_ener_label.reshape(-1)
            l2_atom_ener_loss = torch.square(
                atom_ener_label_reshape - atom_ener_reshape
            ).mean()
            if not self.inference:
                more_loss["l2_atom_ener_loss"] = self.display_if_exist(
                    l2_atom_ener_loss.detach(), find_atom_ener
                )
            if not self.use_huber:
                loss += (pref_ae * l2_atom_ener_loss).to(GLOBAL_PT_FLOAT_PRECISION)
            else:
                l_huber_loss = custom_huber_loss(
                    atom_ener_reshape,
                    atom_ener_label_reshape,
                    delta=self.huber_delta,
                )
                loss += pref_ae * l_huber_loss
            rmse_ae = l2_atom_ener_loss.sqrt()
            more_loss["rmse_ae"] = self.display_if_exist(
                rmse_ae.detach(), find_atom_ener
            )

        if not self.inference:
            more_loss["rmse"] = torch.sqrt(loss.detach())
        return model_pred, loss, more_loss

    @property
    def label_requirement(self) -> list[DataRequirementItem]:
        """Return data label requirements needed for this loss calculation."""
        label_requirement = []
        if self.has_e:
            label_requirement.append(
                DataRequirementItem(
                    "energy",
                    ndof=1,
                    atomic=False,
                    must=False,
                    high_prec=True,
                )
            )
        if self.has_f:
            label_requirement.append(
                DataRequirementItem(
                    "force",
                    ndof=3,
                    atomic=True,
                    must=False,
                    high_prec=False,
                )
            )
        if self.has_v:
            label_requirement.append(
                DataRequirementItem(
                    "virial",
                    ndof=9,
                    atomic=False,
                    must=False,
                    high_prec=False,
                )
            )
        if self.has_ae:
            label_requirement.append(
                DataRequirementItem(
                    "atom_ener",
                    ndof=1,
                    atomic=True,
                    must=False,
                    high_prec=False,
                )
            )
        if self.has_pf:
            label_requirement.append(
                DataRequirementItem(
                    "atom_pref",
                    ndof=1,
                    atomic=True,
                    must=False,
                    high_prec=False,
                    repeat=3,
                )
            )
        if self.has_gf > 0:
            label_requirement.append(
                DataRequirementItem(
                    "drdq",
                    ndof=self.numb_generalized_coord * 3,
                    atomic=True,
                    must=False,
                    high_prec=False,
                )
            )
        if self.enable_atom_ener_coeff:
            label_requirement.append(
                DataRequirementItem(
                    "atom_ener_coeff",
                    ndof=1,
                    atomic=True,
                    must=False,
                    high_prec=False,
                    default=1.0,
                )
            )
        return label_requirement

    def serialize(self) -> dict:
        """Serialize the loss module.

        Returns
        -------
        dict
            The serialized loss module
        """
        return {
            "@class": "EnergyLoss",
            "@version": 2,
            "starter_learning_rate": self.starter_learning_rate,
            "start_pref_e": self.start_pref_e,
            "limit_pref_e": self.limit_pref_e,
            "start_pref_f": self.start_pref_f,
            "limit_pref_f": self.limit_pref_f,
            "start_pref_v": self.start_pref_v,
            "limit_pref_v": self.limit_pref_v,
            "start_pref_ae": self.start_pref_ae,
            "limit_pref_ae": self.limit_pref_ae,
            "start_pref_pf": self.start_pref_pf,
            "limit_pref_pf": self.limit_pref_pf,
            "relative_f": self.relative_f,
            "enable_atom_ener_coeff": self.enable_atom_ener_coeff,
            "start_pref_gf": self.start_pref_gf,
            "limit_pref_gf": self.limit_pref_gf,
            "numb_generalized_coord": self.numb_generalized_coord,
            "use_huber": self.use_huber,
            "huber_delta": self.huber_delta,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "TaskLoss":
        """Deserialize the loss module.

        Parameters
        ----------
        data : dict
            The serialized loss module

        Returns
        -------
        Loss
            The deserialized loss module
        """
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 2, 1)
        data.pop("@class")
        return cls(**data)


class EnergyHessianStdLoss(EnergyStdLoss):
    def __init__(
        self,
        start_pref_h: float = 0.0,
        limit_pref_h: float = 0.0,
        **kwargs: Any,
    ) -> None:
        r"""Enable the layer to compute loss on hessian.

        Parameters
        ----------
        start_pref_h : float
            The prefactor of hessian loss at the start of the training.
        limit_pref_h : float
            The prefactor of hessian loss at the end of the training.
        **kwargs
            Other keyword arguments.
        """
        super().__init__(**kwargs)
        self.has_h = (start_pref_h != 0.0 and limit_pref_h != 0.0) or self.inference

        self.start_pref_h = start_pref_h
        self.limit_pref_h = limit_pref_h

    def forward(
        self,
        input_dict: dict[str, torch.Tensor],
        model: torch.nn.Module,
        label: dict[str, torch.Tensor],
        natoms: int,
        learning_rate: float,
        mae: bool = False,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
        model_pred, loss, more_loss = super().forward(
            input_dict, model, label, natoms, learning_rate, mae=mae
        )
        coef = learning_rate / self.starter_learning_rate
        pref_h = self.limit_pref_h + (self.start_pref_h - self.limit_pref_h) * coef

        if self.has_h and "hessian" in model_pred and "hessian" in label:
            find_hessian = label.get("find_hessian", 0.0)
            pref_h = pref_h * find_hessian
            diff_h = label["hessian"].reshape(
                -1,
            ) - model_pred["hessian"].reshape(
                -1,
            )
            l2_hessian_loss = torch.mean(torch.square(diff_h))
            if not self.inference:
                more_loss["l2_hessian_loss"] = self.display_if_exist(
                    l2_hessian_loss.detach(), find_hessian
                )
            loss += pref_h * l2_hessian_loss
            rmse_h = l2_hessian_loss.sqrt()
            more_loss["rmse_h"] = self.display_if_exist(rmse_h.detach(), find_hessian)
            if mae:
                mae_h = torch.mean(torch.abs(diff_h))
                more_loss["mae_h"] = self.display_if_exist(mae_h.detach(), find_hessian)

        if not self.inference:
            more_loss["rmse"] = torch.sqrt(loss.detach())
        return model_pred, loss, more_loss

    @property
    def label_requirement(self) -> list[DataRequirementItem]:
        """Add hessian label requirement needed for this loss calculation."""
        label_requirement = super().label_requirement
        if self.has_h:
            label_requirement.append(
                DataRequirementItem(
                    "hessian",
                    ndof=1,  # 9=3*3 --> 3N*3N=ndof*natoms*natoms
                    atomic=True,
                    must=False,
                    high_prec=False,
                )
            )
        return label_requirement
