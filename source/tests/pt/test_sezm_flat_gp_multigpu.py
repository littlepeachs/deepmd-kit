# SPDX-License-Identifier: LGPL-3.0-or-later
"""Distributed tests for SeZM flat graph parallelism.

Run with, for example:

    torchrun --standalone --nproc_per_node=2 source/tests/pt/test_sezm_flat_gp_multigpu.py
"""

from __future__ import (
    annotations,
)

import signal
import sys
import unittest
import os

import torch
import torch.distributed as dist

from deepmd.pt.model.model import (
    get_sezm_model,
)
from deepmd.pt.utils.graph_parallel import (
    clear_graph_parallel_context,
    set_graph_parallel_context,
)
from deepmd.pt.utils.graph_parallel_flat import (
    build_flat_graph_partition,
)
from deepmd.pt.utils.nlist import (
    build_precomputed_flat_graph,
)
from deepmd.pt.utils.sezm_moe_ep_dp import (
    sync_moe_gradients,
)


def setup_dist() -> tuple[int, int, torch.device]:
    """Initialize distributed process group from torchrun environment."""
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world = dist.get_world_size()
    if torch.cuda.is_available():
        torch.cuda.set_device(rank % torch.cuda.device_count())
        device = torch.device("cuda", rank % torch.cuda.device_count())
    else:
        device = torch.device("cpu")
    return rank, world, device


def teardown_dist() -> None:
    """Destroy distributed state."""
    clear_graph_parallel_context()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def deadlock_alarm(signum, frame) -> None:
    """Fail when a collective hangs."""
    raise TimeoutError(f"rank {dist.get_rank()} timed out")


def _model_params(*, use_moe: bool, ep_size: int = 1) -> dict:
    descriptor = {
        "type": "SeZM",
        "sel": [4, 4],
        "rcut": 2.2,
        "ntypes": 2,
        "channels": 4,
        "n_focus": 1,
        "n_radial": 3,
        "radial_mlp": [6],
        "use_env_seed": False,
        "l_schedule": [1, 0],
        "mmax": 1,
        "so2_norm": False,
        "so2_layers": 1,
        "n_atten_head": 0,
        "ffn_neurons": 8,
        "ffn_blocks": 1,
        "mlp_bias": True,
        "layer_scale": False,
        "use_amp": False,
        "activation_function": "silu",
        "glu_activation": True,
        "precision": "float64",
        "seed": 20260626,
    }
    if use_moe:
        descriptor.update(
            {
                "use_moe": True,
                "n_focus": 3,
                "n_routing_experts": 8,
                "topk": 2,
                "n_shared_experts": 1,
                "ep_size": ep_size,
                "routing_input": "dst",
                "so2_attn_res": "none",
                "use_compile": False,
            }
        )
    return {
        "type": "SeZM",
        "type_map": ["O", "H"],
        "descriptor": descriptor,
        "fitting_net": {
            "neuron": [8],
            "activation_function": "silu",
            "precision": "float64",
            "seed": 20260626,
        },
        "use_compile": False,
    }


def _flat_inputs(device: torch.device) -> dict[str, torch.Tensor]:
    """Build a two-frame flat batch with enough atoms for 8-way partitions."""
    coord = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.2, 0.1, 0.0],
            [1.1, 0.2, 0.1],
            [0.1, 1.2, 0.2],
            [0.2, 0.0, 1.1],
            [1.2, 1.1, 0.1],
        ],
        dtype=torch.float64,
        device=device,
    )
    atype = torch.tensor(
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.long, device=device
    )
    batch = torch.tensor(
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.long, device=device
    )
    ptr = torch.tensor([0, 5, 10], dtype=torch.long, device=device)
    graph = build_precomputed_flat_graph(
        coord,
        atype,
        batch,
        ptr,
        rcut=2.2,
        sel=[4, 4],
        a_rcut=2.2,
        a_sel=4,
        mixed_types=True,
        box=None,
        ntypes=2,
    )
    graph.update({"coord": coord, "atype": atype, "batch": batch, "ptr": ptr})
    return graph


def _model_kwargs(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    keys = (
        "coord",
        "atype",
        "batch",
        "ptr",
        "extended_atype",
        "extended_batch",
        "extended_image",
        "mapping",
        "central_ext_index",
        "nlist",
        "nlist_ext",
        "a_nlist",
        "a_nlist_ext",
        "nlist_mask",
        "a_nlist_mask",
        "edge_index",
        "angle_index",
    )
    return {key: inputs[key] for key in keys}


def _partition(
    inputs: dict[str, torch.Tensor],
    *,
    rank: int,
    world: int,
) -> dict:
    return build_flat_graph_partition(
        int(inputs["ptr"][-1].item()),
        inputs["edge_index"],
        inputs["angle_index"],
        batch=inputs["batch"],
        rank=rank,
        world_size=world,
    ).asdict()


def _reshard_state_for_ep(
    full_state: dict[str, torch.Tensor],
    *,
    ep_rank: int,
    ep_size: int,
    n_routing_experts: int,
) -> dict[str, torch.Tensor]:
    n_per_gpu = n_routing_experts // ep_size
    start = ep_rank * n_per_gpu
    end = start + n_per_gpu
    local_state: dict[str, torch.Tensor] = {}
    for key, value in full_state.items():
        if ".routing_matrix" in key or ".routing_bias" in key:
            local_state[key] = value[start:end].clone()
        else:
            local_state[key] = value.clone()
    return local_state


def _build_matched_models(
    *,
    use_moe: bool,
    world: int,
    rank: int,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    ref = get_sezm_model(_model_params(use_moe=use_moe, ep_size=1)).to(device)
    gp = get_sezm_model(_model_params(use_moe=use_moe, ep_size=world)).to(device)
    ref.eval()
    gp.eval()
    if use_moe:
        state = _reshard_state_for_ep(
            ref.state_dict(),
            ep_rank=rank,
            ep_size=world,
            n_routing_experts=8,
        )
        gp.load_state_dict(state, strict=True)
    else:
        gp.load_state_dict(ref.state_dict(), strict=True)
    return ref, gp


def _all_gather_variable(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """All-gather tensors with variable size on ``dim``."""
    world = dist.get_world_size()
    local_size = torch.tensor(
        [tensor.size(dim)], dtype=torch.long, device=tensor.device
    )
    size_list = [torch.zeros_like(local_size) for _ in range(world)]
    dist.all_gather(size_list, local_size)
    sizes = [int(item.item()) for item in size_list]
    max_size = max(sizes)
    if tensor.size(dim) < max_size:
        pad_shape = list(tensor.shape)
        pad_shape[dim] = max_size - tensor.size(dim)
        tensor = torch.cat(
            [tensor, tensor.new_zeros(pad_shape)],
            dim=dim,
        )
    gathered = [torch.empty_like(tensor) for _ in range(world)]
    dist.all_gather(gathered, tensor.contiguous())
    trimmed = [part.narrow(dim, 0, sizes[idx]) for idx, part in enumerate(gathered)]
    return torch.cat(trimmed, dim=dim).contiguous()


def _assert_world_consistent(tensor: torch.Tensor) -> None:
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor.contiguous())
    for other in gathered:
        torch.testing.assert_close(tensor, other, atol=0.0, rtol=0.0)


def _assert_outputs_close(
    ref: dict[str, torch.Tensor],
    got: dict[str, torch.Tensor],
    *,
    atol: float = 1e-8,
    rtol: float = 1e-8,
) -> None:
    for key in ("atom_energy", "energy", "force", "mask"):
        torch.testing.assert_close(got[key], ref[key], atol=atol, rtol=rtol, msg=key)


def _core_compute_flat(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    *,
    flat_graph_partition: dict | None = None,
) -> dict[str, torch.Tensor]:
    coord = inputs["coord"].detach().clone()
    atype = inputs["atype"].reshape(-1)
    extended_coord = coord.index_select(0, inputs["mapping"])
    return model.core_compute_flat(
        coord=coord,
        atype=atype,
        batch=inputs["batch"],
        ptr=inputs["ptr"],
        extended_coord=extended_coord,
        extended_atype=inputs["extended_atype"],
        extended_batch=inputs["extended_batch"],
        mapping=inputs["mapping"],
        central_ext_index=inputs["central_ext_index"],
        nlist_ext=inputs["nlist_ext"],
        nlist_mask=inputs["nlist_mask"],
        edge_index=inputs["edge_index"],
        flat_graph_partition=flat_graph_partition,
    )


def _assert_core_outputs_close(
    ref: dict[str, torch.Tensor],
    got: dict[str, torch.Tensor],
    *,
    atol: float = 1e-8,
    rtol: float = 1e-8,
) -> None:
    for key in ("energy", "energy_redu", "mask"):
        torch.testing.assert_close(got[key], ref[key], atol=atol, rtol=rtol, msg=key)


class TestSeZMFlatGraphParallel(unittest.TestCase):
    """Distributed flat GP tests for descriptor, model, and training smoke."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.rank, cls.world, cls.device = setup_dist()

    @classmethod
    def tearDownClass(cls) -> None:
        """Keep process group alive until run_tests aggregates results."""

    def setUp(self) -> None:
        signal.signal(signal.SIGALRM, deadlock_alarm)
        signal.alarm(180)

    def tearDown(self) -> None:
        signal.alarm(0)
        clear_graph_parallel_context()

    def _require_supported_world(self) -> None:
        if self.world not in {2, 4, 8}:
            self.skipTest("flat GP tests are defined for 2, 4, or 8 ranks")

    def _set_gp_context(self, gp_model: torch.nn.Module) -> None:
        descriptor = gp_model.atomic_model.descriptor
        group = (
            descriptor.moe_ep_group
            if descriptor.moe_ep_group is not None
            else dist.group.WORLD
        )
        set_graph_parallel_context(
            True,
            group,
            self.rank,
            self.world,
            reduce_backward=False,
        )

    def _descriptor_compare(self, *, use_moe: bool) -> None:
        self._require_supported_world()
        inputs = _flat_inputs(self.device)
        part = _partition(inputs, rank=self.rank, world=self.world)
        ref, gp = _build_matched_models(
            use_moe=use_moe,
            world=self.world,
            rank=self.rank,
            device=self.device,
        )

        coord_ref = inputs["coord"].detach().clone().requires_grad_(True)
        ext_ref = coord_ref.index_select(0, inputs["mapping"])
        edge_index_ref, edge_vec_ref, edge_mask_ref = (
            ref.build_edge_list_from_flat_graph(
                extended_coord=ext_ref,
                central_ext_index=inputs["central_ext_index"],
                nlist_ext=inputs["nlist_ext"],
                nlist_mask=inputs["nlist_mask"],
                edge_index=inputs["edge_index"],
            )
        )
        desc_ref, _ = ref.atomic_model.descriptor.forward_with_edges(
            extended_coord=coord_ref.reshape(1, -1, 3),
            extended_atype=inputs["atype"].reshape(1, -1),
            edge_index=edge_index_ref,
            edge_vec=edge_vec_ref,
            edge_mask=edge_mask_ref,
        )
        grad_ref = (
            None if use_moe else torch.autograd.grad(desc_ref.sum(), coord_ref)[0]
        )

        coord_gp = inputs["coord"].detach().clone().requires_grad_(True)
        ext_gp = coord_gp.index_select(0, inputs["mapping"])
        edge_index_gp, edge_vec_gp, edge_mask_gp = gp.build_edge_list_from_flat_graph(
            extended_coord=ext_gp,
            central_ext_index=inputs["central_ext_index"],
            nlist_ext=inputs["nlist_ext"],
            nlist_mask=inputs["nlist_mask"],
            edge_index=part["edge_index"],
            edge_ids=part["edge_ids"],
            dummy_owner=(part["local_start"] if part["local_size"] > 0 else None),
        )
        self._set_gp_context(gp)
        desc_local, _ = gp.atomic_model.descriptor.forward_with_edges(
            extended_coord=coord_gp.reshape(1, -1, 3),
            extended_atype=inputs["atype"].reshape(1, -1),
            edge_index=edge_index_gp,
            edge_vec=edge_vec_gp,
            edge_mask=edge_mask_gp,
            flat_graph_partition=part,
        )
        desc_gp = _all_gather_variable(desc_local.squeeze(0), dim=0).reshape(
            1, -1, desc_local.shape[-1]
        )

        torch.testing.assert_close(desc_gp, desc_ref, atol=1e-8, rtol=1e-8)
        if grad_ref is not None:
            grad_gp = torch.autograd.grad(desc_local.sum(), coord_gp)[0]
            dist.all_reduce(grad_gp, op=dist.ReduceOp.SUM)
            torch.testing.assert_close(grad_gp, grad_ref, atol=1e-8, rtol=1e-8)
        if self.rank == 0:
            sys.stdout.write(
                f"descriptor use_moe={use_moe} maxdiff="
                f"{(desc_gp - desc_ref).abs().max().item():.17e}\n"
            )

    def test_descriptor_gp_matches_full_sparse(self) -> None:
        self._descriptor_compare(use_moe=False)

    def test_descriptor_moe_gp_matches_full_sparse(self) -> None:
        self._descriptor_compare(use_moe=True)

    def _model_compare(self, *, use_moe: bool) -> None:
        self._require_supported_world()
        inputs = _flat_inputs(self.device)
        part = _partition(inputs, rank=self.rank, world=self.world)
        ref, gp = _build_matched_models(
            use_moe=use_moe,
            world=self.world,
            rank=self.rank,
            device=self.device,
        )

        if use_moe:
            clear_graph_parallel_context()
            out_ref = _core_compute_flat(ref, inputs)
            self._set_gp_context(gp)
            out_gp = _core_compute_flat(gp, inputs, flat_graph_partition=part)
            _assert_core_outputs_close(out_ref, out_gp)
            maxdiff = (out_gp["energy"] - out_ref["energy"]).abs().max()
        else:
            clear_graph_parallel_context()
            out_ref = ref(**_model_kwargs(inputs))
            self._set_gp_context(gp)
            out_gp = gp(**_model_kwargs(inputs), flat_graph_partition=part)
            _assert_outputs_close(out_ref, out_gp)
            maxdiff = (out_gp["force"] - out_ref["force"]).abs().max()
        if self.rank == 0:
            sys.stdout.write(f"model use_moe={use_moe} maxdiff={maxdiff.item():.17e}\n")

    def test_flat_model_gp_matches_non_gp(self) -> None:
        self._model_compare(use_moe=False)

    def test_flat_model_moe_gp_matches_non_gp(self) -> None:
        self._model_compare(use_moe=True)

    def test_moe_gp_backward_and_gradient_sync_smoke(self) -> None:
        if os.environ.get("DEEPMD_RUN_SEZM_FLAT_GP_BACKWARD") != "1":
            self.skipTest(
                "set DEEPMD_RUN_SEZM_FLAT_GP_BACKWARD=1 for the full-model "
                "MoE+GP backward smoke"
            )
        self._require_supported_world()
        inputs = _flat_inputs(self.device)
        part = _partition(inputs, rank=self.rank, world=self.world)
        _, gp = _build_matched_models(
            use_moe=True,
            world=self.world,
            rank=self.rank,
            device=self.device,
        )
        gp.train()
        self._set_gp_context(gp)
        out = _core_compute_flat(gp, inputs, flat_graph_partition=part)
        loss = out["energy_redu"].square().mean()
        loss.backward()

        descriptor = gp.atomic_model.descriptor
        sync_moe_gradients(
            gp,
            descriptor.moe_dp_group,
            None,
            descriptor.moe_dp_size,
            self.world,
            non_routing_divisor=1.0,
            routing_expert_divisor=1.0,
        )

        routing_grad_found = False
        synced_grad_checked = False
        for name, param in gp.named_parameters():
            if param.grad is None:
                continue
            self.assertTrue(torch.isfinite(param.grad).all().item(), name)
            if ".routing_matrix" in name:
                routing_grad_found = True
            elif not synced_grad_checked:
                _assert_world_consistent(param.grad)
                synced_grad_checked = True

        self.assertTrue(routing_grad_found)
        self.assertTrue(synced_grad_checked)
        if self.rank == 0:
            sys.stdout.write(f"moe gp train loss={loss.detach().item():.17e}\n")


def run_tests() -> bool:
    """Run unittest suite and aggregate success across ranks."""
    rank, _, device = setup_dist()
    if rank == 0:
        sys.stdout.write("Running SeZM flat GP distributed tests\n")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSeZMFlatGraphParallel)
    result = unittest.TextTestRunner(verbosity=2 if rank == 0 else 0).run(suite)
    success = torch.tensor(
        [1 if result.wasSuccessful() else 0],
        dtype=torch.int32,
        device=device,
    )
    dist.all_reduce(success, op=dist.ReduceOp.MIN)
    if rank == 0:
        status = "PASS" if success.item() == 1 else "FAIL"
        sys.stdout.write(f"{status}: SeZM flat GP distributed tests\n")
    teardown_dist()
    return success.item() == 1


if __name__ == "__main__":
    raise SystemExit(0 if run_tests() else 1)
