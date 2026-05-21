# SPDX-License-Identifier: LGPL-3.0-or-later
"""Multi-GPU tests for SeZM MoE SO(2) convolution."""

from __future__ import (
    annotations,
)

import signal
import sys
import unittest

import torch
import torch.distributed as dist

from deepmd.pt.model.descriptor.sezm_nn.moe.conv import (
    MoESO2Convolution,
)


def setup_dist() -> tuple[int, int, torch.device]:
    """Initialize torchrun distributed state."""
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
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def deadlock_alarm(signum, frame) -> None:
    """Fail the process when a collective hangs."""
    raise TimeoutError(f"rank {dist.get_rank()} timed out in multi-GPU test")


def make_ep_group(ep_size: int) -> tuple[dist.ProcessGroup, int]:
    """Create contiguous EP groups and return this rank's group and ep_rank."""
    rank = dist.get_rank()
    world = dist.get_world_size()
    if world % ep_size != 0:
        raise ValueError(f"world size {world} must be divisible by ep_size={ep_size}")
    if ep_size == world:
        return dist.group.WORLD, rank

    my_group = None
    my_ep_rank = -1
    for start in range(0, world, ep_size):
        ranks = list(range(start, start + ep_size))
        group = dist.new_group(ranks=ranks)
        if rank in ranks:
            my_group = group
            my_ep_rank = rank - start
    assert my_group is not None
    return my_group, my_ep_rank


def make_conv(
    *,
    ep_size: int,
    n_routing_experts: int = 8,
    topk: int = 2,
    n_shared_experts: int = 1,
    seed: int = 20260518,
) -> MoESO2Convolution:
    """Build a deterministic conv module for tests."""
    conv = MoESO2Convolution(
        lmax=3,
        mmax=1,
        focus_dim=8,
        n_routing_experts=n_routing_experts,
        topk=topk,
        n_shared_experts=n_shared_experts,
        ep_size=ep_size,
        routing_input="dst",
        routing_key_dim=n_routing_experts,
        so2_layers=4,
        activation_function="silu",
        mlp_bias=False,
        use_layer_scale=False,
        precision="float64",
        seed=seed,
    )
    with torch.no_grad():
        conv.router.gate.matrix.zero_()
        diag = min(conv.router.gate.matrix.shape)
        for idx in range(diag):
            conv.router.gate.matrix[idx, idx] = 10.0
    return conv


def make_inputs(
    conv: MoESO2Convolution,
    n_edge: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create identical inputs on every rank."""
    torch.manual_seed(13579)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(13579)
    device = conv.experts.routing_stack.layers[0].routing_matrix_m0.device
    x_local = torch.randn(
        n_edge,
        conv.n_focus,
        conv.reduced_dim,
        conv.focus_dim,
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    routing_key = torch.zeros(
        n_edge,
        conv.routing_key_dim,
        dtype=torch.float64,
        device=device,
    )
    for edge_idx in range(n_edge):
        routing_key[edge_idx, edge_idx % conv.routing_key_dim] = 10.0
        routing_key[edge_idx, (edge_idx + 1) % conv.routing_key_dim] = 10.0
    routing_key.requires_grad_(True)
    return x_local, routing_key


def reshard_routing_state_for_ep(
    full_state_dict: dict[str, torch.Tensor],
    ep_rank: int,
    ep_size: int,
    n_routing_experts: int,
) -> dict[str, torch.Tensor]:
    """Extract the local routing expert shard for one EP rank."""
    n_per_gpu = n_routing_experts // ep_size
    start = ep_rank * n_per_gpu
    end = start + n_per_gpu
    local_state = {}
    for key, value in full_state_dict.items():
        if ".routing_matrix" in key or ".routing_bias" in key:
            local_state[key] = value[start:end].clone()
        else:
            local_state[key] = value.clone()
    return local_state


def assert_routing_grads_present(module: MoESO2Convolution) -> None:
    """Check local routing expert parameters received gradients."""
    found = False
    for name, param in module.named_parameters():
        if ".routing_matrix" in name:
            found = True
            assert param.grad is not None, name
            assert torch.isfinite(param.grad).all(), name
    assert found


class TestMoESO2ConvolutionMultiGPU(unittest.TestCase):
    """Multi-GPU tests for Step 4 MoESO2Convolution."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.rank, cls.world, cls.device = setup_dist()

    @classmethod
    def tearDownClass(cls) -> None:
        """Keep process group alive for result aggregation."""

    def setUp(self) -> None:
        signal.signal(signal.SIGALRM, deadlock_alarm)
        signal.alarm(120)

    def tearDown(self) -> None:
        signal.alarm(0)

    def test_multigpu_forward_shape(self) -> None:
        if self.world != 4:
            self.skipTest("T_D1 runs with 4 GPUs")
        ep_group, _ = make_ep_group(ep_size=4)
        conv = make_conv(ep_size=4)
        x_local, routing_key = make_inputs(conv)

        out = conv(x_local, routing_key, ep_group=ep_group)

        self.assertEqual(out.shape, x_local.shape)
        self.assertTrue(torch.isfinite(out).all().item())

    def test_multigpu_backward_no_deadlock(self) -> None:
        if self.world != 4:
            self.skipTest("T_D2 runs with 4 GPUs")
        ep_group, _ = make_ep_group(ep_size=4)
        conv = make_conv(ep_size=4)
        x_local, routing_key = make_inputs(conv)

        loss = conv(x_local, routing_key, ep_group=ep_group).sum()
        loss.backward()

        assert_routing_grads_present(conv)

    def test_multigpu_second_backward_no_deadlock(self) -> None:
        if self.world != 4:
            self.skipTest("T_D3 runs with 4 GPUs")
        ep_group, _ = make_ep_group(ep_size=4)
        conv = make_conv(ep_size=4)
        x_local, routing_key = make_inputs(conv)

        loss = conv(x_local, routing_key, ep_group=ep_group).sum()
        grad_x, grad_key = torch.autograd.grad(
            loss,
            (x_local, routing_key),
            create_graph=True,
        )
        (grad_x.sum() + grad_key.sum()).backward()

        assert_routing_grads_present(conv)

    def test_single_vs_multi_gpu_equivalence(self) -> None:
        if self.world != 4:
            self.skipTest("T_D4 runs with 4 GPUs")
        ep_group, ep_rank = make_ep_group(ep_size=4)
        single = make_conv(ep_size=1)
        multi = make_conv(ep_size=4)
        local_state = reshard_routing_state_for_ep(
            single.state_dict(),
            ep_rank=ep_rank,
            ep_size=4,
            n_routing_experts=8,
        )
        multi.load_state_dict(local_state)
        x_local, routing_key = make_inputs(single)

        with torch.no_grad():
            y_single = single(x_local, routing_key, ep_group=None)
            y_multi = multi(x_local, routing_key, ep_group=ep_group)
        max_diff = (y_single - y_multi).abs().max()
        dist.all_reduce(max_diff, op=dist.ReduceOp.MAX)
        if self.rank == 0:
            sys.stdout.write(f"T_D4 max_diff={max_diff.item():.17e}\n")

        torch.testing.assert_close(y_multi, y_single, atol=1e-10, rtol=1e-10)

    def test_multigpu_ep2_forward_backward(self) -> None:
        if self.world != 8:
            self.skipTest("T_D5 runs with 8 GPUs")
        # With world=8 and ep_size=2 this creates four EP groups; it is a
        # forward/backward smoke for mixed EP+replica execution. Gradient sync
        # across replicas belongs to Step 6.
        ep_group, _ = make_ep_group(ep_size=2)
        conv = make_conv(ep_size=2, n_routing_experts=4, topk=2, n_shared_experts=1)
        x_local, routing_key = make_inputs(conv)

        out = conv(x_local, routing_key, ep_group=ep_group)
        self.assertEqual(out.shape, x_local.shape)
        self.assertTrue(torch.isfinite(out).all().item())
        out.sum().backward()
        assert_routing_grads_present(conv)


def run_tests() -> bool:
    """Run unittest suite and aggregate success across ranks."""
    rank, _, device = setup_dist()
    if rank == 0:
        sys.stdout.write("Running MoESO2Convolution multi-GPU tests\n")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMoESO2ConvolutionMultiGPU)
    result = unittest.TextTestRunner(verbosity=2 if rank == 0 else 0).run(suite)
    success = torch.tensor(
        [1 if result.wasSuccessful() else 0],
        dtype=torch.int32,
        device=device,
    )
    dist.all_reduce(success, op=dist.ReduceOp.MIN)
    if rank == 0:
        status = "PASS" if success.item() == 1 else "FAIL"
        sys.stdout.write(f"{status}: MoESO2Convolution multi-GPU tests\n")
    teardown_dist()
    return success.item() == 1


if __name__ == "__main__":
    raise SystemExit(0 if run_tests() else 1)
