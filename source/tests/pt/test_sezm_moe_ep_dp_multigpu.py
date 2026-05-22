# SPDX-License-Identifier: LGPL-3.0-or-later
"""Multi-GPU tests for SeZM MoE EP/DP gradient synchronization."""

from __future__ import (
    annotations,
)

import signal
import sys
import unittest

import torch
import torch.distributed as dist

from deepmd.pt.utils.sezm_moe_ep_dp import (
    init_ep_dp_groups,
    sync_moe_gradients,
)


class TinyGradModel(torch.nn.Module):
    """Tiny model with routing, shared, router, and non-MoE parameters."""

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.routing_stack = torch.nn.Module()
        self.routing_stack.routing_matrix_m0 = torch.nn.Parameter(
            torch.ones(2, 3, dtype=torch.float64, device=device)
        )
        self.routing_stack.routing_bias = torch.nn.Parameter(
            torch.ones(2, dtype=torch.float64, device=device)
        )
        self.shared_stack = torch.nn.Module()
        self.shared_stack.shared_matrix_m0 = torch.nn.Parameter(
            torch.ones(2, 3, dtype=torch.float64, device=device)
        )
        self.router = torch.nn.Module()
        self.router.gate_matrix = torch.nn.Parameter(
            torch.ones(2, 3, dtype=torch.float64, device=device)
        )
        self.other_param = torch.nn.Parameter(
            torch.ones(2, 3, dtype=torch.float64, device=device)
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
    """Destroy process group."""
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def deadlock_alarm(signum, frame) -> None:
    """Raise when a collective hangs."""
    raise TimeoutError(f"rank {dist.get_rank()} timed out")


def assign_known_grads(model: TinyGradModel, rank: int) -> dict[str, torch.Tensor]:
    """Assign deterministic per-rank gradients and return pre-sync copies."""
    pre: dict[str, torch.Tensor] = {}
    for idx, (name, param) in enumerate(model.named_parameters(), start=1):
        grad = torch.full_like(param, float((rank + 1) * idx))
        param.grad = grad
        pre[name] = grad.detach().clone()
    return pre


def expected_routing_grad(
    pre_grad: torch.Tensor,
    dp_group: object | None,
    world_size: int,
) -> torch.Tensor:
    """Compute expected routing gradient after DP-group sync."""
    expected = pre_grad.detach().clone()
    dist.all_reduce(expected, op=dist.ReduceOp.SUM, group=dp_group)
    expected.div_(world_size)
    return expected


def expected_world_grad(pre_grad: torch.Tensor, world_size: int) -> torch.Tensor:
    """Compute expected non-routing gradient after world sync."""
    expected = pre_grad.detach().clone()
    dist.all_reduce(expected, op=dist.ReduceOp.SUM)
    expected.div_(world_size)
    return expected


def assert_dp_group_consistent(tensor: torch.Tensor, group: object | None) -> None:
    """Assert all ranks in a DP group have the same tensor."""
    group_size = dist.get_world_size(group=group) if group is not None else 1
    gathered = [torch.empty_like(tensor) for _ in range(group_size)]
    if group is None:
        gathered[0].copy_(tensor)
    else:
        dist.all_gather(gathered, tensor, group=group)
    for other in gathered:
        torch.testing.assert_close(tensor, other, atol=0.0, rtol=0.0)


def assert_world_consistent(tensor: torch.Tensor) -> None:
    """Assert all ranks in the world have the same tensor."""
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor)
    for other in gathered:
        torch.testing.assert_close(tensor, other, atol=0.0, rtol=0.0)


class TestSeZMMoEEPDP(unittest.TestCase):
    """EP/DP group and gradient sync tests."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.rank, cls.world, cls.device = setup_dist()

    @classmethod
    def tearDownClass(cls) -> None:
        """Keep process group alive until run_tests aggregates results."""

    def setUp(self) -> None:
        signal.signal(signal.SIGALRM, deadlock_alarm)
        signal.alarm(120)

    def tearDown(self) -> None:
        signal.alarm(0)

    def test_init_ep_dp_groups_4gpu_ep2_dp2(self) -> None:
        if self.world != 4:
            self.skipTest("T1 runs with 4 GPUs")
        ep_group, dp_group, ep_rank, ep_size, dp_rank, dp_size = init_ep_dp_groups(2)

        self.assertEqual(ep_size, 2)
        self.assertEqual(dp_size, 2)
        self.assertEqual(ep_rank, self.rank % 2)
        self.assertEqual(dp_rank, self.rank // 2)
        self.assertEqual(dist.get_world_size(group=ep_group), 2)
        self.assertEqual(dist.get_world_size(group=dp_group), 2)
        self.assertEqual(dist.get_rank(group=ep_group), ep_rank)
        self.assertEqual(dist.get_rank(group=dp_group), dp_rank)

    def test_routing_expert_grad_dp_synced_4gpu(self) -> None:
        if self.world != 4:
            self.skipTest("T2 runs with 4 GPUs")
        _, dp_group, _, _, _, dp_size = init_ep_dp_groups(2)
        model = TinyGradModel(self.device)
        pre = assign_known_grads(model, self.rank)
        expected = expected_routing_grad(
            pre["routing_stack.routing_matrix_m0"], dp_group, self.world
        )

        sync_moe_gradients(model, dp_group, None, dp_size, self.world)

        torch.testing.assert_close(model.routing_stack.routing_matrix_m0.grad, expected)
        assert_dp_group_consistent(model.routing_stack.routing_matrix_m0.grad, dp_group)
        if self.rank == 0:
            sys.stdout.write(
                "T2 routing_matrix_grad="
                f"{model.routing_stack.routing_matrix_m0.grad.flatten()[0].item():.17e} "
                f"expected={expected.flatten()[0].item():.17e}\n"
            )

    def test_other_params_world_synced_4gpu(self) -> None:
        if self.world != 4:
            self.skipTest("T3 runs with 4 GPUs")
        _, dp_group, _, _, _, dp_size = init_ep_dp_groups(2)
        model = TinyGradModel(self.device)
        pre = assign_known_grads(model, self.rank)
        expected = expected_world_grad(pre["shared_stack.shared_matrix_m0"], self.world)

        sync_moe_gradients(model, dp_group, None, dp_size, self.world)

        torch.testing.assert_close(model.shared_stack.shared_matrix_m0.grad, expected)
        assert_world_consistent(model.shared_stack.shared_matrix_m0.grad)

    def test_pure_ep_dp_size_1(self) -> None:
        if self.world != 4:
            self.skipTest("T4 runs with 4 GPUs")
        _, dp_group, _, _, _, dp_size = init_ep_dp_groups(4)
        model = TinyGradModel(self.device)
        pre = assign_known_grads(model, self.rank)
        expected = pre["routing_stack.routing_matrix_m0"] / self.world

        sync_moe_gradients(model, dp_group, None, dp_size, self.world)

        torch.testing.assert_close(model.routing_stack.routing_matrix_m0.grad, expected)
        if self.rank == 0:
            sys.stdout.write(
                "T4 routing_matrix_grad="
                f"{model.routing_stack.routing_matrix_m0.grad.flatten()[0].item():.17e} "
                f"expected={expected.flatten()[0].item():.17e}\n"
            )

    def test_8gpu_ep4_dp2(self) -> None:
        if self.world != 8:
            self.skipTest("T5 runs with 8 GPUs")
        _, dp_group, _, _, _, dp_size = init_ep_dp_groups(4)
        model = TinyGradModel(self.device)
        pre = assign_known_grads(model, self.rank)
        expected_routing = expected_routing_grad(
            pre["routing_stack.routing_matrix_m0"], dp_group, self.world
        )
        expected_shared = expected_world_grad(
            pre["shared_stack.shared_matrix_m0"], self.world
        )

        sync_moe_gradients(model, dp_group, None, dp_size, self.world)

        torch.testing.assert_close(
            model.routing_stack.routing_matrix_m0.grad, expected_routing
        )
        torch.testing.assert_close(
            model.shared_stack.shared_matrix_m0.grad, expected_shared
        )
        assert_dp_group_consistent(model.routing_stack.routing_matrix_m0.grad, dp_group)
        assert_world_consistent(model.shared_stack.shared_matrix_m0.grad)


def run_tests() -> bool:
    """Run unittest suite and aggregate success across ranks."""
    rank, _, device = setup_dist()
    if rank == 0:
        sys.stdout.write("Running SeZM MoE EP/DP tests\n")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSeZMMoEEPDP)
    result = unittest.TextTestRunner(verbosity=2 if rank == 0 else 0).run(suite)
    success = torch.tensor(
        [1 if result.wasSuccessful() else 0],
        dtype=torch.int32,
        device=device,
    )
    dist.all_reduce(success, op=dist.ReduceOp.MIN)
    if rank == 0:
        status = "PASS" if success.item() == 1 else "FAIL"
        sys.stdout.write(f"{status}: SeZM MoE EP/DP tests\n")
    teardown_dist()
    return success.item() == 1


if __name__ == "__main__":
    raise SystemExit(0 if run_tests() else 1)
