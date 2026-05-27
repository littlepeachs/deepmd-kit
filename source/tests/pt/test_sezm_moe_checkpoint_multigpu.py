# SPDX-License-Identifier: LGPL-3.0-or-later
"""Multi-GPU tests for SeZM MoE checkpoint routing expert gather."""

from __future__ import (
    annotations,
)

import signal
import sys
import unittest

import torch
import torch.distributed as dist

from deepmd.pt.utils.sezm_moe_checkpoint import (
    gather_state_dict_for_ep_save,
    slice_state_dict_for_ep_load,
)


def setup_dist() -> tuple[int, int, torch.device]:
    """Initialize distributed state."""
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
    """Fail on collectives that hang."""
    raise TimeoutError(f"rank {dist.get_rank()} timed out")


class TestSeZMMoECheckpointMultiGPU(unittest.TestCase):
    """Checkpoint routing shard gather tests."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.rank, cls.world, cls.device = setup_dist()

    @classmethod
    def tearDownClass(cls) -> None:
        """Keep process group alive until result aggregation."""

    def setUp(self) -> None:
        signal.signal(signal.SIGALRM, deadlock_alarm)
        signal.alarm(120)

    def tearDown(self) -> None:
        signal.alarm(0)

    def test_4gpu_shards_gather_to_full_and_reslice(self) -> None:
        if self.world != 4:
            self.skipTest("checkpoint gather test runs with 4 GPUs")
        local = torch.full(
            (2, 3),
            float(self.rank + 1),
            dtype=torch.float64,
            device=self.device,
        )
        state = {
            "model.experts.routing_matrix_m0": local,
            "model.experts.shared_matrix_m0": torch.ones(
                2, 3, dtype=torch.float64, device=self.device
            ),
        }

        gathered = gather_state_dict_for_ep_save(
            state,
            ep_group=dist.group.WORLD,
            ep_rank=self.rank,
            ep_size=4,
            n_routing_experts=8,
        )
        expected = torch.cat(
            [
                torch.full(
                    (2, 3), float(rank + 1), dtype=torch.float64, device=self.device
                )
                for rank in range(4)
            ],
            dim=0,
        )
        torch.testing.assert_close(
            gathered["model.experts.routing_matrix_m0"], expected
        )

        resliced = slice_state_dict_for_ep_load(
            gathered,
            ep_rank=self.rank,
            ep_size=4,
            n_routing_experts=8,
        )
        torch.testing.assert_close(resliced["model.experts.routing_matrix_m0"], local)


def run_tests() -> bool:
    """Run unittest suite and aggregate success."""
    rank, _, device = setup_dist()
    if rank == 0:
        sys.stdout.write("Running SeZM MoE checkpoint multi-GPU tests\n")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSeZMMoECheckpointMultiGPU)
    result = unittest.TextTestRunner(verbosity=2 if rank == 0 else 0).run(suite)
    success = torch.tensor(
        [1 if result.wasSuccessful() else 0],
        dtype=torch.int32,
        device=device,
    )
    dist.all_reduce(success, op=dist.ReduceOp.MIN)
    if rank == 0:
        status = "PASS" if success.item() == 1 else "FAIL"
        sys.stdout.write(f"{status}: SeZM MoE checkpoint multi-GPU tests\n")
    teardown_dist()
    return success.item() == 1


if __name__ == "__main__":
    raise SystemExit(0 if run_tests() else 1)
