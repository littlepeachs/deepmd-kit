# SPDX-License-Identifier: LGPL-3.0-or-later
"""SeZM MoE Expert Parallelism + Data Parallelism utilities."""

from __future__ import (
    annotations,
)

import torch
import torch.distributed as dist


def init_ep_dp_groups(
    ep_size: int = 1,
) -> tuple[object | None, object | None, int, int, int, int]:
    """Initialize EP and DP process groups from the flat distributed world."""
    if ep_size <= 1 or not dist.is_initialized():
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        return (None, None, 0, 1, rank, world_size)

    world_size = dist.get_world_size()
    world_rank = dist.get_rank()
    if world_size % ep_size != 0:
        raise ValueError(
            f"world_size ({world_size}) must be divisible by ep_size ({ep_size})"
        )

    dp_size = world_size // ep_size

    my_ep_group = None
    for dp_idx in range(dp_size):
        ranks = [dp_idx * ep_size + ep_idx for ep_idx in range(ep_size)]
        group = dist.new_group(ranks)
        if world_rank in ranks:
            my_ep_group = group

    my_dp_group = None
    for ep_idx in range(ep_size):
        ranks = [dp_idx * ep_size + ep_idx for dp_idx in range(dp_size)]
        group = dist.new_group(ranks)
        if world_rank in ranks:
            my_dp_group = group

    ep_rank = world_rank % ep_size
    dp_rank = world_rank // ep_size
    return (my_ep_group, my_dp_group, ep_rank, ep_size, dp_rank, dp_size)


def _is_routing_expert_param(name: str) -> bool:
    """Return whether a parameter belongs to a sharded routing expert."""
    return ".routing_matrix" in name or ".routing_bias" in name


def sync_moe_gradients(
    model: torch.nn.Module,
    dp_group: object | None,
    world_group: object | None,
    dp_size: int,
    world_size: int,
) -> None:
    """Synchronize SeZM MoE gradients with the correct group and divisor."""
    if world_size == 1:
        return

    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if _is_routing_expert_param(name):
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=dp_group)
        else:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=world_group)
        param.grad.div_(world_size)


__all__ = ["_is_routing_expert_param", "init_ep_dp_groups", "sync_moe_gradients"]
