# SPDX-License-Identifier: LGPL-3.0-or-later
"""Checkpoint resharding helpers for SeZM MoE routing experts."""

from __future__ import (
    annotations,
)

from copy import (
    deepcopy,
)
from typing import (
    Any,
)

import torch
import torch.distributed as dist


def is_routing_expert_tensor_key(key: str) -> bool:
    """Return whether a state-dict tensor stores sharded routing experts."""
    return ".routing_matrix" in key or ".routing_bias" in key


def slice_state_dict_for_ep_load(
    state_dict: dict[str, Any],
    *,
    ep_rank: int,
    ep_size: int,
    n_routing_experts: int,
) -> dict[str, Any]:
    """Slice full routing expert tensors to the local EP shard."""
    if ep_size <= 1:
        return state_dict
    n_per_gpu = n_routing_experts // ep_size
    start = ep_rank * n_per_gpu
    end = start + n_per_gpu
    out: dict[str, Any] = {}
    for key, value in state_dict.items():
        if (
            is_routing_expert_tensor_key(key)
            and isinstance(value, torch.Tensor)
            and value.ndim > 0
            and value.shape[0] == n_routing_experts
        ):
            out[key] = value[start:end].clone()
        else:
            out[key] = value
    return out


def gather_state_dict_for_ep_save(
    state_dict: dict[str, Any],
    *,
    ep_group: object | None,
    ep_rank: int,
    ep_size: int,
    n_routing_experts: int,
) -> dict[str, Any]:
    """Gather local routing expert shards into full global tensors for saving."""
    if ep_group is None or ep_size <= 1 or not dist.is_initialized():
        return deepcopy(state_dict)

    n_per_gpu = n_routing_experts // ep_size
    out: dict[str, Any] = {}
    for key, value in state_dict.items():
        if (
            is_routing_expert_tensor_key(key)
            and isinstance(value, torch.Tensor)
            and value.ndim > 0
            and value.shape[0] == n_per_gpu
        ):
            gathered = [torch.empty_like(value) for _ in range(ep_size)]
            dist.all_gather(gathered, value.contiguous(), group=ep_group)
            out[key] = torch.cat(gathered, dim=0)
        elif isinstance(value, torch.Tensor):
            out[key] = value.detach().clone()
        else:
            out[key] = deepcopy(value)
    return out


__all__ = [
    "gather_state_dict_for_ep_save",
    "is_routing_expert_tensor_key",
    "slice_state_dict_for_ep_load",
]
