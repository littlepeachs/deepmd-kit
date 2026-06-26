# SPDX-License-Identifier: LGPL-3.0-or-later
"""Lightweight graph-parallel context for PyTorch mixed-batch training.

The first mixed-batch GP implementation uses the same rank set as MoE
expert parallelism, but stores a separate process group and context here so
the old EP path stays unchanged when ``training.graph_parallel`` is false.
"""

from __future__ import annotations

import logging
import os
import time
from typing import (
    TYPE_CHECKING,
    Any,
)

import torch
import torch.distributed as dist
from torch.autograd import Function

from deepmd.pt.utils.collective_order import (
    chain_collective_input,
    register_collective_output,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

_ENABLED = False
_GP_GROUP: Any | None = None
_GP_RANK = 0
_GP_WORLD_SIZE = 1
_REDUCE_BACKWARD = True
_PROFILE_GP_COLLECTIVE = bool(int(os.environ.get("DEEPMD_GP_COLLECTIVE_PROFILE", "0")))
_GP_COLLECTIVE_COUNTER = 0
log = logging.getLogger(__name__)


def _gp_collective_profile(label: str, tensor: torch.Tensor) -> None:
    if not _PROFILE_GP_COLLECTIVE:
        return
    global _GP_COLLECTIVE_COUNTER
    _GP_COLLECTIVE_COUNTER += 1
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    log.info(
        "GP_COLLECTIVE_PROFILE idx=%s t=%.6f rank=%s label=%s shape=%s",
        _GP_COLLECTIVE_COUNTER,
        time.time(),
        rank,
        label,
        tuple(tensor.shape),
    )


def set_graph_parallel_context(
    enabled: bool,
    group: Any | None,
    rank: int,
    world_size: int,
    reduce_backward: bool = True,
) -> None:
    """Set graph-parallel context for the current process."""
    global _ENABLED, _GP_GROUP, _GP_RANK, _GP_WORLD_SIZE, _REDUCE_BACKWARD
    _ENABLED = bool(enabled)
    _GP_GROUP = group
    _GP_RANK = int(rank)
    _GP_WORLD_SIZE = int(world_size)
    _REDUCE_BACKWARD = bool(reduce_backward)


def clear_graph_parallel_context() -> None:
    """Disable graph parallelism in the current process."""
    set_graph_parallel_context(False, None, 0, 1, True)


def graph_parallel_enabled() -> bool:
    """Return whether graph parallelism is active for the current process."""
    return _ENABLED and _GP_WORLD_SIZE > 1


def get_gp_group() -> Any | None:
    """Return the process group used for graph parallel collectives."""
    return _GP_GROUP


def get_gp_rank() -> int:
    """Return this rank's index inside the graph-parallel group."""
    return _GP_RANK


def get_gp_world_size() -> int:
    """Return the graph-parallel group size."""
    return _GP_WORLD_SIZE


def graph_reduce_backward_enabled() -> bool:
    """Return whether GP output reductions should all-reduce in backward."""
    return _REDUCE_BACKWARD


def get_gp_root_global_rank() -> int:
    """Return the global rank corresponding to GP rank 0.

    ``init_ep_dp_groups`` builds contiguous EP groups, and the initial GP path
    mirrors that rank set.  Therefore the group root is the current global rank
    minus the local GP rank.  In the required first mode ``gp_size ==
    world_size``, this is simply rank 0.
    """
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank() - _GP_RANK


def _balanced_partition_sizes(total: int, parts: int) -> list[int]:
    """Split ``total`` items into ``parts`` balanced contiguous chunks."""
    if parts <= 0:
        raise ValueError("parts must be positive")
    if total == 0:
        return [0] * parts
    base = total // parts
    remainder = total % parts
    return [base + (1 if idx < remainder else 0) for idx in range(parts)]


def _build_partition_offsets(
    sizes: Iterable[int],
    device: torch.device | str,
) -> torch.Tensor:
    """Build start offsets for a sequence of partition sizes."""
    offsets = [0]
    running = 0
    sizes_list = list(sizes)
    for size in sizes_list[:-1]:
        running += int(size)
        offsets.append(running)
    return torch.tensor(offsets, device=device, dtype=torch.long)


def _pad_to_size(tensor: torch.Tensor, dim: int, target_size: int) -> torch.Tensor:
    curr_size = tensor.size(dim)
    if curr_size == target_size:
        return tensor
    pad_shape = list(tensor.shape)
    pad_shape[dim] = target_size - curr_size
    pad_tensor = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, pad_tensor], dim=dim)


class _GraphAllGather(Function):
    """All-gather with recursive backward for higher-order force gradients."""

    @staticmethod
    def forward(
        ctx: Any,
        tensor: torch.Tensor,
        rank: int,
        world_size: int,
        group: Any,
        dim: int,
        label: str,
    ) -> torch.Tensor:
        ctx.rank = rank
        ctx.world_size = world_size
        ctx.group = group
        ctx.dim = dim
        ctx.label = label
        gathered = [torch.empty_like(tensor) for _ in range(world_size)]
        _gp_collective_profile(f"{label}:enter", tensor)
        dist.all_gather(gathered, tensor.contiguous(), group=group)
        out = torch.cat(gathered, dim=dim).contiguous()
        _gp_collective_profile(f"{label}:exit", out)
        return out

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[Any, ...]:
        ordered_grad_output = chain_collective_input(grad_output.contiguous())
        grad_input = _GraphReduceScatter.apply(
            ordered_grad_output,
            ctx.rank,
            ctx.world_size,
            ctx.group,
            ctx.dim,
            f"backward_of:{ctx.label}",
        )
        grad_input = register_collective_output(grad_input)
        return grad_input, None, None, None, None, None


class _GraphAllGatherForwardOnly(Function):
    """All-gather in forward while slicing the local shard in backward."""

    @staticmethod
    def forward(
        ctx: Any,
        tensor: torch.Tensor,
        rank: int,
        world_size: int,
        group: Any,
        dim: int,
        label: str,
    ) -> torch.Tensor:
        ctx.rank = rank
        ctx.world_size = world_size
        ctx.dim = dim
        ctx.local_size = tensor.size(dim)
        ctx.label = label
        gathered = [torch.empty_like(tensor) for _ in range(world_size)]
        _gp_collective_profile(f"{label}:enter", tensor)
        dist.all_gather(gathered, tensor.contiguous(), group=group)
        out = torch.cat(gathered, dim=dim).contiguous()
        _gp_collective_profile(f"{label}:exit", out)
        return out

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[Any, ...]:
        start = ctx.rank * ctx.local_size
        grad_input = grad_output.narrow(ctx.dim, start, ctx.local_size).contiguous()
        _gp_collective_profile(
            f"backward_local_of:{ctx.label}",
            grad_input,
        )
        return grad_input, None, None, None, None, None


class _GraphReduceScatter(Function):
    """Sum reduce-scatter with recursive all-gather backward."""

    @staticmethod
    def forward(
        ctx: Any,
        tensor: torch.Tensor,
        rank: int,
        world_size: int,
        group: Any,
        dim: int,
        label: str,
    ) -> torch.Tensor:
        if dim != 0:
            raise NotImplementedError("Graph reduce-scatter currently supports dim=0.")
        ctx.rank = rank
        ctx.world_size = world_size
        ctx.group = group
        ctx.dim = dim
        ctx.label = label
        if tensor.shape[dim] % world_size != 0:
            raise RuntimeError(
                "Graph reduce-scatter input first dimension must be divisible "
                f"by world_size, got shape={tuple(tensor.shape)}, "
                f"world_size={world_size}."
            )
        out_shape = list(tensor.shape)
        out_shape[dim] //= world_size
        out = torch.empty(out_shape, dtype=tensor.dtype, device=tensor.device)
        _gp_collective_profile(f"{label}:enter", tensor)
        dist.reduce_scatter_tensor(
            out,
            tensor.contiguous(),
            op=dist.ReduceOp.SUM,
            group=group,
        )
        _gp_collective_profile(f"{label}:exit", out)
        return out

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[Any, ...]:
        ordered_grad_output = chain_collective_input(grad_output.contiguous())
        grad_input = _GraphAllGather.apply(
            ordered_grad_output,
            ctx.rank,
            ctx.world_size,
            ctx.group,
            ctx.dim,
            f"backward_of:{ctx.label}",
        )
        grad_input = register_collective_output(grad_input)
        return grad_input, None, None, None, None, None


class _GraphAllReduce(Function):
    """Sum all-reduce with recursive backward."""

    @staticmethod
    def forward(
        ctx: Any,
        tensor: torch.Tensor,
        group: Any,
        label: str,
    ) -> torch.Tensor:
        ctx.group = group
        ctx.label = label
        out = tensor.clone()
        _gp_collective_profile(f"{label}:enter", tensor)
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=group)
        _gp_collective_profile(f"{label}:exit", out)
        return out

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[Any, ...]:
        ordered_grad_output = chain_collective_input(grad_output.contiguous())
        grad_input = _GraphAllReduce.apply(
            ordered_grad_output,
            ctx.group,
            f"backward_of:{ctx.label}",
        )
        grad_input = register_collective_output(grad_input)
        return grad_input, None, None


class _GraphAllReduceForwardOnly(Function):
    """Sum all-reduce in forward while leaving backward local.

    GP+MoE force/virial backward already synchronizes replicated parameters
    explicitly after ``loss.backward``.  Avoiding a backward all-reduce here
    prevents output-level GP reductions from racing with MoE second-order A2A
    collectives on the same rank set.
    """

    @staticmethod
    def forward(
        ctx: Any,
        tensor: torch.Tensor,
        group: Any,
        label: str,
    ) -> torch.Tensor:
        ctx.label = label
        out = tensor.clone()
        _gp_collective_profile(f"{label}:enter", tensor)
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=group)
        _gp_collective_profile(f"{label}:exit", out)
        return out

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[Any, ...]:
        _gp_collective_profile(
            f"backward_local_of:{ctx.label}",
            grad_output,
        )
        return grad_output.contiguous(), None, None


def gather_node_tensor(
    tensor: torch.Tensor,
    dim: int = 0,
    *,
    backward: bool = True,
) -> torch.Tensor:
    """Gather GP node shards and concatenate them along ``dim``.

    This mirrors the arch GP helper for the first mixed-batch GP path.  The
    distributed autograd functional preserves gradients from the gathered full
    tensor back to the local shard.
    """
    if not graph_parallel_enabled():
        return tensor
    if not dist.is_available() or not dist.is_initialized():
        return tensor

    group = get_gp_group()
    world_size = get_gp_world_size()
    local_size = tensor.size(dim)
    local_size_tensor = torch.tensor(
        [local_size],
        dtype=torch.long,
        device=tensor.device,
    )
    size_list = [torch.zeros_like(local_size_tensor) for _ in range(world_size)]
    dist.all_gather(size_list, local_size_tensor, group=group)
    sizes = [int(sz.item()) for sz in size_list]
    max_size = max(sizes)

    padded = _pad_to_size(tensor, dim=dim, target_size=max_size)
    ordered_padded = chain_collective_input(padded)
    if backward:
        gathered_tensor = _GraphAllGather.apply(
            ordered_padded,
            get_gp_rank(),
            world_size,
            group,
            dim,
            "gp_all_gather",
        )
    else:
        gathered_tensor = _GraphAllGatherForwardOnly.apply(
            ordered_padded,
            get_gp_rank(),
            world_size,
            group,
            dim,
            "gp_all_gather_forward_only",
        )
    gathered_tensor = register_collective_output(gathered_tensor)
    gathered = list(torch.chunk(gathered_tensor, world_size, dim=dim))

    trimmed = []
    for idx, part in enumerate(gathered):
        part_size = sizes[idx]
        if part_size == max_size:
            trimmed.append(part)
        else:
            trimmed.append(part.narrow(dim, 0, part_size))
    return torch.cat(trimmed, dim=dim).contiguous()


def reduce_graph_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce a tensor across the GP group with autograd support."""
    if not graph_parallel_enabled():
        return tensor
    if not dist.is_available() or not dist.is_initialized():
        return tensor
    if graph_reduce_backward_enabled():
        ordered_tensor = chain_collective_input(tensor)
        out = _GraphAllReduce.apply(ordered_tensor, get_gp_group(), "gp_all_reduce")
        return register_collective_output(out)
    ordered_tensor = chain_collective_input(tensor)
    out = _GraphAllReduceForwardOnly.apply(
        ordered_tensor,
        get_gp_group(),
        "gp_all_reduce_forward_only",
    )
    return register_collective_output(out)


def _tensor_signature(tensor: torch.Tensor) -> tuple[Any, ...]:
    """Build a cheap deterministic signature for consistency checks."""
    detached = tensor.detach().cpu()
    if detached.numel() == 0:
        return (tuple(detached.shape), str(detached.dtype), 0, 0.0, 0.0)
    values = detached.to(torch.float64)
    return (
        tuple(detached.shape),
        str(detached.dtype),
        int(detached.numel()),
        float(values.sum().item()),
        float(values.abs().sum().item()),
    )


def assert_batch_consistent(
    batch_data: dict[str, Any],
    keys: tuple[str, ...] = ("ptr", "batch", "coord"),
    group: Any | None = None,
) -> None:
    """Check that selected batch tensors are identical across GP ranks.

    This is intended as a first-step guard for the GP same-batch contract.  The
    object broadcast already enforces equality; the check catches wiring errors
    during integration without comparing the full tensors every step.
    """
    if not graph_parallel_enabled():
        return
    if not dist.is_available() or not dist.is_initialized():
        return
    signatures = {}
    for key in keys:
        value = batch_data.get(key)
        if isinstance(value, torch.Tensor):
            signatures[key] = _tensor_signature(value)

    world_size = dist.get_world_size(group=group)
    gathered: list[dict[str, tuple[Any, ...]] | None] = [None] * world_size
    dist.all_gather_object(gathered, signatures, group=group)
    reference = gathered[0]
    for rank, item in enumerate(gathered):
        if item != reference:
            raise RuntimeError(
                "Graph-parallel mixed batch is inconsistent across ranks: "
                f"rank0={reference}, rank{rank}={item}"
            )


def broadcast_batch_data(
    batch_data: dict[str, Any] | None,
    group: Any | None = None,
    src: int | None = None,
    validate: bool = False,
) -> dict[str, Any]:
    """Broadcast a Python batch dict from GP rank 0 to all GP ranks."""
    if not graph_parallel_enabled():
        if batch_data is None:
            raise RuntimeError("batch_data is None while graph parallel is disabled.")
        return batch_data
    if not dist.is_available() or not dist.is_initialized():
        if batch_data is None:
            raise RuntimeError("Distributed graph parallel requires a source batch.")
        return batch_data

    if group is None:
        group = get_gp_group()
    if src is None:
        src = get_gp_root_global_rank()

    objects: list[dict[str, Any] | None] = [
        batch_data if dist.get_rank() == src else None
    ]
    dist.broadcast_object_list(objects, src=src, group=group)
    result = objects[0]
    if result is None:
        raise RuntimeError("Graph-parallel batch broadcast returned None.")
    if validate:
        assert_batch_consistent(result, group=group)
    return result
