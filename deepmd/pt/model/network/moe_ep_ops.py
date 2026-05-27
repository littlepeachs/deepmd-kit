# SPDX-License-Identifier: LGPL-3.0-or-later
"""Differentiable All-to-All communication operators for MoE Expert Parallelism.

Provides `_AllToAllDouble`, a recursive autograd Function whose backward
calls `.apply()` again, creating a fresh autograd node so that
`create_graph=True` (required for force -> virial second derivatives)
works correctly to arbitrary order.

Public API
----------
all_to_all_differentiable(x, send_splits, recv_splits, group)
    When *group* is ``None`` (single-GPU / no EP), returns *x* unchanged.
    Otherwise dispatches through ``_AllToAllDouble``.
"""

from __future__ import annotations

import os
import time
import logging
import contextvars
from typing import (
    Any,
)

import torch
import torch.distributed as dist
from torch.autograd import (
    Function,
)

from deepmd.pt.utils.collective_order import (
    chain_collective_input,
    register_collective_output,
)

_PROFILE_A2A = bool(int(os.environ.get("DEEPMD_MOE_A2A_PROFILE", "0")))
_A2A_COUNTER = 0
log = logging.getLogger(__name__)
_HIGHER_ORDER_DEPS: contextvars.ContextVar[list[torch.Tensor] | None] = (
    contextvars.ContextVar("moe_a2a_higher_order_deps", default=None)
)


def _a2a_profile(
    label: str, x: torch.Tensor, send_splits: list[int], recv_splits: list[int]
) -> None:
    if not _PROFILE_A2A:
        return
    global _A2A_COUNTER
    _A2A_COUNTER += 1
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    rank = dist.get_rank() if dist.is_initialized() else 0
    log.info(
        "MOE_A2A_PROFILE idx=%s t=%.6f rank=%s label=%s shape=%s send=%s recv=%s",
        _A2A_COUNTER,
        time.time(),
        rank,
        label,
        tuple(x.shape),
        send_splits,
        recv_splits,
    )


def _a2a_raw(
    x: torch.Tensor,
    send_splits: list[int],
    recv_splits: list[int],
    group: dist.ProcessGroup,
    label: str,
) -> torch.Tensor:
    """Raw All-to-All without autograd.

    Parameters
    ----------
    x : Tensor
        Input tensor whose first dimension equals ``sum(send_splits)``.
    send_splits : list[int]
        Number of rows to send to each rank.
    recv_splits : list[int]
        Number of rows to receive from each rank.
    group : ProcessGroup
        The communication group.

    Returns
    -------
    Tensor
        Output tensor with first dimension ``sum(recv_splits)``.
    """
    total_recv = sum(recv_splits)
    out = torch.empty((total_recv, *x.shape[1:]), dtype=x.dtype, device=x.device)
    _a2a_profile(f"{label}:enter", x, send_splits, recv_splits)
    dist.all_to_all_single(
        out,
        x.contiguous(),
        output_split_sizes=recv_splits,
        input_split_sizes=send_splits,
        group=group,
    )
    _a2a_profile(f"{label}:exit", out, recv_splits, send_splits)
    return out


class _AllToAllDouble(Function):
    """Recursively differentiable All-to-All.

    The backward pass calls ``.apply()`` with swapped send/recv splits,
    which creates a *new* autograd node.  This means the graph built by
    ``create_graph=True`` (1st backward) can itself be differentiated
    (2nd backward), giving correct second-order derivatives through
    the communication boundary.

    Callers must keep the same A2A participation order on every EP rank.
    Graph-parallel force/virial paths may add higher-order dependencies
    explicitly so autograd does not prune a collective on only some ranks.
    """

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        send_splits: list[int],
        recv_splits: list[int],
        group: dist.ProcessGroup,
        label: str,
    ) -> torch.Tensor:
        ctx.group = group
        ctx.send_splits = send_splits
        ctx.recv_splits = recv_splits
        ctx.label = label
        return _a2a_raw(x, send_splits, recv_splits, group, label)

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None, None]:
        # Recursive call: backward of this node is itself an A2A with
        # swapped splits.  Because we call .apply(), a new autograd node
        # is inserted into the graph, enabling higher-order derivatives.
        ordered_grad_output = chain_collective_input(grad_output)
        grad_input = _AllToAllDouble.apply(
            ordered_grad_output,
            ctx.recv_splits,
            ctx.send_splits,
            ctx.group,
            f"backward_of:{ctx.label}",
        )
        grad_input = register_collective_output(grad_input)
        deps = _HIGHER_ORDER_DEPS.get()
        if deps is not None and torch.is_grad_enabled():
            deps.append(grad_input)
        return grad_input, None, None, None, None


def all_to_all_differentiable(
    x: torch.Tensor,
    send_splits: list[int],
    recv_splits: list[int],
    group: dist.ProcessGroup | None,
    label: str = "a2a",
) -> torch.Tensor:
    """Public API for differentiable All-to-All.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    send_splits : list[int]
        Number of rows to send to each rank.
    recv_splits : list[int]
        Number of rows to receive from each rank.
    group : ProcessGroup or None
        Communication group.  When ``None`` (single-GPU / no EP),
        *x* is returned unchanged with gradients flowing through.

    Returns
    -------
    Tensor
        Result of All-to-All, or *x* itself when ``group is None``.
    """
    if group is None:
        return x
    total_recv = sum(recv_splits)
    local_rows = sum(send_splits) + total_recv
    has_rows = torch.tensor([int(local_rows > 0)], dtype=torch.int32, device=x.device)
    dist.all_reduce(has_rows, op=dist.ReduceOp.MAX, group=group)
    if has_rows.item() == 0:
        out = x.new_zeros((total_recv, *x.shape[1:]))
        if x.is_floating_point() or x.is_complex():
            out = out + x.sum() * 0
        return out
    ordered_x = chain_collective_input(x)
    out = _AllToAllDouble.apply(ordered_x, send_splits, recv_splits, group, label)
    return register_collective_output(out)


def begin_higher_order_dependency_capture() -> contextvars.Token:
    """Start collecting recursive A2A outputs built during create_graph backward."""
    return _HIGHER_ORDER_DEPS.set([])


def end_higher_order_dependency_capture(
    token: contextvars.Token,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Stop collection and return a zero-valued dependency scalar."""
    deps = _HIGHER_ORDER_DEPS.get()
    _HIGHER_ORDER_DEPS.reset(token)
    dep = reference.sum() * 0
    if deps is None:
        return dep
    for tensor in deps:
        if tensor.is_floating_point() or tensor.is_complex():
            dep = dep + tensor.sum() * 0
    return dep
