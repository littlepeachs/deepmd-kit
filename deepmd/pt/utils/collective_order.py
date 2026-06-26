# SPDX-License-Identifier: LGPL-3.0-or-later
"""Autograd ordering helpers for mixed GP/MoE collectives.

Collective autograd Functions that share the same ranks must be traversed in
the same order on every rank.  GP+MoE force/virial training creates multiple
second-order collective branches, so this module provides an opt-in chain that
serializes those branches with zero-valued tensor dependencies.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    import torch

_STATE: contextvars.ContextVar[tuple[int, torch.Tensor | None]] = (
    contextvars.ContextVar("deepmd_collective_order_state", default=(0, None))
)


def _is_differentiable_tensor(tensor: torch.Tensor) -> bool:
    return tensor.is_floating_point() or tensor.is_complex()


@contextmanager
def collective_ordering() -> Iterator[None]:
    """Enable a per-context zero-dependency chain for collective nodes."""
    depth, _ = _STATE.get()
    token = _STATE.set((depth + 1, None))
    try:
        yield
    finally:
        _STATE.reset(token)


def chain_collective_input(tensor: torch.Tensor) -> torch.Tensor:
    """Attach the previous collective output as a zero dependency."""
    depth, dep = _STATE.get()
    if depth <= 0 or not _is_differentiable_tensor(tensor):
        return tensor
    if dep is None or not _is_differentiable_tensor(dep):
        return tensor
    return tensor + dep.sum() * 0


def register_collective_output(tensor: torch.Tensor) -> torch.Tensor:
    """Register the current collective output as the next chain dependency."""
    depth, _ = _STATE.get()
    if depth > 0 and _is_differentiable_tensor(tensor):
        _STATE.set((depth, tensor))
    return tensor
