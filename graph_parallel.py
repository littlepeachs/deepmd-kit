from __future__ import annotations

from collections.abc import Iterable
from typing import List

import torch
import torch.distributed as dist
from torch.distributed.nn import functional as dist_fn

try:
    from deepmd.utils.local_distutils import get_repo_distutils

    _gp_distutils = get_repo_distutils()
except Exception:
    _gp_distutils = None

_MOCK_RANK = 0
_MOCK_WORLD_SIZE = 4
_ENABLED = True


def set_graph_parallel_enabled(val: bool) -> None:
    global _ENABLED
    _ENABLED = val


def set_gp_rank(rank: int) -> None:
    global _MOCK_RANK
    _MOCK_RANK = rank


def set_gp_world_size(world_size: int) -> None:
    global _MOCK_WORLD_SIZE
    _MOCK_WORLD_SIZE = world_size


def _dist_initialized() -> bool:
    if _distutils_gp_ready():
        return True
    return dist.is_available() and dist.is_initialized()


def _distutils_gp_ready() -> bool:
    if _gp_distutils is None:
        return False
    if not hasattr(_gp_distutils, "initialized"):
        return False
    if not _gp_distutils.initialized():
        return False
    if not hasattr(_gp_distutils, "get_gp_world_size"):
        return False
    try:
        _ = _gp_distutils.get_gp_world_size()
        return True
    except Exception:
        return False


def graph_parallel_enabled() -> bool:
    if not _ENABLED:
        return False
    if _distutils_gp_ready():
        return _gp_distutils.get_gp_world_size() > 1
    if _dist_initialized() and dist.is_available() and dist.is_initialized():
        return dist.get_world_size() > 1
    return _MOCK_WORLD_SIZE > 1


def get_gp_rank() -> int:
    if _distutils_gp_ready() and hasattr(_gp_distutils, "get_gp_rank"):
        return _gp_distutils.get_gp_rank()
    if _dist_initialized() and dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return _MOCK_RANK


def get_gp_world_size() -> int:
    if _distutils_gp_ready() and hasattr(_gp_distutils, "get_gp_world_size"):
        return _gp_distutils.get_gp_world_size()
    if _dist_initialized() and dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return _MOCK_WORLD_SIZE


def _balanced_partition_sizes(total: int, parts: int) -> List[int]:
    if parts <= 0:
        raise ValueError("parts must be positive")
    if total == 0:
        return [0] * parts
    base = total // parts
    remainder = total % parts
    return [base + (1 if idx < remainder else 0) for idx in range(parts)]


def _build_partition_offsets(sizes: Iterable[int], device: torch.device) -> torch.Tensor:
    offsets = [0]
    running = 0
    sizes_list = list(sizes)
    for size in sizes_list[:-1]:
        running += size
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


def gather_node_tensor(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
    if not graph_parallel_enabled():
        return tensor
    if _distutils_gp_ready() and hasattr(
        _gp_distutils, "gather_from_model_parallel_region_sum_grad"
    ):
        return _gp_distutils.gather_from_model_parallel_region_sum_grad(tensor, dim=dim)
    if not _dist_initialized() or not dist.is_available() or not dist.is_initialized():
        return tensor

    world_size = dist.get_world_size()
    local_size = tensor.size(dim)
    local_size_tensor = torch.tensor([local_size], dtype=torch.long, device=tensor.device)
    size_list = [torch.zeros_like(local_size_tensor) for _ in range(world_size)]
    dist.all_gather(size_list, local_size_tensor)
    sizes = [int(sz.item()) for sz in size_list]
    max_size = max(sizes)

    padded = _pad_to_size(tensor, dim=dim, target_size=max_size)
    gathered = dist_fn.all_gather(padded)

    trimmed = []
    for idx, part in enumerate(gathered):
        part_size = sizes[idx]
        if part_size == max_size:
            trimmed.append(part)
        else:
            trimmed.append(part.narrow(dim, 0, part_size))
    return torch.cat(trimmed, dim=dim).contiguous()


def gather_node_tensor_no_sum_grad(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
    if not graph_parallel_enabled():
        return tensor
    if _distutils_gp_ready() and hasattr(
        _gp_distutils, "gather_from_model_parallel_region"
    ):
        return _gp_distutils.gather_from_model_parallel_region(tensor, dim=dim)
    if not _dist_initialized() or not dist.is_available() or not dist.is_initialized():
        return tensor

    world_size = dist.get_world_size()
    local_size = tensor.size(dim)
    local_size_tensor = torch.tensor([local_size], dtype=torch.long, device=tensor.device)
    size_list = [torch.zeros_like(local_size_tensor) for _ in range(world_size)]
    dist.all_gather(size_list, local_size_tensor)
    sizes = [int(sz.item()) for sz in size_list]
    max_size = max(sizes)

    padded = _pad_to_size(tensor, dim=dim, target_size=max_size)
    gathered = dist_fn.all_gather(padded)

    trimmed = []
    for idx, part in enumerate(gathered):
        part_size = sizes[idx]
        if part_size == max_size:
            trimmed.append(part)
        else:
            trimmed.append(part.narrow(dim, 0, part_size))
    return torch.cat(trimmed, dim=dim).contiguous()


def reduce_graph_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if not graph_parallel_enabled():
        return tensor
    if _distutils_gp_ready() and hasattr(
        _gp_distutils, "reduce_from_model_parallel_region"
    ):
        return _gp_distutils.reduce_from_model_parallel_region(tensor)
    if not _dist_initialized() or not dist.is_available() or not dist.is_initialized():
        return tensor
    return dist_fn.all_reduce(tensor)
