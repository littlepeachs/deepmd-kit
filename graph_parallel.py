from __future__ import annotations

from typing import Any, Dict, Iterable, List

import torch

# import core.distutils as distutils

# Mock State for Testing
_MOCK_RANK = 0
_MOCK_WORLD_SIZE = 4

def get_gp_rank():
    return _MOCK_RANK

def set_gp_rank(rank):
    global _MOCK_RANK
    _MOCK_RANK = rank

def get_gp_world_size():
    return _MOCK_WORLD_SIZE

def gather_node_tensor(tensor):
    # For single-card simulation with nlayers=1, we just return the local slice
    return tensor

_ENABLED = True

def set_graph_parallel_enabled(val: bool):
    global _ENABLED
    _ENABLED = val

def graph_parallel_enabled():
    return _ENABLED

def _balanced_partition_sizes(total: int, parts: int) -> List[int]:
    """Return a list of balanced chunk sizes that sum to ``total``."""
    if parts <= 0:
        raise ValueError("parts must be positive")
    if total == 0:
        return [0] * parts
    base = total // parts
    remainder = total % parts
    return [base + (1 if idx < remainder else 0) for idx in range(parts)]


def _build_partition_offsets(sizes: Iterable[int], device: torch.device) -> torch.Tensor:
    """Compute cumulative offsets from the balanced sizes."""
    offsets = [0]
    running = 0
    for size in sizes[:-1]:
        running += size
        offsets.append(running)
    return torch.tensor(offsets, device=device, dtype=torch.long)


def _slice_node_tensors(node_indices: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
    """Slice node-aligned tensors safely (handles empty partitions)."""
    if tensor is None:
        return None
    if node_indices.numel() == 0:
        # Preserve dtype/device while returning an empty view
        new_shape = list(tensor.shape)
        new_shape[0] = 0
        return tensor.new_empty(new_shape)
    return tensor.index_select(0, node_indices)


def _slice_attn_tensor(
    attn_tensor: torch.Tensor | None,
    node_offset: int,
    node_count: int,
    num_heads: int,
) -> torch.Tensor | None:
    """Slice attention tensors that are flattened along nodes * heads."""
    if attn_tensor is None:
        return None
    if node_count == 0:
        new_shape = list(attn_tensor.shape)
        new_shape[0] = 0
        return attn_tensor.new_empty(new_shape)
    start = node_offset * num_heads
    end = start + node_count * num_heads
    return attn_tensor[start:end]


def partition_batch_graph(batch_graph: Dict[str, Any]) -> Dict[str, Any]:
    """Split node-aligned tensors in ``batch_graph`` across the GP ranks.

    This function keeps graph-level tensors (e.g., volumes, atoms_per_graph) intact
    while slicing every node-level tensor so that each rank works on a contiguous
    range of atoms. Metadata describing the global layout is attached so later
    modules (interaction blocks, readout) can perform all-gather/reduce steps.
    """
    if "atomic_numbers" not in batch_graph:
        raise KeyError("batch_graph must contain 'atomic_numbers'")

    if (not distutils.initialized()) or distutils.get_gp_world_size() == 1:
        return batch_graph

    atomic_numbers = batch_graph["atomic_numbers"]
    device = atomic_numbers.device
    num_nodes = atomic_numbers.shape[0]
    world_size = distutils.get_gp_world_size()
    rank = distutils.get_gp_rank()

    if num_nodes == 0:
        return batch_graph

    partition_sizes = _balanced_partition_sizes(num_nodes, world_size) # 
    partition_offsets = _build_partition_offsets(partition_sizes, device=device)
    local_size = partition_sizes[rank]
    node_offset = int(partition_offsets[rank].item())
    node_indices = torch.arange(
        node_offset,
        node_offset + local_size,
        device=device,
        dtype=torch.long,
    ) # local rank拿到的node list[ 0 1  2 3 ...]

    # When `padding_atoms` is enabled, it is defined in global node space.
    # After slicing into GP ranks, each rank must use its *local* padded size
    # so attention-layer padding stays consistent with the sliced attn_mask.
    padding_atoms = batch_graph.get("padding_atoms")
    if isinstance(padding_atoms, int) and padding_atoms > 0:
        batch_graph["padding_atoms"] = int(local_size)
    
    # Save full node_batch before slicing (needed for GP energy reference)
    if "node_batch" in batch_graph:
        batch_graph["node_batch_full"] = batch_graph["node_batch"]
    if "atomic_numbers" in batch_graph:
        batch_graph["atomic_numbers_full"] = batch_graph["atomic_numbers"]
    if "batch_cart_coords" in batch_graph:
        batch_graph["batch_cart_coords_full"] = batch_graph["batch_cart_coords"]
    # NOTE: batch_strains is graph-level tensor [num_graph, 3, 3], not sliced, so no need to save full version
    
    # Slice node-aligned tensors.
    # NOTE: batch_strains is graph-level tensor [num_data, 3, 3], not node-level, so don't slice it
    node_keys = [
        "atomic_numbers", # 
        "edge_distance_basis",
        "edge_direction",
        "neighbor_list",
        "neighbor_mask",
        "node_batch",
        "node_padding_mask",
        # "batch_cart_coords",  # 这个不需要切分，最后只有反向求导需要用，后面的模型计算不需要了
        "smooth_two_body",
        "smooth_tri_body",
        "three_body_basis",
    ]
    import pdb; pdb.set_trace()
    for key in node_keys:
        tensor = batch_graph.get(key)
        if tensor is None:
            continue
        batch_graph[key] = _slice_node_tensors(node_indices, tensor)

    # Handle attention-specific tensors that flatten the node dimension.
    attn_mask = batch_graph.get("attn_mask")
    if attn_mask is not None:
        num_heads = attn_mask.shape[0] // num_nodes
        batch_graph["attn_mask"] = _slice_attn_tensor(
            attn_mask, node_offset, local_size, num_heads
        )
    import pdb; pdb.set_trace()
    angle_embedding = batch_graph.get("angle_embedding")
    if angle_embedding is not None:
        num_heads = angle_embedding.shape[0] // num_nodes
        batch_graph["angle_embedding"] = _slice_attn_tensor(
            angle_embedding, node_offset, local_size, num_heads
        )
    
    return batch_graph


def graph_parallel_enabled() -> bool:
    """Return True when graph parallel is active."""
    if not distutils.initialized():
        return False
    return distutils.get_gp_world_size() > 1


def gather_node_tensor(
    tensor: torch.Tensor, dim: int = 0
) -> torch.Tensor:
    """All-gather a node-aligned tensor across the GP ranks."""
    if not graph_parallel_enabled():
        return tensor
    # 使用带梯度求和的 gather，这样所有 rank 在后向时的梯度会先 all-reduce 再 split，
    # 保证参数梯度一致（等价于对完整图的计算）
    gathered = distutils.gather_from_model_parallel_region_sum_grad(tensor, dim=dim)
    return gathered


def reduce_graph_tensor(
    tensor: torch.Tensor
) -> torch.Tensor:
    """Sum a tensor across the GP ranks."""
    if not graph_parallel_enabled():
        return tensor
    return distutils.reduce_from_model_parallel_region(tensor)
