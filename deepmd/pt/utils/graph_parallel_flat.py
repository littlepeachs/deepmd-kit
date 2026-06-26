# SPDX-License-Identifier: LGPL-3.0-or-later
"""Flat-graph partition helpers for mixed-batch graph parallelism."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from deepmd.pt.utils import graph_parallel


@dataclass(frozen=True)
class FlatGraphPartition:
    """The graph shard owned by one GP rank.

    The partition is defined over flattened local atom indices.  Edges and
    angles are assigned by their owner/center atom, i.e. row 0 of
    ``edge_index`` and ``angle_index``.
    """

    rank: int
    world_size: int
    local_start: int
    local_end: int
    local_size: int
    atom_index: torch.Tensor
    batch: torch.Tensor | None
    edge_mask: torch.Tensor
    edge_ids: torch.Tensor
    edge_index: torch.Tensor
    angle_mask: torch.Tensor
    angle_ids: torch.Tensor
    angle_index: torch.Tensor

    def asdict(self) -> dict[str, torch.Tensor | int | None]:
        """Return a dict representation convenient for later model plumbing."""
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "local_start": self.local_start,
            "local_end": self.local_end,
            "local_size": self.local_size,
            "atom_index": self.atom_index,
            "batch": self.batch,
            "edge_mask": self.edge_mask,
            "edge_ids": self.edge_ids,
            "edge_index": self.edge_index,
            "angle_mask": self.angle_mask,
            "angle_ids": self.angle_ids,
            "angle_index": self.angle_index,
        }


def build_flat_graph_partition(
    total_atoms: int,
    edge_index: torch.Tensor,
    angle_index: torch.Tensor,
    batch: torch.Tensor | None = None,
    rank: int | None = None,
    world_size: int | None = None,
) -> FlatGraphPartition:
    """Build one rank's contiguous center-atom partition for a flat graph.

    Parameters
    ----------
    total_atoms
        Number of flattened local atoms, normally ``int(ptr[-1])``.
    edge_index
        Graph edge indices with shape ``[2, nedge]``. Row 0 is the owner center
        atom and row 1 is the neighbor atom.
    angle_index
        Angle indices with shape ``[3, nangle]``. Row 0 is the owner center
        atom; rows 1 and 2 reference global edge ids and are remapped to local
        edge ids in the returned partition.
    batch
        Optional atom-to-frame map with shape ``[total_atoms]``.
    rank, world_size
        Optional override for tests.  Defaults to current GP context.
    """
    if rank is None:
        rank = graph_parallel.get_gp_rank()
    if world_size is None:
        world_size = graph_parallel.get_gp_world_size()
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank ({rank}) must be in [0, {world_size})")
    if total_atoms < 0:
        raise ValueError("total_atoms must be non-negative")
    if edge_index.dim() != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, nedge]")
    if angle_index.dim() != 2 or angle_index.shape[0] != 3:
        raise ValueError("angle_index must have shape [3, nangle]")
    if batch is not None and batch.numel() != total_atoms:
        raise ValueError(
            f"batch length ({batch.numel()}) must equal total_atoms ({total_atoms})"
        )

    device = edge_index.device
    sizes = graph_parallel._balanced_partition_sizes(total_atoms, world_size)
    offsets = graph_parallel._build_partition_offsets(sizes, device=device)
    local_start = int(offsets[rank].item())
    local_size = sizes[rank]
    local_end = local_start + local_size

    atom_index = torch.arange(local_start, local_end, device=device, dtype=torch.long)
    local_batch = batch[local_start:local_end] if batch is not None else None

    edge_mask = (edge_index[0] >= local_start) & (edge_index[0] < local_end)
    edge_ids = torch.nonzero(edge_mask, as_tuple=False).view(-1)
    local_edge_index = edge_index[:, edge_ids].clone()

    angle_owner_mask = (angle_index[0] >= local_start) & (angle_index[0] < local_end)
    angle_ids = torch.nonzero(angle_owner_mask, as_tuple=False).view(-1)
    local_angle_index = angle_index[:, angle_ids].clone()

    num_total_edges = edge_index.shape[1]
    edge_remap = torch.full(
        (num_total_edges,),
        -1,
        device=device,
        dtype=torch.long,
    )
    edge_remap[edge_mask] = torch.arange(
        local_edge_index.shape[1],
        device=device,
        dtype=torch.long,
    )

    if local_angle_index.shape[1] > 0:
        valid_edge_ref = (
            (local_angle_index[1] >= 0)
            & (local_angle_index[1] < num_total_edges)
            & (local_angle_index[2] >= 0)
            & (local_angle_index[2] < num_total_edges)
        )
        local_angle_index = local_angle_index[:, valid_edge_ref]
        angle_ids = angle_ids[valid_edge_ref]

    if local_angle_index.shape[1] > 0:
        local_angle_index[1] = edge_remap[local_angle_index[1]]
        local_angle_index[2] = edge_remap[local_angle_index[2]]
        valid_local_edge_ref = (local_angle_index[1] >= 0) & (local_angle_index[2] >= 0)
        local_angle_index = local_angle_index[:, valid_local_edge_ref]
        angle_ids = angle_ids[valid_local_edge_ref]

    angle_mask = torch.zeros(
        angle_index.shape[1],
        device=angle_index.device,
        dtype=torch.bool,
    )
    if angle_ids.numel() > 0:
        angle_mask[angle_ids] = True

    return FlatGraphPartition(
        rank=rank,
        world_size=world_size,
        local_start=local_start,
        local_end=local_end,
        local_size=local_size,
        atom_index=atom_index,
        batch=local_batch,
        edge_mask=edge_mask,
        edge_ids=edge_ids,
        edge_index=local_edge_index,
        angle_mask=angle_mask,
        angle_ids=angle_ids,
        angle_index=local_angle_index,
    )


def build_all_flat_graph_partitions(
    total_atoms: int,
    edge_index: torch.Tensor,
    angle_index: torch.Tensor,
    batch: torch.Tensor | None = None,
    world_size: int | None = None,
) -> list[FlatGraphPartition]:
    """Build flat-graph partitions for every GP rank."""
    if world_size is None:
        world_size = graph_parallel.get_gp_world_size()
    return [
        build_flat_graph_partition(
            total_atoms,
            edge_index,
            angle_index,
            batch=batch,
            rank=rank,
            world_size=world_size,
        )
        for rank in range(world_size)
    ]
