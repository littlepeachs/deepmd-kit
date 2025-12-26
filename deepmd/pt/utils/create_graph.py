from __future__ import annotations

import sys
from abc import ABC, abstractmethod
import gc
import sys
import warnings
from typing import TYPE_CHECKING, Any
import os
import numpy as np
import torch
from torch import nn
from torch import Tensor

if TYPE_CHECKING:
    from typing import Literal

    from pymatgen.core import Structure
    from typing_extensions import Self

TORCH_DTYPE = torch.float32
from typing import Dict, List, Tuple
import numpy as np
import math


class Node:
    """A node in a graph."""

    def __init__(self, index: int, info: dict | None = None) -> None:
        """Initialize a Node.

        Args:
            index (int): the index of this node
            info (dict, optional): any additional information about this node.
        """
        self.index = index
        self.info = info
        self.neighbors: dict[int, list[UndirectedEdge]] = {}

    def add_neighbor(self, index, edge) -> None:
        """Draw an directed edge between self and the node specified by index.

        Args:
            index (int): the index of neighboring node
            edge (DirectedEdge): an DirectedEdge object pointing from self to the node.
        """
        if index not in self.neighbors:
            self.neighbors[index] = [edge]
        else:
            self.neighbors[index].append(edge)


class Edge(ABC):
    """Abstract base class for edges in a graph."""

    def __init__(
        self, nodes: list, index: int | None = None, info: dict | None = None
    ) -> None:
        """Initialize an Edge."""
        self.nodes = nodes
        self.index = index
        self.info = info

    def __repr__(self) -> str:
        """String representation of this edge."""
        nodes, index, info = self.nodes, self.index, self.info
        return f"{type(self).__name__}({nodes=}, {index=}, {info=})"

    def __hash__(self) -> int:
        """Hash this edge."""
        img = (self.info or {}).get("image")
        img_str = "" if img is None else img.tobytes()
        return hash((self.nodes[0], self.nodes[1], img_str))

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        """Check if two edges are equal."""
        raise NotImplementedError

class UndirectedEdge:
    def __init__(self, nodes: Tuple[int, int], index: int, distance: float):
        # nodes 一定是 (min, max) 方便去重
        self.nodes = nodes
        self.index = index
        self.distance = distance

    def __repr__(self):
        return f"UndirectedEdge(nodes={self.nodes}, index={self.index}, dist={self.distance})"


class Graph:
    """无向原子图：只存无向边，用来生成角列表."""

    def __init__(self, nodes: List["Node"]) -> None:
        self.nodes = nodes
        # key: frozenset({i, j})  -> edge_idx
        self.edge_map: Dict[frozenset, int] = {}
        # 真正的无向边列表，位置就是 edge_index
        self.edges: List[UndirectedEdge] = []
        # 每个原子到底连了哪些 edge_index，方便后面做两两组合
        self.atom2edges: List[List[int]] = [[] for _ in range(len(nodes))]

    def add_edge(self, i: int, j: int, distance: float, dist_tol: float = 1e-6) -> None:
        
        # 直接添加边，不检查重复
        edge_idx = len(self.edges)
        u, v = sorted((i, j))
        edge = UndirectedEdge((u, v), edge_idx, distance)
        self.edges.append(edge)
        
        # 只记录原节点
        self.atom2edges[i].append(edge_idx)
        # self.atom2edges[j].append(edge_idx)

        # 如果你的 Node 里也希望存邻居，可以在这里调：
        # self.nodes[i].add_neighbor(j, edge_idx)
        # self.nodes[j].add_neighbor(i, edge_idx)

    def adjacency_list(self) -> List[Tuple[int, int]]:
        """返回无向邻接表形式的原子对 (i, j)，i<j."""
        return [e.nodes for e in self.edges]

    def angle_triplets(self, cutoff: float = 3.0, edge_index: torch.Tensor = None) -> List[List[int]]:
        """
        返回三元组：
            [center_atom, edge_idx_1, edge_idx_2]
        语义：以 center_atom 为角点，edge_idx_1、edge_idx_2 是连到它的两条无向边的编号。
        为了“无向”，(e1, e2) 会按编号排序，这样不会出现重复的 (e2, e1)。
        cutoff: 过滤掉距离太长的边（按边的 distance）
        """
        triplets: List[List[int]] = []

        for center, edge_indices in enumerate(self.atom2edges):
            # 先按 edge_idx 排序，保证后面组合的稳定性
            # 并且先过滤掉 >cutoff 的边
            valid_edges = [
                ei for ei in edge_indices
                if self.edges[ei].distance <= cutoff
            ]
            valid_edges.sort()
            n = len(valid_edges)
            if n < 2:
                continue

            # 两两组合
            for i in range(n):
                
                e1 = valid_edges[i]
                for j in range(i + 1, n):
                    e2 = valid_edges[j]
                    triplets.append([center, e1, e2])
        return triplets

    def __repr__(self) -> str:
        return f"Graph(num_nodes={len(self.nodes)}, num_edges={len(self.edges)})"
