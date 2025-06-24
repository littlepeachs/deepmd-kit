"""Atomic moments.

This is an expansion of the moments defined in MTP. Here we add features of different
rank to the moments.

The atomic moment is defined as:

M_v,p = sum_j f * h_v1 contract D_v2,
where D_v2 = r_ij otimes r_ij otimes ...

M_v = sum_p W_p M_v,p,

We require v1 <= v2.

Multiple contractions between h_v1 and D_v2 can lead to the same v, and this is denoted
by p.

Note, the operations are separate for each radial degree u, and thus u is omitted in
the notation.
"""
from typing import Union, Dict

import torch
from torch import Tensor, nn
from deepmd.pt.model.network.mlp import (
    LinearCombination,
)
from .camp_util import MLP
from .camp_util import get_dyadic_tensor

from .atomic_moment_rule import get_atomic_moment_rules
from .camp_util import RadialPart
from .camp_util import scatter

from deepmd.pt.model.network.mlp import (
    MLPLayer,
)
from deepmd.dpmodel.utils.seed import (
    child_seed,
)

class AtomicMoment(nn.Module):
    def __init__(
        self,
        max_u: int,
        max_v1: int,
        max_v2: int,
        n_dim: int,
        e_dim: int,
        rbf_dim: int = 32,
        max_chebyshev_degree: int = 8,
        radial_mlp_hidden_layers: Union[list[int], int] = 2,
        r_cut: float = 5,
        envelope: int = 6,
        precision: str = "float64",
        seed: int = 100,
    ):
        """
        Atomic Moments.

        Args:
            max_u: maximum radial degree u of the moment tensor.
            max_v1: max angular degree v1 of the atomic features. If None, this is set
                to max_v2. This main use of this argument is for the first layer, where
                the atomic features are scalars and thus v1 = 0.
            max_v2: maximum angular degree v2 of the dyadic tensor, that is the maximum
                value of v2 in M_v,p = sum_j f * h_v1 contract D_v2.
            num_atom_types: number of atomic types.
            num_average_neigh:
            max_chebyshev_degree: max degree of the Chebyshev polynomial to use to
                construct the radial basis functions. The total number of chebyshev
                polynomials is `max_chebyshev_degree + 1`; +1 for the zeroth degree.
            radial_mlp_hidden_layers: if list of int, this gives the size of each hidden
                layer in the MLP that is applied to the radial basis functions. If int,
                this gives the number of hidden layers, and the size of each hidden
                layer is set to `max_u + 1`, the number of radial basis functions.
            number of hidden layers in the MLP that is applied to
                the radial basis functions.
            r_cut: cutoff distance.
            envelope: degree of the polynomial envelope function to make the radial
                basis function smooth at r_cut.
        """
        super().__init__()
        self.max_u = max_u
        self.max_v1 = max_v1
        self.max_v2 = max_v2
        self.max_chebyshev_degree = max_chebyshev_degree
        self.radial_mlp_hidden_layers = radial_mlp_hidden_layers
        self.r_cut = r_cut
        self.envelope = envelope
        self.e_dim = e_dim
        self.n_dim = n_dim
        self.rbf_dim = rbf_dim
        atomic_moment_rules = {
            rank: get_atomic_moment_rules(max_in_rank=max_v2, out_rank=rank)
            for rank in range(max_v2 + 1)
        }
        # filter to keep only rules with v1 <= max_v1. By default, the rules consists
        # rules for v1 upto v1<=v2.
        self.atomic_moment_ranks: dict[int, list[list[int]]] = {
            rank: [rule["ranks"] for rule in rules if rule["ranks"][0] <= max_v1]
            for rank, rules in atomic_moment_rules.items()
        }
        self.atomic_moment_einsum_rule: dict[int, list[str]] = {
            rank: [rule["einsum_rule"] for rule in rules if rule["ranks"][0] <= max_v1]
            for rank, rules in atomic_moment_rules.items()
        }

        # MLP on the radial part. This is separate for each combination of v, v1, and v2

        if isinstance(radial_mlp_hidden_layers, int):
            radial_mlp_hidden_layers = [
                max_u + 1 for _ in range(radial_mlp_hidden_layers)
            ]

        self.radial_mlp : Dict[str, nn.Module] = {}
        for v, rules in self.atomic_moment_ranks.items():
            for rule in rules:
                v1 = rule[0]
                v2 = rule[1]
                self.radial_mlp[f"{v}_{v1}_{v2}"] = MLPLayer(
                    max_u,
                    self.rbf_dim,
                    precision=precision,
                    seed=child_seed(seed, 30),
                )

        self.linear_path = nn.ModuleDict(
            {
                str(rank): LinearCombination(len(rules), self.rbf_dim)
                for rank, rules in self.atomic_moment_ranks.items()
                if len(rules) > 1
            }
        )

        self.linear_channel = nn.ModuleDict(
            {
                str(rank): MLPLayer(
                    self.rbf_dim,
                    self.rbf_dim,
                    bias = False,
                    precision=precision,
                    seed=child_seed(seed, 31),
                )
                for rank, _ in self.atomic_moment_ranks.items()
            }
        )

    def forward(
        self,
        atom_feat: dict[int, Tensor],
        edge_index: Tensor,
        rbf_ebd: Tensor,
        diff: Tensor,
        num_average_neigh: float,
        sw: Tensor,
    ) -> dict[int, Tensor]:
        """

        Args:
            edge_vector:
            edge_idx:
            atom_type:
            atom_feats: atomic features. {v: tensor}, where v is the angular degree,
                and the tensor is of shape (n_u, n_atoms, 3, 3, ...). n_u denotes the
                batch dimension of the radial degree u, and the number of 3s is v.

        Returns:
            Atomic moments: {v: tensor}, where the tensor M_uv has shape
                (n_u, n_atoms, 3, 3, ...).
        """

        i_idx = edge_index[:,0]
        j_idx = edge_index[:,1]
        
        # radial part, shape (n_edges, n_u)
        # print(torch.max(atom_feat[0]))
        # import pdb; pdb.set_trace()
        dyad_tensors = {
            # TODO get_dyadic_tensor cause NAN
            v: get_dyadic_tensor(diff, rank=v, normalize=True)
            for v in range(self.max_v2 + 1)
        }  # (n_edges, 3, 3, ...), number of 3: v
        
        M: dict[int, Tensor] = {}
        for v, rules in self.atomic_moment_ranks.items():
            # atomic moments of rank v from different paths
            M_uvp = []

            einsum_rules = self.atomic_moment_einsum_rule[v]
            
            for rule, equation in zip(rules, einsum_rules):
                v1 = rule[0]
                v2 = rule[1]

                # Make indexing ModuleDict work
                # See https://github.com/pytorch/pytorch/issues/68568
                fn = self.radial_mlp[f"{v}_{v1}_{v2}"]
                R = fn.forward(rbf_ebd).transpose(0, 1)  # shape (n_u, n_edges)
                
                # neighbor_ebd = atom_feat.view(-1, atom_feat.shape[-1])[j_idx].transpose(0, 1)  # shape (n_edges, n_dim)

                neighbor_ebd = atom_feat[v1]
                neighbor_ebd = neighbor_ebd[:, j_idx, ...]
                if v1 == 0 or v2 == 0:
                    t = torch.einsum("ue,ue...,e...->ue...", R, neighbor_ebd, dyad_tensors[v2])
                else:
                    # shape (n_u, n_edges, 1, 1, ...,), number of 1: rank
                    shaped_R = R.reshape(R.shape + torch.Size([1] * v))

                    t = shaped_R * torch.einsum(equation, neighbor_ebd, dyad_tensors[v2])
                
                t = torch.einsum('ue...,e->ue...', t, sw)
                # aggregate atoms j (src) to atom i (dst)
                # shape (n_u, n_atoms, 3, 3, ...), number of 3: rank
                t = (
                    scatter(t, i_idx, reduce="sum", dim=1)
                    / (num_average_neigh**0.5)
                )
                
                
                M_uvp.append(t)
            
            # linear combination of different paths
            
            if len(M_uvp) > 1:
                M_uvp2 = torch.stack(M_uvp)  # shape (n_rules, n_u, n_atoms, 3, 3, ...)
                fn = self.linear_path[str(v)]
                M_uv = fn.forward(M_uvp2)  # shape (n_u, n_atoms, 3, 3, ...)
            else:
                M_uv = M_uvp[0]  # shape (n_u, n_atoms, 3, 3, ...)
                
            # linear mix of different channels

            fn = self.linear_channel[str(v)]
            M_uv = fn.forward(M_uv,dims=0)  # shape (n_u, n_atoms, 3, 3, ...)
            
            M[v] = M_uv
        # print(torch.max(M[0]))
        # import pdb; pdb.set_trace()
        return M


def main():
    """
    测试AtomicMoment模块的主函数
    """
    # 设置测试参数
    max_u = 2
    max_v1 = 1
    max_v2 = 2
    num_atom_types = 2
    num_average_neigh = 10.0
    max_chebyshev_degree = 8
    radial_mlp_hidden_layers = [16, 16]
    r_cut = 5.0
    envelope = 6
    
    # 创建AtomicMoment模型
    model = AtomicMoment(
        max_u=max_u,
        max_v1=max_v1,
        max_v2=max_v2,
        num_atom_types=num_atom_types,
        num_average_neigh=num_average_neigh,
        max_chebyshev_degree=max_chebyshev_degree,
        radial_mlp_hidden_layers=radial_mlp_hidden_layers,
        r_cut=r_cut,
        envelope=envelope,
    )
    
    # 创建测试数据
    n_atoms = 10
    n_edges = 50
    
    # 边向量和边索引
    edge_vector = torch.randn(n_edges, 3) * 2.0  # 随机边向量
    edge_idx = torch.randint(0, n_atoms, (2, n_edges))  # 随机边索引
    atom_type = torch.randint(0, num_atom_types, (n_atoms,))  # 原子类型
    
    # 原子特征
    atom_feats = {}
    for v in range(max_v1 + 1):
        if v == 0:
            # 标量特征
            atom_feats[v] = torch.randn(max_u + 1, n_atoms)
        else:
            # 向量特征
            shape = [max_u + 1, n_atoms] + [3] * v
            atom_feats[v] = torch.randn(*shape)
    
    # 前向传播
    print("开始测试AtomicMoment模型...")
    print(f"模型参数: max_u={max_u}, max_v1={max_v1}, max_v2={max_v2}")
    print(f"原子数量: {n_atoms}, 边数量: {n_edges}")
    print(f"边向量形状: {edge_vector.shape}")
    print(f"边索引形状: {edge_idx.shape}")
    print(f"原子类型形状: {atom_type.shape}")
    
    for v, feat in atom_feats.items():
        print(f"原子特征 v={v} 形状: {feat.shape}")
    
    # 执行前向传播
    with torch.no_grad():
        output = model(edge_vector, edge_idx, atom_type, atom_feats)
    
    print("\n输出结果:")
    for v, tensor in output.items():
        print(f"原子矩 v={v} 形状: {tensor.shape}")
        print(f"原子矩 v={v} 数据类型: {tensor.dtype}")
        print(f"原子矩 v={v} 均值: {tensor.mean().item():.6f}")
        print(f"原子矩 v={v} 标准差: {tensor.std().item():.6f}")
        print()
    
    print("测试完成!")


if __name__ == "__main__":
    main()
