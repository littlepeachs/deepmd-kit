# SPDX-License-Identifier: LGPL-3.0-or-later
"""MoE SO(2) convolution core for SeZM."""

from __future__ import (
    annotations,
)

import torch
import torch.distributed as dist
import torch.nn as nn

from deepmd.dpmodel.utils.seed import (
    child_seed,
)

from .a2a_ops import (
    all_to_all_differentiable,
)
from .experts import (
    MoESO2ExpertCollection,
)
from .router import (
    MoESO2Router,
)

_VALID_ROUTING_INPUTS = frozenset({"dst", "src", "src+dst"})


class MoESO2Convolution(nn.Module):
    """MoE replacement for the inner SO(2) stack in SeZM convolution."""

    def __init__(
        self,
        lmax: int,
        mmax: int,
        focus_dim: int,
        n_routing_experts: int,
        topk: int,
        n_shared_experts: int,
        ep_size: int = 1,
        routing_input: str = "dst",
        routing_key_dim: int | None = None,
        so2_layers: int = 4,
        activation_function: str = "silu",
        mlp_bias: bool = False,
        use_layer_scale: bool = False,
        precision: str = "float64",
        seed: int | list[int] | None = None,
    ) -> None:
        super().__init__()
        if lmax < 0:
            raise ValueError("`lmax` must be non-negative")
        if mmax < 0:
            raise ValueError("`mmax` must be non-negative")
        if mmax > lmax:
            raise ValueError("`mmax` must be <= `lmax`")
        if focus_dim <= 0:
            raise ValueError("`focus_dim` must be positive")
        if so2_layers <= 0:
            raise ValueError("`so2_layers` must be positive")
        if topk <= 0:
            raise ValueError("`topk` must be positive")
        if n_routing_experts < topk:
            raise ValueError("`n_routing_experts` must be >= `topk`")
        if ep_size <= 0:
            raise ValueError("`ep_size` must be positive")
        if n_routing_experts % ep_size != 0:
            raise ValueError("`n_routing_experts` must be divisible by `ep_size`")
        if n_shared_experts < 0:
            raise ValueError("`n_shared_experts` must be >= 0")
        if routing_input not in _VALID_ROUTING_INPUTS:
            raise ValueError(
                "`routing_input` must be one of 'dst', 'src', or 'src+dst'"
            )
        if routing_key_dim is None or routing_key_dim <= 0:
            raise ValueError("`routing_key_dim` must be positive")

        self.lmax = int(lmax)
        self.mmax = int(mmax)
        self.focus_dim = int(focus_dim)
        self.n_routing_experts = int(n_routing_experts)
        self.topk = int(topk)
        self.n_shared_experts = int(n_shared_experts)
        self.n_focus = self.topk + self.n_shared_experts
        self.ep_size = int(ep_size)
        self.n_experts_per_gpu = self.n_routing_experts // self.ep_size
        self.routing_input = routing_input
        self.routing_key_dim = int(routing_key_dim)
        self.so2_layers = int(so2_layers)
        self.mlp_bias = bool(mlp_bias)
        self.use_layer_scale = bool(use_layer_scale)
        self.reduced_dim = (
            self.lmax
            + 1
            + sum(2 * (self.lmax - m + 1) for m in range(1, self.mmax + 1))
        )

        self.router = MoESO2Router(
            input_dim=self.routing_key_dim,
            n_routing_experts=self.n_routing_experts,
            topk=self.topk,
            routing_input=self.routing_input,
            precision=precision,
            seed=child_seed(seed, 0),
        )
        self.experts = MoESO2ExpertCollection(
            lmax=self.lmax,
            mmax=self.mmax,
            focus_dim=self.focus_dim,
            n_experts_per_gpu=self.n_experts_per_gpu,
            n_shared_experts=self.n_shared_experts,
            so2_layers=self.so2_layers,
            activation_function=activation_function,
            mlp_bias=self.mlp_bias,
            use_layer_scale=self.use_layer_scale,
            precision=precision,
            seed=child_seed(seed, 1),
        )

    def _build_expert_gather_idx(
        self,
        local_eids: torch.Tensor,
        recv_counts: list[int],
        ep_size: int,
        n_per_gpu: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, list[int], torch.Tensor]:
        """Build expert-major gather index from a recv-order buffer."""
        n_token = local_eids.shape[0]
        if n_token == 0:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return empty, [0] * n_per_gpu, empty

        sender_counts: list[list[int]] = []
        sender_offsets: list[list[int]] = []
        offset = 0
        for sender_idx in range(ep_size):
            seg_count = recv_counts[sender_idx]
            if seg_count == 0:
                sender_counts.append([0] * n_per_gpu)
                sender_offsets.append([offset] * n_per_gpu)
            else:
                seg_eids = local_eids[offset : offset + seg_count]
                counts = torch.bincount(seg_eids, minlength=n_per_gpu).tolist()
                offsets: list[int] = []
                running = offset
                for local_eid in range(n_per_gpu):
                    offsets.append(running)
                    running += counts[local_eid]
                sender_counts.append(counts)
                sender_offsets.append(offsets)
            offset += seg_count

        gather_parts: list[torch.Tensor] = []
        split_sizes: list[int] = []
        for local_eid in range(n_per_gpu):
            expert_total = 0
            for sender_idx in range(ep_size):
                count = sender_counts[sender_idx][local_eid]
                if count > 0:
                    start = sender_offsets[sender_idx][local_eid]
                    gather_parts.append(
                        torch.arange(
                            start,
                            start + count,
                            dtype=torch.long,
                            device=device,
                        )
                    )
                    expert_total += count
            split_sizes.append(expert_total)

        gather_idx = (
            torch.cat(gather_parts)
            if gather_parts
            else torch.empty(0, dtype=torch.long, device=device)
        )
        ungather_idx = torch.empty(n_token, dtype=torch.long, device=device)
        ungather_idx[gather_idx] = torch.arange(
            n_token,
            dtype=torch.long,
            device=device,
        )
        return gather_idx, split_sizes, ungather_idx

    def _forward_single_gpu(
        self,
        routing_input_slots: torch.Tensor,
        topk_indices: torch.Tensor,
        rad_routing: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run routing experts locally without A2A."""
        n_edge, topk, reduced_dim, focus_dim = routing_input_slots.shape
        if topk != self.topk:
            raise ValueError(f"Expected topk={self.topk}, got {topk}.")
        if reduced_dim != self.reduced_dim or focus_dim != self.focus_dim:
            raise ValueError(
                "Expected routing input slots with trailing shape "
                f"({self.reduced_dim}, {self.focus_dim}), got "
                f"({reduced_dim}, {focus_dim})."
            )
        if topk_indices.shape != (n_edge, self.topk):
            raise ValueError(
                f"Expected `topk_indices` shape {(n_edge, self.topk)}, "
                f"got {tuple(topk_indices.shape)}."
            )

        tokens_flat = routing_input_slots.reshape(
            n_edge * self.topk,
            reduced_dim,
            focus_dim,
        )
        expert_ids_flat = topk_indices.reshape(n_edge * self.topk)
        sort_order = torch.argsort(expert_ids_flat, stable=True)
        sorted_tokens = tokens_flat.index_select(0, sort_order)
        sorted_expert_ids = expert_ids_flat.index_select(0, sort_order)
        split_sizes = torch.bincount(
            sorted_expert_ids,
            minlength=self.n_experts_per_gpu,
        ).tolist()

        if rad_routing is None:
            sorted_rad_factor = None
        else:
            rad_flat = rad_routing.reshape(n_edge * self.topk, focus_dim)
            sorted_rad_factor = rad_flat.index_select(0, sort_order)

        sorted_output = self.experts.forward_routing(
            sorted_tokens,
            sorted_expert_ids,
            split_sizes,
            sorted_rad_factor,
        )
        unsort_idx = torch.empty_like(sort_order)
        unsort_idx[sort_order] = torch.arange(
            sort_order.numel(),
            dtype=torch.long,
            device=sort_order.device,
        )
        output_flat = sorted_output.index_select(0, unsort_idx)
        return output_flat.reshape(n_edge, self.topk, reduced_dim, focus_dim)

    def _forward_multi_gpu(
        self,
        routing_input_slots: torch.Tensor,
        topk_indices: torch.Tensor,
        rad_routing: torch.Tensor | None,
        ep_group: object,
    ) -> torch.Tensor:
        """Run routing experts through EP all-to-all dispatch/combine."""
        # NOTE: Single argsort by global expert id on sender side serves dual
        # purpose: (a) buckets tokens per target GPU for A2A, (b) within each
        # GPU's chunk, additionally orders by local expert id. Receiver side
        # then reorders via O(N) structured gather (no argsort) since each
        # sender's segment arrives pre-sorted by local eid. See DPA3
        # moe_layer.py _build_expert_gather_idx.
        n_edge, topk, reduced_dim, focus_dim = routing_input_slots.shape
        device = routing_input_slots.device
        ep_size = dist.get_world_size(group=ep_group)
        if ep_size != self.ep_size:
            raise ValueError(f"Expected ep_size={self.ep_size}, got {ep_size}.")

        tokens_flat = routing_input_slots.reshape(
            n_edge * topk,
            reduced_dim,
            focus_dim,
        )
        global_eids_flat = topk_indices.reshape(n_edge * topk)
        sort_idx = torch.argsort(global_eids_flat, stable=True)
        sorted_tokens = torch.index_select(tokens_flat, 0, sort_idx)
        sorted_global_eids = torch.index_select(global_eids_flat, 0, sort_idx)

        unsort_idx = torch.empty_like(sort_idx)
        unsort_idx[sort_idx] = torch.arange(
            sort_idx.numel(),
            dtype=torch.long,
            device=device,
        )

        if rad_routing is None:
            sorted_rad = None
        else:
            rad_flat = rad_routing.reshape(n_edge * topk, focus_dim)
            sorted_rad = torch.index_select(rad_flat, 0, sort_idx)

        sorted_target_gpu = sorted_global_eids // self.n_experts_per_gpu
        send_counts = torch.bincount(sorted_target_gpu, minlength=ep_size)
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts, group=ep_group)
        send_counts_list = send_counts.tolist()
        recv_counts_list = recv_counts.tolist()

        recv_tokens = all_to_all_differentiable(
            sorted_tokens,
            send_counts_list,
            recv_counts_list,
            ep_group,
        )
        recv_global_eids = torch.empty(
            sum(recv_counts_list),
            dtype=torch.long,
            device=device,
        )
        dist.all_to_all_single(
            recv_global_eids,
            sorted_global_eids.long().contiguous(),
            output_split_sizes=recv_counts_list,
            input_split_sizes=send_counts_list,
            group=ep_group,
        )
        if sorted_rad is None:
            recv_rad = None
        else:
            recv_rad = all_to_all_differentiable(
                sorted_rad,
                send_counts_list,
                recv_counts_list,
                ep_group,
            )

        recv_local_eids = recv_global_eids % self.n_experts_per_gpu
        gather_idx, local_split_sizes, ungather_idx = self._build_expert_gather_idx(
            recv_local_eids,
            recv_counts_list,
            ep_size,
            self.n_experts_per_gpu,
            device,
        )
        gathered_tokens = torch.index_select(recv_tokens, 0, gather_idx)
        gathered_local_eids = torch.index_select(recv_local_eids, 0, gather_idx)
        if recv_rad is None:
            gathered_rad = None
        else:
            gathered_rad = torch.index_select(recv_rad, 0, gather_idx)

        out_gathered = self.experts.forward_routing(
            gathered_tokens,
            gathered_local_eids,
            local_split_sizes,
            gathered_rad,
        )
        out_recv_order = torch.index_select(out_gathered, 0, ungather_idx)
        returned = all_to_all_differentiable(
            out_recv_order,
            recv_counts_list,
            send_counts_list,
            ep_group,
        )
        output_flat = torch.index_select(returned, 0, unsort_idx)
        return output_flat.reshape(n_edge, topk, reduced_dim, focus_dim)

    def forward(
        self,
        x_local: torch.Tensor,
        routing_key: torch.Tensor,
        rad_factor: torch.Tensor | None = None,
        ep_group: object | None = None,
    ) -> torch.Tensor:
        """Apply MoE experts and alpha weighting to local SO(2) features."""
        if x_local.dim() != 4:
            raise ValueError("`x_local` must have shape (E, F, D_m, Cf)")
        n_edge, n_focus, reduced_dim, focus_dim = x_local.shape
        if n_focus != self.n_focus:
            raise ValueError(
                f"`x_local.shape[1]` must equal n_focus={self.n_focus}, got {n_focus}."
            )
        if reduced_dim != self.reduced_dim or focus_dim != self.focus_dim:
            raise ValueError(
                "Expected `x_local` trailing shape "
                f"({self.reduced_dim}, {self.focus_dim}), got "
                f"({reduced_dim}, {focus_dim})."
            )
        if routing_key.shape != (n_edge, self.routing_key_dim):
            raise ValueError(
                f"Expected `routing_key` shape {(n_edge, self.routing_key_dim)}, "
                f"got {tuple(routing_key.shape)}."
            )
        if self.mlp_bias:
            if rad_factor is None:
                raise ValueError("`mlp_bias=True` requires `rad_factor`")
            if rad_factor.shape != (n_edge, self.n_focus, self.focus_dim):
                raise ValueError(
                    f"Expected `rad_factor` shape "
                    f"{(n_edge, self.n_focus, self.focus_dim)}, "
                    f"got {tuple(rad_factor.shape)}."
                )

        topk_weights, topk_indices = self.router(routing_key)
        routing_input_slots = x_local[:, : self.topk, :, :]
        shared_input_slots = x_local[:, self.topk :, :, :]
        if rad_factor is None:
            rad_routing = None
            rad_shared = None
        else:
            rad_routing = rad_factor[:, : self.topk, :]
            rad_shared = rad_factor[:, self.topk :, :]

        shared_output = self.experts.forward_shared(shared_input_slots, rad_shared)
        if ep_group is None:
            routing_output = self._forward_single_gpu(
                routing_input_slots,
                topk_indices,
                rad_routing,
            )
        else:
            routing_output = self._forward_multi_gpu(
                routing_input_slots,
                topk_indices,
                rad_routing,
                ep_group,
            )

        full = torch.cat([routing_output, shared_output], dim=1)
        alpha_shared = torch.ones(
            n_edge,
            self.n_shared_experts,
            dtype=x_local.dtype,
            device=x_local.device,
        )
        alpha = torch.cat([topk_weights, alpha_shared], dim=-1)
        return full * alpha[:, :, None, None]


__all__ = ["MoESO2Convolution"]
