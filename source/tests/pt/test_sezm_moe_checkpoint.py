# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for SeZM MoE checkpoint resharding helpers."""

from __future__ import (
    annotations,
)

import torch

from deepmd.pt.utils.sezm_moe_checkpoint import (
    gather_state_dict_for_ep_save,
    is_routing_expert_tensor_key,
    slice_state_dict_for_ep_load,
)


def test_is_routing_expert_tensor_key() -> None:
    assert is_routing_expert_tensor_key("model.experts.routing_matrix_m0")
    assert is_routing_expert_tensor_key("model.experts.routing_bias")
    assert not is_routing_expert_tensor_key("model.experts.shared_matrix_m0")
    assert not is_routing_expert_tensor_key("model.router.gate.matrix")


def test_slice_full_state_to_ep_rank() -> None:
    full = {
        "model.experts.routing_matrix_m0": torch.arange(24, device="cpu").reshape(8, 3),
        "model.experts.routing_bias": torch.arange(8, device="cpu"),
        "model.experts.shared_matrix_m0": torch.ones(2, 3, device="cpu"),
    }

    local = slice_state_dict_for_ep_load(
        full,
        ep_rank=2,
        ep_size=4,
        n_routing_experts=8,
    )

    torch.testing.assert_close(
        local["model.experts.routing_matrix_m0"],
        full["model.experts.routing_matrix_m0"][4:6],
    )
    torch.testing.assert_close(
        local["model.experts.routing_bias"],
        full["model.experts.routing_bias"][4:6],
    )
    torch.testing.assert_close(
        local["model.experts.shared_matrix_m0"],
        full["model.experts.shared_matrix_m0"],
    )


def test_same_ep_size_local_state_is_identity() -> None:
    local_state = {
        "model.experts.routing_matrix_m0": torch.arange(6, device="cpu").reshape(2, 3),
        "model.experts.shared_matrix_m0": torch.ones(2, 3, device="cpu"),
    }

    sliced = slice_state_dict_for_ep_load(
        local_state,
        ep_rank=1,
        ep_size=4,
        n_routing_experts=8,
    )

    torch.testing.assert_close(
        sliced["model.experts.routing_matrix_m0"],
        local_state["model.experts.routing_matrix_m0"],
    )


def test_gather_without_dist_returns_copy() -> None:
    state = {
        "model.experts.routing_matrix_m0": torch.arange(6, device="cpu").reshape(2, 3),
        "model.experts.shared_matrix_m0": torch.ones(2, 3, device="cpu"),
    }

    gathered = gather_state_dict_for_ep_save(
        state,
        ep_group=None,
        ep_rank=0,
        ep_size=1,
        n_routing_experts=2,
    )

    torch.testing.assert_close(
        gathered["model.experts.routing_matrix_m0"],
        state["model.experts.routing_matrix_m0"],
    )
    assert (
        gathered["model.experts.routing_matrix_m0"]
        is not state["model.experts.routing_matrix_m0"]
    )
