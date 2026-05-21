# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for single-GPU SeZM MoE SO(2) convolution."""

from __future__ import (
    annotations,
)

import pytest
import torch

from deepmd.pt.model.descriptor.sezm_nn.moe.conv import (
    MoESO2Convolution,
)


def _make_conv(
    *,
    lmax: int = 3,
    mmax: int = 1,
    focus_dim: int = 8,
    n_routing_experts: int = 4,
    topk: int = 2,
    n_shared_experts: int = 1,
    ep_size: int = 1,
    routing_input: str = "dst",
    routing_key_dim: int = 16,
    so2_layers: int = 4,
    mlp_bias: bool = False,
    use_layer_scale: bool = False,
    seed: int = 20260518,
) -> MoESO2Convolution:
    return MoESO2Convolution(
        lmax=lmax,
        mmax=mmax,
        focus_dim=focus_dim,
        n_routing_experts=n_routing_experts,
        topk=topk,
        n_shared_experts=n_shared_experts,
        ep_size=ep_size,
        routing_input=routing_input,
        routing_key_dim=routing_key_dim,
        so2_layers=so2_layers,
        activation_function="silu",
        mlp_bias=mlp_bias,
        use_layer_scale=use_layer_scale,
        precision="float64",
        seed=seed,
    )


def _device_of(conv: MoESO2Convolution) -> torch.device:
    return conv.experts.routing_stack.layers[0].routing_matrix_m0.device


def _inputs(
    conv: MoESO2Convolution,
    n_edge: int = 20,
    *,
    requires_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = _device_of(conv)
    x_local = torch.randn(
        n_edge,
        conv.n_focus,
        conv.reduced_dim,
        conv.focus_dim,
        dtype=torch.float64,
        device=device,
        requires_grad=requires_grad,
    )
    routing_key = torch.randn(
        n_edge,
        conv.routing_key_dim,
        dtype=torch.float64,
        device=device,
        requires_grad=requires_grad,
    )
    return x_local, routing_key


def _rad_factor(conv: MoESO2Convolution, n_edge: int) -> torch.Tensor:
    return torch.randn(
        n_edge,
        conv.n_focus,
        conv.focus_dim,
        dtype=torch.float64,
        device=_device_of(conv),
    )


def test_invalid_config_raises() -> None:
    cases = [
        ({"n_routing_experts": 1, "topk": 2}, "`n_routing_experts`"),
        ({"topk": 0}, "`topk` must be positive"),
        ({"n_routing_experts": 4, "ep_size": 3}, "divisible"),
        ({"routing_input": "wrong"}, "`routing_input`"),
        ({"routing_key_dim": 0}, "`routing_key_dim`"),
        ({"n_shared_experts": -1}, "`n_shared_experts`"),
        ({"lmax": -1}, "`lmax`"),
        ({"mmax": -1}, "`mmax`"),
        ({"lmax": 1, "mmax": 2}, "`mmax` must be <= `lmax`"),
        ({"focus_dim": 0}, "`focus_dim`"),
        ({"so2_layers": 0}, "`so2_layers`"),
    ]
    for kwargs, message in cases:
        with pytest.raises(ValueError, match=message):
            _make_conv(**kwargs)


def test_forward_shape_basic() -> None:
    conv = _make_conv(topk=2, n_shared_experts=1)
    x_local, routing_key = _inputs(conv)

    out = conv(x_local, routing_key, ep_group=None)

    assert out.shape == (20, 3, conv.reduced_dim, conv.focus_dim)
    assert torch.isfinite(out).all()


def test_forward_shape_with_shared_zero() -> None:
    conv = _make_conv(topk=2, n_shared_experts=0)
    x_local, routing_key = _inputs(conv)

    out = conv(x_local, routing_key, ep_group=None)

    assert out.shape == (20, 2, conv.reduced_dim, conv.focus_dim)
    assert torch.isfinite(out).all()


def test_forward_shape_with_topk_one() -> None:
    conv = _make_conv(topk=1, n_shared_experts=2)
    x_local, routing_key = _inputs(conv)

    out = conv(x_local, routing_key, ep_group=None)

    assert out.shape == (20, 3, conv.reduced_dim, conv.focus_dim)
    assert torch.isfinite(out).all()


def test_mlp_bias_true_shape() -> None:
    conv = _make_conv(topk=2, n_shared_experts=1, mlp_bias=True)
    x_local, routing_key = _inputs(conv)
    rad_factor = _rad_factor(conv, x_local.shape[0])

    out = conv(x_local, routing_key, rad_factor, ep_group=None)

    assert out.shape == x_local.shape
    assert torch.isfinite(out).all()


def test_alpha_shared_is_identity() -> None:
    conv = _make_conv(topk=1, n_shared_experts=2)
    x_local, routing_key = _inputs(conv, n_edge=10)

    out = conv(x_local, routing_key, ep_group=None)
    shared_only = conv.experts.forward_shared(x_local[:, conv.topk :, :, :], None)

    torch.testing.assert_close(
        out[:, conv.topk :, :, :],
        shared_only,
        atol=1e-12,
        rtol=1e-12,
    )


def test_routing_alpha_weighting() -> None:
    conv = _make_conv(topk=2, n_shared_experts=1)
    x_local, routing_key = _inputs(conv, n_edge=10)

    out = conv(x_local, routing_key, ep_group=None)
    topk_weights, topk_indices = conv.router(routing_key)
    routing_unweighted = conv._forward_single_gpu(
        x_local[:, : conv.topk, :, :],
        topk_indices,
        None,
    )
    expected = routing_unweighted * topk_weights[:, :, None, None]

    torch.testing.assert_close(
        out[:, : conv.topk, :, :],
        expected,
        atol=1e-12,
        rtol=1e-12,
    )


def test_forward_determinism() -> None:
    conv = _make_conv(topk=2, n_shared_experts=1)
    x_local, routing_key = _inputs(conv, n_edge=10)

    out_a = conv(x_local, routing_key, ep_group=None)
    out_b = conv(x_local, routing_key, ep_group=None)

    torch.testing.assert_close(out_a, out_b, atol=1e-15, rtol=1e-15)


def test_backward_grads_present() -> None:
    conv = _make_conv(topk=2, n_shared_experts=1)
    x_local, routing_key = _inputs(conv, requires_grad=True)

    loss = conv(x_local, routing_key, ep_group=None).sum()
    loss.backward()

    assert conv.router.gate.matrix.grad is not None
    assert conv.router.gate.matrix.grad.abs().sum().item() > 0.0
    for name, param in conv.named_parameters():
        if ".routing_matrix" in name or ".shared_matrix" in name:
            assert param.grad is not None, name
            assert param.grad.abs().sum().item() > 0.0, name


def test_create_graph_second_backward() -> None:
    conv = _make_conv(topk=2, n_shared_experts=1)
    x_local, routing_key = _inputs(conv, requires_grad=True)

    loss = conv(x_local, routing_key, ep_group=None).sum()
    grad_x, grad_key = torch.autograd.grad(
        loss,
        (x_local, routing_key),
        create_graph=True,
    )
    (grad_x.sum() + grad_key.sum()).backward()

    assert conv.router.gate.matrix.grad is not None
    assert conv.router.gate.matrix.grad.abs().sum().item() > 0.0
    for name, param in conv.named_parameters():
        if ".routing_matrix" in name or ".shared_matrix" in name:
            assert param.grad is not None, name
            assert param.grad.abs().sum().item() > 0.0, name


def test_n_routing_experts_equal_topk_edge_case() -> None:
    conv = _make_conv(n_routing_experts=2, topk=2, n_shared_experts=0)
    x_local, routing_key = _inputs(conv)

    out = conv(x_local, routing_key, ep_group=None)

    assert out.shape == x_local.shape
    assert torch.isfinite(out).all()


def test_build_expert_gather_idx_basic() -> None:
    conv = _make_conv()
    device = _device_of(conv)
    local_eids = torch.tensor([0, 0, 2, 0, 1, 1, 2], dtype=torch.long, device=device)

    gather_idx, split_sizes, ungather_idx = conv._build_expert_gather_idx(
        local_eids,
        recv_counts=[3, 4],
        ep_size=2,
        n_per_gpu=4,
        device=device,
    )

    torch.testing.assert_close(
        gather_idx,
        torch.tensor([0, 1, 3, 4, 5, 2, 6], dtype=torch.long, device=device),
    )
    assert split_sizes == [3, 2, 2, 0]
    inverse = torch.empty_like(gather_idx)
    inverse[gather_idx] = torch.arange(gather_idx.numel(), device=device)
    torch.testing.assert_close(ungather_idx, inverse)


def test_build_expert_gather_idx_empty_expert() -> None:
    conv = _make_conv()
    device = _device_of(conv)
    local_eids = torch.tensor([0, 2, 2, 3], dtype=torch.long, device=device)

    gather_idx, split_sizes, _ = conv._build_expert_gather_idx(
        local_eids,
        recv_counts=[2, 2],
        ep_size=2,
        n_per_gpu=4,
        device=device,
    )

    torch.testing.assert_close(
        gather_idx,
        torch.tensor([0, 1, 2, 3], dtype=torch.long, device=device),
    )
    assert split_sizes == [1, 0, 2, 1]


def test_build_expert_gather_idx_empty_sender() -> None:
    conv = _make_conv()
    device = _device_of(conv)
    local_eids = torch.tensor([0, 1, 3, 3], dtype=torch.long, device=device)

    gather_idx, split_sizes, _ = conv._build_expert_gather_idx(
        local_eids,
        recv_counts=[0, 2, 2],
        ep_size=3,
        n_per_gpu=4,
        device=device,
    )

    torch.testing.assert_close(
        gather_idx,
        torch.tensor([0, 1, 2, 3], dtype=torch.long, device=device),
    )
    assert split_sizes == [1, 1, 0, 2]


def test_build_expert_gather_idx_n_zero() -> None:
    conv = _make_conv()
    device = _device_of(conv)
    local_eids = torch.empty(0, dtype=torch.long, device=device)

    gather_idx, split_sizes, ungather_idx = conv._build_expert_gather_idx(
        local_eids,
        recv_counts=[0, 0],
        ep_size=2,
        n_per_gpu=4,
        device=device,
    )

    assert gather_idx.numel() == 0
    assert ungather_idx.numel() == 0
    assert split_sizes == [0, 0, 0, 0]


def test_gather_ungather_inverse() -> None:
    conv = _make_conv()
    device = _device_of(conv)
    local_eids = torch.tensor([0, 1, 3, 0, 2, 3, 3], dtype=torch.long, device=device)

    gather_idx, _, ungather_idx = conv._build_expert_gather_idx(
        local_eids,
        recv_counts=[3, 4],
        ep_size=2,
        n_per_gpu=4,
        device=device,
    )
    x = torch.randn(7, 3, dtype=torch.float64, device=device)
    gathered = torch.index_select(x, 0, gather_idx)
    restored = torch.index_select(gathered, 0, ungather_idx)

    torch.testing.assert_close(restored, x, atol=0.0, rtol=0.0)
