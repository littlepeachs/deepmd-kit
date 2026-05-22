# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for SeZMInteractionBlock MoE argument plumbing."""

from __future__ import (
    annotations,
)

import pytest
import torch

from deepmd.pt.model.descriptor.sezm_nn.block import (
    SeZMInteractionBlock,
)
from deepmd.pt.model.descriptor.sezm_nn.edge_cache import (
    EdgeFeatureCache,
)
from deepmd.pt.utils.sezm_moe_ep_dp import (
    _is_routing_expert_param,
)


def _make_block(
    *,
    use_moe: bool = False,
    attn_case: str = "none",
    moe_config: dict | None = None,
    seed: int = 20260522,
) -> SeZMInteractionBlock:
    full_attn_res = "independent" if attn_case == "full" else "none"
    block_attn_res = "independent" if attn_case == "block" else "none"
    return SeZMInteractionBlock(
        lmax=1,
        mmax=1,
        channels=8,
        n_focus=3,
        focus_dim=0,
        focus_compete=True,
        so2_norm=False,
        so2_layers=2,
        so2_attn_res="none",
        n_atten_head=0,
        mixed_attention=False,
        legacy_attention=True,
        so2_pre_norm=False,
        so2_post_norm=False,
        ffn_pre_norm=False,
        ffn_post_norm=False,
        ffn_neurons=16,
        grid_mlp=False,
        ffn_blocks=1,
        layer_scale=False,
        full_attn_res=full_attn_res,
        block_attn_res=block_attn_res,
        so2_s2_activation=False,
        ffn_s2_activation=False,
        so2_lebedev_quadrature=False,
        ffn_lebedev_quadrature=False,
        so2_activation_function="silu",
        ffn_activation_function="silu",
        ffn_glu_activation=True,
        mlp_bias=False,
        use_triton=False,
        eps=1e-7,
        dtype=torch.float64,
        seed=seed,
        trainable=True,
        use_moe=use_moe,
        moe_config=moe_config,
        use_compile=False,
    )


def _moe_config() -> dict[str, int | str]:
    return {
        "n_routing_experts": 4,
        "topk": 2,
        "n_shared_experts": 1,
        "ep_size": 1,
        "routing_input": "dst",
        "type_embedding_dim": 8,
    }


def _device_of(block: SeZMInteractionBlock) -> torch.device:
    return next(block.parameters()).device


def _edge_cache(
    block: SeZMInteractionBlock, n_node: int, n_edge: int
) -> EdgeFeatureCache:
    device = _device_of(block)
    dtype = block.dtype
    full_dim = (block.lmax + 1) ** 2
    src = torch.tensor([0, 1, 2, 1, 3, 0], dtype=torch.long, device=device)[:n_edge]
    dst = torch.tensor([1, 2, 3, 0, 0, 2], dtype=torch.long, device=device)[:n_edge]
    eye = (
        torch.eye(full_dim, dtype=dtype, device=device)
        .expand(n_edge, full_dim, full_dim)
        .clone()
    )
    return EdgeFeatureCache(
        src=src,
        dst=dst,
        edge_type_feat=torch.zeros(n_edge, block.channels, dtype=dtype, device=device),
        edge_vec=torch.zeros(n_edge, 3, dtype=dtype, device=device),
        edge_rbf=torch.zeros(n_edge, block.lmax + 1, dtype=dtype, device=device),
        edge_env=torch.ones(n_edge, 1, dtype=dtype, device=device),
        deg=torch.ones(n_node, dtype=dtype, device=device),
        inv_sqrt_deg=torch.ones(n_node, 1, 1, dtype=dtype, device=device),
        D_full=eye,
        Dt_full=eye,
        D_to_m_cache={},
        Dt_from_m_cache={},
        edge_src_gate=None,
    )


def _inputs(
    block: SeZMInteractionBlock,
    *,
    requires_grad: bool = False,
) -> tuple[torch.Tensor, EdgeFeatureCache, torch.Tensor]:
    device = _device_of(block)
    dtype = block.dtype
    n_node = 4
    n_edge = 6
    x = torch.randn(
        n_node,
        (block.lmax + 1) ** 2,
        1,
        block.channels,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )
    cache = _edge_cache(block, n_node, n_edge)
    radial_feat = torch.randn(
        n_edge,
        block.lmax + 1,
        block.channels,
        dtype=dtype,
        device=device,
    )
    return x, cache, radial_feat


def _type_embedding(block: SeZMInteractionBlock) -> torch.Tensor:
    return torch.randn(4, block.channels, dtype=block.dtype, device=_device_of(block))


def _randomize_parameters(module: torch.nn.Module) -> None:
    gen = torch.Generator(device=_device_of(module))
    gen.manual_seed(13579)
    with torch.no_grad():
        for param in module.parameters():
            param.normal_(mean=0.0, std=0.1, generator=gen)


@pytest.mark.parametrize("attn_case", ["none", "full", "block"])
def test_block_use_moe_false_unchanged(attn_case: str) -> None:
    base = _make_block(attn_case=attn_case, seed=123)
    explicit = _make_block(use_moe=False, attn_case=attn_case, seed=123)
    x, cache, radial_feat = _inputs(base)
    x_exp = x.detach().clone().requires_grad_(x.requires_grad)
    unit_history = [x.detach().clone()] if attn_case in {"full", "block"} else None
    unit_history_exp = (
        [x_exp.detach().clone()] if attn_case in {"full", "block"} else None
    )

    out_base = base(x, cache, radial_feat, unit_history)
    out_explicit = explicit(x_exp, cache, radial_feat, unit_history_exp)

    for lhs, rhs in zip(out_base, out_explicit, strict=True):
        if lhs is None or rhs is None:
            assert lhs is rhs
        elif isinstance(lhs, list):
            assert isinstance(rhs, list)
            for lhs_item, rhs_item in zip(lhs, rhs, strict=True):
                torch.testing.assert_close(lhs_item, rhs_item, atol=0.0, rtol=0.0)
        else:
            torch.testing.assert_close(lhs, rhs, atol=0.0, rtol=0.0)


def test_block_use_moe_true_forward() -> None:
    block = _make_block(use_moe=True, moe_config=_moe_config())
    x, cache, radial_feat = _inputs(block)
    type_embedding = _type_embedding(block)

    out, _, _, _ = block(
        x,
        cache,
        radial_feat,
        type_embedding=type_embedding,
        ep_group=None,
    )

    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_block_use_moe_true_requires_type_embedding() -> None:
    block = _make_block(use_moe=True, moe_config=_moe_config())
    x, cache, radial_feat = _inputs(block)

    with pytest.raises(ValueError, match="type_embedding"):
        block(x, cache, radial_feat)


def test_block_use_moe_true_backward() -> None:
    block = _make_block(use_moe=True, moe_config=_moe_config())
    _randomize_parameters(block)
    x, cache, radial_feat = _inputs(block, requires_grad=True)
    type_embedding = _type_embedding(block)

    loss = (
        block(
            x,
            cache,
            radial_feat,
            type_embedding=type_embedding,
            ep_group=None,
        )[0]
        .square()
        .sum()
    )
    loss.backward()

    assert block.so2_conv.moe_conv.router.gate.matrix.grad is not None
    for name, param in block.so2_conv.moe_conv.named_parameters():
        if ".routing_matrix" in name or ".shared_matrix" in name:
            assert param.grad is not None, name


def test_block_moe_param_routing_for_sync() -> None:
    block = _make_block(use_moe=True, moe_config=_moe_config())
    names = [name for name, _ in block.named_parameters()]
    routing_names = [name for name in names if ".routing_matrix" in name]
    shared_names = [name for name in names if ".shared_matrix" in name]

    assert routing_names
    assert shared_names
    assert all(_is_routing_expert_param(name) for name in routing_names)
    assert all(".routing_matrix" not in name for name in shared_names)
    assert all(not _is_routing_expert_param(name) for name in shared_names)
