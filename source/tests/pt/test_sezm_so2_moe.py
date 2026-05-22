# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for SO2Convolution MoE integration."""

from __future__ import (
    annotations,
)

import pytest
import torch

from deepmd.pt.model.descriptor.sezm_nn.edge_cache import (
    EdgeFeatureCache,
)
from deepmd.pt.model.descriptor.sezm_nn.so2 import (
    SO2Convolution,
)


def _device_of(module: torch.nn.Module) -> torch.device:
    return next(module.parameters()).device


def _edge_cache(
    *,
    n_node: int,
    n_edge: int,
    lmax: int,
    channels: int,
    dtype: torch.dtype,
    device: torch.device,
) -> EdgeFeatureCache:
    full_dim = (lmax + 1) ** 2
    src_all = torch.tensor([0, 1, 2, 1, 3, 0, 2, 3], dtype=torch.long, device=device)
    dst_all = torch.tensor([1, 2, 3, 0, 0, 2, 1, 1], dtype=torch.long, device=device)
    src = src_all[:n_edge]
    dst = dst_all[:n_edge]
    eye = (
        torch.eye(full_dim, dtype=dtype, device=device)
        .expand(n_edge, full_dim, full_dim)
        .clone()
    )
    edge_env = torch.ones(n_edge, 1, dtype=dtype, device=device)
    return EdgeFeatureCache(
        src=src,
        dst=dst,
        edge_type_feat=torch.zeros(n_edge, channels, dtype=dtype, device=device),
        edge_vec=torch.zeros(n_edge, 3, dtype=dtype, device=device),
        edge_rbf=torch.zeros(n_edge, lmax + 1, dtype=dtype, device=device),
        edge_env=edge_env,
        deg=torch.ones(n_node, dtype=dtype, device=device),
        inv_sqrt_deg=torch.ones(n_node, 1, 1, dtype=dtype, device=device),
        D_full=eye,
        Dt_full=eye,
        D_to_m_cache={},
        Dt_from_m_cache={},
        edge_src_gate=None,
    )


def _inputs(
    module: SO2Convolution,
) -> tuple[torch.Tensor, EdgeFeatureCache, torch.Tensor]:
    device = _device_of(module)
    dtype = module.dtype
    n_node = 4
    n_edge = 6
    x = torch.randn(
        n_node,
        (module.lmax + 1) ** 2,
        module.channels,
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    cache = _edge_cache(
        n_node=n_node,
        n_edge=n_edge,
        lmax=module.lmax,
        channels=module.channels,
        dtype=dtype,
        device=device,
    )
    radial_feat = torch.randn(
        n_edge,
        module.lmax + 1,
        module.channels,
        dtype=dtype,
        device=device,
    )
    return x, cache, radial_feat


def _moe_config(
    *,
    topk: int = 2,
    n_shared_experts: int = 1,
    routing_input: str = "dst",
    type_embedding_dim: int = 8,
) -> dict[str, int | str]:
    return {
        "n_routing_experts": 4,
        "topk": topk,
        "n_shared_experts": n_shared_experts,
        "ep_size": 1,
        "routing_input": routing_input,
        "type_embedding_dim": type_embedding_dim,
    }


def _make_so2(
    *,
    use_moe: bool = False,
    moe_config: dict | None = None,
    n_focus: int = 3,
    mlp_bias: bool = False,
    so2_norm: bool = False,
    so2_attn_res: str = "none",
    use_compile: bool = False,
    seed: int = 20260518,
) -> SO2Convolution:
    return SO2Convolution(
        lmax=1,
        mmax=1,
        channels=8,
        n_focus=n_focus,
        focus_dim=0,
        focus_compete=True,
        so2_norm=so2_norm,
        so2_layers=2,
        so2_attn_res=so2_attn_res,
        layer_scale=False,
        n_atten_head=0,
        mixed_attention=False,
        legacy_attention=True,
        s2_activation=False,
        lebedev_quadrature=False,
        activation_function="silu",
        mlp_bias=mlp_bias,
        use_triton=False,
        eps=1e-7,
        dtype=torch.float64,
        seed=seed,
        trainable=True,
        use_moe=use_moe,
        moe_config=moe_config,
        use_compile=use_compile,
    )


def _type_embedding(
    module: SO2Convolution, *, requires_grad: bool = False
) -> torch.Tensor:
    device = _device_of(module)
    return torch.randn(
        4,
        module.channels,
        dtype=module.dtype,
        device=device,
        requires_grad=requires_grad,
    )


def _randomize_parameters(module: torch.nn.Module) -> None:
    gen = torch.Generator(device=_device_of(module))
    gen.manual_seed(20260522)
    with torch.no_grad():
        for param in module.parameters():
            param.normal_(mean=0.0, std=0.1, generator=gen)


def test_use_moe_false_unchanged() -> None:
    base = _make_so2(seed=123)
    explicit = _make_so2(use_moe=False, seed=123)
    x, cache, radial_feat = _inputs(base)
    x_exp = x.detach().clone().requires_grad_(True)

    out_base = base(x, cache, radial_feat)
    out_explicit = explicit(x_exp, cache, radial_feat)

    torch.testing.assert_close(out_base, out_explicit, atol=0.0, rtol=0.0)


def test_use_moe_true_invalid_config_raises() -> None:
    cases = [
        ({"so2_norm": True, "moe_config": _moe_config()}, "so2_norm=False"),
        (
            {"so2_attn_res": "independent", "moe_config": _moe_config()},
            "so2_attn_res='none'",
        ),
        ({"use_compile": True, "moe_config": _moe_config()}, "use_compile"),
        ({"moe_config": None}, "requires moe_config"),
        ({"moe_config": _moe_config(type_embedding_dim=0)}, "type_embedding"),
        ({"n_focus": 3, "moe_config": _moe_config(topk=1)}, "n_focus"),
        ({"moe_config": _moe_config(routing_input="wrong")}, "routing_input"),
    ]
    for kwargs, message in cases:
        with pytest.raises(ValueError, match=message):
            _make_so2(use_moe=True, **kwargs)


def test_use_moe_true_forward_shape() -> None:
    module = _make_so2(use_moe=True, moe_config=_moe_config())
    x, cache, radial_feat = _inputs(module)
    type_embedding = _type_embedding(module)

    out = module(x, cache, radial_feat, type_embedding=type_embedding)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_use_moe_true_requires_type_embedding() -> None:
    module = _make_so2(use_moe=True, moe_config=_moe_config())
    x, cache, radial_feat = _inputs(module)

    with pytest.raises(ValueError, match="type_embedding"):
        module(x, cache, radial_feat)


@pytest.mark.parametrize("routing_input", ["dst", "src", "src+dst"])
def test_use_moe_true_routing_input_variants(routing_input: str) -> None:
    module = _make_so2(
        use_moe=True,
        moe_config=_moe_config(routing_input=routing_input),
    )
    x, cache, radial_feat = _inputs(module)
    type_embedding = _type_embedding(module)

    out = module(x, cache, radial_feat, type_embedding=type_embedding)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_use_moe_true_backward() -> None:
    module = _make_so2(use_moe=True, moe_config=_moe_config())
    _randomize_parameters(module)
    x, cache, radial_feat = _inputs(module)
    type_embedding = _type_embedding(module, requires_grad=True)

    loss = module(x, cache, radial_feat, type_embedding=type_embedding).square().sum()
    loss.backward()

    assert module.moe_conv.router.gate.matrix.grad is not None
    assert module.moe_conv.router.gate.matrix.grad.abs().sum().item() > 0.0
    for name, param in module.moe_conv.named_parameters():
        if ".routing_matrix" in name or ".shared_matrix" in name:
            assert param.grad is not None, name
            assert param.grad.abs().sum().item() > 0.0, name


def test_use_moe_true_second_backward() -> None:
    module = _make_so2(use_moe=True, moe_config=_moe_config())
    _randomize_parameters(module)
    x, cache, radial_feat = _inputs(module)
    type_embedding = _type_embedding(module, requires_grad=True)

    loss = module(x, cache, radial_feat, type_embedding=type_embedding).square().sum()
    grad_x, grad_type = torch.autograd.grad(
        loss,
        (x, type_embedding),
        create_graph=True,
    )
    (grad_x.sum() + grad_type.sum()).backward()

    assert module.moe_conv.router.gate.matrix.grad is not None
    for name, param in module.moe_conv.named_parameters():
        if ".routing_matrix" in name or ".shared_matrix" in name:
            assert param.grad is not None, name


def test_use_moe_true_mlp_bias_smoke() -> None:
    module = _make_so2(use_moe=True, moe_config=_moe_config(), mlp_bias=True)
    x, cache, radial_feat = _inputs(module)
    type_embedding = _type_embedding(module)

    out = module(x, cache, radial_feat, type_embedding=type_embedding)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()
