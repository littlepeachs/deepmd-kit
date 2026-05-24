# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for DescrptSeZM MoE top-level configuration."""

from __future__ import (
    annotations,
)

import pytest
import torch

from deepmd.pt.model.descriptor.sezm import (
    DescrptSeZM,
)
from deepmd.pt.utils import (
    env,
)


def _tiny_system() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = env.DEVICE
    coord = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]],
        dtype=torch.float32,
        device=device,
    )
    atype = torch.tensor([[0, 1]], dtype=torch.int32, device=device)
    nlist = torch.tensor([[[1, 1], [0, 0]]], dtype=torch.int64, device=device)
    return coord.reshape(1, -1), atype, nlist


def _descriptor_kwargs(**overrides) -> dict:
    kwargs = {
        "rcut": 3.0,
        "sel": [1, 1],
        "ntypes": 2,
        "l_schedule": [1, 0],
        "channels": 4,
        "n_radial": 3,
        "radial_mlp": [6],
        "ffn_neurons": 8,
        "ffn_blocks": 1,
        "random_gamma": False,
        "n_atten_head": 0,
        "mlp_bias": False,
        "precision": "float32",
        "trainable": True,
        "seed": 20260524,
    }
    kwargs.update(overrides)
    return kwargs


def _moe_kwargs(**overrides) -> dict:
    kwargs = _descriptor_kwargs(
        use_moe=True,
        n_focus=3,
        n_routing_experts=4,
        topk=2,
        n_shared_experts=1,
        ep_size=1,
        routing_input="dst",
        so2_norm=False,
        so2_attn_res="none",
        use_compile=False,
    )
    kwargs.update(overrides)
    return kwargs


def test_descrpt_use_moe_false_unchanged() -> None:
    base = DescrptSeZM(**_descriptor_kwargs(seed=123))
    explicit = DescrptSeZM(**_descriptor_kwargs(seed=123, use_moe=False))
    coord, atype, nlist = _tiny_system()
    coord_a = coord.detach().clone().requires_grad_(True)
    coord_b = coord.detach().clone().requires_grad_(True)

    out_a = base(coord_a, atype, nlist, mapping=None, comm_dict=None)[0]
    out_b = explicit(coord_b, atype, nlist, mapping=None, comm_dict=None)[0]

    torch.testing.assert_close(out_a, out_b, atol=0.0, rtol=0.0)


def test_descrpt_use_moe_invalid_config_raises() -> None:
    cases = [
        ({"use_compile": True}, "use_compile=False"),
        ({"so2_norm": True}, "so2_norm=False"),
        ({"so2_attn_res": "independent"}, "so2_attn_res='none'"),
        ({"n_focus": 2}, "n_focus"),
        ({"topk": 0}, "topk"),
        ({"n_routing_experts": 1}, "n_routing_experts"),
        ({"n_shared_experts": -1}, "n_shared_experts"),
        ({"ep_size": 3}, "divisible"),
        ({"routing_input": "bad"}, "routing_input"),
    ]
    for overrides, message in cases:
        with pytest.raises(ValueError, match=message):
            DescrptSeZM(**_moe_kwargs(**overrides))


def test_descrpt_use_moe_forward_shape() -> None:
    model = DescrptSeZM(**_moe_kwargs())
    coord, atype, nlist = _tiny_system()

    desc, *_ = model(coord, atype, nlist, mapping=None, comm_dict=None)

    assert desc.shape == (1, 2, model.channels)
    assert torch.isfinite(desc).all()


def test_descrpt_blocks_construct_moe() -> None:
    model = DescrptSeZM(**_moe_kwargs())

    assert model.moe_config is not None
    assert model.moe_ep_group is None
    assert model.moe_dp_group is None
    for block in model.blocks:
        assert block.so2_conv.use_moe
        assert block.so2_conv.moe_conv is not None
        assert block.so2_conv.moe_conv.routing_key_dim == model.channels


def test_descrpt_moe_param_names_for_sync() -> None:
    model = DescrptSeZM(**_moe_kwargs())
    names = [name for name, _ in model.named_parameters()]

    assert any(".routing_matrix" in name for name in names)
    assert any(".shared_matrix" in name for name in names)
    assert all(
        ".routing_matrix" not in name for name in names if ".shared_matrix" in name
    )


def test_descrpt_moe_serialize_roundtrip() -> None:
    model = DescrptSeZM(**_moe_kwargs())
    data = model.serialize()

    cfg = data["config"]
    assert cfg["use_moe"] is True
    assert cfg["n_routing_experts"] == 4
    assert cfg["topk"] == 2
    assert cfg["n_shared_experts"] == 1
    assert cfg["ep_size"] == 1
    assert cfg["routing_input"] == "dst"

    restored = DescrptSeZM.deserialize(data)
    assert restored.use_moe is True
    assert restored.moe_config == model.moe_config
