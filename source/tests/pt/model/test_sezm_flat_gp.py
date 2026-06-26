# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for SeZM flat graph-parallel guards and validation."""

from __future__ import (
    annotations,
)

from copy import (
    deepcopy,
)

import pytest
import torch

from deepmd.pt.model.descriptor.sezm import (
    DescrptSeZM,
)
from deepmd.pt.model.model import (
    get_sezm_model,
)
from deepmd.pt.train.training import (
    validate_graph_parallel_config,
    validate_graph_parallel_runtime,
)
from deepmd.pt.utils.graph_parallel_flat import (
    build_flat_graph_partition,
)
from deepmd.pt.utils.nlist import (
    build_precomputed_flat_graph,
)


def _model_params(**overrides) -> dict:
    params = {
        "type": "SeZM",
        "type_map": ["O", "H"],
        "descriptor": {
            "type": "SeZM",
            "sel": [4, 4],
            "rcut": 3.0,
            "ntypes": 2,
            "channels": 4,
            "n_focus": 1,
            "n_radial": 3,
            "radial_mlp": [6],
            "use_env_seed": False,
            "l_schedule": [1, 0],
            "mmax": 1,
            "so2_norm": False,
            "so2_layers": 1,
            "n_atten_head": 0,
            "ffn_neurons": 8,
            "ffn_blocks": 1,
            "mlp_bias": True,
            "layer_scale": False,
            "use_amp": False,
            "activation_function": "silu",
            "glu_activation": True,
            "precision": "float32",
            "seed": 7,
        },
        "fitting_net": {
            "neuron": [8],
            "activation_function": "silu",
            "precision": "float32",
            "seed": 7,
        },
        "use_compile": False,
    }
    for key, value in overrides.items():
        if key == "descriptor":
            params["descriptor"].update(value)
        elif key == "fitting_net":
            params["fitting_net"].update(value)
        else:
            params[key] = value
    return params


def _moe_model_params(**overrides) -> dict:
    params = _model_params(
        descriptor={
            "use_moe": True,
            "n_focus": 3,
            "n_routing_experts": 8,
            "topk": 2,
            "n_shared_experts": 1,
            "ep_size": 2,
            "routing_input": "dst",
            "so2_attn_res": "none",
        }
    )
    for key, value in overrides.items():
        if key == "descriptor":
            params["descriptor"].update(value)
        elif key == "fitting_net":
            params["fitting_net"].update(value)
        else:
            params[key] = value
    return params


def _base_gp_config(**overrides) -> dict:
    config = {
        "model": _moe_model_params(),
        "training": {
            "graph_parallel": True,
            "graph_parallel_size": 2,
        },
    }
    for key, value in overrides.items():
        if key == "model":
            config["model"] = value
        elif key == "training":
            config["training"].update(value)
        else:
            config[key] = value
    return config


def _flat_inputs() -> dict[str, torch.Tensor]:
    with torch.device("cpu"):
        coord = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.1, 0.0],
                [0.2, 1.1, 0.0],
                [0.0, 0.0, 0.0],
                [1.2, 0.2, 0.1],
            ],
            dtype=torch.float32,
        )
        atype = torch.tensor([0, 1, 0, 1, 0], dtype=torch.long)
        batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
        ptr = torch.tensor([0, 3, 5], dtype=torch.long)
        graph = build_precomputed_flat_graph(
            coord,
            atype,
            batch,
            ptr,
            rcut=3.0,
            sel=[4, 4],
            a_rcut=3.0,
            a_sel=4,
            mixed_types=True,
            box=None,
            ntypes=2,
        )
    graph.update(
        {
            "coord": coord,
            "atype": atype,
            "batch": batch,
            "ptr": ptr,
        }
    )
    return graph


def _model_flat_kwargs(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    keys = (
        "coord",
        "atype",
        "batch",
        "ptr",
        "extended_atype",
        "extended_batch",
        "extended_image",
        "mapping",
        "central_ext_index",
        "nlist",
        "nlist_ext",
        "a_nlist",
        "a_nlist_ext",
        "nlist_mask",
        "a_nlist_mask",
        "edge_index",
        "angle_index",
    )
    return {key: inputs[key] for key in keys}


def _partition(inputs: dict[str, torch.Tensor]) -> dict:
    return build_flat_graph_partition(
        int(inputs["ptr"][-1].item()),
        inputs["edge_index"],
        inputs["angle_index"],
        batch=inputs["batch"],
        rank=0,
        world_size=2,
    ).asdict()


def test_validate_graph_parallel_accepts_shared_axis_sezm() -> None:
    assert validate_graph_parallel_config(_base_gp_config()) is True
    lower = _base_gp_config()
    lower["model"]["type"] = "sezm"
    assert validate_graph_parallel_config(lower) is True


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda cfg: cfg["model"].update({"model_dict": {"task": {}}}),
            "single-task",
        ),
        (
            lambda cfg: cfg["model"].update({"type": "dos"}),
            "model.type",
        ),
        (
            lambda cfg: cfg["model"]["descriptor"].update({"type": "dpa2"}),
            "descriptor.type",
        ),
        (
            lambda cfg: cfg["model"]["descriptor"].update({"use_moe": False}),
            "use_moe=True",
        ),
        (
            lambda cfg: cfg["model"]["fitting_net"].update({"type": "property"}),
            "fitting_net.type",
        ),
    ],
)
def test_validate_graph_parallel_rejects_unsupported_configs(
    mutate,
    message: str,
) -> None:
    config = _base_gp_config()
    mutate(config)

    with pytest.raises(ValueError, match=message):
        validate_graph_parallel_config(config)


def test_validate_graph_parallel_runtime_accepts_shared_axis() -> None:
    validate_graph_parallel_runtime(
        use_graph_parallel=True,
        use_moe_ep=True,
        gp_size=4,
        gp_rank=2,
        moe_ep_size=4,
        moe_ep_rank=2,
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"use_moe_ep": False}, "active SeZM MoE EP"),
        ({"moe_ep_size": 2}, "graph_parallel_size == moe_ep_size"),
        ({"moe_ep_rank": 1}, "gp_rank == ep_rank"),
    ],
)
def test_validate_graph_parallel_runtime_rejects_axis_mismatch(
    kwargs: dict,
    message: str,
) -> None:
    params = {
        "use_graph_parallel": True,
        "use_moe_ep": True,
        "gp_size": 4,
        "gp_rank": 2,
        "moe_ep_size": 4,
        "moe_ep_rank": 2,
    }
    params.update(kwargs)

    with pytest.raises(RuntimeError, match=message):
        validate_graph_parallel_runtime(**params)


def test_descriptor_gp_rejects_descriptor_level_attention() -> None:
    with torch.device("cpu"):
        coord = torch.zeros(1, 2, 3, dtype=torch.float32)
        atype = torch.tensor([[0, 1]], dtype=torch.long)
        edge_index = torch.tensor([[1], [0]], dtype=torch.long)
        edge_vec = torch.zeros(1, 3, dtype=torch.float32)
        edge_mask = torch.ones(1, dtype=torch.bool)
        part = build_flat_graph_partition(
            total_atoms=2,
            edge_index=torch.tensor([[0], [1]], dtype=torch.long),
            angle_index=torch.empty(3, 0, dtype=torch.long),
            rank=0,
            world_size=2,
        ).asdict()

    descriptor = DescrptSeZM(
        **_model_params(
            descriptor={
                "full_attn_res": "dependent",
                "n_atten_head": 1,
                "channels": 4,
            }
        )["descriptor"]
    )

    with pytest.raises(NotImplementedError, match="full_attn_res/block_attn_res"):
        descriptor.forward_with_edges(
            extended_coord=coord,
            extended_atype=atype,
            edge_index=edge_index,
            edge_vec=edge_vec,
            edge_mask=edge_mask,
            flat_graph_partition=part,
        )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("fparam", "frame parameters"),
        ("charge_spin", "charge_spin"),
        ("atomic_virial", "Atomic virial"),
        ("force_input", "force_input/noise_mask"),
    ],
)
def test_flat_model_rejects_unsupported_flat_inputs(
    case: str,
    message: str,
) -> None:
    model = get_sezm_model(_model_params())
    inputs = _flat_inputs()
    kwargs = _model_flat_kwargs(inputs)
    with torch.device("cpu"):
        if case == "fparam":
            kwargs["fparam"] = torch.zeros(2, 1)
        elif case == "charge_spin":
            kwargs["charge_spin"] = torch.zeros(2, 2)
        elif case == "atomic_virial":
            kwargs["do_atomic_virial"] = True
        elif case == "force_input":
            kwargs["force_input"] = torch.zeros(5, 3)
            kwargs["noise_mask"] = torch.zeros(5, dtype=torch.bool)
        else:
            raise AssertionError(case)

    with pytest.raises(NotImplementedError, match=message):
        model(**kwargs)


def test_flat_model_rejects_dens_mode() -> None:
    model = get_sezm_model(_model_params())
    model.set_active_mode("dens")
    inputs = _flat_inputs()

    with pytest.raises(NotImplementedError, match="dens mode"):
        model(**_model_flat_kwargs(inputs))


def test_flat_gp_rejects_inter_potential() -> None:
    model = get_sezm_model(_model_params(bridging_method="ZBL"))
    inputs = _flat_inputs()
    kwargs = _model_flat_kwargs(inputs)
    kwargs["flat_graph_partition"] = _partition(inputs)

    with pytest.raises(NotImplementedError, match="inter_potential"):
        model(**kwargs)


def test_flat_model_requires_precomputed_graph_fields() -> None:
    model = get_sezm_model(_model_params())
    inputs = _flat_inputs()
    kwargs = _model_flat_kwargs(inputs)
    kwargs.pop("edge_index")

    with pytest.raises(RuntimeError, match="precomputed flat graph fields"):
        model(**kwargs)


def test_flat_gp_partition_consistency_guard() -> None:
    model = get_sezm_model(_model_params())
    inputs = _flat_inputs()
    bad_part = deepcopy(_partition(inputs))
    bad_part["local_size"] += 1
    kwargs = _model_flat_kwargs(inputs)
    kwargs["flat_graph_partition"] = bad_part

    with pytest.raises(RuntimeError, match="local_size is inconsistent"):
        model(**kwargs)
