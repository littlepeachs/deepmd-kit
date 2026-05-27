#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: TID253
"""Compare dense same-nloc and flat mixed-batch EP forwards on fixed pkl data."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from deepmd.common import (
    j_loader,
)
from deepmd.dpmodel.utils.lmdb_data import (
    LmdbDataReader,
)
from deepmd.pt.train.training import (
    get_model_for_wrapper,
)
from deepmd.pt.train.wrapper import (
    ModelWrapper,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.lmdb_dataset import (
    _collate_lmdb_batch,
    _collate_lmdb_mixed_batch,
)
from deepmd.pt.utils.moe_checkpoint import (
    moe_load_state_dict_from_global,
)
from deepmd.pt.utils.moe_context import (
    set_moe_ep_context,
)
from deepmd.pt.utils.moe_ep_dp import (
    init_ep_dp_groups,
)
from deepmd.pt.utils.nlist import (
    build_precomputed_flat_graph,
)
from deepmd.utils.argcheck import (
    normalize,
)
from deepmd.utils.compat import (
    update_deepmd_input,
)

FLAT_GRAPH_INPUT_KEYS = (
    "extended_atype",
    "extended_batch",
    "extended_image",
    "extended_ptr",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build/read a same-nloc pkl fixture and compare the normal dense "
            "same-nloc path with the flat mixed-batch path under MoE EP."
        )
    )
    parser.add_argument("--input", default="input.json", help="DeepMD input json")
    parser.add_argument(
        "--lmdb",
        default="/aisi-nas/liwentao/mptraj_v024.lmdb",
        help="Source LMDB used only to generate the pkl fixture",
    )
    parser.add_argument(
        "--pkl",
        default="same_nloc_ep_compare.pkl",
        help="Output/input pkl fixture path",
    )
    parser.add_argument("--nframes", type=int, default=2)
    parser.add_argument(
        "--nloc",
        type=int,
        default=4,
        help="Atom count per frame. If unavailable, use the smallest valid group.",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate the pkl fixture from LMDB before comparing.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional EP global checkpoint, e.g. model_lmdb_mixed-1.pt",
    )
    parser.add_argument(
        "--result-json",
        default="same_nloc_ep_compare_result.json",
        help="Rank0 comparison summary output",
    )
    parser.add_argument("--atol", type=float, default=5.0e-5)
    parser.add_argument("--rtol", type=float, default=5.0e-5)
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    config = j_loader(str(path))
    config = update_deepmd_input(config, warning=False, dump=None)
    return normalize(config, multi_task="model_dict" in config.get("model", {}))


def init_distributed(ep_size: int) -> tuple[object | None, int, int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(env.LOCAL_RANK)
    ep_group, _dp_group, ep_rank, ep_size, dp_rank, dp_size = init_ep_dp_groups(ep_size)
    set_moe_ep_context(ep_group, ep_rank, ep_size)
    return ep_group, ep_rank, ep_size, dp_rank, dp_size


def rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def generate_same_nloc_pkl(
    pkl_path: Path,
    lmdb_path: str,
    type_map: list[str],
    nframes: int,
    requested_nloc: int | None,
) -> dict[str, Any]:
    reader = LmdbDataReader(
        lmdb_path,
        type_map,
        batch_size=nframes,
        mixed_batch=False,
    )
    groups = reader.nloc_groups
    if requested_nloc in groups and len(groups[requested_nloc]) >= nframes:
        nloc = int(requested_nloc)
    else:
        valid = [
            (int(group_nloc), len(indices))
            for group_nloc, indices in groups.items()
            if len(indices) >= nframes
        ]
        if not valid:
            raise RuntimeError(f"No nloc group has at least {nframes} frames")
        nloc = min(valid)[0]

    frame_indices = list(groups[nloc][:nframes])
    frames = [reader[index] for index in frame_indices]
    fixture = {
        "format_version": 1,
        "source_lmdb": str(lmdb_path),
        "nloc": nloc,
        "nframes": nframes,
        "frame_indices": frame_indices,
        "equivalence": (
            "All frames have the same nloc. The dense path stacks tensors as "
            "[nframes, nloc, ...], while the mixed path concatenates atom-wise "
            "fields as [nframes * nloc, ...] with ptr=[0, nloc, ...]."
        ),
        "frames": frames,
    }
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with pkl_path.open("wb") as fp:
        pickle.dump(fixture, fp, protocol=pickle.HIGHEST_PROTOCOL)
    return fixture


def load_or_generate_fixture(
    pkl_path: Path,
    lmdb_path: str,
    type_map: list[str],
    nframes: int,
    nloc: int | None,
    regenerate: bool,
) -> dict[str, Any]:
    if rank() == 0 and (regenerate or not pkl_path.exists()):
        fixture = generate_same_nloc_pkl(
            pkl_path,
            lmdb_path,
            type_map,
            nframes,
            nloc,
        )
        sys.stdout.write(
            json.dumps(
                {
                    "generated_pkl": str(pkl_path),
                    "source_lmdb": fixture["source_lmdb"],
                    "nframes": fixture["nframes"],
                    "nloc": fixture["nloc"],
                    "frame_indices": fixture["frame_indices"],
                },
                indent=2,
            )
            + "\n"
        )
        sys.stdout.flush()
    barrier()
    with pkl_path.open("rb") as fp:
        fixture = pickle.load(fp)
    if len({len(frame["atype"]) for frame in fixture["frames"]}) != 1:
        raise RuntimeError("The pkl fixture is not a same-nloc fixture")
    return fixture


def to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    return value


def batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: to_device(value, device) for key, value in batch.items()}


def graph_config_from_model(wrapper: ModelWrapper) -> dict[str, Any]:
    model = wrapper.model["Default"]
    descriptor = model.atomic_model.descriptor
    if not hasattr(descriptor, "repflows"):
        raise RuntimeError("This comparison requires a DPA3/RepFlow descriptor")
    return {
        "rcut": descriptor.get_rcut(),
        "sel": descriptor.get_sel(),
        "a_rcut": descriptor.repflows.a_rcut,
        "a_sel": descriptor.repflows.a_sel,
        "mixed_types": descriptor.mixed_types(),
    }


def get_experts_per_gpu(wrapper: ModelWrapper) -> int:
    for key, tensor in wrapper.state_dict().items():
        if key.endswith(".routing_matrix") or key.endswith(".routing_bias"):
            return int(tensor.shape[-1])
    return 1


def load_checkpoint_if_requested(
    wrapper: ModelWrapper,
    checkpoint: str | None,
    ep_rank: int,
    ep_size: int,
) -> None:
    if checkpoint is None:
        return
    state = torch.load(checkpoint, map_location=env.DEVICE, weights_only=True)
    if "model" in state:
        state = state["model"]
    moe_load_state_dict_from_global(
        wrapper,
        state,
        ep_rank=ep_rank,
        ep_size=ep_size,
        experts_per_gpu=get_experts_per_gpu(wrapper),
    )


def run_forward(
    wrapper: ModelWrapper,
    batch: dict[str, Any],
    flat: bool,
) -> dict[str, torch.Tensor]:
    kwargs = {
        "coord": batch["coord"],
        "atype": batch["atype"],
        "box": batch.get("box"),
        "inference_only": True,
    }
    if flat:
        kwargs["batch"] = batch["batch"]
        kwargs["ptr"] = batch["ptr"]
        for key in FLAT_GRAPH_INPUT_KEYS:
            kwargs[key] = batch.get(key)
    pred, _, _ = wrapper(**kwargs)
    return {key: value.detach() for key, value in pred.items()}


def normalize_output_shape(
    key: str,
    dense_value: torch.Tensor,
    flat_value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        key in {"atom_energy", "force", "mask"}
        and dense_value.shape != flat_value.shape
    ):
        dense_value = dense_value.reshape(flat_value.shape)
    return dense_value, flat_value


def compare_predictions(
    dense_pred: dict[str, torch.Tensor],
    flat_pred: dict[str, torch.Tensor],
    atol: float,
    rtol: float,
) -> dict[str, dict[str, float | bool | list[int]]]:
    summary: dict[str, dict[str, float | bool | list[int]]] = {}
    compare_keys = ["energy", "atom_energy", "force", "virial", "mask"]
    for key in compare_keys:
        if key not in dense_pred or key not in flat_pred:
            continue
        dense_value, flat_value = normalize_output_shape(
            key,
            dense_pred[key],
            flat_pred[key],
        )
        diff = dense_value - flat_value
        max_abs = torch.max(torch.abs(diff))
        scale = torch.max(torch.maximum(torch.abs(dense_value), torch.abs(flat_value)))
        max_abs_global = max_abs.clone()
        scale_global = scale.clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(max_abs_global, op=dist.ReduceOp.MAX)
            dist.all_reduce(scale_global, op=dist.ReduceOp.MAX)
        threshold = atol + rtol * float(scale_global.item())
        summary[key] = {
            "dense_shape": list(dense_value.shape),
            "flat_shape": list(flat_value.shape),
            "max_abs": float(max_abs_global.item()),
            "scale": float(scale_global.item()),
            "threshold": threshold,
            "passed": float(max_abs_global.item()) <= threshold,
        }
    return summary


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    pkl_path = Path(args.pkl)
    result_path = Path(args.result_json)

    config = load_config(input_path)
    ep_size_config = int(config["training"].get("moe_ep_size", 1))
    _ep_group, ep_rank, ep_size, dp_rank, dp_size = init_distributed(ep_size_config)

    fixture = load_or_generate_fixture(
        pkl_path,
        args.lmdb,
        config["model"]["type_map"],
        args.nframes,
        args.nloc,
        args.regenerate,
    )

    model = get_model_for_wrapper(config["model"], _loss_params=config["loss"])
    wrapper = ModelWrapper(model).to(env.DEVICE)
    load_checkpoint_if_requested(wrapper, args.checkpoint, ep_rank, ep_size)
    wrapper.eval()

    graph_config = graph_config_from_model(wrapper)
    frames = fixture["frames"]
    dense_batch = batch_to_device(_collate_lmdb_batch(frames), env.DEVICE)
    flat_batch = batch_to_device(_collate_lmdb_mixed_batch(frames), env.DEVICE)
    flat_batch.update(
        build_precomputed_flat_graph(
            flat_batch["coord"],
            flat_batch["atype"],
            flat_batch["batch"],
            flat_batch["ptr"],
            graph_config["rcut"],
            graph_config["sel"],
            graph_config["a_rcut"],
            graph_config["a_sel"],
            mixed_types=graph_config["mixed_types"],
            box=flat_batch.get("box"),
        )
    )

    dense_pred = run_forward(wrapper, dense_batch, flat=False)
    flat_pred = run_forward(wrapper, flat_batch, flat=True)
    summary = compare_predictions(dense_pred, flat_pred, args.atol, args.rtol)
    all_passed = all(item["passed"] for item in summary.values())

    if rank() == 0:
        result = {
            "pkl": str(pkl_path),
            "source_lmdb": fixture["source_lmdb"],
            "nframes": fixture["nframes"],
            "nloc": fixture["nloc"],
            "frame_indices": fixture["frame_indices"],
            "ep_size": ep_size,
            "dp_size": dp_size,
            "checkpoint": args.checkpoint,
            "atol": args.atol,
            "rtol": args.rtol,
            "passed": all_passed,
            "metrics": summary,
        }
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        sys.stdout.write(json.dumps(result, indent=2) + "\n")
        sys.stdout.flush()

    barrier()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
