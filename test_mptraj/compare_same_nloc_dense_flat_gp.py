#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: TID253
"""Compare dense, full-flat mixed, and GP-flat mixed paths on same-nloc data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from compare_same_nloc_pkl_ep import (
    barrier,
    batch_to_device,
    compare_predictions,
    graph_config_from_model,
    load_config,
    load_or_generate_fixture,
    rank,
    run_forward,
)
from compare_same_nloc_pkl_gp import (
    init_ep_gp,
    run_flat_forward,
    sync_model_parameters_for_compare,
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
from deepmd.pt.utils.graph_parallel import (
    clear_graph_parallel_context,
    set_graph_parallel_context,
)
from deepmd.pt.utils.graph_parallel_flat import (
    build_flat_graph_partition,
)
from deepmd.pt.utils.lmdb_dataset import (
    _collate_lmdb_batch,
    _collate_lmdb_mixed_batch,
)
from deepmd.pt.utils.moe_checkpoint import (
    moe_load_state_dict_from_global,
)
from deepmd.pt.utils.nlist import (
    build_precomputed_flat_graph,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build/read a same-nloc pkl fixture and compare three equivalent "
            "MoE 8-card paths: dense non-mixed, full flat mixed, and GP flat mixed."
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
        help="Output/input same-nloc pkl fixture path",
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
        help="Optional EP global checkpoint, e.g. model_lmdb_mixed_5000-5000.pt",
    )
    parser.add_argument(
        "--result-json",
        default="same_nloc_dense_flat_gp_compare_result.json",
        help="Rank0 comparison summary output",
    )
    parser.add_argument("--ep-size", type=int, default=8, help=argparse.SUPPRESS)
    parser.add_argument("--atol", type=float, default=5.0e-5)
    parser.add_argument("--rtol", type=float, default=5.0e-5)
    return parser.parse_args()


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


def all_summary_passed(summary: dict[str, dict[str, Any]]) -> bool:
    return all(bool(item["passed"]) for item in summary.values())


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    pkl_path = Path(args.pkl)
    result_path = Path(args.result_json)

    config = load_config(input_path)
    repflow = config["model"]["descriptor"]["repflow"]
    if not repflow.get("use_moe", False):
        raise RuntimeError("This comparison is intentionally MoE-only.")
    config["training"]["moe_ep_size"] = args.ep_size
    config["training"]["graph_parallel_size"] = args.ep_size
    use_moe = True
    ep_size_config = args.ep_size
    _ep_group, _dp_group, gp_group, ep_rank, ep_size, dp_rank, dp_size = init_ep_gp(
        ep_size_config,
        use_moe,
    )
    if ep_size != 8 or dist.get_world_size() != 8:
        raise RuntimeError(
            "This comparison must be launched as MoE 8-card: "
            f"ep_size={ep_size}, world_size={dist.get_world_size()}."
        )

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
    sync_model_parameters_for_compare(wrapper, use_moe)
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

    clear_graph_parallel_context()
    dense_pred = run_forward(wrapper, dense_batch, flat=False)
    full_flat_pred = run_flat_forward(wrapper, flat_batch, use_moe_gp=use_moe)

    set_graph_parallel_context(
        True,
        gp_group,
        ep_rank,
        ep_size,
        reduce_backward=False,
    )
    partition = build_flat_graph_partition(
        int(flat_batch["ptr"][-1].item()),
        flat_batch["edge_index"],
        flat_batch["angle_index"],
        batch=flat_batch["batch"],
        rank=ep_rank,
        world_size=ep_size,
    )
    gp_flat_pred = run_flat_forward(
        wrapper,
        flat_batch,
        flat_graph_partition=partition,
        use_moe_gp=use_moe,
    )
    clear_graph_parallel_context()

    dense_vs_flat = compare_predictions(
        dense_pred,
        full_flat_pred,
        args.atol,
        args.rtol,
    )
    flat_vs_gp = compare_predictions(
        full_flat_pred,
        gp_flat_pred,
        args.atol,
        args.rtol,
    )
    dense_vs_gp = compare_predictions(
        dense_pred,
        gp_flat_pred,
        args.atol,
        args.rtol,
    )
    passed = (
        all_summary_passed(dense_vs_flat)
        and all_summary_passed(flat_vs_gp)
        and all_summary_passed(dense_vs_gp)
    )

    if rank() == 0:
        result = {
            "pkl": str(pkl_path),
            "source_lmdb": fixture["source_lmdb"],
            "nframes": fixture["nframes"],
            "nloc": fixture["nloc"],
            "frame_indices": fixture["frame_indices"],
            "ep_size": ep_size,
            "dp_size": dp_size,
            "gp_size": ep_size,
            "use_moe": use_moe,
            "checkpoint": args.checkpoint,
            "atol": args.atol,
            "rtol": args.rtol,
            "passed": passed,
            "comparisons": {
                "dense_nonmixed_vs_full_flat_mixed": dense_vs_flat,
                "full_flat_mixed_vs_gp_flat_mixed": flat_vs_gp,
                "dense_nonmixed_vs_gp_flat_mixed": dense_vs_gp,
            },
        }
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        sys.stdout.write(json.dumps(result, indent=2) + "\n")
        sys.stdout.flush()

    barrier()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
