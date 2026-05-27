#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: TID253
"""Profile MoE 8-card dense/full-flat/GP-flat memory on same-nloc data."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from compare_same_nloc_pkl_ep import (
    batch_to_device,
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
from deepmd.pt.utils.nlist import (
    build_precomputed_flat_graph,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile same-nloc MoE 8-card forward memory/time for dense "
            "non-mixed, full flat mixed, and GP flat mixed paths."
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
        default="same_nloc_profile.pkl",
        help="Output/input same-nloc pkl fixture path",
    )
    parser.add_argument("--nframes", type=int, default=1)
    parser.add_argument("--nloc", type=int, default=148)
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate the pkl fixture from LMDB before profiling.",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument(
        "--result-json",
        default="same_nloc_moe8_gp_memory_profile.json",
        help="Rank0 profile summary output",
    )
    return parser.parse_args()


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def profile_forward(
    name: str,
    fn: Any,
    *,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    barrier()
    torch.cuda.empty_cache()
    for _ in range(warmup):
        fn()
    cuda_sync()
    barrier()
    torch.cuda.empty_cache()
    cuda_sync()

    baseline_alloc = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    cuda_sync()
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / max(repeat, 1)
    peak_alloc = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    current_alloc = torch.cuda.memory_allocated()
    current_reserved = torch.cuda.memory_reserved()
    local = {
        "rank": rank(),
        "path": name,
        "baseline_allocated_mb": baseline_alloc / 1024**2,
        "baseline_reserved_mb": baseline_reserved / 1024**2,
        "peak_allocated_mb": peak_alloc / 1024**2,
        "peak_reserved_mb": peak_reserved / 1024**2,
        "extra_peak_allocated_mb": max(peak_alloc - baseline_alloc, 0) / 1024**2,
        "extra_peak_reserved_mb": max(peak_reserved - baseline_reserved, 0) / 1024**2,
        "current_allocated_mb": current_alloc / 1024**2,
        "current_reserved_mb": current_reserved / 1024**2,
        "avg_forward_ms": elapsed_ms,
    }
    gathered: list[dict[str, Any]] | None = [None] * dist.get_world_size()
    dist.gather_object(local, gathered if rank() == 0 else None, dst=0)
    barrier()
    if rank() == 0:
        assert gathered is not None
        return {"per_rank": gathered}
    return {}


def mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def add_path_summary(profile: dict[str, Any]) -> None:
    ranks = profile["per_rank"]
    profile["summary"] = {
        "mean_peak_allocated_mb": mean([x["peak_allocated_mb"] for x in ranks]),
        "max_peak_allocated_mb": max(x["peak_allocated_mb"] for x in ranks),
        "mean_extra_peak_allocated_mb": mean(
            [x["extra_peak_allocated_mb"] for x in ranks]
        ),
        "max_extra_peak_allocated_mb": max(x["extra_peak_allocated_mb"] for x in ranks),
        "mean_peak_reserved_mb": mean([x["peak_reserved_mb"] for x in ranks]),
        "max_peak_reserved_mb": max(x["peak_reserved_mb"] for x in ranks),
        "mean_forward_ms": mean([x["avg_forward_ms"] for x in ranks]),
        "max_forward_ms": max(x["avg_forward_ms"] for x in ranks),
    }


def safe_ratio(num: float, den: float) -> float | None:
    if den == 0:
        return None
    return num / den


def build_ratio_summary(
    full_flat: dict[str, Any],
    gp_flat: dict[str, Any],
) -> dict[str, Any]:
    full_by_rank = {int(item["rank"]): item for item in full_flat["per_rank"]}
    gp_by_rank = {int(item["rank"]): item for item in gp_flat["per_rank"]}
    rows = []
    for rnk in sorted(full_by_rank):
        full = full_by_rank[rnk]
        gp = gp_by_rank[rnk]
        rows.append(
            {
                "rank": rnk,
                "peak_allocated_full_over_gp": safe_ratio(
                    full["peak_allocated_mb"],
                    gp["peak_allocated_mb"],
                ),
                "extra_peak_allocated_full_over_gp": safe_ratio(
                    full["extra_peak_allocated_mb"],
                    gp["extra_peak_allocated_mb"],
                ),
                "forward_ms_full_over_gp": safe_ratio(
                    full["avg_forward_ms"],
                    gp["avg_forward_ms"],
                ),
            }
        )
    extra_ratios = [
        item["extra_peak_allocated_full_over_gp"]
        for item in rows
        if item["extra_peak_allocated_full_over_gp"] is not None
    ]
    peak_ratios = [
        item["peak_allocated_full_over_gp"]
        for item in rows
        if item["peak_allocated_full_over_gp"] is not None
    ]
    time_ratios = [
        item["forward_ms_full_over_gp"]
        for item in rows
        if item["forward_ms_full_over_gp"] is not None
    ]
    return {
        "per_rank": rows,
        "mean_peak_allocated_full_over_gp": mean(peak_ratios),
        "mean_extra_peak_allocated_full_over_gp": mean(extra_ratios),
        "mean_forward_ms_full_over_gp": mean(time_ratios),
        "interpretation": (
            "Ratios are full_flat / gp_flat. Values above 1 mean GP used less "
            "memory or less time for this same global batch."
        ),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.input)
    repflow = config["model"]["descriptor"]["repflow"]
    if not repflow.get("use_moe", False):
        raise RuntimeError("This profiler is intentionally MoE-only.")
    config["training"]["moe_ep_size"] = 8
    config["training"]["graph_parallel_size"] = 8

    _ep_group, _dp_group, gp_group, ep_rank, ep_size, dp_rank, dp_size = init_ep_gp(
        8,
        True,
    )
    if ep_size != 8 or dist.get_world_size() != 8:
        raise RuntimeError("Launch with torchrun --nproc_per_node=8.")

    fixture = load_or_generate_fixture(
        Path(args.pkl),
        args.lmdb,
        config["model"]["type_map"],
        args.nframes,
        args.nloc,
        args.regenerate,
    )

    model = get_model_for_wrapper(config["model"], _loss_params=config["loss"])
    wrapper = ModelWrapper(model).to(env.DEVICE)
    sync_model_parameters_for_compare(wrapper, True)
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
    partition = build_flat_graph_partition(
        int(flat_batch["ptr"][-1].item()),
        flat_batch["edge_index"],
        flat_batch["angle_index"],
        batch=flat_batch["batch"],
        rank=ep_rank,
        world_size=ep_size,
    )

    def dense_forward() -> None:
        clear_graph_parallel_context()
        run_forward(wrapper, dense_batch, flat=False)

    def full_flat_forward() -> None:
        clear_graph_parallel_context()
        run_flat_forward(wrapper, flat_batch, use_moe_gp=True)

    def gp_flat_forward() -> None:
        set_graph_parallel_context(
            True,
            gp_group,
            ep_rank,
            ep_size,
            reduce_backward=False,
        )
        run_flat_forward(
            wrapper,
            flat_batch,
            flat_graph_partition=partition,
            use_moe_gp=True,
        )
        clear_graph_parallel_context()

    profiles = {
        "dense_nonmixed": profile_forward(
            "dense_nonmixed",
            dense_forward,
            warmup=args.warmup,
            repeat=args.repeat,
        ),
        "full_flat_mixed": profile_forward(
            "full_flat_mixed",
            full_flat_forward,
            warmup=args.warmup,
            repeat=args.repeat,
        ),
        "gp_flat_mixed": profile_forward(
            "gp_flat_mixed",
            gp_flat_forward,
            warmup=args.warmup,
            repeat=args.repeat,
        ),
    }

    if rank() == 0:
        for profile in profiles.values():
            add_path_summary(profile)
        result = {
            "pkl": str(args.pkl),
            "source_lmdb": fixture["source_lmdb"],
            "nframes": fixture["nframes"],
            "nloc": fixture["nloc"],
            "total_atoms": int(fixture["nframes"]) * int(fixture["nloc"]),
            "frame_indices": fixture["frame_indices"],
            "ep_size": ep_size,
            "dp_size": dp_size,
            "gp_size": ep_size,
            "use_moe": True,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "profiles": profiles,
            "full_flat_vs_gp_ratio": build_ratio_summary(
                profiles["full_flat_mixed"],
                profiles["gp_flat_mixed"],
            ),
        }
        result_path = Path(args.result_json)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        sys.stdout.write(json.dumps(result, indent=2) + "\n")
        sys.stdout.flush()

    barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
