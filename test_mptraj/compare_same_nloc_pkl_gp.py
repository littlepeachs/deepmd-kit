#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: TID253
"""Compare full-flat and graph-parallel flat forwards on fixed pkl data."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from compare_same_nloc_pkl_ep import (
    FLAT_GRAPH_INPUT_KEYS,
    barrier,
    batch_to_device,
    compare_predictions,
    get_experts_per_gpu,
    graph_config_from_model,
    load_config,
    load_or_generate_fixture,
    rank,
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
from deepmd.pt.utils.collective_order import (
    collective_ordering,
)
from deepmd.pt.utils.graph_parallel import (
    clear_graph_parallel_context,
    set_graph_parallel_context,
)
from deepmd.pt.utils.graph_parallel_flat import (
    build_flat_graph_partition,
)
from deepmd.pt.utils.lmdb_dataset import (
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build/read a same-nloc pkl fixture and compare full flat forward "
            "with graph-parallel flat forward under the same model."
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
        default="same_nloc_gp_compare_result.json",
        help="Rank0 comparison summary output",
    )
    parser.add_argument(
        "--disable-moe",
        action="store_true",
        help="Set model.descriptor.repflow.use_moe=false before model creation.",
    )
    parser.add_argument(
        "--ep-size",
        type=int,
        default=None,
        help="Override training.moe_ep_size/graph_parallel_size for this check.",
    )
    parser.add_argument("--atol", type=float, default=5.0e-5)
    parser.add_argument("--rtol", type=float, default=5.0e-5)
    parser.add_argument(
        "--check-gradients",
        action="store_true",
        help=(
            "Also compare non-routing parameter gradients after a small "
            "energy/force/virial scalar loss."
        ),
    )
    parser.add_argument("--grad-atol", type=float, default=5.0e-4)
    parser.add_argument("--grad-rtol", type=float, default=5.0e-4)
    return parser.parse_args()


def init_ep_gp(
    ep_size_config: int,
    use_moe: bool,
) -> tuple[object | None, object | None, object | None, int, int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("GP comparison requires torchrun distributed launch.")
    if torch.cuda.is_available():
        torch.cuda.set_device(env.LOCAL_RANK)

    world_size = dist.get_world_size()
    if ep_size_config != world_size:
        raise RuntimeError(
            "This GP comparison expects ep_size == world_size. "
            f"Got ep_size={ep_size_config}, world_size={world_size}."
        )
    ep_group, dp_group, ep_rank, ep_size, dp_rank, dp_size = init_ep_dp_groups(
        ep_size_config
    )
    if use_moe:
        set_moe_ep_context(ep_group, ep_rank, ep_size)
    gp_ranks = [dist.get_rank() - ep_rank + idx for idx in range(ep_size)]
    gp_group = dist.new_group(gp_ranks)
    return ep_group, dp_group, gp_group, ep_rank, ep_size, dp_rank, dp_size


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


def is_routing_expert_param(name: str) -> bool:
    return (
        ".routing_matrix" in name
        or ".routing_bias" in name
        or ".routing_experts." in name
    )


def sync_model_parameters_for_compare(wrapper: ModelWrapper, use_moe: bool) -> None:
    if not dist.is_available() or not dist.is_initialized():
        return
    for name, param in wrapper.named_parameters():
        if use_moe and is_routing_expert_param(name):
            continue
        dist.broadcast(param.data, src=0)
    for name, buffer in wrapper.named_buffers():
        if use_moe and is_routing_expert_param(name):
            continue
        dist.broadcast(buffer.data, src=0)


def run_flat_forward(
    wrapper: ModelWrapper,
    batch: dict[str, Any],
    *,
    flat_graph_partition: Any | None = None,
    use_moe_gp: bool = False,
    detach: bool = True,
) -> dict[str, torch.Tensor]:
    kwargs = {
        "coord": batch["coord"],
        "atype": batch["atype"],
        "box": batch.get("box"),
        "batch": batch["batch"],
        "ptr": batch["ptr"],
        "inference_only": True,
        "flat_graph_partition": flat_graph_partition,
    }
    for key in FLAT_GRAPH_INPUT_KEYS:
        kwargs[key] = batch.get(key)

    context = collective_ordering() if use_moe_gp else torch.enable_grad()
    with context:
        if use_moe_gp:
            with torch.autograd.set_multithreading_enabled(False):
                pred, _, _ = wrapper(**kwargs)
        else:
            pred, _, _ = wrapper(**kwargs)
    if detach:
        return {key: value.detach() for key, value in pred.items()}
    return pred


def scalar_prediction_loss(pred: dict[str, torch.Tensor]) -> torch.Tensor:
    loss = pred["energy"].square().sum()
    if "force" in pred:
        loss = loss + pred["force"].square().sum() * 0.01
    if "virial" in pred:
        loss = loss + pred["virial"].square().sum() * 0.01
    return loss


def zero_gradients(wrapper: ModelWrapper) -> None:
    for param in wrapper.parameters():
        param.grad = None


def sync_non_moe_gradients(wrapper: ModelWrapper, divisor: float = 1.0) -> None:
    if not dist.is_available() or not dist.is_initialized():
        return
    for param in wrapper.parameters():
        if param.grad is None:
            continue
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad.div_(divisor)


def collect_non_routing_gradients(wrapper: ModelWrapper) -> dict[str, torch.Tensor]:
    gradients = {}
    for name, param in wrapper.named_parameters():
        if is_routing_expert_param(name):
            continue
        if param.grad is not None:
            gradients[name] = param.grad.detach().clone()
    return gradients


def broadcast_reference_gradients(wrapper: ModelWrapper) -> None:
    if not dist.is_available() or not dist.is_initialized():
        return
    for name, param in wrapper.named_parameters():
        if is_routing_expert_param(name):
            continue
        has_grad = torch.tensor(
            [int(param.grad is not None)],
            dtype=torch.int32,
            device=param.device,
        )
        dist.broadcast(has_grad, src=0)
        if has_grad.item() == 0:
            param.grad = None
            continue
        if param.grad is None:
            param.grad = torch.zeros_like(param)
        dist.broadcast(param.grad, src=0)


def run_gradient_check(
    wrapper: ModelWrapper,
    batch: dict[str, Any],
    *,
    flat_graph_partition: Any,
    gp_group: object,
    ep_rank: int,
    ep_size: int,
    dp_group: object | None,
    dp_size: int,
    use_moe: bool,
    grad_atol: float,
    grad_rtol: float,
) -> dict[str, Any]:
    from deepmd.pt.utils.moe_ep_dp import sync_moe_gradients

    clear_graph_parallel_context()
    zero_gradients(wrapper)
    ref_pred = run_flat_forward(
        wrapper,
        batch,
        use_moe_gp=use_moe,
        detach=False,
    )
    ref_loss = scalar_prediction_loss(ref_pred)
    if use_moe:
        with torch.autograd.set_multithreading_enabled(False):
            ref_loss.backward()
        sync_moe_gradients(wrapper, dp_group, None, dp_size, dist.get_world_size())
    else:
        ref_loss.backward()
        broadcast_reference_gradients(wrapper)
    ref_grad = collect_non_routing_gradients(wrapper)

    set_graph_parallel_context(
        True,
        gp_group,
        ep_rank,
        ep_size,
        reduce_backward=False,
    )
    zero_gradients(wrapper)
    gp_pred = run_flat_forward(
        wrapper,
        batch,
        flat_graph_partition=flat_graph_partition,
        use_moe_gp=use_moe,
        detach=False,
    )
    gp_loss = scalar_prediction_loss(gp_pred)
    if use_moe:
        with torch.autograd.set_multithreading_enabled(False):
            gp_loss.backward()
        sync_moe_gradients(
            wrapper,
            dp_group,
            None,
            dp_size,
            dist.get_world_size(),
            non_routing_divisor=1.0,
        )
    else:
        gp_loss.backward()
        sync_non_moe_gradients(wrapper)
    gp_grad = collect_non_routing_gradients(wrapper)
    clear_graph_parallel_context()

    names = sorted(set(ref_grad) | set(gp_grad))
    max_abs = torch.zeros((), device=env.DEVICE)
    scale = torch.ones((), device=env.DEVICE)
    missing = []
    checked = 0
    for name in names:
        ref_value = ref_grad.get(name)
        gp_value = gp_grad.get(name)
        if ref_value is None or gp_value is None:
            missing.append(name)
            continue
        if ref_value.shape != gp_value.shape:
            missing.append(name)
            continue
        checked += 1
        max_abs = torch.maximum(max_abs, torch.max(torch.abs(ref_value - gp_value)))
        scale = torch.maximum(
            scale,
            torch.max(torch.maximum(torch.abs(ref_value), torch.abs(gp_value))),
        )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(max_abs, op=dist.ReduceOp.MAX)
        dist.all_reduce(scale, op=dist.ReduceOp.MAX)
    threshold = grad_atol + grad_rtol * float(scale.item())
    return {
        "checked_non_routing_params": checked,
        "missing_or_shape_mismatch": missing[:20],
        "missing_count": len(missing),
        "max_abs": float(max_abs.item()),
        "scale": float(scale.item()),
        "threshold": threshold,
        "passed": len(missing) == 0 and float(max_abs.item()) <= threshold,
    }


def maybe_disable_moe(config: dict[str, Any], disable_moe: bool) -> bool:
    repflow = config["model"]["descriptor"]["repflow"]
    if disable_moe:
        repflow["use_moe"] = False
    return bool(repflow.get("use_moe", False))


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    pkl_path = Path(args.pkl)
    result_path = Path(args.result_json)

    config = load_config(input_path)
    if args.ep_size is not None:
        config["training"]["moe_ep_size"] = args.ep_size
        config["training"]["graph_parallel_size"] = args.ep_size
    use_moe = maybe_disable_moe(config, args.disable_moe)
    ep_size_config = int(config["training"].get("moe_ep_size", 1))
    _ep_group, dp_group, gp_group, ep_rank, ep_size, dp_rank, dp_size = init_ep_gp(
        ep_size_config, use_moe
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
    ref_pred = run_flat_forward(wrapper, flat_batch)

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
    gp_pred = run_flat_forward(
        wrapper,
        flat_batch,
        flat_graph_partition=partition,
        use_moe_gp=use_moe,
    )
    clear_graph_parallel_context()

    summary = compare_predictions(ref_pred, gp_pred, args.atol, args.rtol)
    gradient_summary = None
    if args.check_gradients:
        gradient_summary = run_gradient_check(
            wrapper,
            flat_batch,
            flat_graph_partition=partition,
            gp_group=gp_group,
            ep_rank=ep_rank,
            ep_size=ep_size,
            dp_group=dp_group,
            dp_size=dp_size,
            use_moe=use_moe,
            grad_atol=args.grad_atol,
            grad_rtol=args.grad_rtol,
        )
    all_passed = all(item["passed"] for item in summary.values())
    if gradient_summary is not None:
        all_passed = all_passed and bool(gradient_summary["passed"])

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
            "passed": all_passed,
            "metrics": summary,
            "gradient_metrics": gradient_summary,
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
