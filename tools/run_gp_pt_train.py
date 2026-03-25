#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deepmd.main import main as deepmd_main
from deepmd.utils.local_distutils import get_repo_distutils


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def _rank_selected(rank: int, ranks_spec: str) -> bool:
    spec = ranks_spec.strip().lower()
    if spec in {"", "all", "*"}:
        return True
    try:
        selected = {int(item.strip()) for item in ranks_spec.split(",") if item.strip()}
    except ValueError:
        return False
    return rank in selected


def _setup_rank_mapped_debugpy() -> None:
    if os.environ.get("DP_DEBUGPY_ENABLE", "0") != "1":
        return

    rank_key = os.environ.get("DP_DEBUGPY_RANK_KEY", "RANK").strip().upper()
    if rank_key not in {"RANK", "LOCAL_RANK"}:
        rank_key = "RANK"

    rank = int(os.environ.get(rank_key, os.environ.get("RANK", "0")))
    ranks_spec = os.environ.get("DP_DEBUGPY_RANKS", "all")
    if not _rank_selected(rank, ranks_spec):
        return

    host = os.environ.get("DP_DEBUGPY_HOST", "127.0.0.1")
    base_port = int(os.environ.get("DP_DEBUGPY_BASE_PORT", "5678"))
    port_stride = int(os.environ.get("DP_DEBUGPY_PORT_STRIDE", "1"))
    port = base_port + rank * port_stride
    wait_for_client = os.environ.get("DP_DEBUGPY_WAIT_FOR_CLIENT", "0") == "1"

    try:
        import debugpy
    except ImportError:
        print("[debugpy] debugpy not installed, skip attach setup.", flush=True)
        return

    debugpy.listen((host, port))
    print(
        f"[debugpy] pid={os.getpid()} {rank_key}={rank} listening on {host}:{port}",
        flush=True,
    )
    if wait_for_client:
        print(f"[debugpy] waiting for client on {host}:{port}", flush=True)
        debugpy.wait_for_client()


def _maybe_setup_distributed_gp() -> None:
    if os.environ.get("DISABLE_GP_MODE", "0") == "1":
        return

    if os.environ.get("DP_ENABLE_DISTUTILS_GP", "1") != "1":
        return

    world_size = _env_int("WORLD_SIZE", 1)
    if world_size <= 1:
        return

    exec_mode = os.environ.get("DP_GP_EXEC_MODE", "auto").strip().lower()
    if exec_mode in {"single-process-loop", "single_process_loop", "loop"}:
        return

    gp_distutils = get_repo_distutils()

    if gp_distutils.initialized():
        return

    gp_size = _env_int("DP_GP_SIZE", world_size)
    dp_size = _env_int("DP_DP_SIZE", 1)
    pp_size = _env_int("DP_PP_SIZE", 1)
    ep_size = _env_int("DP_EP_SIZE", 1)
    has_fsdp = os.environ.get("DP_HAS_FSDP", "0") == "1"

    dp_replica_size_env = os.environ.get("DP_DP_REPLICA_SIZE")
    dp_shard_size_env = os.environ.get("DP_DP_SHARD_SIZE")
    dp_replica_size = int(dp_replica_size_env) if dp_replica_size_env else None
    dp_shard_size = int(dp_shard_size_env) if dp_shard_size_env else None

    expected_world_size = pp_size * dp_size * ep_size * gp_size
    if expected_world_size != world_size:
        raise SystemExit(
            "Invalid distributed topology: "
            f"PP({pp_size}) * DP({dp_size}) * EP({ep_size}) * GP({gp_size}) "
            f"!= WORLD_SIZE({world_size})"
        )

    gp_distutils.setup(
        dp_size=dp_size,
        dp_replica_size=dp_replica_size,
        dp_shard_size=dp_shard_size,
        pp_size=pp_size,
        ep_size=ep_size,
        gp_size=gp_size,
        has_fsdp=has_fsdp,
    )

    print(
        "[gp-launch] distutils initialized "
        f"(DP={dp_size}, GP={gp_size}, PP={pp_size}, EP={ep_size}, WORLD_SIZE={world_size})",
        flush=True,
    )


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: run_gp_pt_train.py train [args...]\n"
            "Example: run_gp_pt_train.py train --skip-neighbor-stat input_dpa3.json"
        )

    os.environ.setdefault("DISABLE_GP_MODE", "0")
    os.environ.setdefault("DP_PT_DIST_BACKEND", "cuda:gloo,cpu:gloo")
    _maybe_setup_distributed_gp()
    _setup_rank_mapped_debugpy()
    cli_args = ["--pt", *sys.argv[1:]]
    deepmd_main(cli_args)


if __name__ == "__main__":
    main()
