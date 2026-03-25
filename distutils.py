"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any
import os

import numpy as np
import torch
from torch import distributed as dist
from torch.distributed.nn.functional import all_reduce
from torch.distributed.nn.functional import all_to_all_single
import time
from torch.distributed.device_mesh import init_device_mesh
try:
    from torch.distributed.tensor import DeviceMesh
except ImportError:
    from torch.distributed.device_mesh import DeviceMesh
# from deepspeed.utils.timer import SynchronizedWallClockTimer
import json
try:
    import janus.core.distutils as distutils
except ImportError:
    distutils = None
from torch.profiler import record_function
"""
Functions to support graph parallel training.
This is based on the Megatron-LM implementation:
https://github.com/facebookresearch/fairscale/blob/main/fairscale/nn/model_parallel/initialize.py
"""

_GP_GATHER_DEBUG_PRINTED = False

########## INITIALIZATION ##########


# --- 全局变量来存储进程组和 mesh ---
_DP_GROUP = None
_DP_REPLICA_GROUP = None
_DP_SHARD_GROUP = None
_PP_GROUP = None
_GP_GROUP = None
_DATA_GROUP = None # EP + DP or EP+HSDP
# DP+GP 合并组（同一 PP stage 内的所有 DP×GP 进程，用于统一规约）
_DP_GP_GROUP = None # sync expert param
_DP_GP_EP_GROUP = None # sync non expert param
_DP_EP_GROUP = None # partition dataset in datasampler
_PP_STAGES_2_RANKS = None  # 用于流水线并行的stage和rank的映射关系
_DP_SHARD_GP_GROUP = None # sync expert param
_DP_SHARD_EP_GP_GROUP = None # sync non expert param
_DP_REPLICA_EP_DP_SHARD_GROUP = None # partition dataset in datasampler
_DEVICE_MESH: DeviceMesh | None = None  # 原始 DeviceMesh: (pp, dp_replica, dp_shard, ep, gp)
Decouple_FE_FF = False  # 确定是否de couple了FE 和FF
GP_OPT = False  # 确定是否de couple了FE 和FF
GARS = False


def _device_mesh_debug_enabled() -> bool:
    return os.environ.get("MATRIS_DEVICE_MESH_DEBUG", "0") == "1"


def ensure_div(a: int, b: int) -> None:
    assert a % b == 0


def divide_and_check_no_remainder(a: int, b: int) -> int:
    ensure_div(a, b)
    return a // b


def _manual_mesh_group(mesh, vary_dim_names: tuple[str, ...]):
    if not dist.is_initialized():
        return None
    mesh_tensor = mesh.mesh
    dim_names = tuple(mesh.mesh_dim_names)
    vary_dims = tuple(dim_names.index(name) for name in vary_dim_names)
    fixed_dims = tuple(idx for idx in range(len(dim_names)) if idx not in vary_dims)
    current_rank = dist.get_rank()
    selected_group = None
    fixed_ranges = [range(mesh_tensor.shape[idx]) for idx in fixed_dims]

    for fixed_coords in np.ndindex(*[len(rng) for rng in fixed_ranges] or [1]):
        selector = [slice(None)] * mesh_tensor.ndim
        for dim_idx, coord_idx in enumerate(fixed_coords):
            selector[fixed_dims[dim_idx]] = coord_idx
        ranks = mesh_tensor[tuple(selector)].reshape(-1).tolist()
        group = dist.new_group(ranks=ranks)
        if current_rank in ranks:
            selected_group = group

    return selected_group


def setup_dist_group(
    pp_size: int = 1,
    dp_size: int = 1,
    dp_replica_size: int | None = None,
    dp_shard_size: int | None = None,
    ep_size: int = 1,
    gp_size: int = 1,
    *,
    has_fsdp: bool = False,
) -> None:
    """
    使用 DeviceMesh 为 4D/5D 并行 (PP, DP_REPLICA, DP_SHARD, EP, GP) 初始化分布式进程组。

    Args:
        pp_size: 流水线并行的大小。
        dp_size: DP 总大小（dp_replica_size * dp_shard_size）。
        dp_replica_size: DP 的 replica 维度大小；若未指定，则默认 dp_size。
        dp_shard_size: DP 的 shard 维度大小；若未指定，则默认 1
        ep_size: Expert Parallel 维度大小（1 表示未启用 EP）。
        gp_size: 图/张量并行的大小。
    """
    
    global _DP_GROUP, _PP_GROUP, _GP_GROUP, _EP_GROUP
    global _DP_GP_GROUP, _PP_STAGES_2_RANKS
    global _DP_EP_GROUP, _DP_REPLICA_GROUP, _DP_SHARD_GROUP
    global _DP_GP_EP_GROUP
    global _DEVICE_MESH, _DATA_GROUP
    assert dist.is_initialized(), "PyTorch distributed must be initialized first"

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if dp_replica_size is None and dp_shard_size is None:
        # Backward-compatible default: interpret dp_size as "dp_shard_size" (pure sharding).
        dp_replica_size = dp_size
        dp_shard_size = 1
    elif dp_replica_size * dp_shard_size != dp_size:
            raise ValueError(
                f"dp_replica_size({dp_replica_size}) * dp_shard_size({dp_shard_size}) must equal dp_size({dp_size})"
            )

    if pp_size * dp_replica_size * dp_shard_size * ep_size * gp_size != world_size:
        raise ValueError(
            f" pp_size({pp_size}) * dp_replica_size({dp_replica_size}) * dp_shard_size({dp_shard_size}) "
            f"* ep_size({ep_size}) * gp_size({gp_size}) must equal world_size({world_size})"
        )

    if rank == 0:
        logging.info(
            f"> Initializing parallelism with DeviceMesh: "
            f"PP={pp_size}, DP(replica={dp_replica_size}, shard={dp_shard_size}, total={dp_size}), "
            f"EP={ep_size}, GP={gp_size}"
        )

    device_type = "cuda" if torch.cuda.is_available() else "cpu"

    # NOTE: dim order is fixed; we rely on DeviceMesh flatten+slicing (no permute+reshape).
    if has_fsdp:
        mesh = init_device_mesh(
            device_type,
            (pp_size, dp_replica_size, gp_size, ep_size, dp_shard_size),
            mesh_dim_names=("pp", "dp_replica",  "gp", "ep", "dp_shard"),
        )
    else:
        mesh = init_device_mesh(
            device_type,
            (pp_size, dp_size, ep_size, gp_size),
            mesh_dim_names=("pp", "dp", "ep", "gp"),
        )


    _DEVICE_MESH = mesh
    # 流水线并行组 (PP Group)
    _PP_GROUP = mesh.get_group(mesh_dim="pp")
    _PP_STAGES_2_RANKS = dist.get_process_group_ranks(_PP_GROUP)

    # DP 维度：replica vs shard
    if has_fsdp:
        _DP_REPLICA_GROUP = mesh.get_group(mesh_dim="dp_replica")
        _DP_SHARD_GROUP = mesh.get_group(mesh_dim="dp_shard")
    else:
        _DP_GROUP = mesh.get_group(mesh_dim="dp")

    # Expert Parallel 组
    _EP_GROUP = mesh.get_group(mesh_dim="ep")
    # 图/张量并行组 (GP Group)
    _GP_GROUP = mesh.get_group(mesh_dim="gp")


    # NOTE: keep tuples in ascending base-dim index order for
    # mesh_dim_names=("pp","dp_replica","gp","ep","dp_shard").
    if has_fsdp:
        # FSDP: DP_SHARD×GP and DP_SHARD×GP×EP
        # mesh['dp_replica', 'gp']._flatten(mesh_dim_name="dp_replica_gp")
        # mesh['dp_replica', 'gp', 'ep']._flatten(mesh_dim_name="dp_replica_ep_gp")
        mesh[("gp", "dp_shard")]._flatten(mesh_dim_name="dp_shard_gp")
        mesh[("gp","ep", "dp_shard")]._flatten(mesh_dim_name="dp_shard_ep_gp")
        mesh[("dp_replica", "ep", "dp_shard")]._flatten(mesh_dim_name="dp_replica_ep_dp_shard")
        # _DP_SHARD_GP_GROUP = mesh.get_group(mesh_dim="dp_shard_gp")
        # _DP_SHARD_EP_GP_GROUP = mesh.get_group(mesh_dim="dp_shard_ep_gp")
        _DP_REPLICA_EP_DP_SHARD_GROUP = mesh.get_group(mesh_dim="dp_replica_ep_dp_shard")
        _DATA_GROUP = _DP_REPLICA_EP_DP_SHARD_GROUP
    else:
        # No FSDP: DP×GP and DP×GP×EP
        try:
            mesh[("dp", "gp")]._flatten(mesh_dim_name="dp_gp")
            mesh[("dp", "ep", "gp")]._flatten(mesh_dim_name="dp_ep_gp")
            mesh[("dp", "ep")]._flatten(mesh_dim_name="dp_ep")
            _DP_GP_GROUP = mesh.get_group(mesh_dim="dp_gp")
            _DP_GP_EP_GROUP = mesh.get_group(mesh_dim="dp_ep_gp")
            _DATA_GROUP = mesh.get_group(mesh_dim="dp_ep")
        except KeyError:
            # PyTorch 2.3 的 DeviceMesh 不支持 tuple flatten 命名索引时，
            # 手动按 mesh 维度构造需要的复合 group。
            if dp_size == 1 and ep_size == 1:
                _DP_GP_GROUP = _GP_GROUP
                _DP_GP_EP_GROUP = _GP_GROUP
                _DATA_GROUP = _DP_GROUP
            else:
                _DP_GP_GROUP = _manual_mesh_group(mesh, ("dp", "gp"))
                _DP_GP_EP_GROUP = _manual_mesh_group(mesh, ("dp", "ep", "gp"))
                _DATA_GROUP = _manual_mesh_group(mesh, ("dp", "ep"))

    # 日志
    setup_logging()
    if _device_mesh_debug_enabled():
        logging.warning(
            "DeviceMesh::debug mesh_dim_names=%s mesh_shape=%s",
            getattr(mesh, "mesh_dim_names", None),
            tuple(mesh.mesh.shape) if getattr(mesh, "mesh", None) is not None else None,
        )
    logging.info(f"Rank {rank} successfully initialized process groups via DeviceMesh.")


def _safe_group_info(group):
    if group is None or not dist.is_initialized():
        return -1, []
    try:
        return dist.get_world_size(group=group), dist.get_process_group_ranks(group)
    except Exception:
        return -1, []


def setup(
    dp_size: int = 1,
    pp_size: int = 1,
    dp_replica_size: int | None = None,
    dp_shard_size: int | None = None,
    ep_size: int = 1,
    gp_size: int = 1,
    *,
    has_fsdp: bool = False,
) -> None:
    """
    Setup the distributed backend for training with torchrun.
    This function initializes the distributed process group and sets the device
    based on the local rank provided by torchrun environment variables.
    """
    pid = os.getpid()
    if dist.is_initialized():  # 如果已经初始化，直接返回
        logging.warning("Distributed backend already initialized, skipping setup.")
        return
    else:
        # 1. 从环境变量获取 torchrun 设置的 rank 信息
        # torchrun 保证会设置这些环境变量
        local_rank = int(os.environ.get("LOCAL_RANK"))
        global_rank = int(os.environ.get("RANK"))
        world_size_from_env = int(os.environ.get("WORLD_SIZE"))

        if local_rank is None or global_rank is None or world_size_from_env is None:
            logging.error(
                f"[{pid}] Critical environment variables (LOCAL_RANK, RANK, WORLD_SIZE) not set by torchrun. "
                "This indicates a problem with the torchrun launch or environment."
            )
            raise RuntimeError("Torchrun environment variables not found.")

        # 2. 根据 local_rank 分配设备 (非常重要，必须在 init_process_group 之前)
        # 当使用 CUDA_VISIBLE_DEVICES 时，多个进程可能共享同一张 GPU
        # 使用取模操作确保设备索引在有效范围内
        num_visible_devices = torch.cuda.device_count()
        if num_visible_devices > 0:
            device_id = local_rank % num_visible_devices
            torch.cuda.set_device(device_id)
            logging.info(
                f"[{pid}] Local rank {local_rank} mapped to device {device_id} "
                f"(total visible devices: {num_visible_devices})"
            )

        # 3. 准备 init_process_group 参数
        logging.info(
            f"[{pid}] Torchrun setup parameters: local_rank={local_rank}, global_rank={global_rank} (from env), "
            f"world_size={world_size_from_env} (from env), init_method='env', backend='nccl' "
        )

        # 4. 初始化进程组
        dist.init_process_group(
            backend='nccl',
            init_method='env://',  # 如需显式指定可以打开这行
        )
        logging.info(
            f"[{pid}] Process group initialized for RANK {dist.get_rank()} / "
            f"WORLD_SIZE {dist.get_world_size()} (reported by torch.distributed)."
        )

        # 5. 设置通信组（这里面用 DeviceMesh 初始化 DP/PP/EP/GP 几个 group）
        setup_dist_group(
            dp_size=dp_size,
            dp_replica_size=dp_replica_size,
            dp_shard_size=dp_shard_size,
            pp_size=pp_size,
            gp_size=gp_size,
            ep_size=ep_size,
            has_fsdp=has_fsdp,
        )

        # 打印一下各个并行维度的 group 信息
        # 确保这里能访问到这三个全局变量
        global _DP_GROUP, _PP_GROUP, _GP_GROUP

        logging.info(
            f"Rank {global_rank} successfully initialized process groups via DeviceMesh."
        )
        pp_ws, pp_ranks = _safe_group_info(_PP_GROUP)
        ep_ws, ep_ranks = _safe_group_info(_EP_GROUP)
        gp_ws, gp_ranks = _safe_group_info(_GP_GROUP)
        dp_ws, dp_ranks = _safe_group_info(_DP_GROUP)
        data_ws, data_ranks = _safe_group_info(_DATA_GROUP)
        dpgp_ws, dpgp_ranks = _safe_group_info(_DP_GP_GROUP)

        if os.environ.get("DP_DEBUG_2X2", "0") == "1":
            logging.info(
                "Rank %d topology summary -> global=%d/%d data=%d/%d gp=%d/%d pp=%d/%d ep=%d/%d",
                global_rank,
                global_rank,
                dist.get_world_size(),
                get_data_rank(),
                get_data_world_size(),
                get_gp_rank(),
                get_gp_world_size(),
                get_pp_rank(),
                get_pp_world_size(),
                get_ep_rank(),
                get_ep_world_size(),
            )
            logging.info(
                "Rank %d group members -> DP=%s DATA=%s GP=%s PP=%s EP=%s DPxGP=%s",
                global_rank,
                dp_ranks,
                data_ranks,
                gp_ranks,
                pp_ranks,
                ep_ranks,
                dpgp_ranks,
            )

        # if has_fsdp:
        #     dp_replica_ws, dp_replica_ranks = _group_info(_DP_REPLICA_GROUP)
        #     dp_shard_ws, dp_shard_ranks = _group_info(_DP_SHARD_GROUP)
        #     logging.info(
        #         "Rank %d -> "
        #         "PP=%d %s, "
        #         "DP_REPLICA=%d %s, "
        #         "DP_SHARD=%d %s, "
        #         "EP=%d %s, "
        #         "GP=%d %s, "
        #         "DP_SHARD×GP=%d %s",
        #         global_rank,
        #         pp_ws, pp_ranks,
        #         dp_replica_ws, dp_replica_ranks,
        #         dp_shard_ws, dp_shard_ranks,
        #         ep_ws, ep_ranks,
        #         gp_ws, gp_ranks,
        #         dpgp_ws, dpgp_ranks,
        #     )
        # else:
        #     dp_ws, dp_ranks = _group_info(_DP_GROUP)
        #     logging.info(
        #         "Rank %d -> "
        #         "PP=%d %s, "
        #         "DP=%d %s, "
        #         "EP=%d %s, "
        #         "GP=%d %s, "
        #         "DP×GP=%d %s",
        #         global_rank,
        #         pp_ws, pp_ranks,
        #         dp_ws, dp_ranks,
        #         ep_ws, ep_ranks,
        #         gp_ws, gp_ranks,
        #         dpgp_ws, dpgp_ranks,
        #     )



# --- 辅助函数，用于从外部访问这些组 ---

def initialized() -> bool:
    # return _GP_GROUP is not None
    return (
        dist.is_initialized()
        and DeviceMesh is not None
    )


def get_dp_group():
    assert _DATA_GROUP is not None, "DP group not initialized"
    return _DATA_GROUP

def get_dp_replica_group():
    assert _DP_REPLICA_GROUP is not None, "DP replica group not initialized"
    return _DP_REPLICA_GROUP

def get_dp_shard_group():
    assert _DP_SHARD_GROUP is not None, "DP shard group not initialized"
    return _DP_SHARD_GROUP


def get_device_mesh() -> DeviceMesh:
    """返回原始 DeviceMesh: (pp, dp_replica, gp, ep, dp_shard)。"""
    assert initialized(), "Distributed not initialized"
    assert _DEVICE_MESH is not None, "DeviceMesh not initialized"
    return _DEVICE_MESH

def get_fsdp2_expert_mesh() -> DeviceMesh:
    """FSDP2 expert mesh（HSDP 2D）：replica=dp_replica, shard=dp_shard×gp。"""
    assert initialized(), "Distributed not initialized"
    assert _DEVICE_MESH is not None, "DeviceMesh not initialized"
    assert get_pp_world_size() == 1, "FSDP2 expert mesh currently requires pp_size == 1"
    # return _DEVICE_MESH[("dp_replica_gp", "dp_shard")]
    return _DEVICE_MESH[("dp_replica", "dp_shard_gp")]

def get_fsdp2_non_expert_mesh() -> DeviceMesh:
    """FSDP2 non-expert/root mesh（HSDP 2D）：replica=dp_replica, shard=dp_shard×ep×gp。"""
    assert initialized(), "Distributed not initialized"
    assert _DEVICE_MESH is not None, "DeviceMesh not initialized"
    assert get_pp_world_size() == 1, "FSDP2 non-expert mesh currently requires pp_size == 1"
    # return _DEVICE_MESH[("dp_replica_ep_gp", "dp_shard")]
    return _DEVICE_MESH[("dp_replica", "dp_shard_ep_gp")]

def get_pp_group():
    assert _PP_GROUP is not None, "PP group not initialized"
    return _PP_GROUP

def get_gp_group():
    assert _GP_GROUP is not None, "GP group not initialized"
    return _GP_GROUP

def get_dp_gp_group():
    """获取 DP+GP 合并组（同一 PP stage 内的所有 DP×GP）"""
    assert _DP_GP_GROUP is not None, "DP+GP group not initialized"
    return _DP_GP_GROUP


def get_dp_gp_ep_group():
    """获取 DP+EP+GP 合并组（同一 PP stage 内的所有 DP×EP×GP）"""
    assert _DP_GP_EP_GROUP is not None, "DP+EP+GP group not initialized"
    return _DP_GP_EP_GROUP



def get_data_group():
    """获取 Data Group（dp_total×ep，dp_total=dp_replica×dp_shard），用于数据切分"""
    return _DATA_GROUP

def get_dp_ep_group():
    return _DATA_GROUP
def get_ep_group():
    """返回 Expert Parallel 进程组（如果未启用 EP，则返回 None）"""
    return _EP_GROUP


def get_ep_world_size() -> int:
    """获取 Expert Parallel 维度的 world size（未启用时为 1）"""
    return dist.get_world_size(group=_EP_GROUP)

def get_dp_ep_rank() -> int:
    """获取当前进程在 DP+EP 合并组中的 rank"""
    return dist.get_rank(group=get_data_group()) if initialized() else 0

def get_data_rank() -> int:
    """获取当前进程在 Data Group 中的 rank"""
    return dist.get_rank(group=get_data_group()) if initialized() else 0

def get_ep_rank() -> int:
    """获取当前进程在 Expert Parallel 组中的 rank（未启用时为 0）"""
    if not initialized() or _EP_GROUP is None:
        return 0
    return dist.get_rank(group=_EP_GROUP)

def get_dp_rank() -> int:
    """获取当前进程在数据并行组中的 rank"""
    return dist.get_rank(group=get_data_group()) if initialized() else 0

def get_dp_group_rank0_global_rank() -> int:
    """获取 DP group 内 rank 0 对应的 global rank
    
    用于在 broadcast 等操作中指定 src rank。
    在 DP+GP 模式下，DP group 内的 rank 0 对应的 global rank 可能不是 0。
    
    Returns:
        DP group 内 rank 0 对应的 global rank
    """
    if not initialized():
        return 0
    dp_group = get_dp_group()
    # 获取 DP group 内所有 global rank 的列表，第一个就是 rank 0 对应的 global rank
    dp_group_ranks = dist.get_process_group_ranks(dp_group)
    return dp_group_ranks[0] if dp_group_ranks else 0

def get_pp_rank() -> int:
    """获取当前进程在流水线并行组中的 rank"""
    return dist.get_rank(group=get_pp_group()) if initialized() else 0

def get_gp_rank() -> int:
    """获取当前进程在图/张量并行组中的 rank"""
    return dist.get_rank(group=get_gp_group()) if initialized() else 0

def get_gp_group_rank0_global_rank() -> int:
    """获取 GP group 内 rank 0 对应的 global rank
    
    用于在 broadcast 等操作中指定 src rank。
    在 DP+GP 模式下，GP group 内的 rank 0 对应的 global rank 可能不是 0。
    
    Returns:
        GP group 内 rank 0 对应的 global rank
    """
    if not initialized():
        return 0
    gp_group = get_gp_group()
    # 获取 GP group 内所有 global rank 的列表，第一个就是 rank 0 对应的 global rank
    gp_group_ranks = dist.get_process_group_ranks(gp_group)
    return gp_group_ranks[0] if gp_group_ranks else 0

def get_dp_world_size() -> int:
    """获取 DP 总大小（dp_replica×dp_shard）"""
    return dist.get_world_size(group=get_dp_group()) if initialized() else 1

def get_dp_replica_world_size() -> int:
    """获取 DP replica 维度大小"""
    return dist.get_world_size(group=get_dp_replica_group()) if initialized() else 1

def get_dp_shard_world_size() -> int:
    """获取 DP shard 维度大小"""
    return dist.get_world_size(group=get_dp_shard_group()) if initialized() else 1

def get_pp_world_size() -> int:
    """获取流水线并行组的大小"""
    return dist.get_world_size(group=get_pp_group()) if initialized() else 1

def get_gp_world_size() -> int:
    """获取图/张量并行组的大小"""
    return dist.get_world_size(group=get_gp_group()) if initialized() else 1

def get_dp_gp_world_size() -> int:
    """获取 DP+GP 合并组大小（用于最终规约）"""
    return dist.get_world_size(group=get_dp_gp_group()) if initialized() else 1


def get_dp_gp_ep_world_size() -> int:
    """获取 DP+EP+GP 合并组大小"""
    return dist.get_world_size(group=get_dp_gp_ep_group()) if initialized() else 1

def get_dp_ep_world_size() -> int:
    """获取 dp_total×ep 的大小"""
    return dist.get_world_size(group=get_dp_ep_group()) if initialized() else 1

def get_data_world_size() -> int:
    """获取 Data Group 大小（dp_total×ep）"""
    return dist.get_world_size(group=get_data_group()) if initialized() else 1

def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", -1))


def get_rank() -> int:
    return dist.get_rank() if initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if initialized() else 1

def is_master() ->bool:
    return get_rank() == 0


def is_ep_enabled() -> bool:
    """
    判断是否启用 Expert Parallel：
    - dist 已初始化；
    - EP 组 world_size > 1。
    """
    if not initialized():
        return False
    return get_ep_world_size() > 1

def is_last_stage():
    """这里的stage指的是Model stage是Model的切片
    """
    if Decouple_FE_FF == False:
        return get_pp_rank() == get_pp_world_size()-1 and get_pp_world_size() > 1
    else:
        # 中间两个都是last stage
        return ((get_pp_rank() == int(get_pp_world_size() / 2) - 1) or (get_pp_rank() == int(get_pp_world_size() / 2)) ) and get_pp_world_size() > 1
def is_first_stage():
    """这里的stage指的是Model stage是Model的切片
    """
    if Decouple_FE_FF == False:
        return get_pp_rank() == 0
    else:
        # stage 0 和最后一个stage 需要之前的first stage的Model
        return get_pp_rank() == 0 or get_pp_rank() == get_pp_world_size() - 1
def is_inter_stage():
    """是否是中间的stage
    """
    return (not is_first_stage()) and (not is_last_stage())

def stage_id_2_rank(stage_id: int) -> int:
    """将流水线并行的stage id转换为rank,方便dist send和recv发送接收数据
    Args:
        stage_id (int): 流水线并行的stage id    
    """
    assert _PP_STAGES_2_RANKS is not None, "PP stages to ranks mapping is not initialized"
    
    return _PP_STAGES_2_RANKS[stage_id] 




def cleanup_group() -> None:
    if dist.is_initialized():
        with contextlib.suppress(Exception):
            dist.destroy_process_group()
 
 





########## GRAPH PARALLEL DIST METHODS ##########


def pad_tensor(
    tensor: torch.Tensor, dim: int = -1, target_size: int | None = None
) -> torch.Tensor:
    """这在需要将张量均匀切分到不同 GPU 时很有用
    
    填充tensor，如果target_size为None，则根据GP的world_size填充，否则根据target_size填充,
    将tensor的某个维度填充为可以被GP的world_size整除的大小，方便做Graph Parallel

    Args:
        tensor (torch.Tensor): 需要填充的tensor
        dim (int, optional): 需要填充的维度. Defaults to -1.
        target_size (int | None, optional): 需要填充到的目标大小. Defaults to None.

    Returns:
        torch.Tensor: 填充后的tensor
    """
    size = tensor.size(dim)
    if target_size is None: # 这个好像没用到
        world_size = get_gp_world_size()
        pad_size = 0 if size % world_size == 0 else world_size - size % world_size # 
    else: # 一般会给定target size
        pad_size = target_size - size
    if pad_size == 0:
        return tensor
    pad_shape = list(tensor.shape)
    pad_shape[dim] = pad_size
    padding = torch.empty(pad_shape, device=tensor.device, dtype=tensor.dtype)
    return torch.cat([tensor, padding], dim=dim) # 在dim维度padding


def trim_tensor(tensor: torch.Tensor, sizes: torch.Tensor | None = None, dim: int = 0):
    """与 pad_tensor 相对，用于在聚合后去除可能由 pad_tensor 添加的填充部分，或者确保张量大小符合预期 。
    
    裁剪tensor，如果sizes为None，则根据GP的world_size裁剪，否则根据sizes裁剪, 

    Args:
        tensor (torch.Tensor): 需要裁剪的tensor
        sizes (torch.Tensor | None, optional): 需要裁剪到的目标大小. Defaults to None.
        dim (int, optional): 需要裁剪的维度. Defaults to 0.

    Returns:
        tuple: 裁剪后的tensor和目标大小
    """
    size = tensor.size(dim)
    world_size = get_gp_world_size()
    if size % world_size == 0:
        return tensor, sizes
    trim_size = size - size % world_size
    if dim == 0:
        tensor = tensor[:trim_size]
    elif dim == 1:
        tensor = tensor[:, :trim_size]
    else:
        raise ValueError
    if sizes is not None:
        sizes[-1] = sizes[-1] - size % world_size
    return tensor, sizes


def _tensor_to_split_partitions(tensor: torch.Tensor, dim: int = -1):
    """ 计算如何将张量在指定维度上切分成若干块，每块的大小使得能均匀（或尽可能均匀）分配给图并行组中的各个 GPU 。

    Args:
        tensor (torch.Tensor): 需要切分的tensor
        dim (int, optional): 需要切分的维度. Defaults to -1.

    Returns:
        list: 每块的大小
    """
    group = get_gp_group()
    num_parts = dist.get_world_size(group=group)
    return [len(part) for part in np.array_split(np.zeros(tensor.size(dim)), num_parts)]


def _split_tensor(
    tensor: torch.Tensor,
    dim: int = -1,
    contiguous_chunks: bool = False,
):
    """根据上述计算的切分方式，实际将张量切分成一个张量列表 

    Args:
        tensor (torch.Tensor): 需要切分的tensor
        dim (int, optional): 需要切分的维度. Defaults to -1.
        contiguous_chunks (bool, optional): 是否需要连续的块. Defaults to False.

    Returns:
        list: 切分后的tensor列表
    """
    tensor_list = torch.split(tensor, _tensor_to_split_partitions(tensor, dim), dim=dim)
    if contiguous_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)
    return tensor_list


def _reduce(ctx: Any, input: torch.Tensor) -> torch.Tensor:
    group = get_gp_group()
    # ctx的本质是torch autograd的上下文对象。用于在forward和backward之间传递梯度。
    # 只有在用到自定义的autograd Function的backward和forward方法里才会用到ctx
    if ctx:
        ctx.mark_dirty(input) #这个input可能会被in place的修改
    torch.cuda.synchronize()
    dist.all_reduce(input, group=group)
    return input


def _split(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    rank = get_gp_rank()
    input_list = _split_tensor(input, dim=dim)
    return input_list[rank].clone().contiguous() # 返回一个克隆的连续的tensor，不contiguous的tensor会传输错误。


def _gather(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    group = get_gp_group()
    rank = get_gp_rank()
    world_size = dist.get_world_size(group=group)
    if world_size == 1:
        return input
    tensor_list = [torch.empty_like(input) for _ in range(world_size)]
    tensor_list[rank] = input
    torch.cuda.synchronize()
    dist.all_gather(tensor_list, input, group=group)
    return torch.cat(tensor_list, dim=dim).contiguous()


def _gather_with_padding(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """将所有的rank的input收集起来，然后cat起来

    Args:
        input (torch.Tensor): _description_
        dim (int, optional): _description_. Defaults to -1.

    Returns:
        torch.Tensor: _description_
    """
    group = get_gp_group()
    rank = get_gp_rank()
    world_size = dist.get_world_size(group=group) # 这个是获取Graph Parallel Group的size 

    # ranks_in_group = dist.get_process_group_ranks(group)
    # # 获取全局rank, 以便和组内rank对比
    # global_rank = dist.get_rank() 
    # # 获取当前进程在这个特定组内的rank
    # rank_in_group = dist.get_rank(group=group)
    # # 获取这个特定组的大小
    # size_of_group = dist.get_world_size(group=group)
    # logging.warning(
    #     f"GLOBAL_RANK {global_rank}: "
    #     f"Ready for SIZES all_gather. "
    #     f"In group of size {size_of_group}, my rank is {rank_in_group}."
    #     f"ranks_in_group {ranks_in_group}"
    # )

    
    if world_size == 1:
        return input

    # Gather sizes，list中每个元素是每个rank的input的dim维度的size
    size_list = [
        torch.empty(1, device=input.device, dtype=torch.int32) for _ in range(world_size)
    ]
    # NOTE: Prefer int32 for NCCL collectives; int64 may fall back or behave poorly in some environments.
    size = torch.tensor([input.size(dim)], device=input.device, dtype=torch.int32)
    size_list[rank] = size
    # logging.info(f"Gathering tensor of shape {input.device} on rank {rank} in group {id(group)}, world_size={world_size} size_list { size_list}, size { size}")
    
    dist.all_gather(size_list, size, group=group)

    # Gather the inputs
    max_size = int(max([size.item() for size in size_list]))
    input = pad_tensor(input, dim, max_size) # 在dim维度padding为max_size
    shape = list(input.shape) # shape返回的是tensor，将其变成list
    shape[dim] = max_size # 将dim的维度赋值为max_size
    # 用来收集每个rank的input，然后cat起来
    tensor_list = [
        torch.empty(shape, device=input.device, dtype=input.dtype)
        for _ in range(world_size)
    ]
    # 将每个rank的input收集起来
    with torch.profiler.record_function("gather_with_padding"):
        dist.all_gather(tensor_list, input, group=group)
    tensor_list[rank] = input  # pop back in our local copy (requires grad)

    # Trim and cat
    # 使用narrow在dim裁切为0-size的tensor，然后在dim维度上cat
    return torch.cat(
        [tensor.narrow(dim, 0, size) for tensor, size in zip(tensor_list, size_list)],
        dim=dim,
    ).contiguous()


class CopyToModelParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        return input

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return _reduce(None, grad_output)

# 主要使用
class ReduceFromModelParallelRegion(torch.autograd.Function):
    """all reduce Graph Group中的tensor

    Args:
        torch (_type_): _description_

    Returns:
        _type_: _description_
    """
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        # return _reduce(ctx, input) # this operates in place
        torch.cuda.synchronize()
        return all_reduce(input, group=get_gp_group())  # this operats out of place

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


# this returns the values in place
class ScatterToModelParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, dim: int = -1) -> torch.Tensor:
        result = _split(input, dim)
        ctx.save_for_backward(torch.tensor(dim))
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (dim,) = ctx.saved_tensors
        return _gather_with_padding(grad_output.clone(), dim.item()), None

# 主要使用
class GatherFromModelParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, dim: int = -1) -> torch.Tensor:
        ctx.save_for_backward(torch.tensor(dim))
        return _gather_with_padding(input, dim)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (dim,) = ctx.saved_tensors
        result = _split(grad_output, dim.item())
        return result, None


class GatherFromModelParallelRegionSumGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, dim: int = -1) -> torch.Tensor:
        # 保存下来dim
        with record_function("GP_allgather foward"):
            ctx.save_for_backward(torch.tensor(dim))
            temp_result = _gather_with_padding(input, dim)
        return temp_result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (dim,) = ctx.saved_tensors
        group = get_gp_group()
        # use dist internal # does not work
        # reduced_grad_output = grad_output.clone()
        # dist.all_reduce(
        #    reduced_grad_output, group=group
        # )  # This is an inplace operation
        # grad_output = reduced_grad_output

        # use functional version instead torch 中这样的通信函数才支持二阶导数，旧的dist的all reduce不支持二阶导数
        # 反向传播过来的是整个node的feature的grad，需要进行all reduce之后再split
        # torch.cuda.synchronize()
        with record_function("GP_allgather backward"):
            grad_output = all_reduce(grad_output, group=group) 

        result = _split(grad_output, dim.item())
        return result, None


# Leave forward untouched but upscale the gradient by a factor of gp_group_size
# DDP reduces a mean across the loss, if we have gp_group_size=2 and 6 ranks
# that means we do (a_1+a_2+a_3+b_1+b_2+b_3)/6 in ddp mean. This gets us the
# correct loss but the grad is wrong by a factor of gp_group_size
# dL/d_a1 = 1/6 but it should be dL/da = 1/2 (for the equivalanet non GP run
# with 2 ranks)
# we coud perform an extra round of all_reduce, but this would increase
# communication overhead, instead we can just upscsale the gradient only and
# avoid over head communication
class ScaleBackwardGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        return input

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # 修正梯度。
        return dist.get_world_size(get_gp_group()) * grad_output


def copy_to_model_parallel_region(input: torch.Tensor) -> torch.Tensor:
    assert initialized(), "Cannot use graph parallel with initializing gp group, must call setup_gp from gp_utils.py!"
    return CopyToModelParallelRegion.apply(input)

# NOTE 主要使用的函数-1
def reduce_from_model_parallel_region(input: torch.Tensor) -> torch.Tensor:
    """
    all reduce，
    input: tensor,
    output tensor = tensor1 + tensor2 + tensor3 + ...
    """
    assert initialized(), "Cannot use graph parallel with initializing gp group, must call setup_gp from gp_utils.py!"
    return ReduceFromModelParallelRegion.apply(input)


def scatter_to_model_parallel_region(
    input: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    
    assert initialized(), "Cannot use graph parallel with initializing gp group, must call setup_gp from gp_utils.py!"
    return ScatterToModelParallelRegion.apply(input, dim)

# NOTE 主要使用的函数-2
def gather_from_model_parallel_region(
    input: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    
    assert initialized(), "Cannot use graph parallel with initializing gp group, must call setup_gp from gp_utils.py!"
    return GatherFromModelParallelRegion.apply(input, dim)


def gather_from_model_parallel_region_sum_grad(
    input: torch.Tensor, dim: int = -1
) -> torch.Tensor: 
    # 将input gather起来，实际上是不同的rank的node feature
    assert initialized(), "Cannot use graph parallel with initializing gp group, must call setup_gp from gp_utils.py!"
    return GatherFromModelParallelRegionSumGrad.apply(input, dim)


def scale_backward_grad(input: torch.Tensor) -> torch.Tensor:
    
    assert initialized(), "Cannot use graph parallel with initializing gp group, must call setup_gp from gp_utils.py!"
    return ScaleBackwardGrad.apply(input)

def _set_seeds(seed: int) -> None:
    import os, random, numpy as np, torch 
    os.environ['PYTHONHASHSEED'] = str(seed) 
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1' 
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' 
    random.seed(seed)
    np.random.seed(seed) 
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.use_deterministic_algorithms(True) 
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = False
    cudnn.deterministic = True 
    torch.set_printoptions(precision=20)


def synchronize_parameters(model: torch.nn.Module, group=None):
    """
    在训练的开始的时候需要统一参数。保证训练开始一致
    将 group 内 rank 0 的模型参数广播到指定组中的所有其他进程。
    确保所有模型的初始状态完全一致。
    
    Args:
        model: 模型
        group: 进程组，默认为DP group。如果指定，则同步到该group。
    """
    if group is None:
        group = get_dp_group()
        world_size = get_dp_world_size()
        # 获取 DP group 内 rank 0 对应的 global rank
        src_rank = get_dp_group_rank0_global_rank()
    else:
        world_size = dist.get_world_size(group=group) if dist.is_initialized() else 1
        # 获取指定 group 内 rank 0 对应的 global rank
        # 判断 group 类型，使用对应的辅助函数
        if dist.is_initialized():
            group_ranks = dist.get_process_group_ranks(group)
            src_rank = group_ranks[0] if group_ranks else 0
        else:
            src_rank = 0
    
    if world_size == 1:
        return

    for param in model.parameters():
        dist.broadcast(param.data, src=src_rank, group=group)



def allreduce_gradients(model, dp_group=None, world_size=None, average=True):
    """
    对模型的所有梯度在指定组内进行 All-Reduce。
    - 在 backward() 之后, optimizer.step() 之前调用。
    - 能够处理部分 rank 上梯度为 None 的情况，通过将其替换为零张量来避免死锁。
    
    Args:
        model: 模型
        dp_group: 进程组，默认为 DP group
        world_size: 进程组大小，默认为 DP world size
        average: 是否求平均。对于 DP 应该求平均，对于 GP 应该求和（average=False）
    """
    if dp_group is None:
        dp_group = get_dp_group()
    if world_size is None:
        world_size = dist.get_world_size(group=dp_group) if dist.is_initialized() else 1

    # 如果只有一个 GPU，则无需通信
    if world_size == 1:
        return

    for p in model.parameters():
        # 关键修改：如果梯度是 None，则创建一个零张量
        # 这对于解决因数据依赖的控制流（if/else）导致部分 GPU 上的参数未使用而梯度为 None 的问题至关重要
        if p.grad is None:
            # 使用 p.data 来获取正确的 shape, device, 和 dtype
            p.grad = torch.zeros_like(p.data)
        
        # 现在可以保证所有 rank 的 p.grad 都是一个张量，可以安全地进行 all_reduce
        dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM, group=dp_group)
        # 根据 average 参数决定是求和还是求平均 ,GP 求和，DP 求平均
        if average:
            p.grad.data /= float(world_size)


def split_moe_params(model: torch.nn.Module) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Split parameters into expert and non-expert lists by name convention."""
    from torch.nn.parallel import DistributedDataParallel as DDP

    actual_model = model.module if isinstance(model, DDP) else model
    expert_params = []
    non_expert_params = []
    for name, param in actual_model.named_parameters():
        if not param.requires_grad:
            continue
        if ".moe.experts." in name:
            expert_params.append(param)
        else:
            non_expert_params.append(param)
    return expert_params, non_expert_params


def _allreduce_param_grads(
    params: list[torch.nn.Parameter],
    group,
    average: bool,
    extra_divisor: float | None = None,
) -> None:
    if not initialized():
        return
    world_size = dist.get_world_size(group=group)
    if world_size == 1:
        return
    for param in params:
        if param.grad is None:
            param.grad = torch.zeros_like(param.data)
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM, group=group)
        if average:
            param.grad.data /= float(world_size)
        # if extra_divisor is not None and extra_divisor > 1:
        if extra_divisor is not None:
            param.grad.data /= float(extra_divisor)


def sync_moe_gradients(model: torch.nn.Module) -> None:
    """Sync gradients with GP/DP/EP semantics for MoE models.
        Loss averaged in batch. No need to divide by world_size in _allreduce_param_grads
    """
    expert_params, non_expert_params = split_moe_params(model)
    dp_gp_ep_group = get_dp_gp_ep_group()
    dp_gp_group = get_dp_gp_group()
    dp_ep_world_size = get_dp_ep_world_size()
    dp_world_size = get_dp_world_size()

    # non-expert: DP×GP×EP SUM, then divide by DP×EP
    _allreduce_param_grads(
        non_expert_params, dp_gp_ep_group, average=False, extra_divisor=dp_world_size
    )
    # expert: DP×GP SUM (fixed ep), then divide by DP
    _allreduce_param_grads(
        expert_params, dp_gp_group, average=False, extra_divisor=dp_world_size
    )

class RankFilter(logging.Filter):
    def __init__(self, rank, pp_rank, dp_rank, ep_rank, gp_rank):
        super().__init__()
        self.rank = rank
        self.pp_rank = pp_rank
        self.dp_rank = dp_rank
        self.ep_rank = ep_rank
        self.gp_rank = gp_rank
    def filter(self, record):
        # 为 LogRecord 添加自定义字段
        record.rank = self.rank
        record.pp_rank = self.pp_rank
        record.dp_rank = self.dp_rank
        record.ep_rank = self.ep_rank
        record.gp_rank = self.gp_rank
        return True

def setup_logging(level=logging.INFO):
    # 1) 先获取 rank  
    rank = get_rank()
    pp_rank = get_pp_rank()
    dp_rank = get_dp_rank()
    ep_rank = get_ep_rank()
    gp_rank = get_gp_rank()

    # 2) 清理并建立 handler + formatter
    fmt = '%(asctime)s %(levelname)s [rank:%(rank)s pp:%(pp_rank)s dp:%(dp_rank)s ep:%(ep_rank)s gp:%(gp_rank)s] %(filename)s:%(lineno)d - %(message)s'
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt))
    rank_filter = RankFilter(rank, pp_rank, dp_rank, ep_rank, gp_rank)
    handler.addFilter(rank_filter)

    root = logging.getLogger()
    # 移除已有 handler（避免重复打印）
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(handler)
    root.setLevel(level)

    # 3) 添加 filter 注入 rank 字段（保证任何 logger 都有这些字段）
    root.addFilter(rank_filter)

def reinit_model_params(model, init_type="uniform", value=1.0, low=-0.1, high=0.1, mean=0.0, std=0.02):
    """
    重新初始化模型参数，不依赖任何随机数生成器。
    
    Args:
        model: nn.Module
        init_type: "ones" | "zeros" | "uniform" | "normal" | "constant"
        value: 常数填充值 (仅当 init_type="constant")
        low, high: 均匀分布范围 (init_type="uniform"，但注意：这里会依赖随机数)
        mean, std: 正态分布参数 (init_type="normal"，同样依赖随机数)
    """
    for name, param in model.named_parameters():
        # 跳过 reference_energy 的权重，因为它已经有预训练的值
        # reference_energy 的参数名称通常是 "reference_energy.fc.weight"
        if 'reference_energy' in name:
            continue
            
        numel = param.numel()  # 参数里元素总数
        if init_type == "ones":
            param.data.fill_(1.0)

        elif init_type == "zeros":
            param.data.fill_(0.0)

        elif init_type == "constant":
            param.data.fill_(value)

        elif init_type == "uniform":
            # 用等间距确定性序列替代随机初始化
            values = torch.linspace(low, high, steps=numel, dtype=param.data.dtype, device=param.data.device)
            param.data.copy_(values.view_as(param.data))


        elif init_type == "normal":
            # 这里仍然会用到全局随机状态
            _set_seeds(2025)
            param.data.normal_(mean, std)

        else:
            raise ValueError(f"Unknown init_type: {init_type}")
        

def _print_first_10(name: str, tensor: torch.Tensor) -> None:
    """Flatten tensor to 1D and print first 10 elements with 20 decimal places."""
    if not isinstance(tensor, torch.Tensor):
        logging.info(f"{name}: is not a tensor, type: {type(tensor)}")
        return
    flat = tensor.detach().reshape(-1).cpu()
    num = min(10, flat.numel())
    if num == 0:
        logging.info(f"{name}: empty tensor")
        return
    vals = flat[:num].tolist()
    formatted = " ".join(f"{v:.20f}" for v in vals)
    logging.info(f"{name} first {num} elements: {formatted}")
