#!/usr/bin/env bash
set -euo pipefail

# Two-GPU pure graph-parallel launcher.
#
# Default topology:
#   - DP = 1
#   - GP = 2
#   - PP = 1
#   - EP = 1
#   - one datapoint batch is split across 2 GP ranks / 2 GPUs
#
# This script is the preserved two-card GP entrypoint.
#
# Usage:
#   ./run_4rank_single_gpu_gp.sh [train-args...]
# Example:
#   ./run_4rank_single_gpu_gp.sh train --skip-neighbor-stat test_mptraj/input_dpa3.json

export DISABLE_GP_MODE=${DISABLE_GP_MODE:-0}
export DP_GP_EXEC_MODE=${DP_GP_EXEC_MODE:-distributed}
export DP_ENABLE_DISTUTILS_GP=${DP_ENABLE_DISTUTILS_GP:-1}

# Parallel topology. Default is pure graph parallel on 2 GPUs.
export DP_GP_SIZE=${DP_GP_SIZE:-2}
export DP_DP_SIZE=${DP_DP_SIZE:-1}
export DP_PP_SIZE=${DP_PP_SIZE:-1}
export DP_EP_SIZE=${DP_EP_SIZE:-1}
export NPROC_PER_NODE=${NPROC_PER_NODE:-$((DP_DP_SIZE * DP_GP_SIZE * DP_PP_SIZE * DP_EP_SIZE))}

# Each rank gets a distinct GPU by default.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

# Optional bring-up debug logs.
export DP_DEBUG_2X2=${DP_DEBUG_2X2:-0}

# Used only when DeePMD falls back to its own process-group initialization.
export DP_PT_DIST_BACKEND=${DP_PT_DIST_BACKEND:-cuda:nccl,cpu:gloo}

LOG_ROOT=${LOG_ROOT:-logs/torchrun}
RUN_ID=${RUN_ID:-dp1_gp2_$(date +%Y%m%d_%H%M%S)}
RUN_LOG_DIR="${LOG_ROOT}/${RUN_ID}"
mkdir -p "${RUN_LOG_DIR}"

CONDA_ENV_PREFIX=${CONDA_ENV_PREFIX:-/aisi-nas/liwentao/miniconda/deepmd_gp}
CONDA_BIN=${CONDA_BIN:-$(command -v conda || true)}

if [[ -z "${CONDA_BIN}" ]]; then
  echo "[run_4rank_single_gpu_gp] conda not found. Expected environment: ${CONDA_ENV_PREFIX}" >&2
  exit 127
fi

if [[ ! -d "${CONDA_ENV_PREFIX}" ]]; then
  echo "[run_4rank_single_gpu_gp] conda env not found: ${CONDA_ENV_PREFIX}" >&2
  exit 127
fi

LAUNCHER_CMD=(
  "${CONDA_BIN}" run --no-capture-output -p "${CONDA_ENV_PREFIX}"
  python -m torch.distributed.run
)

echo "[run_4rank_single_gpu_gp] log dir: ${RUN_LOG_DIR}"
echo "[run_4rank_single_gpu_gp] DP_GP_EXEC_MODE=${DP_GP_EXEC_MODE}"
echo "[run_4rank_single_gpu_gp] DP topology: DP=${DP_DP_SIZE}, GP=${DP_GP_SIZE}, PP=${DP_PP_SIZE}, EP=${DP_EP_SIZE}"
echo "[run_4rank_single_gpu_gp] NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "[run_4rank_single_gpu_gp] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[run_4rank_single_gpu_gp] DP_DEBUG_2X2=${DP_DEBUG_2X2}"
echo "[run_4rank_single_gpu_gp] conda env: ${CONDA_ENV_PREFIX}"

"${LAUNCHER_CMD[@]}" \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --log-dir "${RUN_LOG_DIR}" \
  --redirects 3 \
  --tee 3 \
  tools/run_gp_pt_train.py "$@"
