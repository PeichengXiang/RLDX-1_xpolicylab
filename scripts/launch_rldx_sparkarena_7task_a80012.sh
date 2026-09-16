#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -n "${RLDX_WANDB_ENV_FILE:-}" ]]; then
    set -a
    source "${RLDX_WANDB_ENV_FILE}"
    set +a
fi

export RLDX_WANDB_PROJECT=xpolicylab-SparkArena-7task-relative
export RLDX_RUN_NAME="${RLDX_RUN_NAME:-rldx-SparkArena_7task-joint54-relarm-bs64-80k}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-rldx_sparkarena_7task_relarm}"
export WANDB_MODE=online
export RLDX_RESUME="${RLDX_RESUME:-0}"
export RLDX_VIDEO_BACKEND=opencv
export RLDX_SKIP_TORCHCODEC=1
export RLDX_DATALOADER_WORKERS=2
export RLDX_VIDEO_CACHE_DIR="${RLDX_VIDEO_CACHE_DIR:-${XPL_ROOT}/.cache/rldx_video}"
export RLDX_SOURCE_BASE_MODEL="${RLDX_SOURCE_BASE_MODEL:-${XPL_ROOT}/pretrain_model/RLDX-1-PT-1592013a}"
export RLDX_BASE_MODEL="${RLDX_BASE_MODEL:-${XPL_ROOT}/pretrain_model/RLDX-1-PT-1592013a-sparkarena-relarm}"
export REAL_STATUS_PATH="${REAL_STATUS_PATH:-${XPL_ROOT}/logs/rldx_sparkarena_7task_relarm_status.json}"

LOG="${RLDX_LOG_PATH:-${XPL_ROOT}/logs/SparkArena_7task_relarm_train.log}"
mkdir -p "$(dirname "${LOG}")"
echo "==== rldx relative-arm start $(date -Iseconds) host=$(hostname) ====" | tee -a "${LOG}"
set +e
bash "${XPL_ROOT}/scripts/train_rldx_sparkarena_7task_8gpu.sh" 2>&1 | tee -a "${LOG}"
status=${PIPESTATUS[0]}
set -e
echo "==== rldx relative-arm exited $(date -Iseconds) code=${status} ====" | tee -a "${LOG}"
exit "${status}"
