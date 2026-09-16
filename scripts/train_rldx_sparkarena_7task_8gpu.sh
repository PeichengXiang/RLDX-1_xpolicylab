#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_POLICY="${REPO_ROOT}/policy/RLDX_1"
ROOT="${RLDX_TRAINING_ROOT:-${REPO_ROOT}}"
RUNTIME_CODE="${RLDX_RUNTIME_CODE:-${REPO_ROOT}}"
RUNTIME_POLICY="${RLDX_RUNTIME_POLICY:-${REPO_POLICY}}"
UPSTREAM="${RLDX_UPSTREAM:-${RUNTIME_POLICY}/RLDX-1}"
LAUNCHER="${RLDX_LAUNCHER:-${RUNTIME_POLICY}/launch_train_egovla.py}"
DATASET="${RLDX_DATASET:-${REPO_ROOT}/data/spark0_bench_7task_0908}"
MODALITY_CONFIG="${RLDX_MODALITY_CONFIG:-${REPO_POLICY}/spark_joint54_relative_config.py}"
PYTHON="${RLDX_PYTHON:-${UPSTREAM}/.venv/bin/python}"
SITE="${RLDX_SITE_PACKAGES:-${UPSTREAM}/.venv/lib/python3.10/site-packages}"
SOURCE_BASE_MODEL="${RLDX_SOURCE_BASE_MODEL:-${REPO_ROOT}/pretrain_model/RLDX-1-PT-1592013a}"
BASE_MODEL="${RLDX_BASE_MODEL:-${REPO_ROOT}/pretrain_model/RLDX-1-PT-1592013a-sparkarena-relarm}"
RUN_NAME="${RLDX_RUN_NAME:-rldx-SparkArena_7task-joint54-relarm-bs64-80k}"
CHECKPOINT_ROOT="${RLDX_CHECKPOINT_ROOT:-${REPO_POLICY}/checkpoints}"
CKPT_DIR="${CHECKPOINT_ROOT}/${RUN_NAME}"

if [[ -n "${RLDX_WANDB_ENV_FILE:-}" ]]; then
    # shellcheck disable=SC1090
    source "${RLDX_WANDB_ENV_FILE}"
elif [[ -f /root/.config/training_0908-real/wandb.env ]]; then
    # shellcheck disable=SC1091
    source /root/.config/training_0908-real/wandb.env
fi
if [[ -z "${WANDB_API_KEY:-}" ]] && ! grep -qs 'api.wandb.ai' "${HOME}/.netrc" 2>/dev/null; then
    echo "ERROR: W&B is not authenticated (WANDB_API_KEY or ~/.netrc required)" >&2
    exit 1
fi
if [[ -e "${CKPT_DIR}" ]] && [[ "${RLDX_RESUME:-0}" != 1 ]]; then
    echo "ERROR: checkpoint directory already exists: ${CKPT_DIR}" >&2
    echo "Set RLDX_RESUME=1 only when resuming this relative-arm run." >&2
    exit 1
fi
for required in \
    "${PYTHON}" \
    "${SITE}" \
    "${UPSTREAM}" \
    "${LAUNCHER}" \
    "${MODALITY_CONFIG}" \
    "${REPO_POLICY}/prepare_sparkarena_relative_base_overlay.sh" \
    "${REPO_ROOT}/data_scripts/compute_sparkarena_7task_relative_stats.py"; do
    if [[ ! -e "${required}" ]]; then
        echo "ERROR: missing required path: ${required}" >&2
        exit 1
    fi
done
for required in meta/info.json meta/episodes.jsonl meta/tasks.jsonl meta/modality.json meta/stats.json; do
    if [[ ! -f "${DATASET}/${required}" ]]; then
        echo "ERROR: dataset is incomplete; missing ${DATASET}/${required}" >&2
        exit 1
    fi
done

export WANDB_PROJECT="${RLDX_WANDB_PROJECT:-xpolicylab-SparkArena-7task-relative}"
export WANDB_NAME="${RUN_NAME}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-SparkArena_7task_relarm}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-rldx_sparkarena_7task_relarm}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="${WANDB_DIR:-${REPO_ROOT}/logs}"
export PYTHONPATH="${SITE}:${UPSTREAM}:${REPO_POLICY}:${RUNTIME_POLICY}:${RUNTIME_CODE}${PYTHONPATH:+:${PYTHONPATH}}"
export RLDX_SERIALIZE_IMPORT="${RLDX_SERIALIZE_IMPORT:-1}"
export RLDX_IMPORT_LOCK="${RLDX_IMPORT_LOCK:-/tmp/rldx_hf_import.lock}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export USE_TF=0
export NO_ALBUMENTATIONS_UPDATE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export TORCHINDUCTOR_COMPILE_THREADS=2
export HF_HOME="${HF_HOME:-${REPO_ROOT}/pretrain_model/hf_home}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB__DISABLE_STATS=true
export RLDX_SAVE_STEPS_OVERRIDE=10000
# A800_12: keep OpenCV; skip torchcodec import (it can hang after FFmpeg install).
export RLDX_VIDEO_BACKEND="${RLDX_VIDEO_BACKEND:-opencv}"
export RLDX_SKIP_TORCHCODEC="${RLDX_SKIP_TORCHCODEC:-1}"
export RLDX_VIDEO_RETRIES="${RLDX_VIDEO_RETRIES:-5}"
export RLDX_VIDEO_CACHE_DIR="${RLDX_VIDEO_CACHE_DIR:-${REPO_ROOT}/.cache/rldx_video}"
export REAL_STATUS_PATH="${REAL_STATUS_PATH:-${REPO_ROOT}/logs/rldx_sparkarena_7task_relarm_status.json}"
export TORCH_HOME="${TORCH_HOME:-${REPO_ROOT}/.cache/torch}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${REPO_ROOT}/.cache/torch_extensions}"
export MASTER_PORT="${MASTER_PORT:-29524}"
mkdir -p "${WANDB_DIR}" "${TORCH_HOME}" "${TORCH_EXTENSIONS_DIR}" "${CHECKPOINT_ROOT}"

RLDX_SOURCE_BASE_MODEL_PATH="${SOURCE_BASE_MODEL}" \
RLDX_RELATIVE_BASE_MODEL_PATH="${BASE_MODEL}" \
RLDX_UPSTREAM_PATH="${UPSTREAM}" \
RLDX_PYTHON_BIN="${PYTHON}" \
    bash "${REPO_POLICY}/prepare_sparkarena_relative_base_overlay.sh"

"${PYTHON}" "${REPO_ROOT}/data_scripts/compute_sparkarena_7task_relative_stats.py" \
    --dataset "${DATASET}" \
    --modality-config "${MODALITY_CONFIG}"

if [[ -e "${CKPT_DIR}" ]] && [[ "${RLDX_RESUME:-0}" == 1 ]]; then
    latest_checkpoint="$(find "${CKPT_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' -print | sort -V | tail -n 1)"
    if [[ -z "${latest_checkpoint}" ]]; then
        echo "ERROR: RLDX_RESUME=1 but no checkpoint-* directory exists under ${CKPT_DIR}" >&2
        exit 1
    fi
    "${PYTHON}" - "${latest_checkpoint}" <<'PY'
import json
from pathlib import Path
import sys

import yaml

checkpoint = Path(sys.argv[1])
experiment = yaml.safe_load((checkpoint / "experiment_cfg/config.yaml").read_text())
if experiment.get("model", {}).get("use_relative_action") is not True:
    raise SystemExit(f"refusing to resume non-relative checkpoint: {checkpoint}")
processor = json.loads((checkpoint / "processor/processor_config.json").read_text())
kwargs = processor["processor_kwargs"]
if kwargs.get("use_relative_action") is not True:
    raise SystemExit(f"checkpoint processor disabled relative actions: {checkpoint}")
action = kwargs["modality_configs"]["general_embodiment"]["action"]
expected = ["RELATIVE", "ABSOLUTE", "RELATIVE", "ABSOLUTE"]
actual = [item["rep"] for item in action["action_configs"]]
if actual != expected:
    raise SystemExit(f"checkpoint action contract is {actual}, expected {expected}")
statistics = json.loads((checkpoint / "processor/statistics.json").read_text())
relative = statistics.get("general_embodiment", {}).get("relative_action", {})
if set(relative) != {"left_arm", "right_arm"}:
    raise SystemExit(f"checkpoint relative statistics are incomplete: {relative.keys()}")
print(f"READY resume contract checkpoint={checkpoint}")
PY
fi

echo "RLDX-1 SparkArena7 dataset=${DATASET} global_batch=64 GPUs=8"
echo "action_contract=arms:relative-to-current hands:absolute horizon=16"
echo "steps=80000 save_steps=10000 save_total_limit=8 checkpoint=${CKPT_DIR}"
cd "${UPSTREAM}"
exec "${PYTHON}" -m torch.distributed.run --nproc_per_node=8 --master_port="${MASTER_PORT}" \
    "${LAUNCHER}" \
    --base-model-path "${BASE_MODEL}" \
    --backbone-path RLWRLD/RLDX-1-VLM \
    --dataset-path "${DATASET}" \
    --embodiment-tag GENERAL_EMBODIMENT \
    --modality-config-path "${MODALITY_CONFIG}" \
    --tune-llm --tune-visual --tune-top-llm-layers 0 --tune-projector --tune-diffusion-model \
    --no-backbone-use-lora --no-action-model-use-lora --no-freeze-cog-tokens \
    --state-dropout-prob 0.0 --n-cog-tokens 64 --action-horizon 16 --video-length 4 --video-stride 2 \
    --global-batch-size 64 --gradient-accumulation-steps 1 --num-gpus 8 --dataloader-num-workers "${RLDX_DATALOADER_WORKERS:-2}" \
    --learning-rate 2e-5 \
    --max-steps 80000 --save-steps 10000 --save-total-limit 8 \
    --output-dir "${CHECKPOINT_ROOT}" \
    --experiment-name "${RUN_NAME}" \
    --use-wandb --wandb-project "${WANDB_PROJECT}"
