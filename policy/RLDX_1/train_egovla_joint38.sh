#!/usr/bin/env bash
set -euo pipefail

# A800_13 launcher for RLDX-1 on the audited EgoVLA joint38 LeRobot v2.1 data.
# Usage:
#   WANDB_API_KEY=... bash train_egovla_joint38.sh start
#   WANDB_API_KEY=... bash train_egovla_joint38.sh resume
#   bash train_egovla_joint38.sh smoke

mode="${1:-start}"
if [[ "${mode}" != "start" && "${mode}" != "resume" && "${mode}" != "smoke" ]]; then
  echo "usage: $0 [start|resume|smoke]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UPSTREAM="${SCRIPT_DIR}/RLDX-1"
DATASET="${RLDX_DATASET_PATH:-${XPL_ROOT}/data/EgoVLA_benchmark_rldx_v21}"
MODALITY_CONFIG="${SCRIPT_DIR}/egovla_joint38_config.py"
TRAIN_ENTRYPOINT="${SCRIPT_DIR}/launch_train_egovla.py"
BASE_MODEL="${RLDX_BASE_MODEL_PATH:-${XPL_ROOT}/pretrain_model/RLDX-1-PT-1592013a-egovla-full}"
PYTHON_BIN="${RLDX_PYTHON_BIN:-${UPSTREAM}/.venv/bin/python}"
RLDX_SITE_PACKAGES="${RLDX_SITE_PACKAGES:-${UPSTREAM}/.venv/lib/python3.10/site-packages}"
RLDX_HF_HOME="${RLDX_HF_HOME:-${XPL_ROOT}/pretrain_model/hf_home}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a gpu_ids <<<"${CUDA_VISIBLE_DEVICES}"
if [[ "${#gpu_ids[@]}" -ne 8 ]]; then
  echo "exactly eight GPUs are required: ${CUDA_VISIBLE_DEVICES}" >&2
  exit 1
fi
if [[ "$(printf '%s\n' "${gpu_ids[@]}" | sort -u | wc -l)" -ne 8 ]]; then
  echo "GPU ids must be unique: ${CUDA_VISIBLE_DEVICES}" >&2
  exit 1
fi

MAX_STEPS="${MAX_STEPS:-30000}"
SAVE_STEPS="${SAVE_STEPS:-5000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-4}"
GLOBAL_BATCH_SIZE=64
NUM_GPUS=8
GRADIENT_ACCUMULATION_STEPS=1
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
STATE_DROPOUT_PROB=0.0

if [[ "${mode}" == "smoke" ]]; then
  MAX_STEPS="${SMOKE_MAX_STEPS:-1}"
  SAVE_STEPS=1000
  SAVE_TOTAL_LIMIT=1
  default_run_name="EgoVLA-benchmark-rldx1_joint38-strict-full-smoke"
  use_wandb=0
else
  default_run_name="EgoVLA-benchmark-rldx1_joint38-strict-full_30k-ego_h1_inspire-joint-0"
  use_wandb=1
  : "${WANDB_API_KEY:?WANDB_API_KEY must be supplied through the environment}"
fi
RUN_NAME="${RLDX_RUN_NAME:-${default_run_name}}"
CHECKPOINTS_ROOT="${SCRIPT_DIR}/checkpoints"
OUTPUT_DIR="${CHECKPOINTS_ROOT}/${RUN_NAME}"

for required in \
  "${PYTHON_BIN}" \
  "${RLDX_SITE_PACKAGES}" \
  "${RLDX_HF_HOME}/hub/models--RLWRLD--RLDX-1-VLM/refs/main" \
  "${UPSTREAM}/rldx/experiment/launch_train.py" \
  "${TRAIN_ENTRYPOINT}" \
  "${BASE_MODEL}/config.json" \
  "${BASE_MODEL}/model.safetensors.index.json" \
  "${BASE_MODEL}/experiment_cfg/config.yaml" \
  "${DATASET}/meta/info.json" \
  "${DATASET}/meta/modality.json" \
  "${DATASET}/meta/stats.json" \
  "${DATASET}/meta/relative_stats.json" \
  "${DATASET}/meta/rldx_egovla_audit.json" \
  "${MODALITY_CONFIG}"; do
  [[ -e "${required}" ]] || {
    echo "missing required input: ${required}" >&2
    exit 1
  }
done

export PYTHONPATH="${RLDX_SITE_PACKAGES}:${UPSTREAM}:${XPL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export NO_ALBUMENTATIONS_UPDATE=1
export CUDA_VISIBLE_DEVICES
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="${RLDX_HF_HOME}"
export TORCH_HOME="${RLDX_LOCAL_CACHE_ROOT:-/tmp/rldx-egovla-hpc}/torch"
export WANDB__DISABLE_STATS=true
export WANDB_PROJECT="${WANDB_PROJECT:-XPolicyLab-RLDX1-EgoVLA}"
export TOKENIZERS_PARALLELISM=false
export MASTER_PORT="${MASTER_PORT:-$(shuf -i 20000-30000 -n 1)}"
mkdir -p "${TORCH_HOME}" "${CHECKPOINTS_ROOT}"

exec 9>"${CHECKPOINTS_ROOT}/.${RUN_NAME}.launch.lock"
if ! flock -n 9; then
  echo "another launcher owns this run: ${RUN_NAME}" >&2
  exit 1
fi

hf_revision="$(<"${HF_HOME}/hub/models--RLWRLD--RLDX-1-VLM/refs/main")"
hf_snapshot="${HF_HOME}/hub/models--RLWRLD--RLDX-1-VLM/snapshots/${hf_revision}"
for required in \
  "${hf_snapshot}/config.json" \
  "${hf_snapshot}/tokenizer_config.json" \
  "${hf_snapshot}/preprocessor_config.json"; do
  [[ -e "${required}" ]] || {
    echo "incomplete offline RLDX VLM cache: ${required}" >&2
    exit 1
  }
done

"${PYTHON_BIN}" - "${DATASET}" "${MODALITY_CONFIG}" <<'PY'
import hashlib
import importlib.util
import json
import pathlib
import sys

dataset = pathlib.Path(sys.argv[1])
config_path = pathlib.Path(sys.argv[2])
info = json.loads((dataset / "meta/info.json").read_text())
marker = json.loads((dataset / "meta/rldx_egovla_audit.json").read_text())
expected = {
    "codebase_version": "v2.1",
    "total_episodes": 1903,
    "total_frames": 510546,
    "total_tasks": 12,
    "fps": 30,
}
for key, value in expected.items():
    if info.get(key) != value:
        raise SystemExit(f"dataset {key}={info.get(key)!r}, expected {value!r}")
marker_expected = {
    "audit_version": "2.0.0",
    "action_type": "joint",
    "action_dim": 38,
    "action_horizon": 16,
    "fps": 30,
    "total_episodes": 1903,
    "total_frames": 510546,
    "total_tasks": 12,
    "total_videos": 5709,
    "lossless_packet_slices_verified": 5709,
    "lossless_video_packets_verified": 1531638,
}
if any(marker.get(key) != value for key, value in marker_expected.items()):
    raise SystemExit(f"invalid audit marker: {marker}")
for name, expected_hash in marker["sha256"].items():
    actual = hashlib.sha256((dataset / "meta" / name).read_bytes()).hexdigest()
    if actual != expected_hash:
        raise SystemExit(f"audited metadata changed: {name}")

spec = importlib.util.spec_from_file_location("egovla_joint38_config", config_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
mods = module.EGOVLA_JOINT38_MODALITY_CONFIG
if mods["state"].modality_keys != ["left_arm", "left_hand", "right_arm", "right_hand"]:
    raise SystemExit("unexpected state modality order")
representations = [item.rep.name for item in mods["action"].action_configs]
if representations != ["RELATIVE", "ABSOLUTE", "RELATIVE", "ABSOLUTE"]:
    raise SystemExit(f"unexpected action semantics: {representations}")
print("validated audited EgoVLA joint38 dataset and RLDX modality semantics")
PY

"${PYTHON_BIN}" - "${BASE_MODEL}/experiment_cfg/config.yaml" <<'PY'
import pathlib
import sys
import yaml

overlay = yaml.safe_load(pathlib.Path(sys.argv[1]).read_text())
model = overlay.get("model", {})
training = overlay.get("training", {})
if model.get("model_name") != "RLWRLD/RLDX-1-VLM":
    raise SystemExit(f"strict full FT requires the RLDX-1-VLM backbone: {model}")
if model.get("backbone_model_type") != "vtc_qwen3_vl" or model.get("action_horizon") != 16:
    raise SystemExit("full-FT overlay architecture does not match the clean checkpoint")
if model.get("state_dropout_prob") != 0.0:
    raise SystemExit("strict clean-base full FT requires state_dropout_prob=0.0")
if training.get("deepspeed_stage") != 3:
    raise SystemExit(f"strict full FT requires ZeRO-3: {training}")
if training.get("gradient_checkpointing") is not True:
    raise SystemExit(f"strict full FT requires gradient checkpointing: {training}")

from rldx.configs.train_config import TrainConfig

defaults = TrainConfig()
if not defaults.tune_projector or not defaults.tune_diffusion_model:
    raise SystemExit("RLDX projector/diffusion full-tune defaults changed")
if defaults.action_model_use_lora or defaults.backbone_use_lora or defaults.freeze_cog_tokens:
    raise SystemExit("strict full FT refuses LoRA or frozen cognition tokens")
print("validated strict full-parameter flags, ZeRO-3, and gradient checkpointing")
PY

"${PYTHON_BIN}" - <<'PY'
import av
import deepspeed
import flash_attn
import torch
import torchcodec
import wandb

assert torch.cuda.device_count() == 8, torch.cuda.device_count()
print(
    f"runtime torch={torch.__version__} cuda={torch.version.cuda} "
    f"torchcodec={torchcodec.__version__} flash_attn={flash_attn.__version__} "
    f"av={av.__version__} deepspeed={deepspeed.__version__} wandb={wandb.__version__}"
)
PY

if [[ "${mode}" == "start" || "${mode}" == "smoke" ]]; then
  if [[ -e "${OUTPUT_DIR}" ]]; then
    echo "refusing to overwrite existing output: ${OUTPUT_DIR}" >&2
    exit 1
  fi
else
  if [[ ! -d "${OUTPUT_DIR}" ]] || ! compgen -G "${OUTPUT_DIR}/checkpoint-*" >/dev/null; then
    echo "resume requested but no checkpoint exists: ${OUTPUT_DIR}" >&2
    exit 1
  fi
fi

while IFS=, read -r index memory_used; do
  memory_used="${memory_used// /}"
  if (( memory_used > 1024 )); then
    echo "GPU ${index} is not idle: ${memory_used} MiB in use" >&2
    exit 1
  fi
done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)

echo "[RLDX EgoVLA] mode=${mode} run=${RUN_NAME}"
echo "[RLDX EgoVLA] action_type=joint action_dim=38 horizon=16 arms=relative hands=absolute"
echo "[RLDX EgoVLA] global_batch=64 gpu_count=8 per_gpu_batch=8 max_steps=${MAX_STEPS}"
echo "[RLDX EgoVLA] strict_full_ft=true zero_stage=3 gradient_checkpointing=true state_dropout_prob=${STATE_DROPOUT_PROB} learning_rate=${LEARNING_RATE}"
echo "[RLDX EgoVLA] dataset=${DATASET} base_model=${BASE_MODEL} output=${OUTPUT_DIR}"

wandb_args=()
if [[ "${use_wandb}" == "1" ]]; then
  wandb_args+=(--use-wandb --wandb-project "${WANDB_PROJECT}")
fi

cd "${UPSTREAM}"
exec "${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node=8 \
  --master_port="${MASTER_PORT}" \
  "${TRAIN_ENTRYPOINT}" \
  --base-model-path "${BASE_MODEL}" \
  --backbone-path RLWRLD/RLDX-1-VLM \
  --dataset-path "${DATASET}" \
  --embodiment-tag GENERAL_EMBODIMENT \
  --modality-config-path "${MODALITY_CONFIG}" \
  --tune-llm \
  --tune-visual \
  --tune-top-llm-layers 0 \
  --tune-projector \
  --tune-diffusion-model \
  --no-backbone-use-lora \
  --no-action-model-use-lora \
  --no-freeze-cog-tokens \
  --state-dropout-prob "${STATE_DROPOUT_PROB}" \
  --n-cog-tokens 64 \
  --action-horizon 16 \
  --video-length 4 \
  --video-stride 2 \
  --global-batch-size "${GLOBAL_BATCH_SIZE}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --num-gpus "${NUM_GPUS}" \
  --dataloader-num-workers "${DATALOADER_NUM_WORKERS}" \
  --learning-rate "${LEARNING_RATE}" \
  --max-steps "${MAX_STEPS}" \
  --save-steps "${SAVE_STEPS}" \
  --save-total-limit "${SAVE_TOTAL_LIMIT}" \
  --output-dir "${CHECKPOINTS_ROOT}" \
  --experiment-name "${RUN_NAME}" \
  "${wandb_args[@]}"
