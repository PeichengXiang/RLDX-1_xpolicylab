#!/usr/bin/env bash
set -euo pipefail

# Publish an RLDX LeRobot v2.1 dataset whose action column is the recorded
# EgoVLA HDF5 /action. The audited legacy dataset is cloned into independent
# files; parquet action/metadata files are then atomically replaced in staging.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SOURCE_V21="${1:-${XPL_ROOT}/data/EgoVLA_benchmark_rldx_v21}"
SOURCE_V3="${2:-${XPL_ROOT}/../pi05/data/EgoVLA_benchmark}"
RAW_ROOT="${3:-/personal/xiangpc/EgoVLA benchmark/data/EgoVLA/raw_remove_deprecated}"
OUTPUT="${4:-${XPL_ROOT}/data/EgoVLA_benchmark_rldx_v21_raw_action}"
SCHEMA_PATH="${EGOVLA_SCHEMA_PATH:-/personal/xiangpc/EgoVLA benchmark/dataset_tool/egovla_dataset_tool/schema.py}"
MODALITY_CONFIG="${XPL_ROOT}/policy/RLDX_1/egovla_joint38_config.py"
RLDX_UPSTREAM="${XPL_ROOT}/policy/RLDX_1/RLDX-1"
PYTHON_BIN="${RLDX_PYTHON_BIN:-${RLDX_UPSTREAM}/.venv/bin/python}"

for required in \
  "${SOURCE_V21}/meta/rldx_egovla_audit.json" \
  "${SOURCE_V21}/meta/xpolicylab_source_conversion.json" \
  "${SOURCE_V3}/meta/info.json" \
  "${RAW_ROOT}" \
  "${SCHEMA_PATH}" \
  "${PYTHON_BIN}" \
  "${MODALITY_CONFIG}"; do
  [[ -e "${required}" ]] || { echo "missing required input: ${required}" >&2; exit 1; }
done
[[ ! -e "${OUTPUT}" ]] || { echo "refusing to overwrite output: ${OUTPUT}" >&2; exit 1; }

output_parent="$(dirname "${OUTPUT}")"
output_name="$(basename "${OUTPUT}")"
mkdir -p "${output_parent}"
staging="${output_parent}/.${output_name}.incomplete-$(date -u +%Y%m%dT%H%M%S)-$$"
[[ ! -e "${staging}" ]] || { echo "staging already exists: ${staging}" >&2; exit 1; }
trap 'echo "FAILED retained staging='"${staging}"'" >&2' ERR

echo "[RLDX EgoVLA] cloning audited v2.1 videos and parquet into independent files"
cp -a --reflink=auto "${SOURCE_V21}" "${staging}"
for name in stats.json relative_stats.json rldx_egovla_audit.json; do
  if [[ -e "${staging}/meta/${name}" ]]; then
    mv "${staging}/meta/${name}" "${staging}/meta/legacy_next_state_${name}"
  fi
done

echo "[RLDX EgoVLA] replacing next-state targets with recorded /action"
"${PYTHON_BIN}" "${SCRIPT_DIR}/rewrite_egovla_raw_action_v21.py" \
  "${staging}" --raw-root "${RAW_ROOT}" --schema "${SCHEMA_PATH}" \
  --workers "${REWRITE_WORKERS:-32}"

echo "[RLDX EgoVLA] computing raw-action absolute and relative-arm statistics"
PYTHONPATH="${RLDX_UPSTREAM}:${XPL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  NO_ALBUMENTATIONS_UPDATE=1 \
  "${PYTHON_BIN}" "${SCRIPT_DIR}/compute_egovla_rldx_stats.py" \
  "${staging}" "${MODALITY_CONFIG}"

echo "[RLDX EgoVLA] auditing all actions, prompts, cameras, and packet slices"
"${PYTHON_BIN}" "${SCRIPT_DIR}/audit_egovla_rldx_v21.py" \
  "${staging}" --source-v3 "${SOURCE_V3}" --raw-root "${RAW_ROOT}" \
  --schema "${SCHEMA_PATH}" --workers "${AUDIT_WORKERS:-32}"

[[ ! -e "${OUTPUT}" ]] || { echo "output appeared during validation: ${OUTPUT}" >&2; exit 1; }
mv -T "${staging}" "${OUTPUT}"
trap - ERR
echo "DONE output=${OUTPUT} episodes=1903 frames=510546 action_source=raw_hdf5/action"
