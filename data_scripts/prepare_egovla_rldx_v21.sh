#!/usr/bin/env bash
set -euo pipefail

# Build an independent LeRobot v2.1 view for RLDX from the verified Pi0.5
# LeRobot v3 dataset. The Pi0.5 dataset is never renamed or modified.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_PARENT="$(cd "${XPL_ROOT}/.." && pwd)"

SOURCE_DATASET="${SOURCE_DATASET:-${WORKSPACE_PARENT}/pi05/data/EgoVLA_benchmark}"
OUTPUT_DATASET="${OUTPUT_DATASET:-${XPL_ROOT}/data/EgoVLA_benchmark_rldx_v21}"
CONVERTER_SCRIPT="${CONVERTER_SCRIPT:-${SCRIPT_DIR}/convert_v3_to_v2_egovla.py}"
CONVERTER_PYTHON="${CONVERTER_PYTHON:-${WORKSPACE_PARENT}/pi05/policy/Pi_05/openpi/.venv-egovla/bin/python}"
RLDX_PYTHON="${RLDX_PYTHON:-${XPL_ROOT}/policy/RLDX_1/RLDX-1/.venv/bin/python}"
RLDX_UPSTREAM="${XPL_ROOT}/policy/RLDX_1/RLDX-1"
MODALITY_CONFIG="${XPL_ROOT}/policy/RLDX_1/egovla_joint38_config.py"

for required in \
  "${SOURCE_DATASET}/meta/info.json" \
  "${SOURCE_DATASET}/meta/xpolicylab_conversion.json" \
  "${CONVERTER_SCRIPT}" \
  "${CONVERTER_PYTHON}" \
  "${RLDX_PYTHON}" \
  "${MODALITY_CONFIG}" \
  "${SCRIPT_DIR}/egovla_rldx_modality.json" \
  "${SCRIPT_DIR}/compute_egovla_rldx_stats.py" \
  "${SCRIPT_DIR}/audit_egovla_rldx_v21.py"; do
  [[ -e "${required}" ]] || {
    echo "missing required input: ${required}" >&2
    exit 1
  }
done

"${CONVERTER_PYTHON}" - "${SOURCE_DATASET}/meta/info.json" <<'PY'
import json
import pathlib
import sys

info = json.loads(pathlib.Path(sys.argv[1]).read_text())
expected = {
    "codebase_version": "v3.0",
    "robot_type": "ego_h1_inspire",
    "total_episodes": 1903,
    "total_frames": 510546,
    "total_tasks": 12,
    "fps": 30,
}
for key, value in expected.items():
    if info.get(key) != value:
        raise SystemExit(f"source info {key}={info.get(key)!r}, expected {value!r}")
for feature in ("observation.state", "action"):
    if info["features"][feature]["shape"] != [38]:
        raise SystemExit(f"source {feature} is not joint38")
print("validated immutable Pi0.5 source dataset")
PY

if [[ -e "${OUTPUT_DATASET}" ]]; then
  echo "refusing to overwrite output: ${OUTPUT_DATASET}" >&2
  exit 1
fi

output_parent="$(dirname "${OUTPUT_DATASET}")"
output_name="$(basename "${OUTPUT_DATASET}")"
mkdir -p "${output_parent}"
stamp="$(date -u +%Y%m%dT%H%M%S)"
staging="${output_parent}/.${output_name}.incomplete-${stamp}-$$"
source_backup="${staging}_v3.0"
if [[ -e "${staging}" || -e "${source_backup}" ]]; then
  echo "unique staging path already exists" >&2
  exit 1
fi

on_error() {
  status=$?
  echo "FAILED status=${status}; retained staging=${staging} source_backup=${source_backup}" >&2
  exit "${status}"
}
trap on_error ERR

echo "[RLDX EgoVLA] hard-linking verified v3 source into private staging"
cp -a --link "${SOURCE_DATASET}" "${staging}"

echo "[RLDX EgoVLA] converting private staging v3.0 -> v2.1"
"${CONVERTER_PYTHON}" "${CONVERTER_SCRIPT}" \
  --root "${output_parent}" \
  --repo-id "$(basename "${staging}")"

install -m 0644 "${SCRIPT_DIR}/egovla_rldx_modality.json" \
  "${staging}/meta/modality.json"
install -m 0644 "${SOURCE_DATASET}/meta/xpolicylab_conversion.json" \
  "${staging}/meta/xpolicylab_source_conversion.json"

# Preserve the v3 aggregate stats for provenance, but force RLDX to compute
# exact v2.1 absolute quantiles and arm-relative horizon statistics.
if [[ -f "${staging}/meta/stats.json" ]]; then
  mv "${staging}/meta/stats.json" "${staging}/meta/source_v3_stats.json"
fi
if [[ -f "${staging}/meta/relative_stats.json" ]]; then
  mv "${staging}/meta/relative_stats.json" \
    "${staging}/meta/source_relative_stats.json"
fi

echo "[RLDX EgoVLA] computing exact absolute/relative statistics"
PYTHONPATH="${RLDX_UPSTREAM}:${XPL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  NO_ALBUMENTATIONS_UPDATE=1 \
  "${RLDX_PYTHON}" "${SCRIPT_DIR}/compute_egovla_rldx_stats.py" \
  "${staging}" "${MODALITY_CONFIG}"

echo "[RLDX EgoVLA] auditing all parquet episodes and all video frame counts"
"${CONVERTER_PYTHON}" "${SCRIPT_DIR}/audit_egovla_rldx_v21.py" \
  "${staging}" --source-v3 "${source_backup}" --workers "${AUDIT_WORKERS:-32}"

if [[ -e "${OUTPUT_DATASET}" ]]; then
  echo "output appeared during conversion; refusing publish: ${OUTPUT_DATASET}" >&2
  exit 1
fi
mv -T "${staging}" "${OUTPUT_DATASET}"
trap - ERR

echo "DONE output=${OUTPUT_DATASET} episodes=1903 frames=510546 fps=30 action_type=joint action_dim=38"
echo "RLDX source-v3 hard-link backup retained at ${source_backup} until training smoke passes"
