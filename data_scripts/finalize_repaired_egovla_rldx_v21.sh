#!/usr/bin/env bash
set -euo pipefail

# Finish the one retained v2.1 staging dataset after its cross-episode tail
# packets have been removed by trim_egovla_rldx_v21_videos.py.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
STAGING="${1:?usage: $0 STAGING SOURCE_V3 [OUTPUT]}"
SOURCE_V3="${2:?usage: $0 STAGING SOURCE_V3 [OUTPUT]}"
OUTPUT="${3:-${XPL_ROOT}/data/EgoVLA_benchmark_rldx_v21}"
MODALITY_CONFIG="${XPL_ROOT}/policy/RLDX_1/egovla_joint38_config.py"
RLDX_UPSTREAM="${XPL_ROOT}/policy/RLDX_1/RLDX-1"

PYTHON_BIN="${RLDX_PYTHON_BIN:-/mnt/xspark-data/xiangpc/.uv-python/cpython-3.10.20-linux-x86_64-gnu/bin/python3.10}"
RLDX_SITE_PACKAGES="${RLDX_SITE_PACKAGES:-/mnt/xspark-data/xiangpc/old_sim_eval/0807_RLDX-1/policy/RLDX_1/RLDX-1/.venv/lib/python3.10/site-packages}"
AUDIT_PYTHON="${AUDIT_PYTHON:-/mnt/xspark-data/xiangpc/.uv-python/cpython-3.11.15-linux-x86_64-gnu/bin/python3.11}"
AUDIT_SITE_PACKAGES="${AUDIT_SITE_PACKAGES:-/mnt/xspark-data/xiangpc/0811_Xpolicylab_bench/pi05/policy/Pi_05/openpi/.venv-egovla/lib/python3.11/site-packages}"

data_parent="$(realpath -m "${XPL_ROOT}/data")"
staging_resolved="$(realpath -m "${STAGING}")"
source_resolved="$(realpath -m "${SOURCE_V3}")"
output_resolved="$(realpath -m "${OUTPUT}")"
[[ "$(dirname "${staging_resolved}")" == "${data_parent}" ]] || {
  echo "staging is outside the RLDX data directory: ${staging_resolved}" >&2
  exit 1
}
[[ "$(dirname "${source_resolved}")" == "${data_parent}" ]] || {
  echo "v3 source backup is outside the RLDX data directory: ${source_resolved}" >&2
  exit 1
}
[[ "$(dirname "${output_resolved}")" == "${data_parent}" ]] || {
  echo "output is outside the RLDX data directory: ${output_resolved}" >&2
  exit 1
}
[[ -d "${staging_resolved}" && -d "${source_resolved}" ]] || {
  echo "staging or v3 source backup is missing" >&2
  exit 1
}
[[ ! -e "${output_resolved}" ]] || {
  echo "refusing to overwrite output: ${output_resolved}" >&2
  exit 1
}

for required in \
  "${PYTHON_BIN}" \
  "${RLDX_SITE_PACKAGES}" \
  "${AUDIT_PYTHON}" \
  "${AUDIT_SITE_PACKAGES}" \
  "${MODALITY_CONFIG}" \
  "${staging_resolved}/meta/rldx_v21_video_trim.json"; do
  [[ -e "${required}" ]] || {
    echo "missing required input: ${required}" >&2
    exit 1
  }
done

python3 - "${staging_resolved}/meta/rldx_v21_video_trim.json" <<'PY'
import json
import pathlib
import sys

marker = json.loads(pathlib.Path(sys.argv[1]).read_text())
if marker.get("total_videos") != 5709:
    raise SystemExit(f"invalid trim marker total: {marker}")
if marker.get("trimmed", 0) + marker.get("already_exact", 0) != 5709:
    raise SystemExit(f"incomplete trim marker: {marker}")
PY

if [[ -e "${staging_resolved}/meta/stats.json" || \
      -e "${staging_resolved}/meta/relative_stats.json" ]]; then
  echo "refusing to reuse pre-existing computed statistics" >&2
  exit 1
fi

echo "[RLDX EgoVLA] computing exact absolute/relative statistics"
PYTHONPATH="${RLDX_SITE_PACKAGES}:${RLDX_UPSTREAM}:${XPL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  NO_ALBUMENTATIONS_UPDATE=1 \
  "${PYTHON_BIN}" "${SCRIPT_DIR}/compute_egovla_rldx_stats.py" \
  "${staging_resolved}" "${MODALITY_CONFIG}"

echo "[RLDX EgoVLA] auditing parquet, decoded counts, prompts, and lossless packet slices"
PYTHONPATH="${AUDIT_SITE_PACKAGES}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${AUDIT_PYTHON}" "${SCRIPT_DIR}/audit_egovla_rldx_v21.py" \
  "${staging_resolved}" --source-v3 "${source_resolved}" \
  --workers "${AUDIT_WORKERS:-32}"

[[ ! -e "${output_resolved}" ]] || {
  echo "output appeared during validation; refusing publish: ${output_resolved}" >&2
  exit 1
}
mv -T "${staging_resolved}" "${output_resolved}"
echo "DONE output=${output_resolved} episodes=1903 frames=510546 videos=5709 action_type=joint action_dim=38"
echo "RLDX v3 hard-link backup retained at ${source_resolved} until training smoke passes"
