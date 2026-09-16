#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UPSTREAM="${SCRIPT_DIR}/RLDX-1"
CONVERTER="${XPL_ROOT}/data_scripts/convert_sparkarena_7task_v21_joint54.py"
PYTHON_BIN="${RLDX_PYTHON_BIN:-${UPSTREAM}/.venv/bin/python}"

if [[ ! -f "${CONVERTER}" ]]; then
    echo "Missing Spark converter: ${CONVERTER}" >&2
    exit 1
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Missing RLDX conversion Python: ${PYTHON_BIN}" >&2
    echo "Run ${SCRIPT_DIR}/install.sh or set RLDX_PYTHON_BIN." >&2
    exit 1
fi

# With no arguments, the converter uses its documented SparkArena 7-task source
# and writes the same dataset directory that train.sh uses by default. Any
# provided flags are forwarded verbatim so callers can supply portable paths.
exec "${PYTHON_BIN}" "${CONVERTER}" "$@"
