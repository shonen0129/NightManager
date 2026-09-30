#!/bin/bash
# Standalone, read-only 09:10 quote collection. This job never submits orders.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
if [ -x "${PROJECT_DIR}/.venv/bin/python" ]; then
    PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
else
    echo "[ERROR] no Python interpreter found" >&2
    exit 127
fi

DATESTR=$(TZ=Asia/Tokyo /bin/date +%Y%m%d)
STATE_DB="${PROJECT_DIR}/var/live/pipeline_data/execution/execution_state.sqlite"
GUARD_LOG="${PROJECT_DIR}/var/logs/job_guard/microstructure_0910_${DATESTR}.json"
OUTPUT_DIR="${LEADLAG_CAPTURE_OUTPUT_DIR:-${PROJECT_DIR}/var/live/pipeline_data/microstructure_0910}"
REQUEST_TIMEOUT="${TACHIBANA_REQUEST_TIMEOUT:-8}"

mkdir -p "${PROJECT_DIR}/var/logs/job_guard" "${OUTPUT_DIR}"
cd "${PROJECT_DIR}"

exec env \
    PYTHONPATH="${PROJECT_DIR}/src" \
    TACHIBANA_REQUEST_TIMEOUT="${REQUEST_TIMEOUT}" \
    "${PYTHON_BIN}" -m leadlag.execution.job_guard \
    --scope "live:microstructure_0910" \
    --timeout "65" \
    --grace "5" \
    --state-db "${STATE_DB}" \
    --guard-log "${GUARD_LOG}" \
    -- "${PYTHON_BIN}" tools/validation/collect_0910_microstructure.py \
    --capture-only \
    --window-seconds 30 \
    --attempts 2 \
    --retry-delay-seconds 2 \
    --output-dir "${OUTPUT_DIR}"
