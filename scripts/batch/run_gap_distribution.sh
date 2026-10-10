#!/bin/bash
# ============================================================
# macOS用 production V2 gap cache publisher
#
# Frozen 09:10 quote + run-owned df_execから、本番decisionと同じ
# on-demand計算境界を使ってcanonical GapStoreへpublishする。
# research diagnostics / V1 backtestは運用入力にしない。
# ============================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR="${PROJECT_DIR}/var/logs"
PIPELINE_DIR="${PROJECT_DIR}/var/live/pipeline_data"
GAP_DIR="${PIPELINE_DIR}/gap_adjusted_distribution"
GAP_STORE="${GAP_DIR}/gap_store.sqlite"

if [ -z "${LEADLAG_LEASE_OWNER:-}" ]; then
    if [ -f "${PROJECT_DIR}/.venv/bin/python" ]; then
        GUARD_PYTHON="${PROJECT_DIR}/.venv/bin/python"
    else
        GUARD_PYTHON="$(command -v python3)"
    fi
    DATESTR=$(date +%Y%m%d)
    exec env PYTHONPATH="${PROJECT_DIR}/src" \
        "${GUARD_PYTHON}" -m leadlag.execution.job_guard \
        --scope "live:production_v2" \
        --timeout "${LEADLAG_GAP_TIMEOUT_SECONDS:-1800}" \
        --grace "${LEADLAG_JOB_GRACE_SECONDS:-10}" \
        --state-db "${PROJECT_DIR}/var/live/pipeline_data/execution/execution_state.sqlite" \
        --guard-log "${PROJECT_DIR}/var/logs/job_guard/gap_${DATESTR}.json" \
        -- bash "$0" "$@"
fi

mkdir -p "${LOG_DIR}" "${GAP_DIR}/acceptance"
DATESTR=$(date +%Y%m%d)
TODAY=$(date +%Y-%m-%d)
LOG_FILE="${LOG_DIR}/gap_distribution_${DATESTR}.log"
PHASE_LOG_DIR="${PROJECT_DIR}/var/logs/job_guard"
ACCEPTANCE_FILE="${GAP_DIR}/acceptance/gap_${DATESTR}.json"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === gap distribution 開始 ===" >> "${LOG_FILE}"

if [ -f "${PROJECT_DIR}/.venv/bin/python" ]; then
    PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
else
    echo "[ERROR] no python3 interpreter found" >> "${LOG_FILE}"
    exit 1
fi

cd "${PROJECT_DIR}"

run_phase() {
    local label="$1"
    local timeout_seconds="$2"
    shift 2
    PYTHONPATH=src "${PYTHON_BIN}" scripts/tools/phase_deadline.py \
        --label "${label}" \
        --timeout "${timeout_seconds}" \
        --grace "${LEADLAG_JOB_GRACE_SECONDS:-10}" \
        --log "${PHASE_LOG_DIR}/gap_${DATESTR}_${label}.json" \
        -- "$@"
}

# Compute every configured horizon first, then publish each atomic SQLite
# bundle and read it back through FileCacheDistributionSource. Missing/stale
# inputs are a non-zero batch result; yesterday's matrices are never copied.
set +e
run_phase "compute" "${LEADLAG_GAP_COMPUTE_TIMEOUT_SECONDS:-1500}" \
    "${PYTHON_BIN}" tools/production/publish_gap_distribution.py \
    --config configs/production/production.yaml \
    --trade-date "${TODAY}" \
    --gap-store "${GAP_STORE}" \
    --acceptance-output "${ACCEPTANCE_FILE}" \
    >> "${LOG_FILE}" 2>&1
EXIT_CODE=$?
set -e

if [ ${EXIT_CODE} -ne 0 ]; then
    echo "[ERROR] gap distribution publication failed (exit=${EXIT_CODE})" >> "${LOG_FILE}"
    exit ${EXIT_CODE}
fi

set +e
run_phase "store_check" "${LEADLAG_GAP_STORE_CHECK_TIMEOUT_SECONDS:-30}" \
    "${PYTHON_BIN}" -m leadlag.execution.gap_store_check \
    --store "${GAP_STORE}" \
    --trade-date "${TODAY}" \
    >> "${LOG_FILE}" 2>&1
STORE_CHECK_EXIT=$?
set -e

if [ ${STORE_CHECK_EXIT} -ne 0 ]; then
    echo "[WARNING] Canonical gap SQLite check failed (exit=${STORE_CHECK_EXIT}). Decision will use its configured fallback policy." >> "${LOG_FILE}"
    exit ${STORE_CHECK_EXIT}
fi

echo "[INFO] Gap acceptance: ${ACCEPTANCE_FILE}" >> "${LOG_FILE}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === gap distribution 終了コード: 0 ===" >> "${LOG_FILE}"
exit 0
