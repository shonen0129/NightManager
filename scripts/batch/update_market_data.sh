#!/bin/bash
# ============================================================
# macOS用 市場データ強制更新スクリプト
# 米国市場クローズ後（日本時間早朝）に実行
# これを実行後、run_distribution_diagnostics.sh を実行すること
# ============================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR="${PROJECT_DIR}/var/logs"

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
        --timeout "${LEADLAG_MARKET_DATA_TIMEOUT_SECONDS:-1200}" \
        --grace "${LEADLAG_JOB_GRACE_SECONDS:-10}" \
        --state-db "${PROJECT_DIR}/var/live/pipeline_data/execution/execution_state.sqlite" \
        --guard-log "${PROJECT_DIR}/var/logs/job_guard/market_data_${DATESTR}.json" \
        -- bash "$0" "$@"
fi

mkdir -p "${LOG_DIR}"
DATESTR=$(date +%Y%m%d)
LOG_FILE="${LOG_DIR}/update_market_data_${DATESTR}.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === update market data 開始 ===" >> "${LOG_FILE}"

# 仮想環境を優先、なければシステムの python3 を使用
if [ -f "${PROJECT_DIR}/.venv/bin/python" ]; then
    PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
else
    echo "[ERROR] no python3 interpreter found" >> "${LOG_FILE}"
    exit 1
fi

cd "${PROJECT_DIR}"

set +e
PYTHONPATH=src "${PYTHON_BIN}" scripts/tools/phase_deadline.py \
    --label market_data_update --timeout "${LEADLAG_MARKET_DATA_PHASE_SECONDS:-1100}" \
    --grace "${LEADLAG_JOB_GRACE_SECONDS:-10}" \
    --log "${PROJECT_DIR}/var/logs/job_guard/market_data_phase_${DATESTR}.json" \
    -- "${PYTHON_BIN}" scripts/batch/_update_market_data.py >> "${LOG_FILE}" 2>&1

EXIT_CODE=$?
set -e

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === update market data 終了コード: ${EXIT_CODE} ===" >> "${LOG_FILE}"
exit ${EXIT_CODE}
