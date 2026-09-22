#!/bin/bash
# ============================================================
# macOS用 引け損益レポート送信スクリプト
# 15:40 実行想定（15:30 大引け後に約定情報を再取得してレポート作成・送信）
# ============================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR="${PROJECT_DIR}/var/logs"

# Reconciliation and reporting share a bounded guard. A read-only recovery
# failure is recorded without skipping the independent PnL report.
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
        --timeout "${LEADLAG_RECONCILIATION_TIMEOUT_SECONDS:-300}" \
        --grace "${LEADLAG_JOB_GRACE_SECONDS:-10}" \
        --state-db "${PROJECT_DIR}/var/live/pipeline_data/execution/execution_state.sqlite" \
        --guard-log "${PROJECT_DIR}/var/logs/job_guard/pnl_${DATESTR}.json" \
        -- bash "$0" "$@"
fi

mkdir -p "${LOG_DIR}"
DATESTR=$(date +%Y%m%d)
LOG_FILE="${LOG_DIR}/pnl_report_${DATESTR}.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === pnl report 開始 ===" >> "${LOG_FILE}"

# 仮想環境を優先、なければシステムの python3 を使用
if [ -f "${PROJECT_DIR}/.venv/bin/python" ]; then
    PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
else
    echo "[ERROR] no python3 interpreter found" >> "${LOG_FILE}"
    exit 1
fi
# .env があれば読み込む
if [ -f "${PROJECT_DIR}/.env" ]; then
    # shellcheck source=/dev/null
    export $(grep -v '^#' "${PROJECT_DIR}/.env" | xargs)
fi

# スクリプト実行
cd "${PROJECT_DIR}"
set +e
PYTHONPATH=src "${PYTHON_BIN}" -m leadlag.execution.reconcile --pending \
    --state-db "${PROJECT_DIR}/var/live/pipeline_data/execution/execution_state.sqlite" \
    --output-dir "${PROJECT_DIR}/var/results/reconciliation" \
    >> "${LOG_FILE}" 2>&1
RECONCILE_EXIT=$?
echo "[$(date '+%Y-%m-%d %H:%M:%S')] reconciliation exit: ${RECONCILE_EXIT}" >> "${LOG_FILE}"

PYTHONPATH=src "${PYTHON_BIN}" tools/production/send_daily_close_pnl_report.py \
    >> "${LOG_FILE}" 2>&1

EXIT_CODE=$?
set -e
if [ ${EXIT_CODE} -eq 0 ] && [ ${RECONCILE_EXIT} -ne 0 ]; then
    EXIT_CODE=${RECONCILE_EXIT}
fi
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === 終了コード: ${EXIT_CODE} ===" >> "${LOG_FILE}"
exit ${EXIT_CODE}
