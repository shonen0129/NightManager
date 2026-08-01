#!/bin/bash
# ============================================================
# macOS用 引け損益レポート送信スクリプト
# 15:40 実行想定（15:30 大引け後に約定情報を再取得してレポート作成・送信）
# ============================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR="${PROJECT_DIR}/logs"
VENV_DIR="${PROJECT_DIR}/.venv-mac"

mkdir -p "${LOG_DIR}"

DATESTR=$(date +%Y%m%d)
LOG_FILE="${LOG_DIR}/pnl_report_${DATESTR}.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === pnl report 開始 ===" >> "${LOG_FILE}"

# 仮想環境のPython（activateが壊れている可能性があるため直接指定）
PYTHON_BIN="${VENV_DIR}/bin/python"
if [ -f "${PYTHON_BIN}" ]; then
    :
else
    echo "[ERROR] venv python not found: ${PYTHON_BIN}" >> "${LOG_FILE}"
    exit 1
fi

# .env があれば読み込む
if [ -f "${PROJECT_DIR}/.env" ]; then
    # shellcheck source=/dev/null
    export $(grep -v '^#' "${PROJECT_DIR}/.env" | xargs)
fi

# スクリプト実行
cd "${PROJECT_DIR}"
PYTHONPATH=src "${PYTHON_BIN}" tools/production/send_daily_close_pnl_report.py \
    >> "${LOG_FILE}" 2>&1

EXIT_CODE=$?
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === 終了コード: ${EXIT_CODE} ===" >> "${LOG_FILE}"
exit ${EXIT_CODE}
