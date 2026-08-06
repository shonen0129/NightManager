#!/bin/bash
# ============================================================
# macOS用 市場データ強制更新スクリプト
# 米国市場クローズ後（日本時間早朝）に実行
# これを実行後、run_distribution_diagnostics.sh を実行すること
# ============================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR="${PROJECT_DIR}/logs"
VENV_DIR="${PROJECT_DIR}/.venv-mac"

mkdir -p "${LOG_DIR}"

DATESTR=$(date +%Y%m%d)
LOG_FILE="${LOG_DIR}/update_market_data_${DATESTR}.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === update market data 開始 ===" >> "${LOG_FILE}"

PYTHON_BIN="${VENV_DIR}/bin/python"
if [ ! -f "${PYTHON_BIN}" ]; then
    echo "[ERROR] venv python not found: ${PYTHON_BIN}" >> "${LOG_FILE}"
    exit 1
fi

cd "${PROJECT_DIR}"

PYTHONPATH=src "${PYTHON_BIN}" scripts/batch/_update_market_data.py >> "${LOG_FILE}" 2>&1

EXIT_CODE=$?

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === update market data 終了コード: ${EXIT_CODE} ===" >> "${LOG_FILE}"
exit ${EXIT_CODE}
