#!/bin/bash
# ============================================================
# macOS用 ポジションクローズスクリプト
# leadlag close (14:50実行)
# ============================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${PROJECT_DIR}/logs"
VENV_DIR="${PROJECT_DIR}/.venv-mac"

mkdir -p "${LOG_DIR}"

DATESTR=$(date +%Y%m%d)
LOG_FILE="${LOG_DIR}/close_positions_${DATESTR}.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === close positions 開始 ===" >> "${LOG_FILE}"

# 仮想環境のアクティベート
if [ -f "${VENV_DIR}/bin/activate" ]; then
    source "${VENV_DIR}/bin/activate"
else
    echo "[ERROR] venv not found: ${VENV_DIR}" >> "${LOG_FILE}"
    exit 1
fi

# スクリプト実行
cd "${PROJECT_DIR}"
PYTHONPATH=src python -m leadlag.cli close \
    >> "${LOG_FILE}" 2>&1

EXIT_CODE=$?
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === 終了コード: ${EXIT_CODE} ===" >> "${LOG_FILE}"
exit ${EXIT_CODE}
