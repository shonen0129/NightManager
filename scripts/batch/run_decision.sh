#!/bin/bash
# ============================================================
# macOS用 発注自動化スクリプト
# leadlag decision (朝9:00実行)
# ============================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR="${PROJECT_DIR}/logs"
VENV_DIR="${PROJECT_DIR}/.venv-mac"

mkdir -p "${LOG_DIR}"

DATESTR=$(date +%Y%m%d)
LOG_FILE="${LOG_DIR}/decision_${DATESTR}.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === decision 開始 ===" >> "${LOG_FILE}"

# 仮想環境のPython（activateが壊れている可能性があるため直接指定）
PYTHON_BIN="${VENV_DIR}/bin/python"
if [ -f "${PYTHON_BIN}" ]; then
    :
else
    echo "[ERROR] venv python not found: ${PYTHON_BIN}" >> "${LOG_FILE}"
    exit 1
fi

# スクリプト実行
cd "${PROJECT_DIR}"
PYTHONPATH=src "${PYTHON_BIN}" -m leadlag.cli decision \
    --api-enable \
    --fast-mode \
    --capital-from-wallet \
    --text-output \
    >> "${LOG_FILE}" 2>&1

EXIT_CODE=$?
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === 終了コード: ${EXIT_CODE} ===" >> "${LOG_FILE}"
exit ${EXIT_CODE}
