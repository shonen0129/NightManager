#!/bin/bash
# ============================================================
# macOS用 Distribution Diagnostics生成スクリプト (Step 1)
# 米国市場クローズ後（日本時間早朝）に実行
# ============================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR="${PROJECT_DIR}/logs"
VENV_DIR="${PROJECT_DIR}/.venv-mac"

mkdir -p "${LOG_DIR}"

DATESTR=$(date +%Y%m%d)
LOG_FILE="${LOG_DIR}/distribution_diagnostics_${DATESTR}.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === distribution diagnostics (Step 1) 開始 ===" >> "${LOG_FILE}"

PYTHON_BIN="${VENV_DIR}/bin/python"
if [ -f "${PYTHON_BIN}" ]; then
    :
else
    echo "[ERROR] venv python not found: ${PYTHON_BIN}" >> "${LOG_FILE}"
    exit 1
fi

cd "${PROJECT_DIR}"

# Step 1: distribution_diagnosticsの実行
# archive/tools/compute_structured_prediction_covariance.pyを使用
set +e
PYTHONPATH=src "${PYTHON_BIN}" archive/tools/compute_structured_prediction_covariance.py \
    --config configs/production/production.yaml \
    --model production_residual_blpx \
    --start "2020-01-01" \
    --end "$(date +%Y-%m-%d)" \
    --results-dir live/pipeline_data/diagnostics_weights \
    --output-dir live/pipeline_data/distribution_diagnostics \
    --slippage-bps 5.0 \
    --save-daily-matrices true \
    --save-psd-projection true \
    --compare-existing-pred-var true \
    --vol-state-panel null \
    --run-backtest-if-missing \
    >> "${LOG_FILE}" 2>&1

EXIT_CODE=$?
set -e

if [ ${EXIT_CODE} -ne 0 ]; then
    echo "[ERROR] distribution diagnostics computation failed (exit=${EXIT_CODE})" >> "${LOG_FILE}"
    exit ${EXIT_CODE}
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === distribution diagnostics (Step 1) 終了コード: ${EXIT_CODE} ===" >> "${LOG_FILE}"
exit ${EXIT_CODE}
