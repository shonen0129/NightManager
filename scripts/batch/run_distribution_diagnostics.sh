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
    --vol-state-panel "" \
    --run-backtest-if-missing \
    --incremental \
    >> "${LOG_FILE}" 2>&1

EXIT_CODE=$?
set -e

if [ ${EXIT_CODE} -ne 0 ]; then
    echo "[ERROR] distribution diagnostics computation failed (exit=${EXIT_CODE})" >> "${LOG_FILE}"
    exit ${EXIT_CODE}
fi

# Update latest symlink (exclude 'latest' from matching)
LATEST_DIAG_DIR=$(ls -td ${PROJECT_DIR}/live/pipeline_data/distribution_diagnostics/*/ 2>/dev/null | grep -v '/latest/' | head -1)
if [ -n "${LATEST_DIAG_DIR}" ]; then
    ln -sfn "$(basename ${LATEST_DIAG_DIR})" ${PROJECT_DIR}/live/pipeline_data/distribution_diagnostics/latest
    echo "[INFO] Updated latest symlink -> ${LATEST_DIAG_DIR}" >> "${LOG_FILE}"
fi

# --- 鮮度チェック: 出力 diagnostics が本日取引日を含んでいるか ---
# distribution_panel_long.csv の列: signal_date, trade_date, ticker, ...
# 本日 T の決定には trade_date == TODAY の行が必要。
EXPECTED_TRADE_DATE="$(date +%Y-%m-%d)"

if [ -n "${LATEST_DIAG_DIR}" ] && [ -f "${LATEST_DIAG_DIR}/distribution_panel_long.csv" ]; then
    MAX_TRADE_DATE=$(awk -F, 'NR>1 {gsub(/ .*/, "", $2); print $2}' "${LATEST_DIAG_DIR}/distribution_panel_long.csv" | sort | tail -1)
    if [ "${MAX_TRADE_DATE}" != "${EXPECTED_TRADE_DATE}" ]; then
        echo "[WARNING] distribution_diagnostics output is stale: max trade_date=${MAX_TRADE_DATE}, expected=${EXPECTED_TRADE_DATE}" >> "${LOG_FILE}"
        echo "[WARNING] The US close data for the signal of ${EXPECTED_TRADE_DATE} may not have been available at run time." >> "${LOG_FILE}"
        echo "[WARNING] Consider rerunning this script later, or delay the schedule to after US data is processed." >> "${LOG_FILE}"
        # Not failing here; gap_distribution.sh will fail its own freshness check and produce flat position.
    else
        echo "[INFO] Freshness check passed: max trade_date=${MAX_TRADE_DATE}" >> "${LOG_FILE}"
    fi
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === distribution diagnostics (Step 1) 終了コード: ${EXIT_CODE} ===" >> "${LOG_FILE}"
exit ${EXIT_CODE}
