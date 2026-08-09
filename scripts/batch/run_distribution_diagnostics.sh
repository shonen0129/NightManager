#!/bin/bash
# ============================================================
# macOS用 Step 1 パイプラインスクリプト
# 米国市場クローズ後（日本時間早朝）に実行
# 1) distribution_diagnostics 生成
# 2) distribution_validation 生成
# 3) vol_state_diagnostics 生成
# これらは run_gap_distribution.sh / compute_gap_adjusted_distribution.py の入力。
# ============================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR="${PROJECT_DIR}/var/logs"

mkdir -p "${LOG_DIR}"
DATESTR=$(date +%Y%m%d)
LOG_FILE="${LOG_DIR}/distribution_diagnostics_${DATESTR}.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Step 1 pipeline 開始 ===" >> "${LOG_FILE}"

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
PIPELINE_DIR="${PROJECT_DIR}/live/pipeline_data"
TODAY="$(date +%Y-%m-%d)"

# ---------------------------------------------------------------------------
# Step 1a: distribution_diagnostics
# ---------------------------------------------------------------------------
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Step 1a: distribution_diagnostics ===" >> "${LOG_FILE}"
set +e
PYTHONPATH=src "${PYTHON_BIN}" tools/production/compute_structured_prediction_covariance.py \
    --config configs/production/production.yaml \
    --model production_residual_blpx \
    --start "2020-01-01" \
    --end "${TODAY}" \
    --results-dir var/live/pipeline_data/diagnostics_weights \
    --output-dir var/live/pipeline_data/distribution_diagnostics \
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
LATEST_DIAG_DIR=$(ls -td ${PIPELINE_DIR}/distribution_diagnostics/*/ 2>/dev/null | grep -v '/latest/' | head -1)
if [ -n "${LATEST_DIAG_DIR}" ]; then
    ln -sfn "$(basename ${LATEST_DIAG_DIR})" ${PIPELINE_DIR}/distribution_diagnostics/latest
    echo "[INFO] Updated distribution_diagnostics latest symlink -> ${LATEST_DIAG_DIR}" >> "${LOG_FILE}"
fi

# --- 鮮度チェック: 出力 diagnostics が本日取引日を含んでいるか ---
# distribution_panel_long.csv の列: signal_date, trade_date, ticker, ...
# 本日 T の決定には trade_date == TODAY の行が必要。
if [ -n "${LATEST_DIAG_DIR}" ] && [ -f "${LATEST_DIAG_DIR}/distribution_panel_long.csv" ]; then
    MAX_TRADE_DATE=$(awk -F, 'NR>1 {gsub(/ .*/, "", $2); print $2}' "${LATEST_DIAG_DIR}/distribution_panel_long.csv" | sort | tail -1)
    if [ "${MAX_TRADE_DATE}" != "${TODAY}" ]; then
        echo "[WARNING] distribution_diagnostics output is stale: max trade_date=${MAX_TRADE_DATE}, expected=${TODAY}" >> "${LOG_FILE}"
        echo "[WARNING] The US close data for the signal of ${TODAY} may not have been available at run time." >> "${LOG_FILE}"
        echo "[WARNING] Consider rerunning this script later, or delay the schedule to after US data is processed." >> "${LOG_FILE}"
        # Not failing here; gap_distribution.sh will fail its own freshness check and produce flat position.
    else
        echo "[INFO] Freshness check passed: max trade_date=${MAX_TRADE_DATE}" >> "${LOG_FILE}"
    fi
fi

# ---------------------------------------------------------------------------
# Step 1b: distribution_validation
# ---------------------------------------------------------------------------
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Step 1b: distribution_validation ===" >> "${LOG_FILE}"
set +e
PYTHONPATH=src "${PYTHON_BIN}" tools/validation/validate_distribution_prediction_step1.py \
    --input-dir "${LATEST_DIAG_DIR}" \
    --output-dir var/live/pipeline_data/distribution_validation \
    --start "2020-01-01" \
    --end "${TODAY}" \
    --cost-mode all \
    --include-vol-state true \
    >> "${LOG_FILE}" 2>&1
VALIDATION_EXIT=$?
set -e

if [ ${VALIDATION_EXIT} -ne 0 ]; then
    echo "[ERROR] distribution validation computation failed (exit=${VALIDATION_EXIT})" >> "${LOG_FILE}"
    exit ${VALIDATION_EXIT}
fi

LATEST_VAL_DIR=$(ls -td ${PIPELINE_DIR}/distribution_validation/*/ 2>/dev/null | grep -v '/latest/' | head -1)
if [ -n "${LATEST_VAL_DIR}" ]; then
    ln -sfn "$(basename ${LATEST_VAL_DIR})" ${PIPELINE_DIR}/distribution_validation/latest
    echo "[INFO] Updated distribution_validation latest symlink -> ${LATEST_VAL_DIR}" >> "${LOG_FILE}"
fi

# ---------------------------------------------------------------------------
# Step 1c: vol_state_diagnostics
# ---------------------------------------------------------------------------
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Step 1c: vol_state_diagnostics ===" >> "${LOG_FILE}"
set +e
PYTHONPATH=src "${PYTHON_BIN}" tools/validation/diagnose_us_vol_states.py \
    --config configs/production/production.yaml \
    --model production_residual_blpx \
    --start "2020-01-01" \
    --end "${TODAY}" \
    --results-dir var/live/pipeline_data/diagnostics_weights \
    --output-dir var/live/pipeline_data/vol_state_diagnostics \
    --run-backtest-if-missing \
    --slippage-bps 5.0 \
    >> "${LOG_FILE}" 2>&1
VOLSTATE_EXIT=$?
set -e

if [ ${VOLSTATE_EXIT} -ne 0 ]; then
    echo "[ERROR] vol state diagnostics computation failed (exit=${VOLSTATE_EXIT})" >> "${LOG_FILE}"
    exit ${VOLSTATE_EXIT}
fi

LATEST_VOL_DIR=$(ls -td ${PIPELINE_DIR}/vol_state_diagnostics/*/ 2>/dev/null | grep -v '/latest/' | head -1)
if [ -n "${LATEST_VOL_DIR}" ]; then
    ln -sfn "$(basename ${LATEST_VOL_DIR})" ${PIPELINE_DIR}/vol_state_diagnostics/latest
    echo "[INFO] Updated vol_state_diagnostics latest symlink -> ${LATEST_VOL_DIR}" >> "${LOG_FILE}"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Step 1 pipeline 終了コード: 0 ===" >> "${LOG_FILE}"
exit 0
