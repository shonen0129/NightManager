#!/bin/bash
# ============================================================
# macOS用 V2発注自動化スクリプト
# leadlag decision v2 (朝9:00実行)
# Production Residual-BLPX-RA v2 (mu_over_sigma + RuleD)
# ============================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR="${PROJECT_DIR}/logs"
VENV_DIR="${PROJECT_DIR}/.venv-mac"

mkdir -p "${LOG_DIR}"

DATESTR=$(date +%Y%m%d)
LOG_FILE="${LOG_DIR}/decision_${DATESTR}.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === decision v2 開始 ===" >> "${LOG_FILE}"

# 仮想環境のPython
PYTHON_BIN="${VENV_DIR}/bin/python"
if [ -f "${PYTHON_BIN}" ]; then
    :
else
    echo "[ERROR] venv python not found: ${PYTHON_BIN}" >> "${LOG_FILE}"
    exit 1
fi

# スクリプト実行
cd "${PROJECT_DIR}"
set +e
PYTHONPATH=src "${PYTHON_BIN}" -c "
import sys
sys.path.insert(0, 'src')
from leadlag.execution.v2_bridge import run_v2_decision
run_v2_decision(
    config_path='configs/production/production.yaml',
    gap_input_dir='live/pipeline_data/gap_adjusted_distribution/latest',
    v1_weights_file='live/production_residual_blpx/v1_baseline_weights.csv',
    live_dir='live/production_residual_blpx',
    api_enable=True,
    api_dry_run=False,
    capital_from_wallet=True,
    text_output=True,
    output_root='results',
)
" >> "${LOG_FILE}" 2>&1
EXIT_CODE=$?
set -e
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === 終了コード: ${EXIT_CODE} ===" >> "${LOG_FILE}"
exit ${EXIT_CODE}
