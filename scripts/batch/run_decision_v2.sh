#!/bin/bash
# ============================================================
# macOS用 V2発注自動化スクリプト（gap distribution + decision 統合）
# 朝9:10実行: 立花API価格でgap行列生成 → 発注
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

cd "${PROJECT_DIR}"

# --- Step 1: gap distribution（立花API価格注入） ---
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [1/2] gap distribution 開始" >> "${LOG_FILE}"
set +e
bash scripts/batch/run_gap_distribution.sh >> "${LOG_FILE}" 2>&1
GAP_EXIT=$?
set -e
if [ ${GAP_EXIT} -ne 0 ]; then
    echo "[ERROR] gap distribution failed (exit=${GAP_EXIT}). Proceeding to decision (will be flat)." >> "${LOG_FILE}"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [1/2] gap distribution 完了" >> "${LOG_FILE}"
fi

# --- Step 2: decision v2 ---
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [2/2] decision v2 開始" >> "${LOG_FILE}"
set +e
PYTHONPATH=src "${PYTHON_BIN}" -c "
import sys
sys.path.insert(0, 'src')
from leadlag.execution.v2_bridge import run_v2_decision
run_v2_decision(
    config_path='configs/production/production.yaml',
    gap_input_dir='live/pipeline_data/gap_adjusted_distribution/latest',
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
