#!/bin/bash
# ============================================================
# macOS用 Gap調整済み予測分布生成スクリプト
# 米国市場クローズ後（日本時間早朝）に実行
# Step 5: compute_gap_adjusted_distribution.py
# ============================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR="${PROJECT_DIR}/logs"
VENV_DIR="${PROJECT_DIR}/.venv-mac"

mkdir -p "${LOG_DIR}"

DATESTR=$(date +%Y%m%d)
LOG_FILE="${LOG_DIR}/gap_distribution_${DATESTR}.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === gap distribution 開始 ===" >> "${LOG_FILE}"

PYTHON_BIN="${VENV_DIR}/bin/python"
if [ -f "${PYTHON_BIN}" ]; then
    :
else
    echo "[ERROR] venv python not found: ${PYTHON_BIN}" >> "${LOG_FILE}"
    exit 1
fi

cd "${PROJECT_DIR}"

# Step 1: distribution_diagnostics (Step 1) と distribution_validation は
# 事前計算済みの結果を再利用するため、最新のものを検索
PIPELINE_DIR="${PROJECT_DIR}/live/pipeline_data"
DIST_DIR=$(ls -td ${PIPELINE_DIR}/distribution_diagnostics/*/ 2>/dev/null | head -1)
VAL_DIR=$(ls -td ${PIPELINE_DIR}/distribution_validation/*/ 2>/dev/null | head -1)
VOL_STATE=$(ls -t ${PIPELINE_DIR}/vol_state_diagnostics/*/state_panel.csv 2>/dev/null | head -1)

if [ -z "${DIST_DIR}" ]; then
    echo "[ERROR] distribution_diagnostics not found. Run Step 1 first." >> "${LOG_FILE}"
    exit 1
fi

if [ -z "${VAL_DIR}" ]; then
    echo "[ERROR] distribution_validation not found. Run Step 1 first." >> "${LOG_FILE}"
    exit 1
fi

if [ -z "${VOL_STATE}" ]; then
    echo "[ERROR] vol_state_diagnostics not found." >> "${LOG_FILE}"
    exit 1
fi

echo "[INFO] Using distribution_diagnostics: ${DIST_DIR}" >> "${LOG_FILE}"
echo "[INFO] Using distribution_validation: ${VAL_DIR}" >> "${LOG_FILE}"
echo "[INFO] Using vol_state_panel: ${VOL_STATE}" >> "${LOG_FILE}"

# 当日の日付を取得（過去3日間をカバーして週末実行時の欠落を防ぐ）
TODAY=$(date +%Y-%m-%d)
TODAY_NUMERIC=$(date +%Y%m%d)
START_DATE=$(date -v-3d +%Y-%m-%d)

# 前回のlatest実績ディレクトリを保存（フォールバック用）
PREV_LATEST=""
if [ -L ${PIPELINE_DIR}/gap_adjusted_distribution/latest ]; then
    PREV_LATEST=$(readlink ${PIPELINE_DIR}/gap_adjusted_distribution/latest)
fi

# Step 2: gap調整済み分布の計算（過去3日間）
# 過去3日をカバーして週末実行時の直近営業日データ欠落を防ぐ
set +e
PYTHONPATH=src "${PYTHON_BIN}" tools/production/compute_gap_adjusted_distribution.py \
    --distribution-input-dir "${DIST_DIR}" \
    --validation-input-dir "${VAL_DIR}" \
    --vol-state-panel "${VOL_STATE}" \
    --config configs/production/production.yaml \
    --output-dir ${PIPELINE_DIR}/gap_adjusted_distribution \
    --start "${START_DATE}" \
    --end "${TODAY}" \
    --save-daily-matrices true \
    --save-multi-horizon false \
    --save-rank-reversal false \
    --compare-pre-gap false \
    >> "${LOG_FILE}" 2>&1

EXIT_CODE=$?
set -e

if [ ${EXIT_CODE} -ne 0 ]; then
    echo "[ERROR] gap distribution computation failed (exit=${EXIT_CODE})" >> "${LOG_FILE}"
    exit ${EXIT_CODE}
fi

# latest シンボリックリンクを更新 (exclude 'latest' symlink from matching)
LATEST_DIR=$(ls -td ${PIPELINE_DIR}/gap_adjusted_distribution/*/ 2>/dev/null | grep -v '/latest/' | head -1)
if [ -n "${LATEST_DIR}" ]; then
    ln -sfn "$(basename ${LATEST_DIR})" ${PIPELINE_DIR}/gap_adjusted_distribution/latest
    echo "[INFO] Updated latest symlink -> ${LATEST_DIR}" >> "${LOG_FILE}"
fi

# 当日の行列ファイルが生成されたか確認（非営業日などで空の場合はフォールバック）
MU_FILE="${PIPELINE_DIR}/gap_adjusted_distribution/latest/matrices/mu_gap_${TODAY_NUMERIC}.npy"
if [ ! -f "${MU_FILE}" ] && [ -n "${PREV_LATEST}" ]; then
    echo "[ALERT] Fallback triggered: Today's mu_gap not found. Copying from previous latest: ${PREV_LATEST}" >> "${LOG_FILE}"
    echo "[ALERT] This indicates the gap distribution computation may have failed or today is a non-trading day." >> "${LOG_FILE}"
    PREV_DIR="${PIPELINE_DIR}/gap_adjusted_distribution/${PREV_LATEST}"
    if [ -d "${PREV_DIR}/matrices" ]; then
        # 過去ファイルをコピー
        cp "${PREV_DIR}/matrices/"*.npy ${PIPELINE_DIR}/gap_adjusted_distribution/latest/matrices/ 2>/dev/null || true
        
        # コピーされた中で最新の日付のファイルを探し、TODAY_NUMERIC にコピーして配置する
        LATEST_MAT_DIR="${PIPELINE_DIR}/gap_adjusted_distribution/latest/matrices"
        PREV_DATE_FILE=$(ls -t "${LATEST_MAT_DIR}"/mu_gap_*.npy 2>/dev/null | grep -v "${TODAY_NUMERIC}" | head -1 || true)
        if [ -n "${PREV_DATE_FILE}" ]; then
            PREV_DATE=$(basename "${PREV_DATE_FILE}" | grep -o '[0-9]\{8\}' || true)
            if [ -n "${PREV_DATE}" ] && [ "${PREV_DATE}" != "${TODAY_NUMERIC}" ]; then
                echo "[INFO] Copying matrices from ${PREV_DATE} to ${TODAY_NUMERIC}..." >> "${LOG_FILE}"
                # Core matrices (always required)
                cp "${LATEST_MAT_DIR}/mu_gap_${PREV_DATE}.npy" "${LATEST_MAT_DIR}/mu_gap_${TODAY_NUMERIC}.npy" 2>/dev/null || true
                cp "${LATEST_MAT_DIR}/omega_gap_${PREV_DATE}.npy" "${LATEST_MAT_DIR}/omega_gap_${TODAY_NUMERIC}.npy" 2>/dev/null || true
                # Multi-horizon matrices (optional - only if they exist)
                if [ -f "${LATEST_MAT_DIR}/mu_gap_h3_${PREV_DATE}.npy" ]; then
                    cp "${LATEST_MAT_DIR}/mu_gap_h3_${PREV_DATE}.npy" "${LATEST_MAT_DIR}/mu_gap_h3_${TODAY_NUMERIC}.npy" 2>/dev/null || true
                    cp "${LATEST_MAT_DIR}/omega_gap_h3_${PREV_DATE}.npy" "${LATEST_MAT_DIR}/omega_gap_h3_${TODAY_NUMERIC}.npy" 2>/dev/null || true
                fi
                if [ -f "${LATEST_MAT_DIR}/mu_gap_h5_${PREV_DATE}.npy" ]; then
                    cp "${LATEST_MAT_DIR}/mu_gap_h5_${PREV_DATE}.npy" "${LATEST_MAT_DIR}/mu_gap_h5_${TODAY_NUMERIC}.npy" 2>/dev/null || true
                    cp "${LATEST_MAT_DIR}/omega_gap_h5_${PREV_DATE}.npy" "${LATEST_MAT_DIR}/omega_gap_h5_${TODAY_NUMERIC}.npy" 2>/dev/null || true
                fi
                # Rank reversal signal (optional - only if it exists)
                if [ -f "${LATEST_MAT_DIR}/rank_reversal_${PREV_DATE}.npy" ]; then
                    cp "${LATEST_MAT_DIR}/rank_reversal_${PREV_DATE}.npy" "${LATEST_MAT_DIR}/rank_reversal_${TODAY_NUMERIC}.npy" 2>/dev/null || true
                fi
                echo "[ALERT] Fallback completed: Copied matrices from ${PREV_DATE} to ${TODAY_NUMERIC}" >> "${LOG_FILE}"
            fi
        fi
    fi
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === gap distribution 終了コード: ${EXIT_CODE} ===" >> "${LOG_FILE}"
exit ${EXIT_CODE}
