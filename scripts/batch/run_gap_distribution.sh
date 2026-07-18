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
    --save-multi-horizon true \
    --save-rank-reversal true \
    --compare-pre-gap false \
    --use-tachibana-prices true \
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

# PIT IR履歴のマージ: 過去のdiagnostics CSVを新しいlatestに統合
# compute_gap_adjusted_distribution.pyは過去3日分のみ計算するため、
# 新しいdiagnostics CSVには数行しかない。RuleD PIT binningは252日以上の
# 履歴を必要とするため、前回のlatestのdiagnostics CSVをマージする。
NEW_DIAG="${LATEST_DIR}/portfolio_gap_distribution_diagnostics.csv"
if [ -n "${PREV_LATEST}" ] && [ -f "${PIPELINE_DIR}/gap_adjusted_distribution/${PREV_LATEST}/portfolio_gap_distribution_diagnostics.csv" ]; then
    PREV_DIAG="${PIPELINE_DIR}/gap_adjusted_distribution/${PREV_LATEST}/portfolio_gap_distribution_diagnostics.csv"
    if [ -f "${NEW_DIAG}" ]; then
        PYTHONPATH=src "${PYTHON_BIN}" -c "
import pandas as pd
old = pd.read_csv('${PREV_DIAG}')
new = pd.read_csv('${NEW_DIAG}')
combined = pd.concat([old, new], ignore_index=True)
combined = combined.drop_duplicates(subset='trade_date', keep='last')
combined = combined.sort_values('trade_date').reset_index(drop=True)
combined.to_csv('${NEW_DIAG}', index=False)
print(f'Merged diagnostics: {len(old)} old + {len(new)} new -> {len(combined)} total')
" >> "${LOG_FILE}" 2>&1
        echo "[INFO] Merged PIT IR history into new diagnostics CSV" >> "${LOG_FILE}"
    else
        cp "${PREV_DIAG}" "${NEW_DIAG}"
        echo "[INFO] Copied previous diagnostics CSV (new one not found)" >> "${LOG_FILE}"
    fi
fi

# 過去のmh行列・rank_reversal行列を前回latestからコピー
# ライブ実行は過去3日分のみ計算するため、それ以前のmh行列・rank_reversal行列は
# 前回のlatestからコピーする。これらはPIT-safe（一度計算されると不変）。
if [ -n "${PREV_LATEST}" ] && [ -d "${PIPELINE_DIR}/gap_adjusted_distribution/${PREV_LATEST}/matrices" ]; then
    PREV_MAT="${PIPELINE_DIR}/gap_adjusted_distribution/${PREV_LATEST}/matrices"
    NEW_MAT="${LATEST_DIR}/matrices"
    # mh行列（mu_gap_h*, omega_gap_h*）
    for f in "${PREV_MAT}"/mu_gap_h*_2*.npy "${PREV_MAT}"/omega_gap_h*_2*.npy; do
        [ -f "$f" ] || continue
        fname=$(basename "$f")
        if [ ! -f "${NEW_MAT}/${fname}" ]; then
            cp "$f" "${NEW_MAT}/${fname}"
        fi
    done
    # rank_reversal行列
    for f in "${PREV_MAT}"/rank_reversal_2*.npy; do
        [ -f "$f" ] || continue
        fname=$(basename "$f")
        if [ ! -f "${NEW_MAT}/${fname}" ]; then
            cp "$f" "${NEW_MAT}/${fname}"
        fi
    done
    echo "[INFO] Copied historical mh and rank_reversal matrices from ${PREV_LATEST}" >> "${LOG_FILE}"
fi

# 当日の行列ファイルが生成されたか確認
# 前日行列のコピーは行わない — 前日のgap行列で発注すると誤ったポジションとなるリスクがあるため
# 当日の行列がない場合は decision_v2 が flat position (w_final=0) を返すのが正しい挙動
MU_FILE="${PIPELINE_DIR}/gap_adjusted_distribution/latest/matrices/mu_gap_${TODAY_NUMERIC}.npy"
if [ ! -f "${MU_FILE}" ]; then
    echo "[WARNING] Today's mu_gap_${TODAY_NUMERIC}.npy not found. Decision will return flat position (no trading)." >> "${LOG_FILE}"
    echo "[WARNING] This indicates the gap distribution computation did not produce today's matrices." >> "${LOG_FILE}"
    echo "[WARNING] Possible causes: (1) etf_data.pkl cache stale (2) Step 1 omega_struct missing (3) non-trading day" >> "${LOG_FILE}"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === gap distribution 終了コード: ${EXIT_CODE} ===" >> "${LOG_FILE}"
exit ${EXIT_CODE}
