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
DIST_DIR=$(ls -td ${PIPELINE_DIR}/distribution_diagnostics/*/ 2>/dev/null | grep -v '/latest/' | head -1)
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

# 当日の日付を取得
TODAY=$(date +%Y-%m-%d)
TODAY_NUMERIC=$(date +%Y%m%d)

# --- 鮮度チェック: distribution_diagnostics が当日取引日を含んでいるか ---
# 本日 T の gap 分布を計算するには、diagnostics に trade_date == TODAY の行が
# 含まれている必要がある。trade_date T は signal_date T-1（前日米国クローズ）
# のデータを使うため、T が欠けている = 最新 US クローズが未処理。
PANEL_LONG="${DIST_DIR}/distribution_panel_long.csv"
EXPECTED_TRADE_DATE="${TODAY}"
if [ -f "${PANEL_LONG}" ]; then
    # distribution_panel_long.csv の列: signal_date, trade_date, ticker, ...
    # trade_date は第2列
    MAX_TRADE_DATE=$(awk -F, 'NR>1 {gsub(/ .*/, "", $2); print $2}' "${PANEL_LONG}" | sort | tail -1)
    if [ "${MAX_TRADE_DATE}" != "${EXPECTED_TRADE_DATE}" ]; then
        echo "[ERROR] distribution_diagnostics is stale: max trade_date=${MAX_TRADE_DATE}, expected=${EXPECTED_TRADE_DATE}" >> "${LOG_FILE}"
        echo "[ERROR] The US close data for the signal of ${EXPECTED_TRADE_DATE} is not yet processed." >> "${LOG_FILE}"
        echo "[ERROR] Please update market data and rerun distribution_diagnostics before gap_adjusted_distribution." >> "${LOG_FILE}"
        exit 1
    fi
    echo "[INFO] Freshness check passed: max trade_date=${MAX_TRADE_DATE} (expected ${EXPECTED_TRADE_DATE})" >> "${LOG_FILE}"
else
    echo "[WARNING] distribution_panel_long.csv not found: ${PANEL_LONG}. Skipping freshness check." >> "${LOG_FILE}"
fi

# 前回のlatest実績ディレクトリを保存（フォールバック用）
PREV_LATEST=""
if [ -L ${PIPELINE_DIR}/gap_adjusted_distribution/latest ]; then
    PREV_LATEST=$(readlink ${PIPELINE_DIR}/gap_adjusted_distribution/latest)
fi

# 過去3日間をデフォルトの開始日とする（週末実行時の欠落を防ぐ）
DEFAULT_START=$(date -v-3d +%Y-%m-%d)
START_DATE="${DEFAULT_START}"

# 前回のlatestに既存のmu_gap行列があれば、不足日のみ再計算するよう開始日を調整
if [ -n "${PREV_LATEST}" ] && [ -d "${PIPELINE_DIR}/gap_adjusted_distribution/${PREV_LATEST}/matrices" ]; then
    PREV_MAT="${PIPELINE_DIR}/gap_adjusted_distribution/${PREV_LATEST}/matrices"
    # 過去5営業日のmu_gapファイルを確認し、最新の存在日を取得
    LATEST_EXISTING_DATE=""
    for d in $(date -v-0d +%Y%m%d) $(date -v-1d +%Y%m%d) $(date -v-2d +%Y%m%d) $(date -v-3d +%Y%m%d) $(date -v-4d +%Y%m%d) $(date -v-5d +%Y%m%d) $(date -v-6d +%Y%m%d) $(date -v-7d +%Y%m%d); do
        if [ -f "${PREV_MAT}/mu_gap_${d}.npy" ]; then
            LATEST_EXISTING_DATE="${d}"
            break
        fi
    done
    if [ -n "${LATEST_EXISTING_DATE}" ]; then
        # 既存最新日の翌日を開始日とする（重複計算を避ける）
        # date -v は BSD/macOS date
        EXISTING_FMT="${LATEST_EXISTING_DATE:0:4}-${LATEST_EXISTING_DATE:4:2}-${LATEST_EXISTING_DATE:6:2}"
        START_DATE=$(date -j -v+1d -f "%Y-%m-%d" "${EXISTING_FMT}" +%Y-%m-%d 2>/dev/null || echo "${DEFAULT_START}")
        # 開始日が当日を超えないようクランプ（再実行時など）
        if [[ "${START_DATE}" > "${TODAY}" ]]; then
            START_DATE="${TODAY}"
            echo "[INFO] Existing matrix found for today; recomputing from ${START_DATE}" >> "${LOG_FILE}"
        else
            echo "[INFO] Found existing mu_gap up to ${EXISTING_FMT}, starting from ${START_DATE}" >> "${LOG_FILE}"
        fi
    fi
fi

# Step 2: gap調整済み分布の計算
# 不足営業日のみ再計算（前回latestにmu_gapが存在する日はスキップ）
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

# PIT IR履歴のマージ: 正本 full_history_diagnostics.csv に新しい行を追加
# compute_gap_adjusted_distribution.pyは過去数日分のみ計算するため、
# 新しいdiagnostics CSVには数行しかない。RuleD PIT binningは252日以上の
# 履歴を必要とするため、正本の full_history_diagnostics.csv を更新する。
# load_pit_ir_history はこの正本ファイルを優先して読み込む。
CANONICAL_DIAG="${PIPELINE_DIR}/gap_adjusted_distribution/full_history_diagnostics.csv"
NEW_DIAG="${LATEST_DIR}/portfolio_gap_distribution_diagnostics.csv"
if [ -f "${NEW_DIAG}" ]; then
    if [ -f "${CANONICAL_DIAG}" ]; then
        PYTHONPATH=src "${PYTHON_BIN}" scripts/batch/merge_gap_distribution_diagnostics.py "${CANONICAL_DIAG}" "${NEW_DIAG}" --output "${CANONICAL_DIAG}" >> "${LOG_FILE}" 2>&1
        echo "[INFO] Merged new diagnostics into canonical full_history_diagnostics.csv" >> "${LOG_FILE}"
    else
        cp "${NEW_DIAG}" "${CANONICAL_DIAG}"
        echo "[INFO] Initialized canonical full_history_diagnostics.csv from new diagnostics" >> "${LOG_FILE}"
    fi
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
