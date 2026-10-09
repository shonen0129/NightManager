#!/bin/bash
# ============================================================
# macOS用 V2発注自動化スクリプト（gap distribution + decision 統合）
# 朝9:10実行: 立花API価格でgap行列生成 → decision（通常モードでは発注）
# Production Residual-BLPX-RA v2 (mu_over_sigma + RuleD)
# ============================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR="${PROJECT_DIR}/var/logs"

# The guard owns the whole gap-generation + decision process.  It uses a
# process-group deadline and a durable single-flight lease. The child receives
# a lease owner token which the execution entry verifies against SQLite.
if [ -z "${LEADLAG_LEASE_OWNER:-}" ]; then
    if [ -f "${PROJECT_DIR}/.venv/bin/python" ]; then
        GUARD_PYTHON="${PROJECT_DIR}/.venv/bin/python"
    else
        GUARD_PYTHON="$(command -v python3)"
    fi
    DATESTR=$(date +%Y%m%d)
    exec env PYTHONPATH="${PROJECT_DIR}/src" \
        "${GUARD_PYTHON}" -m leadlag.execution.job_guard \
        --scope "live:production_v2" \
        --timeout "${LEADLAG_DECISION_TIMEOUT_SECONDS:-1800}" \
        --grace "${LEADLAG_JOB_GRACE_SECONDS:-10}" \
        --state-db "${PROJECT_DIR}/var/live/pipeline_data/execution/execution_state.sqlite" \
        --guard-log "${PROJECT_DIR}/var/logs/job_guard/decision_${DATESTR}.json" \
        -- bash "$0" "$@"
fi

mkdir -p "${LOG_DIR}"
DATESTR=$(date +%Y%m%d)
LOG_FILE="${LOG_DIR}/decision_${DATESTR}.log"
PHASE_LOG_DIR="${PROJECT_DIR}/var/logs/job_guard"

run_phase() {
    local label="$1"
    local timeout_seconds="$2"
    shift 2
    PYTHONPATH=src "${PYTHON_BIN}" -m leadlag.execution.phase_deadline \
        --label "${label}" \
        --timeout "${timeout_seconds}" \
        --grace "${LEADLAG_JOB_GRACE_SECONDS:-10}" \
        --log "${PHASE_LOG_DIR}/decision_${DATESTR}_${label}.json" \
        -- "$@"
}

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === decision v2 開始 ===" >> "${LOG_FILE}"

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

SHADOW_ONLY_ARG=""
if [ "${LEADLAG_SHADOW_ONLY:-0}" = "1" ]; then
    SHADOW_ONLY_ARG="--shadow-only"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] shadow-only mode: no production portfolio outputs or orders" >> "${LOG_FILE}"
fi

# --- Read-only 09:10 market microstructure capture ---
# This phase only calls Tachibana market-price/LOB queries and never submits
# or cancels an order.  Set LEADLAG_CAPTURE_0910=0 to skip it.
if [ "${LEADLAG_CAPTURE_0910:-1}" = "1" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [0/2] 09:10 microstructure capture 開始" >> "${LOG_FILE}"
    set +e
    run_phase "microstructure_capture" "${LEADLAG_CAPTURE_PHASE_TIMEOUT_SECONDS:-120}" \
        "${PYTHON_BIN}" tools/validation/collect_0910_microstructure.py \
        --collect-live \
        --capture-only \
        --window-seconds "${LEADLAG_CAPTURE_WINDOW_SECONDS:-30}" \
        --output-dir "${LEADLAG_CAPTURE_OUTPUT_DIR:-var/shadow_runs/ml_overlay_value/microstructure}" \
        >> "${LOG_FILE}" 2>&1
    CAPTURE_EXIT=$?
    set -e
    if [ ${CAPTURE_EXIT} -ne 0 ]; then
        echo "[WARN] microstructure capture failed (exit=${CAPTURE_EXIT}); decision continues without changing order behavior." >> "${LOG_FILE}"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [0/2] 09:10 microstructure capture 完了" >> "${LOG_FILE}"
    fi
fi

# --- Step 1: gap distribution（立花API価格注入） ---
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [1/2] gap distribution 開始" >> "${LOG_FILE}"
set +e
run_phase "gap_distribution" "${LEADLAG_GAP_PHASE_TIMEOUT_SECONDS:-900}" \
    bash scripts/batch/run_gap_distribution.sh >> "${LOG_FILE}" 2>&1
GAP_EXIT=$?
set -e
if [ ${GAP_EXIT} -ne 0 ]; then
    echo "[ERROR] gap distribution failed (exit=${GAP_EXIT}). Decision will check today's cache, then permitted on-demand fallback, then flat." >> "${LOG_FILE}"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [1/2] gap distribution 完了" >> "${LOG_FILE}"
fi

# --- Step 2: decision v2 ---
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [2/2] decision v2 開始" >> "${LOG_FILE}"
set +e
run_phase "decision" "${LEADLAG_DECISION_PHASE_TIMEOUT_SECONDS:-900}" \
    "${PYTHON_BIN}" -m leadlag.cli decision \
    --config configs/production/production.yaml \
    --live-dir var/live/production_residual_blpx \
    --ml-overlay-shadow-dir var/shadow_runs/ml_overlay_research_20261009 \
    --ml-overlay-shadow-config configs/research/ml_overlay_forward_shadow_20261009.yaml \
    ${SHADOW_ONLY_ARG} \
    --api-enable \
    --capital-from-wallet \
    --text-output \
    --output-root var/results \
    >> "${LOG_FILE}" 2>&1
EXIT_CODE=$?
set -e
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === 終了コード: ${EXIT_CODE} ===" >> "${LOG_FILE}"
exit ${EXIT_CODE}
