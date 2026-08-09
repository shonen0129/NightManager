#!/bin/bash
# ============================================================
# macOS launchd 自動スケジューラ セットアップ
# 使用方法: bash scripts/batch/setup_scheduler_macos.sh
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LAUNCH_AGENT_DIR="${HOME}/Library/LaunchAgents"

echo "============================================"
echo " leadlag-fund 自動スケジューラ セットアップ (macOS)"
echo "============================================"
echo ""
echo "プロジェクトディレクトリ: ${PROJECT_DIR}"
echo ""

# ログディレクトリ作成
mkdir -p "${PROJECT_DIR}/var/logs"
echo "[OK] ログディレクトリ: ${PROJECT_DIR}/var/logs"

# LaunchAgents ディレクトリ作成
mkdir -p "${LAUNCH_AGENT_DIR}"

# --- タスク1: Distribution Diagnostics (Step 1) (毎朝 8:15, 月-土) ---
PLIST_DIST_DIAG="${LAUNCH_AGENT_DIR}/com.leadlag.distribution-diagnostics.plist"
sed "s|__PROJECT_DIR__|${PROJECT_DIR}|g" "${SCRIPT_DIR}/com.leadlag.distribution-diagnostics.plist" > "${PLIST_DIST_DIAG}"
echo "[OK] Distribution Diagnostics plist: ${PLIST_DIST_DIAG} (毎朝 8:15, 月-土)"

# --- タスク2: Decision (毎朝 9:10, 月-金) ---
PLIST_DECISION="${LAUNCH_AGENT_DIR}/com.leadlag.decision.plist"
sed "s|__PROJECT_DIR__|${PROJECT_DIR}|g" "${SCRIPT_DIR}/com.leadlag.decision.plist" > "${PLIST_DECISION}"
echo "[OK] Decision plist: ${PLIST_DECISION} (毎朝 9:10, 月-金)"

# --- タスク3: Close (毎日 14:50) ---
PLIST_CLOSE="${LAUNCH_AGENT_DIR}/com.leadlag.close.plist"
sed "s|__PROJECT_DIR__|${PROJECT_DIR}|g" "${SCRIPT_DIR}/com.leadlag.close.plist" > "${PLIST_CLOSE}"
echo "[OK] Close plist: ${PLIST_CLOSE} (毎日 14:50)"

# --- タスク4: P&L レポート (毎日 15:40) ---
PLIST_PNL_REPORT="${LAUNCH_AGENT_DIR}/com.leadlag.pnl_report.plist"
sed "s|__PROJECT_DIR__|${PROJECT_DIR}|g" "${SCRIPT_DIR}/com.leadlag.pnl_report.plist" > "${PLIST_PNL_REPORT}"
echo "[OK] P&L report plist: ${PLIST_PNL_REPORT} (毎日 15:40)"

# --- launchd に登録 ---
# 既存のものがあればアンロード
launchctl unload "${PLIST_DIST_DIAG}" 2>/dev/null || true
launchctl unload "${PLIST_DECISION}" 2>/dev/null || true
launchctl unload "${PLIST_CLOSE}" 2>/dev/null || true
launchctl unload "${PLIST_PNL_REPORT}" 2>/dev/null || true

# ロード
launchctl load "${PLIST_DIST_DIAG}"
launchctl load "${PLIST_DECISION}"
launchctl load "${PLIST_CLOSE}"
launchctl load "${PLIST_PNL_REPORT}"

echo ""
echo "============================================"
echo " セットアップ完了！"
echo "============================================"
echo ""
echo "登録済みジョブ:"
echo "  com.leadlag.distribution-diagnostics — 毎朝 8:15 (月-土)"
echo "  com.leadlag.decision                  — 毎朝 9:10 (月-金)"
echo "  com.leadlag.close                     — 毎日 14:50 (月-金)"
echo "  com.leadlag.pnl_report                — 毎日 15:40 (月-金)"
echo ""
echo "ログ出力先: ${PROJECT_DIR}/var/logs/"
echo ""
echo "手動テスト実行:"
echo "  bash ${SCRIPT_DIR}/run_distribution_diagnostics.sh"
echo "  bash ${SCRIPT_DIR}/run_gap_distribution.sh"
echo "  bash ${SCRIPT_DIR}/run_decision_v2.sh"
echo "  bash ${SCRIPT_DIR}/run_close_positions.sh"
echo "  bash ${SCRIPT_DIR}/run_pnl_report.sh"
echo ""
echo "launchd 状態確認:"
echo "  launchctl list | grep leadlag"
echo ""
echo "登録解除:"
echo "  launchctl unload ${PLIST_DIST_DIAG}"
echo "  launchctl unload ${PLIST_DECISION}"
echo "  launchctl unload ${PLIST_CLOSE}"
echo "  launchctl unload ${PLIST_PNL_REPORT}"
echo ""
