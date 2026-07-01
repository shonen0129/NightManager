#!/bin/bash
# ============================================================
# macOS launchd 自動スケジューラ セットアップ
# 使用方法: bash scripts/setup_scheduler_macos.sh
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LAUNCH_AGENT_DIR="${HOME}/Library/LaunchAgents"

echo "============================================"
echo " 日米ラグ 自動スケジューラ セットアップ (macOS)"
echo "============================================"
echo ""
echo "プロジェクトディレクトリ: ${PROJECT_DIR}"
echo ""

# ログディレクトリ作成
mkdir -p "${PROJECT_DIR}/logs"
echo "[OK] ログディレクトリ: ${PROJECT_DIR}/logs"

# LaunchAgents ディレクトリ作成
mkdir -p "${LAUNCH_AGENT_DIR}"

# --- タスク1: Decision (毎朝 9:05) ---
PLIST_DECISION="${LAUNCH_AGENT_DIR}/com.leadlag.decision.plist"
sed "s|__PROJECT_DIR__|${PROJECT_DIR}|g" "${SCRIPT_DIR}/com.leadlag.decision.plist" > "${PLIST_DECISION}"
echo "[OK] Decision plist: ${PLIST_DECISION} (毎朝 9:05)"

# --- タスク2: Close (毎日 14:50) ---
PLIST_CLOSE="${LAUNCH_AGENT_DIR}/com.leadlag.close.plist"
sed "s|__PROJECT_DIR__|${PROJECT_DIR}|g" "${SCRIPT_DIR}/com.leadlag.close.plist" > "${PLIST_CLOSE}"
echo "[OK] Close plist: ${PLIST_CLOSE} (毎日 14:50)"

# --- launchd に登録 ---
# 既存のものがあればアンロード
launchctl unload "${PLIST_DECISION}" 2>/dev/null || true
launchctl unload "${PLIST_CLOSE}" 2>/dev/null || true

# ロード
launchctl load "${PLIST_DECISION}"
launchctl load "${PLIST_CLOSE}"

echo ""
echo "============================================"
echo " セットアップ完了！"
echo "============================================"
echo ""
echo "登録済みジョブ:"
echo "  com.leadlag.decision  — 毎朝 9:05 (月-金)"
echo "  com.leadlag.close     — 毎日 14:50 (月-金)"
echo ""
echo "ログ出力先: ${PROJECT_DIR}/logs/"
echo ""
echo "手動テスト実行:"
echo "  bash ${SCRIPT_DIR}/run_decision.sh"
echo "  bash ${SCRIPT_DIR}/run_close_positions.sh"
echo ""
echo "launchd 状態確認:"
echo "  launchctl list | grep leadlag"
echo ""
echo "登録解除:"
echo "  launchctl unload ${PLIST_DECISION}"
echo "  launchctl unload ${PLIST_CLOSE}"
echo ""
