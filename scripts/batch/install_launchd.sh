#!/bin/bash
# ============================================================
# launchd ジョブのインストール・再読み込み
# ============================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
LAUNCH_AGENTS_DIR="${HOME}/Library/LaunchAgents"

mkdir -p "${LAUNCH_AGENTS_DIR}"

# --- update-market-data ---
cp "${PROJECT_DIR}/scripts/batch/com.leadlag.update-market-data.plist" "${LAUNCH_AGENTS_DIR}/"
launchctl unload "${LAUNCH_AGENTS_DIR}/com.leadlag.update-market-data.plist" 2>/dev/null || true
launchctl load "${LAUNCH_AGENTS_DIR}/com.leadlag.update-market-data.plist"
echo "[OK] com.leadlag.update-market-data installed (08:00 JST, Mon-Sat)"

# --- distribution-diagnostics ---
cp "${PROJECT_DIR}/scripts/batch/com.leadlag.distribution-diagnostics.plist" "${LAUNCH_AGENTS_DIR}/"
launchctl unload "${LAUNCH_AGENTS_DIR}/com.leadlag.distribution-diagnostics.plist" 2>/dev/null || true
launchctl load "${LAUNCH_AGENTS_DIR}/com.leadlag.distribution-diagnostics.plist"
echo "[OK] com.leadlag.distribution-diagnostics updated (08:15 JST, Mon-Sat)"

# --- decision (unchanged, 09:10 JST, Mon-Fri) ---
# --- close (unchanged, 14:50 JST, Mon-Fri) ---

echo ""
echo "=== 現在の launchd ジョブ ==="
launchctl list | grep leadlag || true

echo ""
echo "=== スケジュール確認 ==="
echo "08:00  update_market_data.sh        (Mon-Sat)"
echo "08:15  run_distribution_diagnostics.sh (Mon-Sat)"
echo "09:10  run_decision_v2.sh          (Mon-Fri)"
echo "14:50  run_close_positions.sh      (Mon-Fri)"
