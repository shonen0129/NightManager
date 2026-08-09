#!/bin/bash
# ============================================================
# launchd ジョブのインストール・再読み込み
# ============================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
LAUNCH_AGENTS_DIR="${HOME}/Library/LaunchAgents"

mkdir -p "${LAUNCH_AGENTS_DIR}"

# --- update-market-data ---
PLIST_UPDATE="${LAUNCH_AGENTS_DIR}/com.leadlag.update-market-data.plist"
sed "s|__PROJECT_DIR__|${PROJECT_DIR}|g" "${PROJECT_DIR}/scripts/batch/com.leadlag.update-market-data.plist" > "${PLIST_UPDATE}"
launchctl unload "${PLIST_UPDATE}" 2>/dev/null || true
launchctl load "${PLIST_UPDATE}"
echo "[OK] com.leadlag.update-market-data installed (08:00 JST, Mon-Sat)"

# --- distribution-diagnostics ---
PLIST_DIST="${LAUNCH_AGENTS_DIR}/com.leadlag.distribution-diagnostics.plist"
sed "s|__PROJECT_DIR__|${PROJECT_DIR}|g" "${PROJECT_DIR}/scripts/batch/com.leadlag.distribution-diagnostics.plist" > "${PLIST_DIST}"
launchctl unload "${PLIST_DIST}" 2>/dev/null || true
launchctl load "${PLIST_DIST}"
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
