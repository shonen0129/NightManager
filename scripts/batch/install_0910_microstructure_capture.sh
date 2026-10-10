#!/bin/bash
# Install or refresh only the independent read-only 09:10 LaunchAgent.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LABEL="com.leadlag.microstructure-0910"
AGENT_DIR="${HOME}/Library/LaunchAgents"
INSTALLED_PLIST="${AGENT_DIR}/${LABEL}.plist"
TEMP_PLIST="${INSTALLED_PLIST}.tmp.$$"
DOMAIN="gui/$(id -u)"

if [ "$(uname -s)" != "Darwin" ]; then
    echo "This LaunchAgent installer requires macOS." >&2
    exit 2
fi

mkdir -p "${AGENT_DIR}" "${PROJECT_DIR}/var/logs"
sed "s|__PROJECT_DIR__|${PROJECT_DIR}|g" \
    "${SCRIPT_DIR}/${LABEL}.plist" > "${TEMP_PLIST}"
/usr/bin/plutil -lint "${TEMP_PLIST}"
chmod 644 "${TEMP_PLIST}"

if launchctl print "${DOMAIN}/${LABEL}" >/dev/null 2>&1; then
    launchctl bootout "${DOMAIN}/${LABEL}"
fi
mv "${TEMP_PLIST}" "${INSTALLED_PLIST}"
launchctl bootstrap "${DOMAIN}" "${INSTALLED_PLIST}"
launchctl print "${DOMAIN}/${LABEL}" >/dev/null

echo "Installed ${LABEL} for weekdays at 09:10 JST."
echo "Output: ${PROJECT_DIR}/var/shadow_runs/ml_overlay_value/microstructure"
echo "Log: ${PROJECT_DIR}/var/logs/microstructure_0910_launchd.log"
