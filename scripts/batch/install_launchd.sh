#!/bin/bash
# Compatibility entry point. Keep one launchd registration implementation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "${SCRIPT_DIR}/setup_scheduler_macos.sh" "$@"
