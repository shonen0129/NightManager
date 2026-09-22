#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TEST_ROOT="$(mktemp -d /private/tmp/leadlag-scheduler-test.XXXXXX)"
trap 'rm -rf "${TEST_ROOT}"' EXIT
mkdir -p "${TEST_ROOT}/bin"
cat > "${TEST_ROOT}/bin/launchctl" <<'EOF'
#!/bin/bash
set -euo pipefail
printf '%s\n' "$*" >> "${LEADLAG_LAUNCHCTL_LOG}"
EOF
chmod +x "${TEST_ROOT}/bin/launchctl"

export HOME="${TEST_ROOT}/home"
export LEADLAG_LAUNCHCTL_LOG="${TEST_ROOT}/launchctl.log"
export PATH="${TEST_ROOT}/bin:/usr/bin:/bin"

bash "${ROOT}/scripts/batch/setup_scheduler_macos.sh" > "${TEST_ROOT}/setup.log"

expected=(
  com.leadlag.update-market-data
  com.leadlag.distribution-diagnostics
  com.leadlag.decision
  com.leadlag.close
  com.leadlag.pnl_report
)
for label in "${expected[@]}"; do
  plist="${HOME}/Library/LaunchAgents/${label}.plist"
  test -f "${plist}"
  ! grep -q '__PROJECT_DIR__' "${plist}"
  grep -q "${ROOT}" "${plist}"
done
test "$(grep -c '^load ' "${LEADLAG_LAUNCHCTL_LOG}")" -eq 5
test "$(grep -c '^unload ' "${LEADLAG_LAUNCHCTL_LOG}")" -eq 5
echo "scheduler template generation and five-job registration: PASS"
