#!/bin/bash
# Fast unit-only test run (~1 min). Runs tests under tests/unit/ and
# tests/features/, excluding slow / integration tests via markers.
#
# Usage::
#
#     bash scripts/run_tests_unit_only.sh
#     bash scripts/run_tests_unit_only.sh -v

set -e
cd "$(dirname "$0")/.."

if [ -f ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
else
    PYTHON_BIN="python3"
fi

exec env PYTHONPATH=src "$PYTHON_BIN" -m leadlag.execution.phase_deadline \
    --label unit_tests --timeout "${LEADLAG_UNIT_TEST_TIMEOUT_SECONDS:-600}" \
    --grace 10 --log /tmp/pytest_unit_guard.json -- \
    "$PYTHON_BIN" -m pytest tests/unit tests/features \
    -m "unit and not slow and not integration" \
    -q -n auto "$@"
