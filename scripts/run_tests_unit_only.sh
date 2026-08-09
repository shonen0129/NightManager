#!/bin/bash
# Fast unit-only test run (~1 min). Runs tests under tests/unit/ and
# tests/features/, excluding the known slow sprint backtest files.
#
# Usage::
#
#     bash scripts/run_tests_unit_only.sh
#     bash scripts/run_tests_unit_only.sh -v

set -e
cd "$(dirname "$0")/.."

VENV_DIR="${VENV_DIR:-.venv-mac}"
PYTHON_BIN="$VENV_DIR/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

EXTRA_ARGS="$@"

exec "$PYTHON_BIN" -m pytest tests/unit tests/features \
    --ignore=tests/unit/test_sprint0_diagnostics.py \
    --ignore=tests/unit/test_sprint0_qa.py \
    --ignore=tests/unit/test_sprint1.py \
    -q -n auto $EXTRA_ARGS
