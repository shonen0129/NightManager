#!/bin/bash
# Fast unit-only test run (~1 min). Excludes integration and slow tests.
# Uses pytest markers introduced in pyproject.toml.
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

exec "$PYTHON_BIN" -m pytest tests/ -m "not integration and not slow" -q -n auto $EXTRA_ARGS
