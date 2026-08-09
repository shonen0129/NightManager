#!/bin/bash
# Run all tests in parallel (8 processes). ~8min vs ~32min serial.
# Heavy tests are distributed 1-per-process; remaining tests use pytest-xdist.

set -e
cd "$(dirname "$0")/.."

# Fail fast on undefined-name lint (prevents NameError regressions like 2026-07-27).
VENV_DIR="${VENV_DIR:-.venv-mac}"
RUFF="$VENV_DIR/bin/ruff"
if [ -x "$RUFF" ]; then
    echo "Running ruff F821 lint (undefined names)..."
    "$RUFF" check src/leadlag tools/production --select F821
else
    echo "WARNING: $RUFF not found, skipping F821 lint"
fi

LOGDIR=/tmp/pytest_parallel
mkdir -p "$LOGDIR"

echo "Starting 7-process parallel test run..."

PYTHON_BIN="$VENV_DIR/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
    echo "ERROR: venv python not found: $PYTHON_BIN"
    exit 1
fi

"$PYTHON_BIN" -m pytest tests/research/test_sprint0_diagnostics.py -q -n 0 > "$LOGDIR/p1.log" 2>&1 &
P1=$!
"$PYTHON_BIN" -m pytest tests/research/test_sprint0_qa.py -q -n 0 > "$LOGDIR/p2.log" 2>&1 &
P2=$!
"$PYTHON_BIN" -m pytest "tests/research/test_sprint1.py::test_backtest_simulation" -q -n 0 > "$LOGDIR/p3.log" 2>&1 &
P3=$!
"$PYTHON_BIN" -m pytest "tests/research/test_sprint1.py::test_calibration_rolling" -q -n 0 > "$LOGDIR/p4.log" 2>&1 &
P4=$!
"$PYTHON_BIN" -m pytest tests/research/test_sprint1.py --deselect "tests/research/test_sprint1.py::test_backtest_simulation" --deselect "tests/research/test_sprint1.py::test_calibration_rolling" -q -n 0 > "$LOGDIR/p5.log" 2>&1 &
P5=$!
"$PYTHON_BIN" -m pytest tests/integration/ -q -n auto > "$LOGDIR/p6.log" 2>&1 &
P6=$!
"$PYTHON_BIN" -m pytest tests/unit/ -q -n auto > "$LOGDIR/p7.log" 2>&1 &
P7=$!
"$PYTHON_BIN" -m pytest tests/research/ --ignore=tests/research/test_sprint0_diagnostics.py --ignore=tests/research/test_sprint0_qa.py --ignore=tests/research/test_sprint1.py -q -n auto > "$LOGDIR/p8.log" 2>&1 &
P8=$!

FAIL=0
for PID in $P1 $P2 $P3 $P4 $P5 $P6 $P7 $P8; do
    wait "$PID" || FAIL=1
done

echo ""
echo "=== P1 (sprint0_diagnostics) ===" && tail -1 "$LOGDIR/p1.log"
echo "=== P2 (sprint0_qa) ===" && tail -1 "$LOGDIR/p2.log"
echo "=== P3 (sprint1::backtest) ===" && tail -1 "$LOGDIR/p3.log"
echo "=== P4 (sprint1::calibration) ===" && tail -1 "$LOGDIR/p4.log"
echo "=== P5 (sprint1 rest) ===" && tail -1 "$LOGDIR/p5.log"
echo "=== P6 (integration) ===" && tail -1 "$LOGDIR/p6.log"
echo "=== P7 (unit rest) ===" && tail -1 "$LOGDIR/p7.log"
echo "=== P8 (research rest) ===" && tail -1 "$LOGDIR/p8.log"

if [ "$FAIL" -eq 0 ]; then
    echo ""
    echo "ALL PASSED"
else
    echo ""
    echo "SOME TESTS FAILED — check $LOGDIR/ for details"
    exit 1
fi
