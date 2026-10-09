#!/bin/bash
# Run all tests in parallel (10 processes). ~8min vs ~32min serial.
# Heavy tests are distributed 1-per-process; remaining tests use pytest-xdist.

set -e
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# Fail fast on undefined-name lint (prevents NameError regressions like 2026-07-27).
if [ -f ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
else
    PYTHON_BIN="python3"
fi

# Bound the entire parallel run and all pytest-xdist workers with a process-
# group watchdog. Set the guard flag in the child so it does not wrap itself.
if [ "${LEADLAG_TEST_RUNNER_GUARDED:-0}" != "1" ]; then
    exec env PYTHONPATH="$PROJECT_ROOT/src" "$PYTHON_BIN" "$PROJECT_ROOT/scripts/tools/phase_deadline.py" \
        --label parallel_tests --timeout "${LEADLAG_TEST_SUITE_TIMEOUT_SECONDS:-1800}" \
        --grace 10 --log /tmp/pytest_parallel_guard.json -- \
        env LEADLAG_TEST_RUNNER_GUARDED=1 bash "$0" "$@"
fi

RUFF_CMD=()
if [ -x ".venv/bin/ruff" ]; then
    RUFF_CMD=(".venv/bin/ruff")
elif command -v ruff >/dev/null 2>&1; then
    RUFF_CMD=("$(command -v ruff)")
elif "$PYTHON_BIN" -m ruff --version >/dev/null 2>&1; then
    RUFF_CMD=("$PYTHON_BIN" -m ruff)
else
    echo "ERROR: ruff is required for the full validation run" >&2
    exit 2
fi

echo "Running ruff F821 lint (undefined names)..."
"${RUFF_CMD[@]}" check src/leadlag tools/production tools/research --select F821

LOGDIR=/tmp/pytest_parallel
mkdir -p "$LOGDIR"

echo "Starting 10-process parallel test run (including features and regression tests)..."

"$PYTHON_BIN" -m pytest tests/regression/test_v2_baseline.py -q -n 0 \
    > "$LOGDIR/p0.log" 2>&1 &
P0=$!

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
"$PYTHON_BIN" -m pytest tests/features/ -q -n auto > "$LOGDIR/p9.log" 2>&1 &
P9=$!

FAIL=0
for PID in $P0 $P1 $P2 $P3 $P4 $P5 $P6 $P7 $P8 $P9; do
    wait "$PID" || FAIL=1
done

echo ""
echo "=== P0 (regression baseline) ===" && tail -1 "$LOGDIR/p0.log"
echo "=== P1 (sprint0_diagnostics) ===" && tail -1 "$LOGDIR/p1.log"
echo "=== P2 (sprint0_qa) ===" && tail -1 "$LOGDIR/p2.log"
echo "=== P3 (sprint1::backtest) ===" && tail -1 "$LOGDIR/p3.log"
echo "=== P4 (sprint1::calibration) ===" && tail -1 "$LOGDIR/p4.log"
echo "=== P5 (sprint1 rest) ===" && tail -1 "$LOGDIR/p5.log"
echo "=== P6 (integration) ===" && tail -1 "$LOGDIR/p6.log"
echo "=== P7 (unit rest) ===" && tail -1 "$LOGDIR/p7.log"
echo "=== P8 (research rest) ===" && tail -1 "$LOGDIR/p8.log"
echo "=== P9 (features) ===" && tail -1 "$LOGDIR/p9.log"

if [ "$FAIL" -eq 0 ]; then
    echo ""
    echo "ALL PASSED"
else
    echo ""
    echo "SOME TESTS FAILED — check $LOGDIR/ for details"
    exit 1
fi
