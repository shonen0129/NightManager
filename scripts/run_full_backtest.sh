#!/bin/bash
# Run the full-period V2 backtest in the background and log to /tmp/full_backtest.log.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

.venv/bin/python3 -m leadlag.cli backtest \
  --config configs/production/production.yaml \
  --start-date 2015-01-05 \
  --end-date 2026-08-13 \
  --gap-dir var/live/pipeline_data/gap_adjusted_distribution/20260731_024303 \
  --n-jobs -1 \
  --output-level minimal \
  --run-tag full_2015_20260813 \
  > /tmp/full_backtest.log 2>&1
