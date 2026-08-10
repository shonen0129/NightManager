#!/bin/bash
# Recompute gap-adjusted distribution with vol_adjusted_target=true (baseline)
# using the same Step 1 inputs as the false run.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${PROJECT_DIR}"

DIST_DIR=$(ls -td var/live/pipeline_data/distribution_diagnostics/*/ 2>/dev/null | grep -v '/latest/' | head -1)
VAL_DIR=$(ls -td var/live/pipeline_data/distribution_validation/*/ 2>/dev/null | head -1)
VOL_STATE=$(ls -t var/live/pipeline_data/vol_state_diagnostics/*/state_panel.csv 2>/dev/null | head -1)

echo "[INFO] distribution_diagnostics: ${DIST_DIR}"
echo "[INFO] distribution_validation: ${VAL_DIR}"
echo "[INFO] vol_state_panel: ${VOL_STATE}"

python3 tools/research/compute_gap_adjusted_distribution.py \
  --distribution-input-dir "${DIST_DIR}" \
  --validation-input-dir "${VAL_DIR}" \
  --vol-state-panel "${VOL_STATE}" \
  --config configs/experiments/vol_adjusted_true.yaml \
  --output-dir var/live/pipeline_data/gap_adjusted_distribution \
  --start 2020-01-01 \
  --end latest \
  --save-daily-matrices true \
  --save-multi-horizon true \
  --save-rank-reversal true \
  --compare-pre-gap false \
  --use-tachibana-prices false \
  --n-jobs 1
