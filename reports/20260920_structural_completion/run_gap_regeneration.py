"""Regenerate a small real-data gap window with a bounded deadline."""

from __future__ import annotations

import subprocess
import sys


def main() -> int:
    command = [
        sys.executable,
        "tools/research/compute_gap_adjusted_distribution.py",
        "--distribution-input-dir", "var/results/distribution_diagnostics/20260708_184812",
        "--validation-input-dir", "var/results/distribution_validation/20260614_235836",
        "--vol-state-panel", "var/results/vol_state_diagnostics/20260614_115821/state_panel.csv",
        "--results-dir", "var/live/pipeline_data/diagnostics_weights",
        "--output-dir", "var/results/20260920_structural_completion/r3_regenerated_gap",
        "--start", "2026-08-14",
        "--end", "2026-08-17",
        "--save-to-gap-store", "false",
        "--save-daily-matrices", "true",
        "--save-multi-horizon", "true",
        "--save-rank-reversal", "true",
        "--n-jobs", "1",
    ]
    try:
        return subprocess.run(command, check=False, timeout=900).returncode
    except subprocess.TimeoutExpired:
        return 124


if __name__ == "__main__":
    raise SystemExit(main())
