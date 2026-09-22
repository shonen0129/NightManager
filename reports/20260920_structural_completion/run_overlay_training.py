"""Run the production overlay trainer with an outer process deadline."""

from __future__ import annotations

import subprocess
import sys


def main() -> int:
    command = [
        sys.executable,
        "tools/production/train_ml_order_overlay.py",
        "--train-start",
        "2015-01-05",
        "--train-end",
        "2024-12-31",
        "--gap-input-dir",
        "var/live/pipeline_data/gap_adjusted_distribution/20260723_091009",
        "--output-dir",
        "var/results/20260920_structural_completion/ml_overlay_retrained",
    ]
    try:
        completed = subprocess.run(command, timeout=240, check=False)
    except subprocess.TimeoutExpired:
        return 124
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
