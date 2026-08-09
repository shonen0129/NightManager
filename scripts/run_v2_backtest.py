#!/usr/bin/env python3
"""Deprecated V2 production backtest wrapper.

This entry point is preserved for backward compatibility. New code should use
``python3 -m leadlag.cli backtest`` directly, which offers the same (and
additional) options.
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    warnings.warn(
        "scripts/run_v2_backtest.py is deprecated; use "
        "'python3 -m leadlag.cli backtest' instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    parser = argparse.ArgumentParser(description="V2 Production Backtest (deprecated wrapper)")
    parser.add_argument("--config", default="configs/production/production.yaml", help="Path to V2 config YAML")
    parser.add_argument("--gap-dir", default="var/live/pipeline_data/gap_adjusted_distribution/latest",
                        help="Directory with mu_gap/omega_gap matrices (default: latest symlink)")
    parser.add_argument("--start-date", default="2015-01-05", help="Backtest start date")
    parser.add_argument("--end-date", default="latest", help="Backtest end date ('latest' for last available)")
    parser.add_argument("--output-dir", default="var/results/v2_backtest", help="Output root")
    parser.add_argument("--side-leverage", type=float, default=None, help="Side leverage")
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of parallel workers")
    args = parser.parse_args()

    from leadlag.execution.backtest import run_production

    run_production(
        start_date=args.start_date,
        end_date=args.end_date,
        output_root=args.output_dir,
        run_tag="v2_backtest",
        skip_chart=False,
        config_path=args.config,
        gap_input_dir=args.gap_dir,
        side_leverage=args.side_leverage,
        n_jobs=args.n_jobs,
        data_source="cache",
        output_level="detailed",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
