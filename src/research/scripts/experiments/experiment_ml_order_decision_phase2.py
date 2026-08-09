#!/usr/bin/env python
"""Phase 2 experiment entry point for ML order decision overlay (LightGBM)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.cache import load_df_exec_from_local_cache
from research.experiments.ml_order_decision.phase2 import run_phase2_experiment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Phase 2 ML order decision overlay (LightGBM)")
    parser.add_argument(
        "--gap-input-dir",
        default=str(ROOT / "results" / "gap_adjusted_distribution" / "20260615_004113"),
        help="Directory with mu_gap / Omega_gap .npy files",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "reports" / "ml_order_decision" / "phase2_results"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--train-start",
        default="2020-01-06",
        help="Training start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--train-end",
        default="2022-12-31",
        help="Training end date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--test-start",
        default="2023-01-01",
        help="Test start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--test-end",
        default="2024-12-31",
        help="Test end date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel workers for V2 backtest (-1 = all cores)",
    )
    args = parser.parse_args()

    logger.info("Loading production config...")
    with open(ROOT / "configs" / "production" / "production.yaml") as f:
        cfg = yaml.safe_load(f)

    logger.info("Loading df_exec...")
    df_exec = load_df_exec_from_local_cache()

    gap_input_dir = Path(args.gap_input_dir)
    output_dir = Path(args.output_dir)

    logger.info("Running Phase 2 experiment...")
    result = run_phase2_experiment(
        df_exec=df_exec,
        gap_input_dir=gap_input_dir,
        cfg=cfg,
        output_dir=output_dir,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        n_jobs=args.n_jobs,
    )

    logger.info("--- Baseline metrics ---")
    for k, v in result["baseline_metrics"].items():
        logger.info("%s: %s", k, v)

    logger.info("--- Overlay metrics ---")
    for k, v in result["overlay_metrics"].items():
        logger.info("%s: %s", k, v)

    logger.info("Results saved to %s", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
