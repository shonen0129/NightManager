#!/usr/bin/env python
"""Phase 2.5 experiment: LightGBM overlay with EMA smoothing and stronger regularization."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.cache import load_df_exec_from_local_cache
from experiments.ml_order_decision.phase2 import run_phase2_experiment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Phase 2.5 ML order decision overlay")
    parser.add_argument(
        "--gap-input-dir",
        default=str(ROOT / "results" / "gap_adjusted_distribution" / "20260615_004113"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "reports" / "ml_order_decision" / "phase2_5_results"),
    )
    parser.add_argument("--train-start", default="2020-01-06")
    parser.add_argument("--train-end", default="2022-12-31")
    parser.add_argument("--test-start", default="2023-01-01")
    parser.add_argument("--test-end", default="2024-12-31")
    parser.add_argument("--p-trade-ema-span", type=float, default=3.0)
    parser.add_argument("--no-ticker", action="store_true", help="Remove raw ticker categorical feature")
    parser.add_argument("--use-classification", action="store_true", help="Use binary positive-contribution target")
    parser.add_argument("--per-ticker-interactions", action="store_true", help="Add explicit one-hot ticker x score/gap interaction features")
    parser.add_argument("--n-jobs", type=int, default=-1)
    # LightGBM hyperparameters
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--num-leaves", type=int, default=20)
    parser.add_argument("--min-child-samples", type=int, default=300)
    parser.add_argument("--reg-alpha", type=float, default=0.5)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    args = parser.parse_args()

    lgbm_kwargs = {
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "num_leaves": args.num_leaves,
        "min_child_samples": args.min_child_samples,
        "reg_alpha": args.reg_alpha,
        "reg_lambda": args.reg_lambda,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
    }

    with open(ROOT / "configs" / "production" / "production.yaml") as f:
        cfg = yaml.safe_load(f)

    df_exec = load_df_exec_from_local_cache()

    result = run_phase2_experiment(
        df_exec=df_exec,
        gap_input_dir=Path(args.gap_input_dir),
        cfg=cfg,
        output_dir=Path(args.output_dir),
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        n_jobs=args.n_jobs,
        lgbm_kwargs=lgbm_kwargs,
        p_trade_ema_span=args.p_trade_ema_span,
        use_ticker=not args.no_ticker,
        use_classification=args.use_classification,
        per_ticker_interactions=args.per_ticker_interactions,
    )

    logger.info("--- Baseline metrics ---")
    for k, v in result["baseline_metrics"].items():
        logger.info("%s: %s", k, v)
    logger.info("--- Overlay metrics ---")
    for k, v in result["overlay_metrics"].items():
        logger.info("%s: %s", k, v)

    logger.info("Results saved to %s", args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
