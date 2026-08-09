#!/usr/bin/env python
"""Train and save the ML order-decision overlay model for production.

The model is trained on a historical PIT window and saved to a directory that
``run_daily_production_v2.py`` can load at runtime.  It is a one-off or
scheduled training step, not part of the daily execution path.

Example::

    python tools/production/train_ml_order_overlay.py \
        --train-start 2015-01-05 \
        --train-end 2024-12-31 \
        --gap-input-dir results/gap_adjusted_distribution/20260615_004113 \
        --output-dir models/ml_order_overlay/phase2_8
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.cache import load_df_exec_from_local_cache
from leadlag.models.ml_order_overlay import train_overlay_model
from leadlag.models.production_v2 import parse_run_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("TrainMLOrderOverlay")


def parse_arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the ML order-decision overlay model"
    )
    p.add_argument("--config", default="configs/production/production.yaml")
    p.add_argument("--train-start", default="2020-01-06")
    p.add_argument("--train-end", default="2024-12-31")
    p.add_argument(
        "--gap-input-dir",
        default=None,
        help="Directory containing mu_gap / omega_gap .npy files",
    )
    p.add_argument(
        "--output-dir",
        default="models/ml_order_overlay/phase2_8",
        help="Directory where model.pkl and metadata.json are saved",
    )
    p.add_argument("--no-per-ticker-interactions", action="store_true",
                   help="Disable per-ticker interaction features")
    p.add_argument("--no-ticker", action="store_true", help="Remove ticker categorical")
    p.add_argument("--use-classification", action="store_true")
    p.add_argument("--n-estimators", type=int, default=100)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--max-depth", type=int, default=3)
    p.add_argument("--num-leaves", type=int, default=20)
    p.add_argument("--min-child-samples", type=int, default=300)
    p.add_argument("--reg-alpha", type=float, default=0.5)
    p.add_argument("--reg-lambda", type=float, default=1.0)
    p.add_argument("--subsample", type=float, default=0.8)
    p.add_argument("--colsample-bytree", type=float, default=0.8)
    p.add_argument(
        "--p-trade-scale",
        type=float,
        default=1.0,
        help="Multiplier for the p_trade sigmoid. Default 1.0 (legacy). 2.0 centers the neutral prediction at 1.0 instead of 0.5.",
    )
    p.add_argument(
        "--target-type",
        choices=["raw", "residual", "residual_sign", "classification"],
        default="raw",
        help="Target type for LGBM overlay. 'raw' = side*realized - cost, 'residual' = raw - OLS(score), 'residual_sign' = sign of residual, 'classification' = profitable or not.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_arguments()

    config_path = ROOT / args.config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    per_ticker = not args.no_per_ticker_interactions

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
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
    }

    gap_input_dir: Path | None = None
    if args.gap_input_dir:
        gap_input_dir = (
            ROOT / args.gap_input_dir
            if not args.gap_input_dir.startswith("/")
            else Path(args.gap_input_dir)
        )
    else:
        default_gap = cfg.get("gap_distribution", {}).get("dir", "")
        if default_gap:
            gap_input_dir = (
                ROOT / default_gap if not default_gap.startswith("/") else Path(default_gap)
            )

    if gap_input_dir is None or not gap_input_dir.exists():
        logger.error("Gap input directory not found: %s", gap_input_dir)
        return 1

    logger.info("Loading df_exec...")
    df_exec = load_df_exec_from_local_cache()
    logger.info("df_exec loaded: %d rows", len(df_exec))

    output_dir = ROOT / args.output_dir

    train_overlay_model(
        df_exec=df_exec,
        gap_input_dir=gap_input_dir,
        run_cfg=parse_run_config(cfg),
        train_start=args.train_start,
        train_end=args.train_end,
        output_dir=output_dir,
        lgbm_kwargs=lgbm_kwargs,
        use_ticker=not args.no_ticker,
        use_classification=args.use_classification,
        per_ticker_interactions=per_ticker,
        p_trade_scale=args.p_trade_scale,
        target_type=args.target_type,
    )

    logger.info("Training complete. Model saved to %s", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
