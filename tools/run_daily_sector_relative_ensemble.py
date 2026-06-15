#!/usr/bin/env python
"""Daily execution script for Sector Relative Ensemble (SRE) Model.

Loads config, runs standard operational daily pipeline, and writes output csv/audit files.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

# Add src/ to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.fetcher import download_data
from leadlag.data.preprocessor import preprocess_data
from leadlag.data.tickers import JP_TICKERS, TOPIX_TICKER
from leadlag.models.sre import SectorRelativeEnsembleModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Sector Relative Ensemble Daily Operational Runner")
    parser.add_argument("--config", default="configs/production.yaml", help="Path to YAML config file")
    parser.add_argument("--signal-date", default="latest", help="Date of signal generation (YYYY-MM-DD or 'latest')")
    parser.add_argument("--output-dir", default="live/sector_relative_ensemble/", help="Directory for live execution files")
    parser.add_argument("--dry-run", action="store_true", help="Simulate execution without placing actual orders")
    parser.add_argument("--current-weights-csv", default=None, help="Optional CSV of current portfolio weights")
    return parser.parse_args()


def main():
    args = parse_arguments()

    # 1. Load config file
    config_path = ROOT / args.config
    if config_path.exists():
        logger.info(f"Loading YAML config from: {config_path}")
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    else:
        logger.warning(f"Config path {config_path} not found. Running with defaults.")
        cfg = {}

    # Setup directories
    out_dir = Path(args.output_dir) if args.output_dir.startswith("live") else ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    log_file_path = out_dir / "run_log.txt"
    file_handler = logging.FileHandler(log_file_path, mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s]: %(message)s"))
    logging.getLogger().addHandler(file_handler)

    logger.info("=== Daily SRE Execution Started ===")
    if args.dry_run:
        logger.info("[DRY RUN MODE ENABLED] No orders will be submitted.")

    # 2. Download and Preprocess Data
    logger.info("Downloading historical market data...")
    raw_data = download_data(beta_window=60)
    logger.info("Preprocessing market data...")
    df_exec = preprocess_data(raw_data, beta_window=60)

    # Compute TOPIX returns
    topix_close = raw_data["jp_close"][TOPIX_TICKER].copy()
    topix_open = raw_data["jp_open"][TOPIX_TICKER].copy()
    topix_close.index = pd.to_datetime(topix_close.index).tz_localize(None).normalize()
    topix_open.index = pd.to_datetime(topix_open.index).tz_localize(None).normalize()
    r_topix_oc = topix_close / topix_open - 1.0
    df_exec["topix_oc_return"] = r_topix_oc.reindex(df_exec.index).values
    df_exec["topix_cc_trade"] = (1.0 + df_exec["topix_night_return"]) * (1.0 + df_exec["topix_oc_return"]) - 1.0

    # Load current weights if provided
    current_weights = {tk: 0.0 for tk in JP_TICKERS}
    if args.current_weights_csv:
        csv_path = Path(args.current_weights_csv)
        if csv_path.exists():
            logger.info(f"Loading current weights from {csv_path}")
            curr_df = pd.read_csv(csv_path)
            for _, row in curr_df.iterrows():
                ticker = str(row["ticker"])
                if ticker in current_weights:
                    current_weights[ticker] = float(row.get("weight", 0.0))
        else:
            logger.warning(f"Current weights CSV {csv_path} not found. Assuming flat position (0.0).")

    # 3. Instantiate the correct model class and generate daily decisions
    model_name = cfg.get("model", {}).get("name", "sector_relative_ensemble")
    if model_name in ["production_p8p3_blpx", "sector_relative_ensemble_blp_enhanced"]:
        from leadlag.models.sector_relative_ensemble_blp_enhanced import SectorRelativeEnsembleBLPEnhancedModel
        model = SectorRelativeEnsembleBLPEnhancedModel(cfg)
    else:
        model = SectorRelativeEnsembleModel(cfg)
    from leadlag.execution.decision import generate_daily_decision_results
    daily_results = generate_daily_decision_results(model, df_exec, trade_date=args.signal_date, current_weights=current_weights)

    # 4. Save Outputs
    # signals
    daily_results["signal_df"].to_csv(out_dir / "latest_signal.csv", index=False)
    # weights
    daily_results["weights_df"].to_csv(out_dir / "latest_weights.csv", index=False)
    # orders
    daily_results["orders_df"].to_csv(out_dir / "latest_orders.csv", index=False)

    # audit (run single step checks)
    audit_info = {
        "trade_date": daily_results["trade_date"].strftime("%Y-%m-%d"),
        "signal_date": str(daily_results["sig_date"]),
        "audit_checks": {
            "weight_sum_neutral": bool(abs(daily_results["weights_df"]["weight"].sum()) < 1e-10),
            "long_weights_nonnegative": bool((daily_results["weights_df"][daily_results["weights_df"]["side"] == "LONG"]["weight"] >= 0).all()),
            "short_weights_nonpositive": bool((daily_results["weights_df"][daily_results["weights_df"]["side"] == "SHORT"]["weight"] <= 0).all()),
            "no_nan_weights": bool(not daily_results["weights_df"]["weight"].isna().any()),
            "ticker_order_correct": bool(list(daily_results["weights_df"]["ticker"]) == list(JP_TICKERS)),
        }
    }
    audit_info["all_passed"] = all(audit_info["audit_checks"].values())

    with open(out_dir / "latest_audit.json", "w") as f:
        json.dump(audit_info, f, indent=4)

    logger.info(f"Daily execution files generated successfully in {out_dir}")
    logger.info(f"All Audit Checks Passed: {audit_info['all_passed']}")
    logger.info("=== Daily SRE Execution Completed ===")


if __name__ == "__main__":
    main()
