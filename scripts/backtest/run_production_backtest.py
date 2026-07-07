#!/usr/bin/env python
"""Run backtest with production config (production_residual_blpx).

Uses configs/production.yaml and SectorRelativeEnsembleBLPEnhancedModel
through the standard BacktestEngine with overnight holding and cost parameters.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from experiments.backtest_common import load_execution_data
from leadlag.execution.backtester import BacktestEngine
from leadlag.models.sector_relative_ensemble_blp_enhanced import (
    SectorRelativeEnsembleBLPEnhancedModel,
)
from leadlag.reporting.metrics import calculate_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Production Residual-BLPX Backtest")
    parser.add_argument("--config", default="configs/production/production.yaml", help="Path to config YAML")
    parser.add_argument("--start-date", default="2015-01-05", help="Backtest start date")
    parser.add_argument("--output-dir", default="results/production_backtest", help="Output directory")
    args = parser.parse_args()

    config_path = ROOT / args.config
    logger.info("Loading config from %s", config_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    costs = cfg.get("costs", {})
    slippage_bps = float(costs.get("slippage_bps_per_side", 5.0))
    overnight_alpha_long = float(costs.get("overnight_alpha_long", 0.75))
    overnight_alpha_short = float(costs.get("overnight_alpha_short", 0.5))
    buy_interest_annual = float(costs.get("buy_interest_annual", 0.025))
    borrow_fee_annual = float(costs.get("borrow_fee_annual", 0.0115))
    reverse_fee_bps = float(costs.get("reverse_fee_bps", 2.0))

    beta_window = cfg.get("residualization", {}).get("beta_window", 60)
    beta_ewma_halflife = cfg.get("residualization", {}).get("beta_ewma_halflife")
    beta_shrinkage = cfg.get("residualization", {}).get("beta_shrinkage", 0.0)
    beta_winsor_sigma = cfg.get("residualization", {}).get("beta_winsor_sigma")

    logger.info("[1/4] Downloading/loading market data...")
    df_exec = load_execution_data(
        beta_window=beta_window,
        beta_ewma_halflife=beta_ewma_halflife,
        beta_shrinkage=beta_shrinkage,
        beta_winsor_sigma=beta_winsor_sigma,
    )

    logger.info("[2/4] Building production model (Residual-BLPX)...")
    model = SectorRelativeEnsembleBLPEnhancedModel(cfg)

    logger.info("[3/4] Running backtest: start=%s, slippage=%.1f bps, alpha_long=%.2f, alpha_short=%.2f",
                args.start_date, slippage_bps, overnight_alpha_long, overnight_alpha_short)
    results = BacktestEngine.run_backtest(
        model,
        df_exec=df_exec,
        start_date=args.start_date,
        slippage_bps=slippage_bps,
        overnight_alpha_long=overnight_alpha_long,
        overnight_alpha_short=overnight_alpha_short,
        buy_interest_annual=buy_interest_annual,
        borrow_fee_annual=borrow_fee_annual,
        reverse_fee_bps=reverse_fee_bps,
    )

    logger.info("[4/4] Computing metrics...")
    metrics = calculate_metrics(results["daily_returns"])

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    results["daily_returns"].to_csv(out_dir / "daily_net_returns.csv", header=["net_return"])
    results["equity_curve"].to_csv(out_dir / "daily_equity_curve.csv", header=["equity"])
    results["drawdown"].to_csv(out_dir / "daily_drawdown.csv", header=["drawdown"])
    results["daily_turnover"].to_csv(out_dir / "daily_turnover.csv", header=["turnover"])

    print("\n=== Production Backtest Results (Residual-BLPX) ===")
    for key, v in metrics.items():
        if key in ["AR", "RISK", "MDD", "Total Return"]:
            print(f"  {key}: {v*100:.2f}%")
        elif key == "Sharpe":
            print(f"  {key}: {v:.4f}")
        else:
            print(f"  {key}: {v:.2f}")

    logger.info("Artifacts saved in: %s", out_dir)


if __name__ == "__main__":
    main()
