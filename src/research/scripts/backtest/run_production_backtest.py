#!/usr/bin/env python
"""Run V2 production backtest with production config (production_residual_blpx).

Uses configs/production/production.yaml and BacktestEngine.run_v2_backtest
with pre-computed gap-adjusted distribution matrices.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.execution.backtester import BacktestEngine
from leadlag.execution.config import load_config_from_yaml
from leadlag.reporting.metrics import calculate_metrics
from research.backtest_common import load_execution_data

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
    parser.add_argument("--gap-dir", default=None,
                        help="Directory containing mu_gap/omega_gap .npy files. "
                             "Defaults to gap_distribution.dir in the YAML config.")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Number of parallel workers for V2 weight generation (1=sequential, -1=all cores)")
    args = parser.parse_args()

    config_path = ROOT / args.config
    logger.info("Loading config from %s", config_path)
    app_config = load_config_from_yaml(config_path)
    strategy = app_config.strategy

    slippage_bps = strategy.slippage_bps
    overnight_alpha_long = strategy.overnight_alpha_long
    overnight_alpha_short = strategy.overnight_alpha_short
    buy_interest_annual = strategy.buy_interest_annual
    borrow_fee_annual = strategy.borrow_fee_annual
    reverse_fee_bps = strategy.reverse_fee_bps

    logger.info("[1/4] Downloading/loading market data...")
    df_exec = load_execution_data(
        beta_window=strategy.beta_window,
        beta_ewma_halflife=strategy.beta_ewma_halflife,
        beta_shrinkage=strategy.beta_shrinkage,
        beta_winsor_sigma=strategy.beta_winsor_sigma,
    )

    logger.info("[2/4] Resolving V2 gap distribution...")
    if args.gap_dir is None:
        gap_dir = app_config.gap_distribution_dir
    else:
        gap_dir = args.gap_dir
    gap_input_dir = ROOT / gap_dir if gap_dir and not Path(gap_dir).is_absolute() else Path(gap_dir or "/nonexistent")
    if not gap_input_dir.exists():
        logger.warning("Gap input dir not found: %s. V2 backtest will fall back to flat positions.", gap_input_dir)
        gap_input_dir = None
    else:
        gap_input_dir = str(gap_input_dir)

    logger.info("[3/4] Running V2 production backtest: start=%s, slippage=%.1f bps, alpha_long=%.2f, alpha_short=%.2f",
                args.start_date, slippage_bps, overnight_alpha_long, overnight_alpha_short)
    results = BacktestEngine.run_v2_backtest(
        cfg=app_config,
        gap_input_dir=gap_input_dir,
        df_exec=df_exec,
        start_date=args.start_date,
        slippage_bps=slippage_bps,
        overnight_alpha_long=overnight_alpha_long,
        overnight_alpha_short=overnight_alpha_short,
        buy_interest_annual=buy_interest_annual,
        borrow_fee_annual=borrow_fee_annual,
        reverse_fee_bps=reverse_fee_bps,
        n_jobs=args.n_jobs,
    )

    logger.info("[4/4] Computing metrics...")
    metrics = calculate_metrics(results["daily_returns"])

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    results["daily_returns"].to_csv(out_dir / "daily_net_returns.csv", header=["net_return"])
    results["daily_returns_gross"].to_csv(out_dir / "daily_gross_returns.csv", header=["gross_return"])
    results["equity_curve"].to_csv(out_dir / "daily_equity_curve.csv", header=["equity"])
    results["drawdown"].to_csv(out_dir / "daily_drawdown.csv", header=["drawdown"])
    results["daily_turnover"].to_csv(out_dir / "daily_turnover.csv", header=["turnover"])
    results["daily_gross_exps"].to_csv(out_dir / "daily_gross_exposure.csv", header=["gross_exposure"])
    results["daily_costs"].to_csv(out_dir / "daily_costs.csv", header=["cost"])
    results["daily_slip_costs"].to_csv(out_dir / "daily_slip_costs.csv", header=["slip_cost"])
    results["daily_financing_costs"].to_csv(out_dir / "daily_financing_costs.csv", header=["financing_cost"])
    results["daily_borrow_costs"].to_csv(out_dir / "daily_borrow_costs.csv", header=["borrow_cost"])
    results["daily_reverse_costs"].to_csv(out_dir / "daily_reverse_costs.csv", header=["reverse_cost"])
    results["weights"].to_csv(out_dir / "daily_weights.csv")

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
