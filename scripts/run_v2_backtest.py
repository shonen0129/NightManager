#!/usr/bin/env python3
"""Run V2 production backtest via BacktestEngine.run_v2_backtest."""
import argparse
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.cache import load_df_exec_from_local_cache
from leadlag.execution.backtester import BacktestEngine
from leadlag.reporting.metrics import calculate_metrics


def main():
    parser = argparse.ArgumentParser(description="V2 Production Backtest (gap-adjusted distribution)")
    parser.add_argument("--config", default="configs/production/production.yaml", help="Path to V2 config YAML")
    parser.add_argument("--gap-dir", default="live/pipeline_data/gap_adjusted_distribution/latest",
                        help="Directory with mu_gap/omega_gap matrices (default: latest symlink)")
    parser.add_argument("--start-date", default="2015-01-05", help="Backtest start date")
    parser.add_argument("--end-date", default="latest", help="Backtest end date ('latest' for last available)")
    parser.add_argument("--output-dir", default="src/results/v2_backtest", help="Output directory")
    parser.add_argument("--side-leverage", type=float, default=1.5,
                        help="Notional leverage matching allocator DEFAULT_SIDE_LEVERAGE (default: 1.5)")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Number of parallel workers for per-date computation (1=sequential, -1=all cores)")
    args = parser.parse_args()

    config_path = ROOT / args.config
    gap_dir = ROOT / args.gap_dir if not Path(args.gap_dir).is_absolute() else Path(args.gap_dir)
    output_dir = ROOT / args.output_dir

    logger.info("Loading config from %s", config_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    logger.info("Gap dir: %s", gap_dir)
    logger.info("Loading df_exec from local cache...")
    df_exec = load_df_exec_from_local_cache()
    logger.info("df_exec: %s rows, %s -> %s", len(df_exec), df_exec.index[0], df_exec.index[-1])

    results = BacktestEngine.run_v2_backtest(
        cfg=cfg,
        gap_input_dir=gap_dir,
        df_exec=df_exec,
        start_date=args.start_date,
        end_date=args.end_date,
        side_leverage=args.side_leverage,
        n_jobs=args.n_jobs,
    )

    # Save artifacts
    output_dir.mkdir(parents=True, exist_ok=True)
    results["daily_returns"].to_csv(output_dir / "daily_net_returns.csv", header=["net_return"])
    results["equity_curve"].to_csv(output_dir / "daily_equity_curve.csv", header=["equity"])
    results["drawdown"].to_csv(output_dir / "daily_drawdown.csv", header=["drawdown"])
    results["daily_turnover"].to_csv(output_dir / "daily_turnover.csv", header=["turnover"])
    results["daily_gross_exps"].to_csv(output_dir / "daily_gross.csv", header=["gross"])
    results["daily_costs"].to_csv(output_dir / "daily_costs.csv", header=["cost"])
    results["daily_fallback"].to_csv(output_dir / "daily_fallback.csv", header=["fallback"])

    # Metrics (exclude fallback days for Sharpe/DD)
    returns = results["daily_returns"]
    fb = results["daily_fallback"]
    valid = returns[~fb]
    n_valid = len(valid)
    n_fallback = int(fb.sum())
    total_ret = float(np.sum(valid))
    mean_ret = float(np.mean(valid)) if n_valid > 0 else 0.0
    std_ret = float(np.std(valid, ddof=1)) if n_valid > 1 else 0.0
    sharpe = mean_ret / std_ret * np.sqrt(252) if std_ret > 1e-8 else 0.0
    cum = np.cumsum(valid.values)
    running_max = np.maximum.accumulate(cum)
    max_dd = float(np.min(cum - running_max)) if len(cum) > 0 else 0.0
    avg_turnover = float(np.mean(results["daily_turnover"][~fb])) if n_valid > 0 else 0.0
    avg_gross = float(np.mean(results["daily_gross_exps"][~fb])) if n_valid > 0 else 0.0
    fb_rate = n_fallback / len(returns) * 100 if len(returns) > 0 else 0.0

    print("\n" + "=" * 60)
    print("=== V2 Backtest Results (ProductionV2 — gap-adjusted) ===")
    print("=" * 60)
    print(f"  Period:        {returns.index[0].date()} -> {returns.index[-1].date()}")
    print(f"  Total days:    {len(returns)}")
    print(f"  Success:       {n_valid}")
    print(f"  Fallback:      {n_fallback} ({fb_rate:.1f}%)")
    print(f"  Sharpe:        {sharpe:.4f}")
    print(f"  Total Return:  {total_ret*100:.2f}%")
    print(f"  AR (ann):      {mean_ret*252*100:.2f}%")
    print(f"  Vol (ann):     {std_ret*np.sqrt(252)*100:.2f}%")
    print(f"  Max DD:        {max_dd*100:.2f}%")
    print(f"  Avg Turnover:  {avg_turnover:.4f}")
    print(f"  Avg Gross:     {avg_gross:.4f}")
    if n_valid > 0:
        print(f"  Avg Cost/day:  {float(np.mean(results['daily_costs'][~fb]))*10000:.2f} bps")
    print("=" * 60)


if __name__ == "__main__":
    main()
