#!/usr/bin/env python3
"""本番 `run_daily_production_v2.py` と同じ設定（overlay 込み）の V2 バックテスト。"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.cache import load_df_exec_from_local_cache
from leadlag.execution.backtester import BacktestEngine
from leadlag.execution.config import load_config_from_yaml
from research.experiment_registry import Decision
from research.experiment_utils import record_backtest_experiment


def main():
    parser = argparse.ArgumentParser(description="V2 Exact-Production Backtest (with overlay)")
    parser.add_argument("--config", default="configs/production/production.yaml")
    parser.add_argument("--gap-dir", default="var/live/pipeline_data/gap_adjusted_distribution/20260731_024303")
    parser.add_argument("--start-date", default="2020-01-06")
    parser.add_argument("--end-date", default="latest")
    parser.add_argument("--overlay-model-dir", default="models/ml_order_overlay/phase2_8")
    parser.add_argument("--side-leverage", type=float, default=1.5)
    parser.add_argument("--output-dir", default="var/results/v2_backtest_exact_production")
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()

    config_path = ROOT / args.config
    gap_dir = ROOT / args.gap_dir
    overlay_model_dir = ROOT / args.overlay_model_dir
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    app_config = load_config_from_yaml(config_path)

    df_exec = load_df_exec_from_local_cache()
    logger.info("df_exec: %s rows", len(df_exec))

    results = BacktestEngine.run_v2_backtest(
        cfg=app_config,
        gap_input_dir=gap_dir,
        df_exec=df_exec,
        start_date=args.start_date,
        end_date=args.end_date,
        side_leverage=args.side_leverage,
        overlay_model_dir=overlay_model_dir,
        n_jobs=args.n_jobs,
    )

    # Save all backtest outputs
    results["weights"].to_csv(output_dir / "daily_weights.csv")
    results["daily_returns"].to_csv(output_dir / "daily_net_returns.csv", header=["net_return"])
    results["daily_returns_gross"].to_csv(output_dir / "daily_gross_returns.csv", header=["gross_return"])
    results["equity_curve"].to_csv(output_dir / "daily_equity_curve.csv", header=["equity"])
    results["drawdown"].to_csv(output_dir / "daily_drawdown.csv", header=["drawdown"])
    results["daily_turnover"].to_csv(output_dir / "daily_turnover.csv", header=["turnover"])
    results["daily_gross_exps"].to_csv(output_dir / "daily_gross.csv", header=["gross"])
    results["daily_costs"].to_csv(output_dir / "daily_costs_total.csv", header=["cost"])
    results["daily_slip_costs"].to_csv(output_dir / "daily_costs_slip.csv", header=["slip_cost"])
    results["daily_financing_costs"].to_csv(output_dir / "daily_costs_financing.csv", header=["financing_cost"])
    results["daily_borrow_costs"].to_csv(output_dir / "daily_costs_borrow.csv", header=["borrow_cost"])
    results["daily_reverse_costs"].to_csv(output_dir / "daily_costs_reverse.csv", header=["reverse_cost"])
    results["daily_overnight_returns"].to_csv(output_dir / "daily_overnight_returns.csv", header=["overnight_return"])
    results["daily_fallback"].to_csv(output_dir / "daily_fallback.csv", header=["fallback"])

    returns = results["daily_returns"]
    fb = results["daily_fallback"]
    valid = returns[~fb]
    n_valid = len(valid)
    n_fallback = int(fb.sum())
    total_ret = float(np.sum(valid))
    mean_ret = float(np.mean(valid)) if n_valid > 0 else 0.0
    std_ret = float(np.std(valid, ddof=1)) if n_valid > 1 else 0.0
    sharpe = mean_ret / std_ret * np.sqrt(252) if std_ret > 1e-8 else 0.0
    wealth = results["equity_curve"]
    mdd = float(results["drawdown"].min()) if len(results["drawdown"]) > 0 else 0.0
    avg_turnover = float(np.mean(results["daily_turnover"][~fb])) if n_valid > 0 else 0.0
    avg_gross = float(np.mean(results["daily_gross_exps"][~fb])) if n_valid > 0 else 0.0
    fb_rate = n_fallback / len(returns) * 100 if len(returns) > 0 else 0.0
    final_wealth = float(wealth.iloc[-1]) if len(wealth) > 0 else 1.0
    cagr = (final_wealth ** (252 / len(valid)) - 1) * 100 if n_valid > 0 else 0.0

    print("\n" + "=" * 60)
    print("=== V2 Exact-Production Backtest (with overlay) ===")
    print("=" * 60)
    print(f"  Period:        {returns.index[0].date()} -> {returns.index[-1].date()}")
    print(f"  Total days:    {len(returns)}")
    print(f"  Success:       {n_valid}")
    print(f"  Fallback:      {n_fallback} ({fb_rate:.1f}%)")
    print(f"  Sharpe:        {sharpe:.4f}")
    print(f"  Total Return:  {total_ret*100:.2f}%")
    print(f"  CAGR:          {cagr:.2f}%")
    print(f"  Final Wealth:  {final_wealth:,.2f}x")
    print(f"  AR (ann):      {mean_ret*252*100:.2f}%")
    print(f"  Vol (ann):     {std_ret*np.sqrt(252)*100:.2f}%")
    print(f"  Max DD:        {mdd*100:.2f}%")
    print(f"  Avg Turnover:  {avg_turnover:.4f}")
    print(f"  Avg Gross:     {avg_gross:.4f}")
    if n_valid > 0:
        print(f"  Avg Cost/day:  {float(np.mean(results['daily_costs'][~fb]))*10000:.2f} bps")
    print("=" * 60)

    # Save summary
    summary = {
        "period": f"{returns.index[0].date()} -> {returns.index[-1].date()}",
        "total_days": int(len(returns)),
        "fallback_days": n_fallback,
        "fallback_rate_pct": fb_rate,
        "sharpe": float(sharpe),
        "total_return_pct": float(total_ret * 100),
        "cagr_pct": float(cagr),
        "final_wealth": float(final_wealth),
        "annualized_return_pct": float(mean_ret * 252 * 100),
        "annualized_volatility_pct": float(std_ret * np.sqrt(252) * 100),
        "max_drawdown_pct": float(mdd * 100),
        "avg_turnover": float(avg_turnover),
        "avg_gross_exposure": float(avg_gross),
        "side_leverage": args.side_leverage,
        "overlay_model_dir": str(overlay_model_dir),
        "gap_dir": str(gap_dir),
    }
    import json
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    record_backtest_experiment(
        name=Path(__file__).stem,
        hypothesis="V2 exact-production backtest (overlay enabled) reproduces live production behavior.",
        app_config=app_config,
        results=results,
        decision=Decision.PENDING,
    )


if __name__ == "__main__":
    main()
