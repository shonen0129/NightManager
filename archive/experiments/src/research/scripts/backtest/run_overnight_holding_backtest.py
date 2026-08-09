#!/usr/bin/env python
"""Overnight holding backtest with realistic costs: financing, borrow, reverse fee, gap risk.

Current model: daily full close (9:10 entry → 15:00 close, full round-trip cost).
Overnight model: hold fraction α of positions overnight, rebalance at 9:10 next day.

Return decomposition:
  intraday = w_t * r_910toClose(t)          (same as current)
  overnight = α * w_t * r_gap(t+1)           (close-to-open overnight return)

Cost model (realistic):
  slippage = slip * [2*(1-α)*gross + α*turnover]
  financing = α * long_exposure * financing_daily   (long side only)
  borrow   = α * short_exposure * borrow_daily       (short side only)
  reverse  = α * short_exposure * reverse_bps_daily  (逆日歩, short side only)

α=0 → current model (full close every day, no overnight costs)
α=1 → full overnight hold (rebalance + full financing/borrow/reverse)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from research.backtest_common import (
    TRADING_DAYS,
    CostParams,
    extended_metrics,
    load_execution_data,
    prepare_target_and_gap_returns,
    run_baseline_backtest,
    simulate_overnight_holding,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Overnight holding backtest with realistic costs")
    p.add_argument("--start-date", default="2015-01-05")
    p.add_argument("--slippage-bps", type=float, default=5.0)
    p.add_argument("--warmup", type=int, default=60)
    # Realistic cost parameters (matching CostConfig defaults)
    p.add_argument("--buy-interest-annual", type=float, default=0.025,
                   help="Annual financing rate for long positions (default: 2.5%%)")
    p.add_argument("--borrow-fee-annual", type=float, default=0.0115,
                   help="Annual stock borrow fee for short positions (default: 1.15%%)")
    p.add_argument("--reverse-fee-bps", type=float, default=2.0,
                   help="Daily reverse stock lending fee (逆日歩) in bps (default: 2.0 bps/day)")
    p.add_argument("--overnight-margin-mult", type=float, default=1.3,
                   help="Overnight margin multiplier vs intraday (default: 1.3x)")
    return p.parse_args()


def main():
    args = parse_args()

    logger.info("[1/4] Loading data and running baseline backtest...")
    df_exec = load_execution_data(beta_window=60)
    baseline = run_baseline_backtest(
        df_exec, start_date=args.start_date, slippage_bps=args.slippage_bps
    )

    weights = baseline["weights"]
    daily_returns = baseline["daily_returns"]
    sim_dates = weights.index

    logger.info("Baseline: %d trading days", len(daily_returns))

    target_returns_df, gap_returns_df = prepare_target_and_gap_returns(df_exec, sim_dates)

    costs = CostParams(
        slippage_bps=args.slippage_bps,
        buy_interest_annual=args.buy_interest_annual,
        borrow_fee_annual=args.borrow_fee_annual,
        reverse_fee_bps=args.reverse_fee_bps,
    )

    logger.info("[2/4] Running overnight holding backtests for multiple alpha values...")

    alphas = [0.0, 0.25, 0.50, 0.75, 1.0]
    results = {}

    for alpha in alphas:
        res = simulate_overnight_holding(
            weights, target_returns_df, gap_returns_df, alpha=alpha, costs=costs
        )
        results[alpha] = res
        m = extended_metrics(res["daily_returns"])
        label = "current" if alpha == 0 else f"α={alpha}"
        logger.info("  %s: AR=%.2f%%  Sharpe=%.4f  MDD=%.2f%%  Cost=%.4f%%/day  Turnover=%.3f",
                     label,
                     m.get("AR", 0) * 100,
                     m.get("Sharpe", 0),
                     m.get("MDD", 0) * 100,
                     res["daily_costs"].mean() * 100,
                     res["daily_turnover"].mean())

    logger.info("[3/4] Computing comparison metrics...")
    all_metrics = {}
    for alpha in alphas:
        all_metrics[alpha] = extended_metrics(results[alpha]["daily_returns"])

    logger.info("[4/4] Generating report...")

    # Print comparison table
    print("\n" + "=" * 120)
    print("  Overnight Holding Backtest — Position Carry-Over Analysis")
    print("=" * 120)
    print(f"  Start: {args.start_date}  |  Slippage: {args.slippage_bps} bps  |  Warmup: {args.warmup}d")
    print(f"  Financing: {args.buy_interest_annual*100:.2f}%% ann  |  Borrow: {args.borrow_fee_annual*100:.2f}%% ann  |  Reverse: {args.reverse_fee_bps:.1f} bps/day  |  Overnight margin: {args.overnight_margin_mult}x")
    print("-" * 120)
    print(f"  {'Metric':<18}", end="")
    for alpha in alphas:
        label = "Current(α=0)" if alpha == 0 else f"α={alpha}"
        print(f"  {label:>14}", end="")
    print()
    print("-" * 120)

    for key in ["AR", "RISK", "Sharpe", "MDD", "Total Return", "Hit Rate", "Calmar"]:
        print(f"  {key:<18}", end="")
        for alpha in alphas:
            v = all_metrics[alpha].get(key, np.nan)
            if key in ["AR", "RISK", "MDD", "Total Return"]:
                s = f"{v*100:.2f}%" if np.isfinite(v) else "N/A"
            elif key == "Sharpe":
                s = f"{v:.4f}" if np.isfinite(v) else "N/A"
            else:
                s = f"{v:.4f}" if np.isfinite(v) else "N/A"
            print(f"  {s:>14}", end="")
        print()

    print("-" * 120)

    # Cost and turnover comparison
    print(f"  {'Cost (/day)':<18}", end="")
    for alpha in alphas:
        v = results[alpha]["daily_costs"].mean() * 100
        print(f"  {f'{v:.4f}%':>14}", end="")
    print()

    print(f"  {'Cost (total)':<18}", end="")
    for alpha in alphas:
        v = results[alpha]["daily_costs"].sum() * 100
        print(f"  {f'{v:.2f}%':>14}", end="")
    print()

    print(f"  {'Turnover (avg)':<18}", end="")
    for alpha in alphas:
        v = results[alpha]["daily_turnover"].mean()
        print(f"  {f'{v:.3f}':>14}", end="")
    print()

    print(f"  {'Intraday ret':<18}", end="")
    for alpha in alphas:
        v = results[alpha]["daily_intraday"].mean() * 100
        print(f"  {f'{v:.4f}%':>14}", end="")
    print()

    print(f"  {'Overnight ret':<18}", end="")
    for alpha in alphas:
        v = results[alpha]["daily_overnight"].mean() * 100
        print(f"  {f'{v:.4f}%':>14}", end="")
    print()

    print(f"  {'Overnight std':<18}", end="")
    for alpha in alphas:
        v = results[alpha]["daily_overnight"].std() * 100
        print(f"  {f'{v:.4f}%':>14}", end="")
    print()

    print("-" * 120)

    # Sharpe delta from current
    base_sharpe = all_metrics[0.0].get("Sharpe", 0)
    print(f"  {'Δ Sharpe':<18}", end="")
    for alpha in alphas:
        s = all_metrics[alpha].get("Sharpe", 0)
        d = s - base_sharpe if np.isfinite(s) and np.isfinite(base_sharpe) else np.nan
        print(f"  {f'{d:+.4f}':>14}", end="")
    print()

    print(f"  {'Δ Sharpe (%)':<18}", end="")
    for alpha in alphas:
        s = all_metrics[alpha].get("Sharpe", 0)
        d = (s / base_sharpe - 1) * 100 if np.isfinite(s) and np.isfinite(base_sharpe) and base_sharpe != 0 else np.nan
        print(f"  {f'{d:+.1f}%':>14}", end="")
    print()

    print("=" * 120)

    # Cost breakdown by component
    print("\n  --- Cost Breakdown by Component (bps/day) ---")
    print(f"  {'Component':<18}", end="")
    for alpha in alphas:
        label = "Current" if alpha == 0 else f"α={alpha}"
        print(f"  {label:>14}", end="")
    print()
    print("  " + "-" * 100)

    for comp_key, comp_label in [("daily_slip", "Slippage"), ("daily_financing", "Financing"), ("daily_borrow", "Borrow"), ("daily_reverse", "Reverse (逆日歩)")]:
        print(f"  {comp_label:<18}", end="")
        for alpha in alphas:
            v = results[alpha][comp_key].mean() * 10000
            print(f"  {f'{v:.3f}':>14}", end="")
        print()

    print(f"  {'Total':<18}", end="")
    for alpha in alphas:
        v = results[alpha]["daily_costs"].mean() * 10000
        print(f"  {f'{v:.3f}':>14}", end="")
    print()
    print("  (units: bps/day = 1/100 of %)")

    # Overnight return analysis
    print("\n  --- Overnight Return Analysis (α=1.0, full hold) ---")
    on_ret = results[1.0]["daily_overnight"]
    on_ret_nonzero = on_ret[on_ret != 0]
    print(f"  Mean overnight return:   {on_ret_nonzero.mean()*100:.4f}%")
    print(f"  Std overnight return:    {on_ret_nonzero.std()*100:.4f}%")
    print(f"  Sharpe of overnight:     {on_ret_nonzero.mean() / on_ret_nonzero.std() * np.sqrt(TRADING_DAYS):.4f}")
    print(f"  Hit rate (overnight>0):  {(on_ret_nonzero > 0).mean()*100:.1f}%")
    print(f"  Days with overnight:     {len(on_ret_nonzero)} / {len(on_ret)}")

    # Gap risk analysis (VaR/ES of overnight returns)
    print("\n  --- Gap Risk Analysis (α=1.0, full hold) ---")
    on_ret_clean = on_ret_nonzero.dropna()
    if len(on_ret_clean) > 0:
        var_99 = np.percentile(on_ret_clean, 1)
        var_95 = np.percentile(on_ret_clean, 5)
        es_99 = on_ret_clean[on_ret_clean <= var_99].mean()
        es_95 = on_ret_clean[on_ret_clean <= var_95].mean()
        max_loss = on_ret_clean.min()
        max_gain = on_ret_clean.max()
        print(f"  VaR(95%):                {var_95*100:.4f}%")
        print(f"  VaR(99%):                {var_99*100:.4f}%")
        print(f"  ES(95%):                 {es_95*100:.4f}%")
        print(f"  ES(99%):                 {es_99*100:.4f}%")
        print(f"  Max overnight loss:      {max_loss*100:.4f}%")
        print(f"  Max overnight gain:      {max_gain*100:.4f}%")
        print(f"  Days with loss > 1%%:     {(on_ret_clean < -0.01).sum()} / {len(on_ret_clean)} ({(on_ret_clean < -0.01).mean()*100:.1f}%)")
        print(f"  Days with loss > 2%%:     {(on_ret_clean < -0.02).sum()} / {len(on_ret_clean)} ({(on_ret_clean < -0.02).mean()*100:.1f}%)")
        print(f"  Days with loss > 3%%:     {(on_ret_clean < -0.03).sum()} / {len(on_ret_clean)} ({(on_ret_clean < -0.03).mean()*100:.1f}%)")

    # Margin requirement analysis
    print("\n  --- Margin Requirement Analysis ---")
    overnight_margin_mult = args.overnight_margin_mult
    base_gross = results[0.0]["daily_gross"].mean()
    for alpha in alphas:
        carried_gross = alpha * base_gross
        intraday_gross = (1.0 - alpha) * base_gross
        overnight_margin = carried_gross * overnight_margin_mult
        total_margin = intraday_gross + overnight_margin
        margin_ratio = total_margin / base_gross if base_gross > 0 else 0
        label = "Current" if alpha == 0 else f"α={alpha}"
        print(f"  {label:<12}: Intraday={intraday_gross:.3f}  Overnight={carried_gross:.3f}×{overnight_margin_mult}={overnight_margin:.3f}  Total margin={total_margin:.3f}  (ratio={margin_ratio:.3f}x vs current)")

    # Cost saving analysis
    print("\n  --- Cost Saving Analysis (slippage only vs total) ---")
    base_slip = results[0.0]["daily_slip"].sum()
    base_total = results[0.0]["daily_costs"].sum()
    for alpha in alphas:
        slip = results[alpha]["daily_slip"].sum()
        total = results[alpha]["daily_costs"].sum()
        slip_saving = base_slip - slip
        total_saving = base_total - total
        slip_pct = slip_saving / base_slip * 100 if base_slip > 0 else 0
        total_pct = total_saving / base_total * 100 if base_total > 0 else 0
        print(f"  α={alpha:.2f}: Slip cost={slip*100:.2f}%  Total cost={total*100:.2f}%  Slip saving={slip_saving*100:.2f}% ({slip_pct:.1f}%)  Net saving={total_saving*100:.2f}% ({total_pct:.1f}%)")

    # Weight stability analysis
    print("\n  --- Weight Stability (Turnover Distribution) ---")
    turnover = results[0.0]["daily_turnover"]
    print(f"  Mean turnover:   {turnover.mean():.3f}")
    print(f"  Median turnover: {turnover.median():.3f}")
    print(f"  10th pct:        {turnover.quantile(0.1):.3f}")
    print(f"  90th pct:        {turnover.quantile(0.9):.3f}")
    print(f"  Days with turnover < 0.5: {(turnover < 0.5).sum()} / {len(turnover)} ({(turnover < 0.5).mean()*100:.1f}%)")
    print(f"  Days with turnover < 1.0: {(turnover < 1.0).sum()} / {len(turnover)} ({(turnover < 1.0).mean()*100:.1f}%)")

    print("=" * 120)

    # Save results
    output_dir = Path("results/overnight_holding_backtest")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Daily comparison
    comparison_df = pd.DataFrame({
        "baseline_return": results[0.0]["daily_returns"],
        "baseline_cost": results[0.0]["daily_costs"],
        "baseline_turnover": results[0.0]["daily_turnover"],
    })
    for alpha in alphas[1:]:
        comparison_df[f"alpha_{alpha}_return"] = results[alpha]["daily_returns"]
        comparison_df[f"alpha_{alpha}_cost"] = results[alpha]["daily_costs"]
        comparison_df[f"alpha_{alpha}_overnight"] = results[alpha]["daily_overnight"]
        comparison_df[f"alpha_{alpha}_turnover"] = results[alpha]["daily_turnover"]
    comparison_df.to_csv(output_dir / "daily_comparison.csv")

    # Metrics summary
    metrics_rows = []
    for alpha in alphas:
        m = all_metrics[alpha].copy()
        m["alpha"] = alpha
        m["total_cost"] = results[alpha]["daily_costs"].sum()
        m["total_slip"] = results[alpha]["daily_slip"].sum()
        m["total_financing"] = results[alpha]["daily_financing"].sum()
        m["total_borrow"] = results[alpha]["daily_borrow"].sum()
        m["total_reverse"] = results[alpha]["daily_reverse"].sum()
        m["avg_turnover"] = results[alpha]["daily_turnover"].mean()
        m["avg_overnight"] = results[alpha]["daily_overnight"].mean()
        m["std_overnight"] = results[alpha]["daily_overnight"].std()
        metrics_rows.append(m)
    pd.DataFrame(metrics_rows).to_csv(output_dir / "metrics_summary.csv", index=False)

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(4, 1, figsize=(12, 13), sharex=True)

        # Equity curves
        for alpha in alphas:
            label = "Current (α=0)" if alpha == 0 else f"α={alpha}"
            wealth = results[alpha]["equity_curve"]
            axes[0].plot(wealth.index, wealth.values, label=label, alpha=0.8)
        axes[0].set_title("Equity Curve — Overnight Holding Comparison (Realistic Costs)")
        axes[0].set_ylabel("Wealth")
        axes[0].legend()
        axes[0].grid(True)
        axes[0].set_yscale("log")

        # Cost comparison (total)
        for alpha in alphas:
            label = f"α={alpha}" if alpha > 0 else "Current"
            axes[1].plot(results[alpha]["daily_costs"].index,
                        results[alpha]["daily_costs"].rolling(60).mean().values * 10000,
                        label=label, alpha=0.7)
        axes[1].set_title("Rolling 60-day Average Total Cost (bps/day)")
        axes[1].set_ylabel("Cost (bps/day)")
        axes[1].legend()
        axes[1].grid(True)

        # Cost breakdown (α=1.0)
        for comp_key, comp_label, color in [("daily_slip", "Slippage", "blue"), ("daily_financing", "Financing", "green"), ("daily_borrow", "Borrow", "orange"), ("daily_reverse", "Reverse", "red")]:
            axes[2].plot(results[1.0][comp_key].index,
                        results[1.0][comp_key].rolling(60).mean().values * 10000,
                        label=comp_label, alpha=0.7, color=color)
        axes[2].set_title("Cost Breakdown (α=1.0, 60-day rolling mean, bps/day)")
        axes[2].set_ylabel("Cost (bps/day)")
        axes[2].legend()
        axes[2].grid(True)

        # Overnight return contribution (alpha=1.0)
        on_ret = results[1.0]["daily_overnight"]
        axes[3].plot(on_ret.index, on_ret.rolling(60).mean().values * 100, color="red", alpha=0.7, label="Overnight (60d MA)")
        axes[3].axhline(y=0, color="black", linestyle="--", alpha=0.3)
        axes[3].set_title("Overnight Return Contribution (α=1.0, 60-day rolling mean)")
        axes[3].set_ylabel("Return (%)")
        axes[3].legend()
        axes[3].grid(True)

        plt.tight_layout()
        plt.savefig(output_dir / "overnight_holding_comparison.png", dpi=150)
        plt.close()
        logger.info("Chart saved: %s", output_dir / "overnight_holding_comparison.png")
    except Exception as e:
        logger.warning("Chart generation failed: %s", e)

    logger.info("Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
