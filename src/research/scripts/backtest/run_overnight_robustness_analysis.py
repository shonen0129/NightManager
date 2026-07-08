#!/usr/bin/env python
"""Overnight holding robustness analysis: year-by-year and regime-by-regime stability.

Validates that the positive overnight return bias and Sharpe improvement
are not period-specific artifacts.
"""

from __future__ import annotations

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
    load_execution_data,
    prepare_target_and_gap_returns,
    run_baseline_backtest,
    simulate_overnight_holding,
    yearly_metrics,
)
from leadlag.reporting.metrics import calculate_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("[1/3] Loading data and running baseline backtest...")
    df_exec = load_execution_data(beta_window=60)
    baseline = run_baseline_backtest(df_exec, start_date="2015-01-05", slippage_bps=5.0)

    weights = baseline["weights"]
    sim_dates = weights.index

    target_returns_df, gap_returns_df = prepare_target_and_gap_returns(df_exec, sim_dates)

    costs = CostParams()

    res0 = simulate_overnight_holding(weights, target_returns_df, gap_returns_df, alpha=0.0, costs=costs)
    res05 = simulate_overnight_holding(weights, target_returns_df, gap_returns_df, alpha=0.5, costs=costs)
    res1 = simulate_overnight_holding(weights, target_returns_df, gap_returns_df, alpha=1.0, costs=costs)

    alpha0_ret = res0["daily_returns"]
    alpha05_ret = res05["daily_returns"]
    alpha1_ret = res1["daily_returns"]
    intraday_daily = res1["daily_intraday"]
    overnight_daily = res1["daily_overnight"]

    logger.info("[2/3] Computing year-by-year statistics...")

    # Yearly analysis
    print("\n" + "=" * 110)
    print("  Overnight Holding Robustness Analysis — Year-by-Year")
    print("=" * 110)

    # --- Yearly: Overnight return component ---
    print("\n  --- Overnight Return Component (α=1.0) by Year ---")
    print(f"  {'Year':<6} {'Days':>6} {'Mean':>10} {'Std':>10} {'Sharpe':>10} {'HitRate':>10} {'Min':>10} {'Max':>10}")
    print("  " + "-" * 82)
    for year in sorted(overnight_daily.index.year.unique()):
        yr = overnight_daily[overnight_daily.index.year == year]
        if len(yr) < 20:
            continue
        mean = yr.mean() * 100
        std = yr.std() * 100
        sharpe = yr.mean() / yr.std() * np.sqrt(TRADING_DAYS) if yr.std() > 0 else 0
        hit = (yr > 0).mean() * 100
        print(f"  {year:<6} {len(yr):>6} {f'{mean:.4f}%':>10} {f'{std:.4f}%':>10} {sharpe:>10.4f} {f'{hit:.1f}%':>10} {f'{yr.min()*100:.4f}%':>10} {f'{yr.max()*100:.4f}%':>10}")

    # --- Yearly: Strategy Sharpe comparison ---
    print("\n  --- Strategy Sharpe by Year (Realistic Costs) ---")
    print(f"  {'Year':<6} {'α=0 Sharpe':>12} {'α=0.5 Sharpe':>14} {'α=1.0 Sharpe':>14} {'Δ(0.5)':>10} {'Δ(1.0)':>10} {'α=0 AR':>10} {'α=1.0 AR':>10}")
    print("  " + "-" * 98)
    for year in sorted(alpha0_ret.index.year.unique()):
        r0 = alpha0_ret[alpha0_ret.index.year == year]
        r05 = alpha05_ret[alpha05_ret.index.year == year]
        r1 = alpha1_ret[alpha1_ret.index.year == year]
        if len(r0) < 20:
            continue
        s0 = calculate_metrics(r0)
        s05 = calculate_metrics(r05)
        s1 = calculate_metrics(r1)
        d05 = s05.get("Sharpe", 0) - s0.get("Sharpe", 0)
        d1 = s1.get("Sharpe", 0) - s0.get("Sharpe", 0)
        ar0 = s0.get("AR", 0) * 100
        ar1 = s1.get("AR", 0) * 100
        print(f"  {year:<6} {s0.get('Sharpe',0):>12.4f} {s05.get('Sharpe',0):>14.4f} {s1.get('Sharpe',0):>14.4f} {d05:>+10.4f} {d1:>+10.4f} {ar0:>9.2f}% {ar1:>9.2f}%")

    # --- Yearly: MDD comparison ---
    print("\n  --- MDD by Year (Realistic Costs) ---")
    print(f"  {'Year':<6} {'α=0 MDD':>12} {'α=0.5 MDD':>12} {'α=1.0 MDD':>12} {'Δ(0.5)':>10} {'Δ(1.0)':>10}")
    print("  " + "-" * 66)
    for year in sorted(alpha0_ret.index.year.unique()):
        r0 = alpha0_ret[alpha0_ret.index.year == year]
        r05 = alpha05_ret[alpha05_ret.index.year == year]
        r1 = alpha1_ret[alpha1_ret.index.year == year]
        if len(r0) < 20:
            continue
        s0 = calculate_metrics(r0)
        s05 = calculate_metrics(r05)
        s1 = calculate_metrics(r1)
        m0 = s0.get("MDD", 0) * 100
        m05 = s05.get("MDD", 0) * 100
        m1 = s1.get("MDD", 0) * 100
        d05 = m05 - m0
        d1 = m1 - m0
        print(f"  {year:<6} {f'{m0:.2f}%':>12} {f'{m05:.2f}%':>12} {f'{m1:.2f}%':>12} {f'{d05:+.2f}%':>10} {f'{d1:+.2f}%':>10}")

    logger.info("[3/3] Computing regime-based statistics...")

    # Regime analysis: bull/bear/sideways based on TOPIX overnight returns
    topix_gap_col = "jp_gap_1306.T"
    if topix_gap_col in df_exec.columns:
        topix_gap = df_exec[topix_gap_col].loc[sim_dates]
    else:
        # Use average of all JP gaps as proxy
        topix_gap = gap_returns_df.mean(axis=1)

    # Define regimes: bull = rolling 60d mean > 0.03%, bear = < -0.03%, sideways = in between
    rolling_market = topix_gap.rolling(60, min_periods=20).mean()
    regime = pd.Series("sideways", index=sim_dates)
    regime[rolling_market > 0.0003] = "bull"
    regime[rolling_market < -0.0003] = "bear"

    print("\n  --- Regime Analysis (Market regime based on 60d rolling TOPIX gap) ---")
    print(f"  {'Regime':<10} {'Days':>6} {'α=0 Sharpe':>12} {'α=0.5 Sharpe':>14} {'α=1.0 Sharpe':>14} {'Δ(1.0)':>10} {'ON Mean':>10} {'ON Sharpe':>10}")
    print("  " + "-" * 86)
    for reg in ["bull", "sideways", "bear"]:
        mask = regime == reg
        if mask.sum() < 20:
            continue
        r0 = alpha0_ret[mask]
        r05 = alpha05_ret[mask]
        r1 = alpha1_ret[mask]
        on = overnight_daily[mask]
        s0 = calculate_metrics(r0)
        s05 = calculate_metrics(r05)
        s1 = calculate_metrics(r1)
        d1 = s1.get("Sharpe", 0) - s0.get("Sharpe", 0)
        on_mean = on.mean() * 100
        on_sharpe = on.mean() / on.std() * np.sqrt(TRADING_DAYS) if on.std() > 0 else 0
        print(f"  {reg:<10} {mask.sum():>6} {s0.get('Sharpe',0):>12.4f} {s05.get('Sharpe',0):>14.4f} {s1.get('Sharpe',0):>14.4f} {f'{d1:+.4f}':>10} {f'{on_mean:.4f}%':>10} {on_sharpe:>10.4f}")

    # Volatility regime
    rolling_vol = alpha0_ret.rolling(60, min_periods=20).std()
    vol_median = rolling_vol.median()
    vol_regime = pd.Series("normal", index=sim_dates)
    vol_regime[rolling_vol > vol_median * 1.5] = "high_vol"
    vol_regime[rolling_vol < vol_median * 0.5] = "low_vol"

    print("\n  --- Volatility Regime Analysis ---")
    print(f"  {'Regime':<10} {'Days':>6} {'α=0 Sharpe':>12} {'α=0.5 Sharpe':>14} {'α=1.0 Sharpe':>14} {'Δ(1.0)':>10} {'ON Mean':>10} {'ON Sharpe':>10}")
    print("  " + "-" * 86)
    for reg in ["low_vol", "normal", "high_vol"]:
        mask = vol_regime == reg
        if mask.sum() < 20:
            continue
        r0 = alpha0_ret[mask]
        r05 = alpha05_ret[mask]
        r1 = alpha1_ret[mask]
        on = overnight_daily[mask]
        s0 = calculate_metrics(r0)
        s05 = calculate_metrics(r05)
        s1 = calculate_metrics(r1)
        d1 = s1.get("Sharpe", 0) - s0.get("Sharpe", 0)
        on_mean = on.mean() * 100
        on_sharpe = on.mean() / on.std() * np.sqrt(TRADING_DAYS) if on.std() > 0 else 0
        print(f"  {reg:<10} {mask.sum():>6} {s0.get('Sharpe',0):>12.4f} {s05.get('Sharpe',0):>14.4f} {s1.get('Sharpe',0):>14.4f} {f'{d1:+.4f}':>10} {f'{on_mean:.4f}%':>10} {on_sharpe:>10.4f}")

    # Monthly overnight return heatmap data
    print("\n  --- Monthly Average Overnight Return (α=1.0, %) ---")
    monthly_on = overnight_daily.groupby([overnight_daily.index.year, overnight_daily.index.month]).mean() * 100
    monthly_on_df = monthly_on.unstack()
    print(f"  {'Year':<6}", end="")
    for mo in range(1, 13):
        print(f"  {f'{mo}月':>8}", end="")
    print()
    for year in sorted(monthly_on_df.index):
        print(f"  {year:<6}", end="")
        for mo in range(1, 13):
            v = monthly_on_df.loc[year, mo] if mo in monthly_on_df.columns else np.nan
            if np.isfinite(v):
                sign = "+" if v >= 0 else ""
                print(f"  {f'{sign}{v:.3f}':>8}", end="")
            else:
                print(f"  {'---':>8}", end="")
        print()

    # Consistency check: rolling 250d Sharpe of overnight
    print("\n  --- Rolling 250-day Overnight Sharpe Consistency ---")
    rolling_on_sharpe = overnight_daily.rolling(250, min_periods=100).apply(
        lambda x: x.mean() / x.std() * np.sqrt(TRADING_DAYS) if x.std() > 0 else 0, raw=True
    )
    positive_pct = (rolling_on_sharpe > 0).mean() * 100
    above_1_pct = (rolling_on_sharpe > 1.0).mean() * 100
    print(f"  Rolling 250d ON Sharpe > 0:   {positive_pct:.1f}% of days")
    print(f"  Rolling 250d ON Sharpe > 1.0: {above_1_pct:.1f}% of days")
    print(f"  Min rolling 250d ON Sharpe:   {rolling_on_sharpe.min():.4f}")
    print(f"  Max rolling 250d ON Sharpe:   {rolling_on_sharpe.max():.4f}")
    print(f"  Median rolling 250d ON Sharpe: {rolling_on_sharpe.median():.4f}")

    # Rolling delta Sharpe (α=1.0 vs α=0)
    def rolling_sharpe(rets, window=250):
        return rets.rolling(window, min_periods=100).apply(
            lambda x: x.mean() / x.std() * np.sqrt(TRADING_DAYS) if x.std() > 0 else 0, raw=True
        )

    rs0 = rolling_sharpe(alpha0_ret)
    rs1 = rolling_sharpe(alpha1_ret)
    delta_rs = rs1 - rs0
    print(f"\n  --- Rolling 250-day Δ Sharpe (α=1.0 vs α=0) ---")
    print(f"  Δ Sharpe > 0:    {(delta_rs > 0).mean() * 100:.1f}% of days")
    print(f"  Δ Sharpe > 0.1:  {(delta_rs > 0.1).mean() * 100:.1f}% of days")
    print(f"  Min Δ Sharpe:    {delta_rs.min():.4f}")
    print(f"  Max Δ Sharpe:    {delta_rs.max():.4f}")
    print(f"  Median Δ Sharpe: {delta_rs.median():.4f}")

    print("=" * 110)

    # Save results
    output_dir = Path("results/overnight_holding_backtest")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Yearly metrics
    yearly_dfs = []
    for label, rets in [("alpha0", alpha0_ret), ("alpha05", alpha05_ret), ("alpha1", alpha1_ret)]:
        df = yearly_metrics(rets, label)
        yearly_dfs.append(df)
    pd.concat(yearly_dfs).to_csv(output_dir / "yearly_metrics.csv", index=False)

    # Daily data
    daily_data = pd.DataFrame({
        "alpha0_return": alpha0_ret,
        "alpha05_return": alpha05_ret,
        "alpha1_return": alpha1_ret,
        "intraday": intraday_daily,
        "overnight": overnight_daily,
        "regime": regime,
        "vol_regime": vol_regime,
    })
    daily_data.to_csv(output_dir / "robustness_daily.csv")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)

        # 1. Rolling 250d overnight Sharpe
        axes[0].plot(rolling_on_sharpe.index, rolling_on_sharpe.values, color="blue", alpha=0.7)
        axes[0].axhline(y=0, color="black", linestyle="--", alpha=0.3)
        axes[0].axhline(y=1.0, color="green", linestyle=":", alpha=0.3, label="Sharpe=1.0")
        axes[0].set_title("Rolling 250-day Overnight Return Sharpe (α=1.0)")
        axes[0].set_ylabel("Sharpe")
        axes[0].legend()
        axes[0].grid(True)

        # 2. Rolling 250d Δ Sharpe (α=1.0 vs α=0)
        axes[1].plot(delta_rs.index, delta_rs.values, color="red", alpha=0.7)
        axes[1].axhline(y=0, color="black", linestyle="--", alpha=0.3)
        axes[1].set_title("Rolling 250-day Δ Sharpe (α=1.0 − α=0, Realistic Costs)")
        axes[1].set_ylabel("Δ Sharpe")
        axes[1].legend()
        axes[1].grid(True)

        # 3. Yearly Sharpe comparison
        years = sorted(alpha0_ret.index.year.unique())
        s0_vals = []
        s05_vals = []
        s1_vals = []
        for year in years:
            r0 = alpha0_ret[alpha0_ret.index.year == year]
            r05 = alpha05_ret[alpha05_ret.index.year == year]
            r1 = alpha1_ret[alpha1_ret.index.year == year]
            if len(r0) < 20:
                continue
            s0_vals.append(calculate_metrics(r0).get("Sharpe", 0))
            s05_vals.append(calculate_metrics(r05).get("Sharpe", 0))
            s1_vals.append(calculate_metrics(r1).get("Sharpe", 0))
        valid_years = [y for y in years if len(alpha0_ret[alpha0_ret.index.year == y]) >= 20]
        x = np.arange(len(valid_years))
        w = 0.25
        axes[2].bar(x - w, s0_vals, w, label="α=0 (Current)", alpha=0.8, color="steelblue")
        axes[2].bar(x, s05_vals, w, label="α=0.5", alpha=0.8, color="orange")
        axes[2].bar(x + w, s1_vals, w, label="α=1.0", alpha=0.8, color="green")
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(valid_years)
        axes[2].set_title("Yearly Sharpe Comparison (Realistic Costs)")
        axes[2].set_ylabel("Sharpe")
        axes[2].legend()
        axes[2].grid(True, axis="y")

        # 4. Yearly overnight mean return
        on_yearly = overnight_daily.groupby(overnight_daily.index.year).mean() * 100
        on_yearly_std = overnight_daily.groupby(overnight_daily.index.year).std() * 100
        valid_yr = on_yearly.index[on_yearly.index.isin(valid_years)]
        axes[3].bar(valid_yr, on_yearly[valid_yr], yerr=on_yearly_std[valid_yr],
                    color="coral", alpha=0.7, capsize=5, label="Overnight mean ± std")
        axes[3].axhline(y=0, color="black", linestyle="--", alpha=0.3)
        axes[3].set_title("Yearly Average Overnight Return (α=1.0)")
        axes[3].set_ylabel("Return (%)")
        axes[3].legend()
        axes[3].grid(True, axis="y")

        plt.tight_layout()
        plt.savefig(output_dir / "overnight_robustness.png", dpi=150)
        plt.close()
        logger.info("Chart saved: %s", output_dir / "overnight_robustness.png")
    except Exception as e:
        logger.warning("Chart generation failed: %s", e)

    logger.info("Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
