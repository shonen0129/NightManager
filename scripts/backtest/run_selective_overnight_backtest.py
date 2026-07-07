"""選択的オーバーナイト持ち越しバックテスト

一律αではなく、個別銘柄の条件に基づいて持ち越し判定を行う戦略を検証する。

戦略バリエーション:
  1. uniform_alpha=0.5      — 一律50%持ち越し（ベースライン）
  2. hold_winners           — 利益が出ている銘柄のみ持ち越し
  3. hold_losers            — 損失が出ている銘柄のみ持ち越し
  4. hold_strong_signals    — シグナル強度が大きい銘柄のみ持ち越し
  5. hold_weak_signals      — シグナル強度が小さい銘柄のみ持ち越し
  6. hold_long_only         — ロングポジションのみ持ち越し
  7. hold_short_only        — ショートポジションのみ持ち越し
  8. hold_winners_and_strong — 利益＆強シグナルの両方を満たす銘柄のみ
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from experiments.backtest_common import (
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

SLIP_BPS = 5.0
FIN_ANNUAL = 0.025
BORROW_ANNUAL = 0.0115
REVERSE_BPS = 2.0


def make_alpha_mask_fn(strategy: str, alpha_base: float = 0.5):
    """Build a per-asset alpha mask function for use with simulate_overnight_holding."""

    def fn(i, date, w_t, r_target, signals_row):
        n = len(w_t)
        mask = np.zeros(n)

        if strategy == "uniform_alpha":
            mask[:] = alpha_base

        elif strategy == "hold_winners":
            asset_returns = w_t * r_target
            mask[asset_returns > 0] = alpha_base

        elif strategy == "hold_losers":
            asset_returns = w_t * r_target
            mask[asset_returns < 0] = alpha_base

        elif strategy == "hold_strong_signals":
            if signals_row is not None:
                abs_sig = np.abs(signals_row)
                median_sig = np.median(abs_sig[abs_sig > 0]) if np.any(abs_sig > 0) else 0
                mask[abs_sig >= median_sig] = alpha_base

        elif strategy == "hold_weak_signals":
            if signals_row is not None:
                abs_sig = np.abs(signals_row)
                median_sig = np.median(abs_sig[abs_sig > 0]) if np.any(abs_sig > 0) else 0
                mask[abs_sig < median_sig] = alpha_base

        elif strategy == "hold_long_only":
            mask[w_t > 0] = alpha_base

        elif strategy.startswith("hold_long_alpha_"):
            custom_alpha = float(strategy.split("_")[-1])
            mask[w_t > 0] = custom_alpha

        elif strategy.startswith("hold_short_alpha_"):
            custom_alpha = float(strategy.split("_")[-1])
            mask[w_t < 0] = custom_alpha

        elif strategy == "long1_short0.5":
            mask[w_t > 0] = 1.0
            mask[w_t < 0] = 0.5

        elif strategy == "long1_short0.25":
            mask[w_t > 0] = 1.0
            mask[w_t < 0] = 0.25

        elif strategy == "long0.5_short0.25":
            mask[w_t > 0] = 0.5
            mask[w_t < 0] = 0.25

        elif strategy == "long0.75_short0.25":
            mask[w_t > 0] = 0.75
            mask[w_t < 0] = 0.25

        elif strategy == "long0.75_short0.5":
            mask[w_t > 0] = 0.75
            mask[w_t < 0] = 0.5

        elif strategy == "hold_short_only":
            mask[w_t < 0] = alpha_base

        elif strategy == "hold_long_winner":
            asset_returns = w_t * r_target
            mask[(w_t > 0) & (asset_returns > 0)] = alpha_base

        elif strategy == "hold_long_loser":
            asset_returns = w_t * r_target
            mask[(w_t > 0) & (asset_returns < 0)] = alpha_base

        elif strategy == "hold_short_winner":
            asset_returns = w_t * r_target
            mask[(w_t < 0) & (asset_returns > 0)] = alpha_base

        elif strategy == "hold_short_loser":
            asset_returns = w_t * r_target
            mask[(w_t < 0) & (asset_returns < 0)] = alpha_base

        elif strategy == "hold_winners_and_strong":
            asset_returns = w_t * r_target
            if signals_row is not None:
                abs_sig = np.abs(signals_row)
                median_sig = np.median(abs_sig[abs_sig > 0]) if np.any(abs_sig > 0) else 0
                mask[(asset_returns > 0) & (abs_sig >= median_sig)] = alpha_base

        elif strategy == "hold_losers_and_strong":
            asset_returns = w_t * r_target
            if signals_row is not None:
                abs_sig = np.abs(signals_row)
                median_sig = np.median(abs_sig[abs_sig > 0]) if np.any(abs_sig > 0) else 0
                mask[(asset_returns < 0) & (abs_sig >= median_sig)] = alpha_base

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return mask

    return fn


def main():
    logger.info("Loading data and running baseline backtest...")
    df_exec = load_execution_data(beta_window=60)
    baseline = run_baseline_backtest(
        df_exec, start_date="2015-01-05", slippage_bps=SLIP_BPS,
        overnight_alpha=0.0, buy_interest_annual=0.0, borrow_fee_annual=0.0, reverse_fee_bps=0.0,
    )
    weights = baseline["weights"]
    signals = baseline["signals"]

    sim_dates = weights.index

    target_df, gap_df = prepare_target_and_gap_returns(df_exec, sim_dates)

    costs = CostParams(
        slippage_bps=SLIP_BPS,
        buy_interest_annual=FIN_ANNUAL,
        borrow_fee_annual=BORROW_ANNUAL,
        reverse_fee_bps=REVERSE_BPS,
    )

    # alpha=0 with costs (via simulate_overnight_holding)
    alpha0_res = simulate_overnight_holding(weights, target_df, gap_df, alpha=0.0, costs=costs)
    # uniform alpha=0.5 with costs
    uniform_05_res = simulate_overnight_holding(weights, target_df, gap_df, alpha=0.5, costs=costs)

    strategies = [
        ("alpha=0 (no hold)", None),
        ("uniform α=0.5", "uniform_alpha"),
        ("hold_winners", "hold_winners"),
        ("hold_losers", "hold_losers"),
        ("hold_strong_signals", "hold_strong_signals"),
        ("hold_weak_signals", "hold_weak_signals"),
        ("hold_long_only", "hold_long_only"),
        ("hold_short_only", "hold_short_only"),
        ("long α=0.125", "hold_long_alpha_0.125"),
        ("long α=0.25", "hold_long_alpha_0.25"),
        ("long α=0.375", "hold_long_alpha_0.375"),
        ("long α=0.50", "hold_long_alpha_0.50"),
        ("long α=0.75", "hold_long_alpha_0.75"),
        ("long α=1.0", "hold_long_alpha_1.0"),
        ("L1.0+S0.5", "long1_short0.5"),
        ("L1.0+S0.25", "long1_short0.25"),
        ("L0.75+S0.25", "long0.75_short0.25"),
        ("L0.75+S0.5", "long0.75_short0.5"),
        ("L0.5+S0.25", "long0.5_short0.25"),
        ("hold_long_winner", "hold_long_winner"),
        ("hold_long_loser", "hold_long_loser"),
        ("hold_short_winner", "hold_short_winner"),
        ("hold_short_loser", "hold_short_loser"),
        ("hold_winners_and_strong", "hold_winners_and_strong"),
        ("hold_losers_and_strong", "hold_losers_and_strong"),
    ]

    results = {}
    overnight_stats = {}
    hold_stats = {}

    for label, strategy in strategies:
        if strategy is None:
            rets = alpha0_res["daily_returns"]
            on_rets = pd.Series(0.0, index=sim_dates)
            hc = pd.Series(0, index=sim_dates)
        elif strategy == "uniform_alpha":
            rets = uniform_05_res["daily_returns"]
            on_rets = uniform_05_res["daily_overnight"]
            hc = uniform_05_res["daily_hold_counts"]
        else:
            fn = make_alpha_mask_fn(strategy, alpha_base=0.5)
            res = simulate_overnight_holding(
                weights, target_df, gap_df, alpha=fn, costs=costs, signals_df=signals
            )
            rets = res["daily_returns"]
            on_rets = res["daily_overnight"]
            hc = res["daily_hold_counts"]

        results[label] = rets
        overnight_stats[label] = on_rets
        hold_stats[label] = hc

    # Print results
    print()
    print("=" * 110)
    print("  Selective Overnight Holding Backtest — Performance Comparison")
    print(f"  Period: {sim_dates[0].date()} → {sim_dates[-1].date()}  ({len(sim_dates)} days)")
    print(f"  Costs: slippage={SLIP_BPS}bps, financing={FIN_ANNUAL*100:.1f}%, borrow={BORROW_ANNUAL*100:.2f}%, reverse={REVERSE_BPS}bps/day")
    print("=" * 110)
    print()

    # Full period metrics
    print(f"  {'Strategy':<28} {'Sharpe':>8} {'AR':>10} {'RISK':>8} {'MDD':>8} {'Calmar':>8} {'HitRate':>8} {'ON Mean':>10} {'Avg Hold':>9}")
    print("  " + "-" * 105)

    base_sharpe = extended_metrics(results["alpha=0 (no hold)"])["Sharpe"]

    for label, _ in strategies:
        m = extended_metrics(results[label])
        on_mean = overnight_stats[label].mean() * 100
        avg_hold = hold_stats[label].mean()
        delta_s = m["Sharpe"] - base_sharpe
        print(
            f"  {label:<28} {m['Sharpe']:>8.4f} {m['AR']*100:>9.2f}% {m['RISK']*100:>7.2f}% "
            f"{m['MDD']*100:>7.2f}% {m['Calmar']:>8.2f} {m['Hit Rate']:>7.1f}% "
            f"{on_mean:>+9.4f}% {avg_hold:>8.1f}"
        )

    print()
    print(f"  {'Strategy':<28} {'Δ Sharpe':>10} {'Δ Sharpe %':>12} {'Δ AR':>10} {'Δ MDD':>10}")
    print("  " + "-" * 75)
    base_m = extended_metrics(results["alpha=0 (no hold)"])
    for label, _ in strategies:
        m = extended_metrics(results[label])
        ds = m["Sharpe"] - base_m["Sharpe"]
        dsp = (ds / base_m["Sharpe"]) * 100 if base_m["Sharpe"] != 0 else 0
        dar = (m["AR"] - base_m["AR"]) * 100
        dmdd = (m["MDD"] - base_m["MDD"]) * 100
        print(f"  {label:<28} {ds:>+10.4f} {dsp:>+11.1f}% {dar:>+9.2f}% {dmdd:>+9.2f}%")

    # Yearly Sharpe for top strategies
    print()
    print("  --- Yearly Sharpe Comparison ---")
    top_labels = ["alpha=0 (no hold)", "uniform α=0.5", "L1.0+S0.5", "L1.0+S0.25",
                   "L0.75+S0.25", "L0.75+S0.5", "L0.5+S0.25", "long α=0.50"]
    print(f"  {'Year':<6}", end="")
    for label in top_labels:
        short = label.replace("alpha=0 (no hold)", "α=0").replace("uniform α=0.5", "uniform")
        print(f"  {short:>16}", end="")
    print()
    print("  " + "-" * (6 + 18 * len(top_labels)))

    for year in sorted(sim_dates.year.unique()):
        yr_mask = sim_dates.year == year
        if yr_mask.sum() < 20:
            continue
        print(f"  {year:<6}", end="")
        for label in top_labels:
            s = calculate_metrics(results[label][yr_mask])["Sharpe"]
            print(f"  {s:>16.4f}", end="")
        print()

    # Overnight return statistics by strategy
    print()
    print("  --- Overnight Return Statistics (selected positions only) ---")
    print(f"  {'Strategy':<28} {'ON Mean':>10} {'ON Std':>10} {'ON Sharpe':>10} {'HitRate':>8} {'Max Loss':>10} {'Max Gain':>10}")
    print("  " + "-" * 90)
    for label, _ in strategies:
        on = overnight_stats[label]
        if on.std() > 0:
            on_mean = on.mean() * 100
            on_std = on.std() * 100
            on_sharpe = on.mean() / on.std() * np.sqrt(TRADING_DAYS)
            on_hit = (on > 0).mean() * 100
            on_min = on.min() * 100
            on_max = on.max() * 100
        else:
            on_mean = on_std = on_sharpe = on_hit = on_min = on_max = 0
        print(
            f"  {label:<28} {on_mean:>+9.4f}% {on_std:>9.4f}% {on_sharpe:>10.4f} "
            f"{on_hit:>7.1f}% {on_min:>+9.4f}% {on_max:>+9.4f}%"
        )

    # Save CSV
    output_dir = Path("results/overnight_holding_backtest")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_data = []
    for label, _ in strategies:
        m = extended_metrics(results[label])
        summary_data.append({
            "strategy": label,
            "sharpe": m["Sharpe"],
            "ar": m["AR"],
            "risk": m["RISK"],
            "mdd": m["MDD"],
            "calmar": m["Calmar"],
            "hit_rate": m["Hit Rate"],
            "overnight_mean": overnight_stats[label].mean(),
            "overnight_std": overnight_stats[label].std(),
            "avg_hold_count": hold_stats[label].mean(),
        })
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / "selective_overnight_summary.csv", index=False)
    logger.info(f"Summary saved to {output_dir / 'selective_overnight_summary.csv'}")

    print()
    print("=" * 110)


if __name__ == "__main__":
    main()
