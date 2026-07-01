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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TRADING_DAYS = 245
SLIP_BPS = 5.0
FIN_ANNUAL = 0.025
BORROW_ANNUAL = 0.0115
REVERSE_BPS = 2.0


def calculate_metrics(returns: pd.Series) -> dict:
    if len(returns) == 0 or returns.std() == 0:
        return {"AR": 0, "RISK": 0, "Sharpe": 0, "MDD": 0, "Total Return": 0, "Hit Rate": 0, "Calmar": 0}
    ar = returns.mean() * TRADING_DAYS
    risk = returns.std() * np.sqrt(TRADING_DAYS)
    sharpe = ar / risk if risk > 0 else 0
    wealth = (1 + returns).cumprod()
    mdd = ((wealth / wealth.cummax()) - 1).min()
    total_ret = wealth.iloc[-1] - 1
    hit_rate = (returns > 0).mean()
    calmar = ar / abs(mdd) if mdd < 0 else 0
    return {
        "AR": ar, "RISK": risk, "Sharpe": sharpe, "MDD": mdd,
        "Total Return": total_ret, "Hit Rate": hit_rate, "Calmar": calmar,
    }


def run_selective_backtest(
    weights: pd.DataFrame,
    target_returns: pd.DataFrame,
    gap_returns: pd.DataFrame,
    signals: pd.DataFrame,
    strategy: str,
    alpha_base: float = 0.5,
) -> pd.Series:
    """Run backtest with selective overnight holding.

    Args:
        weights: Daily portfolio weights (dates × tickers)
        target_returns: Intraday 9:10-to-close returns (dates × tickers)
        gap_returns: Overnight gap returns (dates × tickers)
        signals: Signal values at each date (dates × tickers)
        strategy: Selection strategy name
        alpha_base: Base alpha for selected positions

    Returns:
        pd.Series of daily net returns
    """
    dates = list(weights.index)
    tickers = list(weights.columns)
    n_assets = len(tickers)

    slip = SLIP_BPS / 10000.0
    fin_daily = FIN_ANNUAL / 365.0
    borrow_daily = BORROW_ANNUAL / 365.0
    reverse_daily = REVERSE_BPS / 10000.0

    net_returns = []
    overnight_returns = []
    hold_counts = []
    w_prev = np.zeros(n_assets)

    for i, date in enumerate(dates):
        w_t = weights.loc[date].values
        r_target = target_returns.loc[date].values

        # Intraday return (always full position)
        gross_ret = float(np.sum(w_t * r_target))
        gross_exp = float(np.sum(np.abs(w_t)))
        long_exp = float(np.sum(np.maximum(w_t, 0.0)))
        short_exp = float(np.sum(np.maximum(-w_t, 0.0)))

        # Determine per-asset alpha mask
        alpha_mask = np.zeros(n_assets)

        if strategy == "uniform_alpha":
            alpha_mask[:] = alpha_base

        elif strategy == "hold_winners":
            # Hold positions where intraday return is positive
            asset_returns = w_t * r_target
            alpha_mask[asset_returns > 0] = alpha_base

        elif strategy == "hold_losers":
            # Hold positions where intraday return is negative
            asset_returns = w_t * r_target
            alpha_mask[asset_returns < 0] = alpha_base

        elif strategy == "hold_strong_signals":
            # Hold positions with above-median absolute signal strength
            sig_vals = signals.loc[date].values
            abs_sig = np.abs(sig_vals)
            median_sig = np.median(abs_sig[abs_sig > 0]) if np.any(abs_sig > 0) else 0
            alpha_mask[abs_sig >= median_sig] = alpha_base

        elif strategy == "hold_weak_signals":
            # Hold positions with below-median absolute signal strength
            sig_vals = signals.loc[date].values
            abs_sig = np.abs(sig_vals)
            median_sig = np.median(abs_sig[abs_sig > 0]) if np.any(abs_sig > 0) else 0
            alpha_mask[abs_sig < median_sig] = alpha_base

        elif strategy == "hold_long_only":
            # Hold only long positions with custom alpha
            alpha_mask[w_t > 0] = alpha_base

        elif strategy.startswith("hold_long_alpha_"):
            # Hold only long positions with specified alpha (e.g. hold_long_alpha_0.25)
            custom_alpha = float(strategy.split("_")[-1])
            alpha_mask[w_t > 0] = custom_alpha

        elif strategy.startswith("hold_short_alpha_"):
            # Hold only short positions with specified alpha
            custom_alpha = float(strategy.split("_")[-1])
            alpha_mask[w_t < 0] = custom_alpha

        elif strategy == "long1_short0.5":
            # Long fully held, short half held
            alpha_mask[w_t > 0] = 1.0
            alpha_mask[w_t < 0] = 0.5

        elif strategy == "long1_short0.25":
            # Long fully held, short quarter held
            alpha_mask[w_t > 0] = 1.0
            alpha_mask[w_t < 0] = 0.25

        elif strategy == "long0.5_short0.25":
            # Long half held, short quarter held
            alpha_mask[w_t > 0] = 0.5
            alpha_mask[w_t < 0] = 0.25

        elif strategy == "long0.75_short0.25":
            # Long 75% held, short quarter held
            alpha_mask[w_t > 0] = 0.75
            alpha_mask[w_t < 0] = 0.25

        elif strategy == "long0.75_short0.5":
            # Long 75% held, short half held
            alpha_mask[w_t > 0] = 0.75
            alpha_mask[w_t < 0] = 0.5

        elif strategy == "hold_short_only":
            # Hold only short positions
            alpha_mask[w_t < 0] = alpha_base

        elif strategy == "hold_long_winner":
            # Hold long positions that are profitable intraday
            asset_returns = w_t * r_target
            alpha_mask[(w_t > 0) & (asset_returns > 0)] = alpha_base

        elif strategy == "hold_long_loser":
            # Hold long positions that are losing intraday
            asset_returns = w_t * r_target
            alpha_mask[(w_t > 0) & (asset_returns < 0)] = alpha_base

        elif strategy == "hold_short_winner":
            # Hold short positions that are profitable intraday
            asset_returns = w_t * r_target
            alpha_mask[(w_t < 0) & (asset_returns > 0)] = alpha_base

        elif strategy == "hold_short_loser":
            # Hold short positions that are losing intraday
            asset_returns = w_t * r_target
            alpha_mask[(w_t < 0) & (asset_returns < 0)] = alpha_base

        elif strategy == "hold_winners_and_strong":
            # Hold positions that are both winners AND strong signal
            asset_returns = w_t * r_target
            sig_vals = signals.loc[date].values
            abs_sig = np.abs(sig_vals)
            median_sig = np.median(abs_sig[abs_sig > 0]) if np.any(abs_sig > 0) else 0
            alpha_mask[(asset_returns > 0) & (abs_sig >= median_sig)] = alpha_base

        elif strategy == "hold_losers_and_strong":
            # Hold positions that are losers AND strong signal (mean reversion bet)
            asset_returns = w_t * r_target
            sig_vals = signals.loc[date].values
            abs_sig = np.abs(sig_vals)
            median_sig = np.median(abs_sig[abs_sig > 0]) if np.any(abs_sig > 0) else 0
            alpha_mask[(asset_returns < 0) & (abs_sig >= median_sig)] = alpha_base

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Overnight return: sum over assets of alpha_mask[j] * w_t[j] * gap(t+1)[j]
        overnight_ret = 0.0
        if i < len(dates) - 1:
            next_date = dates[i + 1]
            if next_date in gap_returns.index:
                r_gap_next = gap_returns.loc[next_date].values
                overnight_ret = float(np.sum(alpha_mask * w_t * r_gap_next))

        # Cost model
        # For each asset j:
        #   (1-alpha_mask[j]) fraction: full round-trip
        #   alpha_mask[j] fraction: only rebalance cost
        turnover = float(np.sum(np.abs(w_t - w_prev)) / 2.0)
        per_asset_roundtrip = np.sum((1.0 - alpha_mask) * np.abs(w_t))
        per_asset_rebalance = np.sum(alpha_mask * np.abs(w_t - w_prev) / 2.0)
        slip_cost = slip * (2.0 * per_asset_roundtrip + per_asset_rebalance)

        # Financing/borrow/reverse only on held portion
        held_long = np.sum(alpha_mask * np.maximum(w_t, 0.0))
        held_short = np.sum(alpha_mask * np.maximum(-w_t, 0.0))
        fin_cost = held_long * fin_daily
        borrow_cost = held_short * borrow_daily
        reverse_cost = held_short * reverse_daily

        cost = slip_cost + fin_cost + borrow_cost + reverse_cost
        net_ret = gross_ret + overnight_ret - cost

        net_returns.append(net_ret)
        overnight_returns.append(overnight_ret)
        hold_counts.append(int(np.sum(alpha_mask > 0)))

        w_prev = w_t

    rets = pd.Series(net_returns, index=weights.index)
    return rets, pd.Series(overnight_returns, index=weights.index), pd.Series(hold_counts, index=weights.index)


def main():
    logger.info("Loading data and running baseline backtest...")
    from leadlag.data.fetcher import download_data
    from leadlag.data.preprocessor import preprocess_data
    from leadlag.data.tickers import JP_TICKERS
    from leadlag.execution.config import StrategyConfig as ProductionConfig
    from leadlag.execution.helpers import build_strategy
    from leadlag.execution.backtester import BacktestEngine
    from leadlag.models.sre import compute_jp_target_returns

    data = download_data(beta_window=60)
    df_exec = preprocess_data(data, beta_window=60)

    config = ProductionConfig(start_date="2015-01-05", slippage_bps=5.0)
    model = build_strategy(config, df_exec)

    # Use alpha=0 baseline to get weights and signals
    baseline = BacktestEngine.run_backtest(
        model, df_exec=df_exec, start_date="2015-01-05", slippage_bps=5.0,
        overnight_alpha=0.0, buy_interest_annual=0.0, borrow_fee_annual=0.0, reverse_fee_bps=0.0,
    )
    weights = baseline["weights"]
    signals = baseline["signals"]

    sim_dates = weights.index

    # Target returns (intraday)
    y_jp_target = compute_jp_target_returns(df_exec, JP_TICKERS)
    target_df = pd.DataFrame(y_jp_target, index=df_exec.index, columns=JP_TICKERS).loc[sim_dates]

    # Gap returns
    gap_cols = [f"jp_gap_{tk}" for tk in JP_TICKERS]
    gap_df = df_exec[gap_cols].copy()
    gap_df.columns = JP_TICKERS
    gap_df = gap_df.loc[sim_dates]

    # Also get uniform alpha=0.5 baseline with costs
    uniform_05 = BacktestEngine.run_backtest(
        model, df_exec=df_exec, start_date="2015-01-05", slippage_bps=5.0,
        overnight_alpha=0.5, buy_interest_annual=FIN_ANNUAL, borrow_fee_annual=BORROW_ANNUAL, reverse_fee_bps=REVERSE_BPS,
    )

    # Also get alpha=0 baseline with costs
    alpha0 = BacktestEngine.run_backtest(
        model, df_exec=df_exec, start_date="2015-01-05", slippage_bps=5.0,
        overnight_alpha=0.0, buy_interest_annual=FIN_ANNUAL, borrow_fee_annual=BORROW_ANNUAL, reverse_fee_bps=REVERSE_BPS,
    )

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
            # alpha=0 with costs
            rets = alpha0["daily_returns"]
            on_rets = pd.Series(0.0, index=sim_dates)
            hc = pd.Series(0, index=sim_dates)
        elif strategy == "uniform_alpha":
            rets = uniform_05["daily_returns"]
            on_rets = uniform_05.get("daily_overnight_returns", pd.Series(0.0, index=sim_dates))
            hc = pd.Series(len(JP_TICKERS) // 2, index=sim_dates)  # approx
        else:
            rets, on_rets, hc = run_selective_backtest(
                weights, target_df, gap_df, signals, strategy, alpha_base=0.5
            )

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

    base_sharpe = calculate_metrics(results["alpha=0 (no hold)"])["Sharpe"]

    for label, _ in strategies:
        m = calculate_metrics(results[label])
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
    base_m = calculate_metrics(results["alpha=0 (no hold)"])
    for label, _ in strategies:
        m = calculate_metrics(results[label])
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
        m = calculate_metrics(results[label])
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
