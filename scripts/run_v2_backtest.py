#!/usr/bin/env python3
"""Run V2 production backtest and compare with actual live results."""
import sys, logging, json, glob, os, csv
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.cache import load_df_exec_from_local_cache
from leadlag.data.tickers import JP_TICKERS
from leadlag.models.production_v2 import generate_v2_production_portfolio
from leadlag.models.sre import compute_jp_target_returns

CONFIG_PATH = ROOT / "configs/production/production.yaml"
GAP_DIR = ROOT / "live/pipeline_data/gap_adjusted_distribution/20260712_231014"
START_DATE = "2020-01-06"
END_DATE = "2026-07-17"
OUTPUT_DIR = ROOT / "src/results/v2_backtest"

# Cost parameters (matching production.yaml)
SLIPPAGE_BPS = 5.0
OVERNIGHT_ALPHA_LONG = 0.75
OVERNIGHT_ALPHA_SHORT = 0.50
BUY_INTEREST_ANNUAL = 0.025
BORROW_FEE_ANNUAL = 0.0115
REVERSE_FEE_BPS = 2.0


def main():
    logger.info("Loading config from %s", CONFIG_PATH)
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    logger.info("Loading df_exec from local cache...")
    df_exec = load_df_exec_from_local_cache()
    logger.info("df_exec: %s rows, %s -> %s", len(df_exec), df_exec.index[0], df_exec.index[-1])

    # Compute JP target returns (9:10 -> close)
    y_jp_target = compute_jp_target_returns(df_exec, JP_TICKERS)
    gap_cols = [f"jp_gap_{tk}" for tk in JP_TICKERS]
    if all(c in df_exec.columns for c in gap_cols):
        gap_returns = df_exec[gap_cols].values
        gap_returns_df = pd.DataFrame(gap_returns, index=df_exec.index, columns=JP_TICKERS)
    else:
        gap_returns_df = pd.DataFrame(0.0, index=df_exec.index, columns=JP_TICKERS)

    # Sim dates
    sim_dates = df_exec.index[(df_exec.index >= START_DATE) & (df_exec.index <= END_DATE)]
    logger.info("Sim dates: %s -> %s (%d days)", sim_dates[0], sim_dates[-1], len(sim_dates))

    # Cost params
    slip = SLIPPAGE_BPS / 10000.0
    financing_daily = BUY_INTEREST_ANNUAL / 365.0
    borrow_daily = BORROW_FEE_ANNUAL / 365.0
    reverse_daily = REVERSE_FEE_BPS / 10000.0

    n_j = len(JP_TICKERS)
    w_prev = np.zeros(n_j)
    daily_returns = []
    daily_gross = []
    daily_turnover = []
    daily_costs = []
    daily_fallback = []
    daily_weights = []
    sim_dates_actual = []

    n_fallback = 0
    n_success = 0

    for idx, dt in enumerate(sim_dates):
        date_str = dt.strftime("%Y-%m-%d")
        i = df_exec.index.get_indexer([dt])[0]

        try:
            result = generate_v2_production_portfolio(date_str, GAP_DIR, cfg)
        except Exception as e:
            logger.warning("Failed on %s: %s", date_str, e)
            daily_returns.append(0.0)
            daily_gross.append(0.0)
            daily_turnover.append(0.0)
            daily_costs.append(0.0)
            daily_fallback.append(True)
            daily_weights.append(np.zeros(n_j))
            sim_dates_actual.append(dt)
            continue

        w = result["w_final"]
        fb = result["fallback"]["gap_data_missing"]

        if fb:
            n_fallback += 1
            daily_returns.append(0.0)
            daily_gross.append(0.0)
            daily_turnover.append(float(np.sum(np.abs(w - w_prev))))
            daily_costs.append(0.0)
            daily_fallback.append(True)
            daily_weights.append(w.copy())
            sim_dates_actual.append(dt)
            w_prev = w.copy()
            continue

        n_success += 1
        gross = float(np.sum(np.abs(w)))
        net = float(np.sum(w))
        turnover = float(np.sum(np.abs(w - w_prev)))

        # Intraday return (9:10 -> close)
        r_target = y_jp_target[i]
        gross_ret = float(np.dot(w, r_target))

        # Overnight return
        alpha_mask = np.where(w > 0, OVERNIGHT_ALPHA_LONG, np.where(w < 0, OVERNIGHT_ALPHA_SHORT, 0.0))
        overnight_ret = 0.0
        if (OVERNIGHT_ALPHA_LONG > 0 or OVERNIGHT_ALPHA_SHORT > 0) and idx < len(sim_dates) - 1:
            r_gap_next = gap_returns_df.loc[sim_dates[idx + 1]].values
            overnight_ret = float(np.sum(alpha_mask * w * r_gap_next))

        # Cost
        # Calendar days
        if idx < len(sim_dates) - 1:
            days_held = (sim_dates[idx + 1] - sim_dates[idx]).days
        else:
            days_held = 1

        slip_cost = slip * (2.0 * np.sum((1.0 - alpha_mask) * np.abs(w)) + np.sum(alpha_mask * np.abs(w - w_prev) / 2.0))
        held_long = float(np.sum(alpha_mask * np.maximum(w, 0.0)))
        held_short = float(np.sum(alpha_mask * np.maximum(-w, 0.0)))
        fin_cost = held_long * financing_daily * days_held
        borrow_cost = held_short * borrow_daily * days_held
        reverse_cost = held_short * reverse_daily * days_held
        cost = slip_cost + fin_cost + borrow_cost + reverse_cost

        net_ret = gross_ret + overnight_ret - cost

        daily_returns.append(net_ret)
        daily_gross.append(gross)
        daily_turnover.append(turnover)
        daily_costs.append(cost)
        daily_fallback.append(False)
        daily_weights.append(w.copy())
        sim_dates_actual.append(dt)
        w_prev = w.copy()

        if (idx + 1) % 200 == 0:
            logger.info("Processed %d/%d dates (success=%d, fallback=%d)", idx + 1, len(sim_dates), n_success, n_fallback)

    logger.info("Done: %d success, %d fallback out of %d total", n_success, n_fallback, len(sim_dates))

    # Build series
    dates_idx = pd.DatetimeIndex(sim_dates_actual)
    returns = pd.Series(daily_returns, index=dates_idx, name="net_return")
    gross_s = pd.Series(daily_gross, index=dates_idx, name="gross")
    turnover_s = pd.Series(daily_turnover, index=dates_idx, name="turnover")
    cost_s = pd.Series(daily_costs, index=dates_idx, name="cost")
    fb_s = pd.Series(daily_fallback, index=dates_idx, name="fallback")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    returns.to_csv(OUTPUT_DIR / "daily_net_returns.csv", header=["net_return"])
    gross_s.to_csv(OUTPUT_DIR / "daily_gross.csv", header=["gross"])
    turnover_s.to_csv(OUTPUT_DIR / "daily_turnover.csv", header=["turnover"])
    cost_s.to_csv(OUTPUT_DIR / "daily_costs.csv", header=["cost"])
    fb_s.to_csv(OUTPUT_DIR / "daily_fallback.csv", header=["fallback"])

    # Metrics
    valid = returns[~fb_s]
    n_valid = len(valid)
    total_ret = float(np.sum(valid))
    mean_ret = float(np.mean(valid))
    std_ret = float(np.std(valid, ddof=1))
    sharpe = mean_ret / std_ret * np.sqrt(252) if std_ret > 1e-8 else 0.0
    cum = np.cumsum(valid.values)
    running_max = np.maximum.accumulate(cum)
    max_dd = float(np.min(cum - running_max))
    avg_turnover = float(np.mean(turnover_s[~fb_s]))
    avg_gross = float(np.mean(gross_s[~fb_s]))
    fb_rate = n_fallback / len(sim_dates) * 100

    print("\n" + "=" * 60)
    print("=== V2 Backtest Results ===")
    print("=" * 60)
    print(f"  Period:        {sim_dates[0].date()} -> {sim_dates[-1].date()}")
    print(f"  Total days:    {len(sim_dates)}")
    print(f"  Success:       {n_success}")
    print(f"  Fallback:      {n_fallback} ({fb_rate:.1f}%)")
    print(f"  Sharpe:        {sharpe:.4f}")
    print(f"  Total Return:  {total_ret*100:.2f}%")
    print(f"  AR (ann):      {mean_ret*252*100:.2f}%")
    print(f"  Vol (ann):     {std_ret*np.sqrt(252)*100:.2f}%")
    print(f"  Max DD:        {max_dd*100:.2f}%")
    print(f"  Avg Turnover:  {avg_turnover:.4f}")
    print(f"  Avg Gross:     {avg_gross:.4f}")
    print(f"  Avg Cost/day:  {float(np.mean(cost_s[~fb_s]))*10000:.2f} bps")
    print("=" * 60)

    # --- Compare with actual live results ---
    print("\n=== V2 Backtest vs Actual Live (July 2026) ===")

    # Actual equity from wallet snapshots
    actual_equity = {}
    for d in sorted(glob.glob(str(ROOT / "results/202607*_production_close_positions"))):
        f = glob.glob(os.path.join(d, "wallet_close_*.json"))
        if f:
            with open(f[0]) as fh:
                data = json.load(fh)
            ts = data.get("timestamp", "")[:10]
            eq = data.get("ukeire_hosyoukin")
            if eq is not None:
                actual_equity[ts] = eq

    # V2 BT returns for July 2026
    july_dates = [d for d in dates_idx.strftime("%Y-%m-%d") if d.startswith("2026-07")]
    print(f'{"Date":<12} {"V2 BT Ret":>10} {"Actual Ret":>11} {"Diff":>8}')
    print("-" * 45)

    prev_actual = None
    for date_str in july_dates:
        bt_ret = returns.get(pd.Timestamp(date_str), None)
        actual_eq = actual_equity.get(date_str, None)

        if bt_ret is not None and not fb_s.get(pd.Timestamp(date_str), True):
            bt_str = f"{bt_ret*100:>9.2f}%"
        else:
            bt_str = "      FB"

        if actual_eq is not None:
            if prev_actual is None:
                actual_daily = 0.0
                prev_actual = actual_eq
            else:
                actual_daily = (actual_eq - prev_actual) / prev_actual
            actual_str = f"{actual_daily*100:>10.2f}%"
            if bt_ret is not None and not fb_s.get(pd.Timestamp(date_str), True):
                diff = actual_daily - bt_ret
                diff_str = f"{diff*100:>7.2f}%"
            else:
                diff_str = "    N/A"
            print(f"{date_str:<12} {bt_str} {actual_str} {diff_str}")
            prev_actual = actual_eq
        else:
            print(f"{date_str:<12} {bt_str} {'   N/A':>11} {'   N/A':>8}")

    # Also compare V1 BT for same period
    print("\n=== V1 Backtest vs V2 Backtest (July 2026) ===")
    v1_returns = {}
    v1_path = ROOT / "src/results/production_backtest/daily_net_returns.csv"
    if v1_path.exists():
        with open(v1_path) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) >= 2:
                    v1_returns[row[0]] = float(row[1])

    print(f'{"Date":<12} {"V1 BT":>8} {"V2 BT":>8} {"Diff":>8}')
    print("-" * 40)
    for date_str in july_dates:
        v1 = v1_returns.get(date_str, None)
        v2 = returns.get(pd.Timestamp(date_str), None)
        fb = fb_s.get(pd.Timestamp(date_str), True)
        if v1 is not None:
            v1_str = f"{v1*100:>7.2f}%"
        else:
            v1_str = "   N/A"
        if v2 is not None and not fb:
            v2_str = f"{v2*100:>7.2f}%"
            diff = v2 - v1 if v1 is not None else 0
            diff_str = f"{diff*100:>7.2f}%"
        else:
            v2_str = "     FB"
            diff_str = "   N/A"
        print(f"{date_str:<12} {v1_str} {v2_str} {diff_str}")


if __name__ == "__main__":
    main()
