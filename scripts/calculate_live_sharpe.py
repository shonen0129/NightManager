#!/usr/bin/env python3
"""Compute live Sharpe ratio from production close wallet snapshots.

Uses ukeire_hosyoukin from wallet_close_*.json as the live equity proxy.
Note that wallet snapshots are taken only on days the close script runs and may
span weekends/holidays; daily_return here is the raw close-to-close interval
return, and annualization uses 245 trading days per year for consistency with
backtest metrics.
"""

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 245


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    pattern = str(root / "results" / "2026*_production_close_positions" / "wallet_close_*.json")

    records = []
    for f in sorted(glob.glob(pattern)):
        stem = Path(f).stem
        date_str = stem.split("_")[-1]
        try:
            dt = pd.to_datetime(date_str, format="%Y%m%d")
        except Exception:
            continue
        with open(f) as fh:
            data = json.load(fh)
        eq = data.get("ukeire_hosyoukin")
        if eq is None:
            continue
        records.append({"date": dt, "equity": float(eq)})

    if not records:
        print("No close wallet snapshots found.")
        return

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    df["daily_return"] = df["equity"].pct_change()
    returns = df["daily_return"].dropna().to_numpy()
    n = len(returns)

    if n < 2:
        print("Need at least two equity observations to compute Sharpe.")
        return

    mean_daily = float(np.mean(returns))
    std_daily = float(np.std(returns, ddof=1))
    ann_ret = mean_daily * TRADING_DAYS_PER_YEAR
    ann_vol = std_daily * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan

    total_ret = (df["equity"].iloc[-1] / df["equity"].iloc[0] - 1) * 100
    wealth = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(wealth)
    mdd = np.min(wealth / running_max - 1) * 100

    print("=== Live Production Sharpe (ukeire_hosyoukin proxy) ===")
    print(f"Period: {df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()}")
    print(f"Close snapshots: {len(df)}")
    print(f"Return observations: {n}")
    print(f"Total return: {total_ret:+.2f}%")
    print(f"Annualized return: {ann_ret*100:+.2f}%")
    print(f"Annualized volatility: {ann_vol*100:.2f}%")
    print(f"Max drawdown: {mdd:.2f}%")
    print(f"Live Sharpe ratio (245-day annualization): {sharpe:.2f}")
    print("\nDate       Equity        Daily Return")
    for _, row in df.iterrows():
        ret_str = f"{row['daily_return']*100:+.2f}%" if pd.notna(row["daily_return"]) else "N/A"
        print(f"{row['date'].date()}  {row['equity']:>12,.0f}  {ret_str}")


if __name__ == "__main__":
    main()
