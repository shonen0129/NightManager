"""Diagnostic script: compare 9:10 (High+Low)/2 approximation vs actual fill prices.

Usage:
    python3 scripts/experiments/diagnose_910_price_approximation.py [--ticker 1320.T] [--start 2025-01-01] [--end 2025-06-30]

If actual fill logs are available (CSV with columns: date, ticker, fill_price, side),
the script compares the (High+Low)/2 approximation against actual fills and reports
bias statistics. Without fill logs, it compares (High+Low)/2 vs Open price as a
sanity check on the approximation error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.cache import load_intraday_cache


def main():
    parser = argparse.ArgumentParser(description="Diagnose 9:10 price approximation bias")
    parser.add_argument("--ticker", default="1320.T", help="JP ticker to analyze")
    parser.add_argument("--start", default="2025-01-01", help="Start date")
    parser.add_argument("--end", default="2025-06-30", help="End date")
    parser.add_argument("--fill-log", default=None, help="Path to actual fill log CSV (date,ticker,fill_price,side)")
    args = parser.parse_args()

    df_5m = load_intraday_cache("5m")
    if df_5m is None or df_5m.empty:
        print("ERROR: No 5-minute data available")
        return

    dates = pd.date_range(args.start, args.end, freq="B")
    ticker = args.ticker

    approx_prices = []
    open_prices = []
    close_prices = []
    valid_dates = []

    for dt in dates:
        idx_910 = pd.Timestamp(f"{dt.date()} 09:10:00")
        if idx_910 not in df_5m.index:
            continue
        row = df_5m.loc[idx_910]

        high = row.get(("High", ticker))
        low = row.get(("Low", ticker))
        close = row.get(("Close", ticker))
        op = row.get(("Open", ticker))

        if pd.isna(high) or pd.isna(low):
            continue

        approx = (high + low) / 2
        approx_prices.append(approx)
        open_prices.append(op if not pd.isna(op) else np.nan)
        close_prices.append(close if not pd.isna(close) else np.nan)
        valid_dates.append(dt)

    if len(valid_dates) < 5:
        print(f"ERROR: Only {len(valid_dates)} valid dates found for {ticker}")
        return

    approx_arr = np.array(approx_prices)
    open_arr = np.array(open_prices)
    close_arr = np.array(close_prices)

    # Compare approx vs open
    diff_vs_open = (approx_arr - open_arr) / open_arr
    print(f"\n=== 9:10 Price Approximation Diagnostic: {ticker} ===")
    print(f"Date range: {valid_dates[0].date()} to {valid_dates[-1].date()} ({len(valid_dates)} days)")
    print(f"\n(High+Low)/2 vs Open:")
    print(f"  Mean bias:  {np.nanmean(diff_vs_open)*10000:.2f} bps")
    print(f"  Std:        {np.nanstd(diff_vs_open)*10000:.2f} bps")
    print(f"  Max abs:    {np.nanmax(np.abs(diff_vs_open))*10000:.2f} bps")
    print(f"  Median:     {np.nanmedian(diff_vs_open)*10000:.2f} bps")

    # Compare approx vs close (intraday drift)
    diff_vs_close = (approx_arr - close_arr) / close_arr
    print(f"\n(High+Low)/2 vs Close (9:10 bar):")
    print(f"  Mean:       {np.nanmean(diff_vs_close)*10000:.2f} bps")
    print(f"  Std:        {np.nanstd(diff_vs_close)*10000:.2f} bps")

    # If fill log provided, compare against actual fills
    if args.fill_log:
        fill_df = pd.read_csv(args.fill_log, parse_dates=["date"])
        fill_df = fill_df[fill_df["ticker"] == ticker]
        fill_df = fill_df.set_index("date")

        fill_approx_diffs = []
        for dt in valid_dates:
            if dt in fill_df.index:
                fill_price = float(fill_df.loc[dt, "fill_price"])
                approx = float(approx_arr[valid_dates.index(dt)])
                fill_approx_diffs.append((approx - fill_price) / fill_price)

        if fill_approx_diffs:
            diffs = np.array(fill_approx_diffs)
            print(f"\n=== Actual Fill Comparison ({len(diffs)} fills) ===")
            print(f"  Mean bias:  {np.mean(diffs)*10000:.2f} bps")
            print(f"  Std:        {np.std(diffs)*10000:.2f} bps")
            print(f"  Max abs:    {np.max(np.abs(diffs))*10000:.2f} bps")
            print(f"  Direction:  {'approx OVERESTIMATES fill' if np.mean(diffs) > 0 else 'approx UNDERESTIMATES fill'}")
        else:
            print("\nWARNING: No matching fill log entries found")
    else:
        print("\nNOTE: No --fill-log provided. Provide actual fill log for real bias measurement.")
        print("      Expected format: CSV with columns date,ticker,fill_price,side")


if __name__ == "__main__":
    main()
