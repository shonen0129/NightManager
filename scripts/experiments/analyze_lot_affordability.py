#!/usr/bin/env python3
"""各銘柄が小資本で購入可能かを分析する。

BacktestEngine の daily_weights を使い、capital * side_leverage に対して
各銘柄の目標ノーショナル・ロット価格・購入可能株数を計算する。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

JP_TICKERS = ["1617.T", "1618.T", "1619.T", "1620.T", "1621.T", "1622.T", "1623.T",
              "1624.T", "1625.T", "1626.T", "1627.T", "1628.T", "1629.T", "1630.T",
              "1631.T", "1632.T", "1633.T"]


def main():
    parser = argparse.ArgumentParser(description="Analyze lot affordability for small capital")
    parser.add_argument("--date", default="2026-07-29", help="Trade date to analyze")
    parser.add_argument("--capital", type=float, default=324_280.0, help="Capital (JPY)")
    parser.add_argument("--side-leverage", type=float, default=1.5, help="Side leverage")
    parser.add_argument("--price-source", default="5m", choices=["5m", "decision"], help="Price source")
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(ROOT / "src"))

    from leadlag.data.tickers import lot_size_for

    # Load weights
    bt_dir = ROOT / "results" / "v2_backtest_exact_production_20260729"
    w_df = pd.read_csv(bt_dir / "daily_weights.csv", index_col=0, parse_dates=True)
    dt = pd.to_datetime(args.date)
    if dt not in w_df.index:
        raise ValueError(f"Date {args.date} not in weights")
    w = w_df.loc[dt].values.astype(float)

    # Get prices
    prices = {}
    if args.price_source == "5m":
        from leadlag.data.cache import load_intraday_cache
        df_5m = load_intraday_cache("5m")
        day_data = df_5m[df_5m.index.date == dt.date()]
        if day_data.empty:
            raise ValueError(f"No 5m data for {args.date}")
        # Use 09:10 bar or first available
        idx = pd.Timestamp(f"{dt.date()} 09:10:00")
        if idx in day_data.index:
            bar = day_data.loc[idx]
        else:
            bar = day_data.iloc[0]
        for tk in JP_TICKERS:
            op = bar.get(("Open", tk))
            close = bar.get(("Close", tk))
            if pd.notna(close):
                prices[tk] = float(close)
            elif pd.notna(op):
                prices[tk] = float(op)
            else:
                prices[tk] = np.nan
    else:
        # decision CSV
        decision_path = ROOT / "live" / "production_residual_blpx" / "daily" / f"decision_{args.date.replace('-', '')}.csv"
        if not decision_path.exists():
            raise FileNotFoundError(f"Decision file not found: {decision_path}")
        df = pd.read_csv(decision_path)
        for _, row in df.iterrows():
            tk = row["ticker"]
            if tk.endswith(".T"):
                tk = tk.replace(".T", "")
            prices[tk] = float(row["open_price"])

    eff_capital = args.capital * args.side_leverage
    records = []
    for j, tk in enumerate(JP_TICKERS):
        weight = float(w[j])
        price = prices.get(tk)
        lot = lot_size_for(tk)
        lot_price = price * lot if pd.notna(price) else np.nan
        target_notional = abs(weight) * eff_capital
        if pd.isna(price) or weight == 0:
            affordable = "N/A"
            q = 0
        else:
            q = int(target_notional / lot_price) * lot
            affordable = "YES" if q >= lot else "NO"
        records.append({
            "ticker": tk,
            "weight": weight,
            "price": price,
            "lot": lot,
            "lot_price": lot_price,
            "target_notional": target_notional,
            "quantity_affordable": q,
            "affordable": affordable,
        })

    df = pd.DataFrame(records)
    print("\n=== Lot Affordability Analysis ===")
    print(f"Date: {args.date}, Capital: {args.capital:,.0f} JPY, Side Leverage: {args.side_leverage}")
    print(f"Effective Capital: {eff_capital:,.0f} JPY")
    print(df.to_string(index=False))
    print(f"\nAffordable: {len(df[df['affordable'] == 'YES'])}/{len(df[df['weight'] != 0])} non-zero weight tickers")
    print(f"Total target notional (non-zero): {df['target_notional'].sum():,.0f} JPY")
    print(f"Total affordable notional: {(df['quantity_affordable'] * df['lot_price']).sum():,.0f} JPY")


if __name__ == "__main__":
    main()
