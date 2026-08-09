#!/usr/bin/env python3
"""V2 バックテストと実際の受入保証金を比較し、ズレを可視化する。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
BT_DIR = ROOT / "results" / "v2_backtest_20200106_20260729_live"
RESULTS_ROOT = ROOT / "results"


def load_wallet_snapshots() -> pd.DataFrame:
    """Scan results/ for wallet_close_*.json and wallet_decision_*.json."""
    rows = []
    for d in sorted(RESULTS_ROOT.iterdir()):
        if not d.is_dir() or "_production_" not in d.name:
            continue
        for f in d.glob("wallet_*.json"):
            try:
                with open(f) as fh:
                    data = json.load(fh)
                ts = data.get("timestamp", "")
                if not ts:
                    continue
                date = pd.to_datetime(ts[:10])
                eq = data.get("ukeire_hosyoukin")
                if eq is None:
                    continue
                rows.append({
                    "date": date,
                    "timestamp": ts,
                    "ukeire_hosyoukin": float(eq),
                    "margin_ratio_str": data.get("sHosyouKinritu", ""),
                    "label": data.get("label", ""),
                    "source_dir": d.name,
                })
            except Exception as e:
                print(f"[WARN] Failed to load {f}: {e}")
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("date")
    return df


def extract_positions_summary(date_dir: Path) -> dict:
    pos_file = date_dir / f"positions_close_{date_dir.name.split('_')[0]}.json"
    if not pos_file.exists():
        pos_file = list(date_dir.glob("positions_*.json"))[0] if list(date_dir.glob("positions_*.json")) else None
    if pos_file is None or not pos_file.exists():
        return {}
    with open(pos_file) as f:
        data = json.load(f)
    positions = data.get("positions", [])
    if not positions:
        return {"total_unrealized_pnl": 0.0, "positions_count": 0}
    total_unr = sum(float(p.get("unrealized_pnl", 0.0)) for p in positions)
    return {
        "total_unrealized_pnl": total_unr,
        "positions_count": len(positions),
        "long_qty": sum(int(p.get("quantity", 0)) for p in positions if p.get("side") == "BUY"),
        "short_qty": sum(int(p.get("quantity", 0)) for p in positions if p.get("side") == "SELL"),
    }


def main():
    bt_returns = pd.read_csv(BT_DIR / "daily_net_returns.csv", index_col=0, parse_dates=True)["net_return"]
    bt_equity = pd.read_csv(BT_DIR / "daily_equity_curve.csv", index_col=0, parse_dates=True)["equity"]

    wallets = load_wallet_snapshots()
    if wallets.empty:
        print("No wallet snapshots found.")
        return

    # Use close (14:50) snapshots; decision (09:05) is before trading
    close_wallets = wallets[wallets["label"] == "close"].drop_duplicates("date", keep="last").set_index("date")

    # Compute actual daily returns from ukeire_hosyoukin close-to-close
    close_wallets["actual_ret"] = close_wallets["ukeire_hosyoukin"].pct_change()

    # Align with backtest dates
    common = bt_returns.index.intersection(close_wallets.index)
    bt_sync = bt_returns.loc[common]
    bt_eq_sync = bt_equity.loc[common]
    act_sync = close_wallets.loc[common]

    # Normalize actual equity to start at same level as backtest equity for comparison
    norm_factor = bt_eq_sync.iloc[0] / act_sync["ukeire_hosyoukin"].iloc[0]
    act_sync["norm_equity"] = act_sync["ukeire_hosyoukin"] * norm_factor
    act_sync["norm_ret"] = act_sync["norm_equity"].pct_change()

    print("=" * 90)
    print("V2 Backtest vs Actual (ukeire_hosyoukin) Comparison")
    print("=" * 90)
    print(f'{"Date":<12} {"BT Ret":>9} {"BT Equity":>12} {"Actual Eq":>12} {"Actual Ret":>11} {"Norm Ret":>10} {"Diff":>9}')
    print("-" * 90)
    diffs = []
    for date in common:
        bt_ret = bt_sync.loc[date]
        bt_eq = bt_eq_sync.loc[date]
        act_eq = act_sync.loc[date, "ukeire_hosyoukin"]
        norm_ret = act_sync.loc[date, "norm_ret"]
        actual_ret = act_sync.loc[date, "actual_ret"]
        diff = norm_ret - bt_ret if pd.notna(norm_ret) and pd.notna(bt_ret) else np.nan
        if pd.notna(diff):
            diffs.append(diff)
        print(
            f"{date.strftime('%Y-%m-%d')} "
            f"{bt_ret*100:>8.2f}% "
            f"{bt_eq:>11,.0f} "
            f"{act_eq:>11,} "
            f"{actual_ret*100:>10.2f}% "
            f"{norm_ret*100 if pd.notna(norm_ret) else 0:>9.2f}% "
            f"{diff*100 if pd.notna(diff) else 0:>8.2f}%"
        )

    if diffs:
        mean_diff = np.mean(diffs)
        rmse = np.sqrt(np.mean(np.square(diffs)))
        print("-" * 90)
        print(f"Mean daily diff (actual - BT): {mean_diff*100:+.3f}%")
        print(f"RMSE of daily diff:            {rmse*100:.3f}%")

    # Total return over overlap
    if len(common) >= 2:
        first_bt = bt_eq_sync.iloc[0]
        last_bt = bt_eq_sync.iloc[-1]
        first_act = act_sync["ukeire_hosyoukin"].iloc[0]
        last_act = act_sync["ukeire_hosyoukin"].iloc[-1]
        bt_total = last_bt / first_bt - 1
        act_total = last_act / first_act - 1
        print("\nTotal Return over overlap:")
        print(f"  BT:    {bt_total*100:+.2f}%  ({first_bt:,.0f} -> {last_bt:,.0f})")
        print(f"  Actual:{act_total*100:+.2f}%  ({first_act:,.0f} -> {last_act:,.0f})")
        print(f"  Gap:   {(bt_total - act_total)*100:+.2f}pp")


if __name__ == "__main__":
    main()
