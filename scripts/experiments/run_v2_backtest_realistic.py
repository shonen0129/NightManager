#!/usr/bin/env python3
"""V2 理論値・悲観的・現実的バックテストの中間版。

執行価格を 5分足 09:10 の (High+Low)/2 としつつ、整数株・ロット制約、
コスト、資本配分を反映。Close/Gap 価格は `--close-mode` / `--overnight-mode`
で切り替え可能。デフォルトは「執行価格のみ (H+L)/2、Close/Gap は悲観的」。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]

JP_TICKERS = ["1617.T", "1618.T", "1619.T", "1620.T", "1621.T", "1622.T", "1623.T",
              "1624.T", "1625.T", "1626.T", "1627.T", "1628.T", "1629.T", "1630.T",
              "1631.T", "1632.T", "1633.T"]


def _bar_value(bar: pd.Series, field: str, ticker: str) -> float | None:
    if bar is None:
        return None
    val = bar.get((field, ticker))
    if val is None or (isinstance(val, float) and not np.isfinite(val)):
        return None
    return float(val) if pd.notna(val) else None


def _find_bar(df_5m: pd.DataFrame, date: pd.Timestamp, ticker: str,
              time_strs: list[str] | None = None, latest_before: str | None = None) -> pd.Series | None:
    day_data = df_5m[df_5m.index.date == date.date()]
    if day_data.empty:
        return None
    if latest_before:
        cutoff = pd.Timestamp(f"{date.date()} {latest_before}")
        day_data = day_data[day_data.index <= cutoff]
        if day_data.empty:
            return None
        for i in range(len(day_data) - 1, -1, -1):
            bar = day_data.iloc[i]
            if pd.notna(bar.get(("Close", ticker))) or pd.notna(bar.get(("High", ticker))) \
                    or pd.notna(bar.get(("Low", ticker))):
                return bar
        return None
    if time_strs:
        for t in time_strs:
            idx = pd.Timestamp(f"{date.date()} {t}")
            if idx in day_data.index:
                bar = day_data.loc[idx]
                if pd.notna(bar.get(("Close", ticker))) or pd.notna(bar.get(("High", ticker))) \
                        or pd.notna(bar.get(("Low", ticker))):
                    return bar
        return None
    return day_data.iloc[0]


def _lot_size(ticker: str) -> int:
    from leadlag.data.tickers import lot_size_for
    return lot_size_for(ticker)


def _allocate_lots(target_notional: np.ndarray, side: np.ndarray, price: np.ndarray,
                   lot: np.ndarray, eff_capital: float, gross_limit_mult: float = 2.0,
                   rounding: str = "floor", reallocate: bool = False) -> np.ndarray:
    """Allocate integer lots to approximate target notional weights.

    Parameters:
        target_notional: signed target notional (long positive, short negative)
        side: sign array (long +1, short -1)
        price: execution price per share
        lot: lot size per ticker
        eff_capital: effective capital (capital * side_leverage)
        gross_limit_mult: gross exposure limit as multiple of eff_capital
        rounding: 'floor' or 'nearest'
        reallocate: redistribute residual to other tickers of the same side

    Returns:
        integer share quantities (signed)
    """
    n = len(target_notional)
    q = np.zeros(n, dtype=int)
    gross_limit_notional = gross_limit_mult * eff_capital

    # Base allocation
    for i in range(n):
        if side[i] == 0 or price[i] <= 0 or lot[i] <= 0:
            continue
        target = abs(target_notional[i])
        lot_price = price[i] * lot[i]
        if rounding == "nearest":
            q_i = int(round(target / lot_price)) * lot[i]
        else:
            q_i = int(target / lot_price) * lot[i]
        q_i = q_i if side[i] > 0 else -q_i
        q[i] = q_i

    # Enforce gross limit by scaling down if needed
    def _gross(q_arr):
        return float(np.sum(np.abs(q_arr * price)))

    current_gross = _gross(q)
    if current_gross > gross_limit_notional:
        scale = gross_limit_notional / current_gross
        for i in range(n):
            if side[i] == 0 or price[i] <= 0 or lot[i] <= 0:
                continue
            scaled = int(round(abs(q[i]) * scale / lot[i])) * lot[i]
            q[i] = scaled if side[i] > 0 else -scaled

    # Reallocation of residual to affordable tickers of the same side
    if reallocate:
        for _ in range(50):  # iterate up to 50 rounds
            current_gross = _gross(q)
            capacity = gross_limit_notional - current_gross
            if capacity < 1e-6:
                break
            best_idx = -1
            best_residual = 0.0
            for i in range(n):
                if side[i] == 0 or price[i] <= 0 or lot[i] <= 0:
                    continue
                target = abs(target_notional[i])
                actual = abs(q[i]) * price[i]
                residual = target - actual
                if residual <= 0:
                    continue
                lot_price = price[i] * lot[i]
                if lot_price > capacity:
                    continue
                if residual > best_residual:
                    best_residual = residual
                    best_idx = i
            if best_idx < 0:
                break
            q[best_idx] += lot[best_idx] if side[best_idx] > 0 else -lot[best_idx]

    return q


def price_from_mode(bar: pd.Series, ticker: str, side: int, mode: str, prefer_open: bool = False) -> float | None:
    """Return a price from the 5m bar based on mode.

    side=+1 long, side=-1 short. mode: 'adverse', 'midpoint', 'close', 'open'.
    """
    high = _bar_value(bar, "High", ticker)
    low = _bar_value(bar, "Low", ticker)
    close = _bar_value(bar, "Close", ticker)
    op = _bar_value(bar, "Open", ticker)

    if mode == "adverse":
        if side > 0:
            return low if low is not None else close
        else:
            return high if high is not None else close
    elif mode == "midpoint":
        if high is not None and low is not None:
            return (high + low) / 2.0
        return close
    elif mode == "close":
        return close
    elif mode == "open":
        return op
    else:
        raise ValueError(f"Unknown price mode: {mode}")


def main():
    parser = argparse.ArgumentParser(description="V2 realistic backtest with configurable 5m prices")
    parser.add_argument("--entry-mode", default="midpoint", choices=["adverse", "midpoint", "close", "open"],
                        help="09:10 execution price mode (default: midpoint)")
    parser.add_argument("--close-mode", default="adverse", choices=["adverse", "midpoint", "close", "open"],
                        help="Close-of-day price mode (default: adverse)")
    parser.add_argument("--overnight-mode", default="adverse", choices=["adverse", "midpoint", "close", "open"],
                        help="Next day open/gap price mode (default: adverse)")
    parser.add_argument("--rounding-mode", default="floor", choices=["floor", "nearest"],
                        help="Lot rounding mode (default: floor)")
    parser.add_argument("--reallocate-unaffordable", action="store_true",
                        help="Redistribute unaffordable/under-allocated notional to other tickers of the same side")
    parser.add_argument("--gross-limit", type=float, default=2.0,
                        help="Gross exposure limit relative to capital*side_leverage (default: 2.0)")
    parser.add_argument("--capital", type=float, default=324_280.0,
                        help="Initial capital (JPY, default: 324,280)")
    parser.add_argument("--output-dir", default=None, help="Output directory (default auto from modes)")
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(ROOT / "src"))

    from leadlag.data.cache import load_intraday_cache

    # Config
    config_path = ROOT / "configs" / "production" / "production.yaml"
    cfg = yaml.safe_load(open(config_path))
    costs = cfg.get("costs", {})
    alpha_long = float(costs.get("overnight_alpha_long", 0.75))
    alpha_short = float(costs.get("overnight_alpha_short", 0.5))
    fin_annual = float(costs.get("buy_interest_annual", 0.025))
    borrow_annual = float(costs.get("borrow_fee_annual", 0.0115))
    rev_bps = float(costs.get("reverse_fee_bps", 2.0))
    slip_bps = float(costs.get("slippage_bps_per_side", 5.0))
    side_leverage = float(
        cfg.get("execution", {}).get("side_leverage", cfg.get("portfolio", {}).get("side_leverage", 1.5))
    )
    initial_capital = float(args.capital)

    financing_daily = fin_annual / 365.0
    borrow_daily = borrow_annual / 365.0
    reverse_daily = rev_bps / 10000.0
    slip = slip_bps / 10000.0

    # Load exact backtest for comparison
    bt_dir = ROOT / "results" / "v2_backtest_exact_production_20260729"
    w_df = pd.read_csv(bt_dir / "daily_weights.csv", index_col=0, parse_dates=True)
    bt_returns = pd.read_csv(bt_dir / "daily_net_returns.csv", index_col=0, parse_dates=True)["net_return"]
    bt_equity = pd.read_csv(bt_dir / "daily_equity_curve.csv", index_col=0, parse_dates=True)["equity"]

    # Load 5m data
    df_5m = load_intraday_cache("5m")
    if df_5m is None or df_5m.empty:
        raise FileNotFoundError("5m intraday cache not found")

    five_dates = sorted(pd.Series(df_5m.index.date).unique())
    five_dates = pd.to_datetime(five_dates)
    overlap = w_df.index.intersection(five_dates)
    if len(overlap) == 0:
        raise ValueError("No overlapping dates between theoretical weights and 5m data")

    records = []
    capital = initial_capital
    w_prev = np.zeros(len(JP_TICKERS))
    equity = 1.0

    for i, trade_dt in enumerate(overlap):
        w_t = w_df.loc[trade_dt].values.astype(float)

        if np.abs(w_t).sum() < 1e-8:
            records.append({
                "trade_date": trade_dt.strftime("%Y-%m-%d"),
                "gross": 0.0, "net": 0.0, "gross_ret": 0.0, "overnight_ret": 0.0,
                "slip_cost": 0.0, "fin_cost": 0.0, "borrow_cost": 0.0, "reverse_cost": 0.0,
                "cost": 0.0, "net_ret": 0.0, "equity": equity, "capital": capital,
                "turnover": 0.0, "positions_count": 0,
            })
            w_prev = np.zeros(len(JP_TICKERS))
            continue

        eff_capital = capital * side_leverage

        # Collect prices and targets
        p_entry_arr = np.zeros(len(JP_TICKERS))
        p_close_arr = np.zeros(len(JP_TICKERS))
        lot_arr = np.zeros(len(JP_TICKERS), dtype=int)
        side_arr = np.zeros(len(JP_TICKERS), dtype=int)
        target_arr = np.zeros(len(JP_TICKERS))
        valid_mask = np.zeros(len(JP_TICKERS), dtype=bool)

        for j, tk in enumerate(JP_TICKERS):
            side = int(np.sign(w_t[j]))
            if side == 0:
                continue

            entry_bar = _find_bar(df_5m, trade_dt, tk, time_strs=["09:10:00", "09:05:00", "09:00:00"])
            close_bar = _find_bar(df_5m, trade_dt, tk, latest_before="15:00:00")
            if entry_bar is None or close_bar is None:
                continue

            p_entry = price_from_mode(entry_bar, tk, side, args.entry_mode)
            p_close = price_from_mode(close_bar, tk, side, args.close_mode)

            if p_entry is None or p_close is None or p_entry <= 0 or p_close <= 0:
                continue

            p_entry_arr[j] = p_entry
            p_close_arr[j] = p_close
            lot_arr[j] = _lot_size(tk)
            side_arr[j] = side
            target_arr[j] = w_t[j] * eff_capital  # signed target notional
            valid_mask[j] = True

        # Integer lot allocation with rounding / reallocation
        quantities = _allocate_lots(
            target_notional=target_arr,
            side=side_arr,
            price=p_entry_arr,
            lot=lot_arr,
            eff_capital=eff_capital,
            gross_limit_mult=args.gross_limit,
            rounding=args.rounding_mode,
            reallocate=args.reallocate_unaffordable,
        )
        notionals = np.where(quantities != 0, quantities * p_entry_arr, 0.0)
        prices_entry = np.where(valid_mask, p_entry_arr, 0.0)
        prices_close = np.where(valid_mask, p_close_arr, 0.0)
        valid_tk = np.where(quantities != 0)[0].tolist()

        w_actual = np.where(notionals != 0, notionals / eff_capital, 0.0)

        r_target = np.zeros(len(JP_TICKERS))
        for j in valid_tk:
            side = int(np.sign(w_t[j]))
            p_entry = prices_entry[j]
            p_close = prices_close[j]
            if side > 0:
                r_target[j] = (p_close / p_entry) - 1.0
            else:
                r_target[j] = (p_entry / p_close) - 1.0

        overnight_ret = 0.0
        if i + 1 < len(overlap):
            next_dt = overlap[i + 1]
            r_gap = np.zeros(len(JP_TICKERS))
            for j in valid_tk:
                tk = JP_TICKERS[j]
                side = int(np.sign(w_t[j]))
                next_bar = _find_bar(df_5m, next_dt, tk, time_strs=["09:00:00", "09:05:00", "09:10:00"])
                if next_bar is None:
                    continue
                p_next = price_from_mode(next_bar, tk, side, args.overnight_mode)
                p_prev_close = prices_close[j]
                if p_next is None or p_prev_close is None or p_prev_close <= 0:
                    continue
                r_gap[j] = (p_next / p_prev_close) - 1.0 if side > 0 else (1.0 - (p_next / p_prev_close))

            alpha_mask = np.where(w_actual > 0, alpha_long, np.where(w_actual < 0, alpha_short, 0.0))
            overnight_ret = side_leverage * float(np.sum(alpha_mask * w_actual * r_gap))

        calendar_days = 1
        if i + 1 < len(overlap):
            calendar_days = (overlap[i + 1] - trade_dt).days

        gross_ret = side_leverage * float(np.sum(w_actual * r_target))
        alpha_mask = np.where(w_actual > 0, alpha_long, np.where(w_actual < 0, alpha_short, 0.0))

        turnover = float(np.sum(np.abs(w_actual - w_prev)) / 2.0)
        slip_cost = side_leverage * slip * (
            2.0 * np.sum((1.0 - alpha_mask) * np.abs(w_actual))
            + np.sum(alpha_mask * np.abs(w_actual - w_prev) / 2.0)
        )
        held_long = float(np.sum(alpha_mask * np.maximum(w_actual, 0.0)))
        held_short = float(np.sum(alpha_mask * np.maximum(-w_actual, 0.0)))
        fin_cost = side_leverage * held_long * financing_daily * calendar_days
        borrow_cost = side_leverage * held_short * borrow_daily * calendar_days
        reverse_cost = side_leverage * held_short * reverse_daily * calendar_days
        cost = slip_cost + fin_cost + borrow_cost + reverse_cost

        net_ret = gross_ret + overnight_ret - cost
        equity *= 1.0 + net_ret
        capital *= 1.0 + net_ret

        records.append({
            "trade_date": trade_dt.strftime("%Y-%m-%d"),
            "gross": float(np.sum(np.abs(w_actual))),
            "net": float(np.sum(w_actual)),
            "gross_ret": gross_ret,
            "overnight_ret": overnight_ret,
            "slip_cost": slip_cost,
            "fin_cost": fin_cost,
            "borrow_cost": borrow_cost,
            "reverse_cost": reverse_cost,
            "cost": cost,
            "net_ret": net_ret,
            "equity": equity,
            "capital": capital,
            "turnover": turnover,
            "positions_count": np.sum(quantities != 0),
        })

        w_prev = w_actual

    df = pd.DataFrame(records)

    out_suffix = f"{args.entry_mode}_{args.close_mode}_{args.overnight_mode}"
    out_dir = ROOT / "results" / f"v2_backtest_realistic_5m_{out_suffix}"
    if args.output_dir:
        out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    df.set_index("trade_date").to_csv(out_dir / "daily_results.csv")

    # Compare
    theory_slice = bt_returns.loc[overlap]
    theory_equity_slice = bt_equity.loc[overlap]
    theory_cumret = theory_equity_slice.iloc[-1] / theory_equity_slice.iloc[0] - 1.0

    pess_cumret = equity - 1.0
    pess_returns = pd.Series(df["net_ret"].values, index=overlap)
    pess_mean = float(pess_returns.mean())
    pess_std = float(pess_returns.std(ddof=1))
    pess_sharpe = pess_mean / pess_std * np.sqrt(252) if pess_std > 1e-8 else 0.0
    pess_ar = pess_mean * 252 * 100
    pess_vol = pess_std * np.sqrt(252) * 100
    pess_mdd = float((pd.Series(df["equity"].values, index=overlap) / pd.Series(df["equity"].values, index=overlap).cummax() - 1.0).min() * 100)
    pess_avg_turn = float(df["turnover"].mean())
    pess_avg_gross = float(df["gross"].mean())
    pess_avg_cost = float(df["cost"].mean()) * 10000
    pess_avg_pos = float(df["positions_count"].mean())

    theory_mean = float(theory_slice.mean())
    theory_std = float(theory_slice.std(ddof=1))
    theory_sharpe = theory_mean / theory_std * np.sqrt(252) if theory_std > 1e-8 else 0.0
    theory_ar = theory_mean * 252 * 100
    theory_vol = theory_std * np.sqrt(252) * 100
    theory_mdd = float((theory_equity_slice / theory_equity_slice.cummax() - 1.0).min() * 100)
    theory_avg_cost = float(pd.read_csv(bt_dir / "daily_costs_total.csv", index_col=0, parse_dates=True).loc[overlap, "cost"].mean()) * 10000

    print("\n" + "=" * 60)
    print(f"=== V2 Realistic Backtest (entry={args.entry_mode}, close={args.close_mode}, overnight={args.overnight_mode}) ===")
    print("=" * 60)
    print(f"  Period: {overlap[0].date()} -> {overlap[-1].date()}")
    print(f"  Days:   {len(df)}")
    print(f"  Theoretical window return: {theory_cumret*100:.2f}%")
    print(f"  Realistic window return:   {pess_cumret*100:.2f}%")
    print(f"  AR: {pess_ar:.2f}%")
    print(f"  Vol: {pess_vol:.2f}%")
    print(f"  Sharpe: {pess_sharpe:.2f}")
    print(f"  Max DD: {pess_mdd:.2f}%")
    print(f"  Avg Turnover: {pess_avg_turn:.4f}")
    print(f"  Avg Gross: {pess_avg_gross:.4f}")
    print(f"  Avg Cost/day: {pess_avg_cost:.2f} bps")
    print(f"  Avg Positions/day: {pess_avg_pos:.1f}")
    print("=" * 60)

    summary = {
        "period_start": str(overlap[0].date()),
        "period_end": str(overlap[-1].date()),
        "days": int(len(df)),
        "initial_capital_jpy": initial_capital,
        "final_capital_jpy": float(capital),
        "realistic_cumret_pct": float(pess_cumret * 100),
        "theoretical_cumret_pct": float(theory_cumret * 100),
        "realistic_ar_pct": float(pess_ar),
        "realistic_vol_pct": float(pess_vol),
        "realistic_sharpe": float(pess_sharpe),
        "realistic_mdd_pct": float(pess_mdd),
        "realistic_avg_turnover": float(pess_avg_turn),
        "realistic_avg_gross": float(pess_avg_gross),
        "realistic_avg_cost_bps": float(pess_avg_cost),
        "realistic_avg_positions": float(pess_avg_pos),
        "theoretical_ar_pct": float(theory_ar),
        "theoretical_vol_pct": float(theory_vol),
        "theoretical_sharpe": float(theory_sharpe),
        "theoretical_mdd_pct": float(theory_mdd),
        "theoretical_avg_cost_bps": float(theory_avg_cost),
        "entry_mode": args.entry_mode,
        "close_mode": args.close_mode,
        "overnight_mode": args.overnight_mode,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Report
    report_lines = [
        "# V2 理論値 vs 現実的バックテスト\n\n",
        f"## 仮定\n\n",
        f"- 9:10 執行価格: **{args.entry_mode}**（'{args.entry_mode}'）\n",
        f"- 15:00 Close 価格: **{args.close_mode}**（'{args.close_mode}'）\n",
        f"- 翌日寄りギャップ: **{args.overnight_mode}**（'{args.overnight_mode}'）\n",
        "- 整数株・ロット（1629.T=10株）切り捨て丸め\n",
        "- 初期資本 324,280 JPY、side_leverage=1.5、slippage=5 bps/side\n\n",
        f"## 期間\n\n",
        f"{overlap[0].date()} 〜 {overlap[-1].date()}（{len(df)} 営業日）\n\n",
        "## 主要指標比較\n\n",
        "| 指標 | 理論値 | 現実的 |\n",
        "|---|---:|---:|\n",
        f"| 累積リターン | {theory_cumret*100:.2f}% | {pess_cumret*100:.2f}% |\n",
        f"| AR | {theory_ar:.2f}% | {pess_ar:.2f}% |\n",
        f"| Volatility | {theory_vol:.2f}% | {pess_vol:.2f}% |\n",
        f"| Sharpe | {theory_sharpe:.2f} | {pess_sharpe:.2f} |\n",
        f"| Max DD | {theory_mdd:.2f}% | {pess_mdd:.2f}% |\n",
        f"| 平均コスト/日 | {theory_avg_cost:.2f} bps | {pess_avg_cost:.2f} bps |\n",
        f"| 平均ターンオーバー | — | {pess_avg_turn:.4f} |\n",
        f"| 平均グロス | — | {pess_avg_gross:.4f} |\n",
        f"| 平均ポジション数 | — | {pess_avg_pos:.1f} |\n\n",
        "## 結論\n\n",
        "執行価格を (High+Low)/2 にするだけでも、理論値と整数株丸めの間で大きな差が出る。\n",
        "特に高額銘柄・1629.T は 1 ロットを買えず、目標ウェイトが実現できない。\n",
    ]
    (out_dir / "report.md").write_text("".join(report_lines), encoding="utf-8")
    print(f"\nReport and outputs written to {out_dir}")


if __name__ == "__main__":
    main()
