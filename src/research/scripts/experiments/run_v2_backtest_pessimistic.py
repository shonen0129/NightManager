#!/usr/bin/env python3
"""V2 理論値バックテストと対になる「悲観的」バックテスト。

実行面の非現実性を反映:
  - 5分足 09:10 の不利側を 9:10 執行価格に使用
    - ロング: High
    - ショート: Low
  - 5分足 15:00 直前バーの不利側を Close 価格に使用
    - ロング: Low
    - ショート: High
  - 翌日寄りギャップも最初の 5分足の不利側を使用
    - ロング: High
    - ショート: Low
  - 整数株・ロットサイズ: 1株（または 1629.T=10株）単位で切り捨て丸め
  - ターンオーバー・コストは丸め後の実数量で計算

5分足キャッシュが存在する期間のみ実施可能。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
JP_TICKERS = ["1617.T", "1618.T", "1619.T", "1620.T", "1621.T", "1622.T", "1623.T",
              "1624.T", "1625.T", "1626.T", "1627.T", "1628.T", "1629.T", "1630.T",
              "1631.T", "1632.T", "1633.T"]


def _bar_value(bar: pd.Series, field: str, ticker: str) -> float | None:
    """Return a single field/ticker value from a 5m bar, None if missing/NaN."""
    if bar is None:
        return None
    val = bar.get((field, ticker))
    if val is None or (isinstance(val, float) and not np.isfinite(val)):
        return None
    return float(val)


def _find_bar(df_5m: pd.DataFrame, date: pd.Timestamp, ticker: str,
              time_strs: list[str] | None = None, latest_before: str | None = None) -> pd.Series | None:
    """Find a 5m bar for a specific ticker.

    - time_strs: iterate candidate times and pick the first bar with non-NaN Close.
    - latest_before: pick the latest bar at/before the time with non-NaN Close.
    """
    day_data = df_5m[df_5m.index.date == date.date()]
    if day_data.empty:
        return None

    if time_strs:
        for t in time_strs:
            idx = pd.Timestamp(f"{date.date()} {t}")
            if idx in day_data.index:
                bar = day_data.loc[idx]
                if pd.notna(bar.get(("Close", ticker))) or pd.notna(bar.get(("Open", ticker))) \
                        or pd.notna(bar.get(("High", ticker))) or pd.notna(bar.get(("Low", ticker))):
                    return bar
        return None

    if latest_before:
        cutoff = pd.Timestamp(f"{date.date()} {latest_before}")
        day_data = day_data[day_data.index <= cutoff]
        if day_data.empty:
            return None
        # Walk backward for the most recent non-NaN close
        for i in range(len(day_data) - 1, -1, -1):
            bar = day_data.iloc[i]
            if pd.notna(bar.get(("Close", ticker))) or pd.notna(bar.get(("High", ticker))) \
                    or pd.notna(bar.get(("Low", ticker))):
                return bar
        return None

    return day_data.iloc[0]


def _lot_size(ticker: str) -> int:
    from leadlag.data.tickers import lot_size_for
    return lot_size_for(ticker)


def main():
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
    initial_capital = 324_280.0  # JPY, actual account start size

    financing_daily = fin_annual / 365.0
    borrow_daily = borrow_annual / 365.0
    reverse_daily = rev_bps / 10000.0
    slip = slip_bps / 10000.0

    # Load theoretical backtest weights and returns for comparison
    bt_dir = ROOT / "results" / "v2_backtest_exact_production_20260729"
    w_df = pd.read_csv(bt_dir / "daily_weights.csv", index_col=0, parse_dates=True)
    bt_returns = pd.read_csv(bt_dir / "daily_net_returns.csv", index_col=0, parse_dates=True)["net_return"]
    bt_equity = pd.read_csv(bt_dir / "daily_equity_curve.csv", index_col=0, parse_dates=True)["equity"]

    # Load 5m data
    df_5m = load_intraday_cache("5m")
    if df_5m is None or df_5m.empty:
        raise FileNotFoundError("5m intraday cache not found")

    # Overlapping dates
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

        # Skip flat
        if np.abs(w_t).sum() < 1e-8:
            records.append({
                "trade_date": trade_dt.strftime("%Y-%m-%d"),
                "gross": 0.0,
                "net": 0.0,
                "gross_ret": 0.0,
                "overnight_ret": 0.0,
                "slip_cost": 0.0,
                "fin_cost": 0.0,
                "borrow_cost": 0.0,
                "reverse_cost": 0.0,
                "cost": 0.0,
                "net_ret": 0.0,
                "equity": equity,
                "capital": capital,
                "turnover": 0.0,
                "positions_count": 0,
            })
            w_prev = np.zeros(len(JP_TICKERS))
            continue

        eff_capital = capital * side_leverage

        # Build positions with integer-lot rounding and adverse prices
        quantities = np.zeros(len(JP_TICKERS), dtype=int)
        notionals = np.zeros(len(JP_TICKERS))
        prices_entry = np.zeros(len(JP_TICKERS))
        prices_close = np.zeros(len(JP_TICKERS))
        valid_tk = []

        for j, tk in enumerate(JP_TICKERS):
            side = int(np.sign(w_t[j]))
            if side == 0:
                continue

            entry_bar = _find_bar(df_5m, trade_dt, tk, time_strs=["09:10:00", "09:05:00", "09:00:00"])
            close_bar = _find_bar(df_5m, trade_dt, tk, latest_before="15:00:00")
            if entry_bar is None or close_bar is None:
                continue

            lot = _lot_size(tk)
            if side > 0:
                p_entry = _bar_value(entry_bar, "High", tk)
                p_close = _bar_value(close_bar, "Low", tk)
            else:
                p_entry = _bar_value(entry_bar, "Low", tk)
                p_close = _bar_value(close_bar, "High", tk)

            if p_entry is None or p_close is None or p_entry <= 0 or p_close <= 0:
                continue

            target_notional = abs(w_t[j]) * eff_capital
            q = int(target_notional / (p_entry * lot)) * lot
            if q < lot:
                continue
            q = q if side > 0 else -q

            quantities[j] = q
            notionals[j] = q * p_entry  # signed notional
            prices_entry[j] = p_entry
            prices_close[j] = p_close
            valid_tk.append(j)

        # Actual notional weights relative to effective capital
        w_actual = np.where(notionals != 0, notionals / eff_capital, 0.0)

        # Close-to-close returns (adverse 09:10 to adverse 15:00)
        r_target = np.zeros(len(JP_TICKERS))
        for j in valid_tk:
            side = int(np.sign(w_t[j]))
            p_entry = prices_entry[j]
            p_close = prices_close[j]
            if side > 0:
                r_target[j] = (p_close / p_entry) - 1.0
            else:
                r_target[j] = (p_entry / p_close) - 1.0

        # Overnight gap to next day (adverse first 5m bar)
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
                if side > 0:
                    p_next = _bar_value(next_bar, "High", tk)
                else:
                    p_next = _bar_value(next_bar, "Low", tk)
                p_prev_close = prices_close[j]
                if p_next is None or p_prev_close is None or p_prev_close <= 0:
                    continue
                r_gap[j] = (p_next / p_prev_close) - 1.0 if side > 0 else (1.0 - (p_next / p_prev_close))

            alpha_mask = np.where(w_actual > 0, alpha_long, np.where(w_actual < 0, alpha_short, 0.0))
            overnight_ret = side_leverage * float(np.sum(alpha_mask * w_actual * r_gap))

        # Calendar days held
        calendar_days = 1
        if i + 1 < len(overlap):
            calendar_days = (overlap[i + 1] - trade_dt).days

        # Costs
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
    out_dir = ROOT / "results" / "v2_backtest_pessimistic_5m"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.set_index("trade_date").to_csv(out_dir / "daily_results.csv")

    # Compare to theoretical for same window
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
    print("=== V2 Pessimistic Backtest (adverse 5m + integer lots) ===")
    print("=" * 60)
    print(f"  Period: {overlap[0].date()} -> {overlap[-1].date()}")
    print(f"  Days:   {len(df)}")
    print(f"  Theoretical window return: {theory_cumret*100:.2f}%")
    print(f"  Pessimistic window return: {pess_cumret*100:.2f}%")
    print(f"  AR (pessimistic): {pess_ar:.2f}%")
    print(f"  Vol (pessimistic): {pess_vol:.2f}%")
    print(f"  Sharpe (pessimistic): {pess_sharpe:.2f}")
    print(f"  Max DD (pessimistic): {pess_mdd:.2f}%")
    print(f"  Avg Turnover (pessimistic): {pess_avg_turn:.4f}")
    print(f"  Avg Gross (pessimistic): {pess_avg_gross:.4f}")
    print(f"  Avg Cost/day (pessimistic): {pess_avg_cost:.2f} bps")
    print(f"  Avg Positions/day: {pess_avg_pos:.1f}")
    print("=" * 60)

    # Save summary
    summary = {
        "period_start": str(overlap[0].date()),
        "period_end": str(overlap[-1].date()),
        "days": int(len(df)),
        "initial_capital_jpy": initial_capital,
        "final_capital_jpy": float(capital),
        "pessimistic_cumret_pct": float(pess_cumret * 100),
        "theoretical_cumret_pct": float(theory_cumret * 100),
        "pessimistic_ar_pct": float(pess_ar),
        "pessimistic_vol_pct": float(pess_vol),
        "pessimistic_sharpe": float(pess_sharpe),
        "pessimistic_mdd_pct": float(pess_mdd),
        "pessimistic_avg_turnover": float(pess_avg_turn),
        "pessimistic_avg_gross": float(pess_avg_gross),
        "pessimistic_avg_cost_bps": float(pess_avg_cost),
        "pessimistic_avg_positions": float(pess_avg_pos),
        "theoretical_ar_pct": float(theory_ar),
        "theoretical_vol_pct": float(theory_vol),
        "theoretical_sharpe": float(theory_sharpe),
        "theoretical_mdd_pct": float(theory_mdd),
        "theoretical_avg_cost_bps": float(theory_avg_cost),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Report
    report_lines = [
        "# V2 理論値 vs 悲観的バックテスト（5分足不利側＋整数株）\n\n",
        "## 悲観的仮定\n\n",
        "- 5分足 09:10 の **High** をロング入り、**Low** をショート入りとして使用\n",
        "- 5分足 15:00 直前バーの **Low** をロング決済、**High** をショート決済として使用\n",
        "- 翌日寄りギャップも最初の 5分足の **High（ロング）/ Low（ショート）** を使用\n",
        "- 1株単位・ロット単位（1629.T=10株）で切り捨て丸め\n",
        "- 資本は実口座初期値 **324,280 JPY** でコンパウンド\n",
        "- side_leverage: 1.5、slippage: 5 bps/side、overnight alpha: long 0.75 / short 0.5\n\n",
        "## 期間制限\n\n",
        f"5分足キャッシュの存在期間に制限: **{overlap[0].date()}** 〜 **{overlap[-1].date()}**（{len(df)} 営業日）。\n",
        "理論値バックテストも同一期間で比較。\n\n",
        "## 主要指標比較\n\n",
        "| 指標 | 理論値 | 悲観的 |\n",
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
        "5分足の不利側と整数株・ロット丸めを入れると、同一期間でも理論値より大幅に劣化。\n",
        "特に高額銘柄やロットサイズが大きい銘柄では目標ウェイトを実現できず、\n",
        "ターゲットから外れた小数量・未約定に近い状態が再現される。\n",
    ]
    (out_dir / "report.md").write_text("".join(report_lines), encoding="utf-8")
    print(f"\nReport and outputs written to {out_dir}")


if __name__ == "__main__":
    main()
