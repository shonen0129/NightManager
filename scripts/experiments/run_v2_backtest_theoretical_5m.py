#!/usr/bin/env python3
"""理論値バックテストを 5分足キャッシュの存在期間（2026-03-03 〜 2026-06-01）に切り出して保存。

これを「理論値的バックテスト」として、悲観的バックテストと対比する。
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

def main():
    bt_dir = ROOT / "results" / "v2_backtest_exact_production_20260729"
    out_dir = ROOT / "results" / "v2_backtest_theoretical_5m"
    out_dir.mkdir(parents=True, exist_ok=True)

    daily_returns = pd.read_csv(bt_dir / "daily_net_returns.csv", index_col=0, parse_dates=True)["net_return"]
    equity = pd.read_csv(bt_dir / "daily_equity_curve.csv", index_col=0, parse_dates=True)["equity"]
    weights = pd.read_csv(bt_dir / "daily_weights.csv", index_col=0, parse_dates=True)
    costs = pd.read_csv(bt_dir / "daily_costs_total.csv", index_col=0, parse_dates=True)["cost"]
    turnover = pd.read_csv(bt_dir / "daily_turnover.csv", index_col=0, parse_dates=True)["turnover"]
    gross = pd.read_csv(bt_dir / "daily_gross.csv", index_col=0, parse_dates=True)["gross"]
    fallback = pd.read_csv(bt_dir / "daily_fallback.csv", index_col=0, parse_dates=True)["fallback"]

    # 5m window used by pessimistic backtest (use exact same trade dates)
    pess_dir = ROOT / "results" / "v2_backtest_pessimistic_5m"
    if (pess_dir / "daily_results.csv").exists():
        pess_dates = pd.read_csv(pess_dir / "daily_results.csv", parse_dates=["trade_date"])["trade_date"]
    elif (pess_dir / "summary.json").exists():
        import json
        with open(pess_dir / "summary.json") as f:
            pess_summary = json.load(f)
        start = pd.to_datetime(pess_summary["period_start"])
        end = pd.to_datetime(pess_summary["period_end"])
        pess_dates = pd.date_range(start, end, freq="B")
    else:
        start = pd.to_datetime("2026-03-03")
        end = pd.to_datetime("2026-06-01")
        pess_dates = pd.date_range(start, end, freq="B")

    start = pess_dates.min()
    end = pess_dates.max()

    mask = daily_returns.index.isin(pess_dates)
    ret_slice = daily_returns.loc[mask]
    eq_slice = equity.loc[mask]
    w_slice = weights.loc[mask]
    cost_slice = costs.loc[mask]
    turn_slice = turnover.loc[mask]
    gross_slice = gross.loc[mask]
    fb_slice = fallback.loc[mask]

    ret_slice.to_csv(out_dir / "daily_net_returns.csv", header=["net_return"])
    eq_slice.to_csv(out_dir / "daily_equity_curve.csv", header=["equity"])
    w_slice.to_csv(out_dir / "daily_weights.csv")
    cost_slice.to_csv(out_dir / "daily_costs_total.csv", header=["cost"])
    turn_slice.to_csv(out_dir / "daily_turnover.csv", header=["turnover"])
    gross_slice.to_csv(out_dir / "daily_gross.csv", header=["gross"])
    fb_slice.to_csv(out_dir / "daily_fallback.csv", header=["fallback"])

    mean_ret = ret_slice.mean()
    std_ret = ret_slice.std(ddof=1)
    sharpe = mean_ret / std_ret * np.sqrt(252) if std_ret > 1e-8 else 0.0
    ar = mean_ret * 252 * 100
    vol = std_ret * np.sqrt(252) * 100
    mdd = float((eq_slice / eq_slice.cummax() - 1.0).min() * 100)
    cumret = eq_slice.iloc[-1] / eq_slice.iloc[0] - 1.0
    avg_cost = float(cost_slice.mean() * 10000)
    avg_turn = float(turn_slice.mean())
    avg_gross = float(gross_slice.mean())
    avg_fb = float(fb_slice.mean())

    print("=" * 60)
    print("=== V2 Theoretical Backtest (5m window) ===")
    print("=" * 60)
    print(f"  Period: {start.date()} -> {end.date()}")
    print(f"  Days:   {len(ret_slice)}")
    print(f"  Cumulative Return: {cumret*100:.2f}%")
    print(f"  AR: {ar:.2f}%")
    print(f"  Vol: {vol:.2f}%")
    print(f"  Sharpe: {sharpe:.2f}")
    print(f"  Max DD: {mdd:.2f}%")
    print(f"  Avg Cost/day: {avg_cost:.2f} bps")
    print(f"  Avg Turnover: {avg_turn:.4f}")
    print(f"  Avg Gross: {avg_gross:.4f}")
    print(f"  Avg Fallback: {avg_fb:.2%}")
    print("=" * 60)

    summary = {
        "period_start": str(start.date()),
        "period_end": str(end.date()),
        "days": int(len(ret_slice)),
        "cumulative_return_pct": float(cumret * 100),
        "ar_pct": float(ar),
        "vol_pct": float(vol),
        "sharpe": float(sharpe),
        "mdd_pct": float(mdd),
        "avg_cost_bps": float(avg_cost),
        "avg_turnover": float(avg_turn),
        "avg_gross": float(avg_gross),
        "avg_fallback": float(avg_fb),
    }
    import json
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    report_lines = [
        "# V2 理論値バックテスト（5分足対象期間）\n\n",
        f"期間: {start.date()} 〜 {end.date()}\n",
        f"日数: {len(ret_slice)}\n\n",
        "## 主要指標\n\n",
        f"- 累積リターン: {cumret*100:.2f}%\n",
        f"- AR: {ar:.2f}%\n",
        f"- Volatility: {vol:.2f}%\n",
        f"- Sharpe: {sharpe:.2f}\n",
        f"- Max DD: {mdd:.2f}%\n",
        f"- 平均コスト/日: {avg_cost:.2f} bps\n",
        f"- 平均ターンオーバー: {avg_turn:.4f}\n",
        f"- 平均グロス: {avg_gross:.4f}\n",
        f"- 平均フォールバック率: {avg_fb:.2%}\n\n",
        "## 説明\n\n",
        "これは `results/v2_backtest_exact_production_20260729` の理論値結果を、\n",
        "5分足キャッシュが存在する期間に切り出したもの。\n",
        "悲観的バックテスト `results/v2_backtest_pessimistic_5m` と同一期間で比較可能。\n",
    ]
    (out_dir / "report.md").write_text("".join(report_lines), encoding="utf-8")
    print(f"\nTheoretical 5m window outputs written to {out_dir}")


if __name__ == "__main__":
    main()
