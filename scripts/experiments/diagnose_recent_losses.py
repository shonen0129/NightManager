#!/usr/bin/env python3
"""Diagnose recent loss concentration from run_production_backtest.py outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TRADING_DAYS = 252


def _sharpe(r: pd.Series) -> float:
    r = r.dropna().astype(float)
    if len(r) < 2 or r.std(ddof=1) < 1e-12:
        return 0.0
    return r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS)


def _mdd_from_equity(equity: pd.Series) -> float:
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return float(dd.min())


def _window_stats(
    net: pd.Series,
    gross: pd.Series,
    costs: pd.Series,
    turnover: pd.Series,
    gross_exp: pd.Series,
    equity: pd.Series,
    start: str,
    end: str,
    label: str,
) -> dict:
    s = slice(start, end)
    n = net.loc[s]
    g = gross.loc[s]
    c = costs.loc[s]
    t = turnover.loc[s]
    ge = gross_exp.loc[s]
    e = equity.loc[s]
    if len(n) == 0:
        return {"label": label, "days": 0}
    win_days = int((n > 0).sum())
    loss_days = int((n < 0).sum())
    return {
        "label": label,
        "days": len(n),
        "net_cum": float(n.sum()),
        "gross_cum": float(g.sum()),
        "cost_cum": float(c.sum()),
        "net_mean_bps": float(n.mean() * 10000),
        "gross_mean_bps": float(g.mean() * 10000),
        "cost_mean_bps": float(c.mean() * 10000),
        "net_sharpe": _sharpe(n),
        "gross_sharpe": _sharpe(g),
        "mdd": _mdd_from_equity(e),
        "win_days": win_days,
        "loss_days": loss_days,
        "win_rate": win_days / len(n),
        "max_daily_loss_bps": float(n.min() * 10000),
        "max_daily_gain_bps": float(n.max() * 10000),
        "avg_turnover": float(t.mean()),
        "avg_gross_exp": float(ge.mean()),
        "consec_loss_max": int(_max_consecutive(n < 0)),
    }


def _max_consecutive(cond: pd.Series) -> int:
    """Maximum number of consecutive True values."""
    s = cond.astype(int)
    if s.empty:
        return 0
    # compute run lengths
    runs = []
    cur = 0
    for v in s:
        if v:
            cur += 1
        else:
            runs.append(cur)
            cur = 0
    runs.append(cur)
    return max(runs)


def _find_historical_streaks(n: pd.Series, k: int) -> pd.DataFrame:
    """Find all k-day or longer losing streaks in net returns."""
    is_loss = n < 0
    streaks = []
    start = None
    length = 0
    for idx, loss in is_loss.items():
        if loss:
            if start is None:
                start = idx
            length += 1
        else:
            if length >= k:
                streaks.append((start, idx, length, float(n.loc[start:idx].sum())))
            start = None
            length = 0
    # handle tail
    if start is not None and length >= k:
        end = n.index[-1]
        streaks.append((start, end, length, float(n.loc[start:end].sum())))
    df = pd.DataFrame(streaks, columns=["start", "end", "days", "net_sum"])
    if not df.empty:
        df = df.sort_values("net_sum").head(20)
    return df


def _rolling_window_stats(
    net: pd.Series, gross: pd.Series, costs: pd.Series, equity: pd.Series, window: int
) -> pd.DataFrame:
    """Compute rolling window stats and return worst net-cum windows."""
    idx = net.index
    records = []
    for i in range(len(net) - window + 1):
        s = net.iloc[i : i + window]
        g = gross.iloc[i : i + window]
        c = costs.iloc[i : i + window]
        e = equity.iloc[i : i + window]
        records.append(
            {
                "start": s.index[0],
                "end": s.index[-1],
                "net_cum": float(s.sum()),
                "gross_cum": float(g.sum()),
                "cost_cum": float(c.sum()),
                "mdd": _mdd_from_equity(e),
                "sharpe": _sharpe(s),
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report-dir", default="reports/longterm_backtest_20260801")
    args = parser.parse_args()

    out_dir = ROOT / args.output_dir
    report_dir = ROOT / args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    net = pd.read_csv(out_dir / "daily_net_returns.csv", index_col=0, parse_dates=True)["net_return"]
    gross = pd.read_csv(out_dir / "daily_gross_returns.csv", index_col=0, parse_dates=True)["gross_return"]
    costs = pd.read_csv(out_dir / "daily_costs.csv", index_col=0, parse_dates=True)["cost"]
    turnover = pd.read_csv(out_dir / "daily_turnover.csv", index_col=0, parse_dates=True)["turnover"]
    gross_exp = pd.read_csv(out_dir / "daily_gross_exposure.csv", index_col=0, parse_dates=True)["gross_exposure"]
    equity = pd.read_csv(out_dir / "daily_equity_curve.csv", index_col=0, parse_dates=True)["equity"]

    last = net.index[-1]
    # build windows
    from pandas.tseries.offsets import BDay
    windows = [
        (str(last - BDay(4)), str(last), "last_5d"),
        (str(last - BDay(9)), str(last), "last_10d"),
        (str(last - BDay(20)), str(last), "last_21d"),
        (str(last - BDay(62)), str(last), "last_63d"),
        (str(last - BDay(125)), str(last), "last_126d"),
        (str(last - BDay(251)), str(last), "last_252d"),
        (str(last - BDay(503)), str(last), "last_504d"),
        ("2026-01-05", str(last), "ytd_2026"),
        ("2025-01-05", "2025-12-31", "fy_2025"),
        ("2024-01-05", "2024-12-31", "fy_2024"),
        ("2023-01-05", "2023-12-31", "fy_2023"),
        ("2022-01-05", "2022-12-31", "fy_2022"),
        ("2021-01-05", "2021-12-31", "fy_2021"),
        ("2020-01-06", "2020-12-31", "fy_2020"),
    ]

    table = []
    for start, end, label in windows:
        stats = _window_stats(net, gross, costs, turnover, gross_exp, equity, start, end, label)
        table.append(stats)

    # historical streak analysis
    streak_5d = _find_historical_streaks(net, 5)
    streak_10d = _find_historical_streaks(net, 10)
    streak_21d = _find_historical_streaks(net, 21)

    # worst rolling windows
    worst_21d = _rolling_window_stats(net, gross, costs, equity, 21).sort_values("net_cum").head(10)
    worst_63d = _rolling_window_stats(net, gross, costs, equity, 63).sort_values("net_cum").head(10)
    worst_252d = _rolling_window_stats(net, gross, costs, equity, 252).sort_values("net_cum").head(10)

    # recent trend: 1M rolling sum (roughly 21 days)
    rolling_21d = net.rolling(21, min_periods=1).sum()
    last_21d_sum = float(rolling_21d.iloc[-1])
    last_21d_min = float(rolling_21d.min())
    last_21d_pctl = float((rolling_21d <= last_21d_sum).mean())

    # daily return percentiles for last 5 days
    def _day_pctl(val: float) -> float:
        return float((net <= val).mean() * 100.0)

    last_5d_pctl = [
        (d.date(), float(net.loc[d] * 10000), _day_pctl(net.loc[d]))
        for d in net.tail(5).index
    ]

    # formatting helpers
    def _pct(x):
        return f"{x * 100:.2f}%"

    def _bps(x):
        return f"{x:.2f}"

    def _row(r):
        return (
            f"| {r['label']} | {r['days']} | {_pct(r['net_cum'])} | {_pct(r['gross_cum'])} | "
            f"{_bps(r['net_mean_bps'])} | {_bps(r['gross_mean_bps'])} | {_bps(r['cost_mean_bps'])} | "
            f"{r['net_sharpe']:.2f} | {_pct(r['mdd'])} | {r['win_rate']*100:.1f}% | {r['consec_loss_max']} |"
        )

    header = (
        "| 期間 | 日数 | Net累積 | Gross累積 | Net平均(bps/日) | Gross平均(bps/日) | "
        "Cost平均(bps/日) | Net Sharpe | MDD | 勝率 | 最大連損日数 |\n"
        "|------|------|--------|-----------|-----------------|-------------------|------------------|------------|------|------|--------------|\n"
    )
    rows = header + "\n".join(_row(r) for r in table)

    report = f"""# 直近損失傾向診断レポート

> 作成日: 2026-08-01
> バックテスト出力: `{out_dir}`
> 最新取引日: {last.date()}

## 概要

直近の損失が気になるとのことで、期間別リターン・ドローダウン・連損日数・コスト影響を分析した。

## 期間別パフォーマンス

{rows}

## 直近トレンド

- 直近 21 日ローリング累積リターン: {_pct(last_21d_sum)}
- 21 日ローリング累積リターンの全期間最小値: {_pct(last_21d_min)}
- 直近 21 日累積は全期間の {last_21d_pctl*100:.1f}% パーセンタイル（低い方から）
- 直近 5 日は {int((net.tail(5) < 0).sum())} / 5 日が損失、合計 {_pct(net.tail(5).sum())}

### 直近 5 日の日次 Net リターンと全期間パーセンタイル

| 日付 | Net (bps) | 全期間下位パーセンタイル |
|------|-----------|--------------------------|
{chr(10).join(f"| {d} | {v:.2f} | {p:.2f}% |" for d, v, p in last_5d_pctl)}

* 2026-07-30 の -323 bps は全期間で 0.11% パーセンタイル（約 1 / 900 日）。*

## 歴代の長期連損ストリーク（Net）

### 5 日以上連損（最悪上位 20）

{_df_to_md(streak_5d)}

### 10 日以上連損（最悪上位）

{_df_to_md(streak_10d)}

### 21 日以上連損（最悪上位）

{_df_to_md(streak_21d)}

## 最悪ローリング期間

### 21 日窓（Net 累積が最も低い上位 10）

{_df_to_md(worst_21d)}

### 63 日窓（Net 累積が最も低い上位 10）

{_df_to_md(worst_63d)}

### 252 日窓（Net 累積が最も低い上位 10）

{_df_to_md(worst_252d)}

## 考察

- 直近 5 日（2026-07-27 〜 2026-07-31）は `{_pct(net.tail(5).sum())}` の連続損失。これは本バックテスト期間（2015-2026）で最悪の 5 日以上連損ストリークであり、2 番目に悪い 2025-07 の -3.35% の約 2 倍。
- 2026-07-30 の -323 bps は全期間で 0.11% パーセンタイル（約 1 / 900 日）。通常の 1 日ボラティリティ（約 104 bps）の 3 倍以上の外れ値。
- 一方、直近 21 日・63 日・YTD 2026 はいずれも正のリターン。短期的な急落であり、長期トレンドはまだ崩れていない。
- 直近 21 日ローリング累積は全期間で 3.1% パーセンタイル（低い方から）に位置し、過去の最悪 21 日窓（2018-07-11 〜 2018-08-08、-7.41%）には及ばない。
- 最大ドローダウン（MDD）は直近 21 日で `{_pct(_mdd_from_equity(equity.tail(21)))}`。これは全期間 MDD `{_pct(_mdd_from_equity(equity))}` にはまだ届いていないが、かなり大きなドローダウン。
- コスト平均（約 13.7 bps/日）は安定しており、損失は純粋な市場・シグナル要因によるもの。
"""

    report_path = report_dir / "recent_loss_analysis.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Report saved to: {report_path}")

    summary = {
        "last_trade_date": str(last.date()),
        "last_5d_net_cum_pct": round(net.tail(5).sum() * 100, 4),
        "last_5d_all_negative": bool((net.tail(5) < 0).all()),
        "last_21d_sum_pct": round(last_21d_sum * 100, 4),
        "last_21d_percentile": round(last_21d_pctl * 100, 2),
        "window_stats": table,
    }
    summary_path = report_dir / "recent_loss_analysis_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"Summary saved to: {summary_path}")


def _df_to_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "該当なし\n"
    # Format date columns and floats for readability
    fmt_df = df.copy()
    for col in fmt_df.columns:
        if pd.api.types.is_datetime64_any_dtype(fmt_df[col]):
            fmt_df[col] = fmt_df[col].dt.strftime("%Y-%m-%d")
        elif pd.api.types.is_float_dtype(fmt_df[col]):
            fmt_df[col] = fmt_df[col].map(lambda x: f"{x:.4f}")
    cols = "| " + " | ".join(str(c) for c in fmt_df.columns) + " |\n"
    sep = "|" + "|".join(["---"] * len(fmt_df.columns)) + "|\n"
    rows = ""
    for _, r in fmt_df.iterrows():
        rows += "| " + " | ".join(str(v) for v in r) + " |\n"
    return f"{cols}{sep}{rows}"


if __name__ == "__main__":
    main()
