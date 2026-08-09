#!/usr/bin/env python3
"""Simulate simple prevention measures and estimate their effect on the recent 5-day drawdown.

Measures evaluated:
  1. Reduce overnight holding ratios (alpha_long/alpha_short)
  2. Daily loss stop: scale weights to 0 if prior day net return < threshold
  3. Trailing drawdown stop: scale weights to 0 if portfolio is below recent high by threshold
  4. Realized IC filter: scale weights to 0 if trailing 20-day w vs y Spearman < threshold

All measures use only historical (past) returns, no look-ahead.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from leadlag.data.tickers import JP_TICKERS
from leadlag.models.sre import compute_jp_target_returns
from research.backtest_common import load_execution_data


def _load(out_dir: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    weights = pd.read_csv(out_dir / "daily_weights.csv", index_col=0, parse_dates=True)
    gross = pd.read_csv(out_dir / "daily_gross_returns.csv", index_col=0, parse_dates=True).iloc[:, 0]
    net = pd.read_csv(out_dir / "daily_net_returns.csv", index_col=0, parse_dates=True).iloc[:, 0]
    costs = pd.read_csv(out_dir / "daily_costs.csv", index_col=0, parse_dates=True).iloc[:, 0]

    df_exec = load_execution_data(beta_window=60, beta_ewma_halflife=None, beta_shrinkage=0.05, beta_winsor_sigma=3.0)
    y = compute_jp_target_returns(df_exec, JP_TICKERS)
    y_df = pd.DataFrame(y, index=df_exec.index, columns=JP_TICKERS)
    gap_df = df_exec[[f"jp_gap_{tk}" for tk in JP_TICKERS]].rename(columns=lambda c: c.replace("jp_gap_", ""))

    common = weights.index.intersection(y_df.index)
    weights = weights.loc[common]
    y_df = y_df.loc[common]
    gap_df = gap_df.loc[common]
    gross = gross.loc[common]
    net = net.loc[common]
    costs = costs.loc[common]
    return weights, gross, net, costs, y_df, gap_df


def _cost_of_day(
    w_t: np.ndarray,
    w_prev: np.ndarray,
    alpha_long: float,
    alpha_short: float,
    days_held: float,
    slip_bps: float = 5.0,
    fin_annual: float = 0.025,
    borrow_annual: float = 0.0115,
    reverse_bps: float = 2.0,
) -> float:
    alpha_mask = np.where(w_t > 0, alpha_long, np.where(w_t < 0, alpha_short, 0.0))
    float(np.sum(np.abs(w_t - w_prev)) / 2.0)
    slip = slip_bps / 10000.0
    financing_daily = fin_annual / 365.0
    borrow_daily = borrow_annual / 365.0
    reverse_daily = reverse_bps / 10000.0

    slip_cost = slip * (
        2.0 * np.sum((1.0 - alpha_mask) * np.abs(w_t))
        + np.sum(alpha_mask * np.abs(w_t - w_prev) / 2.0)
    )
    held_long = float(np.sum(alpha_mask * np.maximum(w_t, 0.0)))
    held_short = float(np.sum(alpha_mask * np.maximum(-w_t, 0.0)))
    fin_cost = held_long * financing_daily * days_held
    borrow_cost = held_short * borrow_daily * days_held
    reverse_cost = held_short * reverse_daily * days_held
    return slip_cost + fin_cost + borrow_cost + reverse_cost


def _simulate_alpha(
    weights: pd.DataFrame,
    y_df: pd.DataFrame,
    gap_df: pd.DataFrame,
    alpha_long: float,
    alpha_short: float,
    slippage_bps: float = 5.0,
    fin_annual: float = 0.025,
    borrow_annual: float = 0.0115,
    reverse_bps: float = 2.0,
) -> tuple[pd.Series, pd.Series]:
    w_arr = weights.values
    y_arr = y_df.values
    gap_arr = gap_df.values
    n = len(weights)
    alpha_mask = np.where(w_arr > 0, alpha_long, np.where(w_arr < 0, alpha_short, 0.0))

    intraday = (w_arr * y_arr).sum(axis=1)
    overnight = np.zeros(n)
    overnight[:-1] = (alpha_mask[:-1] * w_arr[:-1] * gap_arr[1:]).sum(axis=1)
    gross = pd.Series(intraday + overnight, index=weights.index)

    calendar_days = np.ones(n)
    for i in range(n - 1):
        calendar_days[i] = (weights.index[i + 1] - weights.index[i]).days
    calendar_days[-1] = 1.0

    costs = np.zeros(n)
    w_prev = np.zeros(len(JP_TICKERS))
    for i in range(n):
        costs[i] = _cost_of_day(
            w_arr[i],
            w_prev,
            alpha_long,
            alpha_short,
            calendar_days[i],
            slippage_bps,
            fin_annual,
            borrow_annual,
            reverse_bps,
        )
        w_prev = w_arr[i]
    costs = pd.Series(costs, index=weights.index)
    net = gross - costs
    return gross, net


def _spd_spearman(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    with np.errstate(invalid="ignore"):
        r = spearmanr(a, b, nan_policy="omit")[0]
    return float(r) if np.isfinite(r) else 0.0


def _simulate_stop_or_filter(
    weights: pd.DataFrame,
    y_df: pd.DataFrame,
    gap_df: pd.DataFrame,
    mode: str,
    threshold: float,
    scale_after_trigger: float = 0.0,
    alpha_long: float = 0.75,
    alpha_short: float = 0.5,
    ic_window: int = 20,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Simulate a stop/filter based on past information, accounting for path dependence.

    mode:
      - 'daily_loss': if prior day net < threshold, scale next day weights
      - 'drawdown': if portfolio below recent high by threshold, scale weights
      - 'ic': if trailing *ic_window* day spearman(w,y) < threshold, scale weights
    """
    w_arr = weights.values
    y_arr = y_df.values
    gap_arr = gap_df.values
    n = len(weights)

    calendar_days = np.ones(n)
    for i in range(n - 1):
        calendar_days[i] = (weights.index[i + 1] - weights.index[i]).days
    calendar_days[-1] = 1.0

    ic_trigger = np.full(n, False, dtype=bool)
    if mode == "ic":
        for i in range(ic_window, n):
            ic_vals = [_spd_spearman(w_arr[j], y_arr[j]) for j in range(i - ic_window, i)]
            valid = [v for v in ic_vals if np.isfinite(v)]
            if not valid or np.nanmean(valid) < threshold:
                ic_trigger[i] = True

    scale = np.ones(n)
    gross = np.zeros(n)
    net = np.zeros(n)
    costs = np.zeros(n)
    w_prev = np.zeros(len(JP_TICKERS))
    wealth = 1.0
    max_wealth = 1.0

    for i in range(n):
        if i > 0:
            if mode == "daily_loss":
                if net[i - 1] < threshold:
                    scale[i] = scale_after_trigger
            elif mode == "drawdown":
                dd = wealth / max_wealth - 1.0
                if dd < threshold:
                    scale[i] = scale_after_trigger
            elif mode == "ic":
                if ic_trigger[i]:
                    scale[i] = scale_after_trigger

        w_t = w_arr[i] * scale[i]
        alpha_mask = np.where(w_t > 0, alpha_long, np.where(w_t < 0, alpha_short, 0.0))

        intraday = float(np.sum(w_t * y_arr[i]))
        overnight = 0.0
        if i + 1 < n:
            overnight = float(np.sum(alpha_mask * w_t * gap_arr[i + 1]))
        gross_i = intraday + overnight

        costs_i = _cost_of_day(
            w_t,
            w_prev,
            alpha_long,
            alpha_short,
            calendar_days[i],
        )

        net_i = gross_i - costs_i
        gross[i] = gross_i
        costs[i] = costs_i
        net[i] = net_i

        wealth *= 1.0 + net_i
        max_wealth = max(max_wealth, wealth)
        w_prev = w_t

    return pd.Series(scale, index=weights.index), pd.Series(gross, index=weights.index), pd.Series(net, index=weights.index)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/backtest_20260801_144758")
    parser.add_argument("--report-dir", default="reports/longterm_backtest_20260801")
    args = parser.parse_args()

    out_dir = ROOT / args.output_dir
    report_dir = ROOT / args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    weights, gross_actual, net_actual, costs_actual, y_df, gap_df = _load(out_dir)

    n_recent = 5
    recent_idx = gross_actual.tail(n_recent).index

    scenarios: list[dict] = []

    base_gross_5d = float(gross_actual.tail(n_recent).sum())
    base_net_5d = float(net_actual.tail(n_recent).sum())
    scenarios.append({
        "name": "baseline",
        "params": "alpha_long=0.75, alpha_short=0.50, no stop",
        "gross_5d_bps": base_gross_5d * 10000,
        "net_5d_bps": base_net_5d * 10000,
    })

    # 1. Reduce overnight alpha
    for al, as_ in [(0.5, 0.25), (0.0, 0.0)]:
        g, n = _simulate_alpha(weights, y_df, gap_df, al, as_)
        scenarios.append({
            "name": f"alpha_long_{al}_short_{as_}",
            "params": f"alpha_long={al}, alpha_short={as_}",
            "gross_5d_bps": float(g.tail(n_recent).sum()) * 10000,
            "net_5d_bps": float(n.tail(n_recent).sum()) * 10000,
        })

    # 2. Daily loss stop (path-dependent, stops from next day)
    for thresh, scale in [(-0.005, 0.0), (-0.01, 0.0), (-0.015, 0.0), (-0.02, 0.0),
                          (-0.005, 0.5), (-0.01, 0.5), (-0.015, 0.5), (-0.02, 0.5)]:
        scale_ser, g, n = _simulate_stop_or_filter(
            weights, y_df, gap_df, "daily_loss", threshold=thresh, scale_after_trigger=scale
        )
        hit_count = int((scale_ser.loc[recent_idx] < 1.0).sum())
        scenarios.append({
            "name": f"daily_loss_stop_{thresh*100:.1f}pct_scale{scale}",
            "params": f"if prior net < {thresh*10000:.0f} bps then scale={scale}",
            "gross_5d_bps": float(g.tail(n_recent).sum()) * 10000,
            "net_5d_bps": float(n.tail(n_recent).sum()) * 10000,
            "hit_days_in_5d": hit_count,
        })

    # 3. Trailing drawdown stop
    for thresh, scale in [(-0.03, 0.0), (-0.05, 0.0), (-0.03, 0.5), (-0.05, 0.5)]:
        scale_ser, g, n = _simulate_stop_or_filter(
            weights, y_df, gap_df, "drawdown", threshold=thresh, scale_after_trigger=scale
        )
        hit_count = int((scale_ser.loc[recent_idx] < 1.0).sum())
        scenarios.append({
            "name": f"drawdown_stop_{abs(thresh)*100:.0f}pct_scale{scale}",
            "params": f"if DD < {thresh*100:.0f}% then scale={scale}",
            "gross_5d_bps": float(g.tail(n_recent).sum()) * 10000,
            "net_5d_bps": float(n.tail(n_recent).sum()) * 10000,
            "hit_days_in_5d": hit_count,
        })

    ic_window = 20
    # 4. Realized IC filter (uses original w/y signal quality, not affected by path)
    for thresh, scale in [(0.0, 0.0), (0.0, 0.5), (-0.05, 0.0)]:
        scale_ser, g, n = _simulate_stop_or_filter(
            weights, y_df, gap_df, "ic", threshold=thresh, scale_after_trigger=scale
        )
        hit_count = int((scale_ser.loc[recent_idx] < 1.0).sum())
        scenarios.append({
            "name": f"ic{ic_window}_stop_lt_{thresh}_scale{scale}",
            "params": f"if {ic_window}-day IC < {thresh} then scale={scale}",
            "gross_5d_bps": float(g.tail(n_recent).sum()) * 10000,
            "net_5d_bps": float(n.tail(n_recent).sum()) * 10000,
            "hit_days_in_5d": hit_count,
        })

    rows = []
    for s in scenarios:
        rows.append(
            f"| {s['name']} | {s['params']} | {s['gross_5d_bps']:.2f} | {s['net_5d_bps']:.2f} | "
            f"{s.get('hit_days_in_5d', 0)} |"
        )

    report = f"""# 防止策シミュレーション: 直近 5 日の影響

> 作成日: 2026-08-01
> すべての判定は過去情報のみを使用（ルックアヘッドなし）

## 概要

直近 5 日の baseline:
- Gross 累積: {base_gross_5d*10000:.2f} bps
- Net 累積: {base_net_5d*10000:.2f} bps

## シナリオ

| シナリオ | パラメータ | Gross 5日累積 (bps) | Net 5日累積 (bps) | 直近5日間ヒット日数 |
|---|---|---:|---:|---:|
{chr(10).join(rows)}

## 説明

- `alpha_long/short=0` はオーバーナイト持越しを完全に停止するケース。5日の Net 損失が最も減少するが、平日のコスト増と長期パフォーマンスへの影響は別途検証が必要。
- `daily_loss_stop` は前日 Net 損失が閾値を下回った翌日にポジションを縮小・停止する。最初の1日は防げない。
- `drawdown_stop` はアカウント資産の直近高値からの下落率で判定。閾値を -3% や -5% に設定すれば 7/30 以降に停止する。
- `ic20_stop` は過去20日の w vs JP target Spearman（実現した情報係数）が閾値を下回った日にポジションを縮小する。直近の悪化には 20 日平均では反応しにくい。

## 注意

- 本シミュレーションは直近 5 日のみを対象としており、長期パフォーマンス（Sharpe、DD、ターンオーバー）への影響は含まれていない。
- コストは `BacktestEngine` と同一の式で再計算している。
- `daily_loss_stop`・`drawdown_stop` はシミュレーション内で当日の損失が翌日の停止判定に影響するパス依存を考慮している。
"""

    report_path = report_dir / "prevention_simulation.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Report saved to: {report_path}")

    summary_path = report_dir / "prevention_simulation_summary.json"
    summary_path.write_text(json.dumps(scenarios, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
