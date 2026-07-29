#!/usr/bin/env python
"""Generate a markdown report for the Phase 1 ML order decision overlay."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 245


def _annual_metrics(daily_returns: pd.Series) -> dict:
    dr = daily_returns.dropna()
    if len(dr) < 10:
        return {
            "n": len(dr),
            "ar": np.nan,
            "vol": np.nan,
            "sharpe": np.nan,
            "mdd": np.nan,
            "skew": np.nan,
            "kurt": np.nan,
        }
    ar = float(dr.mean() * TRADING_DAYS)
    vol = float(dr.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe = ar / vol if vol > 1e-12 else np.nan
    wealth = (1.0 + dr).cumprod()
    mdd = float(((wealth / wealth.cummax()) - 1.0).min())
    skew = float(dr.skew())
    kurt = float(dr.kurtosis())
    return {
        "n": len(dr),
        "ar": ar,
        "vol": vol,
        "sharpe": sharpe,
        "mdd": mdd,
        "skew": skew,
        "kurt": kurt,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default="reports/ml_order_decision/phase1_results",
        help="Directory with Phase 1 output CSVs",
    )
    parser.add_argument(
        "--output",
        default="reports/ml_order_decision/phase1_report.md",
        help="Output markdown report path",
    )
    parser.add_argument(
        "--test-end",
        default="2024-12-31",
        help="Crop report to dates <= this test-end",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load daily returns and turnover (saved Series have default column name 0)
    base_ret = pd.read_csv(
        input_dir / "baseline_daily_returns.csv",
        index_col=0,
        parse_dates=True,
    ).iloc[:, 0]
    over_ret = pd.read_csv(
        input_dir / "overlay_daily_returns.csv",
        index_col=0,
        parse_dates=True,
    ).iloc[:, 0]

    base_turn = pd.read_csv(
        input_dir / "baseline_daily_turnover.csv",
        index_col=0,
        parse_dates=True,
    ).iloc[:, 0]
    over_turn = pd.read_csv(
        input_dir / "overlay_daily_turnover.csv",
        index_col=0,
        parse_dates=True,
    ).iloc[:, 0]

    base_gross = pd.read_csv(
        input_dir / "baseline_daily_gross_exps.csv",
        index_col=0,
        parse_dates=True,
    ).iloc[:, 0]
    over_gross = pd.read_csv(
        input_dir / "overlay_daily_gross_exps.csv",
        index_col=0,
        parse_dates=True,
    ).iloc[:, 0]

    # Align and optionally crop to the intended test end
    common = base_ret.index.intersection(over_ret.index)
    test_end_dt = pd.to_datetime(args.test_end)
    common = common[common <= test_end_dt]
    base_ret = base_ret.loc[common]
    over_ret = over_ret.loc[common]
    base_turn = base_turn.loc[common]
    over_turn = over_turn.loc[common]
    base_gross = base_gross.loc[common]
    over_gross = over_gross.loc[common]

    diff = over_ret - base_ret
    t_stat, p_value = stats.ttest_rel(over_ret, base_ret)

    base_m = _annual_metrics(base_ret)
    over_m = _annual_metrics(over_ret)

    # Yearly breakdown
    years = sorted({d.year for d in common})
    year_rows = []
    for y in years:
        mask = common.year == y
        if mask.sum() < 10:
            continue
        base_y = _annual_metrics(base_ret.loc[common[mask]])
        over_y = _annual_metrics(over_ret.loc[common[mask]])
        year_rows.append(
            {
                "year": y,
                "base_sharpe": base_y["sharpe"],
                "base_ar": base_y["ar"],
                "base_mdd": base_y["mdd"],
                "over_sharpe": over_y["sharpe"],
                "over_ar": over_y["ar"],
                "over_mdd": over_y["mdd"],
            }
        )
    yearly = pd.DataFrame(year_rows)

    # Coefficients / feature importance
    ridge_path = input_dir / "ridge_coefficients.csv"
    lgbm_path = input_dir / "lgbm_feature_importance.csv"
    if lgbm_path.exists():
        coef_df = pd.read_csv(lgbm_path)
        coef_col = "importance"
        model_header = "# Phase 2 Report: ML Order Decision Overlay (Per-Ticker Gap LightGBM)"
        table_header = "## 4. Top LightGBM feature importance"
        col_header = "| Feature | Importance |"
    else:
        coef_df = pd.read_csv(ridge_path)
        coef_col = "coef"
        model_header = "# Phase 1 Report: ML Order Decision Overlay (Per-Ticker Gap Ridge)"
        table_header = "## 4. Top Ridge coefficients"
        col_header = "| Feature | Coefficient |"

    # Build report
    lines = [
        model_header,
        "",
        f"**Test period:** {common[0].date()} → {common[-1].date()}",
        f"**Trading days:** {len(common)}",
        f"**Output directory:** `{input_dir}`",
        "",
        "## 1. Overall performance (net of costs)",
        "",
        "| Metric | Baseline | Overlay | Diff |",
        "|--------|----------|---------|------|",
    ]

    metrics_table = [
        ("Sharpe", f"{base_m['sharpe']:.3f}", f"{over_m['sharpe']:.3f}", f"{over_m['sharpe'] - base_m['sharpe']:+.3f}"),
        ("Annual Return", f"{base_m['ar']:.3f}", f"{over_m['ar']:.3f}", f"{over_m['ar'] - base_m['ar']:+.3f}"),
        ("Volatility", f"{base_m['vol']:.3f}", f"{over_m['vol']:.3f}", f"{over_m['vol'] - base_m['vol']:+.3f}"),
        ("Max DD", f"{base_m['mdd']:.3f}", f"{over_m['mdd']:.3f}", f"{over_m['mdd'] - base_m['mdd']:+.3f}"),
        ("Skewness", f"{base_m['skew']:.3f}", f"{over_m['skew']:.3f}", f"{over_m['skew'] - base_m['skew']:+.3f}"),
        ("Excess Kurt", f"{base_m['kurt']:.3f}", f"{over_m['kurt']:.3f}", f"{over_m['kurt'] - base_m['kurt']:+.3f}"),
        ("Mean Turnover", f"{base_turn.mean():.3f}", f"{over_turn.mean():.3f}", f"{over_turn.mean() - base_turn.mean():+.3f}"),
        ("Mean Gross Exp", f"{base_gross.mean():.3f}", f"{over_gross.mean():.3f}", f"{over_gross.mean() - base_gross.mean():+.3f}"),
    ]
    for name, b, o, d in metrics_table:
        lines.append(f"| {name} | {b} | {o} | {d} |")

    lines += [
        "",
        "## 2. Statistical test",
        "",
        f"- Mean daily return difference (overlay - baseline): **{diff.mean():.6f}**",
        f"- Paired t-statistic: **{t_stat:.3f}**",
        f"- p-value: **{p_value:.4f}**",
        "",
        "## 3. Yearly breakdown",
        "",
    ]

    if not yearly.empty:
        lines.append("| Year | Base Sharpe | Overlay Sharpe | Base AR | Overlay AR | Base MDD | Overlay MDD |")
        lines.append("|------|-------------|----------------|---------|------------|----------|-------------|")
        for _, row in yearly.iterrows():
            lines.append(
                f"| {int(row['year'])} | {row['base_sharpe']:.3f} | {row['over_sharpe']:.3f} | "
                f"{row['base_ar']:.3f} | {row['over_ar']:.3f} | {row['base_mdd']:.3f} | {row['over_mdd']:.3f} |"
            )
    else:
        lines.append("No yearly breakdown available.")

    lines += [
        "",
        table_header,
        "",
        col_header,
        "|---------|-------------|",
    ]
    for _, row in coef_df.head(15).iterrows():
        lines.append(f"| {row['feature']} | {row[coef_col]:.6f} |")

    lines += [
        "",
        "## 5. Notes and limitations",
        "",
        "- The overlay applies ``p_trade = sigmoid(contribution_hat / target_std)`` to rescale the raw ``mu_gap / sigma_gap`` scores before V2 weight construction.",
        "- RuleD multiplier is taken from the baseline V2 run (PIT history is not recomputed for the overlay because the available gap distribution output lacks a diagnostics CSV; multiplier is 1.0 in this run).",
        "- Cost, financing, borrow, and reverse-fee calculations are exactly those used by ``BacktestEngine.run_v2_backtest`` because the overlay is injected by monkey-patching ``generate_v2_production_portfolio``.",
        "- Training target is ``side * realized_9:10_close - round_trip_cost`` per ticker, where ``realized`` comes from ``compute_jp_target_returns``.",
        "",
    ]

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report written to {output_path}")

    print("--- Baseline metrics ---")
    for k, v in base_m.items():
        print(f"{k}: {v}")
    print("--- Overlay metrics ---")
    for k, v in over_m.items():
        print(f"{k}: {v}")
    print(f"Turnover: base={base_turn.mean():.4f}, overlay={over_turn.mean():.4f}")
    print(f"Gross:    base={base_gross.mean():.4f}, overlay={over_gross.mean():.4f}")
    print(f"Paired t-test: t={t_stat:.3f}, p={p_value:.4f}")


if __name__ == "__main__":
    main()
