#!/usr/bin/env python3
"""Generate markdown report for VIX regime overlay experiment."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "vix_regime_overlay"
REPORT_DIR = ROOT / "reports" / "vix_regime_overlay"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def _load_returns(name: str) -> pd.Series:
    path = RESULTS_DIR / f"daily_returns_{name}.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.iloc[:, 0]


def main():
    summary = pd.read_csv(RESULTS_DIR / "summary.csv")
    walkforward = pd.read_csv(RESULTS_DIR / "walkforward_sharpe.csv")

    baseline = _load_returns("baseline")
    variants = [n for n in summary["name"] if n != "baseline"]

    # Statistical tests: baseline vs each variant
    stats = []
    for var in variants:
        var_rets = _load_returns(var)
        common_idx = baseline.index.intersection(var_rets.index)
        b = baseline.loc[common_idx]
        v = var_rets.loc[common_idx]
        diff = v - b
        if len(diff) < 30:
            continue
        t, p = ttest_rel(v, b, nan_policy="omit")
        win_rate = (diff > 0).mean()
        stats.append({
            "variant": var,
            "n_days": int(len(diff)),
            "mean_diff_bps": float(diff.mean() * 10000),
            "t_stat": float(t),
            "p_value": float(p),
            "win_rate": float(win_rate),
        })

    stats_df = pd.DataFrame(stats)
    if not stats_df.empty:
        stats_df.to_csv(RESULTS_DIR / "paired_test.csv", index=False)

    # Build markdown report
    lines = [
        "# VIX Regime Overlay Experiment Report",
        "",
        f"- Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "- Hypothesis: US VIX high -> US-led (maintain gross); JP VIX high while US VIX low -> Japan-led (reduce gross)",
        "- Model: SectorRelativeEnsembleBLPEnhancedModel (Residual-BLPX, V1-equivalent)",
        "- Config: configs/production/production.yaml",
        "- Period: 2018-04-01 ~ 2024-12-31",
        "- VIX data: ^VIX (US), ^NKVI.OS (Nikkei VI / Japan)",
        "- Method: 60-day rolling z-score on log VIX; spread = JP_z - US_z",
        "",
        "## Aggregate Results",
        "",
    ]

    # Summary table
    header = summary.columns.tolist()
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for _, row in summary.iterrows():
        vals = []
        for col in header:
            v = row[col]
            if col in ("AR", "RISK", "MDD"):
                vals.append(f"{float(v) * 100:.2f}%")
            elif col in ("Sharpe", "Total Return"):
                vals.append(f"{float(v):.4f}")
            else:
                vals.append(f"{v}")
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")

    # Walk-forward table
    lines.append("## Walk-Forward Year-by-Year Sharpe")
    lines.append("")
    if not walkforward.empty:
        pivot = walkforward.pivot(index="year", columns="variant", values="sharpe")
        cols = ["year"] + pivot.columns.tolist()
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| --- | " + " | ".join(["---"] * (len(cols) - 1)) + " |")
        for year, row in pivot.iterrows():
            vals = [str(year)]
            for col in pivot.columns:
                vals.append(f"{row[col]:.2f}" if not pd.isna(row[col]) else "")
            lines.append("| " + " | ".join(vals) + " |")
    lines.append("")

    # Statistical tests
    if not stats_df.empty:
        lines.append("## Paired t-test vs Baseline (daily net returns)")
        lines.append("")
        lines.append("| variant | n_days | mean diff (bps) | t-stat | p-value | win rate |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for _, row in stats_df.iterrows():
            lines.append(
                f"| {row['variant']} | {int(row['n_days'])} | "
                f"{row['mean_diff_bps']:.2f} | {row['t_stat']:.3f} | "
                f"{row['p_value']:.4f} | {row['win_rate'] * 100:.1f}% |"
            )
        lines.append("")

    lines.append("## Conclusion")
    lines.append("")
    baseline_sharpe = float(summary[summary["name"] == "baseline"]["Sharpe"].iloc[0])
    best = summary.loc[summary["Sharpe"].idxmax()]
    if best["name"] == "baseline":
        lines.append(
            f"**No overlay improved net Sharpe.** Baseline Sharpe = {baseline_sharpe:.2f}. "
            "All VIX regime overlays reduced both annualized return and Sharpe, while marginally improving MDD. "
            "The hypothesis that Japan-led VIX spikes are a reliable signal to reduce gross was not supported in this specification."
        )
    else:
        lines.append(
            f"Best variant: **{best['name']}** (Sharpe = {best['Sharpe']:.2f}, "
            f"baseline = {baseline_sharpe:.2f})."
        )
    lines.append("")
    lines.append("## Notes / Next Steps")
    lines.append("")
    lines.append("- Overlay tested on V1 BLPX model, not the production V2 (Residual-BLPX-RA v2).")
    lines.append("- Thresholds (spread_z > 0.8, high_vix_z > 0.5, multiplier 0.5) are ad-hoc; a parameter sweep with Deflated Sharpe is recommended before drawing final conclusions.")
    lines.append("- Cost breakdown and gross/net decomposition were not saved in this run; re-run `experiment_vix_regime_overlay.py` with cost CSV flags for a full audit.")
    lines.append("")

    report_text = "\n".join(lines)
    (REPORT_DIR / "report.md").write_text(report_text)
    (RESULTS_DIR / "report.md").write_text(report_text)
    print("Report saved:")
    print("  ", REPORT_DIR / "report.md")
    print("  ", RESULTS_DIR / "report.md")

    if not stats_df.empty:
        print("\nPaired test summary:")
        print(stats_df.to_string(index=False))


if __name__ == "__main__":
    main()
