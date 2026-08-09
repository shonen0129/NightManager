#!/usr/bin/env python3
"""Compute portfolio-level predicted mean (pred_mean_gap) vs realized return correlation."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))


def _corr(csv: Path) -> float:
    df = pd.read_csv(csv)
    # Realized gross return
    x = df["pred_mean_gap"].to_numpy()
    y = df["gross_return"].to_numpy()
    return float(stats.pearsonr(x, y)[0])


def main() -> int:
    false_csv = ROOT / "var/live/pipeline_data/gap_adjusted_distribution/20260731_024303/portfolio_gap_distribution_diagnostics.csv"
    true_csv = ROOT / "var/live/pipeline_data/gap_adjusted_distribution/20260731_025246/portfolio_gap_distribution_diagnostics.csv"
    print(f"false: pred_mean_gap vs gross_return corr = {_corr(false_csv):.4f}")
    print(f"true:  pred_mean_gap vs gross_return corr = {_corr(true_csv):.4f}")

    false_df = pd.read_csv(false_csv)
    true_df = pd.read_csv(true_csv)
    merged = pd.merge(false_df[["trade_date", "pred_mean_gap", "gross_return", "net_return"]],
                      true_df[["trade_date", "pred_mean_gap", "gross_return", "net_return"]],
                      on="trade_date", suffixes=("_false", "_true"))
    merged = merged.dropna()
    print(f"n days: {len(merged)}")
    print(f"gross_return corr (false vs true): {stats.pearsonr(merged['gross_return_false'], merged['gross_return_true'])[0]:.4f}")
    print(f"pred_mean_gap corr (false vs true): {stats.pearsonr(merged['pred_mean_gap_false'], merged['pred_mean_gap_true'])[0]:.4f}")

    # Paired t on predicted mean
    d = merged["pred_mean_gap_false"] - merged["pred_mean_gap_true"]
    t, p = stats.ttest_rel(merged["pred_mean_gap_false"], merged["pred_mean_gap_true"])
    print(f"pred_mean_gap paired: false={merged['pred_mean_gap_false'].mean():.5f}, true={merged['pred_mean_gap_true'].mean():.5f}, diff={d.mean():.5f}, t={t:.3f}, p={p:.4f}")

    # Paired t on realized net
    d = merged["net_return_false"] - merged["net_return_true"]
    t, p = stats.ttest_rel(merged["net_return_false"], merged["net_return_true"])
    print(f"net_return paired: false={merged['net_return_false'].mean():.5f}, true={merged['net_return_true'].mean():.5f}, diff={d.mean():.5f}, t={t:.3f}, p={p:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
