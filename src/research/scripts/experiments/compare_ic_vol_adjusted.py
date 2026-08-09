#!/usr/bin/env python3
"""Compare cross-sectional IC and portfolio-level R^2 for vol_adjusted_target false vs true.

Loads per-date mu_gap matrices and df_exec realized returns, computes:
- daily cross-sectional Pearson IC (mu_gap vs realized)
- daily rank IC
- portfolio-level realized return vs predicted mean (p_mean) correlation
- per-year averages
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.cache import load_df_exec_from_local_cache
from leadlag.data.tickers import JP_TICKERS


def _ic(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 4:
        return np.nan
    return float(stats.pearsonr(x[mask], y[mask])[0])

def _rank_ic(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 4:
        return np.nan
    return float(stats.spearmanr(x[mask], y[mask])[0])

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gap-false", required=True, type=Path)
    p.add_argument("--gap-true", required=True, type=Path)
    p.add_argument("--start", default="2020-01-06")
    p.add_argument("--output-dir", default=str(ROOT / "outputs" / "experiments" / "vol_adjusted_walkforward"))
    args = p.parse_args()

    df_exec = load_df_exec_from_local_cache()
    start_dt = pd.to_datetime(args.start)
    dates = df_exec.index[df_exec.index >= start_dt]

    # Use jp_oc as realized target (same as backtest for pre-2026 where 5m data missing)
    realized = df_exec[[f"jp_oc_{tk}" for tk in JP_TICKERS]].values
    realized_df = pd.DataFrame(realized, index=df_exec.index, columns=JP_TICKERS)

    rows = []
    for dt in dates:
        dt_str = dt.strftime("%Y%m%d")
        false_file = args.gap_false / "matrices" / f"mu_gap_{dt_str}.npy"
        true_file = args.gap_true / "matrices" / f"mu_gap_{dt_str}.npy"
        if not false_file.exists() or not true_file.exists():
            continue
        mu_f = np.load(false_file)
        mu_t = np.load(true_file)
        r = realized_df.loc[dt].to_numpy()
        if np.isfinite(r).sum() < 4:
            continue
        rows.append({
            "date": dt,
            "pearson_false": _ic(mu_f, r),
            "pearson_true": _ic(mu_t, r),
            "rank_false": _rank_ic(mu_f, r),
            "rank_true": _rank_ic(mu_t, r),
            "mean_mu_false": float(np.nanmean(mu_f)),
            "mean_mu_true": float(np.nanmean(mu_t)),
            "std_mu_false": float(np.nanstd(mu_f, ddof=1)),
            "std_mu_true": float(np.nanstd(mu_t, ddof=1)),
        })

    df = pd.DataFrame(rows).set_index("date")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "daily_ic_comparison.csv")

    # Overall stats
    print("=== Overall IC (full period) ===")
    print(f"{'metric':<25} {'false':>12} {'true':>12} {'diff':>12}")
    print("-" * 65)
    for col in ["pearson_false", "pearson_true", "rank_false", "rank_true"]:
        print(f"{col:<25} {df[col].mean():>12.4f} {df[col].std(ddof=1):>12.4f}")

    print("\n=== Paired comparison ===")
    df["pearson_diff"] = df["pearson_false"] - df["pearson_true"]
    df["rank_diff"] = df["rank_false"] - df["rank_true"]
    for col in ["pearson", "rank"]:
        d = df[f"{col}_diff"].dropna()
        t, pv = stats.ttest_1samp(d, 0.0)
        print(f"{col} IC: mean diff={d.mean():.5f}, t={t:.3f}, p={pv:.4f}, wins={np.sum(d>0)}/{len(d)}")

    # Per-year
    print("\n=== Yearly mean Pearson IC ===")
    yearly = df.groupby(df.index.year)[["pearson_false", "pearson_true", "pearson_diff"]].mean()
    print(yearly.to_string())

    return 0


if __name__ == "__main__":
    sys.exit(main())
