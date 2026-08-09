#!/usr/bin/env python3
"""Compare IC during 5m intraday period (2026-03-03 to 2026-06-01) where target is 9:10-to-close
vs pre-5m period where target falls back to jp_oc."""
from __future__ import annotations

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
from leadlag.data.preprocessor import compute_jp_target_returns
from leadlag.data.tickers import JP_TICKERS


def _ic(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 4:
        return np.nan
    return float(stats.pearsonr(x[mask], y[mask])[0])


def main() -> int:
    gap_f = ROOT / "var/live/pipeline_data/gap_adjusted_distribution/20260731_024303"
    gap_t = ROOT / "var/live/pipeline_data/gap_adjusted_distribution/20260731_025246"
    df_exec = load_df_exec_from_local_cache()
    y_target = compute_jp_target_returns(df_exec, JP_TICKERS)
    y_df = pd.DataFrame(y_target, index=df_exec.index, columns=JP_TICKERS)

    fivem_start = pd.Timestamp("2026-03-03")
    fivem_end = pd.Timestamp("2026-06-01")
    pre = df_exec.index[df_exec.index < fivem_start]
    fivem = df_exec.index[(df_exec.index >= fivem_start) & (df_exec.index <= fivem_end)]

    rows = []
    for label, dates in [("pre_5m", pre), ("5m", fivem)]:
        ic_f, ic_t = [], []
        for dt in dates:
            dt_str = dt.strftime("%Y%m%d")
            f = gap_f / "matrices" / f"mu_gap_{dt_str}.npy"
            t = gap_t / "matrices" / f"mu_gap_{dt_str}.npy"
            if not f.exists() or not t.exists():
                continue
            mu_f = np.load(f)
            mu_t = np.load(t)
            r = y_df.loc[dt].to_numpy()
            if np.isfinite(r).sum() < 4:
                continue
            ic_f.append(_ic(mu_f, r))
            ic_t.append(_ic(mu_t, r))
        ic_f = np.array(ic_f)
        ic_t = np.array(ic_t)
        rows.append({
            "period": label,
            "n_days": len(ic_f),
            "ic_false_mean": float(np.nanmean(ic_f)),
            "ic_true_mean": float(np.nanmean(ic_t)),
            "ic_false_std": float(np.nanstd(ic_f, ddof=1)),
            "ic_true_std": float(np.nanstd(ic_t, ddof=1)),
            "ic_diff": float(np.nanmean(ic_f - ic_t)),
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
