#!/usr/bin/env python
"""LightGBM overlay with VIX: train on 2020-2024, test on 2026-07 (H2 2026 to date)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.cache import load_df_exec_from_local_cache
from research.experiments.ml_order_decision.phase2 import run_phase2_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TRADING_DAYS = 245


def compute_metrics(daily_returns: pd.Series) -> dict:
    dr = daily_returns.dropna()
    if len(dr) < 10:
        return {"sharpe": np.nan, "ar": np.nan, "vol": np.nan, "mdd": np.nan, "n": len(dr)}
    ar = float(dr.mean() * TRADING_DAYS)
    vol = float(dr.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe = ar / vol if vol > 0 else np.nan
    wealth = (1.0 + dr).cumprod()
    mdd = float(((wealth / wealth.cummax()) - 1.0).min())
    return {"sharpe": sharpe, "ar": ar, "vol": vol, "mdd": mdd, "n": len(dr)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="reports/ml_order_decision/vix_overlay_2026h2")
    parser.add_argument("--gap-input-dir", required=True)
    parser.add_argument("--vix-cache", default=str(ROOT / "market_data" / "vix_regime_overlay" / "vix_cache.csv"))
    parser.add_argument("--train-start", default="2020-01-06")
    parser.add_argument("--train-end", default="2024-12-31")
    parser.add_argument("--test-start", default="2026-07-01")
    parser.add_argument("--test-end", default="2026-07-31")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import yaml
    with open(ROOT / "configs" / "production" / "production.yaml") as f:
        cfg = yaml.safe_load(f)

    df_exec = load_df_exec_from_local_cache()
    gap_input_dir = Path(args.gap_input_dir)
    vix_cache_path = Path(args.vix_cache)

    lgbm_kwargs = {
        "n_estimators": 100,
        "learning_rate": 0.05,
        "max_depth": 3,
        "num_leaves": 20,
        "min_child_samples": 300,
        "reg_alpha": 0.5,
        "reg_lambda": 1.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    results = {}
    for label, use_vix in [("no_vix", False), ("vix", True)]:
        variant_out = out_dir / label
        logger.info("=== Running %s on 2026-07 ===", label)
        result = run_phase2_experiment(
            df_exec=df_exec,
            gap_input_dir=gap_input_dir,
            cfg=cfg,
            output_dir=variant_out,
            train_start=args.train_start,
            train_end=args.train_end,
            test_start=args.test_start,
            test_end=args.test_end,
            n_jobs=-1,
            lgbm_kwargs=lgbm_kwargs,
            p_trade_ema_span=None,
            use_ticker=True,
            use_classification=False,
            per_ticker_interactions=True,
            vix_cache_path=vix_cache_path if use_vix else None,
        )
        results[label] = result

    base = results["no_vix"]["baseline_result"]["daily_returns"].dropna()
    no_vix = results["no_vix"]["overlay_result"]["daily_returns"].dropna()
    vix = results["vix"]["overlay_result"]["daily_returns"].dropna()

    common = base.index.intersection(no_vix.index).intersection(vix.index)
    base, no_vix, vix = base.loc[common], no_vix.loc[common], vix.loc[common]

    base_m = compute_metrics(base)
    no_vix_m = compute_metrics(no_vix)
    vix_m = compute_metrics(vix)

    _, p_no_vix = ttest_rel(no_vix, base)
    _, p_vix = ttest_rel(vix, base)
    _, p_vix_vs_no_vix = ttest_rel(vix, no_vix)

    # last 7 days
    def tail7(s):
        last = s.tail(7)
        cum = (1 + last).cumprod() - 1
        mdd = (cum - cum.cummax()).min()
        return float(last.sum()), float(mdd), last.index[0].strftime("%Y-%m-%d"), last.index[-1].strftime("%Y-%m-%d")

    base_t7 = tail7(base)
    no_vix_t7 = tail7(no_vix)
    vix_t7 = tail7(vix)

    lines = [
        "# LGBM VIX Overlay on 2026-07 (H2 2026 to date)",
        "",
        "**Config**: n_estimators=100, num_leaves=20, max_depth=3, reg_alpha=0.5, reg_lambda=1.0, per_ticker_interactions=True",
        f"- Train: {args.train_start} ~ {args.train_end}",
        f"- Test:  {args.test_start} ~ {args.test_end}",
        "- VIX features: 60-day log z-score (US), 60-day log z-score (JP), JP-US spread z-score",
        "",
        "## 2026-07 Pooled Performance",
        f"- Total OOS days: {vix_m['n']}",
        f"- Baseline V2 Sharpe: {base_m['sharpe']:.4f}",
        f"- No-VIX Overlay Sharpe: {no_vix_m['sharpe']:.4f} (p={p_no_vix:.4f})",
        f"- VIX Overlay Sharpe: {vix_m['sharpe']:.4f} (p={p_vix:.4f})",
        f"- VIX vs No-VIX: ΔSharpe={vix_m['sharpe']-no_vix_m['sharpe']:+.4f}, p={p_vix_vs_no_vix:.4f}",
        f"- Mean daily diff (VIX vs no-VIX): {(vix - no_vix).mean():.6f}",
        "",
        "## Last 7 Trading Days",
        f"- Period: {vix_t7[2]} ~ {vix_t7[3]}",
        "",
        "| Model | 7-day total | 7-day MDD |",
        "|-------|-------------|-----------|",
        f"| baseline | {base_t7[0]*100:.4f}% | {base_t7[1]*100:.4f}% |",
        f"| no_vix | {no_vix_t7[0]*100:.4f}% | {no_vix_t7[1]*100:.4f}% |",
        f"| vix | {vix_t7[0]*100:.4f}% | {vix_t7[1]*100:.4f}% |",
        "",
        f"- VIX vs no-VIX 7-day total diff: {(vix_t7[0]-no_vix_t7[0])*100:.4f}%",
        f"- VIX vs no-VIX 7-day MDD diff: {(vix_t7[1]-no_vix_t7[1])*100:.4f}%",
    ]

    report = "\n".join(lines)
    (out_dir / "report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
