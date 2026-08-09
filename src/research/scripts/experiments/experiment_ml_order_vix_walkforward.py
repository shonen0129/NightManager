#!/usr/bin/env python
"""Walk-forward validation: LightGBM overlay with US/JP VIX features.

Compares the production-style per-ticker LightGBM overlay (Phase 2.8)
against an extended version that adds lagged US VIX, JP VIX, and spread
z-scores as market-level continuous features and interactions.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm, ttest_rel

ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.cache import load_df_exec_from_local_cache
from research.experiments.ml_order_decision.phase2 import run_phase2_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TRADING_DAYS = 245
EULER_GAMMA = 0.5772156649015329


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


def compute_deflated_sharpe(
    sharpe_hat: float,
    n_trials: int,
    T_days: int,
    skewness: float = 0.0,
    kurtosis_excess: float = 0.0,
    trials_sharpe_std: float = 0.5,
) -> float:
    sr_daily = sharpe_hat / np.sqrt(TRADING_DAYS)
    if n_trials <= 1:
        sr_0 = 0.0
    else:
        z1 = norm.ppf(1.0 - 1.0 / n_trials)
        z2 = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
        sr_0_annual = trials_sharpe_std * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)
        sr_0 = sr_0_annual / np.sqrt(TRADING_DAYS)
    gamma3 = skewness
    gamma4 = kurtosis_excess + 3.0
    denom = np.sqrt(1.0 - gamma3 * sr_daily + (gamma4 - 1.0) / 4.0 * sr_daily**2)
    if denom < 1e-12:
        return 0.0
    dsr_stat = (sr_daily - sr_0) * np.sqrt(T_days - 1) / denom
    return float(norm.cdf(dsr_stat))


def run_variant(
    df_exec: pd.DataFrame,
    cfg: dict,
    gap_input_dir: Path,
    output_dir: Path,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    lgbm_kwargs: dict,
    vix_cache_path: Path | None,
    per_ticker_interactions: bool,
) -> dict:
    pd.Timestamp.now()
    label = "vix" if vix_cache_path is not None else "no_vix"
    logger.info("=== %s fold %s: train %s->%s | test %s->%s ===", label, test_start[:4], train_start, train_end, test_start, test_end)

    variant_out = output_dir / f"fold_{test_start[:4]}" / label
    result = run_phase2_experiment(
        df_exec=df_exec,
        gap_input_dir=gap_input_dir,
        cfg=cfg,
        output_dir=variant_out,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        n_jobs=-1,
        lgbm_kwargs=lgbm_kwargs,
        p_trade_ema_span=None,
        use_ticker=True,
        use_classification=False,
        per_ticker_interactions=per_ticker_interactions,
        vix_cache_path=vix_cache_path,
    )

    base_ret = result["baseline_result"]["daily_returns"]
    over_ret = result["overlay_result"]["daily_returns"]

    end_dt = pd.to_datetime(test_end)
    base_ret = base_ret[base_ret.index <= end_dt].dropna()
    over_ret = over_ret[over_ret.index <= end_dt].dropna()

    common = base_ret.index.intersection(over_ret.index)
    base_ret = base_ret.loc[common]
    over_ret = over_ret.loc[common]

    base_m = compute_metrics(base_ret)
    over_m = compute_metrics(over_ret)
    _, p_value = ttest_rel(over_ret, base_ret)

    logger.info("[%s/%s] Base=%.4f Overlay=%.4f p=%.4f", test_start[:4], label, base_m["sharpe"], over_m["sharpe"], p_value)
    return {
        "test_year": test_start[:4],
        "label": label,
        "base_metrics": base_m,
        "overlay_metrics": over_m,
        "base_returns": base_ret,
        "overlay_returns": over_ret,
        "p_value": p_value,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="reports/ml_order_decision/vix_overlay_walkforward")
    parser.add_argument("--gap-input-dir", default=str(ROOT / "results" / "gap_adjusted_distribution" / "vix_experiment" / "20260802_001251"))
    parser.add_argument("--vix-cache", default=str(ROOT / "market_data" / "vix_regime_overlay" / "vix_cache.csv"))
    parser.add_argument("--n-trials", type=int, default=2, help="Number of independent model candidates for DSR")
    parser.add_argument("--trials-sharpe-std", type=float, default=0.5)
    parser.add_argument("--per-ticker-interactions", action="store_true", default=True)
    parser.add_argument("--no-per-ticker-interactions", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(ROOT / "configs" / "production" / "production.yaml") as f:
        cfg = yaml.safe_load(f)

    df_exec = load_df_exec_from_local_cache()
    gap_input_dir = Path(args.gap_input_dir)
    vix_cache_path = Path(args.vix_cache)

    per_ticker = not args.no_per_ticker_interactions

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

    folds = [
        ("2020-01-06", "2021-12-31", "2022-01-01", "2022-12-31"),
        ("2020-01-06", "2022-12-31", "2023-01-01", "2023-12-31"),
        ("2020-01-06", "2023-12-31", "2024-01-01", "2024-12-31"),
    ]

    results = []
    for train_start, train_end, test_start, test_end in folds:
        no_vix = run_variant(
            df_exec, cfg, gap_input_dir, out_dir,
            train_start, train_end, test_start, test_end,
            lgbm_kwargs, None, per_ticker,
        )
        vix = run_variant(
            df_exec, cfg, gap_input_dir, out_dir,
            train_start, train_end, test_start, test_end,
            lgbm_kwargs, vix_cache_path, per_ticker,
        )
        results.append({"no_vix": no_vix, "vix": vix})

    # Pooled OOS
    base_all = pd.concat([r["no_vix"]["base_returns"] for r in results]).sort_index()
    no_vix_all = pd.concat([r["no_vix"]["overlay_returns"] for r in results]).sort_index()
    vix_all = pd.concat([r["vix"]["overlay_returns"] for r in results]).sort_index()

    base_m = compute_metrics(base_all)
    no_vix_m = compute_metrics(no_vix_all)
    vix_m = compute_metrics(vix_all)

    common = base_all.index.intersection(no_vix_all.index).intersection(vix_all.index)
    base_all = base_all.loc[common]
    no_vix_all = no_vix_all.loc[common]
    vix_all = vix_all.loc[common]

    diff_no_vix = no_vix_all - base_all
    diff_vix = vix_all - base_all

    _, p_no_vix = ttest_rel(no_vix_all, base_all)
    _, p_vix = ttest_rel(vix_all, base_all)
    _, p_vix_vs_no_vix = ttest_rel(vix_all, no_vix_all)

    dsr_no_vix = compute_deflated_sharpe(
        sharpe_hat=no_vix_m["sharpe"],
        n_trials=args.n_trials,
        T_days=no_vix_m["n"],
        skewness=float(no_vix_all.skew()),
        kurtosis_excess=float(no_vix_all.kurt()),
        trials_sharpe_std=args.trials_sharpe_std,
    )
    dsr_vix = compute_deflated_sharpe(
        sharpe_hat=vix_m["sharpe"],
        n_trials=args.n_trials,
        T_days=vix_m["n"],
        skewness=float(vix_all.skew()),
        kurtosis_excess=float(vix_all.kurt()),
        trials_sharpe_std=args.trials_sharpe_std,
    )

    win_rate_no_vix = sum(1 for r in results if r["no_vix"]["overlay_metrics"]["sharpe"] > r["no_vix"]["base_metrics"]["sharpe"]) / len(results)
    win_rate_vix = sum(1 for r in results if r["vix"]["overlay_metrics"]["sharpe"] > r["vix"]["base_metrics"]["sharpe"]) / len(results)
    win_rate_vix_vs_no_vix = sum(1 for r in results if r["vix"]["overlay_metrics"]["sharpe"] > r["no_vix"]["overlay_metrics"]["sharpe"]) / len(results)

    # Feature importance from the last fold
    last_vix_fold_out = out_dir / f"fold_{folds[-1][2][:4]}" / "vix"
    imp_path = last_vix_fold_out / "lgbm_feature_importance.csv"
    feature_importance = ""
    if imp_path.exists():
        imp_df = pd.read_csv(imp_path)
        imp_df = imp_df[imp_df["importance"] > 0].sort_values("importance", ascending=False).head(20)
        feature_importance = "| feature | importance |\n|---------|------------|\n"
        for _, row in imp_df.iterrows():
            feature_importance += f"| {row['feature']} | {row['importance']:.2f} |\n"

    lines = [
        "# LightGBM Overlay with US/JP VIX Features: Walk-Forward Report",
        "",
        f"**Config**: n_estimators={lgbm_kwargs['n_estimators']}, num_leaves={lgbm_kwargs['num_leaves']}, max_depth={lgbm_kwargs['max_depth']}, min_child_samples={lgbm_kwargs['min_child_samples']}, reg_alpha={lgbm_kwargs['reg_alpha']}, reg_lambda={lgbm_kwargs['reg_lambda']}",
        f"- per_ticker_interactions={per_ticker}",
        "- VIX features: 60-day log z-score (US), 60-day log z-score (JP), 60-day z-score of JP-US spread",
        "- VIX added as: level + `× score` + `× gap` + `× score×gap` + `× score×gap_idio`",
        "",
        "## Pooled OOS Performance",
        f"- Periods: {', '.join(r['no_vix']['test_year'] for r in results)}",
        f"- Total OOS days: {vix_m['n']}",
        f"- Baseline V2 Sharpe: {base_m['sharpe']:.4f}",
        f"- No-VIX Overlay Sharpe: {no_vix_m['sharpe']:.4f} (p={p_no_vix:.4f}, DSR={dsr_no_vix:.4f})",
        f"- VIX Overlay Sharpe: {vix_m['sharpe']:.4f} (p={p_vix:.4f}, DSR={dsr_vix:.4f})",
        f"- Mean daily diff (VIX vs V2 base): {diff_vix.mean():.6f} (p={p_vix:.4f})",
        f"- Mean daily diff (No-VIX vs V2 base): {diff_no_vix.mean():.6f} (p={p_no_vix:.4f})",
        f"- Mean daily diff (VIX vs No-VIX): {(vix_all - no_vix_all).mean():.6f} (p={p_vix_vs_no_vix:.4f})",
        f"- WF win rate vs V2 base: No-VIX {win_rate_no_vix:.0%}, VIX {win_rate_vix:.0%}",
        f"- WF win rate vs No-VIX: {win_rate_vix_vs_no_vix:.0%} ({int(win_rate_vix_vs_no_vix*len(results))}/{len(results)})",
        "",
        "## Per-Fold Metrics",
        "",
        "| Year | Base Sharpe | No-VIX Sharpe | VIX Sharpe | No-VIX p | VIX p | VIX vs No-VIX p |",
        "|------|-------------|---------------|------------|----------|-------|-----------------|",
    ]
    for r in results:
        y = r["no_vix"]["test_year"]
        bs = r["no_vix"]["base_metrics"]["sharpe"]
        ns = r["no_vix"]["overlay_metrics"]["sharpe"]
        vs = r["vix"]["overlay_metrics"]["sharpe"]
        npv = r["no_vix"]["p_value"]
        vpv = r["vix"]["p_value"]
        _, p_vn = ttest_rel(r["vix"]["overlay_returns"], r["no_vix"]["overlay_returns"])
        lines.append(f"| {y} | {bs:.4f} | {ns:.4f} | {vs:.4f} | {npv:.4f} | {vpv:.4f} | {p_vn:.4f} |")

    lines += [
        "",
        "## Per-Fold Additional Metrics",
        "",
        "| Year | Base AR | No-VIX AR | VIX AR | Base Vol | No-VIX Vol | VIX Vol | Base MDD | No-VIX MDD | VIX MDD |",
        "|------|---------|-----------|--------|----------|------------|---------|----------|------------|---------|",
    ]
    for r in results:
        y = r["no_vix"]["test_year"]
        bm = r["no_vix"]["base_metrics"]
        nm = r["no_vix"]["overlay_metrics"]
        vm = r["vix"]["overlay_metrics"]
        lines.append(
            f"| {y} | {bm['ar']:.4f} | {nm['ar']:.4f} | {vm['ar']:.4f} | {bm['vol']:.4f} | "
            f"{nm['vol']:.4f} | {vm['vol']:.4f} | {bm['mdd']:.4f} | {nm['mdd']:.4f} | {vm['mdd']:.4f} |"
        )

    if feature_importance:
        lines += [
            "",
            "## VIX Overlay Top Features (2024 fold)",
            "",
            feature_importance,
        ]

    lines += [
        "",
        "## Verdict against Adoption Criteria",
        f"1. VIX overlay Sharpe > no-VIX overlay by >0.01: {'PASS' if vix_m['sharpe'] - no_vix_m['sharpe'] > 0.01 else 'FAIL'} ({vix_m['sharpe'] - no_vix_m['sharpe']:+.4f})",
        f"2. VIX overlay beats no-VIX in >=2/3 folds: {'PASS' if win_rate_vix_vs_no_vix >= 2/3 else 'FAIL'} ({win_rate_vix_vs_no_vix:.0%})",
        f"3. DSR >= 0.95: {'PASS' if dsr_vix >= 0.95 else 'FAIL'} ({dsr_vix:.4f})",
        f"4. Pooled daily return diff (VIX vs no-VIX) p < 0.10: {'PASS' if p_vix_vs_no_vix < 0.10 else 'FAIL'} (p={p_vix_vs_no_vix:.4f})",
    ]

    report = "\n".join(lines)
    (out_dir / "vix_overlay_walkforward_report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
