#!/usr/bin/env python
"""Phase 2.5 walk-forward validation: EMA-smoothed p_trade + stronger LightGBM."""

from __future__ import annotations

import argparse
import copy
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm, ttest_rel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from experiments.ml_order_decision.phase2 import run_phase2_experiment
from leadlag.data.cache import load_df_exec_from_local_cache

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


def run_fold(
    df_exec: pd.DataFrame,
    cfg: dict,
    gap_input_dir: Path,
    output_dir: Path,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    lgbm_kwargs: dict,
    p_trade_ema_span: float,
    use_ticker: bool = True,
    use_classification: bool = False,
    per_ticker_interactions: bool = False,
) -> dict:
    t0 = time.perf_counter()
    logger.info("=== Fold %s: train %s->%s | test %s->%s ===", test_start[:4], train_start, train_end, test_start, test_end)

    result = run_phase2_experiment(
        df_exec=df_exec,
        gap_input_dir=gap_input_dir,
        cfg=cfg,
        output_dir=output_dir,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        n_jobs=-1,
        lgbm_kwargs=lgbm_kwargs,
        p_trade_ema_span=p_trade_ema_span,
        use_ticker=use_ticker,
        use_classification=use_classification,
        per_ticker_interactions=per_ticker_interactions,
    )

    base_ret = result["baseline_result"]["daily_returns"]
    over_ret = result["overlay_result"]["daily_returns"]
    base_turn = result["baseline_result"]["daily_turnover"]
    over_turn = result["overlay_result"]["daily_turnover"]

    end_dt = pd.to_datetime(test_end)
    base_ret = base_ret[base_ret.index <= end_dt]
    over_ret = over_ret[over_ret.index <= end_dt]
    base_turn = base_turn[base_turn.index <= end_dt]
    over_turn = over_turn[over_turn.index <= end_dt]

    common = base_ret.index.intersection(over_ret.index)
    base_ret = base_ret.loc[common]
    over_ret = over_ret.loc[common]

    base_m = compute_metrics(base_ret)
    over_m = compute_metrics(over_ret)
    _, p_value = ttest_rel(over_ret, base_ret)

    elapsed = time.perf_counter() - t0
    logger.info("[%s] Base=%.4f Overlay=%.4f p=%.4f (%.1fs)", test_start[:4], base_m["sharpe"], over_m["sharpe"], p_value, elapsed)
    return {
        "test_year": test_start[:4],
        "base_metrics": base_m,
        "overlay_metrics": over_m,
        "base_returns": base_ret,
        "overlay_returns": over_ret,
        "base_turnover": base_turn,
        "overlay_turnover": over_turn,
        "p_value": p_value,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="reports/ml_order_decision/phase2_5_walkforward")
    parser.add_argument("--gap-input-dir", default=str(ROOT / "results" / "gap_adjusted_distribution" / "20260615_004113"))
    parser.add_argument("--n-trials", type=int, default=12)
    parser.add_argument("--trials-sharpe-std", type=float, default=0.5)
    parser.add_argument("--p-trade-ema-span", type=float, default=3.0)
    parser.add_argument("--no-ticker", action="store_true", help="Remove raw ticker categorical feature")
    parser.add_argument("--use-classification", action="store_true", help="Use binary positive-contribution target")
    parser.add_argument("--per-ticker-interactions", action="store_true", help="Add explicit one-hot ticker x score/gap interaction features")
    parser.add_argument("--no-sensitivity", action="store_true")
    # LightGBM defaults
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--num-leaves", type=int, default=20)
    parser.add_argument("--min-child-samples", type=int, default=300)
    parser.add_argument("--reg-alpha", type=float, default=0.5)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(ROOT / "configs" / "production" / "production.yaml") as f:
        cfg = yaml.safe_load(f)

    df_exec = load_df_exec_from_local_cache()
    gap_input_dir = Path(args.gap_input_dir)

    lgbm_kwargs = {
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "num_leaves": args.num_leaves,
        "min_child_samples": args.min_child_samples,
        "reg_alpha": args.reg_alpha,
        "reg_lambda": args.reg_lambda,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
    }

    folds = [
        ("2020-01-06", "2021-12-31", "2022-01-01", "2022-12-31"),
        ("2020-01-06", "2022-12-31", "2023-01-01", "2023-12-31"),
        ("2020-01-06", "2023-12-31", "2024-01-01", "2024-12-31"),
    ]

    fold_results = []
    for train_start, train_end, test_start, test_end in folds:
        fold_out = out_dir / f"fold_{test_start[:4]}"
        result = run_fold(
            df_exec, cfg, gap_input_dir, fold_out,
            train_start, train_end, test_start, test_end,
            lgbm_kwargs, args.p_trade_ema_span,
            not args.no_ticker,
            args.use_classification,
            args.per_ticker_interactions,
        )
        fold_results.append(result)

    base_all = pd.concat([r["base_returns"] for r in fold_results]).sort_index()
    over_all = pd.concat([r["overlay_returns"] for r in fold_results]).sort_index()
    base_m = compute_metrics(base_all)
    over_m = compute_metrics(over_all)
    pooled_diff = over_all - base_all

    dsr = compute_deflated_sharpe(
        sharpe_hat=over_m["sharpe"],
        n_trials=args.n_trials,
        T_days=over_m["n"],
        skewness=float(over_all.skew()),
        kurtosis_excess=float(over_all.kurt()),
        trials_sharpe_std=args.trials_sharpe_std,
    )

    win_rate = sum(1 for r in fold_results if r["overlay_metrics"]["sharpe"] > r["base_metrics"]["sharpe"]) / len(fold_results)
    mean_base_turn = float(np.mean([r["base_turnover"].mean() for r in fold_results]))
    mean_over_turn = float(np.mean([r["overlay_turnover"].mean() for r in fold_results]))

    # Sensitivity on 2024 fold
    sens_results = []
    if not args.no_sensitivity:
        train_start, train_end, test_start, test_end = folds[-1]
        for label, mult in [("minus20", 0.8), ("plus20", 1.2)]:
            sens_lgbm = copy.deepcopy(lgbm_kwargs)
            for k in ["n_estimators", "num_leaves", "max_depth", "min_child_samples"]:
                if k in sens_lgbm:
                    sens_lgbm[k] = max(1, int(sens_lgbm[k] * mult))
            for k in ["learning_rate", "reg_alpha", "reg_lambda"]:
                if k in sens_lgbm:
                    sens_lgbm[k] = sens_lgbm[k] * mult
            for k in ["subsample", "colsample_bytree"]:
                if k in sens_lgbm:
                    sens_lgbm[k] = min(0.99, max(0.1, sens_lgbm[k] * mult))
            sens_out = out_dir / f"fold_2024_sens_{label}"
            result = run_fold(
                df_exec, cfg, gap_input_dir, sens_out,
                train_start, train_end, test_start, test_end,
                sens_lgbm, args.p_trade_ema_span,
                not args.no_ticker,
                args.use_classification,
                args.per_ticker_interactions,
            )
            sens_results.append({
                "variant": label,
                "multiplier": mult,
                "overlay_sharpe": result["overlay_metrics"]["sharpe"],
                "overlay_ar": result["overlay_metrics"]["ar"],
                "overlay_vol": result["overlay_metrics"]["vol"],
                "overlay_turnover_mean": result["overlay_turnover"].mean(),
            })

    lines = [
        "# Phase 2.5 Walk-Forward Report (EMA + stronger regularization)\n",
        f"**Config**: EMA span={args.p_trade_ema_span}, n_estimators={args.n_estimators}, num_leaves={args.num_leaves}, reg_lambda={args.reg_lambda}, per_ticker_interactions={args.per_ticker_interactions}\n\n",
        "## Pooled OOS performance\n",
        f"- Periods: {', '.join(r['test_year'] for r in fold_results)}\n",
        f"- Total OOS days: {over_m['n']}\n",
        f"- Baseline Sharpe: {base_m['sharpe']:.4f}\n",
        f"- Overlay Sharpe: {over_m['sharpe']:.4f}\n",
        f"- Mean daily return difference: {pooled_diff.mean():.6f}\n",
        f"- Walk-forward win rate: {win_rate:.1%} ({sum(1 for r in fold_results if r['overlay_metrics']['sharpe'] > r['base_metrics']['sharpe'])}/{len(fold_results)})\n",
        f"- DSR ({args.n_trials} trials): {dsr:.4f}\n",
        f"- Mean turnover: base={mean_base_turn:.4f}, overlay={mean_over_turn:.4f}\n",
        "\n## Per-fold metrics\n",
        "| Year | Base Sharpe | Overlay Sharpe | Base AR | Overlay AR | Base MDD | Overlay MDD | p-value |\n",
        "|------|-------------|----------------|---------|------------|----------|-------------|---------|\n",
    ]
    for r in fold_results:
        bm = r["base_metrics"]
        om = r["overlay_metrics"]
        lines.append(f"| {r['test_year']} | {bm['sharpe']:.4f} | {om['sharpe']:.4f} | {bm['ar']:.4f} | {om['ar']:.4f} | {bm['mdd']:.4f} | {om['mdd']:.4f} | {r['p_value']:.4f} |\n")

    if sens_results:
        lines.append("\n## Sensitivity (2024 fold, ±20% hyperparameters)\n")
        lines.append("| Variant | Overlay Sharpe | Overlay AR | Overlay Vol | Turnover |\n")
        lines.append("|---------|----------------|------------|-------------|----------|\n")
        for s in sens_results:
            lines.append(f"| {s['variant']} | {s['overlay_sharpe']:.4f} | {s['overlay_ar']:.4f} | {s['overlay_vol']:.4f} | {s['overlay_turnover_mean']:.4f} |\n")
        s_min = min(s['overlay_sharpe'] for s in sens_results)
        s_max = max(s['overlay_sharpe'] for s in sens_results)
        s_mean = np.mean([s['overlay_sharpe'] for s in sens_results])
        sens_range = (s_max - s_min) / s_mean * 100 if s_mean != 0 else 0.0
        lines.append(f"\n- Sensitivity Sharpe range: {sens_range:.1f}%\n")

    lines.append("\n## Verdict against adoption criteria\n")
    lines.append(f"1. Pooled net Sharpe > baseline by >0.01: {'PASS' if over_m['sharpe'] - base_m['sharpe'] > 0.01 else 'FAIL'} ({over_m['sharpe'] - base_m['sharpe']:+.4f})\n")
    lines.append(f"2. Walk-forward win rate >= 8/12: {'PASS' if win_rate >= 8/12 else 'FAIL'} ({win_rate:.1%})\n")
    lines.append(f"3. DSR >= 0.95: {'PASS' if dsr >= 0.95 else 'FAIL'} ({dsr:.4f})\n")
    if sens_results:
        lines.append(f"4. ±20% parameter perturbation Sharpe change < 20%: {'PASS' if sens_range < 20 else 'FAIL'} ({sens_range:.1f}%)\n")
    lines.append(f"5. Turnover not significantly increased: {'PASS' if mean_over_turn < mean_base_turn * 1.5 else 'FAIL'}\n")

    report = "".join(lines)
    (out_dir / "phase2_5_walkforward_report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
