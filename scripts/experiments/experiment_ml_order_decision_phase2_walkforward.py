#!/usr/bin/env python
"""Phase 2 walk-forward validation + DSR + sensitivity for ML overlay.

Runs an expanding-window walk-forward:
  - 2022 test: train 2020-01-06 to 2021-12-31
  - 2023 test: train 2020-01-06 to 2022-12-31
  - 2024 test: train 2020-01-06 to 2023-12-31

For each fold a LightGBM overlay is retrained on in-sample data and
backtested on the OOS year.  Pooled returns across folds are used to
compute DSR.  Sensitivity re-runs the 2024 fold with all numeric
LightGBM hyperparameters at -20% and +20%.
"""

from __future__ import annotations

import argparse
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
    """Deflated Sharpe Ratio (Bailey & López de Prado 2014)."""
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
    lgbm_kwargs: dict | None = None,
) -> dict:
    """Run a single walk-forward fold."""
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
    )

    base_ret = result["baseline_result"]["daily_returns"]
    over_ret = result["overlay_result"]["daily_returns"]
    base_turn = result["baseline_result"]["daily_turnover"]
    over_turn = result["overlay_result"]["daily_turnover"]

    # crop to requested test end to handle BacktestEngine inclusive upper bound quirk
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
    t_stat, p_value = ttest_rel(over_ret, base_ret)

    elapsed = time.perf_counter() - t0
    logger.info(
        "[%s] Base Sharpe=%.4f Overlay=%.4f p=%.4f elapsed=%.1fs",
        test_start[:4], base_m["sharpe"], over_m["sharpe"], p_value, elapsed
    )

    return {
        "test_year": test_start[:4],
        "base_metrics": base_m,
        "overlay_metrics": over_m,
        "base_returns": base_ret,
        "overlay_returns": over_ret,
        "base_turnover": base_turn,
        "overlay_turnover": over_turn,
        "p_value": p_value,
        "output_dir": output_dir,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="reports/ml_order_decision/phase2_walkforward")
    parser.add_argument("--gap-input-dir", default=str(ROOT / "results" / "gap_adjusted_distribution" / "20260615_004113"))
    parser.add_argument("--n-trials", type=int, default=12, help="Effective trials for DSR")
    parser.add_argument("--trials-sharpe-std", type=float, default=0.5, help="Cross-trial Sharpe std for DSR")
    parser.add_argument("--no-sensitivity", action="store_true", help="Skip parameter sensitivity")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(ROOT / "configs" / "production" / "production.yaml") as f:
        cfg = yaml.safe_load(f)

    df_exec = load_df_exec_from_local_cache()
    gap_input_dir = Path(args.gap_input_dir)

    # Folds (expanding window)
    folds = [
        ("2020-01-06", "2021-12-31", "2022-01-01", "2022-12-31"),
        ("2020-01-06", "2022-12-31", "2023-01-01", "2023-12-31"),
        ("2020-01-06", "2023-12-31", "2024-01-01", "2024-12-31"),
    ]

    # 1. Walk-forward folds
    fold_results = []
    for train_start, train_end, test_start, test_end in folds:
        fold_out = out_dir / f"fold_{test_start[:4]}"
        result = run_fold(
            df_exec, cfg, gap_input_dir, fold_out,
            train_start, train_end, test_start, test_end,
        )
        fold_results.append(result)

    # 2. Pooled returns
    base_all = pd.concat([r["base_returns"] for r in fold_results]).sort_index()
    over_all = pd.concat([r["overlay_returns"] for r in fold_results]).sort_index()
    base_m = compute_metrics(base_all)
    over_m = compute_metrics(over_all)

    pooled_diff = over_all - base_all
    float(pooled_diff.skew())
    float(pooled_diff.kurt())

    # DSR on the overlay daily returns (not the difference; design uses Pooled net Sharpe)
    dsr = compute_deflated_sharpe(
        sharpe_hat=over_m["sharpe"],
        n_trials=args.n_trials,
        T_days=over_m["n"],
        skewness=float(over_all.skew()),
        kurtosis_excess=float(over_all.kurt()),
        trials_sharpe_std=args.trials_sharpe_std,
    )

    # 3. Sensitivity on the final 2024 fold
    sens_results = []
    if not args.no_sensitivity:
        logger.info("=== Sensitivity analysis (2024 fold) ===")
        fold_results[-1]
        train_start, train_end, test_start, test_end = folds[-1]
        for label, mult in [("minus20", 0.8), ("plus20", 1.2)]:
            lgbm_kwargs = {
                "n_estimators": int(200 * mult),
                "learning_rate": 0.05 * mult,
                "num_leaves": int(31 * mult),
                "max_depth": int(4 * mult) if int(4 * mult) >= 1 else 1,
                "min_child_samples": int(100 * mult),
                "reg_alpha": 0.1 * mult,
                "reg_lambda": 0.1 * mult,
                "subsample": min(0.8 * mult, 0.95),
                "colsample_bytree": min(0.8 * mult, 0.95),
            }
            sens_out = out_dir / f"fold_2024_sens_{label}"
            result = run_fold(
                df_exec, cfg, gap_input_dir, sens_out,
                train_start, train_end, test_start, test_end,
                lgbm_kwargs=lgbm_kwargs,
            )
            sens_results.append(
                {
                    "variant": label,
                    "multiplier": mult,
                    "base_sharpe": result["base_metrics"]["sharpe"],
                    "overlay_sharpe": result["overlay_metrics"]["sharpe"],
                    "overlay_ar": result["overlay_metrics"]["ar"],
                    "overlay_vol": result["overlay_metrics"]["vol"],
                    "overlay_mdd": result["overlay_metrics"]["mdd"],
                    "base_turnover_mean": result["base_turnover"].mean(),
                    "overlay_turnover_mean": result["overlay_turnover"].mean(),
                }
            )

    # 4. Summary report
    win_rate = sum(1 for r in fold_results if r["overlay_metrics"]["sharpe"] > r["base_metrics"]["sharpe"]) / len(fold_results)
    mean_turnover = float(np.mean([r["overlay_turnover"].mean() for r in fold_results]))
    mean_base_turnover = float(np.mean([r["base_turnover"].mean() for r in fold_results]))

    lines = [
        "# Phase 2 Walk-Forward Validation + DSR (LightGBM Overlay)\n",
        "## Pooled OOS performance\n",
        f"- Periods: {', '.join(r['test_year'] for r in fold_results)}\n",
        f"- Total OOS days: {over_m['n']}\n",
        f"- Baseline Sharpe (pooled): {base_m['sharpe']:.4f}\n",
        f"- Overlay Sharpe (pooled): {over_m['sharpe']:.4f}\n",
        f"- Overlay AR: {over_m['ar']:.4f}\n",
        f"- Overlay Vol: {over_m['vol']:.4f}\n",
        f"- Overlay MDD: {over_m['mdd']:.4f}\n",
        f"- Mean daily return difference (O - B): {pooled_diff.mean():.6f}\n",
        f"- Walk-forward win rate: {win_rate:.1%} ({sum(1 for r in fold_results if r['overlay_metrics']['sharpe'] > r['base_metrics']['sharpe'])}/{len(fold_results)})\n",
        f"- DSR ({args.n_trials} trials): {dsr:.4f}\n",
        f"- Mean turnover: base={mean_base_turnover:.4f}, overlay={mean_turnover:.4f}\n",
        "\n## Per-fold metrics\n",
        "| Year | Base Sharpe | Overlay Sharpe | Base AR | Overlay AR | Base MDD | Overlay MDD | p-value |\n",
        "|------|-------------|----------------|---------|------------|----------|-------------|---------|\n",
    ]
    for r in fold_results:
        bm = r["base_metrics"]
        om = r["overlay_metrics"]
        lines.append(
            f"| {r['test_year']} | {bm['sharpe']:.4f} | {om['sharpe']:.4f} | "
            f"{bm['ar']:.4f} | {om['ar']:.4f} | {bm['mdd']:.4f} | {om['mdd']:.4f} | {r['p_value']:.4f} |\n"
        )

    if sens_results:
        lines.append("\n## Sensitivity (2024 fold, ±20% hyperparameters)\n")
        lines.append("| Variant | Overlay Sharpe | Overlay AR | Overlay Vol | Turnover |\n")
        lines.append("|---------|----------------|------------|-------------|----------|\n")
        for s in sens_results:
            lines.append(
                f"| {s['variant']} | {s['overlay_sharpe']:.4f} | {s['overlay_ar']:.4f} | "
                f"{s['overlay_vol']:.4f} | {s['overlay_turnover_mean']:.4f} |\n"
            )
        base_sens_sharpe = sens_results[1]["overlay_sharpe"]  # plus20
        minus_sens_sharpe = sens_results[0]["overlay_sharpe"]
        sens_range = abs(base_sens_sharpe - minus_sens_sharpe) / abs(base_sens_sharpe) * 100 if base_sens_sharpe != 0 else 0.0
        lines.append(f"\n- Sensitivity Sharpe range: {sens_range:.1f}%\n")

    # Verdict against design proposal criteria
    lines.append("\n## Verdict against adoption criteria\n")
    lines.append(f"1. Pooled net Sharpe > baseline by >0.5 SE (≈0.01): {'PASS' if over_m['sharpe'] - base_m['sharpe'] > 0.01 else 'FAIL'} ({over_m['sharpe'] - base_m['sharpe']:+.4f})\n")
    lines.append(f"2. Walk-forward win rate >= 8/12: {'PASS' if win_rate >= 8/12 else 'FAIL'} ({win_rate:.1%})\n")
    lines.append(f"3. DSR >= 0.95: {'PASS' if dsr >= 0.95 else 'FAIL'} ({dsr:.4f})\n")
    if sens_results:
        lines.append(f"4. ±20% parameter perturbation Sharpe change < 20%: {'PASS' if sens_range < 20 else 'FAIL'} ({sens_range:.1f}%)\n")
    lines.append(f"5. Turnover not significantly increased: {'PASS' if mean_turnover < mean_base_turnover * 1.5 else 'FAIL'}\n")

    report_text = "".join(lines)
    (out_dir / "phase2_walkforward_report.md").write_text(report_text)
    logger.info("\n" + report_text)
    print("\n" + report_text)


if __name__ == "__main__":
    main()
