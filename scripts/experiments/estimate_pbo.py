#!/usr/bin/env python
"""Estimate Probability of Backtest Overfitting (PBO) via CSCV.

Implements Combinatorially Symmetric Cross-Validation (CSCV) from
Bailey, Borwein, López de Prado, Zhu (2017) — "The Probability of
Backtest Overfitting".

Methodology:
  1. Generate N strategy variants by perturbing production config parameters.
  2. Run backtest for each variant -> collect daily net returns (T x N matrix).
  3. Split the T-day return history into S blocks of equal size.
  4. For each C(S, S/2) combination of IS/OOS blocks:
     a. Compute IS Sharpe for each strategy.
     b. Find the best-IS strategy (rank 1).
     c. Compute its OOS rank.
     d. If OOS rank > N/2 (bottom half) -> overfitted.
  5. PBO = overfitted combinations / total combinations.

Also computes the Deflated Sharpe Ratio (Bailey & López de Prado, 2014)
with n_trials = N variants, correcting for multiple testing and non-normality.

Usage:
  python3 scripts/experiments/estimate_pbo.py
  python3 scripts/experiments/estimate_pbo.py --n-variants 25 --n-blocks 16
"""

from __future__ import annotations

import argparse
import copy
import itertools
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from research.backtest_common import (
    load_execution_data,
    run_backtest_with_costs,
    compute_backtest_metrics,
)
from leadlag.models.sector_relative_ensemble_blp_enhanced import (
    SectorRelativeEnsembleBLPEnhancedModel,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

TRADING_DAYS = 245

BASELINE_BLPX = {
    "rho": 0.01,
    "alpha_xx": 0.20,
    "alpha_yx": 0.15,
    "alpha_yy": 0.50,
    "lambda_pca": 0.10,
    "lambda_sector": 0.60,
    "beta_conf": 0.25,
    "winsor_sigma": 3.0,
    "blp_window": 504,
    "ewma_halflife": 120,
    "sector_eta": 0.5,
    "sector_gamma": 4.0,
}

INT_PARAMS = {"blp_window", "ewma_halflife"}


def generate_variants(base_cfg: dict, n_variants: int = 25, seed: int = 42) -> list[dict]:
    """Generate config variants by perturbing key BLP parameters."""
    rng = np.random.default_rng(seed)
    variants: list[dict] = []

    # Variant 0: baseline
    variants.append(copy.deepcopy(base_cfg))

    params = list(BASELINE_BLPX.keys())

    # Single-parameter perturbations: 0.5x and 2.0x
    for param in params:
        for mult in [0.5, 2.0]:
            cfg = copy.deepcopy(base_cfg)
            base_val = BASELINE_BLPX[param]
            new_val = base_val * mult
            if param in INT_PARAMS:
                new_val = max(20, int(new_val))
            cfg["blpx"][param] = new_val
            variants.append(cfg)

    # Random multi-parameter perturbations for remaining slots
    n_random = max(0, n_variants - len(variants))
    multipliers = [0.5, 0.75, 1.25, 1.5, 2.0]
    for _ in range(n_random):
        cfg = copy.deepcopy(base_cfg)
        n_to_change = int(rng.integers(2, 6))
        chosen = rng.choice(params, size=n_to_change, replace=False)
        for p in chosen:
            mult = float(rng.choice(multipliers))
            base_val = BASELINE_BLPX[p]
            new_val = base_val * mult
            if p in INT_PARAMS:
                new_val = max(20, int(new_val))
            cfg["blpx"][p] = new_val
        variants.append(cfg)

    return variants[:n_variants]


def variant_label(cfg: dict) -> str:
    """Short label for a variant based on changed parameters."""
    blpx = cfg.get("blpx", {})
    parts = []
    for k, v in BASELINE_BLPX.items():
        cv = blpx.get(k, v)
        if cv != v:
            if isinstance(cv, float):
                parts.append(f"{k}={cv:.3g}")
            else:
                parts.append(f"{k}={cv}")
    if not parts:
        return "baseline"
    return ", ".join(parts)


def compute_sharpe(returns: np.ndarray, trading_days: int = TRADING_DAYS) -> float:
    """Compute annualized Sharpe ratio from daily returns."""
    if len(returns) < 2:
        return np.nan
    mean_r = np.mean(returns)
    std_r = np.std(returns, ddof=1)
    if std_r < 1e-12:
        return np.nan
    return mean_r / std_r * np.sqrt(trading_days)


def cscv_pbo(
    strategy_returns: np.ndarray,
    n_blocks: int = 16,
    trading_days: int = TRADING_DAYS,
) -> dict:
    """Compute PBO via Combinatorially Symmetric Cross-Validation.

    Args:
        strategy_returns: (T, N) array of daily returns for N strategies.
        n_blocks: number of temporal blocks (must be even).
        trading_days: annualization factor for Sharpe.

    Returns:
        dict with pbo, logit_pbo, n_combinations, oos_ranks, etc.
    """
    T, N = strategy_returns.shape
    block_size = T // n_blocks
    if block_size < 20:
        raise ValueError(
            f"Block size {block_size} too small (T={T}, n_blocks={n_blocks}). "
            f"Need T >= {n_blocks * 20}."
        )

    T_trimmed = block_size * n_blocks
    returns_trimmed = strategy_returns[:T_trimmed]

    blocks = returns_trimmed.reshape(n_blocks, block_size, N)

    half = n_blocks // 2
    combinations = list(itertools.combinations(range(n_blocks), half))
    n_comb = len(combinations)

    logger.info(
        "CSCV: N=%d strategies, T=%d days, S=%d blocks (size=%d), C(S,S/2)=%d combinations",
        N, T_trimmed, n_blocks, block_size, n_comb,
    )

    overfit_count = 0
    oos_ranks: list[int] = []
    is_sharpes_best: list[float] = []
    oos_sharpes_best: list[float] = []

    for ci, is_blocks in enumerate(combinations):
        oos_blocks = tuple(i for i in range(n_blocks) if i not in is_blocks)

        is_ret = np.concatenate([blocks[b] for b in is_blocks], axis=0)
        oos_ret = np.concatenate([blocks[b] for b in oos_blocks], axis=0)

        is_sharpes = np.array([
            compute_sharpe(is_ret[:, j], trading_days) for j in range(N)
        ])
        oos_sharpes = np.array([
            compute_sharpe(oos_ret[:, j], trading_days) for j in range(N)
        ])

        best_is_idx = int(np.nanargmax(is_sharpes))

        valid = ~np.isnan(oos_sharpes)
        n_valid = int(np.sum(valid))
        oos_rank = int(np.sum(oos_sharpes[valid] > oos_sharpes[best_is_idx]) + 1)

        if oos_rank > n_valid // 2:
            overfit_count += 1

        oos_ranks.append(oos_rank)
        is_sharpes_best.append(float(is_sharpes[best_is_idx]))
        oos_sharpes_best.append(float(oos_sharpes[best_is_idx]))

        if (ci + 1) % 2000 == 0:
            logger.info("  Processed %d/%d combinations...", ci + 1, n_comb)

    pbo = overfit_count / n_comb
    if 0 < pbo < 1:
        logit_pbo = float(np.log(pbo / (1.0 - pbo)))
    elif pbo == 0:
        logit_pbo = float("-inf")
    else:
        logit_pbo = float("inf")

    return {
        "pbo": pbo,
        "logit_pbo": logit_pbo,
        "n_combinations": n_comb,
        "overfit_count": overfit_count,
        "oos_ranks": oos_ranks,
        "is_sharpes_best": is_sharpes_best,
        "oos_sharpes_best": oos_sharpes_best,
        "n_strategies": N,
        "n_blocks": n_blocks,
        "block_size": block_size,
        "T_trimmed": T_trimmed,
    }


def deflated_sharpe_ratio(
    sharpe_hat: float,
    n_trials: int,
    n_obs: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    trading_days: int = TRADING_DAYS,
) -> tuple[float, float]:
    """Compute Deflated Sharpe Ratio p-value (Bailey & López de Prado, 2014).

    Returns (p_value, sr_star) where p_value is the probability that the
    observed Sharpe beats the expected max SR under H0 after correcting for
    n_trials, and sr_star is the expected max SR under H0.
    """
    if n_trials <= 1:
        return 1.0, 0.0

    euler_mascheroni = 0.5772156649
    e_max = (
        (1 - euler_mascheroni) * norm.ppf(1 - 1.0 / n_trials)
        + euler_mascheroni * norm.ppf(1 - 1.0 / (n_trials * np.e))
    )
    sr_star = float(e_max)

    sr_daily = sharpe_hat / np.sqrt(trading_days)
    sr_star_daily = sr_star / np.sqrt(trading_days)

    dsr_num = (sr_daily - sr_star_daily) * np.sqrt(n_obs - 1)
    dsr_denom = np.sqrt(
        1.0 - skewness * sr_daily + (kurtosis - 1) / 4.0 * sr_daily**2
    )
    if dsr_denom <= 0:
        dsr_denom = 1e-8

    dsr_z = dsr_num / dsr_denom
    p_value = float(norm.cdf(dsr_z))

    return p_value, sr_star


def generate_report(
    pbo_result: dict,
    dsr_result: dict,
    all_metrics: list[dict],
    variant_labels: list[str],
    args: argparse.Namespace,
    report_path: Path,
) -> None:
    """Generate markdown PBO report."""
    pbo = pbo_result["pbo"]
    logit_pbo = pbo_result["logit_pbo"]
    n_comb = pbo_result["n_combinations"]
    overfit = pbo_result["overfit_count"]
    N = pbo_result["n_strategies"]
    S = pbo_result["n_blocks"]
    bs = pbo_result["block_size"]

    oos_ranks = pbo_result["oos_ranks"]
    median_rank = float(np.median(oos_ranks))
    mean_rank = float(np.mean(oos_ranks))
    pct_bottom = float(np.mean(np.array(oos_ranks) > N // 2))

    is_s = np.array(pbo_result["is_sharpes_best"])
    oos_s = np.array(pbo_result["oos_sharpes_best"])
    corr_io = float(np.corrcoef(is_s, oos_s)[0, 1]) if len(is_s) > 1 else np.nan

    lines = [
        "# Probability of Backtest Overfitting (PBO) Estimation Report",
        "",
        f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Config**: `{args.config}`",
        f"**Start date**: {args.start_date}",
        f"**Method**: Combinatorially Symmetric Cross-Validation (CSCV)",
        f"**Reference**: Bailey, Borwein, López de Prado, Zhu (2017)",
        "",
        "## 1. Setup",
        "",
        f"- **N strategies**: {N} (parameter perturbations of production config)",
        f"- **S blocks**: {S} (block size = {bs} days)",
        f"- **T (trimmed)**: {pbo_result['T_trimmed']} days",
        f"- **Combinations**: C({S},{S//2}) = {n_comb}",
        f"- **Performance metric**: Annualized Sharpe ratio (net, {TRADING_DAYS} trading days)",
        "",
        "## 2. PBO Results",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| **PBO** | **{pbo:.4f}** |",
        f"| logit(PBO) | {logit_pbo:.4f} |",
        f"| Overfit combinations | {overfit} / {n_comb} |",
        f"| Mean OOS rank of best-IS strategy | {mean_rank:.2f} / {N} |",
        f"| Median OOS rank | {median_rank:.1f} / {N} |",
        f"| Pct. OOS rank in bottom half | {pct_bottom:.2%} |",
        f"| IS-OOS Sharpe correlation (best-IS) | {corr_io:.4f} |",
        "",
        "### Interpretation",
        "",
    ]

    if pbo < 0.05:
        risk_label = "Very low overfitting risk (PBO < 0.05). The strategy selection process is robust."
    elif pbo < 0.15:
        risk_label = "Low overfitting risk (0.05 <= PBO < 0.15). Some evidence of overfitting but generally robust."
    elif pbo < 0.25:
        risk_label = "Moderate overfitting risk (0.15 <= PBO < 0.25). Notable probability that IS performance does not generalize."
    elif pbo < 0.50:
        risk_label = "High overfitting risk (0.25 <= PBO < 0.50). IS results are unreliable predictors of OOS performance."
    else:
        risk_label = "Very high overfitting risk (PBO >= 0.50). IS optimization is likely overfitting."

    lines.append(f"- PBO = {pbo:.4f}: {risk_label}")
    lines.append(f"- logit(PBO) = {logit_pbo:.4f}: {'negative (favorable)' if logit_pbo < 0 else 'positive (unfavorable)'}")
    lines.append(f"- IS-OOS Sharpe correlation = {corr_io:.4f}: {'high consistency' if corr_io > 0.7 else 'moderate consistency' if corr_io > 0.4 else 'low consistency (overfitting signal)'}")
    lines.append("")

    # DSR section
    lines.extend([
        "## 3. Deflated Sharpe Ratio (DSR)",
        "",
        f"Computed for the **production baseline** (variant 0) with n_trials = {N} (number of strategy variants tested).",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Observed Sharpe (annualized) | {dsr_result['sharpe_hat']:.4f} |",
        f"| n_trials | {dsr_result['n_trials']} |",
        f"| n_obs (days) | {dsr_result['n_obs']} |",
        f"| Skewness | {dsr_result['skewness']:.4f} |",
        f"| Excess Kurtosis | {dsr_result['kurtosis'] - 3:.4f} |",
        f"| Expected max SR under H0 (SR*) | {dsr_result['sr_star']:.4f} |",
        f"| **DSR p-value** | **{dsr_result['p_value']:.4f}** |",
        "",
    ])

    if dsr_result["p_value"] > 0.95:
        dsr_label = "Strongly significant (p > 0.95). The observed Sharpe is very likely real after correcting for multiple testing."
    elif dsr_result["p_value"] > 0.90:
        dsr_label = "Significant (p > 0.90). The observed Sharpe likely survives multiple-testing correction."
    elif dsr_result["p_value"] > 0.50:
        dsr_label = "Weakly significant (0.50 < p < 0.90). Some evidence remains after correction, but not conclusive."
    else:
        dsr_label = "Not significant (p < 0.50). The observed Sharpe may be an artifact of multiple testing."

    lines.append(f"- **DSR Assessment**: {dsr_label}")
    lines.append("")

    # Strategy variants table
    lines.extend([
        "## 4. Strategy Variants",
        "",
        "| # | Config | Sharpe (net) | AR (net) | MDD | Turnover |",
        "|---|--------|-------------|----------|-----|----------|",
    ])
    for i, (m, label) in enumerate(zip(all_metrics, variant_labels)):
        sharpe = m.get("Sharpe_net", m.get("Sharpe", np.nan))
        ar = m.get("AR_net", m.get("AR", np.nan))
        mdd = m.get("MDD", np.nan)
        turnover = m.get("Turnover", np.nan)
        lines.append(
            f"| {i} | {label} | {sharpe:.4f} | {ar*100:.2f}% | {mdd*100:.2f}% | {turnover:.2f} |"
        )

    lines.extend([
        "",
        "## 5. Methodology Notes",
        "",
        "- **CSCV**: The return series is split into S non-overlapping blocks. For each C(S, S/2) partition into IS/OOS halves, the best-IS strategy is identified and its OOS rank is recorded. PBO = fraction of combinations where the OOS rank falls in the bottom half.",
        "- **DSR**: Corrects the observed Sharpe for the number of trials (N variants), non-normality (skewness, kurtosis), and sample length. A p-value close to 1 indicates the Sharpe is likely genuine.",
        "- **Parameter perturbations**: Variants are generated by multiplying key BLP parameters by {0.5, 0.75, 1.25, 1.5, 2.0}, both individually and in combination. This simulates the realistic search space explored during model development.",
        f"- **Total trials context**: The archive contains ~30 experiment scripts and multiple config variants. The n_trials={N} used here is a lower bound. See AGENTS.md for the full overfitting guard policy.",
        "",
    ])

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report saved to: %s", report_path)


def main():
    parser = argparse.ArgumentParser(description="Estimate Probability of Backtest Overfitting (PBO)")
    parser.add_argument("--config", default="configs/production/production.yaml",
                        help="Path to config YAML")
    parser.add_argument("--start-date", default="2015-01-05", help="Backtest start date")
    parser.add_argument("--n-variants", type=int, default=25,
                        help="Number of strategy variants to generate")
    parser.add_argument("--n-blocks", type=int, default=16,
                        help="Number of CSCV blocks (must be even)")
    parser.add_argument("--output-dir", default="reports/pbo",
                        help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for variant generation")
    args = parser.parse_args()

    config_path = ROOT / args.config
    with open(config_path) as f:
        base_cfg = yaml.safe_load(f)

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[1/4] Loading execution data...")
    beta_window = base_cfg.get("residualization", {}).get("beta_window", 60)
    beta_ewma_halflife = base_cfg.get("residualization", {}).get("beta_ewma_halflife")
    beta_shrinkage = base_cfg.get("residualization", {}).get("beta_shrinkage", 0.0)
    beta_winsor_sigma = base_cfg.get("residualization", {}).get("beta_winsor_sigma")
    df_exec = load_execution_data(
        beta_window=beta_window,
        beta_ewma_halflife=beta_ewma_halflife,
        beta_shrinkage=beta_shrinkage,
        beta_winsor_sigma=beta_winsor_sigma,
    )

    logger.info("[2/4] Generating %d strategy variants...", args.n_variants)
    variants = generate_variants(base_cfg, n_variants=args.n_variants, seed=args.seed)
    labels = [variant_label(cfg) for cfg in variants]

    costs = base_cfg.get("costs", {})
    cost_kwargs = dict(
        slippage_bps=float(costs.get("slippage_bps_per_side", 5.0)),
        overnight_alpha_long=float(costs.get("overnight_alpha_long", 0.75)),
        overnight_alpha_short=float(costs.get("overnight_alpha_short", 0.5)),
        buy_interest_annual=float(costs.get("buy_interest_annual", 0.025)),
        borrow_fee_annual=float(costs.get("borrow_fee_annual", 0.0115)),
        reverse_fee_bps=float(costs.get("reverse_fee_bps", 2.0)),
    )

    logger.info("[3/4] Running backtests for %d variants...", len(variants))
    all_returns: list[np.ndarray] = []
    all_metrics: list[dict] = []
    sim_index: pd.Index | None = None

    for vi, cfg in enumerate(variants):
        label = labels[vi]
        logger.info("  Variant %d/%d: %s", vi + 1, len(variants), label)
        model = SectorRelativeEnsembleBLPEnhancedModel(cfg)
        results = run_backtest_with_costs(model, df_exec, start_date=args.start_date, **cost_kwargs)
        dr = results["daily_returns"].dropna()
        if sim_index is None:
            sim_index = dr.index
        else:
            sim_index = sim_index.intersection(dr.index)
        all_returns.append(dr.values)
        m = compute_backtest_metrics(results, name=label)
        all_metrics.append(m)

    # Align all return series to common index
    T_min = min(len(r) for r in all_returns)
    returns_matrix = np.zeros((T_min, len(variants)))
    for vi, r in enumerate(all_returns):
        returns_matrix[:, vi] = r[-T_min:]

    logger.info("  Aligned return matrix: T=%d x N=%d", T_min, len(variants))

    logger.info("[4/4] Computing PBO via CSCV (n_blocks=%d)...", args.n_blocks)
    pbo_result = cscv_pbo(returns_matrix, n_blocks=args.n_blocks)

    # DSR for baseline (variant 0)
    baseline_ret = all_returns[0]
    baseline_sharpe = compute_sharpe(baseline_ret)
    from scipy.stats import skew as scipy_skew, kurtosis as scipy_kurt
    sk = float(scipy_skew(baseline_ret))
    kt = float(scipy_kurt(baseline_ret, fisher=False))  # Pearson kurtosis (3 = normal)
    n_obs = len(baseline_ret)
    dsr_p, sr_star = deflated_sharpe_ratio(
        baseline_sharpe, n_trials=len(variants), n_obs=n_obs,
        skewness=sk, kurtosis=kt,
    )
    dsr_result = {
        "sharpe_hat": baseline_sharpe,
        "n_trials": len(variants),
        "n_obs": n_obs,
        "skewness": sk,
        "kurtosis": kt,
        "p_value": dsr_p,
        "sr_star": sr_star,
    }

    # Print summary
    print("\n" + "=" * 70)
    print("PBO ESTIMATION RESULTS")
    print("=" * 70)
    print(f"  PBO:              {pbo_result['pbo']:.4f}")
    print(f"  logit(PBO):       {pbo_result['logit_pbo']:.4f}")
    print(f"  Overfit combos:   {pbo_result['overfit_count']} / {pbo_result['n_combinations']}")
    print(f"  Mean OOS rank:    {np.mean(pbo_result['oos_ranks']):.2f} / {len(variants)}")
    print(f"  DSR p-value:      {dsr_p:.4f}  (n_trials={len(variants)}, SR*={sr_star:.4f})")
    print(f"  Baseline Sharpe:  {baseline_sharpe:.4f}")
    print("=" * 70)

    # Save results
    np.save(out_dir / "returns_matrix.npy", returns_matrix)
    pd.DataFrame({
        "variant_idx": range(len(variants)),
        "label": labels,
        "sharpe_net": [m.get("Sharpe_net", m.get("Sharpe", np.nan)) for m in all_metrics],
        "AR_net": [m.get("AR_net", m.get("AR", np.nan)) for m in all_metrics],
        "MDD": [m.get("MDD", np.nan) for m in all_metrics],
        "Turnover": [m.get("Turnover", np.nan) for m in all_metrics],
    }).to_csv(out_dir / "variant_metrics.csv", index=False)

    pd.DataFrame({
        "oos_rank": pbo_result["oos_ranks"],
        "is_sharpe_best": pbo_result["is_sharpes_best"],
        "oos_sharpe_best": pbo_result["oos_sharpes_best"],
    }).to_csv(out_dir / "cscv_combinations.csv", index=False)

    # Generate report
    report_path = out_dir / "pbo_report.md"
    generate_report(pbo_result, dsr_result, all_metrics, labels, args, report_path)

    logger.info("All artifacts saved in: %s", out_dir)


if __name__ == "__main__":
    main()
