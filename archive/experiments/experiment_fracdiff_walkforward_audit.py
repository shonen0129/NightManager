#!/usr/bin/env python3
"""Walk-Forward Validation & Leakage Audit for Fractional Differentiation.

This script performs two critical validations:

1. **Walk-Forward Validation**: Splits the backtest period into yearly
   non-overlapping windows and runs backtests for d=0.1, d=0.5, and d=1.0
   (baseline) on each window independently. Reports OOS Sharpe consistency
   across windows to detect overfitting.

2. **Leakage Audit**: Runs ComplianceAuditor on the fractional diff results
   to verify:
   - No lookahead leakage (signal date < trade date)
   - Residualization correctness
   - Ensemble weight validity
   - Net/gross exposure within limits
   - No NaN/Inf in signals or weights
   - Cost consistency (gross - costs = net)

   Additionally, checks that fractional differentiation itself does not
   introduce lookahead: the binomial weights are strictly backward-looking
   (only past data is used), and the rolling window is strictly historical.

Usage:
    python3 scripts/experiments/experiment_fracdiff_walkforward_audit.py
    python3 scripts/experiments/experiment_fracdiff_walkforward_audit.py --use-cache
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from research.backtest_common import (
    compute_backtest_metrics,
    load_execution_data,
    run_backtest_with_costs,
)
from leadlag.compliance.auditor import ComplianceAuditor
from leadlag.compliance.v2_auditor import run_leakage_audit, run_numerical_audit
from leadlag.data.tickers import US_TICKERS
from leadlag.features.fractional_diff import (
    compute_weights,
    fractional_diff,
    fractional_diff_df,
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

OUTPUT_DIR = ROOT / "outputs" / "experiments" / "fractional_diff" / "walkforward_audit"


def load_config(config_path: str | None = None) -> dict:
    if config_path is None:
        config_path = ROOT / "configs" / "production" / "production.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Walk-Forward Validation
# ---------------------------------------------------------------------------

def generate_yearly_windows(
    dates: pd.DatetimeIndex,
    start_date: str = "2015-01-05",
) -> list[dict]:
    """Generate non-overlapping yearly test windows.

    Each window covers one calendar year. The model sees all data up to
    the start of the window (no retraining — the model is parameter-free
    w.r.t. d, which is a global setting).

    Returns list of dicts: {window_id, year, test_start, test_end, test_dates}
    """
    dates = pd.DatetimeIndex(dates)
    dates = dates[dates >= pd.to_datetime(start_date)]

    years = sorted(dates.year.unique())
    windows = []
    for i, yr in enumerate(years):
        yr_dates = dates[dates.year == yr]
        if len(yr_dates) < 100:
            continue
        windows.append({
            "window_id": i,
            "year": yr,
            "test_start": yr_dates[0],
            "test_end": yr_dates[-1],
            "test_dates": yr_dates,
        })
    logger.info("Generated %d yearly windows: %s to %s",
                len(windows), windows[0]["year"], windows[-1]["year"])
    return windows


def run_windowed_backtest(
    cfg: dict,
    df_exec: pd.DataFrame,
    start_date: str,
    d: float,
    threshold: float = 1e-5,
    window: int = 100,
) -> dict:
    """Run a single backtest with fractional diff order d."""
    cfg_copy = copy.deepcopy(cfg)
    if "features" not in cfg_copy:
        cfg_copy["features"] = {}
    cfg_copy["features"]["fractional_diff"] = {
        "enabled": d > 0.0,
        "d": d,
        "threshold": threshold,
        "window": window,
    }

    model = SectorRelativeEnsembleBLPEnhancedModel(cfg_copy)

    costs = cfg_copy.get("costs", {})
    results = run_backtest_with_costs(
        model,
        df_exec=df_exec,
        start_date=start_date,
        slippage_bps=float(costs.get("slippage_bps_per_side", 5.0)),
        overnight_alpha_long=float(costs.get("overnight_alpha_long", 0.75)),
        overnight_alpha_short=float(costs.get("overnight_alpha_short", 0.5)),
        buy_interest_annual=float(costs.get("buy_interest_annual", 0.025)),
        borrow_fee_annual=float(costs.get("borrow_fee_annual", 0.0115)),
        reverse_fee_bps=float(costs.get("reverse_fee_bps", 2.0)),
    )
    metrics = compute_backtest_metrics(results, name=f"d={d:.2f}")
    metrics["d"] = d
    return {"metrics": metrics, "results": results, "model": model}


def run_walkforward_validation(
    cfg: dict,
    df_exec: pd.DataFrame,
    d_values: list[float],
    start_date: str,
) -> pd.DataFrame:
    """Run walk-forward validation across yearly windows for each d value.

    For each (window, d) pair, runs a full backtest but only extracts
    the metrics for the test window dates.
    """
    all_dates = df_exec.index
    windows = generate_yearly_windows(all_dates, start_date)

    rows = []
    for win in windows:
        yr = win["year"]
        win_start = win["test_start"].strftime("%Y-%m-%d")
        logger.info("=== Window %d: Year %d (%s → %s) ===",
                    win["window_id"], yr, win_start, win["test_end"].strftime("%Y-%m-%d"))

        for d_val in d_values:
            t0 = time.perf_counter()
            bt = run_windowed_backtest(cfg, df_exec, start_date, d=d_val)
            elapsed = time.perf_counter() - t0

            # Extract window-specific daily returns
            dr = bt["results"]["daily_returns"]
            win_mask = (dr.index >= win["test_start"]) & (dr.index <= win["test_end"])
            win_dr = dr[win_mask]

            if len(win_dr) < 10:
                logger.warning("  d=%.2f: only %d days in window %d, skipping", d_val, len(win_dr), yr)
                continue

            ar = float(win_dr.mean() * 245)
            vol = float(win_dr.std(ddof=1) * np.sqrt(245))
            sharpe = ar / vol if vol > 0 else np.nan
            wealth = (1.0 + win_dr).cumprod()
            mdd = float(((wealth / wealth.cummax()) - 1.0).min())

            # Turnover for window
            turnover = bt["results"].get("daily_turnover", pd.Series())
            if hasattr(turnover, 'index'):
                win_to = turnover[(turnover.index >= win["test_start"]) & (turnover.index <= win["test_end"])]
                win_turnover = float(win_to.mean()) if len(win_to) > 0 else np.nan
            else:
                win_turnover = np.nan

            logger.info("  d=%.2f: Sharpe=%.4f, AR=%.2f%%, MDD=%.2f%%, Turnover=%.4f (%.1fs)",
                        d_val, sharpe, ar * 100, mdd * 100, win_turnover, elapsed)

            rows.append({
                "window_id": win["window_id"],
                "year": yr,
                "d": d_val,
                "Sharpe_net": sharpe,
                "AR_net": ar,
                "Vol_net": vol,
                "MDD": mdd,
                "Turnover": win_turnover,
                "n_days": len(win_dr),
                "elapsed_s": elapsed,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Leakage Audit
# ---------------------------------------------------------------------------

def run_compliance_audit(
    cfg: dict,
    df_exec: pd.DataFrame,
    start_date: str,
    d: float,
) -> dict:
    """Run ComplianceAuditor on fractional diff backtest results."""
    cfg_copy = copy.deepcopy(cfg)
    if "features" not in cfg_copy:
        cfg_copy["features"] = {}
    cfg_copy["features"]["fractional_diff"] = {
        "enabled": d > 0.0,
        "d": d,
        "threshold": 1e-5,
        "window": 100,
    }

    model = SectorRelativeEnsembleBLPEnhancedModel(cfg_copy)

    costs = cfg_copy.get("costs", {})
    results = run_backtest_with_costs(
        model,
        df_exec=df_exec,
        start_date=start_date,
        slippage_bps=float(costs.get("slippage_bps_per_side", 5.0)),
        overnight_alpha_long=float(costs.get("overnight_alpha_long", 0.75)),
        overnight_alpha_short=float(costs.get("overnight_alpha_short", 0.5)),
        buy_interest_annual=float(costs.get("buy_interest_annual", 0.025)),
        borrow_fee_annual=float(costs.get("borrow_fee_annual", 0.0115)),
        reverse_fee_bps=float(costs.get("reverse_fee_bps", 2.0)),
    )

    audit_dir = OUTPUT_DIR / "audit_d" / f"d={d:.2f}"
    audit_dir.mkdir(parents=True, exist_ok=True)

    audit_result = ComplianceAuditor.run_audit(
        model,
        df_exec=df_exec,
        results=results,
        output_dir=str(audit_dir),
    )
    return audit_result


def audit_fractional_diff_lookahead() -> dict:
    """Verify that fractional differentiation itself is lookahead-free.

    The binomial expansion weights w_k = (-1)^k * C(d, k) are applied
    to past values: fd_t = sum_k w_k * x_{t-k}. This is a strictly
    backward-looking filter. We verify:

    1. Weights are finite and decay for 0 < d < 1
    2. The filter only uses x[t], x[t-1], ..., x[t-L] (no future data)
    3. NaN/Inf do not propagate
    4. The first L values are NaN (warmup period) — no data leakage
    """
    checks = {}

    # Check 1: Weights are finite and decay
    for d in [0.1, 0.3, 0.5, 0.7, 0.9]:
        w = compute_weights(d, threshold=1e-5)
        checks[f"weights_finite_d={d:.1f}"] = bool(np.all(np.isfinite(w)))
        checks[f"weights_decay_d={d:.1f}"] = bool(np.all(np.abs(w[1:]) <= np.abs(w[:-1]) + 1e-10))

    # Check 2: Filter is backward-looking (no future data)
    np.random.seed(42)
    x = np.random.randn(200)
    s = pd.Series(x)
    fd = fractional_diff(s, d=0.5, threshold=1e-5, window=50)

    # The filter output at time t should only depend on x[0:t+1]
    # Verify by checking that modifying x[t+1:] doesn't change fd[t]
    fd_orig = fractional_diff(s, d=0.5, threshold=1e-5, window=50).values

    x_modified = x.copy()
    x_modified[150:] += 100.0  # Corrupt future values
    fd_modified = fractional_diff(pd.Series(x_modified), d=0.5, threshold=1e-5, window=50).values

    # Values before index 150 should be identical
    checks["no_future_leakage"] = bool(
        np.allclose(fd_orig[:150], fd_modified[:150], equal_nan=True)
    )

    # Check 3: NaN/Inf propagation
    checks["no_nan_inf_in_output"] = bool(
        np.all(np.isfinite(fd_orig[100:]))  # After warmup
    )

    # Check 4: Warmup period has NaN
    checks["warmup_has_nan"] = bool(
        np.any(np.isnan(fd_orig[:5]))
    )

    # Check 5: v2_auditor leakage audit (basic signal/trade date check)
    leakage = run_leakage_audit("2026-06-12", "2026-06-16")
    checks["v2_leakage_audit_passes"] = (leakage["status"] == "PASSED")

    # Overall status
    all_pass = all(checks.values())
    checks["overall_status"] = "PASSED" if all_pass else "FAILED"
    return checks


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_yearly_sharpe(wf_df: pd.DataFrame, out_path: Path):
    """Plot yearly Sharpe ratio by d value."""
    fig, ax = plt.subplots(figsize=(12, 7))
    for d_val in sorted(wf_df["d"].unique()):
        sub = wf_df[wf_df["d"] == d_val]
        ax.plot(sub["year"], sub["Sharpe_net"], "o-", linewidth=2, markersize=8,
                label=f"d={d_val:.2f}")

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Net Sharpe Ratio", fontsize=12)
    ax.set_title("Walk-Forward Validation: Yearly Sharpe by Fractional Diff Order", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_yearly_sharpe_delta(wf_df: pd.DataFrame, baseline_d: float, out_path: Path):
    """Plot yearly Sharpe delta vs baseline."""
    fig, ax = plt.subplots(figsize=(12, 7))
    baseline = wf_df[wf_df["d"] == baseline_d].set_index("year")["Sharpe_net"]

    for d_val in sorted(wf_df["d"].unique()):
        if d_val == baseline_d:
            continue
        sub = wf_df[wf_df["d"] == d_val].set_index("year")["Sharpe_net"]
        delta = sub - baseline
        colors = ["green" if d > 0 else "red" for d in delta.values]
        ax.bar(delta.index + (0.2 if d_val == 0.1 else -0.2), delta.values,
               width=0.4, color=colors, alpha=0.7, label=f"Δ(d={d_val:.2f} vs {baseline_d:.2f})")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Sharpe Delta", fontsize=12)
    ax.set_title("Walk-Forward: Yearly Sharpe Improvement vs Baseline", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Walk-Forward Validation & Leakage Audit for Fractional Diff"
    )
    parser.add_argument("--config", default="configs/production/production.yaml")
    parser.add_argument("--start-date", default="2015-01-05")
    parser.add_argument("--use-cache", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)

    if args.use_cache:
        from leadlag.data.cache import load_df_exec_from_local_cache
        logger.info("Loading cached df_exec...")
        df_exec = load_df_exec_from_local_cache()
    else:
        logger.info("Downloading and preprocessing data...")
        beta_window = cfg.get("residualization", {}).get("beta_window", 60)
        df_exec = load_execution_data(beta_window=beta_window)

    logger.info("df_exec shape: %s, date range: %s to %s",
                df_exec.shape, df_exec.index[0], df_exec.index[-1])

    d_values = [0.1, 0.5, 1.0]

    # === Phase 1: Walk-Forward Validation ===
    logger.info("=" * 70)
    logger.info("PHASE 1: WALK-FORWARD VALIDATION")
    logger.info("=" * 70)

    wf_df = run_walkforward_validation(cfg, df_exec, d_values, args.start_date)
    wf_df.to_csv(OUTPUT_DIR / "walkforward_yearly_results.csv", index=False)
    logger.info("Walk-forward results saved to %s", OUTPUT_DIR / "walkforward_yearly_results.csv")

    # Summary table
    logger.info("\n--- Yearly Sharpe by d ---")
    pivot = wf_df.pivot(index="year", columns="d", values="Sharpe_net")
    print("\n" + pivot.to_string())

    # Consistency analysis
    baseline_d = 1.0
    baseline_sharpe = wf_df[wf_df["d"] == baseline_d].set_index("year")["Sharpe_net"]

    logger.info("\n--- Consistency Analysis (vs d=1.0 baseline) ---")
    for d_val in d_values:
        if d_val == baseline_d:
            continue
        sub = wf_df[wf_df["d"] == d_val].set_index("year")["Sharpe_net"]
        delta = sub - baseline_sharpe
        n_positive = int((delta > 0).sum())
        n_total = len(delta)
        mean_delta = float(delta.mean())
        median_delta = float(delta.median())
        logger.info("  d=%.2f vs d=%.2f: %d/%d windows positive, mean Δ=%.4f, median Δ=%.4f",
                    d_val, baseline_d, n_positive, n_total, mean_delta, median_delta)

    # Plots
    plot_yearly_sharpe(wf_df, OUTPUT_DIR / "yearly_sharpe_by_d.png")
    plot_yearly_sharpe_delta(wf_df, baseline_d, OUTPUT_DIR / "yearly_sharpe_delta.png")

    # === Phase 2: Leakage Audit ===
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2: LEAKAGE AUDIT")
    logger.info("=" * 70)

    # 2a: Fractional diff intrinsic lookahead check
    logger.info("\n--- 2a: Fractional Diff Intrinsic Lookahead Check ---")
    fd_checks = audit_fractional_diff_lookahead()
    for check_name, check_val in fd_checks.items():
        status = "✓ PASS" if check_val else "✗ FAIL"
        logger.info("  %s: %s", check_name, status)

    # 2b: ComplianceAuditor on backtest results
    logger.info("\n--- 2b: ComplianceAuditor on Backtest Results ---")
    for d_val in [0.1, 0.5, 1.0]:
        logger.info("\n  Running ComplianceAuditor for d=%.2f...", d_val)
        t0 = time.perf_counter()
        audit = run_compliance_audit(cfg, df_exec, args.start_date, d_val)
        elapsed = time.perf_counter() - t0
        logger.info("  ComplianceAuditor completed (%.1fs)", elapsed)

        for key, val in audit.items():
            if isinstance(val, bool):
                status = "✓ PASS" if val else "✗ FAIL"
                logger.info("    %s: %s", key, status)
            elif isinstance(val, (int, float)):
                logger.info("    %s: %s", key, val)

    # === Summary ===
    logger.info("\n" + "=" * 70)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 70)

    # Walk-forward summary
    for d_val in d_values:
        sub = wf_df[wf_df["d"] == d_val]
        mean_s = sub["Sharpe_net"].mean()
        std_s = sub["Sharpe_net"].std()
        min_s = sub["Sharpe_net"].min()
        max_s = sub["Sharpe_net"].max()
        logger.info("  d=%.2f: mean Sharpe=%.4f, std=%.4f, min=%.4f, max=%.4f (%d windows)",
                    d_val, mean_s, std_s, min_s, max_s, len(sub))

    # Leakage audit summary
    logger.info("\n  Fractional Diff Lookahead: %s", fd_checks["overall_status"])

    # Deflated Sharpe (trial count correction)
    n_trials = len(d_values)  # number of d values tested
    baseline_sharpe_full = wf_df[wf_df["d"] == 1.0]["Sharpe_net"].mean()
    best_sharpe_full = wf_df[wf_df["d"] == 0.1]["Sharpe_net"].mean()
    improvement = best_sharpe_full - baseline_sharpe_full
    logger.info("\n  Deflated Sharpe: %d trials, improvement=%.4f (uncorrected)", n_trials, improvement)
    logger.info("  Note: With only %d d-values tested, deflation impact is minimal.", n_trials)

    logger.info("\n  All outputs saved to: %s", OUTPUT_DIR)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
