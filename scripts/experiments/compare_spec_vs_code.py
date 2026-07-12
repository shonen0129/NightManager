#!/usr/bin/env python
"""Compare spec vs code values for prior subspace vectors and BLP parameters.

Discrepancies identified:
  1. Sensitivity labels w3-w6: spec uses granular values, code uses 7-level grid
  2. BLP parameters: alpha_xx, alpha_yx, blp_window, blp_ewma_halflife differ

This script runs backtests for each variant and reports net Sharpe / MDD / turnover.
"""

from __future__ import annotations

import copy
import logging
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.core import correlation as corr_mod
from research.backtest_common import (
    compute_backtest_metrics,
    load_execution_data,
    run_backtest_with_costs,
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

# --- Spec sensitivity labels (from docs/モデル技術仕様書.md §1.3) ---
SPEC_LABELS = {
    "w3": np.array(
        [
            1.0, 0.3, 0.2, 0.8, 0.9, 0.7, -1.0, 0.4, -0.9, -0.8, 1.0, 0.0, 0.6, -0.2, -0.7,
            -0.9, 0.3, 0.6, 0.9, -0.9, 1.0, 1.0, 0.9, 0.8, -0.3, -1.0, -0.4, 0.7, -0.5, 0.8, 0.6, 0.5,
        ],
        dtype=float,
    ),
    "w4": np.array(
        [
            0.4, 0.0, 0.1, 0.2, 0.7, 0.8, -0.5, -0.4, -0.7, -0.4, 0.6, 0.3, 0.1, 0.6, -0.3,
            -0.6, 0.2, 0.2, 0.5, -0.2, 1.0, 0.6, 0.8, 1.0, -0.2, -0.8, -0.4, 0.8, -0.7, 0.3, 0.0, -0.9,
        ],
        dtype=float,
    ),
    "w5": np.array(
        [
            0.4, 0.0, 1.0, 0.0, 0.2, 0.0, -0.3, 0.0, -0.8, 0.0, -0.3, 0.0, 0.4, -0.1, -0.3,
            -0.3, 1.0, -0.1, 0.3, 0.0, -0.2, 0.2, 0.0, 0.0, 0.0, -0.9, -0.1, 0.7, -0.2, 0.0, 0.0, 0.0,
        ],
        dtype=float,
    ),
    "w6": np.array(
        [
            0.8, -0.3, 1.0, 0.3, 0.3, -0.5, -0.2, 0.4, -0.7, -0.2, -0.4, -0.1, 0.5, -0.4, -0.3,
            -0.4, 1.0, 0.3, 0.7, -0.2, -0.1, 0.6, 0.2, -0.3, -0.3, -0.8, -0.3, 0.8, -0.5, 0.2, 0.1, 0.3,
        ],
        dtype=float,
    ),
}

# --- Spec BLP parameters (from docs/モデル技術仕様書.md §2.2/§3.1) ---
SPEC_BLP_PARAMS = {
    "alpha_xx": 0.50,
    "alpha_yx": 0.05,
    "alpha_yy": 0.50,
    "blp_window": 252,
    "ewma_halflife": 45,  # blp_ewma_halflife
    "lambda_pca": 0.10,
    "lambda_sector": 0.60,  # §2.2 says 0.60, §3.1 table says 0.40
    "rho": 0.01,
    "beta_conf": 0.25,
    "winsor_sigma": 3.0,
}


def apply_spec_labels():
    """Monkey-patch get_static_sensitivity_labels with spec values."""
    original = corr_mod.get_static_sensitivity_labels

    def patched():
        return {k: v.copy() for k, v in SPEC_LABELS.items()}

    corr_mod.get_static_sensitivity_labels = patched
    return original


def restore_labels(original):
    """Restore original sensitivity labels function."""
    corr_mod.get_static_sensitivity_labels = original


def make_spec_config(base_cfg: dict) -> dict:
    """Create config with spec BLP parameters."""
    cfg = copy.deepcopy(base_cfg)
    for key, val in SPEC_BLP_PARAMS.items():
        cfg["blpx"][key] = val
    return cfg


def make_spec_lambda_sector_040(base_cfg: dict) -> dict:
    """Create config with lambda_sector=0.40 (spec §3.1 table value)."""
    cfg = copy.deepcopy(base_cfg)
    cfg["blpx"]["lambda_sector"] = 0.40
    return cfg


def run_variant(name: str, cfg: dict, df_exec, use_spec_labels: bool = False) -> dict:
    """Run a single backtest variant."""
    original_labels = None
    if use_spec_labels:
        original_labels = apply_spec_labels()

    try:
        # Clear caches to avoid contamination
        from leadlag.core.correlation import _BASELINE_CORR_CACHE, _ROLLING_CORR_CACHE
        _BASELINE_CORR_CACHE.clear()
        _ROLLING_CORR_CACHE.clear()

        # Also clear BLP cache
        from leadlag.models.sector_relative_ensemble_blp_enhanced import _BLP_CORR_CACHE
        _BLP_CORR_CACHE.clear()

        model = SectorRelativeEnsembleBLPEnhancedModel(cfg)

        t0 = time.perf_counter()
        results = run_backtest_with_costs(
            model,
            df_exec,
            start_date="2015-01-05",
            slippage_bps=5.0,
            overnight_alpha_long=0.75,
            overnight_alpha_short=0.5,
            buy_interest_annual=0.025,
            borrow_fee_annual=0.0115,
            reverse_fee_bps=2.0,
        )
        elapsed = time.perf_counter() - t0
        metrics = compute_backtest_metrics(results, name=name)
        metrics["elapsed_s"] = elapsed
        return metrics
    finally:
        if original_labels is not None:
            restore_labels(original_labels)


def print_comparison(results: list[dict]):
    """Print comparison table."""
    print("\n" + "=" * 120)
    print(f"{'Variant':<45} {'Sharpe_net':>12} {'Sharpe_gross':>14} {'MDD':>10} {'Turnover':>10} {'AR_net':>10} {'GrossExp':>10}")
    print("-" * 120)
    for m in results:
        print(
            f"{m['name']:<45} "
            f"{m['Sharpe_net']:>12.4f} "
            f"{m['Sharpe_gross']:>14.4f} "
            f"{m['MDD']*100:>9.2f}% "
            f"{m['Turnover']:>10.4f} "
            f"{m['AR_net']*100:>9.2f}% "
            f"{m['GrossExp']:>10.4f}"
        )
    print("=" * 120)


def main():
    config_path = ROOT / "configs/production/production.yaml"
    logger.info("Loading config from %s", config_path)
    with open(config_path) as f:
        base_cfg = yaml.safe_load(f)

    logger.info("Loading execution data...")
    beta_window = base_cfg.get("residualization", {}).get("beta_window", 60)
    beta_shrinkage = base_cfg.get("residualization", {}).get("beta_shrinkage", 0.0)
    beta_winsor_sigma = base_cfg.get("residualization", {}).get("beta_winsor_sigma")
    df_exec = load_execution_data(
        beta_window=beta_window,
        beta_shrinkage=beta_shrinkage,
        beta_winsor_sigma=beta_winsor_sigma,
    )
    logger.info("Data loaded: %d rows", len(df_exec))

    all_results = []

    # A: Baseline (current config + current code labels)
    logger.info("\n=== A: Baseline (current config + code labels) ===")
    m = run_variant("A: Baseline (code params + code labels)", base_cfg, df_exec, use_spec_labels=False)
    all_results.append(m)
    logger.info("Sharpe_net=%.4f, MDD=%.2f%%, elapsed=%.1fs", m["Sharpe_net"], m["MDD"]*100, m["elapsed_s"])

    # B: Spec BLP params + code labels
    logger.info("\n=== B: Spec BLP params + code labels ===")
    spec_cfg = make_spec_config(base_cfg)
    m = run_variant("B: Spec params + code labels", spec_cfg, df_exec, use_spec_labels=False)
    all_results.append(m)
    logger.info("Sharpe_net=%.4f, MDD=%.2f%%, elapsed=%.1fs", m["Sharpe_net"], m["MDD"]*100, m["elapsed_s"])

    # C: Current config params + spec labels
    logger.info("\n=== C: Current params + spec labels ===")
    m = run_variant("C: Code params + spec labels", base_cfg, df_exec, use_spec_labels=True)
    all_results.append(m)
    logger.info("Sharpe_net=%.4f, MDD=%.2f%%, elapsed=%.1fs", m["Sharpe_net"], m["MDD"]*100, m["elapsed_s"])

    # D: Full spec (spec params + spec labels)
    logger.info("\n=== D: Full spec (spec params + spec labels) ===")
    m = run_variant("D: Spec params + spec labels", spec_cfg, df_exec, use_spec_labels=True)
    all_results.append(m)
    logger.info("Sharpe_net=%.4f, MDD=%.2f%%, elapsed=%.1fs", m["Sharpe_net"], m["MDD"]*100, m["elapsed_s"])

    # E: lambda_sector=0.40 (spec §3.1 table) + code labels
    logger.info("\n=== E: lambda_sector=0.40 + code labels ===")
    ls040_cfg = make_spec_lambda_sector_040(base_cfg)
    m = run_variant("E: lambda_sector=0.40 + code labels", ls040_cfg, df_exec, use_spec_labels=False)
    all_results.append(m)
    logger.info("Sharpe_net=%.4f, MDD=%.2f%%, elapsed=%.1fs", m["Sharpe_net"], m["MDD"]*100, m["elapsed_s"])

    print_comparison(all_results)

    # Save results
    out_dir = ROOT / "results" / "spec_vs_code_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    import json
    with open(out_dir / "comparison_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("Results saved to %s", out_dir)


if __name__ == "__main__":
    main()
