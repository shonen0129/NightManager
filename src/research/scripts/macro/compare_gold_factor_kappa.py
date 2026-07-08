#!/usr/bin/env python
"""Compare 3-factor vs 4-factor macro-kappa performance.

Compares baseline (USDJPY, CLF, TNX) vs gold-added (USDJPY, CLF, TNX, GOLD).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from research.backtest_common import load_execution_data, run_backtest_with_costs
from leadlag.core.macro import (
    MACRO_NAMES,
    MACRO_SENS_MATRIX,
    MACRO_SENS_MATRIX_3FACTOR,
    MACRO_TICKERS,
)
from leadlag.models.sector_relative_ensemble_blp_enhanced import (
    SectorRelativeEnsembleBLPEnhancedModel,
)
from leadlag.reporting.metrics import calculate_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_backtest(config: dict, start_date: str, end_date: str, use_3factor: bool = False) -> dict:
    """Run backtest with given config."""
    logger.info(f"Running backtest from {start_date} to {end_date} (3-factor: {use_3factor})")
    
    # Temporarily replace sensitivity matrix for 3-factor version
    if use_3factor:
        import leadlag.core.macro as macro_module
        original_matrix = macro_module.MACRO_SENS_MATRIX.copy()
        original_names = macro_module.MACRO_NAMES.copy()
        original_tickers = macro_module.MACRO_TICKERS.copy()
        
        macro_module.MACRO_SENS_MATRIX = MACRO_SENS_MATRIX_3FACTOR
        macro_module.MACRO_NAMES = ["USDJPY", "CLF", "TNX"]
        macro_module.MACRO_TICKERS = ["JPY=X", "CL=F", "^TNX"]
    
    # For 4-factor version, ensure copper is used (already set in macro.py)
    
    # Load execution data
    df_exec = load_execution_data()
    
    # Filter by date range (keep earlier data for baseline)
    baseline_start = "2010-01-01"
    df_exec_filtered = df_exec.loc[baseline_start:end_date]
    
    # Initialize model
    model = SectorRelativeEnsembleBLPEnhancedModel(config)
    
    # Run backtest with standard costs
    results = run_backtest_with_costs(
        model,
        df_exec=df_exec_filtered,
        start_date=start_date,
    )
    
    # Restore original sensitivity matrix
    if use_3factor:
        macro_module.MACRO_SENS_MATRIX = original_matrix
        macro_module.MACRO_NAMES = original_names
        macro_module.MACRO_TICKERS = original_tickers
    
    # Calculate metrics
    metrics = calculate_metrics(
        daily_returns=results["daily_returns"],
    )
    
    return {
        "returns": results["daily_returns"],
        "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare 3-factor vs 4-factor macro-kappa")
    parser.add_argument("--config", default="configs/production/production.yaml")
    parser.add_argument("--start-date", default="2015-01-05")  # Need data from 2010 for baseline
    parser.add_argument("--end-date", default="2024-12-31")
    args = parser.parse_args()
    
    # Load base config
    base_cfg = _load_config(Path(args.config))
    
    # 3-factor config (baseline)
    cfg_3factor = copy.deepcopy(base_cfg)
    cfg_3factor["blpx"]["macro_confidence_enabled"] = True
    cfg_3factor["blpx"]["macro_kappas"] = [3.0, 0.5, 0.5]  # USDJPY, CLF, TNX
    
    # 4-factor config (with copper)
    cfg_4factor = copy.deepcopy(base_cfg)
    cfg_4factor["blpx"]["macro_confidence_enabled"] = True
    cfg_4factor["blpx"]["macro_kappas"] = [3.0, 0.5, 0.5, 0.7]  # USDJPY, CLF, TNX, COPPER
    
    logger.info("="*60)
    logger.info("Running 3-factor baseline backtest")
    logger.info("="*60)
    results_3factor = run_backtest(cfg_3factor, args.start_date, args.end_date, use_3factor=True)
    
    logger.info("="*60)
    logger.info("Running 4-factor copper-added backtest")
    logger.info("="*60)
    results_4factor = run_backtest(cfg_4factor, args.start_date, args.end_date, use_3factor=False)
    
    # Compare results
    logger.info("="*60)
    logger.info("PERFORMANCE COMPARISON")
    logger.info("="*60)
    
    metrics_3 = results_3factor["metrics"]
    metrics_4 = results_4factor["metrics"]
    
    print(f"\n{'Metric':<25} {'3-Factor':<15} {'4-Factor':<15} {'Delta':<15}")
    print("-"*70)
    
    for key in ["Sharpe", "AR", "MDD", "RISK"]:
        val_3 = metrics_3.get(key, 0)
        val_4 = metrics_4.get(key, 0)
        delta = val_4 - val_3
        print(f"{key:<25} {val_3:<15.4f} {val_4:<15.4f} {delta:<15.4f}")
    
    logger.info("="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    sharpe_delta = metrics_4.get("Sharpe", 0) - metrics_3.get("Sharpe", 0)
    return_delta = metrics_4.get("AR", 0) - metrics_3.get("AR", 0)
    drawdown_delta = metrics_4.get("MDD", 0) - metrics_3.get("MDD", 0)
    
    logger.info(f"Sharpe Ratio Delta: {sharpe_delta:+.4f}")
    logger.info(f"Annual Return Delta: {return_delta:+.4f}")
    logger.info(f"Max Drawdown Delta: {drawdown_delta:+.4f}")
    
    if sharpe_delta > 0:
        logger.info(f"✓ Copper factor improved Sharpe Ratio by {sharpe_delta:.4f}")
    else:
        logger.info(f"✗ Copper factor reduced Sharpe Ratio by {abs(sharpe_delta):.4f}")


if __name__ == "__main__":
    import copy
    main()
