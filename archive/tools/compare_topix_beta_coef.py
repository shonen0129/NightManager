#!/usr/bin/env python
"""Compare topix_beta_coef values (0.6 vs 1.20) via backtest."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Add src/ to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.fetcher import download_data
from leadlag.data.preprocessor import preprocess_data
from leadlag.data.tickers import JP_TICKERS, TOPIX_TICKER
from leadlag.models.sector_relative_ensemble_blp_enhanced import SectorRelativeEnsembleBLPEnhancedModel
from leadlag.reporting.metrics import calculate_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def run_backtest_with_beta_coef(
    cfg: dict,
    df_exec: pd.DataFrame,
    topix_beta_coef: float,
    start_date: str = "2015-01-05",
    end_date: str = "latest",
    slippage_bps: float = 5.0,
) -> dict:
    """Run backtest with specific topix_beta_coef value."""
    # Override topix_beta_coef in config
    if "residualization" not in cfg:
        cfg["residualization"] = {}
    cfg["residualization"]["topix_beta_coef"] = topix_beta_coef

    logger.info(f"Running backtest with topix_beta_coef={topix_beta_coef}")
    
    # Instantiate model
    model = SectorRelativeEnsembleBLPEnhancedModel(cfg)
    
    # Run backtest
    from leadlag.execution.backtester import BacktestEngine
    results = BacktestEngine.run_backtest(
        model,
        df_exec,
        start_date=start_date,
        end_date=end_date,
        slippage_bps=slippage_bps,
    )
    
    # Calculate metrics
    metrics = calculate_metrics(results["daily_returns"])
    
    return {
        "topix_beta_coef": topix_beta_coef,
        "metrics": metrics,
        "daily_returns": results["daily_returns"],
        "equity_curve": results["equity_curve"],
        "drawdown": results["drawdown"],
        "weights": results["weights"],
    }


def main():
    # Load production config
    config_path = ROOT / "configs" / "production.yaml"
    if config_path.exists():
        logger.info(f"Loading config from: {config_path}")
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    else:
        logger.error(f"Config path {config_path} not found.")
        sys.exit(1)

    # Download and preprocess data
    logger.info("Downloading market data...")
    raw_data = download_data(beta_window=60)
    logger.info("Preprocessing market data...")
    df_exec = preprocess_data(raw_data, beta_window=60)

    # Compute TOPIX returns
    topix_close = raw_data["jp_close"][TOPIX_TICKER].copy()
    topix_open = raw_data["jp_open"][TOPIX_TICKER].copy()
    topix_close.index = pd.to_datetime(topix_close.index).tz_localize(None).normalize()
    topix_open.index = pd.to_datetime(topix_open.index).tz_localize(None).normalize()
    r_topix_oc = topix_close / topix_open - 1.0
    df_exec["topix_oc_return"] = r_topix_oc.reindex(df_exec.index).values
    df_exec["topix_cc_trade"] = (1.0 + df_exec["topix_night_return"]) * (1.0 + df_exec["topix_oc_return"]) - 1.0

    # Run backtests with both values
    beta_values = [0.6, 1.20]
    results = {}
    
    for beta in beta_values:
        try:
            result = run_backtest_with_beta_coef(
                cfg.copy(),
                df_exec,
                topix_beta_coef=beta,
                start_date="2015-01-05",
                end_date="latest",
                slippage_bps=5.0,
            )
            results[beta] = result
        except Exception as e:
            logger.error(f"Backtest failed for topix_beta_coef={beta}: {e}")
            import traceback
            traceback.print_exc()

    # Compare results
    logger.info("\n" + "="*80)
    logger.info("COMPARISON RESULTS: topix_beta_coef = 0.6 vs 1.20")
    logger.info("="*80)
    
    comparison_data = []
    for beta, result in results.items():
        m = result["metrics"]
        comparison_data.append({
            "topix_beta_coef": beta,
            "AR (%)": m.get("AR", 0.0) * 100,
            "RISK (%)": m.get("RISK", 0.0) * 100,
            "R/R": m.get("R/R", 0.0),
            "Sharpe": m.get("Sharpe", 0.0),
            "MDD (%)": m.get("MDD", 0.0) * 100,
            "Total Return (%)": m.get("Total Return", 0.0) * 100,
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    logger.info("\n" + comparison_df.to_string(index=False))
    
    # Determine which is better based on Sharpe ratio
    if len(results) == 2:
        sharpe_0_6 = results[0.6]["metrics"].get("Sharpe", 0.0)
        sharpe_1_20 = results[1.20]["metrics"].get("Sharpe", 0.0)
        
        logger.info("\n" + "="*80)
        if sharpe_0_6 > sharpe_1_20:
            logger.info(f"RECOMMENDATION: topix_beta_coef = 0.6 is superior")
            logger.info(f"  Sharpe ratio: {sharpe_0_6:.4f} vs {sharpe_1_20:.4f}")
            logger.info(f"  Difference: {sharpe_0_6 - sharpe_1_20:.4f}")
        elif sharpe_1_20 > sharpe_0_6:
            logger.info(f"RECOMMENDATION: topix_beta_coef = 1.20 is superior")
            logger.info(f"  Sharpe ratio: {sharpe_1_20:.4f} vs {sharpe_0_6:.4f}")
            logger.info(f"  Difference: {sharpe_1_20 - sharpe_0_6:.4f}")
        else:
            logger.info(f"RECOMMENDATION: Both values have equal Sharpe ratio ({sharpe_0_6:.4f})")
        logger.info("="*80)
        
        # Also compare AR and MDD
        ar_0_6 = results[0.6]["metrics"].get("AR", 0.0)
        ar_1_20 = results[1.20]["metrics"].get("AR", 0.0)
        mdd_0_6 = results[0.6]["metrics"].get("MDD", 0.0)
        mdd_1_20 = results[1.20]["metrics"].get("MDD", 0.0)
        
        logger.info(f"\nAR comparison: {ar_0_6*100:.2f}% vs {ar_1_20*100:.2f}% (diff: {(ar_0_6-ar_1_20)*100:.2f}%)")
        logger.info(f"MDD comparison: {mdd_0_6*100:.2f}% vs {mdd_1_20*100:.2f}% (diff: {(mdd_0_6-mdd_1_20)*100:.2f}%)")

    # Save results to CSV
    out_dir = ROOT / "results" / "topix_beta_coef_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    comparison_df.to_csv(out_dir / "comparison_summary.csv", index=False)
    
    # Save detailed results
    for beta, result in results.items():
        result["daily_returns"].to_csv(out_dir / f"daily_returns_beta_{beta}.csv", header=["return"])
        result["equity_curve"].to_csv(out_dir / f"equity_curve_beta_{beta}.csv", header=["equity"])
        result["drawdown"].to_csv(out_dir / f"drawdown_beta_{beta}.csv", header=["drawdown"])
        result["weights"].to_csv(out_dir / f"weights_beta_{beta}.csv")
    
    logger.info(f"\nResults saved to: {out_dir}")


if __name__ == "__main__":
    main()
