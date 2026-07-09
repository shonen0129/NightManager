#!/usr/bin/env python
"""Generate diagnostics weights file for Step 1 diagnostics.

This script runs a simple backtest to generate the daily_positions_Residual-BLPX_only.csv
file that is required by compute_structured_prediction_covariance.py.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Add src/ to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.fetcher import download_data
from leadlag.data.preprocessor import preprocess_data
from leadlag.data.tickers import JP_TICKERS, TOPIX_TICKER
from leadlag.models.sector_relative_ensemble_blp_enhanced import SectorRelativeEnsembleBLPEnhancedModel
from leadlag.models.sre import compute_jp_target_returns


def normalize_cross_sectional(x: np.ndarray) -> np.ndarray:
    """Normalize cross-sectionally (rank-based z-score)."""
    ranks = np.argsort(np.argsort(x, axis=1), axis=1)
    n = x.shape[1]
    z = (ranks - (n - 1) / 2) / np.sqrt(n * (n + 1) / 12)
    return z

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def run_backtest_fast(
    signal_vals: np.ndarray,
    y_jp_target_vals: np.ndarray,
    q: float = 0.3,
    slippage_bps: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized portfolio simulation."""
    T, n_j = signal_vals.shape
    weights = np.zeros((T, n_j))
    num_positions = int(np.round(n_j * q))
    
    sort_order = np.argsort(signal_vals, axis=1)
    short_idx = sort_order[:, :num_positions]
    long_idx = sort_order[:, -num_positions:]
    
    medians = np.take_along_axis(signal_vals, sort_order[:, 8:9], axis=1)
    s_centered = signal_vals - medians
    row_indices = np.arange(T)[:, None]
    
    long_raw = s_centered[row_indices, long_idx]
    long_raw = np.maximum(long_raw, 1e-8)
    long_denom = np.sum(long_raw, axis=1, keepdims=True)
    long_denom_safe = np.where(long_denom > 0, long_denom, 1.0)
    weights[row_indices, long_idx] = np.where(long_denom > 0, long_raw / long_denom_safe, 0.0)
    
    short_raw = -s_centered[row_indices, short_idx]
    short_raw = np.maximum(short_raw, 1e-8)
    short_denom = np.sum(short_raw, axis=1, keepdims=True)
    short_denom_safe = np.where(short_denom > 0, short_denom, 1.0)
    weights[row_indices, short_idx] = np.where(short_denom > 0, -(short_raw / short_denom_safe), 0.0)
    
    return weights


def main():
    parser = argparse.ArgumentParser(description="Generate diagnostics weights file")
    parser.add_argument("--config", default="configs/production/production.yaml", help="Path to config YAML")
    parser.add_argument("--start-date", default="2015-01-05", help="Backtest start date")
    parser.add_argument("--end-date", default="latest", help="Backtest end date")
    parser.add_argument("--output-dir", default="live/pipeline_data/diagnostics_weights", help="Output directory")
    args = parser.parse_args()

    config_path = ROOT / args.config
    logger.info("Loading config from %s", config_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    beta_window = cfg.get("residualization", {}).get("beta_window", 60)
    beta_ewma_halflife = cfg.get("residualization", {}).get("beta_ewma_halflife")
    beta_shrinkage = cfg.get("residualization", {}).get("beta_shrinkage", 0.0)
    beta_winsor_sigma = cfg.get("residualization", {}).get("beta_winsor_sigma")

    logger.info("[1/3] Downloading/loading market data...")
    raw_data = download_data(beta_window=beta_window)
    logger.info("[2/3] Preprocessing market data...")
    df_exec = preprocess_data(raw_data, beta_window=beta_window, 
                             beta_ewma_halflife=beta_ewma_halflife,
                             beta_shrinkage=beta_shrinkage,
                             beta_winsor_sigma=beta_winsor_sigma)

    # Compute TOPIX returns
    topix_close = raw_data["jp_close"][TOPIX_TICKER].copy()
    topix_open = raw_data["jp_open"][TOPIX_TICKER].copy()
    topix_close.index = pd.to_datetime(topix_close.index).tz_localize(None).normalize()
    topix_open.index = pd.to_datetime(topix_open.index).tz_localize(None).normalize()
    r_topix_oc = topix_close / topix_open - 1.0
    df_exec["topix_oc_return"] = r_topix_oc.reindex(df_exec.index).values
    df_exec["topix_cc_trade"] = (1.0 + df_exec["topix_night_return"]) * (1.0 + df_exec["topix_oc_return"]) - 1.0

    # Filter dates
    sim_dates = df_exec.index
    sim_dates_slice = sim_dates[sim_dates >= args.start_date]
    if args.end_date != "latest":
        sim_dates_slice = sim_dates_slice[sim_dates_slice <= args.end_date]
    
    logger.info("Backtest window: %s to %s (%d days)", 
                sim_dates_slice[0].strftime('%Y-%m-%d'), 
                sim_dates_slice[-1].strftime('%Y-%m-%d'), 
                len(sim_dates_slice))

    logger.info("[3/3] Building model and generating signals...")
    model = SectorRelativeEnsembleBLPEnhancedModel(cfg)
    
    # Generate signals using predict_signals (following validate_production_residual_blpx.py pattern)
    blpx_pred = model.predict_signals(df_exec)
    
    # Extract residual BLPX signals and normalize
    z_residual_blpx = normalize_cross_sectional(blpx_pred["residual_blpx_signals"].loc[sim_dates_slice].values)
    
    # Extract realized target returns (9:10-to-close)
    y_jp_target_slice = df_exec.loc[sim_dates_slice, [f"jp_oc_{tk}" for tk in JP_TICKERS]].rename(columns=lambda c: c.replace("jp_oc_", ""))
    y_jp_target_vals = y_jp_target_slice.values
    
    # Run backtest to get weights
    logger.info("Running backtest simulation...")
    weights = run_backtest_fast(z_residual_blpx, y_jp_target_vals, q=0.3, slippage_bps=5.0)
    
    # Save weights
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    weights_df = pd.DataFrame(weights, index=sim_dates_slice, columns=JP_TICKERS)
    weights_file = out_dir / "daily_positions_Residual-BLPX_only.csv"
    weights_df.to_csv(weights_file)
    
    logger.info("Weights file saved to: %s", weights_file)
    logger.info("Total dates: %d", len(weights_df))
    logger.info("Date range: %s to %s", 
                weights_df.index[0].strftime('%Y-%m-%d'),
                weights_df.index[-1].strftime('%Y-%m-%d'))


if __name__ == "__main__":
    main()
