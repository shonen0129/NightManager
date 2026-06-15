"""Unit tests for ML predictor and leakage checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from domain.models.ml_predictor import (
    compute_rolling_z_scores_us,
    compute_jp_volatility,
    MLRollingRunner,
)


def _build_mock_df(n_days: int = 400) -> pd.DataFrame:
    """Create a mock df_exec DataFrame with typical columns and values."""
    dates = pd.date_range("2019-01-01", periods=n_days)
    
    us_tickers = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
    jp_tickers = [f"{t}.T" for t in range(1617, 1634)]
    
    data = {}
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    for tk in us_tickers:
        data[f"us_cc_{tk}"] = np.random.normal(0.0001, 0.015, n_days)
        
    for tk in jp_tickers:
        data[f"jp_cc_{tk}"] = np.random.normal(0.0001, 0.015, n_days)
        data[f"jp_gap_{tk}"] = np.random.normal(0.00005, 0.005, n_days)
        data[f"jp_oc_{tk}"] = np.random.normal(0.00005, 0.010, n_days)
        
    df = pd.DataFrame(data, index=dates)
    return df


def test_us_standardisation_no_leakage():
    """Verify that compute_rolling_z_scores_us does not look ahead."""
    df = _build_mock_df()
    us_tickers = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
    
    # Original calculation
    res_orig = compute_rolling_z_scores_us(df, us_tickers)
    
    # Modify future rows starting from index 200
    df_mod = df.copy()
    for col in df_mod.columns:
        if col.startswith("us_cc_"):
            df_mod.loc[df_mod.index[200:], col] = 999.0
            
    res_mod = compute_rolling_z_scores_us(df_mod, us_tickers)
    
    # Assert that prior to index 200, results are identical
    pd.testing.assert_frame_equal(res_orig.iloc[:200], res_mod.iloc[:200])
    # Assert that starting from index 200, results are different
    assert not res_orig.iloc[200:].equals(res_mod.iloc[200:])


def test_volatility_calculation_no_leakage():
    """Verify that compute_jp_volatility does not look ahead."""
    df = _build_mock_df()
    jp_tickers = ["1617.T", "1618.T"]
    
    res_orig = compute_jp_volatility(df, jp_tickers)
    
    # Modify future rows starting from index 200
    df_mod = df.copy()
    for col in df_mod.columns:
        if col.startswith("jp_cc_"):
            df_mod.loc[df_mod.index[200:], col] = -99.0
            
    res_mod = compute_jp_volatility(df_mod, jp_tickers)
    
    # Assert that prior to index 200, results are identical
    pd.testing.assert_frame_equal(res_orig.iloc[:200], res_mod.iloc[:200])


def test_ml_rolling_runner_leakage():
    """Verify that future data modification doesn't affect past rolling predictions."""
    df = _build_mock_df(n_days=300)
    us_tickers = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
    jp_tickers = [f"{t}.T" for t in range(1617, 1634)]
    
    # 1. Run on original data
    runner = MLRollingRunner(
        df_exec=df,
        us_tickers=us_tickers,
        jp_tickers=jp_tickers,
        train_window=250,
        refit_interval=10,
    )
    res_orig = runner.run_rolling_predictions(start_date=df.index[260].strftime("%Y-%m-%d"))
    
    # 2. Modify future data: we will predict up to index 280, but corrupt rows from 281 onwards
    df_mod = df.copy()
    corrupt_idx = 281
    
    for col in df_mod.columns:
        df_mod.loc[df_mod.index[corrupt_idx:], col] = np.random.normal(99.0, 10.0, len(df_mod) - corrupt_idx)
        
    runner_mod = MLRollingRunner(
        df_exec=df_mod,
        us_tickers=us_tickers,
        jp_tickers=jp_tickers,
        train_window=250,
        refit_interval=10,
    )
    res_mod = runner_mod.run_rolling_predictions(start_date=df.index[260].strftime("%Y-%m-%d"))
    
    # Predictions at and before index 280 must be EXACTLY IDENTICAL
    # because row 280 only uses information from row 280 and prior (no look-ahead to 281+)
    pd.testing.assert_frame_equal(
        res_orig.loc[:df.index[280]], 
        res_mod.loc[:df.index[280]],
        rtol=1e-7,
        atol=1e-7
    )
