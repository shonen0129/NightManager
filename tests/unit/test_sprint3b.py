"""Unit tests for Sprint 3-B Asset-specific Hinge Interaction features and models."""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from research.features.hinge_interactions import (
    compute_within_date_cs_std,
    build_macro_hinge_x_asset_beta,
    build_sector_hinge_x_sector_exposure,
    build_regime_hinge_x_base_signal,
    build_gap_asset_specific_hinge,
    build_all_interaction_features,
)


def test_compute_within_date_cs_std():
    """Verify within-date cross-sectional standard deviation computation."""
    # Create sample long panel
    dates = pd.date_range("2026-01-01", periods=5)
    tickers = ["1306.T", "1610.T", "1613.T"]
    
    records = []
    # feature 1 has variation across tickers
    # feature 2 has no variation across tickers
    for dt in dates:
        for val, tk in enumerate(tickers):
            records.append({
                "date": dt,
                "ticker": tk,
                "feat_var": float(val),
                "feat_const": 1.0,
            })
            
    df = pd.DataFrame(records)
    
    res = compute_within_date_cs_std(
        df, ["feat_var", "feat_const"], date_col="date", ticker_col="ticker"
    )
    
    assert not res.empty
    assert len(res) == 2
    
    var_row = res[res["feature"] == "feat_var"].iloc[0]
    const_row = res[res["feature"] == "feat_const"].iloc[0]
    
    assert var_row["mean_within_date_std"] > 0.0
    assert bool(var_row["use_for_model"]) is True
    
    assert const_row["mean_within_date_std"] == 0.0
    assert bool(const_row["use_for_model"]) is False


def test_build_macro_hinge_x_asset_beta():
    """Verify macro hinge x asset beta interaction feature construction."""
    dates = pd.date_range("2026-01-01", periods=10)
    tickers = ["1306.T", "1610.T"]
    
    macro_z = pd.DataFrame({
        "us10y_change": np.array([0.5, 1.2, -0.8, 1.5, -2.1, 0.1, 0.4, -0.2, 2.5, -1.1])
    }, index=dates)
    
    # Beta columns: beta_ticker_factor_wwindow
    rolling_beta = pd.DataFrame({
        "beta_1306.T_us10y_change_w120": np.ones(10) * 0.8,
        "beta_1306.T_us10y_change_w252": np.ones(10) * 0.6,
        "beta_1610.T_us10y_change_w120": np.ones(10) * 1.2,
        "beta_1610.T_us10y_change_w252": np.ones(10) * 1.0,
    }, index=dates)
    
    res = build_macro_hinge_x_asset_beta(
        macro_z_df=macro_z,
        rolling_beta_df=rolling_beta,
        tickers=tickers,
        macro_cols=["us10y_change"],
        beta_windows=[120, 252],
        thresholds=[1.0, 1.5],
        directions=["positive", "negative"],
    )
    
    assert not res.empty
    # Columns: date, ticker + 8 features (2 thresholds * 2 directions * 1 macro_col * 2 windows)
    assert len(res.columns) == 10
    
    # Check a specific value on date index 1 (value 1.2, kappa 1.0, positive direction)
    # hinge = max(0, 1.2 - 1.0) = 0.2
    # beta for 1306.T w120 is 0.8 -> interaction = 0.2 * 0.8 = 0.16
    row_match = res[(res["date"] == dates[1]) & (res["ticker"] == "1306.T")]
    assert len(row_match) == 1
    val = row_match["int_macro_beta_hinge_pos_us10y_change_k1_0_beta120"].values[0]
    assert np.isclose(val, 0.16)


def test_build_sector_hinge_x_sector_exposure():
    """Verify sector hinge x static sector map interaction feature construction."""
    dates = pd.date_range("2026-01-01", periods=5)
    tickers = ["1306.T", "1610.T"]
    
    sector_z = pd.DataFrame({
        "us_tech_return": np.array([2.2, -0.5, 1.1, -1.8, 0.2])
    }, index=dates)
    
    static_sector_map = pd.DataFrame({
        "us_tech": [0.6, 0.1]
    }, index=tickers)
    
    res = build_sector_hinge_x_sector_exposure(
        sector_z_df=sector_z,
        static_sector_map=static_sector_map,
        rolling_beta_df=None,
        tickers=tickers,
        sector_cols=["us_tech_return"],
        beta_windows=[120],
        thresholds=[1.0],
        directions=["positive"],
        use_rolling_beta=False,
        use_static=True,
    )
    
    assert not res.empty
    # Columns: date, ticker + 1 feature (pos_us_tech_return_k1_0_static)
    assert len(res.columns) == 3
    
    # On date 0: us_tech_return = 2.2, hinge_pos_k1_0 = 2.2 - 1.0 = 1.2
    # Ticker 1306.T static = 0.6 -> interaction = 1.2 * 0.6 = 0.72
    row_match = res[(res["date"] == dates[0]) & (res["ticker"] == "1306.T")]
    val = row_match["int_sector_static_hinge_pos_us_tech_return_k1_0"].values[0]
    assert np.isclose(val, 0.72)


def test_build_regime_hinge_x_base_signal():
    """Verify regime hinge x base signal interaction feature construction."""
    dates = pd.date_range("2026-01-01", periods=5)
    tickers = ["1306.T", "1610.T"]
    
    regime_z = pd.DataFrame({
        "vix_return": np.array([1.8, 0.2, -1.5, 0.9, -0.1])
    }, index=dates)
    
    signal_panel = pd.DataFrame({
        "1306.T": [0.001, -0.002, 0.003, -0.004, 0.005],
        "1610.T": [-0.001, 0.002, -0.003, 0.004, -0.005],
    }, index=dates)
    
    res = build_regime_hinge_x_base_signal(
        regime_z_df=regime_z,
        signal_panel=signal_panel,
        tickers=tickers,
        regime_cols=["vix_return"],
        thresholds=[1.0],
        directions=["positive"],
    )
    
    assert not res.empty
    assert len(res.columns) == 3
    
    # On date 0: vix_return = 1.8, hinge_pos_k1_0 = 1.8 - 1.0 = 0.8
    # Ticker 1306.T signal = 0.001 -> interaction = 0.8 * 0.001 = 0.0008
    row_match = res[(res["date"] == dates[0]) & (res["ticker"] == "1306.T")]
    val = row_match["int_regime_signal_hinge_pos_vix_return_k1_0"].values[0]
    assert np.isclose(val, 0.0008)


def test_build_gap_asset_specific_hinge():
    """Verify gap asset-specific hinge feature construction."""
    dates = pd.date_range("2026-01-01", periods=10)
    tickers = ["1306.T", "1610.T"]
    
    # Gap long panel
    records = []
    # Make jp_open_gap vary by ticker
    for dt in dates:
        records.append({"date": dt, "ticker": "1306.T", "jp_open_gap": 0.5})
        records.append({"date": dt, "ticker": "1610.T", "jp_open_gap": -0.5})
        
    gap_panel = pd.DataFrame(records)
    
    res = build_gap_asset_specific_hinge(
        gap_panel_long=gap_panel,
        tickers=tickers,
        gap_cols=["jp_open_gap"],
        zscore_window=5,
        thresholds=[1.0],
        directions=["positive"],
    )
    
    # Needs enough observations for zscore window, zscore window is 5, dates=10, so it should run.
    assert not res.empty
    assert len(res.columns) == 3


def test_build_all_interaction_features():
    """Verify merging of different interaction feature groups."""
    dates = pd.date_range("2026-01-01", periods=5)
    tickers = ["1306.T", "1610.T"]
    
    # Group A
    macro_df = pd.DataFrame([
        {"date": d, "ticker": t, "feat_a": 1.0}
        for d in dates for t in tickers
    ])
    # Group B
    sector_df = pd.DataFrame([
        {"date": d, "ticker": t, "feat_b": 2.0}
        for d in dates for t in tickers
    ])
    
    combined = build_all_interaction_features(
        macro_interactions=macro_df,
        sector_interactions=sector_df,
        regime_interactions=None,
        gap_interactions=None,
        max_raw_features=5,
    )
    
    assert not combined.empty
    assert "feat_a" in combined.columns
    assert "feat_b" in combined.columns
    assert len(combined) == 10
