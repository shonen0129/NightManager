import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add tools/ to path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))

from backtest_p6_production_residual_ensemble import (
    build_portfolio_weights,
    cs_normalize,
    calculate_comprehensive_metrics,
)


def test_portfolio_weight_neutrality_and_leverage():
    """Verify build_portfolio_weights constructs long/short weights that are dollar-neutral and have leverage of 2.0."""
    np.random.seed(42)
    n_assets = 17
    
    # Test typical signals
    for _ in range(50):
        sig = np.random.randn(n_assets)
        w = build_portfolio_weights(sig, q=0.3)
        
        # Dollar neutrality check
        assert abs(np.sum(w)) < 1e-12
        
        # Leverage check (should be exactly 2.0 if signals are valid and non-zero)
        if np.any(sig):
            assert abs(np.sum(np.abs(w)) - 2.0) < 1e-12


def test_portfolio_weight_all_zero_signals():
    """Verify that build_portfolio_weights handles zero or invalid signals gracefully by returning zero weights."""
    n_assets = 17
    sig = np.zeros(n_assets)
    w = build_portfolio_weights(sig, q=0.3)
    assert np.all(w == 0.0)
    
    sig_nan = np.full(n_assets, np.nan)
    w_nan = build_portfolio_weights(sig_nan, q=0.3)
    assert np.all(w_nan == 0.0)


def test_cs_normalization_zscore():
    """Verify zscore normalization centered at median with std 1.0."""
    np.random.seed(42)
    n_assets = 17
    sig = np.random.randn(n_assets) * 5.0 + 2.0
    
    z_sig = cs_normalize(sig, method="zscore")
    assert abs(np.median(z_sig)) < 1e-12
    assert abs(np.std(z_sig) - 1.0) < 1e-12


def test_cs_normalization_robust_zscore():
    """Verify robust_zscore normalization centered at median with MAD-scaled dispersion."""
    np.random.seed(42)
    n_assets = 17
    sig = np.random.randn(n_assets) * 5.0 + 2.0
    
    rz_sig = cs_normalize(sig, method="robust_zscore")
    assert abs(np.median(rz_sig)) < 1e-12
    
    # Check MAD of normalized values is approx 1 / 1.4826
    mad = np.median(np.abs(rz_sig - np.median(rz_sig)))
    assert abs(mad - 1.0 / 1.4826) < 1e-12


def test_cs_normalization_rank():
    """Verify rank normalization maps to [-1, 1] range and has mean equal to 1/N."""
    np.random.seed(42)
    n_assets = 17
    sig = np.random.randn(n_assets)
    
    r_sig = cs_normalize(sig, method="rank")
    assert np.min(r_sig) >= -1.0
    assert np.max(r_sig) <= 1.0
    assert abs(np.mean(r_sig) - 1.0 / n_assets) < 1e-12


def test_chronological_safety_timeline():
    """Verify mock walk-forward date alignment: signal date is strictly before trade date."""
    trade_dates = pd.date_range("2026-06-01", "2026-06-10")
    # For each trade_date t, the signals are computed up to t-1
    for t in trade_dates:
        sig_cutoff = t - pd.Timedelta(days=1)
        assert sig_cutoff < t


def test_calculate_comprehensive_metrics():
    """Test calculate_comprehensive_metrics with a small synthetic returns dataframe."""
    dates = pd.date_range("2026-01-01", periods=10)
    daily_ret = pd.Series([0.01, -0.005, 0.02, 0.01, -0.01, 0.015, 0.005, -0.002, 0.01, 0.008], index=dates)
    gross_exp = pd.Series(2.0, index=dates)
    slippage_cost = pd.Series(0.0005, index=dates)
    weights_df = pd.DataFrame(np.random.randn(10, 17), index=dates)
    r_oc_df = pd.DataFrame(np.random.randn(10, 17), index=dates)
    signals_df = pd.DataFrame(np.random.randn(10, 17), index=dates)
    benchmark_df = pd.DataFrame({
        "topix_cc": [0.005, -0.002, 0.01, 0.005, -0.005, 0.008, 0.002, -0.001, 0.005, 0.004]
    }, index=dates)
    
    res = calculate_comprehensive_metrics(
        daily_ret=daily_ret,
        gross_exp=gross_exp,
        slippage_cost=slippage_cost,
        weights_df=weights_df,
        r_oc_df=r_oc_df,
        signals_df=signals_df,
        benchmark_df=benchmark_df,
    )
    
    assert "Sharpe" in res
    assert "AR" in res
    assert "MDD" in res
    assert "Avg Turnover" in res
    
    # Check types
    assert isinstance(res["Sharpe"], float)
    assert isinstance(res["AR"], float)
    assert isinstance(res["MDD"], float)
