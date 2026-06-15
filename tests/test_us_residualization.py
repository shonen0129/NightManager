"""Unit tests for SRE with US Residualization (SRE-USR) Model.

Verifies rolling beta shift (no lookahead), gamma parameter sensitivity behavior,
ensemble weights logic, and data sanity checks.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Add src/ to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.models.sre import (
    SectorRelativeEnsembleModel,
    compute_us_residualized_returns,
)


def test_rolling_beta_shift_no_lookahead():
    """Verify that the rolling beta calculation does not use lookahead data.

    Specifically, check that the residual for day t only uses beta estimated from [t-60, ..., t-1].
    """
    np.random.seed(42)
    T = 100
    n_u = 5
    beta_window = 30

    us_returns = np.random.randn(T, n_u)
    spy_returns = np.random.randn(T)

    # Compute residualized returns
    r_us_adj = compute_us_residualized_returns(
        us_returns, spy_returns, beta_window=beta_window, gamma=1.0
    )

    # Manually compute beta and residual for t = 50
    t = 50
    window_us = us_returns[t - beta_window : t]
    window_mkt = spy_returns[t - beta_window : t]

    var_mkt = np.var(window_mkt, ddof=1)
    mean_mkt = np.mean(window_mkt)
    mean_us = np.mean(window_us, axis=0)
    cov = np.sum((window_us - mean_us) * (window_mkt - mean_mkt)[:, np.newaxis], axis=0) / (
        beta_window - 1
    )
    expected_beta = cov / var_mkt

    expected_residual = us_returns[t] - expected_beta * spy_returns[t]

    assert np.allclose(r_us_adj[t], expected_residual, atol=1e-10)


def test_gamma_zero_returns_raw_input():
    """Verify that when gamma = 0, the residualized returns are exactly equal to the raw input returns."""
    np.random.seed(42)
    T = 50
    n_u = 3
    beta_window = 10

    us_returns = np.random.randn(T, n_u)
    spy_returns = np.random.randn(T)

    r_us_adj = compute_us_residualized_returns(
        us_returns, spy_returns, beta_window=beta_window, gamma=0.0
    )

    assert np.allclose(r_us_adj, us_returns, atol=1e-10)


def test_gamma_one_residual_formula():
    """Verify that when gamma = 1.0, the calculation is exactly equal to raw returns minus beta * market_returns."""
    np.random.seed(100)
    T = 60
    n_u = 2
    beta_window = 20

    us_returns = np.random.randn(T, n_u)
    spy_returns = np.random.randn(T)

    r_us_adj = compute_us_residualized_returns(
        us_returns, spy_returns, beta_window=beta_window, gamma=1.0
    )

    # Calculate beta for t = 40
    t = 40
    window_us = us_returns[t - beta_window : t]
    window_mkt = spy_returns[t - beta_window : t]
    cov = np.cov(window_us, window_mkt, rowvar=False)[:-1, -1]
    var_mkt = np.var(window_mkt, ddof=1)
    beta = cov / var_mkt

    expected_val = us_returns[t] - beta * spy_returns[t]
    assert np.allclose(r_us_adj[t], expected_val, atol=1e-10)


def test_ensemble_weights_sum_to_one():
    """Verify that model ensemble weights sum to 1.0."""
    cfg = {
        "model": {"name": "sector_relative_ensemble"},
        "p0_weight": 0.40,
        "p3_weight": 0.40,
        "p4_weight": 0.20,
    }
    model = SectorRelativeEnsembleModel(cfg)
    assert abs(model.p0_weight + model.p3_weight + model.p4_weight - 1.0) < 1e-6


def test_no_nan_inf_returned():
    """Verify that the residualization process does not produce or pass through any NaN or inf values."""
    T = 80
    n_u = 4
    beta_window = 15

    us_returns = np.random.randn(T, n_u)
    # Put NaNs and infs in us_returns
    us_returns[5, 0] = np.nan
    us_returns[10, 1] = np.inf
    us_returns[20, 2] = -np.inf

    spy_returns = np.random.randn(T)
    # Put extremely small variance benchmark section
    spy_returns[30:50] = 0.0000000000001

    r_us_adj = compute_us_residualized_returns(
        us_returns, spy_returns, beta_window=beta_window, gamma=0.5
    )

    assert not np.isnan(r_us_adj).any()
    assert not np.isinf(r_us_adj).any()
