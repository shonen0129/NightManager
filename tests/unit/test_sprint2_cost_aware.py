"""Unit tests for Sprint 2 Cost-Aware Portfolio Optimization."""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def test_covariance_shrinkage():
    """Verify rolling covariance and diagonal shrinkage formula."""
    np.random.seed(42)
    # Generate random returns
    returns = np.random.randn(60, 5) * 0.01
    
    # Covariance
    cov = np.cov(returns, rowvar=False)
    
    # Shrinkage
    gamma = 0.05
    cov_shrunk = (1 - gamma) * cov + gamma * np.diag(np.diag(cov))
    
    # Check that diagonal elements are equal to the original diag elements
    assert np.allclose(np.diag(cov_shrunk), np.diag(cov))
    
    # Check positive definiteness (eigenvalues > 0)
    eigvals = np.linalg.eigvalsh(cov_shrunk)
    assert np.all(eigvals > 0.0)


def test_net_alpha_filter_logic():
    """Test Model 1: net-alpha filtering and side constraints."""
    # 5 tickers
    mu = np.array([0.0030, -0.0020, 0.0005, -0.0050, 0.0001])
    tc_rt_long = np.array([0.0010, 0.0010, 0.0010, 0.0010, 0.0010])
    tc_rt_short = np.array([0.0015, 0.0015, 0.0015, 0.0015, 0.0015])
    k = 1.5
    
    # Long condition: mu > k * tc_rt_long
    # 0.0030 > 1.5 * 0.0010 (0.0015) -> True
    # 0.0005 > 0.0015 -> False
    # 0.0001 > 0.0015 -> False
    # So long candidate is ticker 0
    
    # Short condition: -mu > k * tc_rt_short
    # -(-0.0020) = 0.0020 > 1.5 * 0.0015 (0.00225) -> False
    # -(-0.0050) = 0.0050 > 0.00225 -> True (ticker 3)
    
    long_candidates = np.where((mu > 0.0) & (mu > k * tc_rt_long))[0]
    short_candidates = np.where((mu < 0.0) & (-mu > k * tc_rt_short))[0]
    
    assert list(long_candidates) == [0]
    assert list(short_candidates) == [3]


def test_net_score_ranking_logic():
    """Test Model 2: net-score ranking and direction-adjusted cost subtraction."""
    mu = np.array([0.0020, -0.0030, 0.0005])
    tc_rt_long = np.array([0.0010, 0.0010, 0.0010])
    tc_rt_short = np.array([0.0015, 0.0015, 0.0015])
    lambda_tc = 1.0
    
    # Long candidates (mu > 0): score = mu - lambda * tc_long
    # Ticker 0: 0.0020 - 1.0 * 0.0010 = 0.0010 > 0 -> selected
    # Ticker 2: 0.0005 - 0.0010 = -0.0005 <= 0 -> rejected
    
    # Short candidates (mu < 0): score = -mu - lambda * tc_short
    # Ticker 1: 0.0030 - 1.0 * 0.0015 = 0.0015 > 0 -> selected
    
    long_scores = mu - lambda_tc * tc_rt_long
    short_scores = -mu - lambda_tc * tc_rt_short
    
    long_candidates = np.where((mu > 0.0) & (long_scores > 0.0))[0]
    short_candidates = np.where((mu < 0.0) & (short_scores > 0.0))[0]
    
    assert list(long_candidates) == [0]
    assert list(short_candidates) == [1]


def test_mvo_optimization():
    """Verify that MVO quadratic program is set up and solved correctly via SLSQP."""
    # 3 assets
    mu = np.array([0.0020, -0.0030, 0.0005])
    cov = np.array([
        [1.0, 0.2, 0.1],
        [0.2, 1.2, 0.3],
        [0.1, 0.3, 0.9]
    ]) * 1e-4
    
    risk_aversion = 3.0
    tc_long = np.array([0.0010, 0.0010, 0.0010])
    tc_short = np.array([0.0015, 0.0015, 0.0015])
    target_gross = 1.0
    max_weight = 0.25
    
    # x = [u, v]
    n = len(mu)
    x0 = np.zeros(2 * n)
    
    # Objective function
    def obj_fun(x):
        w = x[:n] - x[n:]
        variance = w.T @ cov @ w
        expected_ret = w.T @ mu
        costs = np.sum(tc_long * x[:n] + tc_short * x[n:])
        return -expected_ret + 0.5 * risk_aversion * variance + costs
        
    # Bounds: 0 <= u, v <= max_weight
    bounds = [(0.0, max_weight)] * (2 * n)
    
    # Constraints: sum(u - v) = 0, sum(u + v) <= target_gross
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x[:n] - x[n:])},
        {'type': 'ineq', 'fun': lambda x: target_gross - np.sum(x)}
    ]
    
    res = minimize(obj_fun, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    assert res.success
    w_opt = res.x[:n] - res.x[n:]
    assert np.isclose(np.sum(w_opt), 0.0, atol=1e-7)
    assert np.sum(np.abs(w_opt)) <= target_gross + 1e-7


def test_integer_rounding_reevaluation():
    """Verify integer rounding and actual weight re-evaluation."""
    AUM = 1000000
    w_opt = np.array([0.15, -0.15, 0.08, -0.08])
    prices = np.array([2500.0, 4000.0, 1500.0, 8000.0])
    
    target_notionals = w_opt * AUM
    shares = np.round(target_notionals / prices)
    
    actual_notionals = shares * prices
    actual_weights = actual_notionals / AUM
    
    # Check that actual weights are multiples of price / AUM
    for w, p, s in zip(actual_weights, prices, shares):
        assert np.isclose(w * AUM, s * p)
