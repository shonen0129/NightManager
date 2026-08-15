"""Unit tests for Convex Portfolio Optimizer."""

import numpy as np

from leadlag.core.convex_optimizer import (
    ConvexOptimizerConfig,
    ensure_psd,
    optimize_portfolio_convex,
)


def test_ensure_psd():
    """Test that ensure_psd produces symmetric PSD matrix with min eigenvalue >= floor."""
    # Non-PSD matrix with negative eigenvalue
    A = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues: 3.0, -1.0
    psd_A = ensure_psd(A, min_eigenvalue=1e-6)

    # Must be symmetric
    assert np.allclose(psd_A, psd_A.T, atol=1e-12)
    # Eigenvalues must be >= 1e-6
    eigvals = np.linalg.eigvalsh(psd_A)
    assert np.all(eigvals >= 1e-6 - 1e-12)


def test_optimize_portfolio_convex_basic():
    """Test basic market-neutral convex portfolio optimization."""
    n_j = 17
    np.random.seed(42)
    mu_gap = np.random.randn(n_j) * 0.01  # 1% standard deviation

    # Random positive definite covariance matrix
    L = np.random.randn(n_j, n_j) * 0.01
    omega_gap = L @ L.T + np.eye(n_j) * 1e-4

    config = ConvexOptimizerConfig(
        lambda_risk=5.0,
        cost_bps=5.0,
        gross_target=2.0,
        max_single_weight=0.25,
    )

    res = optimize_portfolio_convex(
        mu_gap=mu_gap,
        omega_gap=omega_gap,
        config=config,
        gross_multiplier=1.0,
    )

    assert res.converged
    assert abs(res.net_exposure) < 1e-9  # Exact net zero
    assert res.gross_exposure <= 2.0 + 1e-6  # Gross limit
    assert np.all(np.abs(res.weights) <= 0.25 + 1e-6)  # Single weight limit
    assert res.ex_ante_return > 0.0  # Positive expected alpha
    assert res.ex_ante_ir > 0.0


def test_optimize_portfolio_convex_zero_gross():
    """Test zero gross multiplier yields zero weights."""
    n_j = 17
    mu_gap = np.ones(n_j) * 0.01
    omega_gap = np.eye(n_j) * 1e-4

    res = optimize_portfolio_convex(
        mu_gap=mu_gap,
        omega_gap=omega_gap,
        gross_multiplier=0.0,
    )

    assert res.converged
    assert np.allclose(res.weights, 0.0)
    assert res.gross_exposure == 0.0
    assert res.net_exposure == 0.0


def test_optimize_portfolio_convex_turnover_penalty():
    """Test that higher turnover penalty reduces position changes from w_prev."""
    n_j = 17
    np.random.seed(123)
    mu_gap = np.random.randn(n_j) * 0.01
    omega_gap = np.eye(n_j) * 1e-4

    # Previous weights
    w_prev = np.zeros(n_j)
    w_prev[0] = 0.2
    w_prev[1] = -0.2

    # Low turnover penalty
    res_low = optimize_portfolio_convex(
        mu_gap=mu_gap,
        omega_gap=omega_gap,
        w_prev=w_prev,
        config=ConvexOptimizerConfig(turnover_penalty=0.0, cost_bps=0.0),
    )

    # High turnover penalty
    res_high = optimize_portfolio_convex(
        mu_gap=mu_gap,
        omega_gap=omega_gap,
        w_prev=w_prev,
        config=ConvexOptimizerConfig(turnover_penalty=0.1, cost_bps=50.0),
    )

    assert res_high.turnover < res_low.turnover
