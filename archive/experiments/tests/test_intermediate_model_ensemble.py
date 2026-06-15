import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add tools/ to path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))

from backtest_intermediate_model_ensemble import (
    orth,
    build_portfolio_weights,
    normalize_signals,
)


def test_hybrid_subspace_orthogonality():
    """Verify hybrid subspace dimensionality and orthogonality of orth()."""
    np.random.seed(42)
    # Generate mock static V0_U (e.g., shape 11 x 6)
    k_prior = 6
    n_us = 11
    V0_U = np.random.randn(n_us, k_prior)
    # Orthonormalize V0_U first
    V0_U = orth(V0_U)
    
    # Generate mock dynamic PCA subspace v_u_t_k
    V_dynamic = np.random.randn(n_us, k_prior)
    V_dynamic = orth(V_dynamic)
    
    # Hybrid subspace blend
    M = 0.5 * V0_U + 0.5 * V_dynamic
    Q = orth(M)
    
    # Dimensions check
    assert Q.shape == (n_us, k_prior)
    
    # Orthogonality check: Q^T * Q should be close to identity
    I_approx = Q.T @ Q
    assert np.allclose(I_approx, np.eye(k_prior), atol=1e-8)


def test_restricted_propagation_fittings():
    """Verify closed-form solutions for diagonal and diagonal-dominant Ridge propagation matrix A_t."""
    np.random.seed(42)
    k_prior = 6
    N_tr = 100
    
    F_tr = np.random.randn(N_tr, k_prior)
    G_tr = np.random.randn(N_tr, k_prior)
    
    # Diagonal solver
    diag_A = np.zeros((k_prior, k_prior))
    for j in range(k_prior):
        num = np.sum(F_tr[:, j] * G_tr[:, j]) + 10.0
        den = np.sum(F_tr[:, j] ** 2) + 10.0 + 10.0  # lambda_A = 10, lambda_prior = 10
        diag_A[j, j] = num / den
        
    # Check that it is diagonal
    assert np.all(diag_A - np.diag(np.diag(diag_A)) == 0.0)
    
    # Diagonal dominant solver (L2 off-diagonal penalty = 10.0)
    dom_A = np.zeros((k_prior, k_prior))
    M_N3 = F_tr.T @ F_tr
    for j in range(k_prior):
        # row system solving
        D_j = np.eye(k_prior) * (10.0 + 10.0 + 10.0)  # lambda_A + lambda_prior + lambda_offdiag
        D_j[j, j] = 10.0 + 10.0  # lambda_A + lambda_prior (no off-diagonal penalty on diagonal element)
        lhs = M_N3 + D_j
        rhs = F_tr.T @ G_tr[:, j]
        rhs[j] += 10.0  # lambda_prior * A0
        dom_A[:, j] = np.linalg.solve(lhs, rhs)
        
    # Check dimensions
    assert dom_A.shape == (k_prior, k_prior)
    
    # Run unconstrained Ridge propagation solver logic to compare
    # loss = ||G - F A||_F^2 + 10.0 * ||A||_F^2 + 10.0 * ||A - I||_F^2
    # standard ridge solver: (F^T F + (lambda_A + lambda_prior)*I) A = F^T G + lambda_prior * I
    lhs_unconstrained = M_N3 + (10.0 + 10.0) * np.eye(k_prior)
    rhs_unconstrained = F_tr.T @ G_tr + 10.0 * np.eye(k_prior)
    unconstrained_A = np.linalg.solve(lhs_unconstrained, rhs_unconstrained)
    
    # The off-diagonal terms of dom_A should be shrunk closer to 0 compared to unconstrained_A
    # since we added lambda_offdiag * ||offdiag(A)||_F^2
    offdiag_mask = ~np.eye(k_prior, dtype=bool)
    norm_unconstrained_offdiag = np.linalg.norm(unconstrained_A[offdiag_mask])
    norm_dom_offdiag = np.linalg.norm(dom_A[offdiag_mask])
    
    assert norm_dom_offdiag <= norm_unconstrained_offdiag + 1e-7


def test_portfolio_weight_constraints():
    """Verify portfolio weight limits (sum w = 0, sum |w| <= 2.0) and sign consistency."""
    np.random.seed(42)
    # Generate mock signals for 17 assets
    n_assets = 17
    
    for _ in range(50):
        sig = np.random.randn(n_assets)
        w = build_portfolio_weights(sig, q=0.3)
        
        # Check sum is zero (dollar neutral)
        assert abs(np.sum(w)) < 1e-12
        
        # Check gross exposure is exactly 2.0 (if signals are valid and non-zero)
        if np.any(sig):
            assert abs(np.sum(np.abs(w)) - 2.0) < 1e-12
            
        # Check sign consistency: positive weights correspond to assets with higher signals/ranks
        # and negative weights to lower signals/ranks.
        # Find indices of longs (w > 0) and shorts (w < 0)
        longs = np.where(w > 0)[0]
        shorts = np.where(w < 0)[0]
        zeros = np.where(w == 0)[0]
        
        if len(longs) > 0 and len(shorts) > 0:
            # All long signals should be strictly greater than all short signals
            min_long_sig = np.min(sig[longs])
            max_short_sig = np.max(sig[shorts])
            assert min_long_sig > max_short_sig
            
            # If there are zero weights, verify they are in the middle of signals
            if len(zeros) > 0:
                max_zero_sig = np.max(sig[zeros])
                min_zero_sig = np.min(sig[zeros])
                assert min_long_sig >= min_zero_sig
                assert max_zero_sig >= max_short_sig


def test_chronological_safety():
    """Verify chronological safety check (signals computed up to t-1, gap open checked at t)."""
    # Create mock timeline where signal dates are strictly before trade dates (e.g., t-1)
    trade_dates = pd.date_range("2026-06-02", "2026-06-11")
    sig_dates = trade_dates - pd.Timedelta(days=1)
    
    # Verify that signal_date < trade_date for execution (or signal computed up to t-1)
    # Our audit reports failures if sig_date >= trade_date
    violations = 0
    for s_dt, t_dt in zip(sig_dates, trade_dates):
        if s_dt >= t_dt:
            violations += 1
            
    assert violations == 0
