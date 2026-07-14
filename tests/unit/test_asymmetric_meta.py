"""Unit tests for Asymmetric Propagation and Meta-Learning features in SectorRelativeEnsembleBLPEnhancedModel."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
from leadlag.models.sector_relative_ensemble_blp_enhanced import SectorRelativeEnsembleBLPEnhancedModel
from leadlag.data.tickers import JP_TICKERS, US_TICKERS

@pytest.fixture
def base_config() -> dict:
    return {
        "model": {"name": "sector_relative_ensemble_blp_enhanced"},
        "portfolio": {"long_short_frac": 0.3, "weight_mode": "signal"},
        "ensemble": {
            "raw_pca_weight": 0.4,
            "residual_pca_weight": 0.4,
            "raw_blpx_weight": 0.1,
            "residual_blpx_weight": 0.1,
            "normalization": "zscore",
        },
        "costs": {"slippage_bps_per_side": 5.0},
        "blp_window": 252,
        "blp_ewma_halflife": 45,
        "alpha_xx": 0.5,
        "alpha_yx": 0.25,
        "alpha_yy": 0.5,
        "rho": 0.03,
        "rank": "full",
        "lambda_pca": 0.1,
        "lambda_sector": 0.1,
        "beta_conf": 0.25,
        "winsor_sigma": 4.0,
        "exec_adjustment": "none",
        # Asymmetric propagation & meta parameters
        "asymmetry_delta": 0.2,
        "asymmetry_mode": "scalar",
        "gap_open_coef_neg": 0.6,
        "topix_beta_coef_neg": 0.6,
        "asymmetry_post_gap_delta": 0.1,
        "asymmetry_post_gap_mode": "signal_split",
        "meta_learning_enabled": False,
        "meta_learning_model_type": "logistic_regression",
        "meta_learning_train_window": 100,
        "meta_learning_smooth_factor": 1.0,
    }

def test_asymmetric_propagation_init(base_config):
    """Verify that asymmetry parameters are correctly read from configuration."""
    model = SectorRelativeEnsembleBLPEnhancedModel(base_config)
    assert model.asymmetry_delta == 0.2
    assert model.asymmetry_mode == "scalar"
    assert model.gap_open_coef_neg == 0.6
    assert model.topix_beta_coef_neg == 0.6
    assert model.asymmetry_post_gap_delta == 0.1
    assert model.asymmetry_post_gap_mode == "signal_split"
    assert model.meta_enabled is False

def test_estimate_asymmetric_covariance(base_config):
    """Verify that _estimate_asymmetric_covariance splits returns on US factor and uses fallback."""
    model = SectorRelativeEnsembleBLPEnhancedModel(base_config)
    
    n_u = len(US_TICKERS)
    n_j = len(JP_TICKERS)
    total_assets = n_u + n_j
    
    # 1. Fallback case: insufficient samples (e.g. 20 samples)
    window_returns_small = np.random.randn(20, total_assets)
    corr_full = np.eye(total_assets)
    C_YX_pos, C_YX_neg, C_XX, C_YY = model._estimate_asymmetric_covariance(
        window_returns_small, corr_full
    )
    # With fallback, positive and negative should be identical to full corr's C_YX
    assert np.allclose(C_YX_pos, corr_full[n_u:, :n_u])
    assert np.allclose(C_YX_neg, corr_full[n_u:, :n_u])
    
    # 2. Sufficient samples: verify split logic
    # Create window returns with 40 positive US factor days and 40 negative US factor days
    pos_samples = np.ones((40, total_assets)) * 0.05
    neg_samples = np.ones((40, total_assets)) * -0.05
    window_returns = np.vstack([pos_samples, neg_samples])
    
    C_YX_pos, C_YX_neg, C_XX, C_YY = model._estimate_asymmetric_covariance(
        window_returns, corr_full
    )
    # Columns and rows shape checks
    assert C_YX_pos.shape == (n_j, n_u)
    assert C_YX_neg.shape == (n_j, n_u)
    assert C_XX.shape == (n_u, n_u)
    assert C_YY.shape == (n_j, n_j)

def test_solve_asymmetric_blp(base_config):
    """Verify that _solve_asymmetric_blp solves coefficients correctly for pos/neg regimes."""
    model = SectorRelativeEnsembleBLPEnhancedModel(base_config)
    n_u = model.n_u
    n_j = model.n_j
    
    C_YX_pos = np.random.randn(n_j, n_u) * 0.1
    C_YX_neg = np.random.randn(n_j, n_u) * 0.1
    C_XX = np.eye(n_u)
    C_YY = np.eye(n_j)
    B_pca = np.zeros((n_j, n_u))
    M_sector = np.zeros((n_j, n_u))
    
    B_pos, B_neg, inv_A, Sigma_YX = model._solve_asymmetric_blp(
        C_YX_pos, C_YX_neg, C_XX, C_YY, B_pca, M_sector
    )
    assert B_pos.shape == (n_j, n_u)
    assert B_neg.shape == (n_j, n_u)
    assert inv_A.shape == (n_u, n_u)
    assert Sigma_YX.shape == (n_j, n_u)
    assert np.all(np.isfinite(B_pos))
    assert np.all(np.isfinite(inv_A))

def test_predict_meta_weight(base_config):
    """Verify that _predict_meta_weight correctly runs models and yields weights in [0.6, 1.0]."""
    cfg = copy.deepcopy(base_config)
    cfg["meta_learning_enabled"] = True
    model = SectorRelativeEnsembleBLPEnhancedModel(cfg)
    
    # Generate 150 days of dummy meta features
    # features: [us_disp, cond_num, vix, recent_ic_blpx, recent_ic_pca]
    T_train = 150
    us_dispersions = np.random.rand(T_train).tolist()
    cond_nums = (np.random.rand(T_train) * 10.0).tolist()
    vix_vals = (15.0 + np.random.rand(T_train) * 15.0).tolist()
    ic_blpx_vals = (np.random.rand(T_train) * 0.1).tolist()
    ic_pca_vals = (np.random.rand(T_train) * 0.1).tolist()
    
    # 1. Test Logistic Regression Mode
    w_t_lr = model._predict_meta_weight(
        140, us_dispersions, cond_nums, vix_vals, ic_blpx_vals, ic_pca_vals
    )
    assert 0.6 <= w_t_lr <= 1.0
    
    # 2. Test Ridge Regression Mode
    model.meta_model_type = "ridge"
    w_t_ridge = model._predict_meta_weight(
        140, us_dispersions, cond_nums, vix_vals, ic_blpx_vals, ic_pca_vals
    )
    assert 0.6 <= w_t_ridge <= 1.0
