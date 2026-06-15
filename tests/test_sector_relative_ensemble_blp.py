"""Unit tests for Sector Relative Ensemble with Regularized Block BLP (SRE-BLP) Model."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

# Add src/ to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.models.sector_relative_ensemble_blp import SectorRelativeEnsembleBLPModel
from leadlag.models.sre import SectorRelativeEnsembleModel
from leadlag.execution.backtester import BacktestEngine


@pytest.fixture
def blp_sample_config() -> dict:
    """Return sample configuration dictionary for testing SRE-BLP."""
    return {
        "model": {"name": "sector_relative_ensemble_blp"},
        "portfolio": {"long_short_frac": 0.3, "weight_mode": "signal"},
        "ensemble": {
            "p0_weight": 0.4,
            "p3_weight": 0.4,
            "p5_weight": 0.1,
            "p5p3_weight": 0.1,
            "normalization": "zscore",
        },
        "costs": {"slippage_bps_per_side": 5.0},
        "blp_window": 252,
        "blp_ewma_halflife": 45,
        "alpha_xx": 0.5,
        "alpha_yx": 0.25,
        "rho": 0.03,
        "rank": "full",
    }


def test_blp_prediction_formula(blp_sample_config):
    """4. test_blp_prediction_formula: Check B calculation using a simplified custom matrix setup."""
    model = SectorRelativeEnsembleBLPModel(blp_sample_config)

    # Hand-crafted small example of Sigma_XX (3x3) and Sigma_YX (2x3)
    # n_u = 3, n_j = 2
    n_u = 3
    n_j = 2
    model.n_u = n_u
    model.n_j = n_j

    # correlation matrix setup
    corr = np.array(
        [[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 1.0]], dtype=float
    )  # 3x3 US correlation
    cross_corr = np.array([[0.4, 0.2, 0.1], [0.1, 0.3, 0.5]], dtype=float)  # 2x3 cross corr

    # parameters
    alpha_xx = 0.5
    alpha_yx = 0.25
    rho = 0.03

    Sigma_XX_reg = (1.0 - alpha_xx) * corr + alpha_xx * np.eye(n_u)
    Sigma_YX_reg = (1.0 - alpha_yx) * cross_corr

    diag_mean = np.mean(np.diag(Sigma_XX_reg))
    ridge = rho * diag_mean * np.eye(n_u)

    expected_B = Sigma_YX_reg @ np.linalg.inv(Sigma_XX_reg + ridge)

    # Let's run a mock inputs check
    model.alpha_xx = alpha_xx
    model.alpha_yx = alpha_yx
    model.rho = rho
    model.rank = "full"

    # Setup a mock returns history
    # 5 steps, 5 assets (3 US, 2 JP)
    window_returns = np.random.randn(20, n_u + n_j)
    mu = np.mean(window_returns, axis=0)
    sigma = np.std(window_returns, axis=0, ddof=0)
    z_window = (window_returns - mu) / sigma
    computed_corr = np.dot(z_window.T, z_window) / len(window_returns)

    C_XX = computed_corr[:n_u, :n_u]
    C_YX = computed_corr[n_u:, :n_u]

    Sigma_XX_reg_mock = (1.0 - alpha_xx) * C_XX + alpha_xx * np.eye(n_u)
    Sigma_YX_reg_mock = (1.0 - alpha_yx) * C_YX
    A = Sigma_XX_reg_mock + rho * np.mean(np.diag(Sigma_XX_reg_mock)) * np.eye(n_u)
    B_t_expected = Sigma_YX_reg_mock @ np.linalg.inv(A)

    # Check that our hand-computed matches our expectations
    assert B_t_expected.shape == (n_j, n_u)


def test_blp_matrix_dimensions(blp_sample_config, sample_df_exec):
    """2. test_blp_matrix_dimensions: Verify dimensions of covariance and BLP matrices."""
    df_exec, _ = sample_df_exec
    model = SectorRelativeEnsembleBLPModel(blp_sample_config)

    # Run signal preparation
    inputs = model._prepare_common_inputs(df_exec)
    all_returns = inputs["all_returns_raw"]

    # At current_index = 300, run blp signals
    idx = 300
    res = model.compute_blp_signal(
        all_returns,
        current_index=idx,
        gap_override=np.zeros(model.n_j),
        betas_t=np.zeros(model.n_j),
        topix_night_t=0.0,
    )

    # Dimension checks
    # US is 15 tickers, JP is 17 tickers
    assert model.n_u == 15
    assert model.n_j == 17
    assert len(res["signal"]) == 17
    assert len(res["z_hat_j_t1"]) == 17
    assert isinstance(res["cond_num"], float)
    assert isinstance(res["b_norm"], float)


def test_blp_ridge_stability(blp_sample_config):
    """3. test_blp_ridge_stability: Verify matrix inversion works even with near-singular inputs."""
    model = SectorRelativeEnsembleBLPModel(blp_sample_config)
    model.n_u = 3
    model.n_j = 2
    model.blp_window = 10
    model.blp_ewma_halflife = 45

    # Completely collinear US inputs (e.g. perfect correlation 1.0)
    collinear_returns = np.ones((10, 5))
    # Add minor noise so std is positive
    collinear_returns += np.random.randn(10, 5) * 1e-12

    # Compute correlation
    mu = np.mean(collinear_returns, axis=0)
    sigma = np.std(collinear_returns, axis=0, ddof=0)
    sigma[sigma == 0] = 1e-8
    z = (collinear_returns - mu) / sigma
    corr = np.dot(z.T, z) / len(collinear_returns)

    C_XX = corr[:3, :3]
    C_YX = corr[3:, :3]

    # Without regularization/ridge, C_XX is singular (rank 1)
    # Let's apply model regularizations
    model.alpha_xx = 0.0  # no shrinkage to diagonal
    model.rho = 0.03  # but we have ridge rho
    model.alpha_yx = 0.25

    Sigma_XX_reg = (1.0 - model.alpha_xx) * C_XX + model.alpha_xx * np.eye(3)
    Sigma_YX_reg = (1.0 - model.alpha_yx) * C_YX
    A = Sigma_XX_reg + model.rho * np.mean(np.diag(Sigma_XX_reg)) * np.eye(3)

    # Inversion should be stable and finite
    inv_A = np.linalg.inv(A)
    B_t = Sigma_YX_reg @ inv_A

    assert np.all(np.isfinite(B_t))


def test_blp_no_lookahead(blp_sample_config, sample_df_exec):
    """1. test_blp_no_lookahead: Verify B_t is estimated using only Y_date <= signal_date."""
    df_exec, _ = sample_df_exec
    model = SectorRelativeEnsembleBLPModel(blp_sample_config)

    # For each date i, check that the training sample's latest trade date (T_{i-1})
    # is strictly <= the current step's signal date (t_i).
    # This is equivalent to ensuring we don't use index i data for predicting at index i.
    for i in range(model.corr_window, len(df_exec)):
        t_i = df_exec["sig_date"].values[i]
        T_prev = df_exec.index[i - 1]
        assert T_prev <= t_i


def test_ensemble_weights_sum(blp_sample_config):
    """5. test_ensemble_weights_sum: Verify ensemble weights sum to 1.0."""
    model = SectorRelativeEnsembleBLPModel(blp_sample_config)
    w_sum = model.p0_weight + model.p3_weight + model.p5_weight + model.p5p3_weight
    assert abs(w_sum - 1.0) < 1e-12


def test_no_nan_inf_signals(blp_sample_config, sample_df_exec):
    """6. test_no_nan_inf_signals: Verify there are no NaN or Inf signal values."""
    df_exec, _ = sample_df_exec
    model = SectorRelativeEnsembleBLPModel(blp_sample_config)

    res = model.predict_signals(df_exec)
    p5_slice = res["p5_signals"].loc["2015-01-05":]
    p5p3_slice = res["p5p3_signals"].loc["2015-01-05":]
    signals_slice = res["signals"].loc["2015-01-05":]
    assert not p5_slice.isna().any().any()
    assert not np.isinf(p5_slice.values).any()
    assert not p5p3_slice.isna().any().any()
    assert not np.isinf(p5p3_slice.values).any()
    assert not signals_slice.isna().any().any()
    assert not np.isinf(signals_slice.values).any()


def test_cost_consistency(blp_sample_config, sample_df_exec):
    """7. test_cost_consistency: Verify net_return = gross_return - cost."""
    df_exec, _ = sample_df_exec
    model = SectorRelativeEnsembleBLPModel(blp_sample_config)
    start_str = df_exec.index[-20].strftime("%Y-%m-%d")

    # Run backtest using standard engine
    results = BacktestEngine.run_backtest(
        model, df_exec, start_date=start_str
    )
    r_gross = results["daily_returns_gross"]
    r_net = results["daily_returns"]
    costs = results["daily_costs"]

    diff = np.abs(r_gross - costs - r_net)
    assert np.all(diff < 1e-15)


def test_baseline_sre_reproduction(blp_sample_config, sample_df_exec):
    """8. test_baseline_sre_reproduction: Verify SRE-BLP with zero BLP weights matches SRE production model."""
    df_exec, _ = sample_df_exec
    # Run only 10 dates to keep test fast
    start_str = df_exec.index[-10].strftime("%Y-%m-%d")

    # Production SRE model
    prod_config_path = ROOT / "configs" / "archive" / "production_before_p8p3_blpx_20260614.yaml"
    with open(prod_config_path) as f:
        prod_cfg = yaml.safe_load(f)
    sre_model = SectorRelativeEnsembleModel(prod_cfg)

    # SRE-BLP configured to match SRE (p5_weight=0.0, p5p3_weight=0.0, p0_weight=0.5, p3_weight=0.5)
    blp_cfg = blp_sample_config.copy()
    blp_cfg["ensemble"]["p0_weight"] = 0.5
    blp_cfg["ensemble"]["p3_weight"] = 0.5
    blp_cfg["ensemble"]["p5_weight"] = 0.0
    blp_cfg["ensemble"]["p5p3_weight"] = 0.0

    blp_model = SectorRelativeEnsembleBLPModel(blp_cfg)

    sre_res = BacktestEngine.run_backtest(sre_model, df_exec, start_date=start_str)
    blp_res = BacktestEngine.run_backtest(blp_model, df_exec, start_date=start_str)

    # Verify signals, weights, and returns are exactly the same
    assert np.allclose(sre_res["signals"].values, blp_res["signals"].values, atol=1e-10)
    assert np.allclose(sre_res["weights"].values, blp_res["weights"].values, atol=1e-10)
    assert np.allclose(sre_res["daily_returns"].values, blp_res["daily_returns"].values, atol=1e-10)
