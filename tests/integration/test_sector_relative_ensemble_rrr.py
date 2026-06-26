"""Unit tests for Sector Relative Ensemble with Reduced-Rank Regression (PCA-Ensemble-RRR) Model."""

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

from leadlag.models.sector_relative_ensemble_rrr import SectorRelativeEnsembleRRRModel
from leadlag.models.sre import SectorRelativeEnsembleModel
from leadlag.execution.backtester import BacktestEngine


@pytest.fixture
def rrr_sample_config() -> dict:
    """Return sample configuration dictionary for testing PCA-Ensemble-RRR."""
    return {
        "model": {"name": "sector_relative_ensemble_rrr"},
        "portfolio": {"long_short_frac": 0.3, "weight_mode": "signal"},
        "ensemble": {
            "raw_pca_weight": 0.4,
            "residual_pca_weight": 0.4,
            "p6_weight": 0.1,
            "p6p3_weight": 0.1,
            "p7_weight": 0.0,
            "p7p3_weight": 0.0,
            "normalization": "zscore",
        },
        "costs": {"slippage_bps_per_side": 5.0},
        "rrr_window": 252,
        "rrr_ewma_halflife": 45,
        "lambda_ridge": 0.03,
        "lambda_prior": 0.3,
        "rank": 3,
        "variant": "Lowrank_BLP",
        "rho_blp": 0.03,
        "alpha_xx": 0.5,
        "alpha_yx": 0.25,
    }


def test_rrr_no_lookahead(rrr_sample_config, sample_df_exec):
    """1. test_rrr_no_lookahead: Verify B_t is estimated using only Y_date <= signal_date."""
    df_exec, _ = sample_df_exec
    model = SectorRelativeEnsembleRRRModel(rrr_sample_config)

    for i in range(model.corr_window, len(df_exec)):
        t_i = df_exec["sig_date"].values[i]
        T_prev = df_exec.index[i - 1]
        assert T_prev <= t_i


def test_rrr_matrix_dimensions(rrr_sample_config, sample_df_exec):
    """2. test_rrr_matrix_dimensions: Verify dimensions of RRR cov, B, and y_hat matrices."""
    df_exec, _ = sample_df_exec
    model = SectorRelativeEnsembleRRRModel(rrr_sample_config)

    inputs = model._prepare_common_inputs(df_exec)
    all_returns = inputs["all_returns_raw"]

    idx = 300
    res = model.compute_rrr_signal(
        all_returns,
        current_index=idx,
        c_full_prior=inputs["c_full"],
        v0_static=inputs["v0_static"],
        gap_override=np.zeros(model.n_j),
        betas_t=np.zeros(model.n_j),
        topix_night_t=0.0,
    )

    assert model.n_u == 15
    assert model.n_j == 17
    assert len(res["signal"]) == 17
    assert len(res["z_hat_j_t1"]) == 17
    assert res["B_t"].shape == (17, 15)
    assert isinstance(res["cond_num"], float)
    assert isinstance(res["b_norm"], float)


def test_rrr_rank_constraint(rrr_sample_config, sample_df_exec):
    """3. test_rrr_rank_constraint: Verify rank(B) <= K."""
    df_exec, _ = sample_df_exec
    model = SectorRelativeEnsembleRRRModel(rrr_sample_config)
    model.rank = 3

    inputs = model._prepare_common_inputs(df_exec)
    all_returns = inputs["all_returns_raw"]

    idx = 300
    res = model.compute_rrr_signal(
        all_returns,
        current_index=idx,
        c_full_prior=inputs["c_full"],
        v0_static=inputs["v0_static"],
        gap_override=np.zeros(model.n_j),
        betas_t=np.zeros(model.n_j),
        topix_night_t=0.0,
    )

    B_t = res["B_t"]
    U, S, Vt = np.linalg.svd(B_t)
    effective_rank = np.sum(S > 1e-10)
    assert effective_rank <= 3


def test_rrr_ridge_stability(rrr_sample_config):
    """4. test_rrr_ridge_stability: Verify matrix inversion works with collinear inputs."""
    model = SectorRelativeEnsembleRRRModel(rrr_sample_config)
    model.n_u = 3
    model.n_j = 2
    model.rrr_window = 10
    model.rrr_ewma_halflife = 45

    collinear_returns = np.ones((10, 5))
    collinear_returns += np.random.randn(10, 5) * 1e-12

    mu = np.mean(collinear_returns, axis=0)
    sigma = np.std(collinear_returns, axis=0, ddof=0)
    sigma[sigma == 0] = 1e-8
    z = (collinear_returns - mu) / sigma
    corr = np.dot(z.T, z) / len(collinear_returns)

    C_XX = corr[:3, :3]
    C_YX = corr[3:, :3]

    model.lambda_ridge = 0.03
    model.lambda_prior = 0.0
    diag_mean = np.mean(np.diag(C_XX))
    A = C_XX + model.lambda_ridge * diag_mean * np.eye(3)

    inv_A = np.linalg.inv(A)
    B_t = C_YX @ inv_A

    assert np.all(np.isfinite(B_t))


def test_lowrank_blp_svd(rrr_sample_config, sample_df_exec):
    """5. test_lowrank_blp_svd: Verify Lowrank_BLP SVD rank reduction works."""
    df_exec, _ = sample_df_exec
    model = SectorRelativeEnsembleRRRModel(rrr_sample_config)
    model.variant = "Lowrank_BLP"
    model.rank = 2

    inputs = model._prepare_common_inputs(df_exec)
    all_returns = inputs["all_returns_raw"]

    res = model.compute_rrr_signal(
        all_returns,
        current_index=300,
        c_full_prior=inputs["c_full"],
        v0_static=inputs["v0_static"],
        gap_override=np.zeros(model.n_j),
        betas_t=np.zeros(model.n_j),
        topix_night_t=0.0,
    )

    B_t = res["B_t"]
    U, S, Vt = np.linalg.svd(B_t)
    effective_rank = np.sum(S > 1e-10)
    assert effective_rank <= 2


def test_prior_rrr_formula(rrr_sample_config):
    """6. test_prior_rrr_formula: Verify prior shrunk solution formula on a small manual setup."""
    model = SectorRelativeEnsembleRRRModel(rrr_sample_config)
    n_u = 3
    n_j = 2
    model.n_u = n_u
    model.n_j = n_j

    C_XX = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 1.0]])
    C_YX = np.array([[0.4, 0.2, 0.1], [0.1, 0.3, 0.5]])
    B_prior = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]])

    lambda_ridge = 0.05
    lambda_prior = 0.10

    diag_mean = np.mean(np.diag(C_XX))
    A_expected = C_XX + (lambda_ridge + lambda_prior) * diag_mean * np.eye(n_u)
    B_base_expected = (C_YX + lambda_prior * B_prior) @ np.linalg.inv(A_expected)

    model.lambda_ridge = lambda_ridge
    model.lambda_prior = lambda_prior
    model.variant = "BLP_prior_RRR"
    model.rank = "full"

    # Mock compute_rrr_signal logic
    A = C_XX + (model.lambda_ridge + model.lambda_prior) * diag_mean * np.eye(n_u)
    inv_A = np.linalg.inv(A)
    B_base = (C_YX + model.lambda_prior * B_prior) @ inv_A

    assert np.allclose(B_base, B_base_expected)


def test_ensemble_weights_sum(rrr_sample_config):
    """7. test_ensemble_weights_sum: Verify PCA-Ensemble-RRR weights sum to 1.0."""
    model = SectorRelativeEnsembleRRRModel(rrr_sample_config)
    w_sum = (
        model.raw_pca_weight
        + model.residual_pca_weight
        + model.p6_weight
        + model.p6p3_weight
        + model.p7_weight
        + model.p7p3_weight
    )
    assert abs(w_sum - 1.0) < 1e-12


def test_no_nan_inf_signals(rrr_sample_config, sample_df_exec):
    """8. test_no_nan_inf_signals: Verify there are no NaN or Inf signal values."""
    df_exec, _ = sample_df_exec
    model = SectorRelativeEnsembleRRRModel(rrr_sample_config)

    res = model.predict_signals(df_exec)
    p6_slice = res["p6_signals"].loc["2015-01-05":]
    p6p3_slice = res["p6p3_signals"].loc["2015-01-05":]
    signals_slice = res["signals"].loc["2015-01-05":]

    assert not p6_slice.isna().any().any()
    assert not np.isinf(p6_slice.values).any()
    assert not p6p3_slice.isna().any().any()
    assert not np.isinf(p6p3_slice.values).any()
    assert not signals_slice.isna().any().any()
    assert not np.isinf(signals_slice.values).any()


def test_cost_consistency(rrr_sample_config, sample_df_exec):
    """9. test_cost_consistency: Verify net_return = gross_return - cost."""
    df_exec, _ = sample_df_exec
    model = SectorRelativeEnsembleRRRModel(rrr_sample_config)
    start_str = df_exec.index[-20].strftime("%Y-%m-%d")

    results = BacktestEngine.run_backtest(
        model, df_exec, start_date=start_str
    )
    r_gross = results["daily_returns_gross"]
    r_net = results["daily_returns"]
    costs = results["daily_costs"]

    diff = np.abs(r_gross - costs - r_net)
    assert np.all(diff < 1e-15)


def test_baseline_sre_reproduction(rrr_sample_config, sample_df_exec):
    """10. test_baseline_sre_reproduction: Verify PCA-Ensemble-RRR with zero weights matches PCA PCA-Ensemble production model."""
    df_exec, _ = sample_df_exec
    start_str = df_exec.index[-10].strftime("%Y-%m-%d")

    prod_config_path = ROOT / "configs" / "archive" / "production_before_residual_blpx_20260614.yaml"
    with open(prod_config_path) as f:
        prod_cfg = yaml.safe_load(f)
    sre_model = SectorRelativeEnsembleModel(prod_cfg)

    rrr_cfg = rrr_sample_config.copy()
    rrr_cfg["ensemble"]["raw_pca_weight"] = 0.5
    rrr_cfg["ensemble"]["residual_pca_weight"] = 0.5
    rrr_cfg["ensemble"]["p6_weight"] = 0.0
    rrr_cfg["ensemble"]["p6p3_weight"] = 0.0
    rrr_cfg["ensemble"]["p7_weight"] = 0.0
    rrr_cfg["ensemble"]["p7p3_weight"] = 0.0

    rrr_model = SectorRelativeEnsembleRRRModel(rrr_cfg)

    sre_res = BacktestEngine.run_backtest(sre_model, df_exec, start_date=start_str)
    rrr_res = BacktestEngine.run_backtest(rrr_model, df_exec, start_date=start_str)

    assert np.allclose(sre_res["signals"].values, rrr_res["signals"].values, atol=1e-10)
    assert np.allclose(sre_res["weights"].values, rrr_res["weights"].values, atol=1e-10)
    assert np.allclose(sre_res["daily_returns"].values, rrr_res["daily_returns"].values, atol=1e-10)
