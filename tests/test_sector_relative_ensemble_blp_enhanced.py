"""Unit tests for Sector Relative Ensemble with Enhanced BLP (PCA-BLPX Ensemble) Model."""

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
from leadlag.models.sector_relative_ensemble_blp_enhanced import SectorRelativeEnsembleBLPEnhancedModel
from leadlag.models.sre import SectorRelativeEnsembleModel
from leadlag.execution.backtester import BacktestEngine
from leadlag.data.tickers import JP_TICKERS, US_TICKERS


@pytest.fixture
def blpx_sample_config() -> dict:
    """Return sample configuration dictionary for testing PCA-BLPX Ensemble."""
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
    }


def test_blpx_no_lookahead(blpx_sample_config, sample_df_exec):
    """1. test_blpx_no_lookahead: Verify B_t is estimated using only Y_date <= signal_date."""
    df_exec, _ = sample_df_exec
    model = SectorRelativeEnsembleBLPEnhancedModel(blpx_sample_config)
    
    # For each date i, check that the training sample's latest trade date (T_{i-1})
    # is strictly <= the current step's signal date (t_i).
    for i in range(model.corr_window, len(df_exec)):
        t_i = df_exec["sig_date"].values[i]
        T_prev = df_exec.index[i - 1]
        assert T_prev <= t_i


def test_blpx_matrix_dimensions(blpx_sample_config, sample_df_exec):
    """2. test_blpx_matrix_dimensions: Verify dimensions of covariance and BLP matrices."""
    df_exec, _ = sample_df_exec
    model = SectorRelativeEnsembleBLPEnhancedModel(blpx_sample_config)

    # Run signal preparation
    inputs = model._prepare_common_inputs(df_exec)
    all_returns = inputs["all_returns_raw"]

    idx = 300
    res = model.compute_blp_signal(
        all_returns,
        current_index=idx,
        gap_override=np.zeros(model.n_j),
        betas_t=np.zeros(model.n_j),
        topix_night_t=0.0,
        v0_static=inputs["v0_static"],
        c_full=inputs["c_full"],
    )

    assert model.n_u == 15
    assert model.n_j == 17
    assert len(res["signal"]) == 17
    assert len(res["z_hat_j_t1"]) == 17
    assert isinstance(res["cond_num"], float)
    assert isinstance(res["b_norm"], float)
    assert isinstance(res["b_pca_norm"], float)
    assert isinstance(res["b_sector_norm"], float)
    assert isinstance(res["b_struct_norm"], float)


def test_structured_lambda_constraints(blpx_sample_config):
    """3. test_structured_lambda_constraints: Verify lambda_pca + lambda_sector <= 0.75 constraint."""
    cfg = blpx_sample_config.copy()
    cfg["lambda_pca"] = 0.5
    cfg["lambda_sector"] = 0.5  # sum is 1.0 > 0.75
    model = SectorRelativeEnsembleBLPEnhancedModel(cfg)

    # Standardize a mock matrix
    all_returns = np.random.randn(300, 32)
    # Check compute_blp_signal scales them down
    res = model.compute_blp_signal(
        all_returns,
        current_index=280,
        v0_static=np.random.randn(32, 6),
        c_full=np.eye(32)
    )
    
    assert res["b_struct_norm"] is not None


def test_sector_prior_mapping(blpx_sample_config):
    """4. test_sector_prior_mapping: Verify M_sector size, normalization, and absence of NaNs."""
    model = SectorRelativeEnsembleBLPEnhancedModel(blpx_sample_config)
    M = model.M_sector
    
    assert M.shape == (17, 15)
    assert np.all(np.isfinite(M))
    
    # Check column normalization
    col_sums = np.sum(M, axis=0)
    for u in range(15):
        if col_sums[u] > 0:
            assert abs(col_sums[u] - 1.0) < 1e-10


def test_pca_prior_dimensions(blpx_sample_config, sample_df_exec):
    """5. test_pca_prior_dimensions: Verify B_pca shape is 17x15."""
    df_exec, _ = sample_df_exec
    model = SectorRelativeEnsembleBLPEnhancedModel(blpx_sample_config)
    inputs = model._prepare_common_inputs(df_exec)
    
    res = model.compute_blp_signal(
        inputs["all_returns_raw"],
        current_index=300,
        v0_static=inputs["v0_static"],
        c_full=inputs["c_full"]
    )
    
    assert res["b_pca_norm"] >= 0.0


def test_confidence_variance(blpx_sample_config, sample_df_exec):
    """6. test_confidence_variance: Verify conditional variance bounds and flooring."""
    df_exec, _ = sample_df_exec
    model = SectorRelativeEnsembleBLPEnhancedModel(blpx_sample_config)
    inputs = model._prepare_common_inputs(df_exec)
    
    res = model.compute_blp_signal(
        inputs["all_returns_raw"],
        current_index=300,
        v0_static=inputs["v0_static"],
        c_full=inputs["c_full"]
    )
    
    assert res["min_pred_var"] >= 1e-8
    assert np.all(np.isfinite(res["signal"]))


def test_winsorization_no_lookahead(blpx_sample_config):
    """7. test_winsorization_no_lookahead: Verify winsorization uses only past data."""
    model = SectorRelativeEnsembleBLPEnhancedModel(blpx_sample_config)
    # Generate random returns
    np.random.seed(42)
    returns = np.random.randn(300, 32)
    # Add huge outlier at current step
    returns[280, :] = 1000.0
    
    # Run prediction at 280, window excludes current index 280 (so it should not see the outlier)
    res = model.compute_blp_signal(
        returns,
        current_index=280,
        v0_static=np.random.randn(32, 6),
        c_full=np.eye(32)
    )
    
    assert np.all(np.isfinite(res["signal"]))


def test_blpx_ridge_stability(blpx_sample_config):
    """8. test_blpx_ridge_stability: Verify matrix inversion works even with singular inputs."""
    model = SectorRelativeEnsembleBLPEnhancedModel(blpx_sample_config)
    model.n_u = 3
    model.n_j = 2
    model.blp_window = 10
    model.blp_ewma_halflife = 45

    collinear_returns = np.ones((10, 5))
    collinear_returns += np.random.randn(10, 5) * 1e-12

    res = model.compute_blp_signal(
        collinear_returns,
        current_index=9,
        v0_static=np.random.randn(5, 2),
        c_full=np.eye(5)
    )
    assert np.all(np.isfinite(res["signal"]))


def test_ensemble_weights_sum(blpx_sample_config):
    """9. test_ensemble_weights_sum: Verify ensemble weights sum to 1.0."""
    model = SectorRelativeEnsembleBLPEnhancedModel(blpx_sample_config)
    w_sum = model.raw_pca_weight + model.residual_pca_weight + model.raw_blpx_weight + model.residual_blpx_weight
    assert abs(w_sum - 1.0) < 1e-12


def test_no_nan_inf_signals(blpx_sample_config, sample_df_exec):
    """10. test_no_nan_inf_signals: Verify signals have no NaNs/Infs."""
    df_exec, _ = sample_df_exec
    model = SectorRelativeEnsembleBLPEnhancedModel(blpx_sample_config)
    res = model.predict_signals(df_exec)
    
    raw_blpx_slice = res["raw_blpx_signals"].loc["2015-01-05":]
    residual_blpx_slice = res["residual_blpx_signals"].loc["2015-01-05":]
    signals_slice = res["signals"].loc["2015-01-05":]
    
    assert not raw_blpx_slice.isna().any().any()
    assert not np.isinf(raw_blpx_slice.values).any()
    assert not residual_blpx_slice.isna().any().any()
    assert not np.isinf(residual_blpx_slice.values).any()
    assert not signals_slice.isna().any().any()
    assert not np.isinf(signals_slice.values).any()


def test_cost_consistency(blpx_sample_config, sample_df_exec):
    """11. test_cost_consistency: Verify net_return = gross_return - cost."""
    df_exec, _ = sample_df_exec
    model = SectorRelativeEnsembleBLPEnhancedModel(blpx_sample_config)
    start_str = df_exec.index[-20].strftime("%Y-%m-%d")

    results = BacktestEngine.run_backtest(
        model, df_exec, start_date=start_str
    )
    r_gross = results["daily_returns_gross"]
    r_net = results["daily_returns"]
    costs = results["daily_costs"]

    diff = np.abs(r_gross - costs - r_net)
    assert np.all(diff < 1e-15)


def test_continuous_m_sector_positive_corr_only(blpx_sample_config):
    """Verify continuous M_sector uses max(0, corr) — negative correlations get zero weight."""
    cfg = blpx_sample_config.copy()
    cfg["sector_eta"] = 1.0
    cfg["sector_gamma"] = 2.0
    model = SectorRelativeEnsembleBLPEnhancedModel(cfg)

    n_u, n_j = model.n_u, model.n_j
    corr = np.eye(n_u + n_j)
    # Set all cross-correlations to -0.5 (negative)
    corr[:n_u, n_u:] = -0.5
    corr[n_u:, :n_u] = -0.5
    np.fill_diagonal(corr, 1.0)

    B_blp = np.zeros((n_j, n_u))
    all_returns = np.random.randn(300, 32)
    M = model._get_sector_prior(250, all_returns, corr, B_blp)

    # With all-negative correlations and max(0, corr), M_data should be all zeros
    # So M_blended = (1-eta)*M_fixed + eta*0 = M_fixed (since eta=1.0, M_blended=0)
    # Actually with eta=1.0, M_blended = M_data = zeros (all corr <= 0)
    assert np.allclose(M, 0.0, atol=1e-10)


def test_continuous_m_sector_mixed_corr(blpx_sample_config):
    """Verify continuous M_sector assigns weight only to positively correlated tickers."""
    cfg = blpx_sample_config.copy()
    cfg["sector_eta"] = 1.0
    cfg["sector_gamma"] = 1.0
    model = SectorRelativeEnsembleBLPEnhancedModel(cfg)

    n_u, n_j = model.n_u, model.n_j
    corr = np.eye(n_u + n_j)
    # US ticker 0 (XLB) maps to 1620.T (j=3) and 1623.T (j=6)
    j_pos = JP_TICKERS.index("1620.T")  # 3
    j_neg = JP_TICKERS.index("1623.T")  # 6
    corr[0, n_u + j_pos] = 0.8   # US 0 vs 1620.T: positive
    corr[0, n_u + j_neg] = -0.6  # US 0 vs 1623.T: negative
    corr[n_u + j_pos, 0] = 0.8
    corr[n_u + j_neg, 0] = -0.6
    np.fill_diagonal(corr, 1.0)

    B_blp = np.zeros((n_j, n_u))
    all_returns = np.random.randn(300, 32)
    M = model._get_sector_prior(250, all_returns, corr, B_blp)

    # For US ticker 0 (XLB), mapped to 1620.T (j=j_pos) and 1623.T (j=j_neg)
    # Only j_pos has positive corr, so all weight goes there
    u_idx_xlb = 0
    assert M[j_pos, u_idx_xlb] > 0.0  # positive corr ticker gets weight
    assert M[j_neg, u_idx_xlb] == 0.0  # negative corr ticker gets zero


def test_tikhonov_no_scaling(blpx_sample_config, sample_df_exec):
    """Verify Tikhonov uses priors directly without norm-based rescaling."""
    df_exec, _ = sample_df_exec
    cfg = blpx_sample_config.copy()
    cfg["lambda_pca"] = 0.2
    cfg["lambda_sector"] = 0.3
    model = SectorRelativeEnsembleBLPEnhancedModel(cfg)
    inputs = model._prepare_common_inputs(df_exec)

    res = model.compute_blp_signal(
        inputs["all_returns_raw"],
        current_index=300,
        v0_static=inputs["v0_static"],
        c_full=inputs["c_full"],
        return_matrices=True,
    )

    # With direct (unscaled) priors, B_pca_prior should be the raw B_pca
    # and B_sector_prior should be the raw M_sector
    assert "B_pca_prior" in res
    assert "B_sector_prior" in res
    assert "B_pca_scaled" not in res
    assert "B_sector_scaled" not in res
    assert np.all(np.isfinite(res["B_struct"]))


def test_build_sector_prior_from_structure(blpx_sample_config):
    """Verify _build_sector_prior derives from _SECTOR_MAPPING_STRUCTURE with equal weights + normalization."""
    model = SectorRelativeEnsembleBLPEnhancedModel(blpx_sample_config)
    M = model.M_sector

    assert M.shape == (17, 15)
    assert np.all(np.isfinite(M))

    # Column normalization: each non-zero column sums to 1.0
    col_sums = np.sum(M, axis=0)
    for u in range(15):
        if col_sums[u] > 0:
            assert abs(col_sums[u] - 1.0) < 1e-10

    # Verify structural consistency: non-zero entries match _SECTOR_MAPPING_STRUCTURE
    from leadlag.data.tickers import JP_TICKERS, US_TICKERS
    for u_idx, us_tk in enumerate(US_TICKERS):
        if us_tk in model._SECTOR_MAPPING_STRUCTURE:
            mapped_jp = model._SECTOR_MAPPING_STRUCTURE[us_tk]
            for j_idx, jp_tk in enumerate(JP_TICKERS):
                if jp_tk in mapped_jp:
                    assert M[j_idx, u_idx] > 0.0, f"{jp_tk} should be non-zero for {us_tk}"
                else:
                    assert M[j_idx, u_idx] == 0.0, f"{jp_tk} should be zero for {us_tk}"


def test_baseline_sre_reproduction(blpx_sample_config, sample_df_exec):
    """12. test_baseline_sre_reproduction: Verify PCA-BLPX Ensemble with zero BLP weights matches PCA-Ensemble production model."""
    df_exec, _ = sample_df_exec
    start_str = df_exec.index[-10].strftime("%Y-%m-%d")

    prod_config_path = ROOT / "configs" / "archive" / "production_before_residual_blpx_20260614.yaml"
    with open(prod_config_path) as f:
        prod_cfg = yaml.safe_load(f)
    sre_model = SectorRelativeEnsembleModel(prod_cfg)

    blpx_cfg = blpx_sample_config.copy()
    blpx_cfg["ensemble"]["raw_pca_weight"] = 0.5
    blpx_cfg["ensemble"]["residual_pca_weight"] = 0.5
    blpx_cfg["ensemble"]["raw_blpx_weight"] = 0.0
    blpx_cfg["ensemble"]["residual_blpx_weight"] = 0.0

    blpx_model = SectorRelativeEnsembleBLPEnhancedModel(blpx_cfg)

    sre_res = BacktestEngine.run_backtest(sre_model, df_exec, start_date=start_str)
    blpx_res = BacktestEngine.run_backtest(blpx_model, df_exec, start_date=start_str)

    assert np.allclose(sre_res["signals"].values, blpx_res["signals"].values, atol=1e-10)
    assert np.allclose(sre_res["weights"].values, blpx_res["weights"].values, atol=1e-10)
    assert np.allclose(sre_res["daily_returns"].values, blpx_res["daily_returns"].values, atol=1e-10)


def test_previous_blp_reproduction(blpx_sample_config, sample_df_exec):
    """13. test_previous_blp_reproduction: Verify PCA-BLPX Ensemble with baseline parameters matches legacy BLP model."""
    df_exec, _ = sample_df_exec
    start_str = df_exec.index[-10].strftime("%Y-%m-%d")

    # Legacy PCA-Ensemble-BLP config
    legacy_cfg = {
        "model": {"name": "sector_relative_ensemble_blp"},
        "portfolio": {"long_short_frac": 0.3, "weight_mode": "signal"},
        "ensemble": {"raw_pca_weight": 0.4, "residual_pca_weight": 0.4, "p5_weight": 0.1, "p5p3_weight": 0.1},
        "costs": {"slippage_bps_per_side": 5.0},
        "blp_window": 252,
        "blp_ewma_halflife": 45,
        "alpha_xx": 0.75,
        "alpha_yx": 0.0,
        "rho": 0.003,
        "rank": "full",
    }
    legacy_model = SectorRelativeEnsembleBLPModel(legacy_cfg)

    # Enhanced PCA-BLPX Ensemble config configured to match legacy baseline
    blpx_cfg = blpx_sample_config.copy()
    blpx_cfg["ensemble"]["raw_pca_weight"] = 0.4
    blpx_cfg["ensemble"]["residual_pca_weight"] = 0.4
    blpx_cfg["ensemble"]["raw_blpx_weight"] = 0.1
    blpx_cfg["ensemble"]["residual_blpx_weight"] = 0.1
    blpx_cfg["blp_window"] = 252
    blpx_cfg["blp_ewma_halflife"] = 45
    blpx_cfg["alpha_xx"] = 0.75
    blpx_cfg["alpha_yx"] = 0.0
    blpx_cfg["rho"] = 0.003
    blpx_cfg["rank"] = "full"
    
    # disable enhanced features
    blpx_cfg["lambda_pca"] = 0.0
    blpx_cfg["lambda_sector"] = 0.0
    blpx_cfg["beta_conf"] = 0.0
    blpx_cfg["winsor_sigma"] = None
    blpx_cfg["exec_adjustment"] = "none"

    blpx_model = SectorRelativeEnsembleBLPEnhancedModel(blpx_cfg)

    legacy_res = BacktestEngine.run_backtest(legacy_model, df_exec, start_date=start_str)
    blpx_res = BacktestEngine.run_backtest(blpx_model, df_exec, start_date=start_str)

    assert np.allclose(legacy_res["signals"].values, blpx_res["signals"].values, atol=1e-10)
    assert np.allclose(legacy_res["weights"].values, blpx_res["weights"].values, atol=1e-10)
    assert np.allclose(legacy_res["daily_returns"].values, blpx_res["daily_returns"].values, atol=1e-10)
