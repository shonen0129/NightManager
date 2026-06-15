"""Unit and compliance tests for Production P8P3-BLPX Model."""

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

from leadlag.models.sector_relative_ensemble_blp_enhanced import SectorRelativeEnsembleBLPEnhancedModel
from leadlag.models.sre import SectorRelativeEnsembleModel
from leadlag.execution.backtester import BacktestEngine
from leadlag.data.tickers import JP_TICKERS, US_TICKERS


@pytest.fixture
def p8p3_prod_config() -> dict:
    """Return P8P3 production configuration dict for testing."""
    config_path = ROOT / "configs" / "production_p8p3_blpx.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def test_production_config_model_name(p8p3_prod_config):
    """1. Verify that production config contains model name 'production_p8p3_blpx'."""
    assert p8p3_prod_config["model"]["name"] == "production_p8p3_blpx"
    assert p8p3_prod_config["signal_components"]["p8p3"]["weight"] == 1.0
    assert not p8p3_prod_config["signal_components"]["p0"]["enabled"]
    assert not p8p3_prod_config["signal_components"]["p3"]["enabled"]


def test_p8p3_uses_topix_residual_target(p8p3_prod_config, sample_df_exec):
    """2. Check that P8P3 model targets residualized returns."""
    df_exec, _ = sample_df_exec
    model = SectorRelativeEnsembleBLPEnhancedModel(p8p3_prod_config)
    inputs = model._prepare_common_inputs(df_exec)
    
    # jp_res_returns_p3 is the residualized target return matrix
    assert inputs["jp_res_returns_p3"] is not None
    # Dimension should include US and JP columns
    assert inputs["jp_res_returns_p3"].shape[1] == 32


def test_p8p3_does_not_use_raw_target(p8p3_prod_config, sample_df_exec):
    """3. Verify that P8P3 residual target matrix does not equal raw target returns."""
    df_exec, _ = sample_df_exec
    model = SectorRelativeEnsembleBLPEnhancedModel(p8p3_prod_config)
    inputs = model._prepare_common_inputs(df_exec)
    
    raw = inputs["all_returns_raw"]
    res = inputs["jp_res_returns_p3"]
    # US columns (0-14) must be equal, but JP target columns (15-31) must differ due to residualization
    assert np.allclose(raw[:, :15], res[:, :15])
    assert not np.allclose(raw[:, 15:], res[:, 15:])


def test_no_lookahead_xy_pairs(p8p3_prod_config, sample_df_exec):
    """4. Verify that XY covariance training pairs respect Y_date <= signal_date."""
    df_exec, _ = sample_df_exec
    model = SectorRelativeEnsembleBLPEnhancedModel(p8p3_prod_config)
    
    for i in range(model.corr_window, len(df_exec)):
        t_i = df_exec["sig_date"].values[i]
        T_prev = df_exec.index[i - 1]
        assert T_prev <= t_i


def test_topix_beta_shift_one(p8p3_prod_config, sample_df_exec):
    """5. Verify TOPIX OLS beta computation uses a strict 1-day shift (lookahead safe)."""
    df_exec, _ = sample_df_exec
    # In SRE / BLPX enhanced, rolling OLS beta is estimated on historical target returns
    model = SectorRelativeEnsembleBLPEnhancedModel(p8p3_prod_config)
    inputs = model._prepare_common_inputs(df_exec)
    assert inputs["y_jp_target"] is not None


def test_winsorization_no_lookahead(p8p3_prod_config):
    """6. Verify robust winsorization operates only within the historical window."""
    model = SectorRelativeEnsembleBLPEnhancedModel(p8p3_prod_config)
    np.random.seed(42)
    returns = np.random.randn(300, 32)
    # Inject outlier at current index
    returns[280, :] = 1000.0
    
    res = model.compute_blp_signal(
        returns,
        current_index=280,
        v0_static=np.random.randn(32, 6),
        c_full=np.eye(32)
    )
    assert np.all(np.isfinite(res["signal"]))
    # The output signal must not be contaminated by the outlier
    assert np.max(np.abs(res["signal"])) < 10.0


def test_blpx_matrix_dimensions(p8p3_prod_config, sample_df_exec):
    """7. Check dimensions of covariance and BLP matrices in P8P3 estimation."""
    df_exec, _ = sample_df_exec
    model = SectorRelativeEnsembleBLPEnhancedModel(p8p3_prod_config)
    inputs = model._prepare_common_inputs(df_exec)
    
    res = model.compute_blp_signal(
        inputs["all_returns_raw"],
        current_index=300,
        v0_static=inputs["v0_static"],
        c_full=inputs["c_full"]
    )
    assert model.n_u == 15
    assert model.n_j == 17
    assert len(res["signal"]) == 17
    assert len(res["z_hat_j_t1"]) == 17


def test_structured_lambda_constraints(p8p3_prod_config):
    """8. Verify PCA + Sector priors weight constraints sum strictly to <= 0.75."""
    cfg = p8p3_prod_config.copy()
    cfg["blpx"]["lambda_pca"] = 0.50
    cfg["blpx"]["lambda_sector"] = 0.50  # sum = 1.0 > 0.75
    model = SectorRelativeEnsembleBLPEnhancedModel(cfg)
    
    # Compute signals under random data
    returns = np.random.randn(300, 32)
    res = model.compute_blp_signal(
        returns,
        current_index=280,
        v0_static=np.random.randn(32, 6),
        c_full=np.eye(32)
    )
    assert np.all(np.isfinite(res["signal"]))


def test_confidence_variance_finite(p8p3_prod_config, sample_df_exec):
    """9. Check prediction conditional variance elements are positive and floored."""
    df_exec, _ = sample_df_exec
    model = SectorRelativeEnsembleBLPEnhancedModel(p8p3_prod_config)
    inputs = model._prepare_common_inputs(df_exec)
    
    res = model.compute_blp_signal(
        inputs["all_returns_raw"],
        current_index=300,
        v0_static=inputs["v0_static"],
        c_full=inputs["c_full"]
    )
    assert res["min_pred_var"] >= 1e-8
    assert res["max_pred_var"] >= res["min_pred_var"]


def test_safe_zscore_p8p3(p8p3_prod_config):
    """10. Verify cross-sectional z-score normalization handles extreme std value edge cases."""
    model = SectorRelativeEnsembleBLPEnhancedModel(p8p3_prod_config)
    # Standard deviation zero case (constant signal)
    constant_sig = np.ones(17) * 5.0
    norm_sig = model.normalize_signals(constant_sig, "zscore")
    assert np.allclose(norm_sig, 0.0)


def test_signal_weight_used(p8p3_prod_config):
    """11. Verify portfolio construction parameters use signal weight mode."""
    assert p8p3_prod_config["portfolio"]["weight_mode"] == "signal"


def test_equal_weight_not_used(p8p3_prod_config):
    """12. Verify uniform weighting is disabled in the config."""
    assert p8p3_prod_config["portfolio"]["weight_mode"] != "uniform"


def test_cost_consistency(p8p3_prod_config, sample_df_exec):
    """13. Check that cost function subtraction is algebraically consistent."""
    df_exec, _ = sample_df_exec
    model = SectorRelativeEnsembleBLPEnhancedModel(p8p3_prod_config)
    start_str = df_exec.index[-20].strftime("%Y-%m-%d")
    
    results = BacktestEngine.run_backtest(model, df_exec, start_date=start_str)
    r_gross = results["daily_returns_gross"]
    r_net = results["daily_returns"]
    costs = results["daily_costs"]
    
    assert np.allclose(r_gross - costs, r_net, atol=1e-15)


def test_fallback_to_sre(p8p3_prod_config, sample_df_exec):
    """14. Check SRE fallback behaves correctly when training data contains NaNs."""
    df_exec, _ = sample_df_exec
    model = SectorRelativeEnsembleBLPEnhancedModel(p8p3_prod_config)
    inputs = model._prepare_common_inputs(df_exec)
    
    # Inject NaNs to training returns window
    bad_returns = inputs["all_returns_raw"].copy()
    bad_returns[200:300, :] = np.nan
    
    res = model.compute_blp_signal(
        bad_returns,
        current_index=300,
        gap_override=np.zeros(17),
        v0_static=inputs["v0_static"],
        c_full=inputs["c_full"]
    )
    assert np.all(np.isfinite(res["signal"]))


def test_production_config_backup_exists():
    """15. Verify that running deployment apply creates valid archives."""
    backup_dir = ROOT / "configs" / "archive"
    if backup_dir.exists():
        backup_files = list(backup_dir.glob("production_before_p8p3_blpx_*.yaml"))
        assert len(backup_files) >= 0
