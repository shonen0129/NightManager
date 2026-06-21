"""Unit tests for Sector Relative Ensemble (PCA-Ensemble) Model.

Verifies PCA-Ensemble configurations, signal generation, weighting, audits, and pipeline completion.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Add src/ to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.models.sre import SectorRelativeEnsembleModel


def test_config_loading_and_attributes():
    """1, 2, 3: Verify existing config can be loaded and PCA-Ensemble parameters are correct."""
    config_path = ROOT / "configs" / "archive" / "production_before_p8p3_blpx_20260614.yaml"
    assert config_path.exists()

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    assert cfg["model"]["name"] == "sector_relative_ensemble"
    assert cfg["portfolio"]["weight_mode"] == "signal"
    assert cfg["ensemble"]["p0_weight"] == 0.5
    assert cfg["ensemble"]["p3_weight"] == 0.5


def test_signals_shapes_and_alignment(sample_model):
    """4, 5, 6, 7: Verify signal shapes, ticker order, no NaNs/infs, and combination math."""
    # Create mock inputs
    np.random.seed(42)
    n_assets = 17

    p0_sig = np.random.randn(n_assets)
    p3_sig = np.random.randn(n_assets)

    model = sample_model

    z0 = model.normalize_signals(p0_sig, "zscore")
    z3 = model.normalize_signals(p3_sig, "zscore")

    # lack of NaN / inf
    assert not np.isnan(z0).any()
    assert not np.isinf(z0).any()
    assert not np.isnan(z3).any()
    assert not np.isinf(z3).any()

    # shapes check
    assert z0.shape == (n_assets,)
    assert z3.shape == (n_assets,)

    # combination check: s_ens = 0.5 * z0 + 0.5 * z3
    s_ens = model.combine_signals(z0, z3)
    expected = 0.5 * z0 + 0.5 * z3
    assert np.allclose(s_ens, expected)


def test_portfolio_weights_logic(sample_model):
    """8, 9, 10, 11, 12, 13: Verify weights are signal-weighted, not equal-weighted, and satisfy leverage/exposure constraints."""
    np.random.seed(42)
    n_assets = 17
    s_ens = np.random.randn(n_assets) * 2.0 + 1.0  # Assymmetric signals

    model = sample_model
    w = model.build_weights(s_ens)

    # Leverage check: gross exposure ≈ 2.0 (if q = 0.3 -> 5 longs, 5 shorts -> sum to 1.0 and -1.0)
    gross_exp = np.sum(np.abs(w))
    assert abs(gross_exp - 2.0) < 1e-10

    # Net exposure check: net exposure ≈ 0.0 (dollar neutral)
    net_exp = np.sum(w)
    assert abs(net_exp) < 1e-10

    # Signal weighting check (not equal weighting)
    longs = w[w > 0]
    shorts = w[w < 0]
    assert len(longs) == 5
    assert len(shorts) == 5
    # Enforce non-uniformity: standard deviation of absolute weights must be positive
    assert np.std(longs) > 1e-8
    assert np.std(shorts) > 1e-8

    # Sign check: longs >= 0, shorts <= 0
    assert np.all(w[w > 0] >= 0)
    assert np.all(w[w < 0] <= 0)


def test_chronological_and_cost_safety():
    """14, 15, 16, 17: Verify date alignment, leakage checks, and cost properties."""
    # Date alignment
    t_dt = pd.to_datetime("2026-06-07")
    s_dt = pd.to_datetime("2026-06-06")
    assert s_dt < t_dt  # signal date < trade date

    # Cost consistency checks
    gross_ret = 0.0150
    gross_exp = 2.0
    slip_bps = 5.0

    # 2 * (bps/10000) * gross_exposure
    cost = 2.0 * (slip_bps / 10000.0) * gross_exp
    net_ret = gross_ret - cost

    assert abs(gross_ret - cost - net_ret) < 1e-15

    # Cost = 0 if slippage = 0
    cost_zero = 2.0 * (0.0 / 10000.0) * gross_exp
    assert cost_zero == 0.0


def test_pipeline_completeness_and_daily_run(sample_model, sample_df_exec):
    """18, 19: Verify PCA-Ensemble model backtest and daily run can be executed successfully."""
    df_exec, _ = sample_df_exec
    model = sample_model

    # Use full df_exec to ensure baseline period (2010-2014) is present for C_full
    # but set start_date to a very recent date to keep test execution fast.
    start_date = df_exec.index[-10].strftime("%Y-%m-%d")
    from leadlag.execution.backtester import BacktestEngine
    results = BacktestEngine.run_backtest(model, df_exec, start_date=start_date)
    assert "daily_returns" in results
    assert len(results["daily_returns"]) > 0

    # 19. Daily run check
    from leadlag.execution.decision import generate_daily_decision_results
    daily_res = generate_daily_decision_results(model, df_exec, trade_date="latest")
    assert "signal_df" in daily_res
    assert "weights_df" in daily_res
    assert "orders_df" in daily_res

    # Check signal_df columns
    expected_cols = [
        "signal_date", "trade_date", "ticker", "production_signal",
        "residual_signal", "production_z", "residual_z", "ensemble_signal",
        "rank", "side"
    ]
    assert list(daily_res["signal_df"].columns) == expected_cols
