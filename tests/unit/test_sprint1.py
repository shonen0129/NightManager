"""Unit tests for Sprint 1 diagnostics and optimization pipeline."""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.cache import load_df_exec_from_local_cache
from research.diagnostics.sprint1_experiments import (
    generate_targets_panel,
    solve_portfolio_weights,
    run_sprint1_backtests,
    run_ruled_rolling_calibration
)
from research.diagnostics.sprint0 import run_sprint0_calculations


def test_targets_panel_generation():
    """Verify targets panel generation and separation on a subset of dates."""
    df_exec = load_df_exec_from_local_cache()
    # Test on a small end period to be fast
    start_date = "2026-05-15"
    
    df_targets = generate_targets_panel(df_exec, start_date=start_date)
    
    assert not df_targets.empty
    assert "close_to_close_return" in df_targets.columns
    assert "gap_return" in df_targets.columns
    assert "open_to_close_return" in df_targets.columns
    assert "entry_to_close_return" in df_targets.columns
    assert "is_true_0910" in df_targets.columns
    assert "beta_topix_60d" in df_targets.columns
    assert "residual_return_60d" in df_targets.columns
    
    # Check that is_true_0910 contains boolean values
    assert df_targets["is_true_0910"].dtype == bool
    assert set(df_targets["entry_price_type"].unique()).issubset({"true_0910", "open"})


def test_slsqp_solver():
    """Verify SLSQP solver for dollar and beta neutrality under bounds."""
    # 17 JP assets
    np.random.seed(42)
    w_0 = np.random.uniform(-0.1, 0.1, 17)
    # enforce initial dollar neutrality for dummy weights
    w_0 -= w_0.mean()
    
    beta_topix = np.random.uniform(0.5, 1.5, 17)
    cap_bounds = np.full(17, 0.05) # 5% weight cap
    target_gross = 2.0
    
    # Solve with beta neutral
    w_opt, success = solve_portfolio_weights(
        w_0=w_0,
        beta_topix=beta_topix,
        cap_bounds=cap_bounds,
        target_gross_limit=target_gross,
        include_beta_neutral=True
    )
    
    assert success
    # Check dollar neutrality (sum w = 0)
    assert np.isclose(np.sum(w_opt), 0.0, atol=1e-6)
    # Check beta neutrality (sum w * beta = 0)
    assert np.isclose(np.sum(w_opt * beta_topix), 0.0, atol=1e-6)
    # Check box bounds (|w| <= cap)
    assert np.all(np.abs(w_opt) <= cap_bounds + 1e-6)
    # Check gross limit (sum |w| <= target_gross)
    assert np.sum(np.abs(w_opt)) <= target_gross + 1e-6


def test_backtest_simulation():
    """Verify backtest and capacity constraints simulations."""
    df_exec = load_df_exec_from_local_cache()
    start_date = "2026-05-15"
    
    base_results = run_sprint0_calculations(start_date=start_date)
    w_ruled_df = base_results["signal_diagnostics_panel"]["weight_ruled"]
    targets_df = generate_targets_panel(df_exec, start_date=start_date)
    
    config = {
        "aum_scenarios_jpy": [10000000, 100000000],
        "adv_caps": [0.05],
        "min_adv_jpy": [0],
        "adv_windows": [20],
        "beta_windows": [60],
        "impact_eta": [0.05],
        "static_cost_bps": [15],
        "target_gross": 2.0,
        "max_abs_weight_per_name": 0.25
    }
    
    df_backtest = run_sprint1_backtests(df_exec, w_ruled_df, targets_df, config)
    
    assert not df_backtest.empty
    assert "strategy_return" in df_backtest.columns
    assert "net_return_after_cost" in df_backtest.columns
    assert "realized_gross_exposure" in df_backtest.columns
    assert "realized_net_exposure" in df_backtest.columns
    assert "turnover" in df_backtest.columns


def test_calibration_rolling():
    """Verify RuleD rolling calibration calibration."""
    df_exec = load_df_exec_from_local_cache()
    # Need sufficient history to run rolling 252d (at least 300 days)
    start_date = "2025-01-01"
    
    base_results = run_sprint0_calculations(start_date=start_date)
    w_ruled_df = base_results["signal_diagnostics_panel"]["weight_ruled"]
    valid_dates_beta = w_ruled_df.index.intersection(df_exec.index[120:])
    
    df_calib = run_ruled_rolling_calibration(df_exec, w_ruled_df, valid_dates_beta)
    
    if not df_calib.empty:
        assert "full_sample_bin" in df_calib.columns
        assert "rolling_252_bin" in df_calib.columns
        assert "expanding_bin" in df_calib.columns
        assert "pnl_multiplier_rolling" in df_calib.columns
        # Values should be Low, Medium, or High
        assert set(df_calib["rolling_252_bin"].unique()).issubset({"Low", "Medium", "High"})
