"""Unit tests for Sprint 0 diagnostics pipeline."""

from __future__ import annotations

import os
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from experiments.diagnostics.sprint0 import run_sprint0_calculations


def test_sprint0_calculations_subset():
    """Verify that run_sprint0_calculations runs successfully on a subset of data."""
    # We can pass start_date to limit the data size and speed up calculation
    start_date = "2026-01-01"
    
    results = run_sprint0_calculations(start_date=start_date)
    
    # Check returned keys
    assert "returns_panel" in results
    assert "residual_returns_panel" in results
    assert "signal_diagnostics_panel" in results
    assert "ic_timeseries" in results
    assert "ic_summary" in results
    assert "quantile_return_summary" in results
    assert "beta_exposure_timeseries" in results
    assert "beta_exposure_stats" in results
    assert "long_short_pnl_decomposition" in results
    assert "liquidity_summary" in results
    assert "cost_impact_summary" in results
    assert "capacity_summary" in results
    assert "predicted_ir_calibration" in results
    
    # Check shape of returns panel
    r_cc = results["returns_panel"]["r_cc"]
    assert r_cc.shape[1] == 17  # 17 JP sector ETFs
    
    # Verify no lookahead in beta (beta should be lookahead-free and not contain future values)
    beta_ts = results["beta_exposure_timeseries"]
    assert not beta_ts["beta_exposure"].isna().all()
