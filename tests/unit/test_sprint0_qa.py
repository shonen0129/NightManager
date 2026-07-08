"""Unit tests for Sprint 0 diagnostics QA pipeline."""

from __future__ import annotations

import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from research.diagnostics.sprint0_qa import run_sprint0_qa


def test_sprint0_qa_subset():
    """Verify that run_sprint0_qa runs successfully on a subset of data."""
    # Use a small range of dates at the end of the dataset (e.g., May 2026) to make the test fast
    start_date = "2026-05-01"
    
    results = run_sprint0_qa(start_date=start_date)
    
    # Verify that all 8 QA checks returned results
    assert "qa1" in results
    assert "qa2" in results
    assert "qa3" in results
    assert "qa4" in results
    assert "qa5" in results
    assert "qa6" in results
    assert "qa7" in results
    assert "qa8" in results
    
    # Check that individual QA check outputs have expected structure/dataframes
    assert "comparison_table" in results["qa1"]
    assert "alignment_table" in results["qa2"]
    assert "sign_comparison_table" in results["qa3"]
    assert "representative_days_table" in results["qa4"]
    assert "formula_audit" in results["qa4"]
    assert "long_short_leg_pnl_summary" in results["qa5"]
    assert "long_short_leg_pnl_timeseries" in results["qa5"]
    assert "ticker_capacity_audit" in results["qa6"]
    assert "cost_capacity_reconciliation" in results["qa7"]
    assert "calibration_leak_comparison" in results["qa8"]
    
    # Verify some dataframe shapes
    comp_df = results["qa1"]["comparison_table"]
    assert comp_df.shape[0] == 3  # Full period, proxy only, true 9:10 only
    
    align_df = results["qa2"]["alignment_table"]
    assert align_df.shape[0] == 5  # Lags -2, -1, 0, 1, 2
    
    sign_df = results["qa3"]["sign_comparison_table"]
    assert sign_df.shape[1] == 2  # Positive, Negative
