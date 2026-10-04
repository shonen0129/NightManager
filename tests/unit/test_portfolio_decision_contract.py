"""Contract tests for the typed portfolio decision boundary."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from leadlag.data.tickers import JP_TICKERS
from leadlag.domain.portfolio import PortfolioDecision
from leadlag.reporting.production_v2_writer import write_production_files


def _decision() -> PortfolioDecision:
    return PortfolioDecision(
        w_final=np.array([0.5, -0.5] + [0.0] * (len(JP_TICKERS) - 2)),
        scores=np.array([1.0, -1.0]),
        mu_gap=np.zeros(2),
        sigma_gap=np.ones(2),
        Omega_gap=np.eye(2),
        fallback={"gap_data_missing": False, "audit_failure": False},
        pit_binning={"multiplier": 1.0, "assigned_bin": "Medium"},
        leakage={"status": "PASSED"},
        numerical={"status": "PASSED"},
        alerts=[],
        summary={"target_gross": 1.0, "target_net": 0.0, "predicted_portfolio_ir": 0.0},
        run_config=SimpleNamespace(cost_bps_per_gross=10.0),
    )


def test_portfolio_decision_exposes_attributes_only() -> None:
    decision = _decision()

    assert np.array_equal(decision.w_final[:2], [0.5, -0.5])
    assert not hasattr(decision, "get")
    with pytest.raises(TypeError):
        decision["w_final"]  # type: ignore[index]


def test_writer_reads_typed_decision_without_conversion(tmp_path, caplog) -> None:
    decision = _decision()
    with caplog.at_level("INFO"):
        write_production_files("2026-10-02", tmp_path, decision, dry_run=True)
    assert "DRY-RUN SUMMARY" in caplog.text
    assert not list(tmp_path.iterdir())
