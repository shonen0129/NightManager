"""Contract tests for the typed portfolio decision boundary."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from leadlag.domain.portfolio import PortfolioDecision
from leadlag.reporting.production_v2_writer import _coerce_decision


def _decision() -> PortfolioDecision:
    return PortfolioDecision(
        w_final=np.array([0.5, -0.5]),
        scores=np.array([1.0, -1.0]),
        mu_gap=np.zeros(2),
        sigma_gap=np.ones(2),
        Omega_gap=np.eye(2),
        fallback={"gap_data_missing": False, "audit_failure": False},
        pit_binning={"multiplier": 1.0},
        leakage={"status": "PASSED"},
        numerical={"status": "PASSED"},
        alerts=[],
        summary={},
        run_config=SimpleNamespace(cost_bps_per_gross=10.0),
    )


def test_portfolio_decision_exposes_attributes_only() -> None:
    decision = _decision()

    assert np.array_equal(decision.w_final, [0.5, -0.5])
    assert not hasattr(decision, "get")
    with pytest.raises(TypeError):
        decision["w_final"]  # type: ignore[index]


def test_legacy_mapping_is_coerced_at_writer_boundary() -> None:
    decision = _decision()
    raw = {
        "w_final": decision.w_final,
        "scores": decision.scores,
        "mu_gap": decision.mu_gap,
        "sigma_gap": decision.sigma_gap,
        "Omega_gap": decision.Omega_gap,
        "fallback": decision.fallback,
        "pit_binning": decision.pit_binning,
        "leakage": decision.leakage,
        "numerical": decision.numerical,
        "alerts": decision.alerts,
        "summary": decision.summary,
        "run_config": decision.run_config,
    }

    coerced = _coerce_decision(raw)
    assert isinstance(coerced, PortfolioDecision)
    assert np.array_equal(coerced.w_final, decision.w_final)
