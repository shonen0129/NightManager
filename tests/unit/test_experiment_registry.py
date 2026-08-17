"""Unit tests for the experiment registry and deflated Sharpe computation."""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from leadlag.experiment_registry import (
    Decision,
    ExperimentRecord,
    ExperimentRegistry,
    compute_deflated_sharpe,
)


def _temp_registry() -> ExperimentRegistry:
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    return ExperimentRegistry(path)


def test_record_and_iterate():
    reg = _temp_registry()
    rec = ExperimentRecord(
        name="tv_gap_coef",
        hypothesis="Time-varying gap coefficients improve Sharpe.",
        parameters={"alpha": 0.3, "window": 252},
        metrics={"net_sharpe": 1.5, "trials": 1, "n_observations": 1000},
        decision=Decision.REJECTED,
        report_path="reports/tv_gap_coef/report.md",
    )
    reg.record(rec)

    loaded = list(reg)
    assert len(loaded) == 1
    assert loaded[0].name == "tv_gap_coef"
    assert loaded[0].decision == Decision.REJECTED
    assert loaded[0].parameters["alpha"] == 0.3


def test_count_trials_and_filter():
    reg = _temp_registry()
    now = datetime.now(UTC)
    for i in range(3):
        rec = ExperimentRecord(
            name=f"exp_{i}",
            hypothesis=f"h{i}",
            start_time=now - timedelta(days=i),
            metrics={"net_sharpe": 0.8 + i, "trials": i + 1, "n_observations": 500},
            decision=Decision.ADOPTED if i == 0 else Decision.REJECTED,
        )
        reg.record(rec)

    assert reg.count_trials() == 3
    assert reg.count_trials(since=now - timedelta(hours=12)) == 1
    adopted = list(reg.iter_records(decision=Decision.ADOPTED))
    assert len(adopted) == 1
    assert adopted[0].name == "exp_0"


def test_deflated_sharpe_basic():
    # Simple case: many observations, moderate Sharpe, few trials.
    # DSR should be a probability between 0 and 1.
    np.random.seed(42)
    returns = np.random.normal(0.0006, 0.015, 1000)
    metrics = {
        "net_sharpe": 1.0,
        "trials": 10,
        "n_observations": 1000,
        "returns": returns.tolist(),
        "trial_sharpes": np.random.normal(0.0, 0.4, 10).tolist(),
    }
    dsr = compute_deflated_sharpe(metrics)
    assert dsr is not None
    assert 0.0 <= dsr <= 1.0

    # With only 1 trial, DSR should reduce to the PSR (no selection bias).
    # Positive Sharpe -> > 0.5, negative Sharpe -> < 0.5.
    metrics_one = {
        "net_sharpe": 1.0,
        "trials": 1,
        "n_observations": 1000,
        "returns": returns.tolist(),
    }
    dsr_one = compute_deflated_sharpe(metrics_one)
    assert dsr_one is not None
    assert dsr_one > 0.5

    metrics_neg = {
        "net_sharpe": -1.0,
        "trials": 1,
        "n_observations": 1000,
        "returns": (-returns).tolist(),
    }
    dsr_neg = compute_deflated_sharpe(metrics_neg)
    assert dsr_neg is not None
    assert dsr_neg < 0.5


def test_deflated_sharpe_missing_fields():
    assert compute_deflated_sharpe({}) is None
    assert compute_deflated_sharpe({"net_sharpe": 1.0}) is None


def test_deflated_sharpe_trials_penalty():
    # Same observed Sharpe, but with more trials the DSR should fall.
    returns = np.random.default_rng(42).normal(0.0005, 0.015, 1000)
    base = {
        "net_sharpe": 1.0,
        "n_observations": 1000,
        "returns": returns.tolist(),
        "trial_sharpes": np.linspace(-0.5, 0.5, 10).tolist(),
    }
    dsr_10 = compute_deflated_sharpe({**base, "trials": 10})
    dsr_1000 = compute_deflated_sharpe({**base, "trials": 1000})
    assert dsr_10 is not None
    assert dsr_1000 is not None
    assert dsr_1000 < dsr_10


def test_decisions_summary():
    reg = _temp_registry()
    reg.record(
        ExperimentRecord(
            name="us_cs_rank",
            hypothesis="US cross-sectional rank improves BLPX.",
            metrics={"net_sharpe": 0.5, "trials": 1, "n_observations": 2000},
            decision=Decision.REJECTED,
        )
    )
    decisions = reg.decisions()
    assert len(decisions) == 1
    assert decisions[0]["decision"] == "rejected"
    assert decisions[0]["deflated_sharpe"] is not None or "trials" in decisions[0]


def test_registry_jsonl_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "registry.jsonl"
        reg = ExperimentRegistry(path)
        rec = ExperimentRecord(
            name="roundtrip",
            hypothesis="x",
            parameters={"a": [1, 2, 3]},
            metrics={"net_sharpe": 2.0},
        )
        reg.record(rec)

        with path.open("r", encoding="utf-8") as f:
            line = f.readline()
            raw = json.loads(line)
            assert raw["name"] == "roundtrip"
            assert raw["parameters"]["a"] == [1, 2, 3]
