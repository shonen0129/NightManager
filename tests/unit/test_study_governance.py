from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from leadlag.experiment_registry import (
    Decision,
    ExperimentRecord,
    ExperimentRegistry,
    compute_deflated_sharpe,
)
from research.experiment_utils import record_backtest_experiment, record_simple_experiment


def protocol():
    return {
        "target_schema": "9:10-close-v1",
        "cost_schema": "four-costs-v1",
        "is_period": {"start": "2015-01-05", "end": "2024-12-20"},
        "oos_period": {"start": "2024-12-23", "end": "2026-09-18"},
        "purge": {"sessions": 1, "embargo_sessions": 0, "rationale": "one-session labels"},
        "selection_rule": "paired net Sharpe and drawdown",
        "metrics_spec": {
            "frequency": "daily",
            "annualization_periods": 245,
            "include_flat_days": True,
        },
    }


def valid_metrics():
    return {
        "metric_schema_version": "daily-v1",
        "metric_status": "valid",
        "net_sharpe": 2.0,
        "net_sharpe_frequency": "annual",
        "trading_days_per_year": 245,
        "returns": [-0.01, 0.015, -0.005, 0.02],
        "n_observations": 4,
        "trials": 2,
        "trial_sharpes": [1.0, 2.0],
    }


@pytest.mark.parametrize(
    "override",
    [
        {"metric_status": None},
        {"metric_schema_version": "unknown"},
        {"returns": None},
        {"returns": [[0.01, 0.02], [-0.01, -0.02]]},
        {"returns": [0.01, None, -0.01, 0.02]},
        {"net_sharpe": True},
        {"trading_days_per_year": True},
        {"trading_days_per_year": 245.5},
        {"trials": True},
        {"n_observations": 4.5},
        {"include_flat_days": False},
        {"trial_count_status": "lower_bound"},
        {"trial_sharpe_variance": -1.0, "trial_sharpe_variance_basis": "external_estimate"},
        {"trial_sharpe_variance": 0.1, "trial_sharpe_variance_basis": "cross_trial_ddof1"},
        {
            "trial_sharpes": [1.0, np.inf],
            "trial_sharpe_variance": 0.5,
            "trial_sharpe_variance_basis": "external_estimate",
        },
        {"trial_sharpe_variance": 0.5, "trial_sharpe_variance_basis": "unknown"},
        {"trial_sharpes_frequency": "weekly"},
        {"trial_sharpe_variance_frequency": "monthly"},
    ],
)
def test_full_dsr_contract_rejects_unusable_inputs(override):
    assert compute_deflated_sharpe(valid_metrics()) is not None
    assert compute_deflated_sharpe({**valid_metrics(), **override}) is None


def test_variance_and_frequency_contract():
    annual = valid_metrics()
    annual["trial_sharpe_variance"] = 0.5
    annual["trial_sharpe_variance_basis"] = "cross_trial_ddof1"
    daily = {
        **annual,
        "net_sharpe": 2 / np.sqrt(245),
        "net_sharpe_frequency": "daily",
        "trial_sharpes": [1 / np.sqrt(245), 2 / np.sqrt(245)],
        "trial_sharpes_frequency": "daily",
        "trial_sharpe_variance": 0.5 / 245,
        "trial_sharpe_variance_frequency": "daily",
    }
    assert compute_deflated_sharpe(annual) == pytest.approx(
        compute_deflated_sharpe(daily), abs=1e-14
    )
    assert (
        compute_deflated_sharpe(
            {**annual, "trial_sharpes": [1.0, 1.0], "trial_sharpe_variance": 0.0}
        )
        is not None
    )
    assert (
        compute_deflated_sharpe({**annual, "trial_sharpes": None, "trial_sharpe_variance": None})
        is None
    )
    assert (
        compute_deflated_sharpe(
            {**annual, "trials": 1, "trial_sharpes": None, "trial_sharpe_variance": None}
        )
        is not None
    )


def test_missing_metric_declarations_are_unverified():
    for field in (
        "metric_status",
        "metric_schema_version",
        "net_sharpe_frequency",
        "trading_days_per_year",
    ):
        metrics = valid_metrics()
        del metrics[field]
        assert compute_deflated_sharpe(metrics) is None
    rec = ExperimentRecord("a", "h", metrics=valid_metrics(), metric_schema_version="different")
    assert rec.deflated_sharpe() is None


@pytest.mark.parametrize(
    "extra",
    [
        {"returns": []},
        {"metric_status": "valid"},
        {"trading_days_per_year": 245},
        {"include_flat_days": True},
        {"deflated_sharpe": 1.0},
        {"trial_count_status": "complete"},
    ],
)
def test_extra_fields_cannot_claim_computed_results_without_results(tmp_path, extra):
    with pytest.raises(ValueError, match="cannot override"):
        record_backtest_experiment(
            "a", "h", {}, extra_metrics=extra, registry_path=tmp_path / "reg.jsonl"
        )
    assert not (tmp_path / "reg.jsonl").exists()


def test_ledger_counts_started_failed_and_rejected_trials_and_freezes_selection(tmp_path):
    reg = ExperimentRegistry(tmp_path / "reg.jsonl")
    reg.register_study(
        "family",
        hypothesis="h",
        candidates=["a", "b", "c"],
        protocol=protocol(),
        history_complete=True,
    )
    first = reg.start_trial("family", "a", parameters={}, code_hash="a" * 64, data_hash="b" * 64)
    second = reg.start_trial("family", "b", parameters={}, code_hash="a" * 64, data_hash="b" * 64)
    assert reg.count_trials(study_id="family") == 2
    with pytest.raises(ValueError, match="outcomes"):
        reg.select_study("family", selected_trial_id=first, reason="gate")
    rec = record_backtest_experiment(
        "a",
        "h",
        {},
        results={"daily_returns": pd.Series([0.01, -0.02, 0.03, 0.01])},
        extra_metrics={
            "trial_sharpe_variance": 0.5,
            "trial_sharpe_variance_basis": "external_estimate",
        },
        decision=Decision.REJECTED,
        registry_path=reg.path,
        study_id="family",
        trial_id=first,
    )
    record_simple_experiment(
        "b",
        "h",
        {},
        {"reason": "interrupted"},
        registry_path=reg.path,
        study_id="family",
        trial_id=second,
        trial_status="aborted",
    )
    assert reg.count_trials() == 2
    event = reg.select_study("family", selected_trial_id=None, reason="no candidate passed")
    assert [t["status"] for t in event["trials"]] == ["completed", "aborted"]
    assert event["unattempted_candidates"] == ["c"]
    with pytest.raises(ValueError, match="already selected"):
        reg.start_trial("family", "c", parameters={}, code_hash="a" * 64, data_hash="b" * 64)
    before = reg.path.read_text()
    payload = rec.to_dict()
    payload.pop("record_id")
    corrected = ExperimentRecord.from_dict(payload)
    reg.record_correction(rec.record_id, corrected)
    assert reg.count_trials() == 2
    assert reg.path.read_text().startswith(before)
    assert reg.study_events()[-1] == event


def test_preregistration_provenance_and_unknown_history(tmp_path):
    reg = ExperimentRegistry(tmp_path / "reg.jsonl")
    with pytest.raises(KeyError):
        reg.start_trial("missing", "a", parameters={}, code_hash="a" * 64, data_hash="b" * 64)
    reg.register_study(
        "family",
        hypothesis="h",
        candidates=["a"],
        protocol=protocol(),
        historical_trials_lower_bound=25,
    )
    with pytest.raises(ValueError, match="outside"):
        reg.start_trial(
            "family", "unplanned", parameters={}, code_hash="a" * 64, data_hash="b" * 64
        )
    with pytest.raises(ValueError, match="SHA-256"):
        reg.start_trial("family", "a", parameters={}, code_hash="HEAD", data_hash="b" * 64)
    trial = reg.start_trial(
        "family", "a", parameters={"variant": 1}, code_hash="a" * 64, data_hash="b" * 64
    )
    with pytest.raises(ValueError, match="parameters"):
        record_simple_experiment(
            "a", "h", {}, {}, registry_path=reg.path, study_id="family", trial_id=trial
        )
    rec = record_simple_experiment(
        "a", "h", {"variant": 1}, {}, registry_path=reg.path, study_id="family", trial_id=trial
    )
    assert rec.metrics["trials"] == 26
    assert rec.metrics["trial_count_status"] == "lower_bound"
    with pytest.raises(ValueError, match="already has"):
        record_simple_experiment(
            "a", "h", {"variant": 1}, {}, registry_path=reg.path, study_id="family", trial_id=trial
        )
    saved = [json.loads(line) for line in reg.path.read_text().splitlines()]
    assert len(saved[1]["config_hash"]) == 64
    assert (
        reg.select_study("family", selected_trial_id=trial, reason="hold")[
            "selected_deflated_sharpe"
        ]
        is None
    )


@pytest.mark.parametrize(
    "override",
    [
        {"is_period": {"start": "2026-01-01", "end": "2027-01-01"}},
        {"purge": {"sessions": -1}},
        {"metrics_spec": {"frequency": "monthly"}},
        {"cost_schema": ""},
    ],
)
def test_invalid_protocol_does_not_create_study(tmp_path, override):
    reg = ExperimentRegistry(tmp_path / "reg.jsonl")
    with pytest.raises(ValueError):
        reg.register_study(
            "family", hypothesis="h", candidates=["a"], protocol={**protocol(), **override}
        )
    assert not reg.path.exists()


def test_selection_recomputes_for_all_attempts_instead_of_first_success(tmp_path):
    reg = ExperimentRegistry(tmp_path / "reg.jsonl")
    reg.register_study(
        "family", hypothesis="h", candidates=["a", "b"], protocol=protocol(), history_complete=True
    )
    first = reg.start_trial("family", "a", parameters={}, code_hash="a" * 64, data_hash="b" * 64)
    metrics = {
        **valid_metrics(),
        "trials": 1,
        "trial_sharpes": None,
        "include_flat_days": True,
        "trial_sharpe_variance": 0.5,
        "trial_sharpe_variance_basis": "external_estimate",
    }
    rec = record_simple_experiment(
        "a", "h", {}, metrics, registry_path=reg.path, study_id="family", trial_id=first
    )
    second = reg.start_trial("family", "b", parameters={}, code_hash="a" * 64, data_hash="b" * 64)
    record_simple_experiment(
        "b",
        "h",
        {},
        {},
        registry_path=reg.path,
        study_id="family",
        trial_id=second,
        trial_status="failed",
    )
    selection = reg.select_study("family", selected_trial_id=first, reason="preset rule")
    assert selection["selected_deflated_sharpe"] < rec.metrics["deflated_sharpe"]
    assert selection["trials_lower_bound"] == 2


def test_legacy_trial_study_assignment_survives_multiple_corrections(tmp_path):
    reg = ExperimentRegistry(tmp_path / "reg.jsonl")
    original = reg.record(ExperimentRecord("legacy", "h"))
    first = reg.record_correction(
        original.record_id, ExperimentRecord("legacy", "h", study_id="family")
    )
    reg.record_correction(first.record_id, ExperimentRecord("legacy", "h", study_id="family"))
    assert reg.count_trials(study_id="family") == 1
    assert reg.count_trials() == 1
    assert len(list(reg)) == 3
