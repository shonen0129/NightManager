from __future__ import annotations

import hashlib
import json

from leadlag.experiment_registry import ExperimentRecord, ExperimentRegistry
from research.study_history import build_report_index, review_registry


def test_reports_are_indexed_without_manufacturing_trials(tmp_path):
    report = tmp_path / "reports/20260924_sensitivity_pipeline_audit_long/report.md"
    report.parent.mkdir(parents=True)
    report.write_text("# Sensitivity\n10 candidates; 25 prior trials\n")
    output = build_report_index(tmp_path)
    entry = output["reports"][0]
    assert entry["source_sha256"] == hashlib.sha256(report.read_bytes()).hexdigest()
    assert len(entry["candidate_ids"]) == 10
    assert output["families"][0]["trials_lower_bound"] == 35
    assert output["families"][0]["complete_trial_count"] == "unknown"
    assert not (tmp_path / "var/experiments/registry.jsonl").exists()


def test_dsr_review_retains_legacy_id_corrections_and_valid_numerical_uncertainty(tmp_path):
    reg = ExperimentRegistry(tmp_path / "reg.jsonl")
    original = {
        "name": "legacy",
        "hypothesis": "h",
        "start_time": "2026-09-19T00:00:00+00:00",
        "metrics": {
            "net_sharpe": 2.0,
            "n_observations": 1000,
            "trials": 10,
            "returns": [-0.01, 0.015, -0.005, 0.02],
            "deflated_sharpe": 0.993164,
        },
    }
    reg.path.write_text(json.dumps(original) + "\n")
    legacy_id = next(iter(reg)).record_id
    reg.record(
        ExperimentRecord(
            "valid-numerically",
            "h",
            metrics={
                "metric_status": "valid",
                "metric_schema_version": "daily-v1",
                "net_sharpe_frequency": "annual",
                "trading_days_per_year": 245,
                "net_sharpe": 2.0,
                "n_observations": 4,
                "returns": [-0.01, 0.015, -0.005, 0.02],
                "trials": 1,
                "deflated_sharpe": 0.55,
            },
        )
    )
    before = reg.path.read_bytes()
    preview = review_registry(reg)
    assert reg.path.read_bytes() == before
    assert preview["corrections"][0]["record_id"] == legacy_id
    assert "invalid_or_missing_dsr_inputs" in preview["corrections"][0]["reasons"]
    assert "invalid_or_missing_dsr_inputs" not in preview["corrections"][1]["reasons"]
    written = review_registry(reg, write=True)
    assert written["appended"] == 2
    assert reg.path.read_bytes().startswith(before)
    assert len(list(reg)) == 4
    assert reg.count_trials() == 2
    assert all(rec.metrics["deflated_sharpe"] is None for rec in reg.iter_current_records())
    assert review_registry(reg, write=True)["appended"] == 0
    assert list(reg.iter_current_records())[0].correction_of == legacy_id


def test_absent_registry_is_unavailable_not_successfully_reviewed(tmp_path):
    assert (
        review_registry(ExperimentRegistry(tmp_path / "absent.jsonl"))["registry_status"]
        == "unavailable"
    )
