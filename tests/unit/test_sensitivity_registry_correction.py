from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

from leadlag.experiment_registry import ExperimentRecord, ExperimentRegistry

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    ROOT
    / "reports/20260929_frozen_0910_account_risk_forward_eval"
    / "correct_sensitivity_registry_rank_ic.py"
)
SPEC = importlib.util.spec_from_file_location("rank_ic_registry_correction", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
CORRECTION_TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CORRECTION_TOOL)


def _build_sources(tmp_path: Path) -> tuple[Path, Path]:
    registry_path = tmp_path / "registry.jsonl"
    registry = ExperimentRegistry(registry_path)
    variants = [
        "all_w3_w6_zero",
        "drop_w3",
        "drop_w4",
        "drop_w5",
        "drop_w6",
        "user_jpx33_pitcap_annual",
        "legacy_jpx33_pitcap_annual",
        "rate_w6_us_only",
        "rate_w6_jp_only",
        "rate_w6_both",
    ]
    for variant in variants:
        registry.record(
            ExperimentRecord(
                name=f"20260924_sensitivity_pipeline_audit_long_{variant}",
                hypothesis="Test sensitivity variant.",
                metrics={
                    "net_sharpe": 0.5,
                    "rank_ic": {
                        "paired_dates": 102,
                        "mean_rank_ic": -0.4,
                        "rank_ic_ci95": [-0.6, -0.2],
                        "daily": [{"trade_date": "2026-01-02", "rank_ic": -0.4}],
                    },
                },
            )
        )
    csv_path = tmp_path / "rank_ic.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "variant",
                "dates",
                "mean_rank_ic_vs_backtest_target",
                "ci95",
                "dates_all17_target",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "variant": "current",
                "dates": 2779,
                "mean_rank_ic_vs_backtest_target": 0.2,
                "ci95": json.dumps([0.1, 0.3]),
                "dates_all17_target": 2779,
            }
        )
        for variant in variants:
            writer.writerow(
                {
                    "variant": variant,
                    "dates": 2779,
                    "mean_rank_ic_vs_backtest_target": 0.22,
                    "ci95": json.dumps([0.2, 0.24]),
                    "dates_all17_target": 2779,
                }
            )
    return registry_path, csv_path


def test_rank_ic_correction_appends_idempotently_and_relabels_legacy_metric(tmp_path):
    registry_path, csv_path = _build_sources(tmp_path)
    registry = ExperimentRegistry(registry_path)

    preview = CORRECTION_TOOL.correct_registry(registry_path, csv_path)
    assert preview["planned_count"] == 10
    assert preview["appended_count"] == 0
    assert len(list(registry)) == 10

    written = CORRECTION_TOOL.correct_registry(registry_path, csv_path, write=True)
    assert written["appended_count"] == 10
    assert len(list(registry)) == 20
    current = list(registry.iter_current_records())
    assert len(current) == 10
    assert registry.count_trials(study_id=CORRECTION_TOOL.STUDY_ID) == 10
    for record in current:
        assert record.metric_schema_version == CORRECTION_TOOL.METRIC_SCHEMA
        assert "rank_ic" not in record.metrics
        assert record.metrics["rank_ic_open_to_0910_diagnostic"]["mean_rank_ic"] == -0.4
        assert "daily" not in record.metrics["rank_ic_open_to_0910_diagnostic"]
        original = next(
            item for item in registry if item.name == record.name and item.correction_of is None
        )
        assert original.metrics["rank_ic"]["daily"]
        corrected = record.metrics["rank_ic_vs_backtest_target"]
        assert corrected["mean_rank_ic"] == 0.22
        assert corrected["ci95"] == [0.2, 0.24]

    repeated = CORRECTION_TOOL.correct_registry(registry_path, csv_path, write=True)
    assert repeated["appended_count"] == 0
    assert repeated["already_current_count"] == 10
    assert len(list(registry)) == 20
