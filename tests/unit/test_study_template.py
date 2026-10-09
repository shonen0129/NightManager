from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from leadlag.config.schemas import AppConfig
from leadlag.experiment_registry import ExperimentRegistry

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "study_template", ROOT / "src/research/scripts/experiments/_template.py"
)
assert SPEC is not None and SPEC.loader is not None
TEMPLATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TEMPLATE)


@pytest.mark.parametrize("fails", [False, True])
def test_template_records_start_before_evaluation_and_outcome_even_on_failure(
    tmp_path, monkeypatch, fails
):
    plan = {
        "study_id": "family",
        "hypothesis": "test",
        "candidates": ["a"],
        "history_complete": True,
        "historical_trials_lower_bound": 0,
        "protocol": {
            "target_schema": "9:10-close-v1",
            "cost_schema": "four-costs-v1",
            "is_period": {"start": "2015-01-05", "end": "2024-12-20"},
            "oos_period": {"start": "2024-12-23", "end": "2026-09-18"},
            "purge": {"sessions": 1, "embargo_sessions": 0, "rationale": "one-day labels"},
            "selection_rule": "preset",
            "metrics_spec": {
                "frequency": "daily",
                "include_flat_days": True,
                "annualization_periods": 252,
            },
        },
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    registry = ExperimentRegistry(tmp_path / "registry.jsonl")
    monkeypatch.setattr(TEMPLATE, "ROOT", tmp_path)
    monkeypatch.setattr(TEMPLATE, "load_config_from_yaml", lambda _: AppConfig())
    monkeypatch.setattr(
        TEMPLATE, "load_df_exec_from_local_cache", lambda: pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    )
    monkeypatch.setattr(
        TEMPLATE.sys,
        "argv",
        [
            "template",
            "--study-plan",
            str(path),
            "--candidate-id",
            "a",
            "--code-hash",
            "a" * 64,
            "--registry",
            str(registry.path),
        ],
    )

    def evaluate(**kwargs):
        assert registry.count_trials(study_id="family") == 1
        assert registry.study_events()[1]["event_type"] == "trial_started"
        assert len(registry.study_events()[1]["data_hash"]) == 64
        assert kwargs["start_date"] == "2024-12-23"
        assert kwargs["end_date"] == "2026-09-18"
        if fails:
            raise RuntimeError("SYNTHETIC_SECRET")
        return {"daily_returns": pd.Series([0.01, -0.02, 0.03, 0.02])}

    monkeypatch.setattr(TEMPLATE.BacktestEngine, "run_v2_backtest", evaluate)
    if fails:
        with pytest.raises(RuntimeError):
            TEMPLATE.main()
    else:
        assert TEMPLATE.main() == 0
    records = list(registry)
    assert len(records) == 1
    assert records[0].trial_status == ("failed" if fails else "completed")
    assert registry.count_trials() == 1
    assert "SYNTHETIC_SECRET" not in registry.path.read_text()
    if not fails:
        assert records[0].metrics["trading_days_per_year"] == 252
        assert records[0].metrics["deflated_sharpe"] is not None
