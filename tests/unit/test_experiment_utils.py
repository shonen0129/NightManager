"""Unit tests for research.experiment_utils convenience helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from leadlag.experiment_registry import Decision, ExperimentRegistry
from research.experiment_utils import (
    record_backtest_experiment,
    record_simple_experiment,
)


def test_record_backtest_experiment_appends_and_counts_trials(tmp_path):
    registry_path = tmp_path / "registry.jsonl"
    n = 50
    results = {
        "daily_returns": pd.Series(np.random.normal(0.001, 0.01, n)),
        "daily_fallback": pd.Series([False] * 40 + [True] * 10),
        "daily_turnover": pd.Series(np.random.uniform(0.0, 1.0, n)),
        "daily_gross_exps": pd.Series(np.random.uniform(0.5, 2.0, n)),
    }
    app_config = {"test_param": 0.5}

    rec1 = record_backtest_experiment(
        name="test_exp",
        hypothesis="Test hypothesis.",
        app_config=app_config,
        results=results,
        registry_path=registry_path,
    )

    assert rec1.decision == Decision.PENDING
    assert rec1.metrics["trials"] == 1
    assert "net_sharpe" in rec1.metrics
    assert rec1.metrics["n_observations"] == 40
    assert rec1.parameters == app_config

    rec2 = record_backtest_experiment(
        name="test_exp",
        hypothesis="Test hypothesis.",
        app_config=app_config,
        results=results,
        registry_path=registry_path,
    )

    assert rec2.metrics["trials"] == 2

    reg = ExperimentRegistry(registry_path)
    assert reg.count_trials() == 2
    records = list(reg.iter_records(name="test_exp"))
    assert len(records) == 2


def test_record_simple_experiment_appends_and_sets_trial_count(tmp_path):
    registry_path = tmp_path / "simple_registry.jsonl"
    metrics = {
        "net_sharpe": 1.2,
        "n_observations": 100,
        "returns": [0.001] * 100,
    }
    parameters = {"variant": "simple"}

    rec = record_simple_experiment(
        name="simple_exp",
        hypothesis="Simple experiment test.",
        parameters=parameters,
        metrics=metrics,
        registry_path=registry_path,
    )

    assert rec.decision == Decision.PENDING
    assert rec.metrics["trials"] == 1
    assert rec.parameters == parameters
    assert rec.metrics["net_sharpe"] == 1.2

    reg = ExperimentRegistry(registry_path)
    assert reg.count_trials() == 1


def test_record_backtest_experiment_extra_metrics_merged(tmp_path):
    registry_path = tmp_path / "registry.jsonl"
    n = 20
    results = {
        "daily_returns": pd.Series(np.random.normal(0.001, 0.01, n)),
        "daily_fallback": pd.Series([False] * n),
        "daily_turnover": pd.Series(np.full(n, 0.5)),
        "daily_gross_exps": pd.Series(np.full(n, 1.5)),
    }

    rec = record_backtest_experiment(
        name="extra_metrics_exp",
        hypothesis="Extra metrics merge test.",
        app_config={},
        results=results,
        extra_metrics={"custom_metric": 42},
        registry_path=registry_path,
    )

    assert rec.metrics["custom_metric"] == 42
    assert rec.metrics["n_observations"] == 20
