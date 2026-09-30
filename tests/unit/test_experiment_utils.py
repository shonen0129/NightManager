"""Unit tests for research.experiment_utils convenience helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from leadlag.config.schemas import AppConfig, KabuApiConfig, TachibanaApiConfig
from leadlag.experiment_registry import Decision, ExperimentRegistry
from leadlag.reporting.metrics import MetricsSpec, calculate_metrics
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
    # Primary registry metrics include flat fallback dates so the observation
    # count and Sharpe denominator match the backtest report.
    assert rec1.metrics["n_observations"] == 50
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


def test_study_id_counts_trials_across_different_record_names(tmp_path):
    registry_path = tmp_path / "study_registry.jsonl"
    first = record_simple_experiment(
        name="candidate_a",
        hypothesis="Same prespecified family.",
        parameters={},
        metrics={"net_sharpe": 0.1, "n_observations": 20},
        registry_path=registry_path,
        study_id="same-family",
    )
    second = record_simple_experiment(
        name="candidate_b",
        hypothesis="Same prespecified family.",
        parameters={},
        metrics={"net_sharpe": 0.2, "n_observations": 20},
        registry_path=registry_path,
        study_id="same-family",
    )

    assert first.metrics["trials"] == 1
    assert second.metrics["trials"] == 2
    assert second.study_id == "same-family"
    assert ExperimentRegistry(registry_path).count_trials(study_id="same-family") == 2


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


def test_record_backtest_experiment_preserves_explicit_trial_count(tmp_path):
    registry_path = tmp_path / "registry.jsonl"
    n = 20
    results = {
        "daily_returns": pd.Series(np.random.normal(0.001, 0.01, n)),
        "daily_fallback": pd.Series([False] * n),
        "daily_turnover": pd.Series(np.full(n, 0.5)),
        "daily_gross_exps": pd.Series(np.full(n, 1.5)),
    }

    rec = record_backtest_experiment(
        name="related_trials_exp",
        hypothesis="Trial count includes related experiments with different names.",
        app_config={},
        results=results,
        extra_metrics={"trials": 24},
        registry_path=registry_path,
    )

    assert rec.metrics["trials"] == 24
    assert rec.metrics["deflated_sharpe"] is not None


def test_record_backtest_experiment_does_not_persist_broker_credentials(tmp_path):
    config = AppConfig(
        kabu=KabuApiConfig(api_token="SYNTHETIC_TOKEN", api_password="SYNTHETIC_PASSWORD"),
        tachibana=TachibanaApiConfig(
            auth_id="SYNTHETIC_AUTH_ID",
            private_key_path="/tmp/SYNTHETIC_PRIVATE_KEY",
            second_password="SYNTHETIC_SECOND_PASSWORD",
        ),
    )
    registry_path = tmp_path / "safe_registry.jsonl"
    record_backtest_experiment(
        name="safe_config_snapshot",
        hypothesis="Store research settings without broker credentials.",
        app_config=config,
        registry_path=registry_path,
    )

    saved = registry_path.read_text()
    for secret in (
        "SYNTHETIC_TOKEN",
        "SYNTHETIC_PASSWORD",
        "SYNTHETIC_AUTH_ID",
        "SYNTHETIC_PRIVATE_KEY",
        "SYNTHETIC_SECOND_PASSWORD",
    ):
        assert secret not in saved
    assert "research-safe-v1" in saved
    assert '"v2"' in saved


def test_registry_metrics_share_annualization_and_drawdown_definition(tmp_path):
    returns = pd.Series([-0.10, 0.0, 0.02, 0.03])
    spec = MetricsSpec(annualization_periods=252)
    registry_path = tmp_path / "metrics_registry.jsonl"

    record = record_backtest_experiment(
        name="canonical_metrics",
        hypothesis="Use the same daily metric definition as reports.",
        app_config={},
        results={"daily_returns": returns},
        registry_path=registry_path,
        metrics_spec=spec,
    )
    expected = calculate_metrics(returns, spec=spec)

    assert record.metrics["net_sharpe"] == pytest.approx(expected["Sharpe"])
    assert record.metrics["max_dd"] == pytest.approx(expected["MDD"])
    assert record.metrics["total_return"] == pytest.approx(expected["Total Return"])
    assert record.metrics["trading_days_per_year"] == 252
    assert record.metrics["net_sharpe_frequency"] == "annual"
    assert record.metrics["n_observations"] == len(returns)


def test_registry_marks_non_finite_return_series_unusable(tmp_path):
    record = record_backtest_experiment(
        name="invalid_metrics",
        hypothesis="Missing daily returns must not produce a valid Sharpe.",
        app_config={},
        results={"daily_returns": pd.Series([0.01, np.nan, -0.02])},
        registry_path=tmp_path / "invalid_registry.jsonl",
    )

    assert record.metrics["metric_status"] == "invalid_non_finite_returns"
    assert record.metrics["missing_return_count"] == 1
    assert record.metrics["returns"] == [0.01, None, -0.02]
    assert "net_sharpe" not in record.metrics
    assert "max_dd" not in record.metrics
    assert record.metrics["deflated_sharpe"] is None


def test_registry_parameter_dict_redacts_nested_credentials(tmp_path):
    record = record_simple_experiment(
        name="redacted_parameters",
        hypothesis="Nested dict configuration is also sanitized.",
        parameters={"risk": {"var_stop": 0.03}, "broker": {"api_token": "SYNTHETIC_TOKEN"}},
        metrics={"trials": 1},
        registry_path=tmp_path / "simple_safe_registry.jsonl",
    )

    saved = (tmp_path / "simple_safe_registry.jsonl").read_text()
    assert "SYNTHETIC_TOKEN" not in saved
    assert record.parameters["broker"]["api_token"] == "[REDACTED]"
