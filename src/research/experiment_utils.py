"""Convenience helpers for recording experiments to the registry.

This module sits on top of ``research.experiment_registry`` and provides the
canned hooks that experiment scripts use to append an ``ExperimentRecord``
to ``var/experiments/registry.jsonl``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from leadlag.config import default_registry_path
from leadlag.config.schemas import AppConfig
from leadlag.experiment_registry import (
    Decision,
    ExperimentRecord,
    ExperimentRegistry,
)
from leadlag.reporting.metrics import MetricsSpec, calculate_metrics

logger = logging.getLogger(__name__)
_REDACTED = "[REDACTED]"
_SECRET_PARAMETER_NAMES = {
    "api_key",
    "api_password",
    "api_token",
    "auth_id",
    "authorization",
    "credential",
    "credentials",
    "password",
    "private_key",
    "private_key_path",
    "secret",
    "secret_key",
    "second_password",
    "token",
}
_NORMALIZED_SECRET_PARAMETER_NAMES = {
    "".join(character for character in name if character.isalnum())
    for name in _SECRET_PARAMETER_NAMES
}


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _is_secret_parameter(name: object) -> bool:
    normalized = "".join(character for character in str(name).lower() if character.isalnum())
    return (
        normalized in _NORMALIZED_SECRET_PARAMETER_NAMES
        or normalized.endswith(("password", "token", "secret"))
        or "credential" in normalized
    )


def _safe_parameter_value(value: Any, key: object | None = None) -> Any:
    """Copy JSON-like configuration values while removing credentials."""
    if key is not None and _is_secret_parameter(key):
        return _REDACTED
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(name): _safe_parameter_value(item, name) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_parameter_value(item) for item in value]
    return value


def _safe_app_parameters(app_config: AppConfig | dict[str, Any] | None) -> dict[str, Any]:
    """Return research-relevant settings without broker credentials."""
    if isinstance(app_config, AppConfig):
        return {
            "snapshot_schema": "research-safe-v1",
            "strategy": {
                "start_date": app_config.strategy.start_date,
                "side_leverage": app_config.strategy.side_leverage,
            },
            "risk": app_config.risk.model_dump(mode="json"),
            "v2": app_config.v2.model_dump(mode="json"),
            "gap_distribution_dir": app_config.gap_distribution_dir,
        }
    safe = _safe_parameter_value(app_config or {})
    return safe if isinstance(safe, dict) else {}


def _extract_metrics(
    results: dict[str, Any] | None,
    *,
    metrics_spec: MetricsSpec | None = None,
) -> dict[str, Any]:
    """Extract daily performance metrics under the shared MetricsSpec."""
    if results is None:
        return {}

    metrics: dict[str, Any] = {}
    daily_returns = results.get("daily_returns")
    fallback = results.get("daily_fallback")
    turnover = results.get("daily_turnover")
    gross_exps = results.get("daily_gross_exps")

    if isinstance(daily_returns, (pd.Series, np.ndarray)):
        returns = np.asarray(daily_returns, dtype=float)
        spec = metrics_spec or MetricsSpec()
        if spec.frequency != "daily":
            raise ValueError("experiment registry metrics require daily-frequency returns")
        metrics["metric_schema_version"] = "daily-v1"
        metrics["net_sharpe_frequency"] = "annual"
        metrics["trading_days_per_year"] = int(spec.annualization_periods)
        if not np.isfinite(returns).all():
            metrics["metric_status"] = "invalid_non_finite_returns"
            metrics["n_observations_expected"] = int(len(returns))
            metrics["missing_return_count"] = int((~np.isfinite(returns)).sum())
            metrics["returns"] = [float(value) if np.isfinite(value) else None for value in returns]
        else:
            metrics["metric_status"] = "valid"
            summary = calculate_metrics(pd.Series(daily_returns), spec=spec)
            metrics["net_sharpe"] = (
                float(summary["Sharpe"]) if np.isfinite(summary.get("Sharpe", np.nan)) else None
            )
            metrics["n_observations"] = int(len(returns))
            metrics["max_dd"] = float(summary["MDD"])
            metrics["total_return"] = float(summary["Total Return"])
            metrics["returns"] = returns.tolist()

    if isinstance(turnover, (pd.Series, np.ndarray)):
        to_arr = np.asarray(turnover, dtype=float)
        if len(to_arr) > 0:
            metrics["turnover"] = float(np.mean(to_arr))

    if isinstance(gross_exps, (pd.Series, np.ndarray)):
        gross_arr = np.asarray(gross_exps, dtype=float)
        if len(gross_arr) > 0:
            metrics["avg_gross"] = float(np.mean(gross_arr))

    if isinstance(fallback, (pd.Series, np.ndarray)):
        fb_arr = np.asarray(fallback, dtype=bool)
        if len(fb_arr) > 0:
            metrics["fallback_rate"] = float(np.mean(fb_arr))

    return metrics


def record_backtest_experiment(
    name: str,
    hypothesis: str,
    app_config: AppConfig | dict[str, Any] | None,
    results: dict[str, Any] | None = None,
    extra_metrics: dict[str, Any] | None = None,
    decision: Decision = Decision.PENDING,
    reason: str | None = None,
    report_path: str | Path | None = None,
    tags: list[str] | None = None,
    registry_path: str | Path | None = None,
    metrics_spec: MetricsSpec | None = None,
    study_id: str | None = None,
) -> ExperimentRecord:
    """Record a backtest experiment to the registry.

    Args:
        name: Experiment / script name.
        hypothesis: Free-text hypothesis.
        app_config: AppConfig (or dict) to freeze as safe research parameters.
        results: Backtest results dict from ``BacktestEngine.run_v2_backtest``
            or ``run_v1_backtest``.
        extra_metrics: Additional metrics not inferable from *results*.
        decision: ADOPTION / REJECTION decision.
        reason: Optional reason for the decision.
        report_path: Path to a markdown report.
        tags: Optional tags appended to the hypothesis.
        registry_path: Override the default ``var/experiments/registry.jsonl``.
        metrics_spec: Daily return definition. Defaults to the shared 245-day
            annualization and includes flat days.
        study_id: Stable grouping for trials that share one hypothesis family.

    Returns:
        The recorded ``ExperimentRecord``.
    """
    registry = ExperimentRegistry(registry_path or default_registry_path())
    params = _safe_app_parameters(app_config)

    if metrics_spec is None and extra_metrics and "trading_days_per_year" in extra_metrics:
        metrics_spec = MetricsSpec(
            annualization_periods=int(extra_metrics["trading_days_per_year"]),
            include_flat_days=bool(extra_metrics.get("include_flat_days", True)),
        )
    metrics = _extract_metrics(results, metrics_spec=metrics_spec)
    if extra_metrics:
        metrics.update(extra_metrics)

    # Preserve an explicit count when the experiment belongs to a wider study
    # or when the caller has already tallied related trials. Otherwise, use
    # the number of prior records with this name as a conservative proxy.
    if "trials" not in metrics:
        metrics["trials"] = (
            registry.count_trials(study_id=study_id) + 1
            if study_id is not None
            else sum(1 for rec in registry.iter_records(name=name) if rec.correction_of is None) + 1
        )

    if reason:
        metrics["reason"] = reason

    hypothesis_with_tags = hypothesis
    if tags:
        hypothesis_with_tags = f"{hypothesis} [{' '.join(tags)}]"

    record = ExperimentRecord(
        name=name,
        hypothesis=hypothesis_with_tags,
        parameters=params,
        metrics=metrics,
        decision=decision,
        report_path=str(report_path) if report_path is not None else None,
        study_id=study_id,
    )
    record.end_time = _utc_now()
    record.metrics["deflated_sharpe"] = record.deflated_sharpe()
    registry.record(record)
    logger.info(
        "Recorded experiment %s (decision=%s, dsr=%s) to %s",
        record.name,
        record.decision.value,
        record.deflated_sharpe(),
        registry.path,
    )
    return record


def record_simple_experiment(
    name: str,
    hypothesis: str,
    parameters: dict[str, Any] | None,
    metrics: dict[str, Any],
    decision: Decision = Decision.PENDING,
    report_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    study_id: str | None = None,
) -> ExperimentRecord:
    """Record a generic experiment without a full backtest result dict."""
    registry = ExperimentRegistry(registry_path or default_registry_path())
    metrics = dict(metrics)
    if "trials" not in metrics:
        metrics["trials"] = (
            registry.count_trials(study_id=study_id) + 1
            if study_id is not None
            else sum(1 for rec in registry.iter_records(name=name) if rec.correction_of is None) + 1
        )

    if "n_observations" not in metrics and "returns" in metrics:
        metrics["n_observations"] = len(metrics["returns"])

    record = ExperimentRecord(
        name=name,
        hypothesis=hypothesis,
        parameters=_safe_parameter_value(parameters or {}),
        metrics=metrics,
        decision=decision,
        report_path=str(report_path) if report_path is not None else None,
        study_id=study_id,
    )
    record.end_time = _utc_now()
    record.metrics["deflated_sharpe"] = record.deflated_sharpe()
    registry.record(record)
    logger.info(
        "Recorded experiment %s (decision=%s, dsr=%s) to %s",
        record.name,
        record.decision.value,
        record.deflated_sharpe(),
        registry.path,
    )
    return record
