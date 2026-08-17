"""Convenience helpers for recording experiments to the registry.

This module sits on top of ``research.experiment_registry`` and provides the
canned hooks that experiment scripts use to append an ``ExperimentRecord``
to ``var/experiments/registry.jsonl``.
"""

from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _extract_metrics(results: dict[str, Any] | None) -> dict[str, Any]:
    """Best-effort extraction of net Sharpe / MDD / turnover / fallback rate."""
    if results is None:
        return {}

    metrics: dict[str, Any] = {}
    daily_returns = results.get("daily_returns")
    fallback = results.get("daily_fallback")
    turnover = results.get("daily_turnover")
    gross_exps = results.get("daily_gross_exps")

    if isinstance(daily_returns, (pd.Series, np.ndarray)):
        returns = np.asarray(daily_returns, dtype=float)
        if fallback is not None:
            mask = ~np.asarray(fallback, dtype=bool)
            returns = returns[mask]
        if len(returns) > 1 and np.std(returns, ddof=1) > 1e-12:
            metrics["net_sharpe"] = float(
                np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(252)
            )
        else:
            metrics["net_sharpe"] = 0.0
        metrics["n_observations"] = int(len(returns))

        cum = np.cumsum(returns)
        running_max = np.maximum.accumulate(cum)
        mdd = float(np.min(cum - running_max)) if len(cum) > 0 else 0.0
        metrics["max_dd"] = mdd
        metrics["total_return"] = float(np.sum(returns))

    if isinstance(turnover, (pd.Series, np.ndarray)):
        to_arr = np.asarray(turnover, dtype=float)
        if fallback is not None:
            to_arr = to_arr[~np.asarray(fallback, dtype=bool)]
        if len(to_arr) > 0:
            metrics["turnover"] = float(np.mean(to_arr))

    if isinstance(gross_exps, (pd.Series, np.ndarray)):
        gross_arr = np.asarray(gross_exps, dtype=float)
        if fallback is not None:
            gross_arr = gross_arr[~np.asarray(fallback, dtype=bool)]
        if len(gross_arr) > 0:
            metrics["avg_gross"] = float(np.mean(gross_arr))

    if isinstance(fallback, (pd.Series, np.ndarray)):
        fb_arr = np.asarray(fallback, dtype=bool)
        if len(fb_arr) > 0:
            metrics["fallback_rate"] = float(np.mean(fb_arr))

    metrics["returns"] = (
        daily_returns.dropna().tolist()
        if isinstance(daily_returns, pd.Series)
        else (
            np.asarray(daily_returns, dtype=float).tolist()
            if daily_returns is not None
            else []
        )
    )
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
) -> ExperimentRecord:
    """Record a backtest experiment to the registry.

    Args:
        name: Experiment / script name.
        hypothesis: Free-text hypothesis.
        app_config: AppConfig (or dict) to freeze as parameters.
        results: Backtest results dict from ``BacktestEngine.run_v2_backtest``
            or ``run_v1_backtest``.
        extra_metrics: Additional metrics not inferable from *results*.
        decision: ADOPTION / REJECTION decision.
        reason: Optional reason for the decision.
        report_path: Path to a markdown report.
        tags: Optional tags appended to the hypothesis.
        registry_path: Override the default ``var/experiments/registry.jsonl``.

    Returns:
        The recorded ``ExperimentRecord``.
    """
    registry = ExperimentRegistry(registry_path or default_registry_path())
    params = (
        app_config.model_dump()
        if isinstance(app_config, AppConfig)
        else (dict(app_config) if app_config is not None else {})
    )

    metrics = _extract_metrics(results)
    if extra_metrics:
        metrics.update(extra_metrics)

    # Trial count is the number of records already in this name family + 1.
    # This is a conservative proxy for the number of trials tried.
    existing = list(registry.iter_records(name=name))
    metrics["trials"] = len(existing) + 1

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
) -> ExperimentRecord:
    """Record a generic experiment without a full backtest result dict."""
    registry = ExperimentRegistry(registry_path or default_registry_path())
    existing = list(registry.iter_records(name=name))
    metrics = dict(metrics)
    if "trials" not in metrics:
        metrics["trials"] = len(existing) + 1

    if "n_observations" not in metrics and "returns" in metrics:
        metrics["n_observations"] = len(metrics["returns"])

    record = ExperimentRecord(
        name=name,
        hypothesis=hypothesis,
        parameters=parameters or {},
        metrics=metrics,
        decision=decision,
        report_path=str(report_path) if report_path is not None else None,
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
