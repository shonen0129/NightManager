from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from leadlag.reporting.metrics import MetricsSpec, calculate_metrics


def test_max_drawdown_includes_initial_wealth() -> None:
    metrics = calculate_metrics(pd.Series([-0.10, 0.0]))
    assert metrics["MDD"] == pytest.approx(-0.10)
    assert metrics["Total Return"] == pytest.approx(-0.10)


def test_metrics_default_frequency_is_daily_even_with_datetime_index() -> None:
    returns = pd.Series(
        [0.01, -0.005],
        index=pd.to_datetime(["2026-01-02", "2026-02-02"]),
    )
    daily = calculate_metrics(returns)
    expected_std = float(returns.std(ddof=1))
    assert daily["Sharpe"] == pytest.approx(float(returns.mean()) / expected_std * np.sqrt(245))
    assert daily["RISK"] == pytest.approx(expected_std * np.sqrt(245))


def test_metrics_spec_makes_flat_day_treatment_explicit() -> None:
    returns = pd.Series([0.01, 0.0, -0.01])
    with_flat = calculate_metrics(returns, spec=MetricsSpec(include_flat_days=True))
    without_flat = calculate_metrics(returns, spec=MetricsSpec(include_flat_days=False))
    assert with_flat["Total Return"] == pytest.approx(without_flat["Total Return"])
    assert with_flat["RISK"] != pytest.approx(without_flat["RISK"])


def test_metrics_spec_rejects_monthly_non_monthly_annualization() -> None:
    with pytest.raises(ValueError, match="monthly"):
        MetricsSpec(frequency="monthly", annualization_periods=245)
