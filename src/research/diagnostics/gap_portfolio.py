"""Portfolio-level evaluation helpers for gap-adjusted research diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def _portfolio_prediction(
    weights: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
) -> tuple[float, float, float]:
    """Return predicted mean, volatility, and information ratio."""
    pred_mean = float(np.sum(weights * mean))
    pred_var = float(weights.T @ covariance @ weights)
    pred_vol = float(np.sqrt(np.maximum(pred_var, 1e-10)))
    pred_ir = pred_mean / pred_vol if pred_vol > 0 else 0.0
    return pred_mean, pred_vol, pred_ir


def evaluate_gap_portfolio(
    *,
    weights: np.ndarray,
    realized_returns: np.ndarray,
    mu_raw: np.ndarray,
    omega_raw: np.ndarray,
    mu_gap: np.ndarray,
    omega_gap: np.ndarray,
    slippage_bps_per_side: float,
    turnover: float,
) -> dict[str, float]:
    """Evaluate realized and predicted portfolio quantities for one date.

    This function contains only numeric portfolio evaluation.  Baseline signal
    construction and gap summary fields remain orchestration inputs and can be
    added to the returned record by the caller.
    """
    weights = np.asarray(weights, dtype=float)
    realized_returns = np.asarray(realized_returns, dtype=float)
    mu_raw = np.asarray(mu_raw, dtype=float)
    omega_raw = np.asarray(omega_raw, dtype=float)
    mu_gap = np.asarray(mu_gap, dtype=float)
    omega_gap = np.asarray(omega_gap, dtype=float)

    gross_exposure = float(np.sum(np.abs(weights)))
    gross_return = float(np.sum(weights * realized_returns))
    cost = float(2.0 * (slippage_bps_per_side / 10000.0) * gross_exposure)
    raw_mean, raw_vol, raw_ir = _portfolio_prediction(weights, mu_raw, omega_raw)
    gap_mean, gap_vol, gap_ir = _portfolio_prediction(weights, mu_gap, omega_gap)
    return {
        "gross_return": gross_return,
        "net_return": gross_return - cost,
        "cost": cost,
        "pred_mean_raw": raw_mean,
        "pred_vol_raw": raw_vol,
        "pred_ir_raw": raw_ir,
        "pred_mean_gap": gap_mean,
        "pred_vol_gap": gap_vol,
        "pred_ir_gap": gap_ir,
        "pred_ir_gap_realized_cost_diagnostic": (
            (gap_mean - cost) / gap_vol if gap_vol > 0 else 0.0
        ),
        "turnover": float(turnover),
        "gross_exposure": gross_exposure,
        "net_exposure": float(np.sum(weights)),
    }


def add_baseline_diagnostics(
    record: dict[str, Any],
    *,
    weights: np.ndarray,
    mu_gap: np.ndarray,
    omega_gap: np.ndarray,
    baseline_cost_bps_per_gross: float,
    baseline_gross: float | None = None,
) -> dict[str, Any]:
    """Append baseline post-gap IR metrics to a portfolio evaluation record."""
    baseline_weights = np.asarray(weights, dtype=float)
    pred_mean = float(np.sum(baseline_weights * mu_gap))
    pred_var = float(baseline_weights @ omega_gap @ baseline_weights)
    pred_vol = float(np.sqrt(np.maximum(pred_var, 1e-10)))
    gross = (
        float(np.sum(np.abs(baseline_weights))) if baseline_gross is None else float(baseline_gross)
    )
    ex_ante_cost = gross * (baseline_cost_bps_per_gross / 10000.0)
    record.update(
        {
            "pred_mean_gap_baseline": pred_mean,
            "pred_vol_gap_baseline": pred_vol,
            "pred_ir_gap_baseline_cost": (
                (pred_mean - ex_ante_cost) / pred_vol if pred_vol > 1e-6 else 0.0
            ),
        }
    )
    return record


def build_portfolio_diagnostic_record(
    *,
    signal_date: Any,
    trade_date: str,
    evaluation: Mapping[str, Any],
    gap_metrics: Mapping[str, Any],
    baseline_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Combine stable identifiers, evaluation, and diagnostic metrics."""
    record: dict[str, Any] = {
        "signal_date": signal_date,
        "trade_date": trade_date,
    }
    record.update(evaluation)
    record.update(baseline_metrics)
    record.update(gap_metrics)
    return record


__all__ = [
    "add_baseline_diagnostics",
    "build_portfolio_diagnostic_record",
    "evaluate_gap_portfolio",
]
