"""Portfolio optimizer: converts signals to target weights."""

from __future__ import annotations

import numpy as np

from leadlag.core import signal as signals
from leadlag.core.types import GrossExposureAdjustment


def compute_trade_decision(
    signal: np.ndarray,
    sigma_s: float,
    n_j: int,
    q: float,
    weight_mode: str = "signal",
    dispersion_filter: bool = True,
    dispersion_metric: str = "long_short_mean_gap",
    dispersion_history: list[float] | None = None,
    gap_tolerant: bool = False,
    gamma: float = 0.5,
    jp_close_t: np.ndarray | None = None,
    jp_open_t1: np.ndarray | None = None,
    signal_mode: str = "gap_residual",
    enforce_sign: bool = False,
) -> dict:
    """Compute the full trade decision from a signal.

    Returns:
        Dict with: weights, raw_weights, scale, signal, dispersion_indicator, sigma_s
    """
    if dispersion_history is None:
        dispersion_history = []

    if gap_tolerant and jp_close_t is not None and jp_open_t1 is not None:
        # For gap_tolerant, first get base weights then filter
        base_weights = signals.build_weights(signal, q, n_j, weight_mode, enforce_sign)
        weights, long_exec, short_exec, executed = signals.apply_gap_tolerant_filter(
            signal,
            sigma_s,
            base_weights,
            jp_close_t,
            jp_open_t1,
            gamma,
            q,
            n_j,
            weight_mode,
            enforce_sign,
        )
        dispersion_ind = signals.compute_dispersion_indicator(
            signal,
            q,
            n_j,
            dispersion_metric,
            enforce_sign,
        )
        scale = signals.dispersion_scale(
            dispersion_ind,
            dispersion_history,
            dispersion_filter,
        )
        scaled_weights = weights * scale
    else:
        weights = signals.build_weights(signal, q, n_j, weight_mode, enforce_sign)
        dispersion_ind = signals.compute_dispersion_indicator(
            signal,
            q,
            n_j,
            dispersion_metric,
            enforce_sign,
        )
        scale = signals.dispersion_scale(
            dispersion_ind,
            dispersion_history,
            dispersion_filter,
        )
        scaled_weights = weights * scale

    return {
        "weights": scaled_weights,
        "raw_weights": weights,
        "signal": signal,
        "scale": float(scale),
        "sigma_s": float(sigma_s),
        "dispersion_indicator": float(dispersion_ind),
    }


def adjust_gross_exposure(
    weights: np.ndarray,
    max_gross_exposure: float,
) -> GrossExposureAdjustment:
    """Scale weights down if gross exposure exceeds the limit.

    Returns:
        GrossExposureAdjustment with new weights and metadata.
    """
    gross_before = float(np.sum(np.abs(weights)))
    gross_limit = float(max_gross_exposure)

    # Small epsilon avoids unnecessary scaling from floating-point noise
    if gross_before > gross_limit + 1e-12 and gross_before > 0.0:
        factor = gross_limit / gross_before
        adjusted = weights * factor
        return GrossExposureAdjustment(
            gross_before=gross_before,
            gross_after=float(np.sum(np.abs(adjusted))),
            gross_limit=gross_limit,
            adjustment_factor=factor,
            was_adjusted=True,
        )
    else:
        return GrossExposureAdjustment(
            gross_before=gross_before,
            gross_after=gross_before,
            gross_limit=gross_limit,
            adjustment_factor=1.0,
            was_adjusted=False,
        )


def classify_actions(weights: np.ndarray) -> list[str]:
    """Classify each position as BUY, SELL, or HOLD."""
    return list(
        np.where(
            weights > 1e-12,
            "BUY",
            np.where(weights < -1e-12, "SELL", "HOLD"),
        )
    )
