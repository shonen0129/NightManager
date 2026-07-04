"""Portfolio optimizer: converts signals to target weights."""

from __future__ import annotations

import numpy as np

from leadlag.core.types import GrossExposureAdjustment


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


# ---------------------------------------------------------------------------
# Production v2 helpers (mu_over_sigma ranking + RuleD gross scaling)
# ---------------------------------------------------------------------------


def solve_baseline_style(
    scores: np.ndarray,
    long_idx: np.ndarray,
    short_idx: np.ndarray,
    baseline_gross: float = 2.0,
) -> np.ndarray:
    """Compute baseline_style weights normalised to *baseline_gross*.

    Long side sums to +baseline_gross/2; short side sums to -baseline_gross/2.
    Weights are proportional to |score - median(score)| on each side.

    Args:
        scores: Raw signal scores (shape n_j).
        long_idx: Indices of long positions.
        short_idx: Indices of short positions.
        baseline_gross: Target gross exposure (default 2.0).

    Returns:
        Weight array of shape n_j.
    """
    n = len(scores)
    w = np.zeros(n)
    med_score = np.median(scores)
    scores_centered = scores - med_score

    long_raw = np.maximum(scores_centered[long_idx], 1e-12)
    long_denom = np.sum(long_raw)
    if long_denom > 0:
        w[long_idx] = (baseline_gross / 2.0) * (long_raw / long_denom)

    short_raw = np.maximum(-scores_centered[short_idx], 1e-12)
    short_denom = np.sum(short_raw)
    if short_denom > 0:
        w[short_idx] = -(baseline_gross / 2.0) * (short_raw / short_denom)

    return w


def get_rolling_pit_bin(
    history_ir: np.ndarray,
    current_ir: float,
    rolling_window: int = 252,
    low_pct: float = 33.3333,
    high_pct: float = 66.6667,
    mult_low: float = 0.75,
    mult_mid: float = 1.00,
    mult_high: float = 1.00,
) -> tuple[str, float, float, float]:
    """Assign *current_ir* to a PIT tertile bin using strictly historical data.

    Implements RuleD dynamic gross scaling:
      - Low tertile  → gross multiplier 0.75
      - Mid/High     → gross multiplier 1.00

    Args:
        history_ir: Historical ex-ante IR series (shape T).
        current_ir: Current period ex-ante IR (point-in-time; must not include
            today's realised return to avoid lookahead).
        rolling_window: Number of historical observations to use.
        low_pct: Lower percentile boundary (default 33.33).
        high_pct: Upper percentile boundary (default 66.67).
        mult_low: Gross multiplier for Low bin.
        mult_mid: Gross multiplier for Mid bin.
        mult_high: Gross multiplier for High bin.

    Returns:
        Tuple of (bin_label, low_threshold, high_threshold, multiplier).
        Falls back to ('Medium', nan, nan, 1.00) when history is insufficient.
    """
    history_valid = history_ir[np.isfinite(history_ir)]

    if len(history_valid) < rolling_window:
        return "Medium", float("nan"), float("nan"), mult_mid

    history_slice = history_valid[-rolling_window:]
    low_thresh = float(np.percentile(history_slice, low_pct))
    high_thresh = float(np.percentile(history_slice, high_pct))

    if current_ir <= low_thresh:
        return "Low", low_thresh, high_thresh, mult_low
    elif current_ir >= high_thresh:
        return "High", low_thresh, high_thresh, mult_high
    else:
        return "Medium", low_thresh, high_thresh, mult_mid
