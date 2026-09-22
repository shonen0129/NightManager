"""Signal enhancement module for Phase 2A (multi-horizon blend) and Phase 2D (rank reversal overlay).

Provides functions to enhance mu_over_sigma scores with:
  1. Multi-horizon blending: combines scores from h=1, h=3, h=5 gap matrices
  2. Rank reversal overlay: adds cross-sectional rank reversal signal

All functions are lookahead-safe: they only use data from dates strictly before
the trade date.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from leadlag.utils.gap_matrix_io import load_gap_bundle, load_gap_npy

logger = logging.getLogger(__name__)


def cross_sectional_zscore(arr: np.ndarray) -> np.ndarray:
    """Z-score an array cross-sectionally (median-centered, std-normalized).

    Args:
        arr: 1D array of scores.

    Returns:
        Z-scored array. Constant arrays return zeros.
    """
    arr = np.asarray(arr, dtype=float)
    med = float(np.median(arr))
    centered = arr - med
    std = float(np.std(centered))
    if std < 1e-8:
        return np.zeros_like(arr)
    return centered / std


def apply_multi_horizon_blend(
    scores_h1: np.ndarray,
    gap_input_dir: Path | None,
    date_str: str,
    horizons: tuple[int, ...],
    weights: tuple[float, ...],
    mu_pattern: str = "matrices/mu_gap_h{h}_{date}.npy",
    omega_pattern: str = "matrices/omega_gap_h{h}_{date}.npy",
    expected_identity: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Blend mu_over_sigma scores from multiple horizons.

    For each horizon h > 1, loads mu_gap_h and Omega_gap_h, computes
    mu_over_sigma scores, z-scores them cross-sectionally, and blends
    with the h=1 scores using the provided weights.

    If h>1 matrices are not available, falls back to h1-only scores
    (graceful degradation).

    Args:
        scores_h1: Base mu_over_sigma scores from h=1 (shape n_j).
        gap_input_dir: Directory containing gap matrices, or None.
        date_str: Trade date string.
        horizons: Tuple of horizons in days (e.g., (1, 3, 5)).
        weights: Tuple of blend weights (must sum to ~1.0).
        mu_pattern: File pattern for h>1 mu matrices.
        omega_pattern: File pattern for h>1 Omega matrices.
        expected_identity: Run-owned bundle identity.  When omitted, the
            bundle must still contain all identity fields; callers should pass
            this mapping when they own the execution frame.

    Returns:
        Tuple of (blended_scores, alerts).
    """
    alerts: list[str] = []
    n_j = len(scores_h1)

    if gap_input_dir is None:
        alerts.append("Multi-horizon blend: gap_input_dir is None, using h1 only.")
        return scores_h1.copy(), alerts

    # Z-score the h1 scores
    z_h1 = cross_sectional_zscore(scores_h1)
    blended = np.zeros(n_j)
    total_weight = 0.0

    for h, w in zip(horizons, weights):
        if h == 1:
            blended += w * z_h1
            total_weight += w
            continue

        mu_h, omega_h, metadata_h, gap_alerts = load_gap_bundle(
            gap_input_dir,
            date_str,
            mu_pattern=mu_pattern,
            omega_pattern=omega_pattern,
            pattern_kwargs={"h": h},
            require_metadata=True,
            expected_identity=expected_identity,
            require_identity=True,
        )
        if mu_h is None or omega_h is None or any(
            alert.startswith("[FATAL]") for alert in gap_alerts
        ):
            alerts.extend(gap_alerts)
            alerts.append(
                f"Multi-horizon blend: h={h} matrices unavailable or unprovenanced, "
                f"skipping (weight={w:.2f} redistributed to h1)."
            )
            continue
        if metadata_h is None or "gap_inputs_version" not in metadata_h:
            alerts.append(
                f"Multi-horizon blend: h={h} bundle lacks gap_inputs_version, "
                f"skipping (weight={w:.2f} redistributed to h1)."
            )
            continue
        alerts.extend(gap_alerts)

        # Compute mu_over_sigma for this horizon
        sigma_h = np.sqrt(np.maximum(np.diag(omega_h), 1e-6))
        scores_h = mu_h / sigma_h
        z_h = cross_sectional_zscore(scores_h)
        blended += w * z_h
        total_weight += w
        logger.debug("Multi-horizon blend: h=%d loaded, weight=%.2f", h, w)

    if total_weight < 1e-8:
        alerts.append("Multi-horizon blend: no horizons loaded, using h1 scores directly.")
        return scores_h1.copy(), alerts

    # Normalize by total weight used
    blended = blended / total_weight

    # Rescale to match h1 score magnitude (preserve original scale for portfolio construction)
    h1_std = np.std(scores_h1)
    blended_std = np.std(blended)
    if blended_std > 1e-8:
        blended = blended * (h1_std / blended_std)

    # Shift to match h1 median
    blended = blended + np.median(scores_h1)

    return blended, alerts


def apply_rank_reversal_overlay(
    scores: np.ndarray,
    gap_input_dir: Path | None,
    date_str: str,
    weight: float = 0.05,
    file_pattern: str = "matrices/rank_reversal_{date}.npy",
    rank_reversal_signal: np.ndarray | None = None,
    allow_implicit_io: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Apply cross-sectional rank reversal overlay to scores.

    Loads the pre-computed rank reversal signal, z-scores both the base
    scores and the rank reversal signal, and blends them:
        final = z(base) + weight * z(rank_reversal)

    Then rescales to preserve the original score magnitude.

    If the rank reversal file is not found, returns scores unchanged
    (graceful degradation).

    Args:
        scores: Base mu_over_sigma scores (shape n_j).
        gap_input_dir: Directory containing rank reversal file, or None.
        date_str: Trade date string.
        weight: Blend weight for rank reversal signal.
        file_pattern: File pattern with {date} placeholder.
        allow_implicit_io: Whether a missing explicit signal may be loaded from
            ``gap_input_dir``.  Strict runs disable this to keep all inputs
            owned by the run snapshot.

    Returns:
        Tuple of (enhanced_scores, alerts).
    """
    alerts: list[str] = []

    if rank_reversal_signal is None and gap_input_dir is None:
        alerts.append("Rank reversal overlay: gap_input_dir is None, skipping.")
        return scores.copy(), alerts

    if weight <= 0.0:
        return scores.copy(), alerts

    rr_signal = rank_reversal_signal
    if rr_signal is None:
        assert gap_input_dir is not None
        if not allow_implicit_io:
            alerts.append(
                "Rank reversal overlay: explicit signal missing in strict run snapshot, skipping."
            )
            return scores.copy(), alerts
        rr_signal, _ = load_gap_npy(gap_input_dir, date_str, file_pattern)
    if rr_signal is None:
        alerts.append("Rank reversal overlay: signal file not found, skipping.")
        return scores.copy(), alerts

    n_j = len(scores)
    if len(rr_signal) != n_j:
        alerts.append(
            f"Rank reversal overlay: shape mismatch ({len(rr_signal)} vs {n_j}), skipping."
        )
        return scores.copy(), alerts

    # Z-score both signals
    z_base = cross_sectional_zscore(scores)
    z_rr = cross_sectional_zscore(rr_signal)

    # Blend
    blended_z = z_base + weight * z_rr

    # Rescale to match original score magnitude
    orig_std = np.std(scores)
    blended_std = np.std(blended_z)
    if blended_std > 1e-8:
        blended_z = blended_z * (orig_std / blended_std)

    # Shift to match original median
    enhanced = blended_z + np.median(scores)

    logger.debug(
        "Rank reversal overlay applied: weight=%.2f, corr(z_base, z_rr)=%.3f",
        weight,
        float(np.corrcoef(z_base, z_rr)[0, 1]) if np.std(z_rr) > 1e-8 else 0.0,
    )

    return enhanced, alerts
