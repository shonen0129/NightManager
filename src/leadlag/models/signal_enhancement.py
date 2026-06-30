"""Signal enhancement module for Phase 2A (multi-horizon blend) and Phase 2D (rank reversal overlay).

Provides functions to enhance mu_over_sigma scores with:
  1. Multi-horizon blending: combines scores from h=1, h=3, h=5 gap matrices
  2. Rank reversal overlay: adds cross-sectional rank reversal signal

All functions are lookahead-safe: they only use data from dates strictly before
the trade date.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def cross_sectional_zscore(arr: np.ndarray) -> np.ndarray:
    """Z-score an array cross-sectionally (median-centered, std-normalized).

    Args:
        arr: 1D array of scores.

    Returns:
        Z-scored array. Constant arrays return zeros.
    """
    arr = np.asarray(arr, dtype=float)
    med = np.median(arr)
    centered = arr - med
    std = np.std(centered)
    if std < 1e-8:
        return np.zeros_like(arr)
    return centered / std


def load_horizon_gap_matrices(
    gap_input_dir: Path,
    date_str: str,
    horizon: int,
    mu_pattern: str = "matrices/mu_gap_h{h}_{date}.npy",
    omega_pattern: str = "matrices/omega_gap_h{h}_{date}.npy",
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load gap matrices for a specific horizon h > 1.

    Files follow the pattern: mu_gap_h{h}_{YYYYMMDD}.npy

    Args:
        gap_input_dir: Root directory of gap distribution output.
        date_str: Trade date (any parseable format).
        horizon: Horizon in days (e.g., 3, 5).
        mu_pattern: File pattern for mu matrices.
        omega_pattern: File pattern for Omega matrices.

    Returns:
        Tuple of (mu_gap_h, Omega_gap_h). Both None if files missing.
    """
    date_numeric = pd.to_datetime(date_str).strftime("%Y%m%d")
    mu_file = gap_input_dir / mu_pattern.format(h=horizon, date=date_numeric)
    omega_file = gap_input_dir / omega_pattern.format(h=horizon, date=date_numeric)

    if not mu_file.exists() or not omega_file.exists():
        return None, None

    return np.load(mu_file), np.load(omega_file)


def apply_multi_horizon_blend(
    scores_h1: np.ndarray,
    gap_input_dir: Path | None,
    date_str: str,
    horizons: tuple[int, ...],
    weights: tuple[float, ...],
    mu_pattern: str = "matrices/mu_gap_h{h}_{date}.npy",
    omega_pattern: str = "matrices/omega_gap_h{h}_{date}.npy",
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

        mu_h, omega_h = load_horizon_gap_matrices(
            gap_input_dir, date_str, h, mu_pattern, omega_pattern
        )
        if mu_h is None or omega_h is None:
            alerts.append(
                f"Multi-horizon blend: h={h} matrices not found, skipping (weight={w:.2f} redistributed to h1)."
            )
            continue

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


def load_rank_reversal_signal(
    gap_input_dir: Path,
    date_str: str,
    file_pattern: str = "matrices/rank_reversal_{date}.npy",
) -> np.ndarray | None:
    """Load pre-computed rank reversal signal for the given date.

    The rank reversal signal is computed by the upstream gap distribution
    process and saved as a .npy file. It represents the 1-day change in
    cross-sectional rank of JP open-close returns (shifted by 1 day,
    negated for reversal).

    Args:
        gap_input_dir: Root directory of gap distribution output.
        date_str: Trade date string.
        file_pattern: File pattern with {date} placeholder.

    Returns:
        Rank reversal signal array (shape n_j), or None if file missing.
    """
    date_numeric = pd.to_datetime(date_str).strftime("%Y%m%d")
    file_path = gap_input_dir / file_pattern.format(date=date_numeric)

    if not file_path.exists():
        return None

    return np.load(file_path)


def apply_rank_reversal_overlay(
    scores: np.ndarray,
    gap_input_dir: Path | None,
    date_str: str,
    weight: float = 0.05,
    file_pattern: str = "matrices/rank_reversal_{date}.npy",
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

    Returns:
        Tuple of (enhanced_scores, alerts).
    """
    alerts: list[str] = []

    if gap_input_dir is None:
        alerts.append("Rank reversal overlay: gap_input_dir is None, skipping.")
        return scores.copy(), alerts

    if weight <= 0.0:
        return scores.copy(), alerts

    rr_signal = load_rank_reversal_signal(gap_input_dir, date_str, file_pattern)
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
