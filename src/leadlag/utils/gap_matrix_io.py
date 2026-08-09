"""Gap matrix I/O helpers shared by production and signal-enhancement modules."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from leadlag.data.gap_store import GapStore, is_gap_store_path
from leadlag.data.tickers import JP_TICKERS
from leadlag.data.validation import DataValidationError, validate_gap_matrices

logger = logging.getLogger(__name__)


def _format_gap_date(date_str: str) -> str:
    """Convert any parseable date string to the YYYYMMDD gap file suffix."""
    return str(pd.to_datetime(date_str).strftime("%Y%m%d"))


def _parse_pattern_to_matrix_type(
    file_pattern: str,
    pattern_kwargs: dict | None,
) -> tuple[str | None, int | None]:
    """Map a filename pattern like ``matrices/mu_gap_h{h}_{date}.npy`` to a
    (matrix_type, horizon) pair for the SQLite store.
    """
    pattern_kwargs = pattern_kwargs or {}
    basename = Path(file_pattern).name
    if "rank_reversal" in basename:
        matrix_type = "rank_reversal"
    elif "mu_gap" in basename:
        matrix_type = "mu"
    elif "omega_gap" in basename:
        matrix_type = "omega"
    else:
        return None, None

    horizon = pattern_kwargs.get("h")
    if horizon is not None and "h{h}" not in basename and "{h}" not in basename:
        horizon = None
    if horizon is not None:
        horizon = int(horizon)
    return matrix_type, horizon


def _try_load_gap_from_store(
    gap_input_dir: Path,
    date_str: str,
    file_pattern: str,
    pattern_kwargs: dict | None,
) -> tuple[np.ndarray | None, list[str]]:
    """Try to load a gap matrix from a SQLite ``GapStore``.

    Returns (array, alerts).  If *gap_input_dir* is not a store path or the
    matrix is not found, returns (None, [alert]) so the caller can fall back.
    """
    if not is_gap_store_path(gap_input_dir):
        return None, []

    try:
        store = GapStore(gap_input_dir)
    except Exception as e:
        return None, [f"GapStore open failed for {gap_input_dir}: {e}"]

    matrix_type, horizon = _parse_pattern_to_matrix_type(file_pattern, pattern_kwargs)
    if matrix_type is None:
        return None, [f"Cannot map pattern {file_pattern!r} to a GapStore matrix type"]

    arr = store.get(date_str, matrix_type, horizon=horizon)
    if arr is None:
        return None, [f"GapStore missing {matrix_type} (h={horizon}) for {date_str}"]
    return arr, []


def _try_save_gap_to_store(
    gap_output_dir: Path,
    date_str: str,
    file_pattern: str,
    pattern_kwargs: dict | None,
    data: np.ndarray,
) -> bool:
    """Write a single matrix to a SQLite ``GapStore`` if *gap_output_dir* is one.

    Returns True if the store path was used, False otherwise (caller should
    fall back to ``.npy``).
    """
    if not is_gap_store_path(gap_output_dir):
        return False

    try:
        store = GapStore(gap_output_dir)
    except Exception as e:
        logger.warning("GapStore open failed for %s: %s", gap_output_dir, e)
        return False

    matrix_type, horizon = _parse_pattern_to_matrix_type(file_pattern, pattern_kwargs)
    if matrix_type is None:
        logger.warning(
            "Cannot map pattern %r to a GapStore matrix type; skipping store write",
            file_pattern,
        )
        return False

    try:
        store.put(date_str, matrix_type, data, horizon=horizon)
    except Exception as e:
        logger.warning("Failed to write %s to GapStore: %s", file_pattern, e)
        return False
    return True


def load_gap_npy(
    gap_input_dir: Path,
    date_str: str,
    file_pattern: str,
    pattern_kwargs: dict | None = None,
) -> tuple[np.ndarray | None, list[str]]:
    """Load a single gap matrix from a ``.npy`` file or a SQLite ``GapStore``.

    Args:
        gap_input_dir: Root directory containing the file, or a ``.sqlite``
            gap store file.
        date_str: Trade date in any format parseable by ``pd.to_datetime``.
        file_pattern: Path template with ``{date}`` placeholder and optional
            additional named placeholders (e.g. ``{h}`` for horizon).
        pattern_kwargs: Optional extra format arguments for *file_pattern*.

    Returns:
        Tuple of (array, alerts).  Array is ``None`` when the matrix is missing
        or cannot be loaded.
    """
    pattern_kwargs = pattern_kwargs or {}

    # 1. Try SQLite gap store first (opt-in via .sqlite/.db path).
    arr, alerts = _try_load_gap_from_store(
        gap_input_dir, date_str, file_pattern, pattern_kwargs
    )
    if arr is not None:
        return arr, []
    if alerts:
        return None, alerts

    # 2. Fall back to per-date .npy files.
    date_numeric = _format_gap_date(date_str)
    file_path = gap_input_dir / file_pattern.format(date=date_numeric, **pattern_kwargs)

    if not file_path.exists():
        alert = f"Gap file missing: {file_path}"
        logger.debug(alert)
        return None, [alert]

    try:
        arr = np.load(file_path)
    except Exception as e:
        alert = f"Failed to load {file_path}: {e}"
        logger.warning(alert)
        return None, [alert]

    return arr, []


def save_gap_npy(
    gap_output_dir: Path,
    date_str: str,
    data: np.ndarray,
    file_pattern: str,
    pattern_kwargs: dict | None = None,
) -> bool:
    """Save a single gap matrix to a ``.npy`` file or SQLite ``GapStore``.

    Args:
        gap_output_dir: Root directory for the output file, or a ``.sqlite``
            gap store file.
        date_str: Trade date in any format parseable by ``pd.to_datetime``.
        data: Numpy array to store.
        file_pattern: Path template with ``{date}`` placeholder and optional
            additional named placeholders.
        pattern_kwargs: Optional extra format arguments for *file_pattern*.

    Returns:
        True if the matrix was written successfully.
    """
    pattern_kwargs = pattern_kwargs or {}

    # 1. Try SQLite gap store first (opt-in via .sqlite/.db path).
    if _try_save_gap_to_store(gap_output_dir, date_str, file_pattern, pattern_kwargs, data):
        return True

    # 2. Fall back to per-date .npy file.
    date_numeric = _format_gap_date(date_str)
    file_path = gap_output_dir / file_pattern.format(date=date_numeric, **pattern_kwargs)
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(file_path, data)
    except Exception as e:
        logger.warning("Failed to save %s: %s", file_path, e)
        return False
    return True


def load_gap_matrices(
    gap_input_dir: Path,
    date_str: str,
    mu_pattern: str = "matrices/mu_gap_{date}.npy",
    omega_pattern: str = "matrices/omega_gap_{date}.npy",
    pattern_kwargs: dict | None = None,
    *,
    strict: bool = False,
    n_j: int = len(JP_TICKERS),
) -> tuple[np.ndarray | None, np.ndarray | None, list[str]]:
    """Load a pair of mu_gap / Omega_gap ``.npy`` files or SQLite records.

    Args:
        gap_input_dir: Root directory containing the gap matrix files, or a
            ``.sqlite`` gap store file.
        date_str: Trade date in any format parseable by ``pd.to_datetime``.
        mu_pattern: File pattern for the mu matrix.  Must contain ``{date}``.
        omega_pattern: File pattern for the Omega matrix.  Must contain ``{date}``.
        pattern_kwargs: Optional extra format arguments for the patterns
            (e.g. ``{"h": 3}`` for horizon-aware patterns like
            ``matrices/mu_gap_h{h}_{date}.npy``).
        strict: If True, raise ``DataValidationError`` when the matrices are
            missing, have the wrong shape, or fail basic invariants. If False,
            return ``(None, None, alerts)`` to preserve existing fallback flow.
        n_j: Expected number of JP assets for shape validation.

    Returns:
        Tuple of (mu_gap, Omega_gap, alerts).  Both arrays are ``None`` when
        either file is missing or cannot be loaded (non-strict mode).

    Raises:
        DataValidationError: When ``strict=True`` and validation fails.
    """
    pattern_kwargs = pattern_kwargs or {}
    mu_gap, mu_alerts = load_gap_npy(gap_input_dir, date_str, mu_pattern, pattern_kwargs)
    Omega_gap, omega_alerts = load_gap_npy(gap_input_dir, date_str, omega_pattern, pattern_kwargs)

    alerts = mu_alerts + omega_alerts
    if strict and (mu_gap is None or Omega_gap is None):
        raise DataValidationError("; ".join(alerts) if alerts else "Gap matrices unavailable")

    if mu_gap is not None and Omega_gap is not None:
        v_alerts = validate_gap_matrices(mu_gap, Omega_gap, n_j=n_j)
        if v_alerts:
            if strict:
                raise DataValidationError("; ".join(v_alerts))
            alerts.extend(v_alerts)
        # Return the arrays in both strict and non-strict paths.
        # The caller (e.g. production_v2) decides whether to fall back to
        # flat based on the alerts and its own fallback flags.
        return mu_gap, Omega_gap, alerts
    return None, None, alerts


def save_gap_matrices(
    gap_output_dir: Path,
    date_str: str,
    mu_gap: np.ndarray,
    omega_gap: np.ndarray,
    mu_pattern: str = "matrices/mu_gap_{date}.npy",
    omega_pattern: str = "matrices/omega_gap_{date}.npy",
    pattern_kwargs: dict | None = None,
    metadata: dict[str, Any] | None = None,
) -> bool:
    """Save a pair of mu_gap / Omega_gap matrices.

    If *gap_output_dir* is a ``.sqlite`` file the matrices are written through
    :class:`GapStore`; otherwise per-date ``.npy`` files are written under the
    supplied directory.  The ``latest/`` directory of ``.npy`` files is
    preserved for backward compatibility.

    Args:
        gap_output_dir: Directory or SQLite gap store path.
        date_str: Trade date in any format parseable by ``pd.to_datetime``.
        mu_gap: Expected-return vector.
        omega_gap: Covariance matrix.
        mu_pattern: File pattern for the mu matrix.
        omega_pattern: File pattern for the Omega matrix.
        pattern_kwargs: Optional extra format arguments for horizon-aware
            patterns (e.g. ``{"h": 3}``).
        metadata: Optional dict of metadata (sig_date, etc.) stored with the
            pair when writing to a ``GapStore``.

    Returns:
        True if both matrices were written successfully.
    """
    pattern_kwargs = pattern_kwargs or {}

    if is_gap_store_path(gap_output_dir):
        try:
            store = GapStore(gap_output_dir)
        except Exception as e:
            logger.warning("GapStore open failed for %s: %s", gap_output_dir, e)
            return False

        mu_type, mu_horizon = _parse_pattern_to_matrix_type(mu_pattern, pattern_kwargs)
        omega_type, omega_horizon = _parse_pattern_to_matrix_type(
            omega_pattern, pattern_kwargs
        )

        if mu_type is None or omega_type is None:
            logger.warning(
                "Cannot map mu/omega patterns to GapStore types; skipping store write"
            )
            return False

        # If both patterns are the default (no horizon) pair, use the bundled
        # ``save`` API so metadata is stored together with mu/omega.
        if mu_horizon is None and omega_horizon is None:
            try:
                store.save(date_str, mu_gap, omega_gap, metadata=metadata)
            except Exception as e:
                logger.warning("Failed to save gap pair to GapStore: %s", e)
                return False
            return True

        # Horizon-aware: store each matrix with its horizon.  Metadata, if any,
        # is stored under the 'meta' type with the same horizon.
        try:
            store.put(date_str, mu_type, mu_gap, horizon=mu_horizon)
            store.put(date_str, omega_type, omega_gap, horizon=omega_horizon)
            if metadata is not None:
                store.put(date_str, "meta", metadata, horizon=mu_horizon)
        except Exception as e:
            logger.warning("Failed to save horizon gap pair to GapStore: %s", e)
            return False
        return True

    # Directory / .npy fallback.
    date_numeric = _format_gap_date(date_str)
    mu_path = gap_output_dir / mu_pattern.format(date=date_numeric, **pattern_kwargs)
    omega_path = gap_output_dir / omega_pattern.format(date=date_numeric, **pattern_kwargs)
    try:
        mu_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(mu_path, mu_gap)
        omega_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(omega_path, omega_gap)
    except Exception as e:
        logger.warning("Failed to save gap matrices to %s / %s: %s", mu_path, omega_path, e)
        return False
    return True
