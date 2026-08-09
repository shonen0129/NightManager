"""Gap matrix I/O helpers shared by production and signal-enhancement modules."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from leadlag.data.tickers import JP_TICKERS
from leadlag.data.validation import DataValidationError, validate_gap_matrices


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
    from leadlag.data.gap_store import GapStore, is_gap_store_path

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

logger = logging.getLogger(__name__)


def _format_gap_date(date_str: str) -> str:
    """Convert any parseable date string to the YYYYMMDD gap file suffix."""
    return str(pd.to_datetime(date_str).strftime("%Y%m%d"))


def load_gap_npy(
    gap_input_dir: Path,
    date_str: str,
    file_pattern: str,
    pattern_kwargs: dict | None = None,
) -> tuple[np.ndarray | None, list[str]]:
    """Load a single ``.npy`` gap file if it exists.

    Args:
        gap_input_dir: Root directory containing the file.
        date_str: Trade date in any format parseable by ``pd.to_datetime``.
        file_pattern: Path template with ``{date}`` placeholder and optional
            additional named placeholders (e.g. ``{h}`` for horizon).
        pattern_kwargs: Optional extra format arguments for *file_pattern*.

    Returns:
        Tuple of (array, alerts).  Array is ``None`` when the file is missing
        or cannot be loaded.
    """
    pattern_kwargs = pattern_kwargs or {}

    # 1. Try SQLite gap store first (opt-in via .sqlite/.db path).
    arr, alerts = _try_load_gap_from_store(gap_input_dir, date_str, file_pattern, pattern_kwargs)
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
    """Load a pair of mu_gap / Omega_gap ``.npy`` files.

    Args:
        gap_input_dir: Root directory containing the gap matrix files.
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
