"""Gap matrix I/O helpers shared by production and signal-enhancement modules."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from leadlag.data.tickers import JP_TICKERS
from leadlag.data.validation import DataValidationError, validate_gap_matrices

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
        else:
            return mu_gap, Omega_gap, []
    return None, None, alerts
