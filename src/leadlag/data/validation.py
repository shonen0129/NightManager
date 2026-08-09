"""Data validation gates for the lead-lag pipeline.

Validation gates turn silent data corruption (dropped rows, stale gap matrices,
missing tickers) into explicit, actionable errors. By default the existing
pipeline keeps the historical tolerant behaviour; callers can opt into
``strict_validation=True`` to fail fast.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from leadlag.data.tickers import JP_TICKERS, US_TICKERS

logger = logging.getLogger(__name__)


class DataValidationError(ValueError):
    """Raised when input data does not meet quality invariants."""


US_CC_PREFIX = "us_cc_"
JP_CC_PREFIX = "jp_cc_"
JP_GAP_PREFIX = "jp_gap_"
JP_OPEN_PREFIX = "jp_open_trade_"
JP_CLOSE_PREFIX = "jp_close_sig_"


def _required_us_cols() -> list[str]:
    return [f"{US_CC_PREFIX}{tk}" for tk in US_TICKERS]


def _required_jp_cols() -> list[str]:
    return (
        [f"{JP_CC_PREFIX}{tk}" for tk in JP_TICKERS]
        + [f"{JP_GAP_PREFIX}{tk}" for tk in JP_TICKERS]
        + [f"{JP_OPEN_PREFIX}{tk}" for tk in JP_TICKERS]
        + [f"{JP_CLOSE_PREFIX}{tk}" for tk in JP_TICKERS]
    )


def validate_raw_data_sources(data: dict[str, Any]) -> list[str]:
    """Check that raw OHLC source dict has the expected keys and ticker columns.

    Returns a list of alerts (empty if no problems found). Raises
    ``DataValidationError`` only for structural problems (missing keys).
    """
    required = {"us_close", "jp_close", "jp_open"}
    missing_keys = required - set(data.keys())
    if missing_keys:
        raise DataValidationError(f"Raw data missing required keys: {sorted(missing_keys)}")

    alerts: list[str] = []

    us_close = data.get("us_close")
    if isinstance(us_close, pd.DataFrame):
        missing_us = [tk for tk in US_TICKERS if tk not in us_close.columns]
        if missing_us:
            alerts.append(f"us_close missing tickers: {missing_us}")

    jp_close = data.get("jp_close")
    if isinstance(jp_close, pd.DataFrame):
        missing_jp = [tk for tk in JP_TICKERS + ["1306.T"] if tk not in jp_close.columns]
        if missing_jp:
            alerts.append(f"jp_close missing tickers: {missing_jp}")

    return alerts


def validate_exec_record(
    record: dict[str, Any],
    *,
    us_tickers: list[str] | None = None,
    jp_tickers: list[str] | None = None,
) -> list[str]:
    """Validate a single execution record before it is appended to ``df_exec``.

    Required non-NaN fields:
      - All ``us_cc_*`` columns
      - All ``jp_cc_*``, ``jp_gap_*``, ``jp_open_trade_*``, ``jp_close_sig_*`` columns

    The target return columns ``jp_oc_*`` may be NaN for the current day; this is
    handled downstream (0-fill + is_provisional).
    """
    us_tickers = us_tickers or US_TICKERS
    jp_tickers = jp_tickers or JP_TICKERS

    alerts: list[str] = []
    for tk in us_tickers:
        col = f"{US_CC_PREFIX}{tk}"
        if record.get(col) is None or (isinstance(record[col], float) and np.isnan(record[col])):
            alerts.append(f"us_cc missing for {tk} on {record.get('trade_date')}")

    for tk in jp_tickers:
        for prefix in (JP_CC_PREFIX, JP_GAP_PREFIX, JP_OPEN_PREFIX, JP_CLOSE_PREFIX):
            col = f"{prefix}{tk}"
            if record.get(col) is None or (isinstance(record[col], float) and np.isnan(record[col])):
                alerts.append(f"{prefix[:-1]} missing for {tk} on {record.get('trade_date')}")

    return alerts


def validate_gap_matrices(
    mu: np.ndarray | None,
    omega: np.ndarray | None,
    *,
    n_j: int = len(JP_TICKERS),
) -> list[str]:
    """Validate the shape and finite-ness of mu / Omega gap matrices.

    Returns a list of alerts. Callers can decide whether to raise or fall back.
    """
    alerts: list[str] = []
    if mu is None and omega is None:
        alerts.append("Both mu and Omega gap matrices are missing")
        return alerts
    if mu is None:
        alerts.append("mu gap matrix is missing")
    if omega is None:
        alerts.append("Omega gap matrix is missing")
    if mu is None or omega is None:
        return alerts

    mu_arr = np.asarray(mu, dtype=float)
    omega_arr = np.asarray(omega, dtype=float)

    if mu_arr.shape != (n_j,):
        alerts.append(f"mu shape {mu_arr.shape} != ({n_j},)")
    if omega_arr.shape != (n_j, n_j):
        alerts.append(f"Omega shape {omega_arr.shape} != ({n_j}, {n_j})")
    if not np.all(np.isfinite(mu_arr)):
        alerts.append("mu contains non-finite values")
    if not np.all(np.isfinite(omega_arr)):
        alerts.append("Omega contains non-finite values")
    if omega_arr.shape == (n_j, n_j):
        if not np.allclose(omega_arr, omega_arr.T):
            alerts.append("Omega is not symmetric")
        eigvals = np.linalg.eigvalsh(omega_arr + omega_arr.T) / 2.0
        if np.any(eigvals < -1e-8):
            alerts.append(f"Omega is not positive semi-definite (min eigval {eigvals.min():.6f})")

    return alerts
