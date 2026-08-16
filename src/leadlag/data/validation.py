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

from leadlag.data.tickers import JP_TICKERS, TOPIX_TICKER, US_TICKERS

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
        missing_jp = [tk for tk in JP_TICKERS + [TOPIX_TICKER] if tk not in jp_close.columns]
        if missing_jp:
            alerts.append(f"jp_close missing tickers: {missing_jp}")

    return alerts


def _check_series_nan_blocks(series: pd.Series) -> dict[str, Any]:
    """Return NaN block statistics for a single-ticker price series."""
    if series.empty:
        return {
            "all_nan": True,
            "n_nan": len(series),
            "first_valid": None,
            "last_valid": None,
            "leading_nan": len(series),
            "trailing_nan": len(series),
        }

    n_nan = int(series.isna().sum())
    n_total = len(series)
    all_nan = n_nan == n_total
    first_valid = series.first_valid_index()
    last_valid = series.last_valid_index()
    idx = series.index
    leading_nan = 0
    if first_valid is not None:
        leading_nan = int((idx < first_valid).sum())
    trailing_nan = 0
    if last_valid is not None:
        trailing_nan = int((idx > last_valid).sum())

    return {
        "all_nan": all_nan,
        "n_nan": n_nan,
        "first_valid": first_valid,
        "last_valid": last_valid,
        "leading_nan": leading_nan,
        "trailing_nan": trailing_nan,
    }


def validate_etf_raw_data(
    data: dict[str, Any],
    *,
    min_history_days: int = 30,
) -> dict[str, Any]:
    """Validate raw ETF OHLC cache quality.

    Checks for all-NaN tickers, leading / trailing NaN blocks, and short history.
    Returns a dict with ``ok`` (bool), ``alerts`` (list[str]), and ``fatals``
    (list[str]).
    """
    required = {"us_close", "jp_close", "jp_open"}
    missing_keys = required - set(data.keys())
    if missing_keys:
        raise DataValidationError(f"Raw data missing required keys: {sorted(missing_keys)}")

    ok = True
    alerts: list[str] = []
    fatals: list[str] = []
    ticker_stats: dict[str, dict[str, Any]] = {}

    tables = {
        "us_close": (data["us_close"], US_TICKERS),
        "jp_close": (data["jp_close"], JP_TICKERS + [TOPIX_TICKER]),
        "jp_open": (data["jp_open"], JP_TICKERS + [TOPIX_TICKER]),
    }

    for table_name, (df, expected_tickers) in tables.items():
        if not isinstance(df, pd.DataFrame):
            fatals.append(f"{table_name} is not a DataFrame")
            ok = False
            continue
        if df.empty:
            fatals.append(f"{table_name} is empty")
            ok = False
            continue

        missing = [tk for tk in expected_tickers if tk not in df.columns]
        if missing:
            fatals.append(f"{table_name} missing tickers: {missing}")
            ok = False

        for tk in expected_tickers:
            if tk not in df.columns:
                continue
            series = df[tk]
            stats = _check_series_nan_blocks(series)
            ticker_stats[f"{table_name}.{tk}"] = stats

            if stats["all_nan"]:
                fatals.append(f"{table_name}.{tk} is all-NaN")
                ok = False
                continue

            valid_count = len(series) - stats["n_nan"]
            if valid_count < min_history_days:
                fatals.append(
                    f"{table_name}.{tk} has only {valid_count} valid rows "
                    f"(min {min_history_days})"
                )
                ok = False

            if stats["trailing_nan"] > 0:
                last_valid = stats["last_valid"]
                last_idx = df.index[-1]
                alerts.append(
                    f"{table_name}.{tk} has {stats['trailing_nan']} trailing NaN "
                    f"(last valid {last_valid}, last index {last_idx})"
                )

            if stats["leading_nan"] > 0:
                first_valid = stats["first_valid"]
                first_idx = df.index[0]
                alerts.append(
                    f"{table_name}.{tk} has {stats['leading_nan']} leading NaN "
                    f"(first valid {first_valid}, first index {first_idx})"
                )

            isolated = stats["n_nan"] - stats["leading_nan"] - stats["trailing_nan"]
            if isolated > 0:
                alerts.append(
                    f"{table_name}.{tk} has {isolated} isolated NaN(s)"
                )

    return {
        "ok": ok,
        "alerts": alerts,
        "fatals": fatals,
        "ticker_stats": ticker_stats,
    }


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
        if record.get(col) is None or (isinstance(record[col], float) and (np.isnan(record[col]) or np.isinf(record[col]))):
            alerts.append(f"us_cc missing or non-finite for {tk} on {record.get('trade_date')}")

    for tk in jp_tickers:
        for prefix in (JP_CC_PREFIX, JP_GAP_PREFIX, JP_OPEN_PREFIX, JP_CLOSE_PREFIX):
            col = f"{prefix}{tk}"
            val = record.get(col)
            if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                alerts.append(f"{prefix[:-1]} missing or non-finite for {tk} on {record.get('trade_date')}")
                continue
            if prefix == JP_OPEN_PREFIX and isinstance(val, (int, float)) and float(val) <= 0.0:
                alerts.append(f"{prefix[:-1]} non-positive for {tk} on {record.get('trade_date')}")

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
