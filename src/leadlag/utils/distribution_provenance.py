"""Point-in-time provenance validation shared by gap-data consumers."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from leadlag.utils.timestamps import normalize_jst_date


def _is_scalar(value: Any) -> bool:
    return not isinstance(value, (list, tuple, set, Mapping, pd.Series, pd.Index, np.ndarray))


def _missing(value: Any) -> bool:
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(result, (bool, np.bool_)) and bool(result)


def _parse_date(value: Any, field: str) -> tuple[pd.Timestamp | None, str | None]:
    if value is None or not _is_scalar(value) or _missing(value):
        return None, f"{field} must be one non-null scalar date"
    if not isinstance(value, (str, date, datetime, pd.Timestamp, np.datetime64)):
        return None, f"{field} must be one scalar date"
    try:
        parsed = pd.to_datetime(value, errors="raise")
        timestamp = pd.Timestamp(parsed)
        # ``pd.to_datetime`` accepts strings such as ``"NaT"`` and ``""``
        # with ``errors="raise"`` but returns the missing sentinel.  Check
        # the converted value before calling Timestamp-only methods so every
        # invalid provenance value follows the normal validation contract.
        if _missing(timestamp):
            return None, f"{field} is invalid"
        # Provenance dates follow the strategy's JST calendar.  Convert an
        # offset-aware producer value before dropping its timezone; stripping
        # the offset first can turn 00:00 JST into the previous UTC date.
        timestamp = normalize_jst_date(timestamp)
    except (TypeError, ValueError, OverflowError, OSError):
        return None, f"{field} is invalid"
    return timestamp, None


def _parse_horizon(value: Any, field: str) -> tuple[int | None, str | None]:
    if value is None or not _is_scalar(value) or _missing(value) or isinstance(value, bool):
        return None, f"{field} must be one finite integer"
    try:
        if isinstance(value, (float, np.floating)):
            if not np.isfinite(value) or not float(value).is_integer():
                return None, f"{field} must be one finite integer"
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None, f"{field} must be one finite integer"
    return parsed, None


def validate_distribution_provenance(
    metadata: Mapping[str, Any] | None,
    trade_date: Any,
    horizon: Any,
    *,
    label: str = "Distribution provenance",
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate and normalize one distribution's point-in-time metadata."""
    if not isinstance(metadata, Mapping) or not metadata:
        return None, f"{label} metadata is missing"
    trade_dt, error = _parse_date(trade_date, "requested trade_date")
    if error:
        return None, f"{label} {error}"
    expected_horizon, error = _parse_horizon(horizon, "requested horizon")
    if error:
        return None, f"{label} {error}"
    assert trade_dt is not None and expected_horizon is not None

    if "sig_date" not in metadata and "signal_date" not in metadata:
        return None, f"{label} signal date is missing"
    sig_dt: pd.Timestamp | None = None
    if "sig_date" in metadata:
        sig_dt, error = _parse_date(metadata.get("sig_date"), f"{label} sig_date")
        if error:
            return None, error
    alias_dt: pd.Timestamp | None = None
    if "signal_date" in metadata:
        alias_dt, error = _parse_date(metadata.get("signal_date"), f"{label} signal_date")
        if error:
            return None, error
    if sig_dt is not None and alias_dt is not None and sig_dt != alias_dt:
        return None, f"{label} sig_date/signal_date disagree"
    signal_dt = sig_dt or alias_dt
    assert signal_dt is not None
    if signal_dt >= trade_dt:
        return None, (
            f"{label} signal date {signal_dt.date()} is not before "
            f"trade date {trade_dt.date()}"
        )

    normalized = dict(metadata)
    normalized["sig_date"] = signal_dt.strftime("%Y-%m-%d")
    if "signal_date" in normalized:
        normalized["signal_date"] = signal_dt.strftime("%Y-%m-%d")

    if "trade_date" in metadata:
        metadata_trade_dt, error = _parse_date(metadata.get("trade_date"), f"{label} trade_date")
        if error:
            return None, error
        assert metadata_trade_dt is not None
        if metadata_trade_dt != trade_dt:
            return None, f"{label} trade_date disagrees with requested date"
    normalized["trade_date"] = trade_dt.strftime("%Y-%m-%d")

    if "horizon" in metadata:
        metadata_horizon, error = _parse_horizon(metadata.get("horizon"), f"{label} horizon")
        if error:
            return None, error
        assert metadata_horizon is not None
        if metadata_horizon != expected_horizon:
            return None, f"{label} horizon disagrees with requested horizon"
    normalized["horizon"] = expected_horizon
    return normalized, None
