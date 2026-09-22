"""Timestamp normalization shared by strategy input boundaries."""

from __future__ import annotations

from typing import Any

import pandas as pd

JST = "Asia/Tokyo"


def normalize_jst_timestamp(value: Any) -> pd.Timestamp:
    """Coerce one timestamp to a timezone-naive timestamp on the JST clock.

    Naive values are already interpreted as JST by the strategy contracts.
    Offset-aware values are converted before the timezone is removed so their
    trading date cannot shift when an upstream producer emits UTC timestamps.
    """
    result = pd.Timestamp(value)
    if pd.isna(result):
        raise ValueError(f"invalid timestamp: {value!r}")
    if result.tzinfo is not None:
        result = result.tz_convert(JST).tz_localize(None)
    return result


def normalize_jst_date(value: Any) -> pd.Timestamp:
    """Return the JST calendar date for one timestamp as a naive midnight."""
    return normalize_jst_timestamp(value).normalize()


def normalize_jst_index(index: pd.Index) -> pd.DatetimeIndex:
    """Normalize a possibly mixed timezone index to date keys.

    Normalizing scalar-by-scalar also handles an index containing both naive
    and offset-aware values, which ``pd.to_datetime`` cannot reliably parse as
    one timezone in all supported pandas versions.
    """
    return pd.DatetimeIndex([normalize_jst_date(value) for value in index])


__all__ = ["JST", "normalize_jst_date", "normalize_jst_index", "normalize_jst_timestamp"]
