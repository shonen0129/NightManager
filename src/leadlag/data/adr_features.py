"""ADR feature input adapter.

Only this module owns the filesystem read and staleness policy for the
precomputed ADR feature table. Models receive a DataFrame or ``None``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from leadlag.utils.timestamps import normalize_jst_date, normalize_jst_index

logger = logging.getLogger(__name__)

DEFAULT_ADR_FEATURES_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "adr_features.pkl"
)


def normalize_adr_features(adr_df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Return an owned ADR frame keyed by normalized JST trade dates.

    Training consumes the full artifact rather than one date-filtered row, so
    it needs the same timezone normalization used by the live date validator.
    Invalid or duplicate dates are rejected at this adapter boundary.
    """
    if adr_df is None or adr_df.empty:
        return None if adr_df is None else adr_df.copy(deep=True)
    normalized = adr_df.copy(deep=True)
    try:
        normalized.index = normalize_jst_index(normalized.index)
    except (TypeError, ValueError, OverflowError, OSError) as exc:
        logger.warning("ADR feature index is invalid: %s; returning None.", exc)
        return None
    if normalized.index.hasnans or normalized.index.has_duplicates:
        logger.warning("ADR feature index has invalid or duplicate JST dates; returning None.")
        return None
    return normalized.sort_index()


def validate_adr_features(
    adr_df: pd.DataFrame | None,
    trade_date: pd.Timestamp | str,
    max_stale_bdays: int = 3,
) -> pd.DataFrame | None:
    """Return the run-owned ADR frame when it is valid for *trade_date*.

    The filesystem adapter and callers that own a full run snapshot use the
    same date/staleness rule.  This keeps backtest and live overlay decisions
    identical when a row is missing or the artifact has gone stale.
    """
    if adr_df is None:
        return None
    if adr_df.empty:
        logger.warning("ADR feature artifact is empty; returning None.")
        return None
    timestamp = normalize_jst_date(trade_date)
    normalized = normalize_adr_features(adr_df)
    if normalized is None:
        return None
    latest = normalized.index.max()
    if latest is None or pd.isna(latest):
        return adr_df
    if timestamp not in normalized.index:
        logger.warning(
            "ADR feature row missing for trade_date=%s (latest=%s); returning None.",
            timestamp,
            latest,
        )
        return None
    threshold = latest + pd.offsets.BDay(max_stale_bdays)
    if threshold < timestamp:
        logger.warning(
            "ADR features are stale (latest=%s, trade_date=%s, max_stale_bdays=%d); returning None.",
            latest,
            timestamp,
            max_stale_bdays,
        )
        return None
    return normalized


def load_adr_features(
    path: Path | str | None = None,
    trade_date: pd.Timestamp | str | None = None,
    max_stale_bdays: int = 3,
) -> pd.DataFrame | None:
    """Load ADR features when present and within the existing staleness rule."""
    artifact_path = DEFAULT_ADR_FEATURES_PATH if path is None else Path(path)
    if not artifact_path.exists():
        return None
    try:
        adr_df = pd.read_pickle(artifact_path)
        if "sig_date" in adr_df.columns:
            adr_df = adr_df.drop(columns=["sig_date"])
    except Exception as exc:
        logger.warning("Failed to load ADR features from %s: %s", artifact_path, exc)
        return None

    if trade_date is not None:
        return validate_adr_features(adr_df, trade_date, max_stale_bdays)
    return normalize_adr_features(adr_df)


__all__ = [
    "DEFAULT_ADR_FEATURES_PATH",
    "load_adr_features",
    "normalize_adr_features",
    "validate_adr_features",
]
