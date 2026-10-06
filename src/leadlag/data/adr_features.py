"""ADR feature input adapter.

Only this module owns the filesystem read and staleness policy for the
versioned ADR feature bundle. Models receive a DataFrame or ``None``.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from leadlag.config.paths import project_root
from leadlag.data.tickers import ADR_SECTOR_MAP, JP_TICKERS
from leadlag.utils.timestamps import normalize_jst_date, normalize_jst_index

logger = logging.getLogger(__name__)

DEFAULT_ADR_FEATURES_PATH = project_root() / "data" / "adr_features.zip"
FEATURE_COLUMNS = [f"adr_{ticker}" for ticker in JP_TICKERS]


def _publication_frame(
    frame: pd.DataFrame, required_trade_date: pd.Timestamp | str
) -> pd.DataFrame:
    normalized = normalize_adr_features(frame)
    required = normalize_jst_date(required_trade_date)
    if normalized is None or validate_adr_features(normalized, required) is None:
        raise ValueError("ADR publication requires complete features for the required trade date")
    if normalized.index.max() != required:
        raise ValueError("ADR publication latest row must equal the required trade date")
    if "sig_date" not in normalized:
        raise ValueError("ADR publication requires signal-date provenance")
    signals = normalize_jst_index(normalized["sig_date"])
    if signals.hasnans or (signals >= normalized.index).any():
        raise ValueError("ADR signal dates must precede their JP trade dates")
    normalized["sig_date"] = signals
    coverage_columns = [f"coverage_{ticker}" for ticker in JP_TICKERS]
    if not set(coverage_columns).issubset(normalized.columns):
        raise ValueError("ADR publication requires per-sector observed coverage")
    coverage = normalized[coverage_columns].to_numpy(dtype=float)
    if (
        not np.isfinite(coverage).all()
        or (coverage < 0).any()
        or (coverage != np.floor(coverage)).any()
    ):
        raise ValueError("ADR coverage must contain nonnegative integer counts")
    for ticker in JP_TICKERS:
        counts = normalized[f"coverage_{ticker}"].to_numpy(dtype=float)
        values = normalized[f"adr_{ticker}"].to_numpy(dtype=float)
        mapped = ADR_SECTOR_MAP[ticker]
        if mapped:
            if (counts > len(mapped)).any() or not np.array_equal(counts > 0, np.isfinite(values)):
                raise ValueError("ADR feature observations disagree with coverage")
        elif (counts != 0).any() or (values != 0).any():
            raise ValueError("Unmapped ADR sectors require structural zero features and coverage")
    return normalized


def publish_adr_features(
    frame: pd.DataFrame,
    path: Path | str = DEFAULT_ADR_FEATURES_PATH,
    *,
    required_trade_date: pd.Timestamp | str,
) -> dict:
    """Commit pickle, CSV and their quality manifest with one atomic replacement."""
    normalized = _publication_frame(frame, required_trade_date)
    required = normalize_jst_date(required_trade_date)
    signals = normalize_jst_index(normalized["sig_date"])
    pickle_buffer = io.BytesIO()
    normalized.to_pickle(pickle_buffer)
    artifacts = {
        "features.pkl": pickle_buffer.getvalue(),
        "features.csv": normalized.to_csv().encode("utf-8"),
    }
    manifest = {
        "schema_version": 1,
        "published_at": datetime.now(UTC).isoformat(),
        "latest_trade_date": required.date().isoformat(),
        "latest_signal_date": signals[-1].date().isoformat(),
        "rows": len(normalized),
        "incomplete_rows": int(
            (~np.isfinite(normalized[FEATURE_COLUMNS].to_numpy(dtype=float))).any(axis=1).sum()
        ),
        "latest_coverage": {
            ticker: int(normalized.iloc[-1][f"coverage_{ticker}"]) for ticker in JP_TICKERS
        },
        "sha256": {
            name: hashlib.sha256(payload).hexdigest() for name, payload in artifacts.items()
        },
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, suffix=".tmp", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
                for name, payload in artifacts.items():
                    bundle.writestr(name, payload)
                bundle.writestr("manifest.json", json.dumps(manifest, sort_keys=True))
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return manifest


def normalize_adr_features(adr_df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Return an owned ADR frame keyed by normalized JST trade dates.

    Training consumes the full artifact rather than one date-filtered row, so
    it needs the same timezone normalization used by the live date validator.
    Invalid or duplicate dates are rejected at this adapter boundary.
    """
    if adr_df is None or adr_df.empty:
        return None if adr_df is None else adr_df.copy(deep=True)
    normalized = adr_df.copy(deep=True)
    index_name = normalized.index.name
    try:
        normalized.index = normalize_jst_index(normalized.index)
        normalized.index.name = index_name
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
) -> pd.DataFrame | None:
    """Return the run-owned ADR frame when it is valid for *trade_date*.

    The filesystem adapter and callers that own a full run snapshot use the
    same exact-date/completeness rule.  This keeps backtest and live overlay decisions
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
    if not set(FEATURE_COLUMNS).issubset(normalized.columns) or normalized.columns.has_duplicates:
        logger.warning("ADR feature columns are incomplete or duplicated; returning None.")
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
    try:
        complete = np.isfinite(
            normalized.loc[timestamp, FEATURE_COLUMNS].to_numpy(dtype=float)
        ).all()
    except (TypeError, ValueError):
        complete = False
    if not complete:
        logger.warning(
            "ADR feature row is incomplete for trade_date=%s; returning None.", timestamp
        )
        return None
    return normalized


def load_adr_features(
    path: Path | str | None = None,
    trade_date: pd.Timestamp | str | None = None,
) -> pd.DataFrame | None:
    """Load only a complete, verified ADR bundle; old pickle artifacts are not read."""
    artifact_path = DEFAULT_ADR_FEATURES_PATH if path is None else Path(path)
    if not artifact_path.exists():
        return None
    try:
        with zipfile.ZipFile(artifact_path) as bundle:
            manifest = json.loads(bundle.read("manifest.json"))
            if (
                manifest["schema_version"] != 1
                or len(bundle.namelist()) != 3
                or set(bundle.namelist())
                != {
                    "manifest.json",
                    "features.pkl",
                    "features.csv",
                }
            ):
                raise ValueError("Invalid ADR bundle schema")
            artifacts = {name: bundle.read(name) for name in ("features.pkl", "features.csv")}
            if any(
                hashlib.sha256(payload).hexdigest() != manifest["sha256"][name]
                for name, payload in artifacts.items()
            ):
                raise ValueError("ADR bundle digest mismatch")
            adr_df = pd.read_pickle(io.BytesIO(artifacts["features.pkl"]))
            normalized = normalize_adr_features(adr_df)
            if (
                normalized is None
                or normalized.empty
                or len(normalized) != manifest["rows"]
                or normalized.index[-1].date().isoformat() != manifest["latest_trade_date"]
            ):
                raise ValueError("ADR bundle date/row manifest mismatch")
            normalized = _publication_frame(normalized, manifest["latest_trade_date"])
            if (
                normalize_jst_index(normalized["sig_date"])[-1].date().isoformat()
                != manifest["latest_signal_date"]
            ):
                raise ValueError("ADR bundle signal-date manifest mismatch")
            actual_coverage = {
                ticker: int(normalized.iloc[-1][f"coverage_{ticker}"]) for ticker in JP_TICKERS
            }
            if actual_coverage != manifest["latest_coverage"]:
                raise ValueError("ADR bundle coverage manifest mismatch")
            adr_df = normalized[FEATURE_COLUMNS]
    except Exception as exc:
        logger.warning("Failed to load ADR features from %s: %s", artifact_path, exc)
        return None

    if trade_date is not None:
        return validate_adr_features(adr_df, trade_date)
    return normalize_adr_features(adr_df)


__all__ = [
    "DEFAULT_ADR_FEATURES_PATH",
    "load_adr_features",
    "normalize_adr_features",
    "validate_adr_features",
    "publish_adr_features",
]
