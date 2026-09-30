"""Immutable, time-bounded broker quote snapshots for the live decision path."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from leadlag.data.tickers import JP_TICKERS_WITH_TOPIX

QUOTE_SNAPSHOT_SCHEMA = "market-quote-snapshot-v1"
QUOTE_WINDOW_SECONDS = 30


def _jst_timestamp(value: Any, *, field: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError(f"{field} must be a valid timestamp")
    if timestamp.tzinfo is None:
        raise ValueError(f"{field} must include a timezone")
    return timestamp.tz_convert("Asia/Tokyo")


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FrozenQuoteSnapshot:
    """A complete bid/ask snapshot with locally verifiable availability time."""

    trade_date: str
    as_of: pd.Timestamp
    request_started_at: pd.Timestamp
    snapshot_id: str
    prices: Mapping[str, float]
    observed_at: Mapping[str, str]
    quote_rows: Mapping[str, Mapping[str, Any]]
    source: str
    timestamp_source: str


def _snapshot_payload(
    record: Mapping[str, Any],
    *,
    trade_date: str,
    required_tickers: tuple[str, ...],
) -> dict[str, Any]:
    if record.get("status") != "OBSERVED" or record.get("window_valid") is not True:
        raise ValueError("quote snapshot is not a valid in-window observation")
    source = str(record.get("source") or "")
    timestamp_source = str(record.get("timestamp_source") or "")
    if source != "tachibana:CLMMfdsGetMarketPrice":
        raise ValueError("quote snapshot source is not the approved read-only market endpoint")
    if timestamp_source != "local_response_receipt":
        raise ValueError("quote snapshot must use the locally recorded response receipt time")

    started = _jst_timestamp(record.get("request_started_at"), field="request_started_at")
    received = _jst_timestamp(record.get("response_received_at"), field="response_received_at")
    expected_date = date.fromisoformat(trade_date)
    if started.date() != expected_date or received.date() != expected_date:
        raise ValueError("quote snapshot timestamps must match trade_date")
    window_start = started.normalize() + pd.Timedelta(hours=9, minutes=10)
    window_end = window_start + pd.Timedelta(seconds=QUOTE_WINDOW_SECONDS)
    if started < window_start or received > window_end or received < started:
        raise ValueError(
            "quote request and response are outside the 09:10 capture window "
            "(09:10:00-09:10:30 JST)"
        )

    rows_value = record.get("rows")
    if not isinstance(rows_value, list):
        raise ValueError("quote snapshot rows must be a list")
    rows: dict[str, dict[str, Any]] = {}
    for raw_row in rows_value:
        if not isinstance(raw_row, Mapping):
            continue
        ticker = str(raw_row.get("ticker") or "")
        if ticker in rows:
            raise ValueError(f"duplicate quote row for {ticker}")
        rows[ticker] = dict(raw_row)
    missing = sorted(set(required_tickers) - set(rows))
    if missing:
        raise ValueError(f"quote snapshot missing required tickers: {missing}")

    prices: dict[str, float] = {}
    selected_rows: dict[str, dict[str, Any]] = {}
    observed_at: dict[str, str] = {}
    for ticker in required_tickers:
        row = rows[ticker]
        bid = float(row.get("bid_price_1") or 0.0)
        ask = float(row.get("ask_price_1") or 0.0)
        mid = float(row.get("quote_mid_price") or 0.0)
        if (
            row.get("status") != "OBSERVED"
            or row.get("valid_two_sided_quote") is not True
            or not all(math.isfinite(item) and item > 0.0 for item in (bid, ask, mid))
            or ask <= bid
            or not math.isclose(mid, (bid + ask) / 2.0, rel_tol=1e-9, abs_tol=1e-9)
        ):
            raise ValueError(f"quote snapshot has an invalid two-sided quote for {ticker}")
        row_time = _jst_timestamp(row.get("observed_at"), field=f"observed_at[{ticker}]")
        if row_time != received:
            raise ValueError(f"quote row availability time differs from batch receipt for {ticker}")
        row_started = _jst_timestamp(
            row.get("request_started_at"), field=f"request_started_at[{ticker}]"
        )
        row_received = _jst_timestamp(
            row.get("response_received_at"), field=f"response_received_at[{ticker}]"
        )
        if row_started != started or row_received != received:
            raise ValueError(f"quote row request/response times differ from batch times for {ticker}")
        if row.get("timestamp_source") != timestamp_source or row.get("source") != source:
            raise ValueError(f"quote row provenance differs from batch provenance for {ticker}")
        prices[ticker] = mid
        observed_at[ticker] = received.isoformat()
        selected_rows[ticker] = row

    identity = {
        "schema_version": QUOTE_SNAPSHOT_SCHEMA,
        "trade_date": trade_date,
        "request_started_at": started.isoformat(),
        "response_received_at": received.isoformat(),
        "source": source,
        "timestamp_source": timestamp_source,
        "prices": prices,
        "quotes": selected_rows,
    }
    return {
        **identity,
        "status": "OBSERVED",
        "window_valid": True,
        "observed_count": len(required_tickers),
        "lob_count": len(required_tickers),
        "rows": list(selected_rows.values()),
        "snapshot_id": _canonical_hash(identity),
    }


def validate_frozen_payload(
    payload: Mapping[str, Any],
    *,
    trade_date: str,
    required_tickers: tuple[str, ...] = tuple(JP_TICKERS_WITH_TOPIX),
) -> FrozenQuoteSnapshot:
    """Validate a saved immutable snapshot and return its typed view."""
    if payload.get("schema_version") != QUOTE_SNAPSHOT_SCHEMA:
        raise ValueError("unsupported frozen quote snapshot schema")
    if payload.get("trade_date") != trade_date:
        raise ValueError("frozen quote snapshot trade_date mismatch")
    canonical = _snapshot_payload(payload, trade_date=trade_date, required_tickers=required_tickers)
    if canonical.get("snapshot_id") != payload.get("snapshot_id"):
        raise ValueError("frozen quote snapshot fingerprint mismatch")
    prices = canonical["prices"]
    rows = canonical["quotes"]
    observed_at = {ticker: canonical["response_received_at"] for ticker in required_tickers}
    return FrozenQuoteSnapshot(
        trade_date=trade_date,
        as_of=_jst_timestamp(canonical["response_received_at"], field="response_received_at"),
        request_started_at=_jst_timestamp(canonical["request_started_at"], field="request_started_at"),
        snapshot_id=str(canonical["snapshot_id"]),
        prices=prices,
        observed_at=observed_at,
        quote_rows=rows,
        source=str(canonical["source"]),
        timestamp_source=str(canonical["timestamp_source"]),
    )


def freeze_quote_snapshot(
    record: Mapping[str, Any],
    directory: str | Path,
    *,
    trade_date: str,
    required_tickers: tuple[str, ...] = tuple(JP_TICKERS_WITH_TOPIX),
) -> Path:
    """Persist the first valid daily quote snapshot without replacing it."""
    payload = _snapshot_payload(record, trade_date=trade_date, required_tickers=required_tickers)
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"frozen_{trade_date.replace('-', '')}.json"
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        existing = json.loads(path.read_text(encoding="utf-8"))
        validate_frozen_payload(existing, trade_date=trade_date, required_tickers=required_tickers)
        return path
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return path


def load_frozen_quote_snapshot(
    directory: str | Path,
    *,
    trade_date: str,
    required_tickers: tuple[str, ...] = tuple(JP_TICKERS_WITH_TOPIX),
) -> FrozenQuoteSnapshot:
    """Load the immutable snapshot selected for this trade date."""
    path = Path(directory) / f"frozen_{trade_date.replace('-', '')}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("frozen quote snapshot must contain a JSON object")
    return validate_frozen_payload(payload, trade_date=trade_date, required_tickers=required_tickers)


__all__ = [
    "FrozenQuoteSnapshot",
    "QUOTE_SNAPSHOT_SCHEMA",
    "freeze_quote_snapshot",
    "load_frozen_quote_snapshot",
    "validate_frozen_payload",
]
