"""Build immutable price-only outcomes for paired ML-overlay forward records."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from leadlag.data.quote_snapshot import FrozenQuoteSnapshot
from leadlag.data.tickers import JP_TICKERS
from leadlag.reporting.ml_overlay_forward_schema import OUTCOME_SCHEMA

JST = ZoneInfo("Asia/Tokyo")
OUTCOME_TARGET_BASIS = "frozen_0910_quote_mid_to_official_close"
OFFICIAL_CLOSE_SOURCE = "tachibana:CLMMfdsGetMarketPriceHistory:pDPP"


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def official_close_history_ready_at(trade_date: str) -> datetime:
    """Return the earliest safe read time from Tachibana's documented history refresh."""
    target_day = date.fromisoformat(trade_date)
    return datetime.combine(target_day + timedelta(days=1), time(1, 0), tzinfo=JST)


def ensure_official_close_history_ready(trade_date: str, *, now: datetime | None = None) -> None:
    """Refuse a history read before the broker's overnight close-data refresh."""
    current = now or datetime.now(JST)
    if current.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if current.astimezone(JST) < official_close_history_ready_at(trade_date):
        raise ValueError(
            "official close history is not ready yet; retry after 01:00 JST on the day "
            "following trade_date"
        )


def _finite_positive(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive number") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return number


def build_ml_overlay_forward_outcome(
    *,
    trade_date: str,
    input_digest: str,
    quote_snapshot: FrozenQuoteSnapshot,
    close_history_by_ticker: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Build one verified 09:10-mid-to-close label for a shadow input version.

    The caller must load the quote snapshot through ``load_frozen_quote_snapshot``
    and obtain daily rows from the broker's read-only market-history endpoint.
    The function fails on any missing, duplicate, wrong-date, or invalid close.
    """
    target_day = date.fromisoformat(trade_date)
    if quote_snapshot.trade_date != trade_date:
        raise ValueError("frozen quote snapshot trade_date mismatch")
    if quote_snapshot.source != "tachibana:CLMMfdsGetMarketPrice":
        raise ValueError("frozen quote snapshot source is not Tachibana market price")
    if quote_snapshot.timestamp_source != "local_response_receipt":
        raise ValueError("frozen quote snapshot lacks local response receipt time")
    if not input_digest:
        raise ValueError("input_digest is required")
    if set(close_history_by_ticker) != set(JP_TICKERS):
        raise ValueError("official close history must contain exactly the JP17 ticker universe")

    expected_api_date = target_day.strftime("%Y%m%d")
    target_returns: dict[str, float] = {}
    close_prices: dict[str, float] = {}
    close_row_sha256: dict[str, str] = {}
    for ticker in JP_TICKERS:
        quote_mid = _finite_positive(quote_snapshot.prices.get(ticker), f"quote_mid[{ticker}]")
        rows = close_history_by_ticker[ticker]
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise ValueError(f"close history for {ticker} must be a row sequence")
        if any(not isinstance(row, Mapping) for row in rows):
            raise ValueError(f"close history for {ticker} contains a non-object row")
        matched_rows = [row for row in rows if str(row.get("sDate") or "") == expected_api_date]
        if len(matched_rows) != 1:
            raise ValueError(
                f"official close history for {ticker} must contain exactly one row for {trade_date}"
            )
        close_row = dict(matched_rows[0])
        close_price = _finite_positive(close_row.get("pDPP"), f"official_close[{ticker}]")
        target_return = close_price / quote_mid - 1.0
        if not math.isfinite(target_return):
            raise ValueError(f"target return for {ticker} must be finite")
        target_returns[ticker] = target_return
        close_prices[ticker] = close_price
        close_row_sha256[ticker] = _canonical_digest(close_row)

    record: dict[str, Any] = {
        "schema_version": OUTCOME_SCHEMA,
        "trade_date": trade_date,
        "input_version": {"digest": input_digest},
        "quote_snapshot_id": quote_snapshot.snapshot_id,
        "label_status": "complete",
        "target_basis": OUTCOME_TARGET_BASIS,
        "target_source": OFFICIAL_CLOSE_SOURCE,
        "target_returns": target_returns,
        "official_close_prices": close_prices,
        "official_close_row_sha256": close_row_sha256,
    }
    record["outcome_fingerprint"] = _canonical_digest(record)
    return record


def _read_outcomes(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed outcome JSONL at {path}:{line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"outcome JSONL row at {path}:{line_number} is not an object")
            records.append(record)
    return records


def append_ml_overlay_forward_outcome(path: str | Path, record: Mapping[str, Any]) -> bool:
    """Append an outcome once; never replace a different label for the same input."""
    outcome = dict(record)
    fingerprint = outcome.pop("outcome_fingerprint", None)
    if fingerprint != _canonical_digest(outcome):
        raise ValueError("outcome fingerprint does not match its payload")
    if outcome.get("schema_version") != OUTCOME_SCHEMA:
        raise ValueError("unsupported ML-overlay forward outcome schema")
    trade_date = str(outcome.get("trade_date") or "")
    date.fromisoformat(trade_date)
    input_version = outcome.get("input_version")
    digest = input_version.get("digest") if isinstance(input_version, Mapping) else None
    if not digest:
        raise ValueError("outcome requires input_version.digest")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_path.with_name(f".{output_path.name}.lock")
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    with lock_path.open("a", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        existing_for_key = [
            row for row in _read_outcomes(output_path)
            if row.get("trade_date") == trade_date
            and isinstance(row.get("input_version"), Mapping)
            and row["input_version"].get("digest") == digest
        ]
        if len(existing_for_key) > 1:
            raise ValueError(f"multiple immutable outcomes already exist for {(trade_date, digest)}")
        if existing_for_key:
            existing = existing_for_key[0]
            if existing.get("outcome_fingerprint") != fingerprint:
                raise ValueError(f"refusing to replace a different outcome for {(trade_date, digest)}")
            return False
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        return True


__all__ = [
    "OFFICIAL_CLOSE_SOURCE",
    "OUTCOME_TARGET_BASIS",
    "append_ml_overlay_forward_outcome",
    "build_ml_overlay_forward_outcome",
    "ensure_official_close_history_ready",
    "official_close_history_ready_at",
]
