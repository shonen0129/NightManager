#!/usr/bin/env python3
"""Collect auditable 09:10 quote/LOB evidence without submitting orders.

The historical part of the report deliberately labels five-minute midpoint
data as a proxy.  A true 09:10 quote requires a timestamped broker response;
this command only records one as true when it is collected inside the target
window.  Missing bid/ask/depth/fill fields remain missing and are never
replaced by a fixed spread or a zero quantity.
"""

from __future__ import annotations

import argparse
import json
import os
import plistlib
import re
import subprocess
import time as time_module
from datetime import datetime, time, timedelta
from pathlib import Path
from statistics import median
from typing import Any
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from leadlag.data.quote_snapshot import freeze_quote_snapshot
from leadlag.data.tickers import JP_TICKERS_WITH_TOPIX

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "reports/20260923_profitability_order_2"
JST = ZoneInfo("Asia/Tokyo")
JP_TICKERS = [f"{code}.T" for code in range(1617, 1634)]
CAPTURE_TICKERS = list(JP_TICKERS_WITH_TOPIX)
_URL_QUERY_PATTERN = re.compile(r"(?P<url>https?://[^\s?]+)\?[^\s]+")


def _safe_error_message(exc: Exception) -> str:
    """Keep useful request errors while removing authentication query data."""
    return _URL_QUERY_PATTERN.sub(r"\g<url>?[redacted]", str(exc))


def _validated_api_url(api_url: str) -> str:
    """Reject the retired API endpoint before any broker request is made."""
    normalized = api_url.strip()
    if "e_api_v4r9" in normalized.casefold():
        raise ValueError(
            "TACHIBANA_API_URL points to retired v4r9; configure the current v4r10 URL"
        )
    return normalized


def _jst_now() -> datetime:
    return datetime.now(JST)


def _target_window(now: datetime, target: time, seconds: int) -> bool:
    target_dt = now.replace(hour=target.hour, minute=target.minute, second=0, microsecond=0)
    return abs((now - target_dt).total_seconds()) <= seconds


def _forward_capture_window(now: datetime, *, seconds_after: int = 30) -> bool:
    """Accept only timestamps from 09:10:00 through 09:10:30 JST."""
    target_dt = now.replace(hour=9, minute=10, second=0, microsecond=0)
    return timedelta(0) <= now - target_dt <= timedelta(seconds=seconds_after)


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_atomic(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(record, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _historical_proxy_coverage() -> dict[str, Any]:
    """Summarize local 5-minute proxy coverage without changing any cache."""
    from leadlag.data.intraday_inputs import build_5m_910_prices
    from leadlag.data.market_data_cache import (
        load_df_exec_from_local_cache,
        load_intraday_cache,
    )

    df_exec = load_df_exec_from_local_cache()
    bars = load_intraday_cache("5m")
    if bars is None or bars.empty:
        return {
            "source": "local_5m_cache",
            "status": "MISSING",
            "quote_days": 0,
            "complete_cross_section_days": 0,
            "true_0910": False,
        }

    prices = build_5m_910_prices(df_exec, JP_TICKERS, bars)
    available = prices.notna()
    days_with_any = int((available.sum(axis=1) > 0).sum())
    complete_days = int((available.sum(axis=1) == len(JP_TICKERS)).sum())
    per_ticker = {
        ticker: int(available[ticker].sum()) for ticker in JP_TICKERS
    }
    return {
        "source": "local_5m_cache",
        "status": "PROXY_ONLY",
        "true_0910": False,
        "quote_days": days_with_any,
        "complete_cross_section_days": complete_days,
        "per_ticker_days": per_ticker,
        "observed_at_semantics": "5m bar midpoint/close proxy, not a timestamped executable quote",
    }


def _load_stage1_fill_evidence() -> dict[str, Any]:
    path = ROOT / "reports/20260923_profitability_order_1/reconciliation.json"
    if not path.exists():
        return {"status": "MISSING_STAGE1_REPORT"}
    payload = json.loads(path.read_text(encoding="utf-8"))
    coverage = payload.get("coverage", {})
    broker = payload.get("broker_requery", {})
    local_fill_orders = 0
    local_fill_quantity = 0
    local_fill_price_count = 0
    local_fee_detail_count = 0
    local_fee_nonzero_count = 0
    local_fill_order_ids: list[str] = []
    local_fill_records: list[dict[str, Any]] = []
    fee_fields = (
        "sKessaiTateTesuryou",
        "sKessaiZyunHibu",
        "sKessaiGyakuhibu",
        "sKessaiKakikaeryou",
        "sKessaiKanrihi",
        "sKessaiKasikaburyou",
        "sKessaiSonota",
    )

    def _nonzero(value: Any) -> bool:
        try:
            return float(str(value or "0").replace(",", "")) != 0.0
        except (TypeError, ValueError):
            return False

    def _number(value: Any) -> float | None:
        try:
            return float(str(value).replace(",", ""))
        except (TypeError, ValueError):
            return None

    for log_path in sorted(
        (ROOT / "var" / "results").glob("2026*_production_close_positions/close_execution_log.json")
    ):
        close_log = json.loads(log_path.read_text(encoding="utf-8"))
        if close_log.get("dry_run"):
            continue
        for row in close_log.get("close_results", []):
            if not isinstance(row, dict) or int(row.get("fill_quantity") or 0) <= 0:
                continue
            local_fill_orders += 1
            local_fill_quantity += int(row.get("fill_quantity") or 0)
            if row.get("fill_price") is not None:
                local_fill_price_count += 1
            local_fill_order_ids.append(str(row.get("order_id", "")))
            detail = row.get("fill_detail")
            settlement = {}
            if isinstance(detail, dict):
                entries = detail.get("aKessaiOrderTategyokuList")
                if isinstance(entries, list) and entries and isinstance(entries[0], dict):
                    settlement = entries[0]
                elif isinstance(entries, dict):
                    settlement = entries
            if settlement:
                local_fee_detail_count += 1
                if any(_nonzero(settlement.get(field, "0")) for field in fee_fields):
                    local_fee_nonzero_count += 1
            directory_date = log_path.parent.name[:8]
            trade_date = (
                f"{directory_date[:4]}-{directory_date[4:6]}-{directory_date[6:8]}"
                if len(directory_date) == 8 and directory_date.isdigit()
                else None
            )
            local_fill_records.append(
                {
                    "trade_date": trade_date,
                    "order_id": str(row.get("order_id", "")),
                    "ticker": row.get("ticker"),
                    "side": row.get("side"),
                    "requested_quantity": int(row.get("quantity") or 0),
                    "filled_quantity": int(row.get("fill_quantity") or 0),
                    "fill_price": _number(row.get("fill_price")),
                    "original_side": row.get("original_side"),
                    "original_price": _number(row.get("original_price")),
                    "realized_pnl": _number(settlement.get("sKessaiSoneki")),
                    "fee_fields": {
                        field: _number(settlement.get(field))
                        for field in fee_fields
                        if _number(settlement.get(field)) is not None
                    },
                    "slippage_bps": None,
                    "slippage_status": "missing_reference_9_10_quote",
                }
            )
    return {
        "status": "AUTHORITATIVE_CSV_WITHOUT_ORDER_IDS",
        "official_quantity_for_local_keys": coverage.get("official_quantity_for_local_keys"),
        "runtime_confirmed_quantity": coverage.get("local_confirmed_quantity_from_runtime_logs"),
        "runtime_quantity_gap": coverage.get("runtime_log_quantity_gap"),
        "broker_detail_status": broker.get("status"),
        "local_runtime_fill_orders": local_fill_orders,
        "local_runtime_fill_quantity": local_fill_quantity,
        "local_runtime_fill_price_count": local_fill_price_count,
        "local_runtime_fee_detail_count": local_fee_detail_count,
        "local_runtime_fee_nonzero_count": local_fee_nonzero_count,
        "local_runtime_fill_order_ids": local_fill_order_ids,
        "local_runtime_fill_records": local_fill_records,
        "fees_observed": local_fee_detail_count > 0,
        "all_requested_fills_observed": (
            local_fill_quantity == coverage.get("local_requested_quantity")
        ),
        "source": "reports/20260923_profitability_order_1/reconciliation.json",
    }


def _collect_live_quotes(
    tickers: list[str],
    observed_at: datetime,
) -> dict[str, Any]:
    """Read Tachibana market data only; this function has no order methods."""
    from leadlag.broker.base import BrokerConfig
    from leadlag.broker.tachibana.client import TachibanaBrokerClient

    config = BrokerConfig(
        provider="tachibana",
        api_url=_validated_api_url(os.environ.get("TACHIBANA_API_URL", "")),
        api_token=os.environ.get("TACHIBANA_AUTH_ID", ""),
        api_password=os.environ.get("TACHIBANA_SECOND_PASSWORD", ""),
        request_timeout=int(os.environ.get("TACHIBANA_REQUEST_TIMEOUT", "15")),
        extra={"private_key_path": os.environ.get("TACHIBANA_PRIVATE_KEY_PATH", "")},
    )
    client = TachibanaBrokerClient(config)
    request_started_at = observed_at
    try:
        quotes = client.fetch_market_quotes(
            tickers,
            observed_at=request_started_at.isoformat(),
            allow_missing=True,
        )
        response_received_at = _jst_now()
    finally:
        client.close()

    by_ticker = {str(row["ticker"]): row for row in quotes}
    rows: list[dict[str, Any]] = []
    for ticker in tickers:
        row = by_ticker.get(ticker)
        if row is None:
            rows.append(
                {
                    "ticker": ticker,
                    "observed_at": response_received_at.isoformat(),
                    "request_started_at": request_started_at.isoformat(),
                    "response_received_at": response_received_at.isoformat(),
                    "timestamp_source": "local_response_receipt",
                    "source": "tachibana:CLMMfdsGetMarketPrice",
                    "status": "MISSING",
                    "missing_reason": "broker_response_did_not_contain_ticker",
                    "fill_quantity": None,
                    "fill_price": None,
                }
            )
            continue
        row = dict(row)
        row["request_started_at"] = request_started_at.isoformat()
        row["response_received_at"] = response_received_at.isoformat()
        row["timestamp_source"] = "local_response_receipt"
        # Preserve request start separately; observed_at is the auditable
        # local time at which the quote response reached this process.
        row["observed_at"] = response_received_at.isoformat()
        row["timestamp"] = response_received_at.isoformat()
        if row.get("lob_available"):
            bid = row.get("bid_price_1")
            ask = row.get("ask_price_1")
            if bid is not None and ask is not None and bid > 0 and ask > bid:
                mid = (bid + ask) / 2.0
                row["quote_mid_price"] = mid
                row["quoted_spread_bps"] = (ask - bid) / mid * 10000.0
                row["ask_depth_jpy_5"] = sum(
                    (row.get(f"ask_price_{level}") or 0.0)
                    * (row.get(f"ask_size_{level}") or 0.0)
                    for level in range(1, 6)
                )
                row["bid_depth_jpy_5"] = sum(
                    (row.get(f"bid_price_{level}") or 0.0)
                    * (row.get(f"bid_size_{level}") or 0.0)
                    for level in range(1, 6)
                )
            else:
                row["quote_mid_price"] = None
                row["quoted_spread_bps"] = None
                row["ask_depth_jpy_5"] = None
                row["bid_depth_jpy_5"] = None
        else:
            row["quote_mid_price"] = None
            row["quoted_spread_bps"] = None
            row["ask_depth_jpy_5"] = None
            row["bid_depth_jpy_5"] = None
        row["valid_two_sided_quote"] = bool(
            row.get("bid_price_1") is not None
            and row.get("ask_price_1") is not None
            and row["bid_price_1"] > 0
            and row["ask_price_1"] > row["bid_price_1"]
        )
        row["full_five_level_depth"] = all(
            row.get(f"{side}_{field}_{level}") is not None
            and row[f"{side}_{field}_{level}"] > 0
            for side in ("bid", "ask")
            for field in ("price", "size")
            for level in range(1, 6)
        )
        row.update(
            {
                "status": "OBSERVED",
                "target": "09:10",
                "fill_quantity": None,
                "fill_price": None,
                "fill_source": "not_an_order_query",
            }
        )
        rows.append(row)
    spreads = [
        float(row["quoted_spread_bps"])
        for row in rows
        if row.get("quoted_spread_bps") is not None
    ]
    return {
        "status": "OBSERVED",
        "request_started_at": request_started_at.isoformat(),
        "response_received_at": response_received_at.isoformat(),
        "observed_at": response_received_at.isoformat(),
        "timestamp_source": "local_response_receipt",
        "source": "tachibana:CLMMfdsGetMarketPrice",
        "rows": rows,
        "observed_count": sum(row["status"] == "OBSERVED" for row in rows),
        "lob_count": sum(bool(row.get("valid_two_sided_quote")) for row in rows),
        "full_five_level_depth_count": sum(
            bool(row.get("full_five_level_depth")) for row in rows
        ),
        "spread_bps_summary": (
            {
                "count": len(spreads),
                "min": min(spreads),
                "median": median(spreads),
                "max": max(spreads),
            }
            if spreads
            else {"count": 0, "min": None, "median": None, "max": None}
        ),
    }


def _run_capture_only(
    *,
    output_dir: Path,
    attempts: int = 2,
    retry_delay_seconds: float = 2.0,
    window_seconds: int = 30,
) -> int:
    """Collect promptly and persist append-only evidence without report scans."""
    run_started_at = _jst_now()
    run_id = run_started_at.strftime("%Y%m%dT%H%M%S%f%z")
    run_path = output_dir / "capture_runs.jsonl"
    snapshot_path = output_dir / "quote_snapshots.jsonl"
    attempts = max(1, attempts)
    records: list[dict[str, Any]] = []

    from leadlag.core.market_calendar import is_trading_day

    if not is_trading_day(run_started_at):
        record = {
            "run_id": run_id,
            "status": "MARKET_CLOSED",
            "scheduled_date": run_started_at.date().isoformat(),
            "run_started_at": run_started_at.isoformat(),
            "target": "09:10:00-09:10:30 JST",
        }
        _append_jsonl(run_path, record)
        _write_json_atomic(output_dir / f"capture_{run_started_at:%Y%m%d}.json", record)
        print(json.dumps(record, ensure_ascii=False))
        return 0

    if not _forward_capture_window(run_started_at, seconds_after=window_seconds):
        record = {
            "run_id": run_id,
            "status": "OUTSIDE_CAPTURE_WINDOW",
            "scheduled_date": run_started_at.date().isoformat(),
            "run_started_at": run_started_at.isoformat(),
            "target": "09:10:00-09:10:30 JST",
        }
        _append_jsonl(run_path, record)
        _write_json_atomic(output_dir / f"capture_{run_started_at:%Y%m%d}.json", record)
        print(json.dumps(record, ensure_ascii=False))
        return 78

    for attempt in range(1, attempts + 1):
        request_started_at = _jst_now()
        if not _forward_capture_window(request_started_at, seconds_after=window_seconds):
            break
        try:
            snapshot = _collect_live_quotes(CAPTURE_TICKERS, request_started_at)
        except Exception as exc:
            record = {
                "run_id": run_id,
                "attempt": attempt,
                "status": "ERROR",
                "request_started_at": request_started_at.isoformat(),
                "error_type": type(exc).__name__,
                "error": _safe_error_message(exc),
            }
            records.append(record)
            _append_jsonl(run_path, record)
            if attempt < attempts:
                now = _jst_now()
                remaining = (now.replace(hour=9, minute=10, second=0, microsecond=0)
                             + timedelta(seconds=window_seconds) - now).total_seconds()
                if remaining > 0:
                    time_module.sleep(min(max(0.0, retry_delay_seconds), remaining))
                    continue
            break

        received_at = datetime.fromisoformat(snapshot["response_received_at"])
        window_valid = (
            request_started_at.date() == received_at.date()
            and _forward_capture_window(request_started_at, seconds_after=window_seconds)
            and _forward_capture_window(received_at, seconds_after=window_seconds)
        )
        snapshot.update(
            {
                "run_id": run_id,
                "attempt": attempt,
                "window_valid": window_valid,
                "target": "09:10:00-09:10:30 JST",
            }
        )
        _append_jsonl(snapshot_path, snapshot)
        capture_status = (
            "CAPTURED"
            if window_valid
            and snapshot["observed_count"] == len(CAPTURE_TICKERS)
            and snapshot["lob_count"] == len(CAPTURE_TICKERS)
            else "CAPTURED_PARTIAL" if window_valid
            else "RESPONSE_OUTSIDE_CAPTURE_WINDOW"
        )
        record = {
            "run_id": run_id,
            "attempt": attempt,
            "status": capture_status,
            "request_started_at": snapshot["request_started_at"],
            "response_received_at": snapshot["response_received_at"],
            "window_valid": window_valid,
            "observed_count": snapshot["observed_count"],
            "two_sided_quote_count": snapshot["lob_count"],
            "full_five_level_depth_count": snapshot["full_five_level_depth_count"],
            "spread_bps_summary": snapshot["spread_bps_summary"],
        }
        records.append(record)
        _append_jsonl(run_path, record)
        if (
            window_valid
            and snapshot["observed_count"] == len(CAPTURE_TICKERS)
            and snapshot["lob_count"] == len(CAPTURE_TICKERS)
        ):
            freeze_path = freeze_quote_snapshot(
                snapshot,
                output_dir,
                trade_date=run_started_at.date().isoformat(),
                required_tickers=tuple(CAPTURE_TICKERS),
            )
            snapshot["frozen_snapshot_path"] = str(freeze_path)
            _write_json_atomic(
                output_dir / f"capture_{run_started_at:%Y%m%d}.json",
                {"run_id": run_id, "status": capture_status, "attempts": records},
            )
            print(json.dumps(record, ensure_ascii=False))
            return 0
        if attempt < attempts:
            now = _jst_now()
            remaining = (now.replace(hour=9, minute=10, second=0, microsecond=0)
                         + timedelta(seconds=window_seconds) - now).total_seconds()
            if remaining > 0:
                time_module.sleep(min(max(0.0, retry_delay_seconds), remaining))

    final_record = {
        "run_id": run_id,
        "status": "FAILED",
        "scheduled_date": run_started_at.date().isoformat(),
        "run_started_at": run_started_at.isoformat(),
        "target": "09:10:00-09:10:30 JST",
        "attempts": records,
    }
    _append_jsonl(run_path, final_record)
    _write_json_atomic(output_dir / f"capture_{run_started_at:%Y%m%d}.json", final_record)
    print(json.dumps(final_record, ensure_ascii=False))
    return 1


def _load_stored_snapshot_summary(output_dir: Path) -> dict[str, Any] | None:
    """Read prior append-only captures so report regeneration preserves evidence."""
    snapshot_path = output_dir / "quote_snapshots.jsonl"
    if not snapshot_path.exists():
        return None
    records: list[dict[str, Any]] = []
    with snapshot_path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
    if not records:
        return None
    eligible_quotes = [
        record
        for record in records
        if record.get("status") == "OBSERVED"
        and record.get("window_valid") is True
        and record.get("observed_count") == len(CAPTURE_TICKERS)
        and record.get("lob_count") == len(CAPTURE_TICKERS)
    ]
    complete_depth = [
        record for record in eligible_quotes
        if record.get("full_five_level_depth_count") == len(CAPTURE_TICKERS)
    ]
    latest = records[-1]
    return {
        "status": "STORED_CAPTURE_SUMMARY",
        "stored_capture_count": len(records),
        "true_0910_capture_count": len(eligible_quotes),
        "complete_two_sided_quote_capture_count": len(eligible_quotes),
        "complete_five_level_capture_count": len(complete_depth),
        "latest_observed_at": latest.get("observed_at"),
        "observed_count": latest.get("observed_count"),
        "lob_count": latest.get("lob_count"),
        "full_five_level_depth_count": latest.get("full_five_level_depth_count"),
        "spread_bps_summary": latest.get("spread_bps_summary", {}),
        "true_0910_observed": bool(eligible_quotes),
    }


def _inspect_installed_scheduler() -> dict[str, Any]:
    """Inspect the independent capture LaunchAgent without modifying it."""
    path = Path.home() / "Library/LaunchAgents/com.leadlag.microstructure-0910.plist"
    if not path.exists():
        return {
            "status": "NOT_INSTALLED",
            "path": str(path),
            "label": "com.leadlag.microstructure-0910",
        }
    try:
        with path.open("rb") as handle:
            payload = plistlib.load(handle)
        try:
            registered = subprocess.run(
                ["launchctl", "print", f"gui/{os.getuid()}/com.leadlag.microstructure-0910"],
                capture_output=True,
                check=False,
                timeout=3,
            ).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            registered = False
        return {
            "status": "REGISTERED" if registered else "PLIST_PRESENT_NOT_REGISTERED",
            "path": str(path),
            "label": payload.get("Label"),
            "registered": registered,
            "schedule": payload.get("StartCalendarInterval"),
            "program": payload.get("ProgramArguments", []),
        }
    except (OSError, plistlib.InvalidFileException, ValueError) as exc:
        return {
            "status": "UNREADABLE",
            "path": str(path),
            "label": "com.leadlag.microstructure-0910",
            "error": str(exc),
        }


def build_report(
    *,
    output_dir: Path,
    collect_live: bool,
    allow_outside_window: bool,
    window_seconds: int,
) -> dict[str, Any]:
    now = _jst_now()
    target = time(9, 10)
    proxy = _historical_proxy_coverage()
    fills = _load_stage1_fill_evidence()
    stored_summary = _load_stored_snapshot_summary(output_dir)
    live: dict[str, Any]
    in_window = _target_window(now, target, window_seconds)
    if not collect_live:
        live = stored_summary or {
            "status": "NOT_RUN",
            "reason": "--collect-live was not specified",
            "target": "09:10 JST",
        }
    elif not in_window and not allow_outside_window:
        live = {
            "status": "NOT_RUN_OUTSIDE_0910_WINDOW",
            "reason": "read-only quote was not queried outside the 09:10 capture window",
            "now_jst": now.isoformat(),
            "target": "09:10 JST",
            "window_seconds": window_seconds,
        }
    else:
        try:
            live = _collect_live_quotes(CAPTURE_TICKERS, now)
            live["window_valid"] = in_window
            if not in_window:
                live["status"] = "OBSERVED_OUTSIDE_0910_WINDOW"
                live["not_eligible_as_true_0910"] = True
        except Exception as exc:
            live = {
                "status": "ERROR",
                "error_type": type(exc).__name__,
                "error": _safe_error_message(exc),
                "window_valid": in_window,
            }

    true_quote_observed = bool(
        live.get("true_0910_observed")
        or (live.get("status") == "OBSERVED" and live.get("window_valid", True))
    )
    complete = bool(
        true_quote_observed
        and live.get("observed_count") == len(CAPTURE_TICKERS)
        and live.get("lob_count") == len(CAPTURE_TICKERS)
        and fills.get("broker_detail_status") == "PASS"
    )
    return {
        "stage": 2,
        "stage_name": "true_0910_prices_spreads_order_book_and_fills",
        "status": "COMPLETE" if complete else "INSTRUMENTED_HISTORICAL_DATA_INSUFFICIENT",
        "decision": "PROCEED_TO_STAGE_3" if complete else "HOLD_BEFORE_STAGE_3",
        "read_only": True,
        "capture_time_jst": now.isoformat(),
        "target_time_jst": "09:10",
        "historical_proxy": proxy,
        "live_capture": live,
        "fill_evidence": fills,
        "completion_rules": {
            "true_timestamped_0910_quote": true_quote_observed,
            "all_17_jp_plus_topix_two_sided_quotes": (
                live.get("lob_count") == len(CAPTURE_TICKERS)
            ),
            "all_17_jp_plus_topix_five_level_books": (
                live.get("full_five_level_depth_count") == len(CAPTURE_TICKERS)
            ),
            "broker_fill_and_fee_detail": bool(
                fills.get("broker_detail_status") == "PASS"
                or fills.get("all_requested_fills_observed")
            ),
            "no_proxy_promoted_to_true_quote": True,
        },
        "unresolved": [
            "Historical five-minute bars are proxy observations, not executable 09:10 quotes.",
            "The official CSV has no broker order IDs and the historical detail re-query is unavailable.",
            "No fill quantity/price/fee is inferred from a market quote snapshot.",
        ],
    }


def write_report(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_dir / "quote_snapshots.jsonl"
    live = report["live_capture"]
    if live.get("rows") and live.get("status") in {
        "OBSERVED",
        "OBSERVED_OUTSIDE_0910_WINDOW",
    }:
        with snapshot_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(live, ensure_ascii=False) + "\n")
    stored_snapshot_count = 0
    if snapshot_path.exists():
        with snapshot_path.open(encoding="utf-8") as handle:
            stored_snapshot_count = sum(1 for _ in handle)
    report["storage"] = {
        "snapshot_jsonl": str(snapshot_path.relative_to(ROOT)),
        "stored_snapshot_count": stored_snapshot_count,
        "append_only": True,
    }
    report["operational_capture_hook"] = {
        "capture_script": "scripts/batch/run_0910_microstructure_capture.sh",
        "scheduler_plist": "scripts/batch/com.leadlag.microstructure-0910.plist",
        "scheduler_install_executed": False,
        "read_only_market_data_only": True,
        "installed_scheduler": _inspect_installed_scheduler(),
    }
    (output_dir / "microstructure.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    proxy = report["historical_proxy"]
    fills = report["fill_evidence"]
    spread_summary = live.get("spread_bps_summary", {})
    lines = [
        "# 収益改善順序2：9:10価格・spread・板・約定の蓄積",
        "",
        "発注・取消・再送を行わない読み取り専用の検証です。",
        "",
        f"判定: **{report['status']}** / 次段階: **{report['decision']}**",
        "",
        "## 観測状況",
        "",
        f"- historical 5分足 proxy: `{proxy.get('status')}`, quote days={proxy.get('quote_days', 0)}, complete cross-section days={proxy.get('complete_cross_section_days', 0)}",
        f"- live capture: `{live.get('status')}`",
        f"- stored timestamped captures: {stored_snapshot_count} (append-only JSONL)",
        "- independent 9:10 capture job: `com.leadlag.microstructure-0910`",
        f"- installed scheduler: `{report['operational_capture_hook']['installed_scheduler'].get('status')}`",
        f"- live quote spread bps: count={spread_summary.get('count', 0)}, min={spread_summary.get('min')}, median={spread_summary.get('median')}, max={spread_summary.get('max')}",
        f"- two-sided quote count={live.get('lob_count')}; full five-level depth count={live.get('full_five_level_depth_count')}",
        f"- fill detail: `{fills.get('broker_detail_status')}`; fees observed={fills.get('fees_observed')}",
        f"- local confirmed fills: orders={fills.get('local_runtime_fill_orders', 0)}, quantity={fills.get('local_runtime_fill_quantity', 0)}, fee-detail rows={fills.get('local_runtime_fee_detail_count', 0)}",
        "",
        "## 完了条件",
        "",
        "| 条件 | 結果 |",
        "|---|---|",
        f"| timestamped 09:10 quote | {report['completion_rules']['true_timestamped_0910_quote']} |",
        f"| 17銘柄の二面気配 | {report['completion_rules']['all_17_two_sided_quotes']} |",
        f"| 17銘柄の完全な5段板 | {report['completion_rules']['all_17_five_level_books']} |",
        f"| broker約定・手数料明細 | {report['completion_rules']['broker_fill_and_fee_detail']} |",
        f"| proxyの格上げなし | {report['completion_rules']['no_proxy_promoted_to_true_quote']} |",
        "",
        "## 未解決事項",
        "",
    ]
    lines.extend(f"- {item}" for item in report["unresolved"])
    lines.extend(
        [
            "",
            "順序2は収集器と欠損管理を実装済みだが、過去期間の真の9:10 quote/板/約定明細がまだ不足しているため完了扱いにしない。09:10 JSTの稼働日に `--collect-live` を実行し、得られた観測を蓄積してから順序3へ進める。",
            "",
            f"再現JSON: `{(output_dir / 'microstructure.json').relative_to(ROOT)}`",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--collect-live", action="store_true")
    parser.add_argument(
        "--capture-only",
        action="store_true",
        help="Capture quotes promptly and append evidence without scanning historical data.",
    )
    parser.add_argument("--allow-outside-window", action="store_true")
    parser.add_argument("--window-seconds", type=int, default=120)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--retry-delay-seconds", type=float, default=2.0)
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    if args.capture_only:
        return _run_capture_only(
            output_dir=output_dir,
            attempts=args.attempts,
            retry_delay_seconds=args.retry_delay_seconds,
            window_seconds=args.window_seconds,
        )
    report = build_report(
        output_dir=output_dir,
        collect_live=args.collect_live,
        allow_outside_window=args.allow_outside_window,
        window_seconds=args.window_seconds,
    )
    write_report(report, output_dir)
    live = report["live_capture"]
    print(
        json.dumps(
            {
                "status": report["status"],
                "decision": report["decision"],
                "live_capture": {
                    key: live.get(key)
                    for key in (
                        "status",
                        "observed_at",
                        "observed_count",
                        "lob_count",
                        "spread_bps_summary",
                        "window_valid",
                        "true_0910_capture_count",
                    )
                    if key in live
                },
                "stored_snapshot_count": report["storage"]["stored_snapshot_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
