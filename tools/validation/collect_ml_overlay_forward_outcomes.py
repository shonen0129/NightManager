#!/usr/bin/env python3
"""Collect read-only official JP closes for frozen 09:10 ML-shadow inputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from leadlag.broker.base import BrokerConfig
from leadlag.broker.tachibana.client import TachibanaBrokerClient
from leadlag.config.paths import project_root
from leadlag.data.quote_snapshot import load_frozen_quote_snapshot
from leadlag.data.tickers import JP_TICKERS
from leadlag.reporting.ml_overlay_forward_outcome import (
    append_ml_overlay_forward_outcome,
    build_ml_overlay_forward_outcome,
    ensure_official_close_history_ready,
)

ROOT = project_root()
DEFAULT_SHADOW_DIR = ROOT / "var/shadow_runs/ml_overlay_value"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"JSONL row at {path}:{line_number} must be an object")
            records.append(item)
    return records


def _input_versions_for_date(
    shadow_records: list[dict[str, Any]], trade_date: str
) -> dict[str, str]:
    versions: dict[str, str] = {}
    date_records = [item for item in shadow_records if item.get("trade_date") == trade_date]
    if not date_records:
        raise ValueError(f"no paired shadow records exist for {trade_date}")
    for record in date_records:
        input_version = record.get("input_version")
        digest = input_version.get("digest") if isinstance(input_version, dict) else None
        quote_id = str(record.get("quote_snapshot_id") or "")
        if not digest or not quote_id:
            raise ValueError(f"shadow record for {trade_date} lacks input digest or frozen quote ID")
        digest_text = str(digest)
        previous_quote_id = versions.setdefault(digest_text, quote_id)
        if previous_quote_id != quote_id:
            raise ValueError(f"one input digest refers to multiple quote snapshots on {trade_date}")
    quote_ids = set(versions.values())
    if len(quote_ids) != 1:
        raise ValueError(f"multiple frozen quote snapshots exist on {trade_date}")
    return versions


def _build_broker_client() -> TachibanaBrokerClient:
    api_url = os.environ.get(
        "TACHIBANA_API_URL", "https://kabuka.e-shiten.jp/e_api_v4r10"
    )
    if "e_api_v4r9" in api_url:
        raise ValueError("TACHIBANA_API_URL points to retired v4r9; configure the current v4r10 URL")
    return TachibanaBrokerClient(
        BrokerConfig(
            provider="tachibana",
            api_url=api_url,
            api_token=os.environ.get("TACHIBANA_AUTH_ID", ""),
            api_password=os.environ.get("TACHIBANA_SECOND_PASSWORD", ""),
            request_timeout=int(os.environ.get("TACHIBANA_REQUEST_TIMEOUT", "15")),
            extra={"private_key_path": os.environ.get("TACHIBANA_PRIVATE_KEY_PATH", "")},
        )
    )


def main() -> int:
    load_dotenv(ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trade-date", required=True, help="JP session date in YYYY-MM-DD")
    parser.add_argument("--shadow-dir", type=Path, default=DEFAULT_SHADOW_DIR)
    parser.add_argument("--quote-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    shadow_dir = args.shadow_dir if args.shadow_dir.is_absolute() else ROOT / args.shadow_dir
    quote_dir = args.quote_dir
    if quote_dir is None:
        configured_quote_dir = os.environ.get("LEADLAG_CAPTURE_OUTPUT_DIR")
        quote_dir = Path(configured_quote_dir) if configured_quote_dir else shadow_dir / "microstructure"
    if not quote_dir.is_absolute():
        quote_dir = ROOT / quote_dir
    output_path = args.output or (shadow_dir / "outcomes.jsonl")
    if not output_path.is_absolute():
        output_path = ROOT / output_path

    ensure_official_close_history_ready(args.trade_date)
    versions = _input_versions_for_date(_read_jsonl(shadow_dir / "daily.jsonl"), args.trade_date)
    snapshot = load_frozen_quote_snapshot(quote_dir, trade_date=args.trade_date)
    expected_quote_id = next(iter(versions.values()))
    if snapshot.snapshot_id != expected_quote_id:
        raise ValueError("shadow quote ID differs from the immutable 09:10 snapshot")

    client = _build_broker_client()
    try:
        # The broker supports one issue per history request; keep this sequential.
        history_by_ticker = client.fetch_market_price_history(JP_TICKERS)
    finally:
        client.close()

    appended = 0
    already_current = 0
    for input_digest in sorted(versions):
        record = build_ml_overlay_forward_outcome(
            trade_date=args.trade_date,
            input_digest=input_digest,
            quote_snapshot=snapshot,
            close_history_by_ticker=history_by_ticker,
        )
        if append_ml_overlay_forward_outcome(output_path, record):
            appended += 1
        else:
            already_current += 1

    print(json.dumps({
        "trade_date": args.trade_date,
        "quote_snapshot_id": snapshot.snapshot_id,
        "input_version_count": len(versions),
        "appended_count": appended,
        "already_current_count": already_current,
        "actual_execution_pnl": "not_collected",
        "output": str(output_path),
    }, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
