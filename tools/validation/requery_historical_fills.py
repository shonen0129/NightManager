"""Read-only historical Tachibana fill re-query with the correct business day."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from leadlag.execution.broker_ops import build_api_client
from leadlag.execution.pricing import fetch_fill_prices

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "reports/20260923_profitability_order_1/broker_requery.json"


def load_order_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted((ROOT / "var" / "results").glob("2026*_production_close_positions/close_execution_log.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("dry_run"):
            continue
        date_token = path.parent.name[:8]
        if not re.fullmatch(r"\d{8}", date_token):
            continue
        eigyou_day = date_token
        for source in payload.get("close_results", []):
            if not isinstance(source, dict) or not source.get("order_id"):
                continue
            record = {
                "order_id": str(source["order_id"]),
                "ticker": source.get("ticker"),
                "side": source.get("side"),
                "quantity": source.get("quantity"),
                "status": source.get("status"),
                "eigyou_day": eigyou_day,
                "source_path": str(path.relative_to(ROOT)),
            }
            records.append(record)
    return records


def main() -> int:
    records = load_order_records()
    output: dict[str, Any] = {
        "read_only": True,
        "order_count": len(records),
        "business_days": sorted({record["eigyou_day"] for record in records}),
        "orders": [],
        "errors": [],
    }
    broker = None
    try:
        broker = build_api_client(None, None, False)
        try:
            fetch_fill_prices(broker, records, wait_seconds=0.0)
        except Exception as exc:  # noqa: BLE001
            output["errors"].append(str(exc))
    except Exception as exc:  # noqa: BLE001
        output["errors"].append(f"broker_init: {exc}")
    finally:
        if broker is not None:
            broker.close()

    for record in records:
        output["orders"].append(
            {
                key: record.get(key)
                for key in (
                    "order_id",
                    "ticker",
                    "side",
                    "quantity",
                    "status",
                    "eigyou_day",
                    "fill_quantity",
                    "fill_price",
                    "fill_status",
                    "source_path",
                )
            }
        )
    output["confirmed_fill_order_count"] = sum(
        int(record.get("fill_quantity") or 0) > 0 for record in records
    )
    output["confirmed_fill_quantity"] = sum(
        int(record.get("fill_quantity") or 0) for record in records
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if not output["errors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
