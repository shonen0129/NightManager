"""Summarize the user-provided Tachibana transaction CSV for reconciliation."""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = Path("/Users/shonen/Downloads/Tachibana_取引記録.csv")
OUT = Path(__file__).resolve().parent


def parse_number(value: str) -> float | None:
    cleaned = value.replace(",", "").replace("円", "").replace("株", "").strip()
    if not cleaned:
        return None
    sign = -1 if cleaned.startswith("-") else 1
    cleaned = cleaned.lstrip("+-")
    try:
        return sign * float(cleaned)
    except ValueError:
        return None


def parse_trade_date(delivery: str, executed: str) -> str | None:
    if not delivery or not executed:
        return None
    year = delivery.split("/", 1)[0]
    month, day = executed.split("/", 1)
    return date(int(year), int(month), int(day)).isoformat()


def load_csv() -> list[dict[str, str]]:
    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_local_orders() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(
        (ROOT / "var" / "results").glob("2026*_production_close_positions/close_execution_log.json")
    ):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("dry_run"):
            continue
        for record in payload.get("close_results", payload.get("orders", payload.get("order_results", []))):
            if not isinstance(record, dict):
                continue
            row = dict(record)
            row["source_path"] = str(path.relative_to(ROOT))
            match = re.search(r"/([0-9]{8})_[0-9]{6}_production_close_positions/", f"/{path.name}")
            if match is None:
                match = re.search(r"([0-9]{8})_[0-9]{6}_production_close_positions", str(path))
            row["trade_date"] = (
                f"{match.group(1)[:4]}-{match.group(1)[4:6]}-{match.group(1)[6:]}"
                if match
                else None
            )
            rows.append(row)
    return rows


def normalize_side(value: str) -> str | None:
    if "買" in value:
        return "BUY"
    if "売" in value:
        return "SELL"
    return None


def main() -> None:
    rows = load_csv()
    transactions: list[dict[str, Any]] = []
    for row in rows:
        trade = row.get("取引", "")
        executed = row.get("約定日", "")
        ticker = re.sub(r"\D", "", row.get("銘柄コード", ""))
        if not executed or not ticker or "返済" not in trade:
            continue
        transactions.append(
            {
                "trade_date": parse_trade_date(row.get("受渡日", ""), executed),
                "delivery_date": row.get("受渡日", ""),
                "executed_date_raw": executed,
                "ticker": f"{ticker}.T",
                "side": normalize_side(trade),
                "trade": trade,
                "quantity": parse_number(row.get("数量", "")),
                "price": parse_number(row.get("単価", "")),
                "pnl_or_settlement": parse_number(row.get("損益/受渡金額", "")),
                "account": row.get("口座/預かり", ""),
            }
        )

    local_orders = load_local_orders()
    official_groups: defaultdict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for transaction in transactions:
        key = (transaction["trade_date"], transaction["ticker"], transaction["side"])
        official_groups[key].append(transaction)
    local_groups: defaultdict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for order in local_orders:
        key = (order.get("trade_date"), order.get("ticker"), order.get("side"))
        local_groups[key].append(order)

    matches: list[dict[str, Any]] = []
    for key in sorted(local_groups):
        local_rows = local_groups[key]
        official_rows = official_groups.get(key, [])
        local_quantity = sum(float(row.get("quantity") or 0) for row in local_rows)
        official_quantity = sum(float(row.get("quantity") or 0) for row in official_rows)
        matches.append(
            {
                "trade_date": key[0],
                "ticker": key[1],
                "side": key[2],
                "local_order_ids": [str(row.get("order_id", "")) for row in local_rows],
                "local_order_count": len(local_rows),
                "local_quantity": local_quantity,
                "official_transaction_count": len(official_rows),
                "official_quantity": official_quantity,
                "quantity_reconciled": bool(official_rows) and local_quantity == official_quantity,
                "official_prices": sorted({row["price"] for row in official_rows}),
                "official_pnl_or_settlement_total": sum(
                    float(row.get("pnl_or_settlement") or 0) for row in official_rows
                ),
            }
        )
    matched_groups = [row for row in matches if row["quantity_reconciled"]]
    output = {
        "source": str(CSV_PATH),
        "source_exists": CSV_PATH.exists(),
        "csv_row_count": len(rows),
        "official_transaction_row_count": len(transactions),
        "local_real_order_count": len(local_orders),
        "local_order_ids": [str(row.get("order_id", "")) for row in local_orders],
        "local_group_count": len(local_groups),
        "local_group_reconciliations": matches,
        "quantity_reconciled_group_count": len(matched_groups),
        "quantity_reconciled_local_order_count": sum(
            row["local_order_count"] for row in matched_groups
        ),
        "local_quantity_total": sum(row["local_quantity"] for row in matches),
        "official_quantity_for_local_keys_total": sum(
            row["official_quantity"] for row in matches
        ),
        "unreconciled_local_group_count": len(matches) - len(matched_groups),
    }
    target = OUT / "tachibana_csv_analysis.json"
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in output.items() if key != "official_transactions"}, ensure_ascii=False, indent=2))
    for transaction in transactions:
        print(json.dumps(transaction, ensure_ascii=False))


if __name__ == "__main__":
    main()
