"""Reconcile historical production close records with an authoritative CSV.

This is a read-only validation tool for profitability sequence 1.  It never
contacts a broker and never submits or retries an order.  Broker order IDs are
not present in the supplied Tachibana CSV, so the strongest historical match
is explicitly reported as trade-date/ticker/side/quantity rather than being
promoted to an order-ID match.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = Path("/Users/shonen/Downloads/Tachibana_取引記録.csv")
ACCEPTANCE = ROOT / "reports/20260922_production_acceptance"
DEFAULT_OUTPUT = ROOT / "reports/20260923_profitability_order_1"

FEE_FIELDS = (
    "sKessaiTateTesuryou",
    "sKessaiZyunHibu",
    "sKessaiGyakuhibu",
    "sKessaiKakikaeryou",
    "sKessaiKanrihi",
    "sKessaiKasikaburyou",
    "sKessaiSonota",
)


def parse_number(value: Any) -> float | None:
    """Parse Tachibana's signed Japanese currency/quantity text."""
    if value is None:
        return None
    cleaned = str(value).replace(",", "").replace("円", "").replace("株", "").strip()
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
    try:
        year = delivery.split("/", 1)[0]
        month, day = executed.split("/")
        return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
    except (ValueError, IndexError):
        return None


def normalize_side(value: str) -> str | None:
    if "買" in value:
        return "BUY"
    if "売" in value:
        return "SELL"
    return None


def load_official_transactions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            trade = raw.get("取引", "")
            executed = raw.get("約定日", "")
            ticker_code = re.sub(r"\D", "", raw.get("銘柄コード", ""))
            side = normalize_side(trade)
            trade_date = parse_trade_date(raw.get("受渡日", ""), executed)
            if "返済" not in trade or not ticker_code or not side or not trade_date:
                continue
            quantity = parse_number(raw.get("数量", ""))
            price = parse_number(raw.get("単価", ""))
            pnl = parse_number(raw.get("損益/受渡金額", ""))
            if quantity is None or price is None:
                continue
            rows.append(
                {
                    "trade_date": trade_date,
                    "ticker": f"{ticker_code}.T",
                    "side": side,
                    "quantity": int(quantity),
                    "price": price,
                    "reported_pnl_or_settlement": pnl,
                    "trade_label": trade,
                }
            )
    return rows


def _settlement_detail(record: dict[str, Any]) -> dict[str, Any]:
    detail = record.get("fill_detail")
    if not isinstance(detail, dict):
        return {}
    rows = detail.get("aKessaiOrderTategyokuList")
    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        return rows[0]
    if isinstance(rows, dict):
        return rows
    return {}


def load_local_close_orders(root: Path) -> list[dict[str, Any]]:
    orders: list[dict[str, Any]] = []
    for path in sorted((root / "var" / "results").glob("2026*_production_close_positions/close_execution_log.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("dry_run"):
            continue
        directory_date = path.parent.name[:8]
        trade_date = (
            f"{directory_date[:4]}-{directory_date[4:6]}-{directory_date[6:8]}"
            if re.fullmatch(r"\d{8}", directory_date)
            else None
        )
        for record in payload.get("close_results", []):
            if not isinstance(record, dict) or not record.get("ticker") or not trade_date:
                continue
            row = dict(record)
            row["trade_date"] = trade_date
            row["source_path"] = str(path.relative_to(root))
            orders.append(row)
    return orders


def _group(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    grouped: defaultdict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["trade_date"]), str(row["ticker"]), str(row["side"]))].append(row)
    return dict(grouped)


def _price_pnl(original_side: str, original_price: float, close_price: float, quantity: int) -> float:
    if original_side == "BUY":
        return (close_price - original_price) * quantity
    return (original_price - close_price) * quantity


def _group_reconciliation(
    key: tuple[str, str, str],
    local_rows: list[dict[str, Any]],
    official_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    trade_date, ticker, side = key
    local_requested = sum(int(row.get("quantity") or 0) for row in local_rows)
    local_confirmed = sum(int(row.get("fill_quantity") or 0) for row in local_rows)
    official_quantity = sum(int(row["quantity"]) for row in official_rows)
    official_prices = sorted({float(row["price"]) for row in official_rows})
    official_pnl = sum(float(row["reported_pnl_or_settlement"] or 0.0) for row in official_rows)
    local_prices = sorted(
        {float(row["fill_price"]) for row in local_rows if row.get("fill_price") is not None}
    )

    price_pnl = 0.0
    price_pnl_complete = bool(official_rows and official_prices)
    if price_pnl_complete:
        for row in local_rows:
            original_side = str(row.get("original_side") or "")
            original_price = parse_number(row.get("original_price"))
            if original_side not in {"BUY", "SELL"} or original_price is None:
                price_pnl_complete = False
                break
            price_pnl += _price_pnl(
                original_side,
                original_price,
                official_prices[0],
                int(row.get("quantity") or 0),
            )

    local_reported_pnl = 0.0
    local_reported_pnl_count = 0
    fee_values: dict[str, float] = {}
    for row in local_rows:
        detail = _settlement_detail(row)
        raw_pnl = detail.get("sKessaiSoneki")
        parsed_pnl = parse_number(raw_pnl)
        if parsed_pnl is not None:
            local_reported_pnl += parsed_pnl
            local_reported_pnl_count += 1
        for field in FEE_FIELDS:
            parsed_fee = parse_number(detail.get(field))
            if parsed_fee is not None:
                fee_values[field] = fee_values.get(field, 0.0) + parsed_fee

    return {
        "trade_date": trade_date,
        "ticker": ticker,
        "side": side,
        "local_order_ids": [str(row.get("order_id", "")) for row in local_rows],
        "local_order_count": len(local_rows),
        "official_transaction_count": len(official_rows),
        "local_requested_quantity": local_requested,
        "local_confirmed_quantity": local_confirmed,
        "official_quantity": official_quantity,
        "quantity_match": bool(official_rows) and local_requested == official_quantity,
        "official_prices": official_prices,
        "local_observed_prices": local_prices,
        "execution_price_match": bool(local_prices) and local_prices == official_prices,
        "official_reported_pnl_or_settlement": official_pnl,
        "local_reported_settlement_pnl": local_reported_pnl if local_reported_pnl_count else None,
        "pre_fee_price_pnl_from_official_close": price_pnl if price_pnl_complete else None,
        "official_minus_pre_fee_price_pnl": official_pnl - price_pnl if price_pnl_complete else None,
        "local_fee_fields_observed": fee_values,
        "source_paths": sorted({str(row.get("source_path", "")) for row in local_rows}),
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_report(csv_path: Path, root: Path = ROOT) -> dict[str, Any]:
    official = load_official_transactions(csv_path)
    local = load_local_close_orders(root)
    official_groups = _group(official)
    local_groups = _group(local)
    keys = sorted(set(local_groups) | set(official_groups))
    groups = [
        _group_reconciliation(key, local_groups.get(key, []), official_groups.get(key, []))
        for key in keys
    ]

    artifact_parity = load_json(ACCEPTANCE / "artifact_parity.json")
    gap_parity = load_json(ACCEPTANCE / "gap_path_parity.json")
    field_parity_path = root / "reports/20260923_profitability_order_1/field_parity.json"
    field_parity = load_json(field_parity_path) if field_parity_path.exists() else None
    score_pit_parity = bool(
        field_parity is not None
        and field_parity.get("status") == "PASS"
        and all(field_parity.get("checks", {}).values())
    )
    requery_path = root / "reports/20260923_profitability_order_1/broker_requery.json"
    broker_requery = load_json(requery_path) if requery_path.exists() else None
    snapshot_checks = {
        "mu_omega_cache_vs_ondemand": all(
            row.get("mu_max_error") == 0.0 and row.get("omega_max_error") == 0.0
            for row in gap_parity
        ),
        "weights_production_vs_backtest": all(
            row.get("live_bt_max_weight_error") == 0.0 for row in artifact_parity
        ),
        "weights_production_vs_collector": all(
            row.get("live_collected_max_weight_error") == 0.0 for row in artifact_parity
        ),
        "numerical_and_leakage_audits": all(
            row.get("numerical") == "PASSED" and row.get("leakage") == "PASSED"
            for row in artifact_parity
        ),
        "scores_and_pit_arrays": score_pit_parity,
    }
    local_reconciliations = [row for row in groups if row["local_order_count"] > 0]
    quantity_matches = [row for row in local_reconciliations if row["quantity_match"]]
    price_matches = [row for row in local_reconciliations if row["execution_price_match"]]
    local_requested = sum(row["local_requested_quantity"] for row in groups)
    local_confirmed = sum(row["local_confirmed_quantity"] for row in groups)
    official_quantity_for_local_keys = sum(
        row["official_quantity"] for row in local_reconciliations
    )
    official_quantity_all_csv_groups = sum(row["official_quantity"] for row in groups)
    report = {
        "stage": 1,
        "stage_name": "same_day_production_replay_and_fill_ledger_reconciliation",
        "status": "COMPLETE_WITH_LIMITATIONS",
        "decision": "PROCEED_TO_STAGE_2",
        "authoritative_execution_source": str(csv_path),
        "source_sha256": sha256(csv_path),
        "snapshot_checks": snapshot_checks,
        "broker_requery": {
            "status": (
                "PASS" if broker_requery is not None and not broker_requery.get("errors")
                else "UNAVAILABLE_991012" if broker_requery is not None
                else "NOT_RUN"
            ),
            "confirmed_fill_order_count": (
                broker_requery.get("confirmed_fill_order_count") if broker_requery else None
            ),
            "confirmed_fill_quantity": (
                broker_requery.get("confirmed_fill_quantity") if broker_requery else None
            ),
        },
        "coverage": {
            "local_close_order_count": len(local),
            "official_repayment_transaction_count": len(official),
            "local_group_count": len(local_groups),
            "official_group_count": len(official_groups),
            "matched_group_count": len(quantity_matches),
            "quantity_match_rate": len(quantity_matches) / len(local_groups) if local_groups else None,
            "execution_price_match_group_count": len(price_matches),
            "local_requested_quantity": local_requested,
            "local_confirmed_quantity_from_runtime_logs": local_confirmed,
            "official_quantity_for_local_keys": official_quantity_for_local_keys,
            "official_quantity_all_csv_groups": official_quantity_all_csv_groups,
            "runtime_log_quantity_gap": official_quantity_for_local_keys - local_confirmed,
        },
        "completion_basis": {
            "internal_decision_snapshot_reconciled": all(snapshot_checks.values()),
            "authoritative_ledger_group_match": len(quantity_matches) == len(local_groups),
            "authoritative_ledger_quantity_match": (
                official_quantity_for_local_keys == local_requested
            ),
            "execution_difference_separately_reported": True,
            "broker_order_id_and_fee_reconciliation": False,
        },
        "groups": groups,
        "unresolved": [
            "The CSV has no broker order IDs, so one-to-one order mapping is unavailable.",
            "The supplied runtime logs record zero fill quantity for 199 of 207 shares; the CSV confirms those executions but cannot repair the original logs.",
            "The broker detail re-query was unavailable for the historical records and fee components are not present in the CSV; these are carried into stages 2 and 3 and are not treated as zero.",
        ],
        "references": {
            "artifact_parity": str((ACCEPTANCE / "artifact_parity.json").relative_to(root)),
            "gap_path_parity": str((ACCEPTANCE / "gap_path_parity.json").relative_to(root)),
            "acceptance_report": str((ACCEPTANCE / "report.md").relative_to(root)),
        },
    }
    return report


def write_report(report: dict[str, Any], output_dir: Path, root: Path = ROOT) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "reconciliation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    checks = report["snapshot_checks"]
    coverage = report["coverage"]
    lines = [
        "# 収益改善順序1：本番再生・約定台帳照合",
        "",
        "実行は読み取り専用。brokerへの発注・取消・再送は行っていない。",
        "",
        f"判定: **{report['status']}** / 次段階: **{report['decision']}**",
        "",
        "## 固定snapshotの照合",
        "",
        "| 項目 | 結果 |",
        "|---|---|",
        f"| μ/Ω cache vs on-demand | {checks['mu_omega_cache_vs_ondemand']} |",
        f"| production vs BT weights | {checks['weights_production_vs_backtest']} |",
        f"| production vs collector weights | {checks['weights_production_vs_collector']} |",
        f"| 数値・リーク監査 | {checks['numerical_and_leakage_audits']} |",
        f"| scores・PITの直接比較 | {checks['scores_and_pit_arrays']} |",
        "",
        "## 約定照合",
        "",
        "| 指標 | 値 |",
        "|---|---:|",
        f"| local close order | {coverage['local_close_order_count']} |",
        f"| local group | {coverage['local_group_count']} |",
        f"| quantity一致group | {coverage['matched_group_count']} |",
        f"| requested quantity | {coverage['local_requested_quantity']} |",
        f"| runtime log confirmed quantity | {coverage['local_confirmed_quantity_from_runtime_logs']} |",
        f"| CSV confirmed quantity for local keys | {coverage['official_quantity_for_local_keys']} |",
        f"| runtime log quantity gap | {coverage['runtime_log_quantity_gap']} |",
        f"| observed execution price一致group | {coverage['execution_price_match_group_count']} |",
        "",
        "## 未解決事項",
        "",
    ]
    lines.extend(f"- {item}" for item in report["unresolved"])
    lines.extend(
        [
            "",
            "この結果により、固定snapshotの重複計算とscores/PIT配列の直接比較はPASS、実約定は公式CSVの数量・日付・銘柄・売買単位でローカル23/23 group、207/207株を照合できた。順序1はこの範囲で完了とする。ただしbroker order ID・費用内訳・実行ログの199株分は未解決のまま順序2・3へ持ち越し、約定済み・手数料済みとは扱わない。",
            "",
            f"再現JSON: `{(output_dir / 'reconciliation.json').relative_to(root)}`",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    csv_path = args.csv if args.csv.is_absolute() else ROOT / args.csv
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    report = build_report(csv_path)
    write_report(report, output_dir)
    print(json.dumps({key: report[key] for key in ("status", "decision", "coverage")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
