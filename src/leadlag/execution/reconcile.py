"""Restart reconciliation of durable runs. This module never submits orders.

List unresolved runs without broker access with ``python -m
leadlag.execution.reconcile``. Use ``--run-id`` to refresh broker observations
and reconcile confirmed fills, inventory, wallet and the recovery journal.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from leadlag.broker.base import BrokerClient
from leadlag.config.paths import execution_state_path
from leadlag.core.types import OrderRequest, OrderSide, OrderType
from leadlag.execution.contracts import ExecutionPlan
from leadlag.execution.job_guard import execution_lease
from leadlag.execution.pricing import fetch_fill_prices
from leadlag.execution.state_store import ExecutionRunStatus, ExecutionStateStore


def _fill_field_errors(record: dict[str, Any]) -> list[str]:
    """Validate the accounting fields required for any confirmed quantity."""
    quantity = record.get("fill_quantity")
    try:
        has_fill = int(cast(Any, quantity)) > 0
    except (TypeError, ValueError):
        has_fill = False
    if not has_fill:
        return []
    errors: list[str] = []
    try:
        price = float(cast(Any, record.get("fill_price")))
    except (TypeError, ValueError):
        price = math.nan
    if not math.isfinite(price) or price <= 0.0:
        errors.append(f"unconfirmed fill price: {record.get('order_id')}")

    detail = record.get("fill_detail")
    detail = detail if isinstance(detail, Mapping) else {}
    fee_provided = "sBaiBaiTesuryo" in detail or "fee" in record or "commission" in record
    raw_fee = detail.get("sBaiBaiTesuryo") if "sBaiBaiTesuryo" in detail else record.get(
        "fee", record.get("commission")
    )
    try:
        fee = float(cast(Any, raw_fee))
    except (TypeError, ValueError):
        fee = math.nan
    if not fee_provided or not math.isfinite(fee) or fee < 0.0:
        errors.append(f"unconfirmed fill fee: {record.get('order_id')}")
    return errors


def verify_position_checkpoint(plan: dict[str, Any], records: list[dict], snapshot_path: str | None) -> list[str]:
    """Compare observed inventory with starting inventory and confirmed fills."""
    errors: list[str] = []
    expected = dict(plan["current_positions"])
    for record in records:
        quantity = record.get("fill_quantity")
        if quantity is None:
            errors.append(f"unconfirmed fill quantity: {record.get('order_id')}")
            continue
        quantity = int(quantity)
        if quantity < 0 or quantity > int(record["quantity"]):
            errors.append(f"invalid fill quantity: {record.get('order_id')}")
            continue
        if record.get("status") == "FILLED" and quantity != int(record["quantity"]):
            errors.append(f"FILLED quantity mismatch: {record.get('order_id')}")
        errors.extend(_fill_field_errors(record))
        ticker = str(record["ticker"])
        expected[ticker] = expected.get(ticker, 0) + (quantity if record["side"] == "BUY" else -quantity)
    try:
        if snapshot_path is None:
            raise ValueError("missing position snapshot")
        positions = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))["positions"]
        actual: dict[str, int] = {}
        for position in positions:
            quantity = int(position["quantity"]) * (1 if position["side"] == "BUY" else -1)
            actual[position["ticker"]] = actual.get(position["ticker"], 0) + quantity
        if {k: v for k, v in actual.items() if v} != {k: v for k, v in expected.items() if v}:
            errors.append("position quantity mismatch against starting inventory and confirmed fills")
    except Exception as exc:
        errors.append(f"position checkpoint: {exc}")
    return errors


def collect_readonly_account_snapshot(
    broker: BrokerClient,
    output_dir: Path,
    *,
    provider: str,
) -> dict[str, Any]:
    """Collect account, position, and wallet evidence without order actions.

    This is intentionally separate from run reconciliation: it does not need
    a persisted run and never calls submit/cancel/resend.  Fill evidence for a
    specific order remains available through ``--run-id`` because the broker
    neutral interface can only query fills for known broker IDs.
    """
    observed_at = datetime.now(UTC).isoformat()
    positions = [asdict(position) for position in broker.get_positions()]
    wallet = asdict(broker.get_wallet())
    result = {
        "read_only": True,
        "provider": provider,
        "observed_at": observed_at,
        "positions": positions,
        "position_count": len(positions),
        "wallet": wallet,
        "fill_query": "known broker order IDs only; use --run-id for persisted runs",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    journal = output_dir / f"account_snapshot_{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    temporary = journal.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(journal)
    result["path"] = str(journal)
    return result


def reconcile_run(store: ExecutionStateStore, run_id: str, broker: BrokerClient, output_dir: Path) -> dict[str, Any]:
    """Reconcile only persisted IDs; unidentified submissions remain unresolved."""
    run = store.get_run(run_id)
    if run is None or run.status not in {ExecutionRunStatus.EXECUTING, ExecutionRunStatus.RECONCILIATION_REQUIRED}:
        raise ValueError("Recovery requires an executing or reconciliation_required run")
    metadata = (run.metadata or {}).get("execution_plan")
    if not isinstance(metadata, dict):
        raise ValueError("Run has no saved starting inventory/plan; manual reconciliation is required")
    with store._connect() as conn:
        intents = conn.execute("SELECT * FROM order_intents WHERE run_id = ? ORDER BY ordinal", (run_id,)).fetchall()
        observations = conn.execute(
            "SELECT * FROM order_observations WHERE run_id = ? ORDER BY observation_id", (run_id,),
        ).fetchall()
    latest = {row["broker_order_id"]: dict(row) for row in observations if row["broker_order_id"]}
    errors: list[str] = []
    records: list[dict[str, Any]] = []
    allocated: dict[str, int] = {}
    for order_id, observation in latest.items():
        record = {
            "order_id": order_id, "ticker": observation["ticker"], "side": observation["side"],
            "quantity": observation["requested_quantity"], "status": observation["status"],
            "fill_quantity": observation["filled_quantity"], "fill_price": observation["fill_price"],
            "fee": observation["fee"],
            "eigyou_day": run.trade_date.replace("-", ""),
        }
        try:
            record["status"] = str(broker.get_order_status(order_id))
        except Exception as exc:
            errors.append(f"order_status:{order_id}: {exc}")
        records.append(record)
        intent_id = observation["intent_id"]
        if intent_id is None:
            errors.append(f"unmapped broker order: {order_id}")
        else:
            allocated[intent_id] = allocated.get(intent_id, 0) + int(observation["requested_quantity"])
    for intent in intents:
        if allocated.get(intent["intent_id"], 0) != int(intent["quantity"]):
            errors.append(f"unidentified or unsent quantity for intent {intent['intent_id']}; do not resend")
    try:
        fetch_fill_prices(broker, records)
    except Exception as exc:
        errors.append(f"fills: {exc}")

    expected = dict(metadata["current_positions"])
    for record in records:
        status = record["status"]
        if status not in {"FILLED", "CANCELLED", "FAILED"}:
            errors.append(f"non-terminal order: {record['order_id']} ({status})")
        quantity = record.get("fill_quantity")
        if quantity is None or int(quantity) < 0 or int(quantity) > record["quantity"]:
            errors.append(f"unconfirmed fill quantity: {record['order_id']}")
            continue
        if status == "FILLED" and int(quantity) != record["quantity"]:
            errors.append(f"FILLED quantity mismatch: {record['order_id']}")
        errors.extend(_fill_field_errors(record))
        sign = 1 if record["side"] == "BUY" else -1
        expected[record["ticker"]] = expected.get(record["ticker"], 0) + sign * int(quantity)

    positions = []
    wallet = None
    try:
        positions = [asdict(position) for position in broker.get_positions()]
        actual: dict[str, int] = {}
        for position in positions:
            sign = 1 if position["side"] == "BUY" else -1
            actual[position["ticker"]] = actual.get(position["ticker"], 0) + sign * int(position["quantity"])
        if {key: value for key, value in actual.items() if value} != {key: value for key, value in expected.items() if value}:
            errors.append("position quantity mismatch against starting inventory and confirmed fills")
    except Exception as exc:
        errors.append(f"positions: {exc}")
    try:
        wallet = asdict(broker.get_wallet())
    except Exception as exc:
        errors.append(f"wallet: {exc}")

    def order(intent: Any) -> OrderRequest:
        payload = json.loads(intent["payload_json"])
        payload["side"] = OrderSide(payload["side"])
        payload["order_type"] = OrderType(payload["order_type"])
        return OrderRequest(**payload)

    plan = ExecutionPlan(
        decision_id=run.decision_id or "", trade_date=run.trade_date,
        close_orders=tuple(order(intent) for intent in intents if intent["purpose"] == "close"),
        new_orders=tuple(order(intent) for intent in intents if intent["purpose"] == "new"),
    )
    store.record_result_set(run_id, plan, records)
    result = {"run_id": run_id, "complete": not errors, "errors": errors, "orders": records,
              "positions": positions, "expected_positions": expected, "wallet": wallet}
    output_dir.mkdir(parents=True, exist_ok=True)
    journal = output_dir / f"reconciliation_{run_id}.json"
    temporary = journal.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(journal)
    store.record_reconciliation(run_id, outcome="incomplete" if errors else "complete", errors=errors,
                               references={"recovery_journal": str(journal)})
    if errors:
        store.mark_reconciliation_required(run_id, error="; ".join(errors))
    else:
        store.mark_completed(run_id)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-db", type=Path, default=execution_state_path())
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--run-id", help="Reconcile this run using read-only broker queries")
    selection.add_argument("--pending", action="store_true", help="Reconcile pending production runs for the configured broker")
    selection.add_argument(
        "--account-snapshot",
        action="store_true",
        help="Collect read-only wallet and position evidence without a run",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("var/results/reconciliation"))
    args = parser.parse_args(argv)
    store = ExecutionStateStore(args.state_db)
    if args.run_id is None and not args.pending and not args.account_snapshot:
        print(json.dumps([asdict(run) for run in store.list_recovery_candidates()], indent=2))
        return 0
    from leadlag.execution.broker_ops import build_api_client
    from leadlag.execution.config import load_config_from_yaml

    config = load_config_from_yaml(strict=True)
    account_key = f"{config.broker_provider}:default"
    if args.account_snapshot:
        from leadlag.execution.broker_ops import build_api_client

        broker = build_api_client(None, None, False)
        try:
            result = collect_readonly_account_snapshot(
                broker,
                args.output_dir,
                provider=config.broker_provider,
            )
        finally:
            broker.close()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.pending:
        runs = [run for run in store.list_recovery_candidates()
                if run.account_key == account_key and run.strategy_key == "production_v2"]
    else:
        run = store.get_run(args.run_id)
        if run is None or run.account_key != account_key:
            raise ValueError("Run account does not match the configured broker")
        runs = [run]
    if not runs:
        print(json.dumps({"runs": [], "complete": True}))
        return 0
    results = []
    with execution_lease(store, metadata={"job_type": "reconciliation", "run_id": args.run_id}):
        broker = build_api_client(None, None, False)
        try:
            for run in runs:
                try:
                    result = reconcile_run(store, run.run_id, broker, args.output_dir)
                except Exception as exc:
                    # A failed recovery remains unresolved. Continue collecting
                    # evidence for other candidates; never turn failure into a
                    # completed checkpoint or a new submission.
                    result = {"run_id": run.run_id, "complete": False, "errors": [str(exc)]}
                results.append({"run_id": run.run_id, "complete": result["complete"], "errors": result["errors"]})
        finally:
            broker.close()
    complete = all(result["complete"] for result in results)
    print(json.dumps({"runs": results, "complete": complete}, indent=2))
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
