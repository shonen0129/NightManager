"""leadlag/execution/close.py — position closing logic.

Provides ``close_all_positions()`` and ``run_close_positions_mode()`` which
orchestrate the end-of-day 引け時反対売買 flow via the BrokerClient ABC.

These functions are broker-neutral: they work with any BrokerClient
implementation (KabuBrokerClient, DryRunBrokerClient, future SBI, etc.).
"""

from __future__ import annotations

import json
import logging
import math
import os
import time as time_module
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

from leadlag.broker.base import BrokerClient
from leadlag.config.paths import execution_state_path
from leadlag.core.types import OrderRequest, OrderResult, OrderSide, OrderStatus, OrderType
from leadlag.data.tickers import lot_size_for
from leadlag.execution.broker_ops import (
    SPLIT_DELAY_SECONDS,
    build_api_client,
    split_large_orders,
)
from leadlag.execution.config import load_config_from_yaml
from leadlag.execution.contracts import ExecutionPlan, build_decision_id, report_from_records
from leadlag.execution.job_guard import execution_lease
from leadlag.execution.order_lifecycle import poll_order_statuses
from leadlag.execution.output_ops import (
    build_output_dir,
    save_daily_journal,
    save_position_snapshot,
    save_wallet_snapshot,
)
from leadlag.execution.pricing import fetch_fill_prices
from leadlag.execution.runtime_manifest import update_execution_manifest
from leadlag.execution.state_store import ExecutionRun, ExecutionStateStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CloseOrderPlan:
    """Validated, run-owned close intent before any broker side effect."""

    execution_plan: ExecutionPlan
    order_metadata: tuple[dict[str, Any], ...]
    order_requests: tuple[OrderRequest, ...]
    held_overnight: tuple[dict[str, Any], ...]
    metadata_by_ticker: Mapping[str, dict[str, Any]]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "metadata_by_ticker", MappingProxyType(dict(self.metadata_by_ticker))
        )


def _build_close_order_plan(
    positions: Sequence[Any],
    *,
    overnight_alpha_long: float,
    overnight_alpha_short: float,
) -> CloseOrderPlan:
    """Calculate close quantities and durable intent without contacting a broker."""
    close_order_meta: list[dict[str, Any]] = []
    close_order_requests: list[OrderRequest] = []
    held_overnight_meta: list[dict[str, Any]] = []
    for position in positions:
        if position.quantity <= 0:
            continue

        alpha = overnight_alpha_long if position.side == "BUY" else overnight_alpha_short
        close_fraction = 1.0 - alpha
        lot_size = lot_size_for(position.ticker)
        hold_qty_target = position.quantity * alpha
        hold_qty = min(
            position.quantity,
            math.floor(hold_qty_target / lot_size + 0.5) * lot_size,
        )
        close_qty = position.quantity - hold_qty
        close_qty = math.floor(close_qty / lot_size) * lot_size
        close_qty = max(0, min(close_qty, position.quantity))
        hold_qty = position.quantity - close_qty

        if hold_qty > 0:
            held_overnight_meta.append(
                {
                    "ticker": position.ticker,
                    "side": position.side,
                    "hold_quantity": hold_qty,
                    "alpha": alpha,
                }
            )
            logger.info(
                "  Overnight hold: %s %s x%d (alpha=%.2f, held=%.0f%%)",
                position.ticker,
                position.side,
                hold_qty,
                alpha,
                alpha * 100,
            )

        if close_qty <= 0:
            logger.info(
                "  Skipping close for %s %s: close_qty=0 (alpha=%.2f)",
                position.ticker,
                position.side,
                alpha,
            )
            continue

        close_side_str = "SELL" if position.side == "BUY" else "BUY"
        close_side = OrderSide.SELL if position.side == "BUY" else OrderSide.BUY
        close_order_meta.append(
            {
                "ticker": position.ticker,
                "exchange": position.exchange or 27,
                "side": close_side_str,
                "quantity": close_qty,
                "margin_trade_type": position.margin_trade_type,
                "account_type": position.account_type,
                "order_type": "CLO",
                "original_side": position.side,
                "original_price": position.price,
            }
        )
        close_order_requests.append(
            OrderRequest(
                ticker=position.ticker,
                side=close_side,
                quantity=close_qty,
                order_type=OrderType.CLOSE,
                margin_trade_type=position.margin_trade_type,
                account_type=position.account_type,
            )
        )
        logger.info(
            "  Position to close: %s %s x%d/%d → %s (引成（後場）, close=%.0f%%)",
            position.ticker,
            position.side,
            close_qty,
            position.quantity,
            close_side_str,
            close_fraction * 100,
        )

    metadata_by_ticker = {item["ticker"]: item for item in close_order_meta}
    starting_positions: dict[str, int] = {}
    for position in positions:
        signed_quantity = int(position.quantity) * (1 if position.side == "BUY" else -1)
        starting_positions[position.ticker] = (
            starting_positions.get(position.ticker, 0) + signed_quantity
        )
    execution_plan = ExecutionPlan(
        decision_id=build_decision_id(
            datetime.now().date().isoformat(), close_order_requests, ()
        ),
        trade_date=datetime.now().date().isoformat(),
        close_orders=tuple(close_order_requests),
        new_orders=(),
        current_positions=tuple(sorted(starting_positions.items())),
    )
    return CloseOrderPlan(
        execution_plan=execution_plan,
        order_metadata=tuple(close_order_meta),
        order_requests=tuple(close_order_requests),
        held_overnight=tuple(held_overnight_meta),
        metadata_by_ticker=metadata_by_ticker,
    )


MISSING_RESPONSE_STATUS = "MISSING_RESPONSE"


def _request_signature(request: OrderRequest) -> tuple[str, str, int]:
    return request.ticker, request.side.value, int(request.quantity)


def _record_signature(record: Mapping[str, Any]) -> tuple[str, str, int]:
    return (
        str(record.get("ticker", "")),
        str(record.get("side", "")),
        int(record.get("quantity", 0) or 0),
    )


def _result_dict(
    result: OrderResult,
    order_plan: CloseOrderPlan,
    *,
    delayed: bool = False,
) -> dict[str, Any]:
    meta = order_plan.metadata_by_ticker.get(result.ticker, {})
    payload = {
        "order_id": result.order_id,
        "status": result.status.value,
        "ticker": result.ticker,
        "side": result.side.value,
        "quantity": result.quantity,
        "message": result.message,
        "eigyou_day": result.eigyou_day,
        "original_side": meta.get("original_side"),
        "original_price": meta.get("original_price"),
    }
    if delayed:
        payload["delayed"] = True
    return payload


def _normalize_batch_results(
    requests: Sequence[OrderRequest],
    results: Sequence[OrderResult],
    order_plan: CloseOrderPlan,
    *,
    delayed: bool = False,
) -> list[dict[str, Any]]:
    """Keep every broker response and make missing responses explicit."""
    records = [_result_dict(result, order_plan, delayed=delayed) for result in results]
    expected = Counter(_request_signature(request) for request in requests)
    observed = Counter(_record_signature(record) for record in records)
    missing = expected - observed
    for (ticker, side, quantity), count in missing.items():
        meta = order_plan.metadata_by_ticker.get(ticker, {})
        for _ in range(count):
            record = {
                "order_id": "",
                "status": MISSING_RESPONSE_STATUS,
                "ticker": ticker,
                "side": side,
                "quantity": quantity,
                "message": "Broker returned no response for submitted close request",
                "original_side": meta.get("original_side"),
                "original_price": meta.get("original_price"),
            }
            if delayed:
                record["delayed"] = True
            records.append(record)
    return records


def _batch_reconciliation_errors(
    requests: Sequence[OrderRequest],
    records: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Validate one concrete broker batch before any delayed submission."""
    errors: list[str] = []
    expected = Counter(_request_signature(request) for request in requests)
    observed = Counter(_record_signature(record) for record in records)
    if observed != expected:
        errors.append(
            "batch response quantities do not match submitted requests "
            f"(expected={dict(expected)}, observed={dict(observed)})"
        )

    order_ids = [str(record.get("order_id") or "") for record in records]
    real_ids = [order_id for order_id in order_ids if order_id]
    duplicate_ids = sorted(
        order_id for order_id, count in Counter(real_ids).items() if count > 1
    )
    if duplicate_ids:
        errors.append(f"duplicate broker order response(s): {', '.join(duplicate_ids)}")

    for record in records:
        status = str(record.get("status") or "")
        if status != OrderStatus.FILLED.value:
            errors.append(
                "first batch not fully filled: "
                f"{record.get('ticker')} {record.get('side')} x{record.get('quantity')} "
                f"status={status or 'UNKNOWN'}"
            )
        if status == OrderStatus.FILLED.value and not record.get("order_id"):
            errors.append(
                "filled close response is missing broker order id: "
                f"{record.get('ticker')} {record.get('side')} x{record.get('quantity')}"
            )
    return errors


def _close_plan_reconciliation_errors(
    execution_plan: ExecutionPlan,
    records: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Reconcile child broker responses back to the persisted close intent quantities."""
    errors: list[str] = []
    planned: Counter[tuple[str, str]] = Counter()
    for request in execution_plan.close_orders:
        planned[(request.ticker, request.side.value)] += int(request.quantity)

    represented: Counter[tuple[str, str]] = Counter()
    real_order_ids: list[str] = []
    for record in records:
        ticker = str(record.get("ticker") or "")
        side = str(record.get("side") or "")
        try:
            quantity = int(record.get("quantity", 0) or 0)
        except (TypeError, ValueError):
            errors.append(f"invalid response quantity for {ticker} {side}")
            continue
        if not ticker or not side or quantity <= 0:
            errors.append(f"invalid close response identity: {ticker} {side} x{quantity}")
            continue
        represented[(ticker, side)] += quantity
        order_id = str(record.get("order_id") or "")
        if order_id:
            real_order_ids.append(order_id)

        filled_raw = record.get("fill_quantity", record.get("filled_quantity"))
        if filled_raw is not None:
            try:
                filled = int(filled_raw)
            except (TypeError, ValueError):
                errors.append(f"invalid fill quantity: {order_id or ticker}")
            else:
                if filled < 0 or filled > quantity:
                    errors.append(f"invalid fill quantity: {order_id or ticker}")
                if (
                    str(record.get("status") or "") == OrderStatus.FILLED.value
                    and filled != quantity
                ):
                    errors.append(f"FILLED quantity mismatch: {order_id or ticker}")

    if represented != planned:
        errors.append(
            "close response quantities do not match persisted plan "
            f"(planned={dict(planned)}, represented={dict(represented)})"
        )

    duplicate_ids = sorted(
        order_id for order_id, count in Counter(real_order_ids).items() if count > 1
    )
    if duplicate_ids:
        errors.append(f"duplicate broker order response(s): {', '.join(duplicate_ids)}")
    return errors


def _submit_close_order_batches(
    api_client: BrokerClient,
    order_plan: CloseOrderPlan,
    *,
    dry_run: bool,
    close_position_order: int,
    persist_submitted_observations: Callable[[Sequence[OrderResult]], None],
) -> list[dict[str, Any]]:
    """Submit close batches and fail closed before any delayed submission."""
    close_results: list[dict[str, Any]] = []
    if dry_run:
        logger.info("[DRY RUN MODE] Simulating position close (no actual orders sent)...")
        for meta in order_plan.order_metadata:
            clean = meta["ticker"].replace(".T", "")
            simulated = {
                "order_id": f"SIM-CLOSE-{datetime.now().strftime('%Y%m%d%H%M%S')}-{clean}",
                "status": "SIMULATED",
                "ticker": meta["ticker"],
                "side": meta["side"],
                "quantity": meta["quantity"],
                "original_side": meta["original_side"],
                "original_price": meta["original_price"],
            }
            logger.info(
                "  [SIMULATED CLOSE] %s: %d shares (%s → %s)",
                meta["ticker"],
                meta["quantity"],
                meta["original_side"],
                meta["side"],
            )
            close_results.append(simulated)
        return close_results

    immediate_close, delayed_close = split_large_orders(list(order_plan.order_requests))
    immediate_requests = list(immediate_close)
    logger.info("[LIVE MODE] Submitting %d position close orders...", len(immediate_requests))
    first_results = api_client.submit_orders_batch(
        immediate_requests,
        delay_ms=250,
        is_close=True,
        close_position_order=close_position_order,
    )
    persist_submitted_observations(first_results)
    first_records = _normalize_batch_results(
        immediate_requests,
        first_results,
        order_plan,
    )
    for record in first_records:
        logger.info(
            "  [CLOSE RESPONSE] %s: %d shares (status=%s, order_id=%s)",
            record["ticker"],
            record["quantity"],
            record["status"],
            record["order_id"],
        )
    _wait_for_close_fills_sync(api_client, first_records)
    close_results.extend(first_records)

    first_batch_errors = _batch_reconciliation_errors(immediate_requests, first_records)
    if delayed_close:
        if first_batch_errors:
            reason = "; ".join(first_batch_errors)
            logger.warning(
                "[DELAYED CLOSE] Skipping %d delayed close order(s): %s",
                len(delayed_close),
                reason,
            )
            for request in delayed_close:
                meta = order_plan.metadata_by_ticker.get(request.ticker, {})
                close_results.append(
                    {
                        "order_id": "",
                        "status": "SKIPPED",
                        "ticker": request.ticker,
                        "side": request.side.value,
                        "quantity": request.quantity,
                        "message": f"Skipped because first batch was not reconciled: {reason}",
                        "delayed": True,
                        "original_side": meta.get("original_side"),
                        "original_price": meta.get("original_price"),
                    }
                )
        else:
            delayed_requests = list(delayed_close)
            logger.info(
                "[DELAYED CLOSE] Waiting %d seconds before submitting %d delayed close order(s)...",
                SPLIT_DELAY_SECONDS,
                len(delayed_requests),
            )
            time_module.sleep(SPLIT_DELAY_SECONDS)
            logger.info(
                "[DELAYED CLOSE] Submitting %d delayed close orders...",
                len(delayed_requests),
            )
            delayed_results = api_client.submit_orders_batch(
                delayed_requests,
                delay_ms=250,
                is_close=True,
                close_position_order=close_position_order,
            )
            persist_submitted_observations(delayed_results)
            delayed_records = _normalize_batch_results(
                delayed_requests,
                delayed_results,
                order_plan,
                delayed=True,
            )
            for record in delayed_records:
                logger.info(
                    "  [DELAYED CLOSE RESPONSE] %s: %d shares (status=%s, order_id=%s)",
                    record["ticker"],
                    record["quantity"],
                    record["status"],
                    record["order_id"],
                )
            _wait_for_close_fills_sync(api_client, delayed_records)
            close_results.extend(delayed_records)
    return close_results


def _reconcile_close_run(
    api_client: BrokerClient,
    summary: dict[str, Any],
    execution_plan: ExecutionPlan,
    output_dir: str | Path,
    *,
    dry_run: bool,
    state_store: ExecutionStateStore | None,
    execution_run: ExecutionRun | None,
) -> dict[str, Any]:
    """Collect fills, persist reconciliation state, and publish close results."""
    close_results = summary["close_results"]
    if not dry_run and close_results:
        from leadlag.broker.dry_run import DryRunBrokerClient

        if not isinstance(api_client, DryRunBrokerClient):
            try:
                fetch_fill_prices(api_client, close_results, wait_seconds=5.0)
            except Exception as exc:  # noqa: BLE001
                summary.setdefault("reconciliation_errors", []).append(f"fill_prices: {exc}")
                logger.exception("Failed to fetch close fill prices")

    plan_errors = _close_plan_reconciliation_errors(execution_plan, close_results)
    if plan_errors:
        existing_errors = summary.setdefault("reconciliation_errors", [])
        existing_errors.extend(error for error in plan_errors if error not in existing_errors)
    summary["plan_reconciliation_errors"] = plan_errors
    summary["planned_orders_count"] = execution_plan.expected_order_count
    summary["planned_close_quantity"] = sum(
        int(request.quantity) for request in execution_plan.close_orders
    )
    summary["represented_close_quantity"] = sum(
        int(result.get("quantity", 0) or 0) for result in close_results
    )

    terminal_successes = {OrderStatus.FILLED.value, OrderStatus.SIMULATED.value}
    success_count = sum(
        1 for result in close_results if result.get("status") in terminal_successes
    )
    partial_count = sum(
        1
        for result in close_results
        if result.get("status") == OrderStatus.PARTIALLY_FILLED.value
    )
    pending_count = sum(
        1
        for result in close_results
        if result.get("status")
        in {OrderStatus.SUBMITTED.value, OrderStatus.PARTIALLY_FILLED.value}
    )
    failed_count = sum(
        1
        for result in close_results
        if result.get("status")
        in {
            OrderStatus.FAILED.value,
            OrderStatus.CANCELLED.value,
            "SKIPPED",
            MISSING_RESPONSE_STATUS,
        }
    )
    summary["filled_orders_count"] = success_count
    summary["partial_orders_count"] = partial_count
    summary["pending_orders_count"] = pending_count
    summary["failed_orders_count"] = failed_count
    summary["close_incomplete"] = any(
        result.get("status") not in terminal_successes for result in close_results
    ) or bool(summary.get("reconciliation_errors"))
    summary["execution_report"] = report_from_records(
        close_results,
        expected_orders=execution_plan.expected_order_count,
        reconciliation_errors=summary.get("reconciliation_errors", []),
    ).to_dict()

    if state_store is not None and execution_run is not None:
        try:
            state_store.record_result_set(execution_run.run_id, execution_plan, close_results)
            state_store.record_reconciliation(
                execution_run.run_id,
                outcome="incomplete" if summary["close_incomplete"] else "pending",
                errors=summary.get("reconciliation_errors", []),
                references={"close_execution_log": str(Path(output_dir) / "close_execution_log.json")},
            )
            if summary["close_incomplete"]:
                state_store.mark_reconciliation_required(
                    execution_run.run_id,
                    error=(
                        f"filled_responses={success_count}; "
                        f"planned_intents={execution_plan.expected_order_count}; "
                        f"pending={pending_count}; failed={failed_count}"
                    ),
                )
        except Exception as exc:  # noqa: BLE001
            logger.exception("[CLOSE] Durable state update failed")
            summary.setdefault("reconciliation_errors", []).append(f"state_store: {exc}")
            try:
                state_store.mark_reconciliation_required(execution_run.run_id, error=str(exc))
            except Exception:
                logger.exception("[CLOSE] Could not mark run reconciliation_required")

    if summary.get("reconciliation_errors"):
        summary["close_incomplete"] = True
        summary["execution_report"] = report_from_records(
            close_results,
            expected_orders=execution_plan.expected_order_count,
            reconciliation_errors=summary["reconciliation_errors"],
        ).to_dict()

    log_path = os.path.join(output_dir, "close_execution_log.json")
    with open(log_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    logger.info("Close execution log saved: %s", log_path)

    total_close_orders = len(close_results)
    if summary["close_incomplete"]:
        logger.error(
            "Position close incomplete: filled=%d/%d, pending=%d, failed=%d",
            success_count,
            total_close_orders,
            pending_count,
            failed_count,
        )
    else:
        logger.info(
            "Position close completed: %d/%d orders filled", success_count, total_close_orders
        )
    return summary


def _close_result_record(result: OrderResult) -> dict[str, Any]:
    """Convert a broker close response to the durable observation shape."""
    return {
        "order_id": result.order_id,
        "status": result.status.value,
        "ticker": result.ticker,
        "side": result.side.value,
        "quantity": result.quantity,
        "message": result.message,
        # Tachibana's detail endpoint requires the business day returned by
        # the submit response. Preserve it for later read-only reconciliation.
        "eigyou_day": result.eigyou_day,
    }


def _wait_for_close_fills_sync(
    api_client: BrokerClient,
    close_results: list[dict],
    timeout_seconds: float = 60.0,
    poll_interval: float = 1.0,
) -> None:
    """Poll close order statuses until they are no longer SUBMITTED or timeout.

    Updates ``close_results`` entries with the latest status.  This prevents
    moving on before a close order has been confirmed at the exchange.
    """
    poll_order_statuses(
        api_client,
        close_results,
        order_id_getter=lambda result: str(result.get("order_id", "")),
        status_getter=lambda result: result.get("status", ""),
        status_setter=lambda result, status: result.__setitem__("status", status.value),
        message_setter=lambda result, message: result.__setitem__("message", message),
        timeout_seconds=timeout_seconds,
        poll_interval=poll_interval,
        label="CLOSE FILL",
    )


def close_all_positions(
    api_client: BrokerClient,
    output_dir: str | Path,
    dry_run: bool = False,
    margin_trade_type: int = 3,
    account_type: int = 4,
    close_position_order: int = 0,
    overnight_alpha_long: float = 0.0,
    overnight_alpha_short: float = 0.0,
    state_store: ExecutionStateStore | None = None,
    account_key: str = "default",
    strategy_key: str = "production_v2",
) -> dict:
    """Close open margin positions at 引け, respecting overnight holding ratios.

    For each position, only the ``(1 - alpha)`` fraction is closed at 引け.
    The remaining ``alpha`` fraction is held overnight and rebalanced the next morning.

    - Long positions: close ``(1 - overnight_alpha_long)`` fraction
    - Short positions: close ``(1 - overnight_alpha_short)`` fraction

    Args:
        api_client: BrokerClient instance
        output_dir: Directory to save close_execution_log.json
        dry_run: If True, simulate without actual submission
        margin_trade_type: 1=制度信用, 2=一般信用(長期), 3=一般信用(デイトレ)
        account_type: 2=一般口座, 4=特定口座, 12=法人口座
        close_position_order: Close priority (0-7) for credit repayment
        overnight_alpha_long: Fraction of long positions to hold overnight (0=close all, 1=hold all)
        overnight_alpha_short: Fraction of short positions to hold overnight (0=close all, 1=hold all)

    Returns:
        Dict with close order summary
    """
    logger.info(
        "=== Position Close (引け時反対売買) — alpha_long=%.2f, alpha_short=%.2f ===",
        overnight_alpha_long, overnight_alpha_short,
    )

    try:
        positions = api_client.get_positions()
    except Exception as e:
        error_summary = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "dry_run": dry_run,
            "positions_found": None,
            "close_orders": [],
            "close_results": [],
            "filled_orders_count": 0,
            "partial_orders_count": 0,
            "pending_orders_count": 0,
            "failed_orders_count": 0,
            "close_incomplete": True,
            "execution_error": str(e),
        }
        try:
            with open(Path(output_dir) / "close_execution_log.json", "w", encoding="utf-8") as handle:
                json.dump(error_summary, handle, ensure_ascii=False, indent=2)
        except Exception:
            logger.exception("Failed to persist close query error summary")
        raise RuntimeError("Failed to fetch open positions before auto-close") from e

    logger.info("Found %d open position(s)", len(positions))

    # Keep only margin positions (those with an execution_id)
    margin_positions = [pos for pos in positions if pos.execution_id]
    skipped = len(positions) - len(margin_positions)
    if skipped > 0:
        logger.info("Skipping %d cash position(s) without ExecutionID", skipped)
    positions = margin_positions

    if not positions:
        logger.info("No open positions to close")
        empty_summary: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "dry_run": dry_run,
            "positions_found": 0,
            "close_orders": [],
            "close_results": [],
            "filled_orders_count": 0,
            "partial_orders_count": 0,
            "pending_orders_count": 0,
            "failed_orders_count": 0,
            "close_incomplete": False,
        }
        with open(Path(output_dir) / "close_execution_log.json", "w", encoding="utf-8") as handle:
            json.dump(empty_summary, handle, ensure_ascii=False, indent=2)
        return empty_summary

    # This calculation phase has no broker or persistence side effects.
    order_plan = _build_close_order_plan(
        positions,
        overnight_alpha_long=overnight_alpha_long,
        overnight_alpha_short=overnight_alpha_short,
    )
    close_plan = order_plan.execution_plan
    close_order_requests = order_plan.order_requests
    execution_run: ExecutionRun | None = None
    if state_store is not None and close_plan.expected_order_count:
        execution_run = state_store.prepare_run(
            account_key=account_key,
            strategy_key=strategy_key,
            trade_date=close_plan.trade_date,
            job_type="close",
            decision_id=close_plan.decision_id,
            metadata={"expected_orders": close_plan.expected_order_count},
        )
        # This commit is intentionally before the first close submission.
        state_store.record_plan(execution_run.run_id, close_plan)
        state_store.mark_submission_started(execution_run.run_id)

    summary: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dry_run": dry_run,
        "positions_found": len(positions),
        "close_orders_count": len(close_order_requests),
        "planned_orders_count": close_plan.expected_order_count,
        "planned_close_quantity": sum(int(request.quantity) for request in close_order_requests),
        "overnight_alpha_long": overnight_alpha_long,
        "overnight_alpha_short": overnight_alpha_short,
        "held_overnight": list(order_plan.held_overnight),
        "close_results": [],
    }
    if execution_run is not None:
        summary["run_id"] = execution_run.run_id

    observed_records: list[dict[str, Any]] = []

    def persist_submitted_observations(results: Sequence[OrderResult]) -> None:
        """Commit close responses before waiting or submitting a delayed batch."""
        if state_store is None or execution_run is None:
            return
        observed_records.extend(_close_result_record(result) for result in results)
        # Propagate a durable-store failure so no subsequent close batch is
        # submitted. The executing run remains a recovery candidate.
        state_store.record_result_set(execution_run.run_id, close_plan, observed_records)

    summary["close_results"] = _submit_close_order_batches(
        api_client,
        order_plan,
        dry_run=dry_run,
        close_position_order=close_position_order,
        persist_submitted_observations=persist_submitted_observations,
    )

    return _reconcile_close_run(
        api_client,
        summary,
        close_plan,
        output_dir,
        dry_run=dry_run,
        state_store=state_store,
        execution_run=execution_run,
    )


def run_close_positions_mode(
    output_root: str,
    run_tag: str | None,
    api_url: str | None,
    api_token: str | None,
    api_dry_run: bool,
    close_position_order: int,
) -> dict[str, Any]:
    """Entry point for ``--mode close-positions``.

    Builds the broker client, executes position close, and cleans up.
    """
    logger.info("=== CLOSE-POSITIONS MODE ===")
    output_dir = build_output_dir(output_root, run_tag, run_name="production_close_positions")

    api_client: BrokerClient | None = None
    lease_stack = ExitStack()
    try:
        api_client = build_api_client(api_url, api_token, api_dry_run)
        config = load_config_from_yaml()
        state_store = ExecutionStateStore(execution_state_path())
        account_key = f"{config.broker_provider}:default" if not api_dry_run else f"simulation:{output_dir}"
        if config.broker_provider == "tachibana":
            margin_trade_type = config.tachibana.margin_trade_type
            account_type = config.tachibana.account_type
        else:
            margin_trade_type = config.kabu.margin_trade_type
            account_type = config.kabu.account_type
        lease_context = execution_lease(
            state_store, metadata={"trade_date": datetime.now().date().isoformat(), "job_type": "close"},
        )
        lease_stack.enter_context(lease_context)
        close_summary = close_all_positions(
            api_client=api_client,
            output_dir=output_dir,
            dry_run=api_dry_run,
            margin_trade_type=margin_trade_type,
            account_type=account_type,
            close_position_order=close_position_order,
            overnight_alpha_long=config.strategy.overnight_alpha_long,
            overnight_alpha_short=config.strategy.overnight_alpha_short,
            state_store=state_store,
            account_key=account_key,
            strategy_key="production_v2",
        )
        if close_summary.get("close_incomplete"):
            logger.error(
                "Close-positions incomplete: filled=%d/%d, pending=%d, failed=%d",
                close_summary.get("filled_orders_count", 0),
                len(close_summary.get("close_results", [])),
                close_summary.get("pending_orders_count", 0),
                close_summary.get("failed_orders_count", 0),
            )
        else:
            logger.info(
                "Close-positions completed. Orders filled: %d",
                close_summary.get("filled_orders_count", 0),
            )

        # --- Trade journal: collect post-close data ---
        # Reconciliation failures must not erase the close summary or turn an
        # incomplete close into a false success.  Try each artifact
        # independently, persist the errors beside the execution log, and then
        # propagate a processing error after the broker is closed in ``finally``.
        close_log_path = os.path.join(output_dir, "close_execution_log.json")
        reconciliation_errors: list[str] = list(close_summary.get("reconciliation_errors", []))
        try:
            pos_snapshot_path = save_position_snapshot(
                api_client, output_dir, label="close", raise_on_error=True
            )
        except Exception as exc:  # noqa: BLE001
            pos_snapshot_path = None
            reconciliation_errors.append(f"positions: {exc}")
            logger.exception("[JOURNAL] Close position reconciliation failed")
        try:
            wallet_snapshot_path = save_wallet_snapshot(
                api_client, output_dir, label="close", raise_on_error=True
            )
        except Exception as exc:  # noqa: BLE001
            wallet_snapshot_path = None
            reconciliation_errors.append(f"wallet: {exc}")
            logger.exception("[JOURNAL] Close wallet reconciliation failed")
        try:
            save_daily_journal(
                output_dir=output_dir,
                close_execution_log_path=close_log_path,
                position_snapshot_path=pos_snapshot_path,
                wallet_snapshot_path=wallet_snapshot_path,
            )
        except Exception as exc:  # noqa: BLE001
            reconciliation_errors.append(f"daily_journal: {exc}")
            logger.exception("[JOURNAL] Close daily journal save failed")

        if close_summary.get("run_id"):
            try:
                from leadlag.execution.reconcile import verify_position_checkpoint

                if not api_dry_run:
                    run = state_store.get_run(str(close_summary["run_id"]))
                    if run is None or not (run.metadata or {}).get("execution_plan"):
                        raise ValueError("Missing saved execution plan")
                    reconciliation_errors.extend(verify_position_checkpoint(
                        (run.metadata or {})["execution_plan"], close_summary.get("close_results", []), pos_snapshot_path,
                    ))
                incomplete = bool(reconciliation_errors) or bool(close_summary.get("close_incomplete"))
                state_store.record_reconciliation(
                    str(close_summary["run_id"]),
                    outcome="incomplete" if incomplete else "complete",
                    errors=reconciliation_errors,
                    references={"close_execution_log": close_log_path,
                                "position_snapshot": str(pos_snapshot_path) if pos_snapshot_path else None,
                                "wallet_snapshot": str(wallet_snapshot_path) if wallet_snapshot_path else None},
                )
                if incomplete:
                    state_store.mark_reconciliation_required(str(close_summary["run_id"]), error="; ".join(reconciliation_errors))
                else:
                    state_store.mark_completed(str(close_summary["run_id"]))
            except Exception as exc:
                reconciliation_errors.append(f"state_store: {exc}")
                logger.exception("[JOURNAL] Durable close checkpoint failed")

        # Make errors discovered during the durable checkpoint visible to the
        # manifest before it is published.  The detailed execution report is
        # rewritten below for the close log, but the manifest must never say
        # "recorded" while this list is non-empty.
        if reconciliation_errors:
            close_summary["close_incomplete"] = True
            close_summary["reconciliation_errors"] = list(reconciliation_errors)

        try:
            update_execution_manifest(
                output_dir,
                phase="close",
                summary=close_summary,
                position_snapshot_path=pos_snapshot_path,
                wallet_snapshot_path=wallet_snapshot_path,
            )
        except Exception as exc:  # noqa: BLE001
            reconciliation_errors.append(f"runtime_manifest: {exc}")
            logger.exception("[JOURNAL] Close runtime manifest update failed")

        if reconciliation_errors:
            close_summary["close_incomplete"] = True
            close_summary["reconciliation_errors"] = reconciliation_errors
            close_summary["execution_report"] = report_from_records(
                close_summary.get("close_results", []),
                expected_orders=int(
                    close_summary.get(
                        "planned_orders_count",
                        len(close_summary.get("close_results", [])),
                    )
                ),
                reconciliation_errors=reconciliation_errors,
            ).to_dict()
            with open(close_log_path, "w", encoding="utf-8") as handle:
                json.dump(close_summary, handle, ensure_ascii=False, indent=2)
            raise RuntimeError(
                "Close execution completed with reconciliation errors: "
                + "; ".join(reconciliation_errors)
            )

        return close_summary

    finally:
        try:
            if api_client is not None:
                api_client.close()
        finally:
            lease_stack.close()
