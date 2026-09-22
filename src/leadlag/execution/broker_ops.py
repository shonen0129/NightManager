"""Broker client and order-submission helpers.

This module was split from ``execution/helpers.py`` as part of P1-B1 to
isolate broker connectivity, position queries, and order submission from
pricing, output, and risk concerns.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path

import pandas as pd

from leadlag.broker.base import BrokerClient
from leadlag.broker.factory import create_broker_from_args
from leadlag.core.types import (
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
)
from leadlag.data.tickers import lot_size_for
from leadlag.execution.config import load_config_from_yaml
from leadlag.execution.contracts import (
    ExecutionPlan,
    build_decision_id,
    report_from_records,
)
from leadlag.execution.order_lifecycle import poll_order_statuses
from leadlag.execution.state_store import (
    ExecutionRun,
    ExecutionStateConflict,
    ExecutionStateStore,
)

logger = logging.getLogger(__name__)

SPLIT_TICKER = "1629.T"
SPLIT_THRESHOLD = 100
SPLIT_DELAY_SECONDS = 60
FILL_POLL_TIMEOUT_SECONDS = 30.0
FILL_POLL_INTERVAL_SECONDS = 1.0


class OrderExecutionIncomplete(RuntimeError):
    """Raised when submitted orders or their initial record are incomplete."""

    def __init__(
        self,
        message: str,
        summary: dict,
        log_path: str,
        *,
        recording_errors: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.summary = summary
        self.log_path = log_path
        self.recording_errors = recording_errors or []


def build_api_client(
    api_url: str | None,
    api_token: str | None,
    api_dry_run: bool = False,
) -> BrokerClient:
    """Build and validate a BrokerClient.

    Delegates to ``broker.factory.create_broker_from_args``.
    """
    app_cfg = load_config_from_yaml()
    provider = app_cfg.broker_provider

    if provider == "tachibana" and not api_dry_run:
        tachi = app_cfg.tachibana
        final_api_url = api_url if api_url else tachi.api_url
        final_api_token = api_token if api_token else tachi.auth_id
        api_password = tachi.second_password
        margin_trade_type = tachi.margin_trade_type
        account_type = tachi.account_type
        request_timeout = tachi.request_timeout
        extra = {"private_key_path": tachi.private_key_path}
    else:
        kabu = app_cfg.kabu
        final_api_url = api_url if api_url else kabu.api_url
        final_api_token = api_token if api_token else kabu.api_token
        api_password = kabu.api_password or os.environ.get("KABU_API_PASSWORD", "")
        margin_trade_type = kabu.margin_trade_type
        account_type = kabu.account_type
        request_timeout = kabu.request_timeout
        extra = {}

    client = create_broker_from_args(
        api_url=final_api_url,
        api_token=final_api_token or None,
        api_password=api_password or None,
        dry_run=api_dry_run,
        margin_trade_type=margin_trade_type,
        account_type=account_type,
        request_timeout=request_timeout,
        extra=extra,
    )

    # Try to restore a previously-saved Tachibana session to avoid re-login.
    restored = False
    restore_fn = getattr(client, "restore_session", None)
    if restore_fn:
        try:
            restored = bool(restore_fn())
            if restored:
                logger.info("[API] Restored broker session from cache")
        except Exception as e:
            logger.warning("[API] Failed to restore broker session: %s", e)

    logger.info("[API] Checking API connectivity (provider=%s)...", provider)
    healthy = client.health_check()
    if not healthy and restored:
        logger.warning("[API] Restored session is invalid; retrying with fresh login")
        discard_fn = getattr(client, "discard_restored_session", None)
        if discard_fn:
            discard_fn()
        healthy = client.health_check()

    if not healthy:
        if api_dry_run:
            logger.warning("[API] Health check failed, continuing in dry-run mode...")
        else:
            raise RuntimeError(
                f"Failed to connect to broker API (provider={provider}). "
                "Verify API URL, token, and credentials are correct."
            )
    else:
        logger.info("[API] Connection successful")

    return client


def fetch_current_positions(api_client: BrokerClient) -> dict[str, int]:
    """Fetch current open positions and return as a signed-quantity dict.

    Returns:
        Dict mapping ticker → signed quantity (positive=long, negative=short).
    """
    positions = api_client.get_positions()
    current: dict[str, int] = {}
    for pos in positions:
        if pos.quantity <= 0:
            continue
        signed_qty = pos.quantity if pos.side == "BUY" else -pos.quantity
        current[pos.ticker] = current.get(pos.ticker, 0) + signed_qty
    logger.info("[POSITIONS] Current holdings: %s", current or "(none)")
    return current


def resolve_wallet_capital(api_client: BrokerClient) -> float:
    """Resolve capital from broker wallet.

    Prefer 受入保証金 (deposited margin = equity base) for margin trading.
    This value is stable regardless of overnight positions, unlike
    cash_available (現物買付可能額) or margin_available (信用新規建可能額)
    which are reduced by existing positions.
    """
    wallet = api_client.get_wallet()
    ukeire = wallet.extra.get("ukeire_hosyoukin")
    if ukeire is not None and ukeire > 0:
        logger.info(
            "[CAPITAL] Using 受入保証金 (deposited margin) for sizing: %s JPY",
            f"{ukeire:,.0f}",
        )
        return float(ukeire)
    # Fallback for brokers without 受入保証金 (e.g. kabu)
    cash_available = float(wallet.cash_available)
    logger.info(
        "[CAPITAL] Using cash wallet balance for sizing: %s JPY",
        f"{cash_available:,.0f}",
    )
    return cash_available


def split_large_orders(
    orders: list[OrderRequest],
) -> tuple[list[OrderRequest], list[OrderRequest]]:
    """Split large orders for SPLIT_TICKER into immediate and delayed batches.

    Only applies when qty >= SPLIT_THRESHOLD (100 shares). Smaller orders
    are kept in the immediate batch unchanged. All OrderRequest fields
    (including margin_trade_type and account_type) are preserved.

    Returns:
        (immediate_orders, delayed_orders)
    """
    immediate: list[OrderRequest] = []
    delayed: list[OrderRequest] = []
    for req in orders:
        ticker = req.ticker
        qty = req.quantity
        if ticker == SPLIT_TICKER and qty >= SPLIT_THRESHOLD:
            lot = lot_size_for(SPLIT_TICKER)
            first_qty = (qty // 2 // lot) * lot
            second_qty = qty - first_qty
            if first_qty > 0 and second_qty > 0:
                logger.info(
                    "[SPLIT] %s: splitting %d shares → first=%d (immediate), second=%d (delayed %ds)",
                    ticker, qty, first_qty, second_qty, SPLIT_DELAY_SECONDS,
                )
                immediate.append(
                    OrderRequest(
                        ticker=req.ticker,
                        side=req.side,
                        quantity=first_qty,
                        order_type=req.order_type,
                        limit_price=req.limit_price,
                        margin_trade_type=req.margin_trade_type,
                        account_type=req.account_type,
                    )
                )
                delayed.append(
                    OrderRequest(
                        ticker=req.ticker,
                        side=req.side,
                        quantity=second_qty,
                        order_type=req.order_type,
                        limit_price=req.limit_price,
                        margin_trade_type=req.margin_trade_type,
                        account_type=req.account_type,
                    )
                )
            else:
                immediate.append(req)
        else:
            immediate.append(req)
    return immediate, delayed


def _wait_for_fills_sync(
    api_client: BrokerClient,
    results: list[OrderResult],
    timeout_seconds: float = FILL_POLL_TIMEOUT_SECONDS,
    poll_interval: float = FILL_POLL_INTERVAL_SECONDS,
) -> tuple[list[OrderResult], bool]:
    """Poll order statuses until all are terminal or the timeout expires.

    Updates each ``OrderResult`` in place with the latest status. Submitted
    market orders at 9:10 should fill quickly; if they are still SUBMITTED
    after the timeout, the caller must decide whether to proceed.

    Returns:
        (results, polled): the (possibly updated) results and a bool that is
        True when the broker actually supported status polling and was used.
    """
    updated, polled = poll_order_statuses(
        api_client,
        results,
        order_id_getter=lambda result: result.order_id,
        status_getter=lambda result: result.status,
        status_setter=lambda result, status: setattr(result, "status", status),
        message_setter=lambda result, message: setattr(result, "message", message),
        timeout_seconds=timeout_seconds,
        poll_interval=poll_interval,
        label="FILL",
    )
    return list(updated), polled


def _build_order_deltas(
    decision_df: pd.DataFrame,
    current_positions: dict[str, int] | None,
) -> tuple[list[tuple[str, OrderSide, int]], list[tuple[str, OrderSide, int]], list[tuple[str, int, int, int]]]:
    """Compute target/current/delta per ticker and split into close vs new orders.

    Returns:
        (close_orders, new_orders, delta_log_entries)
    """
    current: dict[str, int] = current_positions if current_positions is not None else {}
    close_orders: list[tuple[str, OrderSide, int]] = []
    new_orders: list[tuple[str, OrderSide, int]] = []
    delta_log_entries: list[tuple[str, int, int, int]] = []

    for _, row in decision_df.iterrows():
        ticker = str(row["ticker"])
        target_qty = int(row["quantity"])
        if row["action"] == "BUY":
            target_signed = target_qty
        elif row["action"] == "SELL":
            target_signed = -target_qty
        else:
            target_signed = 0

        current_signed = current.get(ticker, 0)
        delta = target_signed - current_signed
        delta_log_entries.append((ticker, target_signed, current_signed, delta))

        if delta == 0:
            continue

        order_side = OrderSide.BUY if delta > 0 else OrderSide.SELL
        abs_delta = abs(delta)

        if current_signed > 0 and delta < 0:
            close_qty = min(abs_delta, current_signed)
            new_qty = abs_delta - close_qty
        elif current_signed < 0 and delta > 0:
            close_qty = min(abs_delta, abs(current_signed))
            new_qty = abs_delta - close_qty
        else:
            close_qty = 0
            new_qty = abs_delta

        if close_qty > 0:
            close_orders.append((ticker, order_side, close_qty))
        if new_qty > 0:
            new_orders.append((ticker, order_side, new_qty))

    return close_orders, new_orders, delta_log_entries


def build_execution_plan(
    decision_df: pd.DataFrame,
    current_positions: dict[str, int] | None = None,
) -> ExecutionPlan:
    """Convert a target decision frame into an immutable execution plan."""
    close_orders, new_orders, _ = _build_order_deltas(decision_df, current_positions)
    close_requests = tuple(
        OrderRequest(
            ticker=ticker,
            side=side,
            quantity=quantity,
            order_type=OrderType.MARKET,
            is_close=True,
        )
        for ticker, side, quantity in close_orders
    )
    new_requests = tuple(
        OrderRequest(ticker=ticker, side=side, quantity=quantity, order_type=OrderType.MARKET)
        for ticker, side, quantity in new_orders
    )
    trade_date = str(decision_df.attrs.get("trade_date", ""))
    current = tuple(sorted((str(ticker), int(quantity)) for ticker, quantity in (current_positions or {}).items()))
    weight = pd.to_numeric(decision_df.get("weight", pd.Series(dtype=float)), errors="coerce")
    target_net = float(weight.sum()) if not weight.empty else None
    target_gross = float(weight.abs().sum()) if not weight.empty else None
    return ExecutionPlan(
        decision_id=build_decision_id(trade_date, close_requests, new_requests),
        trade_date=trade_date,
        close_orders=close_requests,
        new_orders=new_requests,
        current_positions=current,
        target_net_exposure=target_net,
        target_gross_exposure=target_gross,
    )


def _submit_close_orders(
    api_client: BrokerClient,
    close_orders: list[tuple[str, OrderSide, int]],
    close_position_order: int = 0,
    on_submitted: Callable[[Sequence[OrderResult]], None] | None = None,
) -> tuple[list[OrderResult], list[dict], bool]:
    """Build OrderRequests for close orders, submit them, and return result dicts."""
    close_order_requests = [
        OrderRequest(
            ticker=ticker,
            side=side,
            quantity=qty,
            order_type=OrderType.MARKET,
            is_close=True,
            close_position_order=close_position_order,
        )
        for ticker, side, qty in close_orders
    ]
    close_results: list[OrderResult] = []
    close_result_dicts: list[dict] = []
    close_polled = False

    if close_order_requests:
        logger.info("[CLOSE PHASE] Submitting %d close (返済) orders first...", len(close_order_requests))
        close_results = api_client.submit_orders_batch(
            close_order_requests,
            delay_ms=250,
            is_close=True,
            close_position_order=close_position_order,
        )
        if on_submitted is not None:
            on_submitted(close_results)
        close_results, close_polled = _wait_for_fills_sync(api_client, close_results)
        for result in close_results:
            logger.info(
                "  [CLOSE] %s: %d shares (Order ID: %s, Status: %s)",
                result.ticker,
                result.quantity,
                result.order_id,
                result.status.value,
            )
            close_result_dicts.append({
                "order_id": result.order_id,
                "status": result.status.value,
                "ticker": result.ticker,
                "side": result.side.value,
                "quantity": result.quantity,
                "message": result.message,
                "eigyou_day": result.eigyou_day,
            })

    return close_results, close_result_dicts, close_polled


def _submit_new_orders(
    api_client: BrokerClient,
    immediate_orders: list[OrderRequest],
    on_submitted: Callable[[Sequence[OrderResult]], None] | None = None,
) -> tuple[list[OrderResult], bool, bool]:
    """Submit the immediate new-order batch and detect any first-batch failure."""
    first_batch_failed = False
    results: list[OrderResult] = []
    polled = False

    if immediate_orders:
        logger.info("[NEW PHASE] Submitting %d new (新規) orders...", len(immediate_orders))
        results = api_client.submit_orders_batch(immediate_orders, delay_ms=250)
        if on_submitted is not None:
            on_submitted(results)
        results, polled = _wait_for_fills_sync(api_client, results)
        for result in results:
            result_side = result.side.value
            logger.info(
                "  [%s] %s: %d shares (Order ID: %s, Status: %s)",
                result_side,
                result.ticker,
                result.quantity,
                result.order_id,
                result.status.value,
            )
            if result.status == OrderStatus.FAILED:
                first_batch_failed = True
            elif result.status == OrderStatus.PARTIALLY_FILLED or (
                polled and result.status not in (OrderStatus.FILLED, OrderStatus.SIMULATED)
            ):
                first_batch_failed = True

    return results, first_batch_failed, polled


def _submit_delayed_orders(
    api_client: BrokerClient,
    delayed_orders: list[OrderRequest],
    first_batch_failed: bool,
    summary: dict,
    on_submitted: Callable[[Sequence[OrderResult]], None] | None = None,
) -> bool:
    """Sleep and submit delayed orders, or mark SKIPPED if first batch failed.

    Modifies ``summary["buy_results"]`` and ``summary["sell_results"]`` in place.
    Returns True if the delayed batch itself had a fill/terminal failure.
    """
    if not delayed_orders:
        return False

    if first_batch_failed:
        logger.warning(
            "[DELAYED PHASE] Skipping %d delayed order(s) — first batch had failures",
            len(delayed_orders),
        )
        for req in delayed_orders:
            result_dict = {
                "order_id": "",
                "status": "SKIPPED",
                "ticker": req.ticker,
                "side": req.side.value,
                "quantity": req.quantity,
                "message": "Skipped due to first batch failure",
                "delayed": True,
            }
            if req.side == OrderSide.BUY:
                summary["buy_results"].append(result_dict)
            elif req.side == OrderSide.SELL:
                summary["sell_results"].append(result_dict)
        return False

    delayed_requests = list(delayed_orders)
    logger.info(
        "[DELAYED PHASE] Waiting %d seconds before submitting %d delayed order(s)...",
        SPLIT_DELAY_SECONDS, len(delayed_requests),
    )
    time.sleep(SPLIT_DELAY_SECONDS)
    logger.info("[DELAYED PHASE] Submitting %d delayed (新規) orders...", len(delayed_requests))
    delayed_results = api_client.submit_orders_batch(delayed_requests, delay_ms=250)
    if on_submitted is not None:
        on_submitted(delayed_results)
    delayed_results, delayed_polled = _wait_for_fills_sync(api_client, delayed_results)
    delayed_failed = False
    for result in delayed_results:
        result_side = result.side.value
        logger.info(
            "  [DELAYED %s] %s: %d shares (Order ID: %s, Status: %s)",
            result_side,
            result.ticker,
            result.quantity,
            result.order_id,
            result.status.value,
        )
        if result.status == OrderStatus.FAILED:
            delayed_failed = True
        elif result.status == OrderStatus.PARTIALLY_FILLED or (
            delayed_polled and result.status not in (OrderStatus.FILLED, OrderStatus.SIMULATED)
        ):
            delayed_failed = True
        result_dict = {
            "order_id": result.order_id,
            "status": result.status.value,
            "ticker": result.ticker,
            "side": result_side,
            "quantity": result.quantity,
            "message": result.message,
            "delayed": True,
        }
        if result_side == "BUY":
            summary["buy_results"].append(result_dict)
        elif result_side == "SELL":
            summary["sell_results"].append(result_dict)
    return delayed_failed


def _write_api_execution_log(summary: dict, output_dir: str | Path) -> str:
    """Write `api_execution_log.json` and return its path."""
    log_path = os.path.join(output_dir, "api_execution_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("API execution log saved: %s", log_path)
    return log_path


def _order_result_record(result: OrderResult) -> dict:
    """Convert an immediate broker response to the durable observation shape."""
    return {
        "order_id": result.order_id,
        "status": result.status.value,
        "ticker": result.ticker,
        "side": result.side.value,
        "quantity": result.quantity,
        "message": result.message,
        "eigyou_day": result.eigyou_day,
    }


def submit_orders_via_api(
    decision_df: pd.DataFrame,
    api_client: BrokerClient,
    output_dir: str | Path,
    current_positions: dict[str, int] | None = None,
    *,
    execution_plan: ExecutionPlan | None = None,
    state_store: ExecutionStateStore | None = None,
    account_key: str = "default",
    strategy_key: str = "production_v2",
    run_id: str | None = None,
) -> dict:
    """Submit trade orders to the broker API, accounting for existing positions.

    When ``current_positions`` is provided, only the delta between target and
    current quantities is submitted. This avoids re-ordering the full target
    when positions are already held from overnight carry-over.

    The dry-run vs live distinction is handled entirely by the BrokerClient
    implementation: ``DryRunBrokerClient`` simulates orders without sending
    them, while ``KabuBrokerClient`` submits to the real API. This function
    does not need to know which variant is being used.
    """
    is_dry_run = type(api_client).__name__ == "DryRunBrokerClient"
    current = current_positions or {}

    plan = execution_plan or build_execution_plan(decision_df, current_positions)
    close_orders = [
        (order.ticker, order.side, order.quantity) for order in plan.close_orders
    ]
    new_orders = [
        (order.ticker, order.side, order.quantity) for order in plan.new_orders
    ]
    _, _, delta_log_entries = _build_order_deltas(decision_df, current_positions)

    if current:
        logger.info("[DELTA] Reconciling against %d existing position(s):", len(current))
        for ticker, target_signed, current_signed, delta in delta_log_entries:
            if delta != 0:
                logger.info(
                    "  %s: target=%d, current=%d, delta=%d",
                    ticker, target_signed, current_signed, delta,
                )
    else:
        logger.info("[DELTA] No existing positions; submitting full target quantities")

    buy_count = sum(1 for _, side, _ in new_orders if side == OrderSide.BUY)
    sell_count = sum(1 for _, side, _ in new_orders if side == OrderSide.SELL)

    summary: dict = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dry_run": is_dry_run,
        "buy_orders_count": buy_count,
        "sell_orders_count": sell_count,
        "current_positions": current,
        "execution_plan": plan.to_dict(),
        "buy_results": [],
        "sell_results": [],
        "close_results": [],
    }

    observed_records: list[dict] = []

    def persist_submitted_observations(results: Sequence[OrderResult]) -> None:
        """Commit broker responses before polling or submitting another batch."""
        if state_store is None or run_id is None:
            return
        observed_records.extend(_order_result_record(result) for result in results)
        # A state-store error deliberately propagates.  The run remains
        # executing and no later batch is submitted; recovery must reconcile
        # the broker before a retry.
        state_store.record_result_set(run_id, plan, observed_records)

    # Persist the complete intent before the first broker call.  If a previous
    # process reached the broker but did not finish reconciliation, the durable
    # state store raises and this call must stop rather than submit a duplicate.
    execution_run: ExecutionRun | None = None
    if state_store is not None:
        try:
            if run_id is None:
                execution_run = state_store.prepare_run(
                    account_key=account_key,
                    strategy_key=strategy_key,
                    trade_date=plan.trade_date,
                    job_type="decision",
                    decision_id=plan.decision_id,
                    metadata={"expected_orders": plan.expected_order_count},
                )
                run_id = execution_run.run_id
            else:
                execution_run = state_store.get_run(run_id)
                if execution_run is None:
                    raise ExecutionStateConflict(f"Unknown execution run: {run_id}")
            state_store.record_plan(run_id, plan)
            state_store.mark_submission_started(run_id)
            summary["run_id"] = run_id
        except Exception as exc:  # noqa: BLE001
            summary["execution_error"] = str(exc)
            summary["execution_error_type"] = type(exc).__name__
            log_path = os.path.join(output_dir, "api_execution_log.json")
            try:
                _write_api_execution_log(summary, output_dir)
            except Exception:
                logger.exception("[EXECUTION] Failed to persist state conflict summary")
            raise OrderExecutionIncomplete(
                f"Execution state prevented order submission: {exc}",
                summary,
                log_path,
                recording_errors=[str(exc)],
            ) from exc

    close_results, close_result_dicts, close_polled = _submit_close_orders(
        api_client, close_orders, on_submitted=persist_submitted_observations
    )
    summary["close_results"] = close_result_dicts

    close_failed = any(
        r.status == OrderStatus.FAILED
        or (close_polled and r.status not in (OrderStatus.FILLED, OrderStatus.SIMULATED))
        for r in close_results
    )
    if close_failed:
        n_failed = sum(
            1 for r in close_results
            if r.status == OrderStatus.FAILED
            or (close_polled and r.status not in (OrderStatus.FILLED, OrderStatus.SIMULATED))
        )
        logger.error(
            "[EXECUTION] %d close order(s) did not reach a terminal success state; "
            "skipping new order submission to avoid double exposure.",
            n_failed,
        )

    unsplit_new_order_requests = [
        OrderRequest(
            ticker=ticker,
            side=side,
            quantity=qty,
            order_type=OrderType.MARKET,
        )
        for ticker, side, qty in new_orders
    ]
    immediate_orders, delayed_orders = split_large_orders(unsplit_new_order_requests)

    new_order_requests = list(immediate_orders)
    expected_orders_count = len(close_orders) + len(immediate_orders) + len(delayed_orders)
    summary["expected_orders_count"] = expected_orders_count

    new_results: list[OrderResult] = []
    first_batch_failed = close_failed
    if not close_failed and new_order_requests:
        new_results, first_batch_failed, _ = _submit_new_orders(
            api_client, new_order_requests, on_submitted=persist_submitted_observations
        )
        for result in new_results:
            result_side = result.side.value
            result_dict = {
                "order_id": result.order_id,
                "status": result.status.value,
                "ticker": result.ticker,
                "side": result_side,
                "quantity": result.quantity,
                "message": result.message,
                "delayed": False,
                "eigyou_day": result.eigyou_day,
            }
            if result_side == "BUY":
                summary["buy_results"].append(result_dict)
            elif result_side == "SELL":
                summary["sell_results"].append(result_dict)

    delayed_failed = False
    if delayed_orders:
        delayed_failed = _submit_delayed_orders(
            api_client,
            delayed_orders,
            first_batch_failed,
            summary,
            on_submitted=persist_submitted_observations,
        )

    result_count = (
        len(summary["buy_results"]) + len(summary["sell_results"]) + len(summary["close_results"])
    )
    accepted_statuses = {
        OrderStatus.SUBMITTED.value,
        OrderStatus.PARTIALLY_FILLED.value,
        OrderStatus.FILLED.value,
        OrderStatus.SIMULATED.value,
    }
    all_result_dicts = summary["buy_results"] + summary["sell_results"] + summary["close_results"]
    accepted_orders_count = sum(
        1 for result in all_result_dicts if result.get("status") in accepted_statuses
    )
    failed_orders_count = sum(
        1 for result in all_result_dicts if result.get("status") == OrderStatus.FAILED.value
    )
    partial_orders_count = sum(
        1 for result in all_result_dicts
        if result.get("status") == OrderStatus.PARTIALLY_FILLED.value
    )
    unresolved_orders_count = sum(
        1 for result in all_result_dicts
        if result.get("status") in {
            OrderStatus.SUBMITTED.value,
            OrderStatus.PARTIALLY_FILLED.value,
        }
    )
    summary["result_count"] = result_count
    # Keep the legacy field, but give it its real meaning: an order accepted by
    # the broker or still awaiting terminal fill, never a rejected response.
    summary["submitted_orders_count"] = accepted_orders_count
    summary["accepted_orders_count"] = accepted_orders_count
    summary["filled_orders_count"] = sum(
        1 for result in all_result_dicts
        if result.get("status") in {OrderStatus.FILLED.value, OrderStatus.SIMULATED.value}
    )
    summary["failed_orders_count"] = failed_orders_count
    summary["partial_orders_count"] = partial_orders_count
    summary["unresolved_orders_count"] = unresolved_orders_count
    summary["close_failed"] = close_failed
    summary["first_batch_failed"] = first_batch_failed
    summary["delayed_failed"] = delayed_failed

    execution_report = report_from_records(
        all_result_dicts,
        expected_orders=plan.expected_order_count,
        close_failed=close_failed,
    )
    summary["execution_report"] = execution_report.to_dict()

    if state_store is not None and run_id is not None:
        try:
            state_store.record_result_set(run_id, plan, all_result_dicts)
            execution_incomplete_for_state = (
                failed_orders_count > 0
                or accepted_orders_count < expected_orders_count
                or unresolved_orders_count > 0
                or close_failed
                or first_batch_failed
                or delayed_failed
            )
            if execution_incomplete_for_state:
                state_store.mark_reconciliation_required(
                    run_id,
                    error=(
                        f"accepted={accepted_orders_count}/{expected_orders_count}; "
                        f"failed={failed_orders_count}; unresolved={unresolved_orders_count}"
                    ),
                )
            state_store.record_reconciliation(
                run_id,
                outcome="incomplete" if execution_incomplete_for_state else "pending",
                errors=summary.get("reconciliation_errors", []),
                references={"result_count": result_count, "expected_orders": expected_orders_count},
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("[EXECUTION] Durable state update failed")
            summary.setdefault("reconciliation_errors", []).append(f"state_store: {exc}")
            try:
                state_store.mark_reconciliation_required(run_id, error=str(exc))
            except Exception:
                logger.exception("[EXECUTION] Could not mark run reconciliation_required")

    log_path = os.path.join(output_dir, "api_execution_log.json")
    recording_errors: list[str] = []
    try:
        log_path = _write_api_execution_log(summary, output_dir)
    except Exception as exc:  # noqa: BLE001
        recording_errors.append(f"api_execution_log: {exc}")
        summary["execution_log_errors"] = recording_errors
        summary["execution_report"] = report_from_records(
            all_result_dicts,
            expected_orders=plan.expected_order_count,
            close_failed=close_failed,
            recording_errors=recording_errors,
        ).to_dict()
        logger.exception("[EXECUTION] Failed to write initial API execution log")

    execution_incomplete = bool(recording_errors) or bool(summary.get("reconciliation_errors")) or (
        not is_dry_run
        and (
            failed_orders_count > 0
            or accepted_orders_count < expected_orders_count
            or unresolved_orders_count > 0
            or close_failed
            or first_batch_failed
            or delayed_failed
        )
    )
    if execution_incomplete:
        message = (
            "Order submission incomplete: "
            f"accepted={accepted_orders_count}/expected={expected_orders_count}, "
            f"failed={failed_orders_count}, partial={partial_orders_count}, "
            f"unresolved={unresolved_orders_count}, "
            f"recording_errors={len(recording_errors)}. "
            f"See {log_path} for details."
        )
        raise OrderExecutionIncomplete(
            message,
            summary,
            log_path,
            recording_errors=recording_errors,
        )

    return summary
