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
from collections.abc import Sequence
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
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
from leadlag.execution.state_store import ExecutionRun, ExecutionStateStore

logger = logging.getLogger(__name__)


def _close_result_record(result: OrderResult) -> dict[str, Any]:
    """Convert a broker close response to the durable observation shape."""
    return {
        "order_id": result.order_id,
        "status": result.status.value,
        "ticker": result.ticker,
        "side": result.side.value,
        "quantity": result.quantity,
        "message": result.message,
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

    # Build close-order metadata and OrderRequest list
    # Apply overnight holding ratios: only close (1 - alpha) fraction at 引け
    close_order_meta: list[dict[str, Any]] = []
    close_order_requests: list[OrderRequest] = []
    held_overnight_meta: list[dict[str, Any]] = []
    for pos in positions:
        if pos.quantity <= 0:
            continue

        # Determine alpha based on position side
        alpha = overnight_alpha_long if pos.side == "BUY" else overnight_alpha_short
        close_fraction = 1.0 - alpha
        lot_size = lot_size_for(pos.ticker)

        # Round target hold quantity to lot size, then derive close quantity.
        # This keeps the actual overnight holding ratio as close to alpha as possible
        # and avoids the old rounding approach drifting from the configured alpha.
        hold_qty_target = pos.quantity * alpha
        hold_qty = min(
            pos.quantity,
            math.floor(hold_qty_target / lot_size + 0.5) * lot_size,
        )
        close_qty = pos.quantity - hold_qty

        # Close quantity must be a lot multiple; floor any residual caused by
        # the position not being on a lot boundary.  The residual is held.
        close_qty = math.floor(close_qty / lot_size) * lot_size
        close_qty = max(0, min(close_qty, pos.quantity))
        hold_qty = pos.quantity - close_qty

        if hold_qty > 0:
            held_overnight_meta.append({
                "ticker": pos.ticker,
                "side": pos.side,
                "hold_quantity": hold_qty,
                "alpha": alpha,
            })
            logger.info(
                "  Overnight hold: %s %s x%d (alpha=%.2f, held=%.0f%%)",
                pos.ticker, pos.side, hold_qty, alpha, alpha * 100,
            )

        if close_qty <= 0:
            logger.info(
                "  Skipping close for %s %s: close_qty=0 (alpha=%.2f)",
                pos.ticker, pos.side, alpha,
            )
            continue

        close_side_str = "SELL" if pos.side == "BUY" else "BUY"
        close_side = OrderSide.SELL if pos.side == "BUY" else OrderSide.BUY
        close_order_meta.append(
            {
                "ticker": pos.ticker,
                "exchange": pos.exchange or 27,
                "side": close_side_str,
                "quantity": close_qty,
                "margin_trade_type": pos.margin_trade_type,
                "account_type": pos.account_type,
                "order_type": "CLO",
                "original_side": pos.side,
                "original_price": pos.price,
            }
        )
        close_order_requests.append(
            OrderRequest(
                ticker=pos.ticker,
                side=close_side,
                quantity=close_qty,
                order_type=OrderType.CLOSE,
                margin_trade_type=pos.margin_trade_type,
                account_type=pos.account_type,
            )
        )
        logger.info(
            "  Position to close: %s %s x%d/%d → %s (引成（後場）, close=%.0f%%)",
            pos.ticker,
            pos.side,
            close_qty,
            pos.quantity,
            close_side_str,
            close_fraction * 100,
        )

    close_meta_by_ticker = {m["ticker"]: m for m in close_order_meta}

    starting_positions: dict[str, int] = {}
    for position in positions:
        signed_quantity = int(position.quantity) * (1 if position.side == "BUY" else -1)
        starting_positions[position.ticker] = starting_positions.get(position.ticker, 0) + signed_quantity
    close_plan = ExecutionPlan(
        decision_id=build_decision_id(
            datetime.now().date().isoformat(), close_order_requests, ()
        ),
        trade_date=datetime.now().date().isoformat(),
        close_orders=tuple(close_order_requests),
        new_orders=(),
        current_positions=tuple(sorted(starting_positions.items())),
    )
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
        "overnight_alpha_long": overnight_alpha_long,
        "overnight_alpha_short": overnight_alpha_short,
        "held_overnight": held_overnight_meta,
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

    if dry_run:
        logger.info("[DRY RUN MODE] Simulating position close (no actual orders sent)...")
        for meta in close_order_meta:
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
            summary["close_results"].append(simulated)
    else:
        # Split large 1629.T close orders into immediate + delayed batches
        # close_order_requests already contains OrderRequest objects with metadata
        immediate_close, delayed_close = split_large_orders(close_order_requests)

        immediate_requests = list(immediate_close)

        logger.info("[LIVE MODE] Submitting %d position close orders...", len(immediate_requests))
        close_results = api_client.submit_orders_batch(
            immediate_requests,
            delay_ms=250,
            is_close=True,
            close_position_order=close_position_order,
        )
        persist_submitted_observations(close_results)
        first_batch_failed = False
        for result in close_results:
            logger.info(
                "  [CLOSE SUBMITTED] %s: %d shares (Order ID: %s)",
                result.ticker,
                result.quantity,
                result.order_id,
            )
            meta = close_meta_by_ticker.get(result.ticker, {})
            summary["close_results"].append(
                {
                    "order_id": result.order_id,
                    "status": result.status.value,
                    "ticker": result.ticker,
                    "side": result.side.value,
                    "quantity": result.quantity,
                    "message": result.message,
                    "original_side": meta.get("original_side"),
                    "original_price": meta.get("original_price"),
                }
            )
            if result.status == OrderStatus.FAILED:
                first_batch_failed = True

        # Wait for close fills before deciding whether to proceed.
        _wait_for_close_fills_sync(api_client, summary["close_results"])
        first_batch_failed = any(
            r.get("status") in {
                OrderStatus.FAILED.value,
                OrderStatus.PARTIALLY_FILLED.value,
            }
            for r in summary["close_results"]
            if not r.get("delayed")
        )

        # Delayed close batch (1629.T second half)
        if delayed_close:
            if first_batch_failed:
                logger.warning(
                    "[DELAYED CLOSE] Skipping %d delayed close order(s) — first batch had failures",
                    len(delayed_close),
                )
                for req in delayed_close:
                    meta = close_meta_by_ticker.get(req.ticker, {})
                    summary["close_results"].append({
                        "order_id": "",
                        "status": "SKIPPED",
                        "ticker": req.ticker,
                        "side": req.side.value,
                        "quantity": req.quantity,
                        "message": "Skipped due to first batch failure",
                        "delayed": True,
                        "original_side": meta.get("original_side"),
                        "original_price": meta.get("original_price"),
                    })
            else:
                delayed_requests = list(delayed_close)
                logger.info(
                    "[DELAYED CLOSE] Waiting %d seconds before submitting %d delayed close order(s)...",
                    SPLIT_DELAY_SECONDS, len(delayed_requests),
                )
                time_module.sleep(SPLIT_DELAY_SECONDS)
                logger.info("[DELAYED CLOSE] Submitting %d delayed close orders...", len(delayed_requests))
                delayed_results = api_client.submit_orders_batch(
                    delayed_requests,
                    delay_ms=250,
                    is_close=True,
                    close_position_order=close_position_order,
                )
                persist_submitted_observations(delayed_results)
                for result in delayed_results:
                    logger.info(
                        "  [DELAYED CLOSE SUBMITTED] %s: %d shares (Order ID: %s)",
                        result.ticker,
                        result.quantity,
                        result.order_id,
                    )
                    meta = close_meta_by_ticker.get(result.ticker, {})
                    summary["close_results"].append(
                        {
                            "order_id": result.order_id,
                            "status": result.status.value,
                            "ticker": result.ticker,
                            "side": result.side.value,
                            "quantity": result.quantity,
                            "message": result.message,
                            "delayed": True,
                            "original_side": meta.get("original_side"),
                            "original_price": meta.get("original_price"),
                        }
                    )

                # Wait for delayed close fills as well.
                _wait_for_close_fills_sync(api_client, summary["close_results"])

    # Fetch fill prices for close orders (約定価格取得)
    if not dry_run and summary["close_results"]:
        from leadlag.broker.dry_run import DryRunBrokerClient
        if not isinstance(api_client, DryRunBrokerClient):
            try:
                fetch_fill_prices(api_client, summary["close_results"], wait_seconds=5.0)
            except Exception as exc:  # noqa: BLE001
                summary.setdefault("reconciliation_errors", []).append(f"fill_prices: {exc}")
                logger.exception("Failed to fetch close fill prices")

    terminal_successes = {OrderStatus.FILLED.value, OrderStatus.SIMULATED.value}
    success_count = sum(
        1 for r in summary["close_results"]
        if r.get("status") in terminal_successes
    )
    partial_count = sum(
        1 for r in summary["close_results"]
        if r.get("status") == OrderStatus.PARTIALLY_FILLED.value
    )
    pending_count = sum(
        1 for r in summary["close_results"]
        if r.get("status") in {
            OrderStatus.SUBMITTED.value,
            OrderStatus.PARTIALLY_FILLED.value,
        }
    )
    failed_count = sum(
        1 for r in summary["close_results"]
        if r.get("status") in {
            OrderStatus.FAILED.value,
            OrderStatus.CANCELLED.value,
            "SKIPPED",
        }
    )
    summary["filled_orders_count"] = success_count
    summary["partial_orders_count"] = partial_count
    summary["pending_orders_count"] = pending_count
    summary["failed_orders_count"] = failed_count
    summary["close_incomplete"] = any(
        r.get("status") not in terminal_successes
        for r in summary["close_results"]
    ) or bool(summary.get("reconciliation_errors"))
    summary["execution_report"] = report_from_records(
        summary["close_results"],
        expected_orders=len(summary["close_results"]),
        reconciliation_errors=summary.get("reconciliation_errors", []),
    ).to_dict()

    if state_store is not None and execution_run is not None:
        try:
            state_store.record_result_set(
                execution_run.run_id, close_plan, summary["close_results"]
            )
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
                        f"filled={success_count}/{len(summary['close_results'])}; "
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
            summary["close_results"], expected_orders=close_plan.expected_order_count,
            reconciliation_errors=summary["reconciliation_errors"],
        ).to_dict()

    log_path = os.path.join(output_dir, "close_execution_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("Close execution log saved: %s", log_path)

    total_close_orders = len(summary["close_results"])
    if summary["close_incomplete"]:
        logger.error(
            "Position close incomplete: filled=%d/%d, pending=%d, failed=%d",
            success_count,
            total_close_orders,
            pending_count,
            failed_count,
        )
    else:
        logger.info("Position close completed: %d/%d orders filled", success_count, total_close_orders)
    return summary


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

        if reconciliation_errors:
            close_summary["close_incomplete"] = True
            close_summary["reconciliation_errors"] = reconciliation_errors
            close_summary["execution_report"] = report_from_records(
                close_summary.get("close_results", []),
                expected_orders=len(close_summary.get("close_results", [])),
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
