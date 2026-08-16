"""leadlag/execution/close.py — position closing logic.

Provides ``close_all_positions()`` and ``wait_and_auto_close()`` which
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
from datetime import datetime
from pathlib import Path
from typing import Any

from leadlag.broker.base import BrokerClient
from leadlag.core.types import OrderRequest, OrderSide, OrderStatus, OrderType
from leadlag.data.tickers import lot_size_for
from leadlag.execution.broker_ops import (
    SPLIT_DELAY_SECONDS,
    build_api_client,
    split_large_orders,
)
from leadlag.execution.config import load_config_from_yaml
from leadlag.execution.output_ops import (
    build_output_dir,
    save_daily_journal,
    save_position_snapshot,
    save_wallet_snapshot,
)
from leadlag.execution.pricing import fetch_fill_prices

logger = logging.getLogger(__name__)


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
    if not callable(getattr(api_client, "get_order_status", None)):
        logger.debug("Broker client does not support get_order_status; skip fill polling")
        return

    pending = [r for r in close_results if r.get("status") == OrderStatus.SUBMITTED.value]
    if not pending:
        return

    # Probe whether the broker actually implements status polling.
    test_order_id = pending[0].get("order_id", "")
    try:
        api_client.get_order_status(test_order_id)
    except NotImplementedError:
        logger.debug("Broker client get_order_status is not implemented; skip fill polling")
        return
    except Exception:
        # Non-fatal probe error; continue with polling.
        pass

    deadline = time_module.time() + timeout_seconds
    logger.info(
        "[CLOSE FILL POLL] Waiting for %d close order(s) to fill (timeout=%.1fs)...",
        len(pending),
        timeout_seconds,
    )

    while pending and time_module.time() < deadline:
        for result in pending:
            order_id = result.get("order_id", "")
            if not order_id:
                continue
            try:
                status = api_client.get_order_status(order_id)
                if status != OrderStatus.SUBMITTED:
                    result["status"] = status.value
                    logger.info(
                        "  [CLOSE FILL] %s: %s",
                        result.get("ticker"),
                        status.value,
                    )
            except Exception as e:
                logger.warning(
                    "Failed to poll close order %s: %s",
                    order_id,
                    e,
                )

        pending = [r for r in close_results if r.get("status") == OrderStatus.SUBMITTED.value]
        if pending:
            time_module.sleep(poll_interval)

    if pending:
        logger.warning(
            "%d close order(s) still SUBMITTED after %.1fs",
            len(pending),
            timeout_seconds,
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
        return {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "dry_run": dry_run,
            "positions_found": 0,
            "close_orders": [],
        }

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
            r.get("status") == OrderStatus.FAILED.value
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
            fetch_fill_prices(api_client, summary["close_results"], wait_seconds=5.0)

    log_path = os.path.join(output_dir, "close_execution_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("Close execution log saved: %s", log_path)

    success_count = sum(
        1 for r in summary["close_results"]
        if r.get("status") != "FAILED"
    )
    total_close_orders = len(summary["close_results"])
    logger.info(
        "Position close completed: %d/%d orders submitted",
        success_count,
        total_close_orders,
    )
    return summary


def wait_and_auto_close(
    api_client: BrokerClient,
    output_dir: str | Path,
    auto_close_time: str,
    dry_run: bool = False,
    close_position_order: int = 0,
    max_wait_hours: float = 6.0,
) -> None:
    """Wait until ``auto_close_time`` and automatically close all positions.

    Args:
        api_client: BrokerClient instance
        output_dir: Directory to save close execution log
        auto_close_time: Time to close positions (HH:MM format)
        dry_run: If True, simulate without actual submission
        close_position_order: Close priority (0-7) for credit repayment
        max_wait_hours: Maximum wall-clock hours to wait before aborting.
            Prevents a decision process launched far before the close time
            from running indefinitely.
    """
    config = load_config_from_yaml()
    alpha_long = config.strategy.overnight_alpha_long
    alpha_short = config.strategy.overnight_alpha_short
    if config.broker_provider == "tachibana":
        margin_trade_type = config.tachibana.margin_trade_type
        account_type = config.tachibana.account_type
    else:
        margin_trade_type = config.kabu.margin_trade_type
        account_type = config.kabu.account_type

    hour, minute = map(int, auto_close_time.split(":"))
    now = datetime.now()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

    if target <= now:
        logger.warning(
            "Auto-close time %s has already passed. Closing positions immediately.",
            auto_close_time,
        )
        close_all_positions(
            api_client=api_client,
            output_dir=output_dir,
            dry_run=dry_run,
            margin_trade_type=margin_trade_type,
            account_type=account_type,
            close_position_order=close_position_order,
            overnight_alpha_long=alpha_long,
            overnight_alpha_short=alpha_short,
        )
        return

    wait_seconds = (target - now).total_seconds()
    max_wait_seconds = max_wait_hours * 3600.0
    if wait_seconds > max_wait_seconds:
        logger.error(
            "Auto-close time %s is %.1f hours in the future ( exceeds max_wait_hours=%.1f ). "
            "Aborting in-process wait. Run the 'close' subcommand separately instead.",
            auto_close_time,
            wait_seconds / 3600.0,
            max_wait_hours,
        )
        return

    logger.info(
        "=== AUTO-CLOSE SCHEDULED ===\n"
        "  Positions will be automatically closed at %s\n"
        "  Waiting %.0f seconds (%.1f hours)\n"
        "  Use --auto-close-time to change the close time",
        auto_close_time,
        wait_seconds,
        wait_seconds / 3600,
    )

    check_interval = 300  # 5 minutes
    while True:
        remaining = (target - datetime.now()).total_seconds()
        if remaining <= 0:
            break
        if remaining <= check_interval:
            logger.info("[HEARTBEAT] Auto-close: sleeping final %.0f seconds", remaining)
            time_module.sleep(remaining)
            break
        logger.info("[HEARTBEAT] Auto-close: sleeping %d seconds (%.0f minutes remaining)",
                    check_interval, remaining / 60)
        time_module.sleep(check_interval)

    logger.info("=== AUTO-CLOSE EXECUTION ===")
    close_all_positions(
        api_client=api_client,
        output_dir=output_dir,
        dry_run=dry_run,
        margin_trade_type=margin_trade_type,
        account_type=account_type,
        close_position_order=close_position_order,
        overnight_alpha_long=alpha_long,
        overnight_alpha_short=alpha_short,
    )
    logger.info("=== AUTO-CLOSE COMPLETED ===")


def run_close_positions_mode(
    output_root: str,
    run_tag: str | None,
    api_url: str | None,
    api_token: str | None,
    api_dry_run: bool,
    close_position_order: int,
) -> None:
    """Entry point for ``--mode close-positions``.

    Builds the broker client, executes position close, and cleans up.
    """
    logger.info("=== CLOSE-POSITIONS MODE ===")
    output_dir = build_output_dir(output_root, run_tag, run_name="production_close_positions")

    api_client: BrokerClient | None = None
    try:
        api_client = build_api_client(api_url, api_token, api_dry_run)
        config = load_config_from_yaml()
        if config.broker_provider == "tachibana":
            margin_trade_type = config.tachibana.margin_trade_type
            account_type = config.tachibana.account_type
        else:
            margin_trade_type = config.kabu.margin_trade_type
            account_type = config.kabu.account_type
        close_summary = close_all_positions(
            api_client=api_client,
            output_dir=output_dir,
            dry_run=api_dry_run,
            margin_trade_type=margin_trade_type,
            account_type=account_type,
            close_position_order=close_position_order,
            overnight_alpha_long=config.strategy.overnight_alpha_long,
            overnight_alpha_short=config.strategy.overnight_alpha_short,
        )
        logger.info(
            "Close-positions completed. Positions closed: %d",
            close_summary.get("close_orders_count", 0),
        )

        # --- Trade journal: collect post-close data ---
        close_log_path = os.path.join(output_dir, "close_execution_log.json")
        pos_snapshot_path = save_position_snapshot(api_client, output_dir, label="close")
        wallet_snapshot_path = save_wallet_snapshot(api_client, output_dir, label="close")
        save_daily_journal(
            output_dir=output_dir,
            close_execution_log_path=close_log_path,
            position_snapshot_path=pos_snapshot_path,
            wallet_snapshot_path=wallet_snapshot_path,
        )

    finally:
        if api_client is not None:
            api_client.close()
