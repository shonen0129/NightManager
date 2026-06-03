"""Execution engine: order submission and management."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime

import pandas as pd

from domain.models.types import OrderRequest, OrderResult, OrderStatus, OrderSide

logger = logging.getLogger(__name__)


class ExecutionEngine:
    """Manages order submission via the kabu API."""

    def __init__(self, api_client, dry_run: bool = False):
        self.api_client = api_client
        self.dry_run = dry_run

    def submit_order(self, order: OrderRequest) -> OrderResult:
        """Submit a single order."""
        clean_ticker = order.ticker.replace(".T", "")

        if self.dry_run:
            result = OrderResult(
                order_id=f"SIM-{datetime.now().strftime('%Y%m%d%H%M%S')}-{clean_ticker}",
                status=OrderStatus.SIMULATED,
                ticker=clean_ticker,
                side=order.side,
                quantity=order.quantity,
                order_type=order.order_type,
                limit_price=order.limit_price,
                message="Simulated (dry run)",
            )
            logger.info(f"  [DRY RUN] {order.side.value} {order.quantity} shares of {clean_ticker}")
            return result

        map_type = {"MO": "MO", "LO": "LO", "CLO": "CLO"}
        result = self.api_client.send_order(
            ticker=clean_ticker,
            side=order.side.value,
            quantity=order.quantity,
            order_type=map_type.get(order.order_type.value, "MO"),
            limit_price=order.limit_price,
        )

        if result is None:
            return OrderResult(
                order_id="", status=OrderStatus.FAILED,
                ticker=clean_ticker, side=order.side,
                quantity=order.quantity, order_type=order.order_type,
                limit_price=order.limit_price, message="API returned None",
            )

        return OrderResult(
            order_id=str(result.get("order_id", "")),
            status=OrderStatus.SUBMITTED,
            ticker=clean_ticker, side=order.side,
            quantity=order.quantity, order_type=order.order_type,
            limit_price=order.limit_price,
            margin_trade_type=result.get("margin_trade_type", 3),
        )

    def submit_orders_batch(
        self, orders: list[OrderRequest], output_dir: str | None = None, delay_ms: int = 50,
    ) -> list[OrderResult]:
        """Submit multiple orders with delay between each."""
        if not orders:
            return []

        mode = "DRY RUN" if self.dry_run else "LIVE"
        logger.info(f"[{mode}] Submitting {len(orders)} orders...")

        results = []
        for order in orders:
            result = self.submit_order(order)
            results.append(result)

        if output_dir:
            self._save_execution_log(results, output_dir)

        return results

    def _save_execution_log(self, results: list[OrderResult], output_dir: str) -> str:
        """Save execution log to JSON file."""
        log_data = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "dry_run": self.dry_run,
            "total_orders": len(results),
            "results": [
                {"order_id": r.order_id, "status": r.status.value, "ticker": r.ticker,
                 "side": r.side.value, "quantity": r.quantity, "message": r.message}
                for r in results
            ],
        }

        os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(output_dir, "api_execution_log.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)

        logger.info(f"API execution log saved: {log_path}")
        return log_path


def build_orders_from_decision(decision_df: pd.DataFrame) -> list[OrderRequest]:
    """Build OrderRequest list from a decision DataFrame."""
    active = decision_df[
        (decision_df["action"].isin(["BUY", "SELL"])) & (decision_df["quantity"] > 0)
    ]

    orders = []
    for _, row in active.iterrows():
        side = OrderSide.BUY if row["action"] == "BUY" else OrderSide.SELL
        orders.append(OrderRequest(
            ticker=str(row["ticker"]), side=side, quantity=int(row["quantity"]),
        ))

    return orders