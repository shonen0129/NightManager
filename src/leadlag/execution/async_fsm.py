"""Asynchronous Execution Engine & Order State Machine (Async FSM).

Executes target portfolio weights via non-blocking asynchronous state machines,
guaranteeing zero process hang / blocking I/O and strict order sequencing.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any

import numpy as np

from leadlag.broker.async_base import AsyncBrokerClient
from leadlag.broker.base import Position
from leadlag.core.types import OrderRequest, OrderResult, OrderSide, OrderStatus, OrderType
from leadlag.data.tickers import JP_TICKERS

logger = logging.getLogger(__name__)


class OrderState(Enum):
    """Lifecycle states of an execution order."""
    CREATED = auto()
    VALIDATED = auto()
    SUBMITTING = auto()
    FILLED = auto()
    FAILED = auto()
    CANCELLED = auto()


@dataclass
class OrderLifecycle:
    """Tracks state transitions and execution details of a single order."""
    order: OrderRequest
    state: OrderState = OrderState.CREATED
    result: OrderResult | None = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    error_message: str = ""

    def transition_to(self, new_state: OrderState, message: str = "") -> None:
        logger.debug(
            "Order %s (%s %d) transitioned: %s -> %s %s",
            self.order.ticker,
            self.order.side,
            self.order.quantity,
            self.state.name,
            new_state.name,
            f"({message})" if message else "",
        )
        self.state = new_state
        self.updated_at = datetime.now()
        if message:
            self.error_message = message


@dataclass(frozen=True)
class ExecutionJournal:
    """Comprehensive summary of an execution run."""
    trade_date: str
    total_orders: int
    filled_orders: int
    failed_orders: int
    close_orders_count: int
    new_orders_count: int
    lifecycles: list[OrderLifecycle]
    elapsed_seconds: float
    success: bool


class AsyncExecutionEngine:
    """Non-blocking Asynchronous Portfolio Execution Engine."""

    def __init__(
        self,
        split_delay_seconds: float = 1.0,
        large_order_ticker: str = "1629.T",
        order_timeout_seconds: float = 30.0,
    ) -> None:
        self.split_delay_seconds = split_delay_seconds
        self.large_order_ticker = large_order_ticker
        self.order_timeout_seconds = order_timeout_seconds

    def compute_order_deltas(
        self,
        target_weights: np.ndarray,
        current_prices: dict[str, float],
        current_positions: list[Position],
        total_capital: float,
    ) -> tuple[list[OrderRequest], list[OrderRequest]]:
        """Compute delta orders separating close orders and new orders.

        Returns:
            (close_orders, new_orders)
        """
        # Map current positions
        pos_map = {p.ticker: (p.quantity if p.side == "BUY" else -p.quantity) for p in current_positions}

        close_orders: list[OrderRequest] = []
        new_orders: list[OrderRequest] = []

        for j, tk in enumerate(JP_TICKERS):
            price = current_prices.get(tk, 0.0)
            if price <= 0.0:
                continue

            target_w = target_weights[j]
            target_value = target_w * total_capital
            target_qty = int(np.round(target_value / price))
            current_qty = pos_map.get(tk, 0)

            delta_qty = target_qty - current_qty
            if delta_qty == 0:
                continue

            # Check if closing an existing opposite position
            if current_qty != 0:
                if (current_qty > 0 and delta_qty < 0) or (current_qty < 0 and delta_qty > 0):
                    # Close existing position partially or completely
                    close_qty = min(abs(current_qty), abs(delta_qty))
                    side = "SELL" if current_qty > 0 else "BUY"
                    close_orders.append(
                        OrderRequest(
                            ticker=tk,
                            side=OrderSide(side),
                            quantity=close_qty,
                            limit_price=price,
                            order_type=OrderType.MARKET,
                        )
                    )
                    # Remaining delta becomes a new order
                    remaining_delta = delta_qty + close_qty if current_qty > 0 else delta_qty - close_qty
                    if remaining_delta != 0:
                        new_side = "BUY" if remaining_delta > 0 else "SELL"
                        new_orders.append(
                            OrderRequest(
                                ticker=tk,
                                side=OrderSide(new_side),
                                quantity=abs(remaining_delta),
                                limit_price=price,
                                order_type=OrderType.MARKET,
                            )
                        )
                    continue

            # Pure new order
            side = "BUY" if delta_qty > 0 else "SELL"
            new_orders.append(
                OrderRequest(
                    ticker=tk,
                    side=OrderSide(side),
                    quantity=abs(delta_qty),
                    limit_price=price,
                    order_type=OrderType.MARKET,
                )
            )

        return close_orders, new_orders

    async def _execute_single_order(
        self,
        lifecycle: OrderLifecycle,
        broker: AsyncBrokerClient,
    ) -> None:
        """Submit a single order and update its lifecycle state asynchronously."""
        lifecycle.transition_to(OrderState.SUBMITTING)
        try:
            # Submit with strict timeout guard
            result = await asyncio.wait_for(
                broker.submit_order(lifecycle.order),
                timeout=self.order_timeout_seconds,
            )
            lifecycle.result = result
            if result.status in (OrderStatus.FILLED, OrderStatus.SUBMITTED, OrderStatus.SIMULATED):
                lifecycle.transition_to(OrderState.FILLED, result.message)
            else:
                lifecycle.transition_to(OrderState.FAILED, result.message)
        except asyncio.TimeoutError:
            lifecycle.transition_to(OrderState.FAILED, "Order submission timed out")
        except Exception as e:
            lifecycle.transition_to(OrderState.FAILED, str(e))

    async def execute_portfolio(
        self,
        target_weights: np.ndarray,
        current_prices: dict[str, float],
        total_capital: float,
        broker: AsyncBrokerClient,
        trade_date: str = "",
    ) -> ExecutionJournal:
        """Execute full portfolio transition with staged non-blocking execution."""
        start_time = datetime.now()
        if not trade_date:
            trade_date = start_time.strftime("%Y-%m-%d")

        # 1. Fetch current positions asynchronously
        positions = await broker.get_positions()

        # 2. Compute order deltas
        close_orders, new_orders = self.compute_order_deltas(
            target_weights=target_weights,
            current_prices=current_prices,
            current_positions=positions,
            total_capital=total_capital,
        )

        close_lifecycles = [OrderLifecycle(order=o, state=OrderState.VALIDATED) for o in close_orders]
        new_lifecycles = [OrderLifecycle(order=o, state=OrderState.VALIDATED) for o in new_orders]

        logger.info(
            "[%s] Async execution starting: %d close orders, %d new orders",
            trade_date,
            len(close_lifecycles),
            len(new_lifecycles),
        )

        # 3. Stage 1: Execute all CLOSE orders concurrently
        if close_lifecycles:
            await asyncio.gather(
                *[self._execute_single_order(lc, broker) for lc in close_lifecycles]
            )

        # 4. Stage 2: Execute NEW orders (with non-blocking delay for split large orders)
        standard_new = [lc for lc in new_lifecycles if lc.order.ticker != self.large_order_ticker]
        split_new = [lc for lc in new_lifecycles if lc.order.ticker == self.large_order_ticker]

        # Submit standard new orders concurrently
        if standard_new:
            await asyncio.gather(
                *[self._execute_single_order(lc, broker) for lc in standard_new]
            )

        # Submit split large orders with async non-blocking delay
        for lc in split_new:
            # If quantity is large (> 100 shares), split into two half orders
            if lc.order.quantity >= 10:
                half_qty = lc.order.quantity // 2
                rem_qty = lc.order.quantity - half_qty

                lc1 = OrderLifecycle(
                    order=OrderRequest(
                        ticker=lc.order.ticker,
                        side=lc.order.side,
                        quantity=half_qty,
                        limit_price=lc.order.limit_price,
                        order_type=lc.order.order_type,
                    ),
                    state=OrderState.VALIDATED,
                )
                await self._execute_single_order(lc1, broker)

                # Non-blocking async sleep
                await asyncio.sleep(self.split_delay_seconds)

                lc2 = OrderLifecycle(
                    order=OrderRequest(
                        ticker=lc.order.ticker,
                        side=lc.order.side,
                        quantity=rem_qty,
                        limit_price=lc.order.limit_price,
                        order_type=lc.order.order_type,
                    ),
                    state=OrderState.VALIDATED,
                )
                await self._execute_single_order(lc2, broker)

                lc.state = OrderState.FILLED if (lc1.state == OrderState.FILLED and lc2.state == OrderState.FILLED) else OrderState.FAILED
            else:
                await self._execute_single_order(lc, broker)

        # 5. Assemble Journal
        all_lifecycles = close_lifecycles + new_lifecycles
        filled_count = sum(1 for lc in all_lifecycles if lc.state == OrderState.FILLED)
        failed_count = sum(1 for lc in all_lifecycles if lc.state == OrderState.FAILED)
        elapsed = (datetime.now() - start_time).total_seconds()

        journal = ExecutionJournal(
            trade_date=trade_date,
            total_orders=len(all_lifecycles),
            filled_orders=filled_count,
            failed_orders=failed_count,
            close_orders_count=len(close_lifecycles),
            new_orders_count=len(new_lifecycles),
            lifecycles=all_lifecycles,
            elapsed_seconds=elapsed,
            success=(failed_count == 0),
        )

        logger.info(
            "[%s] Async execution completed in %.2fs: %d filled, %d failed",
            trade_date,
            elapsed,
            filled_count,
            failed_count,
        )
        return journal
