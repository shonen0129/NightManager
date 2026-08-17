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

import numpy as np

from leadlag.broker.async_base import AsyncBrokerClient
from leadlag.broker.base import Position
from leadlag.core.types import OrderRequest, OrderResult, OrderSide, OrderStatus, OrderType
from leadlag.data.tickers import JP_TICKERS, lot_size_for

logger = logging.getLogger(__name__)


class OrderState(Enum):
    """Lifecycle states of an execution order."""
    CREATED = auto()
    VALIDATED = auto()
    SUBMITTING = auto()
    SUBMITTED = auto()
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


def _align_to_lot(qty: int, lot: int) -> int:
    """Truncate ``qty`` to the nearest lot multiple towards zero.

    Python's ``//`` floors toward negative infinity, which over-orders on the
    short side. This helper preserves the sign and truncates the absolute
    quantity to a multiple of ``lot``.
    """
    if lot <= 1:
        return qty
    sign = 1 if qty >= 0 else -1
    return sign * (abs(qty) // lot) * lot


class AsyncRateLimiter:
    """Token-bucket asynchronous rate limiter for broker API guardrails.

    Prevents HTTP 429 Too Many Requests, session drops, and API overloads
    by enforcing strict requests-per-second and burst limits.
    """

    def __init__(self, rate_limit_per_second: float = 5.0, burst_limit: int = 5) -> None:
        self.rate_limit = max(0.1, rate_limit_per_second)
        self.burst_limit = max(1, burst_limit)
        self._tokens: float = float(self.burst_limit)
        self._last_update: float = 0.0
        self._lock: asyncio.Lock | None = None

    def _ensure_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def acquire(self) -> None:
        """Acquire a token, waiting asynchronously if rate limit is reached."""
        lock = self._ensure_lock()
        async with lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            if self._last_update == 0.0:
                self._last_update = now

            # Replenish tokens based on elapsed time
            elapsed = now - self._last_update
            self._tokens = min(self.burst_limit, self._tokens + elapsed * self.rate_limit)
            self._last_update = now

            if self._tokens < 1.0:
                # Need to wait for token replenishment
                wait_seconds = (1.0 - self._tokens) / self.rate_limit
                await asyncio.sleep(wait_seconds)
                # Recompute after sleep
                now_after = loop.time()
                self._tokens = min(self.burst_limit, self._tokens + (now_after - self._last_update) * self.rate_limit)
                self._last_update = now_after

            # Consume 1 token
            self._tokens = max(0.0, self._tokens - 1.0)


class AsyncExecutionEngine:
    """Non-blocking Asynchronous Portfolio Execution Engine."""

    def __init__(
        self,
        split_delay_seconds: float = 1.0,
        large_order_ticker: str = "1629.T",
        order_timeout_seconds: float = 30.0,
        split_threshold: int = 100,
        rate_limit_per_second: float = 5.0,
        burst_limit: int = 5,
        get_positions_timeout_seconds: float = 30.0,
        close_fill_timeout_seconds: float = 60.0,
        new_fill_timeout_seconds: float = 30.0,
    ) -> None:
        self.split_delay_seconds = split_delay_seconds
        self.large_order_ticker = large_order_ticker
        self.order_timeout_seconds = order_timeout_seconds
        self.split_threshold = split_threshold
        self.get_positions_timeout_seconds = get_positions_timeout_seconds
        self.close_fill_timeout_seconds = close_fill_timeout_seconds
        self.new_fill_timeout_seconds = new_fill_timeout_seconds
        self.rate_limiter = AsyncRateLimiter(
            rate_limit_per_second=rate_limit_per_second,
            burst_limit=burst_limit,
        )

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
        pos_meta = {p.ticker: p for p in current_positions}

        close_orders: list[OrderRequest] = []
        new_orders: list[OrderRequest] = []

        for j, tk in enumerate(JP_TICKERS):
            price = current_prices.get(tk, 0.0)
            if price <= 0.0:
                logger.warning("Missing or invalid price for %s, skipping order", tk)
                continue

            target_w = target_weights[j]
            target_value = target_w * total_capital
            lot = lot_size_for(tk)
            # Round half-up (matching core/allocator.py and broker_ops.py convention)
            # and then truncate to the nearest lot multiple toward zero.
            target_qty_raw = int(np.floor(target_value / price + 0.5))
            target_qty = _align_to_lot(target_qty_raw, lot)
            current_qty = pos_map.get(tk, 0)

            delta_qty = target_qty - current_qty
            if delta_qty == 0:
                continue

            # Check if closing an existing opposite position
            if current_qty != 0:
                if (current_qty > 0 and delta_qty < 0) or (current_qty < 0 and delta_qty > 0):
                    # Close existing position partially or completely
                    close_qty_raw = min(abs(current_qty), abs(delta_qty))
                    close_qty = _align_to_lot(close_qty_raw, lot)
                    if close_qty == 0:
                        close_qty = min(lot, abs(current_qty))
                    side = "SELL" if current_qty > 0 else "BUY"
                    pos = pos_meta.get(tk)
                    close_orders.append(
                        OrderRequest(
                            ticker=tk,
                            side=OrderSide(side),
                            quantity=close_qty,
                            limit_price=price,
                            order_type=OrderType.CLOSE,
                            margin_trade_type=pos.margin_trade_type if pos else None,
                            account_type=pos.account_type if pos else None,
                            is_close=True,
                        )
                    )
                    # Remaining delta becomes a new order
                    remaining_delta = delta_qty + close_qty if current_qty > 0 else delta_qty - close_qty
                    if remaining_delta != 0:
                        remaining_delta = _align_to_lot(remaining_delta, lot)
                        if remaining_delta == 0:
                            continue
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
            # Enforce API rate limiter token acquisition before network call
            await self.rate_limiter.acquire()

            # Submit with strict timeout guard
            result = await asyncio.wait_for(
                broker.submit_order(lifecycle.order),
                timeout=self.order_timeout_seconds,
            )
            lifecycle.result = result
            if result.status == OrderStatus.FILLED:
                lifecycle.transition_to(OrderState.FILLED, result.message)
            elif result.status == OrderStatus.SIMULATED:
                lifecycle.transition_to(OrderState.FILLED, result.message)
            elif result.status == OrderStatus.SUBMITTED:
                # Broker accepted the order but has not yet confirmed fill.
                # Do not treat as filled; downstream callers should poll/wait.
                lifecycle.transition_to(OrderState.SUBMITTED, result.message)
            else:
                lifecycle.transition_to(OrderState.FAILED, result.message)
        except TimeoutError:
            lifecycle.transition_to(OrderState.FAILED, "Order submission timed out")
        except Exception as e:
            lifecycle.transition_to(OrderState.FAILED, str(e))

    async def _wait_for_close_fills(
        self,
        lifecycles: list[OrderLifecycle],
        broker: AsyncBrokerClient,
        trade_date: str = "",
    ) -> None:
        """Poll close order lifecycles until they fill, fail, or time out.

        Close orders must reach a terminal state (FILLED, CANCELLED, or FAILED)
        before the NEW order stage can safely begin.  This prevents accidental
        double exposure when a close order is still SUBMITTED.
        """
        if not hasattr(broker, "get_order_status"):
            logger.debug("Broker does not support get_order_status; skipping close fill polling")
            return

        pending = [lc for lc in lifecycles if lc.state == OrderState.SUBMITTED]
        if not pending:
            return

        deadline = asyncio.get_running_loop().time() + self.close_fill_timeout_seconds
        poll_interval = 1.0

        logger.info(
            "[%s] Waiting for %d close order(s) to fill (timeout=%.1fs)...",
            trade_date,
            len(pending),
            self.close_fill_timeout_seconds,
        )

        while pending and asyncio.get_running_loop().time() < deadline:
            for lc in pending:
                if lc.result is None or not lc.result.order_id:
                    continue
                try:
                    # Enforce API rate limiter token acquisition before polling.
                    await self.rate_limiter.acquire()
                    result = await asyncio.wait_for(
                        broker.get_order_status(lc.result.order_id),
                        timeout=self.order_timeout_seconds,
                    )
                    if result in (OrderStatus.FILLED, OrderStatus.SIMULATED):
                        lc.transition_to(OrderState.FILLED, "Filled (polled)")
                    elif result == OrderStatus.CANCELLED:
                        lc.transition_to(OrderState.FAILED, "Close order was cancelled")
                    elif result == OrderStatus.FAILED:
                        lc.transition_to(OrderState.FAILED, "Close order failed")
                except TimeoutError:
                    pass
                except NotImplementedError:
                    logger.debug(
                        "Broker does not implement get_order_status; "
                        "stopping close fill polling"
                    )
                    return
                except Exception as e:
                    logger.warning(
                        "Failed to poll close order %s: %s",
                        lc.result.order_id,
                        e,
                    )

            pending = [lc for lc in lifecycles if lc.state == OrderState.SUBMITTED]
            if pending:
                await asyncio.sleep(poll_interval)

        if pending:
            logger.warning(
                "%d close order(s) still SUBMITTED after %.1fs; treating as failed",
                len(pending),
                self.close_fill_timeout_seconds,
            )
            for lc in pending:
                lc.transition_to(OrderState.FAILED, "Close fill confirmation timeout")

    async def _wait_for_new_fills(
        self,
        lifecycles: list[OrderLifecycle],
        broker: AsyncBrokerClient,
        trade_date: str = "",
    ) -> None:
        """Poll new order lifecycles until they fill, fail, or time out.

        New orders must reach a terminal state (FILLED or FAILED) before the
        execution journal can report success. This avoids treating an order
        that is merely accepted by the broker as an actual fill.
        """
        if not hasattr(broker, "get_order_status"):
            logger.debug("Broker does not support get_order_status; skipping new fill polling")
            return

        pending = [lc for lc in lifecycles if lc.state == OrderState.SUBMITTED]
        if not pending:
            return

        deadline = asyncio.get_running_loop().time() + self.new_fill_timeout_seconds
        poll_interval = 1.0

        logger.info(
            "[%s] Waiting for %d new order(s) to fill (timeout=%.1fs)...",
            trade_date,
            len(pending),
            self.new_fill_timeout_seconds,
        )

        while pending and asyncio.get_running_loop().time() < deadline:
            for lc in pending:
                if lc.result is None or not lc.result.order_id:
                    continue
                try:
                    # Enforce API rate limiter token acquisition before polling.
                    await self.rate_limiter.acquire()
                    result = await asyncio.wait_for(
                        broker.get_order_status(lc.result.order_id),
                        timeout=self.order_timeout_seconds,
                    )
                    if result in (OrderStatus.FILLED, OrderStatus.SIMULATED):
                        lc.transition_to(OrderState.FILLED, "Filled (polled)")
                    elif result == OrderStatus.CANCELLED:
                        lc.transition_to(OrderState.FAILED, "New order was cancelled")
                    elif result == OrderStatus.FAILED:
                        lc.transition_to(OrderState.FAILED, "New order failed")
                except TimeoutError:
                    pass
                except NotImplementedError:
                    logger.debug(
                        "Broker does not implement get_order_status; "
                        "stopping new fill polling"
                    )
                    return
                except Exception as e:
                    logger.warning(
                        "Failed to poll new order %s: %s",
                        lc.result.order_id,
                        e,
                    )

            pending = [lc for lc in lifecycles if lc.state == OrderState.SUBMITTED]
            if pending:
                await asyncio.sleep(poll_interval)

        if pending:
            logger.warning(
                "%d new order(s) still SUBMITTED after %.1fs; treating as failed",
                len(pending),
                self.new_fill_timeout_seconds,
            )
            for lc in pending:
                lc.transition_to(OrderState.FAILED, "New fill confirmation timeout")

    def _should_split(self, lifecycle: OrderLifecycle) -> bool:
        """Return True if an order should be split into delayed children."""
        if lifecycle.order.ticker != self.large_order_ticker:
            return False
        lot = lot_size_for(lifecycle.order.ticker)
        threshold = max(self.split_threshold, lot)
        return lifecycle.order.quantity >= threshold

    def _split_lifecycle(self, lc: OrderLifecycle) -> list[OrderLifecycle]:
        """Split a large lifecycle order into two lot-aligned children."""
        qty = lc.order.quantity
        lot = lot_size_for(lc.order.ticker)
        first_qty = (qty // 2 // lot) * lot
        second_qty = qty - first_qty
        if first_qty <= 0 or second_qty <= 0:
            return [lc]

        return [
            OrderLifecycle(
                order=OrderRequest(
                    ticker=lc.order.ticker,
                    side=lc.order.side,
                    quantity=first_qty,
                    limit_price=lc.order.limit_price,
                    order_type=lc.order.order_type,
                    margin_trade_type=lc.order.margin_trade_type,
                    account_type=lc.order.account_type,
                    is_close=lc.order.is_close,
                    close_position_order=lc.order.close_position_order,
                ),
                state=OrderState.VALIDATED,
            ),
            OrderLifecycle(
                order=OrderRequest(
                    ticker=lc.order.ticker,
                    side=lc.order.side,
                    quantity=second_qty,
                    limit_price=lc.order.limit_price,
                    order_type=lc.order.order_type,
                    margin_trade_type=lc.order.margin_trade_type,
                    account_type=lc.order.account_type,
                    is_close=lc.order.is_close,
                    close_position_order=lc.order.close_position_order,
                ),
                state=OrderState.VALIDATED,
            ),
        ]

    async def _execute_split_order(
        self,
        lc: OrderLifecycle,
        broker: AsyncBrokerClient,
    ) -> list[OrderLifecycle]:
        """Execute a large order as two children with a non-blocking delay between."""
        children = self._split_lifecycle(lc)
        if len(children) == 1:
            await self._execute_single_order(children[0], broker)
            return children

        await self._execute_single_order(children[0], broker)
        await asyncio.sleep(self.split_delay_seconds)
        await self._execute_single_order(children[1], broker)

        # Parent lifecycle reflects combined child state
        if all(c.state == OrderState.FILLED for c in children):
            lc.transition_to(OrderState.FILLED, "Split order fully filled")
        elif any(c.state == OrderState.FILLED for c in children):
            lc.transition_to(OrderState.FILLED, "Split order partially filled")
        else:
            lc.transition_to(OrderState.FAILED, "Split order failed")
        return children

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

        # 1. Fetch current positions asynchronously with timeout
        try:
            positions = await asyncio.wait_for(
                broker.get_positions(),
                timeout=self.get_positions_timeout_seconds,
            )
        except TimeoutError:
            logger.error("[%s] get_positions timed out; aborting execution.", trade_date)
            return ExecutionJournal(
                trade_date=trade_date,
                total_orders=0,
                filled_orders=0,
                failed_orders=0,
                close_orders_count=0,
                new_orders_count=0,
                lifecycles=[],
                elapsed_seconds=(datetime.now() - start_time).total_seconds(),
                success=False,
            )

        # 2. Compute order deltas
        close_orders, new_orders = self.compute_order_deltas(
            target_weights=target_weights,
            current_prices=current_prices,
            current_positions=positions,
            total_capital=total_capital,
        )

        # Ensure new orders carry the broker's default margin/account settings.
        default_margin = (
            broker.config.margin_trade_type
            if broker.config and broker.config.margin_trade_type is not None
            else None
        )
        default_account = (
            broker.config.account_type
            if broker.config and broker.config.account_type is not None
            else None
        )
        new_orders = [
            OrderRequest(
                ticker=o.ticker,
                side=o.side,
                quantity=o.quantity,
                order_type=o.order_type,
                limit_price=o.limit_price,
                margin_trade_type=default_margin,
                account_type=default_account,
                is_close=False,
            )
            for o in new_orders
        ]

        close_lifecycles = [OrderLifecycle(order=o, state=OrderState.VALIDATED) for o in close_orders]
        new_lifecycles = [OrderLifecycle(order=o, state=OrderState.VALIDATED) for o in new_orders]

        logger.info(
            "[%s] Async execution starting: %d close orders, %d new orders",
            trade_date,
            len(close_lifecycles),
            len(new_lifecycles),
        )

        # 3. Stage 1: Execute all CLOSE orders concurrently (with split handling)
        close_split_children = []
        standard_close = [lc for lc in close_lifecycles if not self._should_split(lc)]
        split_close = [lc for lc in close_lifecycles if self._should_split(lc)]

        if standard_close:
            await asyncio.gather(
                *[self._execute_single_order(lc, broker) for lc in standard_close]
            )
        for lc in split_close:
            close_split_children.extend(await self._execute_split_order(lc, broker))

        close_lifecycles_executed = standard_close + close_split_children

        # Wait for close orders to fill before opening new positions.
        await self._wait_for_close_fills(close_lifecycles_executed, broker, trade_date)

        close_phase_failed = any(
            lc.state != OrderState.FILLED
            for lc in close_lifecycles_executed
        )
        if close_phase_failed:
            logger.error(
                "[%s] Close phase had non-FILLED orders; aborting NEW order stage.",
                trade_date,
            )
            # Do NOT proceed to new orders; existing positions remain.
            all_lifecycles = close_lifecycles_executed
            elapsed = (datetime.now() - start_time).total_seconds()
            filled_count = sum(1 for lc in all_lifecycles if lc.state == OrderState.FILLED)
            failed_count = sum(1 for lc in all_lifecycles if lc.state == OrderState.FAILED)

            journal = ExecutionJournal(
                trade_date=trade_date,
                total_orders=len(all_lifecycles),
                filled_orders=filled_count,
                failed_orders=failed_count,
                close_orders_count=len(standard_close) + len(close_split_children),
                new_orders_count=0,
                lifecycles=all_lifecycles,
                elapsed_seconds=elapsed,
                success=all(lc.state == OrderState.FILLED for lc in all_lifecycles),
            )
            logger.info(
                "[%s] Async execution completed in %.2fs: %d filled, %d failed (NEW stage aborted).",
                trade_date,
                elapsed,
                filled_count,
                failed_count,
            )
            return journal

        # 4. Stage 2: Execute NEW orders (with non-blocking delay for split large orders)
        new_split_children = []
        standard_new = [lc for lc in new_lifecycles if not self._should_split(lc)]
        split_new = [lc for lc in new_lifecycles if self._should_split(lc)]

        # Submit standard new orders concurrently
        if standard_new:
            await asyncio.gather(
                *[self._execute_single_order(lc, broker) for lc in standard_new]
            )

        # Submit split large orders with async non-blocking delay
        for lc in split_new:
            new_split_children.extend(await self._execute_split_order(lc, broker))

        # Wait for new orders to fill before reporting success.
        new_lifecycles_executed = standard_new + new_split_children
        await self._wait_for_new_fills(new_lifecycles_executed, broker, trade_date)

        # 5. Assemble Journal
        all_lifecycles = (
            close_lifecycles_executed
            + standard_new
            + new_split_children
        )
        filled_count = sum(1 for lc in all_lifecycles if lc.state == OrderState.FILLED)
        failed_count = sum(1 for lc in all_lifecycles if lc.state == OrderState.FAILED)
        elapsed = (datetime.now() - start_time).total_seconds()

        journal = ExecutionJournal(
            trade_date=trade_date,
            total_orders=len(all_lifecycles),
            filled_orders=filled_count,
            failed_orders=failed_count,
            close_orders_count=len(standard_close) + len(close_split_children),
            new_orders_count=len(standard_new) + len(new_split_children),
            lifecycles=all_lifecycles,
            elapsed_seconds=elapsed,
            success=all(lc.state == OrderState.FILLED for lc in all_lifecycles),
        )

        logger.info(
            "[%s] Async execution completed in %.2fs: %d filled, %d failed",
            trade_date,
            elapsed,
            filled_count,
            failed_count,
        )
        return journal

    async def close_all_positions(
        self,
        broker: AsyncBrokerClient,
        trade_date: str = "",
        close_position_order: int = 0,
    ) -> ExecutionJournal:
        """Close all open positions with async parallel execution and rate limiting."""
        start_time = datetime.now()
        if not trade_date:
            trade_date = start_time.strftime("%Y-%m-%d")

        # 1. Fetch open positions
        try:
            positions = await asyncio.wait_for(
                broker.get_positions(),
                timeout=self.get_positions_timeout_seconds,
            )
        except TimeoutError:
            logger.error("[%s] get_positions timed out during close_all_positions; aborting.", trade_date)
            return ExecutionJournal(
                trade_date=trade_date,
                total_orders=0,
                filled_orders=0,
                failed_orders=0,
                close_orders_count=0,
                new_orders_count=0,
                lifecycles=[],
                elapsed_seconds=(datetime.now() - start_time).total_seconds(),
                success=False,
            )

        if not positions:
            logger.info("[%s] No open positions to close.", trade_date)
            return ExecutionJournal(
                trade_date=trade_date,
                total_orders=0,
                filled_orders=0,
                failed_orders=0,
                close_orders_count=0,
                new_orders_count=0,
                lifecycles=[],
                elapsed_seconds=(datetime.now() - start_time).total_seconds(),
                success=True,
            )

        # 2. Build closing orders for each open position
        close_orders: list[OrderRequest] = []
        for p in positions:
            if p.quantity <= 0:
                continue
            side = "SELL" if p.side.upper() in ("BUY", "LONG") else "BUY"
            close_orders.append(
                OrderRequest(
                    ticker=p.ticker,
                    side=OrderSide(side),
                    quantity=p.quantity,
                    limit_price=p.price if p.price > 0 else 0.0,
                    order_type=OrderType.CLOSE,
                    margin_trade_type=p.margin_trade_type,
                    account_type=p.account_type,
                    is_close=True,
                    close_position_order=close_position_order,
                )
            )

        close_lifecycles = [OrderLifecycle(order=o, state=OrderState.VALIDATED) for o in close_orders]
        logger.info(
            "[%s] Closing %d open positions asynchronously (rate limited + split guard)...",
            trade_date,
            len(close_lifecycles),
        )

        # 3. Execute closing orders (with split handling for large/thin tickers like 1629.T)
        standard_close = [lc for lc in close_lifecycles if not self._should_split(lc)]
        split_close = [lc for lc in close_lifecycles if self._should_split(lc)]
        close_split_children = []

        if standard_close:
            await asyncio.gather(
                *[self._execute_single_order(lc, broker) for lc in standard_close]
            )
        for lc in split_close:
            close_split_children.extend(await self._execute_split_order(lc, broker))

        all_lifecycles = standard_close + close_split_children

        # Wait for close orders to fill before reporting success.
        await self._wait_for_close_fills(all_lifecycles, broker, trade_date)

        filled_count = sum(1 for lc in all_lifecycles if lc.state == OrderState.FILLED)
        failed_count = sum(1 for lc in all_lifecycles if lc.state == OrderState.FAILED)
        elapsed = (datetime.now() - start_time).total_seconds()

        journal = ExecutionJournal(
            trade_date=trade_date,
            total_orders=len(all_lifecycles),
            filled_orders=filled_count,
            failed_orders=failed_count,
            close_orders_count=len(all_lifecycles),
            new_orders_count=0,
            lifecycles=all_lifecycles,
            elapsed_seconds=elapsed,
            success=all(lc.state == OrderState.FILLED for lc in all_lifecycles),
        )

        logger.info(
            "[%s] close_all_positions completed in %.2fs: %d filled, %d failed",
            trade_date,
            elapsed,
            filled_count,
            failed_count,
        )
        return journal
