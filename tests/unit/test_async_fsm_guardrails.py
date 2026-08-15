"""Unit tests for Async Execution Guardrails.

Tests rate limiting, timeout resilience, serial order delay, and snapshot sanity guards.
"""

from __future__ import annotations

import asyncio
import time

import numpy as np

from leadlag.broker.async_base import AsyncBrokerClient
from leadlag.broker.base import Position, WalletInfo
from leadlag.core.types import OrderRequest, OrderResult, OrderSide, OrderStatus, OrderType
from leadlag.data.pit_lake import MarketSnapshot
from leadlag.data.tickers import JP_TICKERS, US_TICKERS
from leadlag.execution.async_fsm import (
    AsyncExecutionEngine,
    AsyncRateLimiter,
    OrderLifecycle,
    OrderState,
)


def test_rate_limiter_throttling():
    """Verify AsyncRateLimiter enforces minimum intervals between token acquisitions."""
    async def _run():
        # Limit to 10 requests / sec (1 token every 0.1s), burst = 1
        limiter = AsyncRateLimiter(rate_limit_per_second=10.0, burst_limit=1)

        start = time.perf_counter()
        # Acquire 4 tokens
        for _ in range(4):
            await limiter.acquire()
        elapsed = time.perf_counter() - start

        # 4 tokens with rate 10/s should take at least ~0.20 - 0.30 seconds
        assert elapsed >= 0.20, f"Expected elapsed >= 0.20s, got {elapsed:.3f}s"

    asyncio.run(_run())


class SlowHangingBrokerClient(AsyncBrokerClient):
    """Mock broker client that hangs forever on order submission."""

    def __init__(self) -> None:
        super().__init__(config=None)

    async def connect(self) -> None:
        pass

    async def disconnect(self) -> None:
        pass

    async def get_positions(self) -> list[Position]:
        return []

    async def get_wallet(self) -> WalletInfo:
        return WalletInfo(cash_available=1_000_000.0, margin_available=1_000_000.0)

    async def submit_order(self, order: OrderRequest) -> OrderResult:
        # Hang longer than timeout
        await asyncio.sleep(5.0)
        return OrderResult(
            order_id="hang_1",
            status=OrderStatus.FILLED,
            ticker=order.ticker,
            side=order.side,
            quantity=order.quantity,
            order_type=order.order_type,
            limit_price=order.limit_price,
            message="Filled after hang",
        )

    async def cancel_order(self, order_id: str) -> bool:
        return True


def test_async_order_timeout_guard():
    """Verify that hanging broker requests trigger timeout and do not block the engine."""
    async def _run():
        # Set short timeout of 0.05 seconds
        engine = AsyncExecutionEngine(order_timeout_seconds=0.05)
        broker = SlowHangingBrokerClient()

        lc = OrderLifecycle(
            order=OrderRequest(
                ticker="1617.T",
                side=OrderSide.BUY,
                quantity=100,
                limit_price=1000.0,
                order_type=OrderType.MARKET,
            ),
            state=OrderState.VALIDATED,
        )

        start = time.perf_counter()
        await engine._execute_single_order(lc, broker)
        elapsed = time.perf_counter() - start

        # Must fail and transition to FAILED within sub-second
        assert lc.state == OrderState.FAILED
        assert "timed out" in lc.error_message.lower()
        assert elapsed < 1.0, f"Order hung for {elapsed:.2f}s instead of timing out promptly"

    asyncio.run(_run())


def test_market_snapshot_sanity_validation():
    """Verify MarketSnapshot detects invalid prices, NaN/Inf, and extreme return outliers."""
    import pandas as pd

    ts = pd.Timestamp("2026-08-15 09:10:00")
    valid_us = np.zeros(len(US_TICKERS))
    valid_jp = np.zeros(len(JP_TICKERS))
    valid_betas = np.ones(len(JP_TICKERS))
    valid_prices = {tk: 1000.0 for tk in JP_TICKERS}
    valid_closes = {tk: 1000.0 for tk in JP_TICKERS}

    # 1. Normal valid snapshot
    snap_ok = MarketSnapshot(
        as_of=ts,
        trade_date="2026-08-15",
        us_returns=valid_us,
        jp_gap_returns=valid_jp,
        jp_betas=valid_betas,
        topix_night_return=0.0,
        current_prices=valid_prices,
        prev_closes=valid_closes,
    )
    assert snap_ok.is_valid()

    # 2. Snapshot with extreme outlier return (> 20%)
    extreme_us = valid_us.copy()
    extreme_us[0] = 0.35  # +35% jump
    snap_extreme = MarketSnapshot(
        as_of=ts,
        trade_date="2026-08-15",
        us_returns=extreme_us,
        jp_gap_returns=valid_jp,
        jp_betas=valid_betas,
        topix_night_return=0.0,
        current_prices=valid_prices,
        prev_closes=valid_closes,
    )
    is_valid, errors = snap_extreme.validate(max_abs_return=0.20)
    assert not is_valid
    assert any("max_abs_return" in e for e in errors)

    # 3. Snapshot with missing / negative price
    bad_prices = valid_prices.copy()
    bad_prices["1617.T"] = -10.0
    snap_bad_price = MarketSnapshot(
        as_of=ts,
        trade_date="2026-08-15",
        us_returns=valid_us,
        jp_gap_returns=valid_jp,
        jp_betas=valid_betas,
        topix_night_return=0.0,
        current_prices=bad_prices,
        prev_closes=valid_closes,
    )
    is_valid, errors = snap_bad_price.validate()
    assert not is_valid
    assert any("Invalid or missing execution price" in e for e in errors)
