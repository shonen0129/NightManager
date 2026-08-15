"""Async Broker Abstraction: base classes and simulator.

Provides non-blocking async interfaces for broker interactions using asyncio.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from leadlag.broker.base import BrokerConfig, Position, WalletInfo
from leadlag.core.types import OrderRequest, OrderResult, OrderStatus

logger = logging.getLogger(__name__)


class AsyncBrokerClient(ABC):
    """Abstract Base Class for asynchronous broker clients."""

    def __init__(self, config: BrokerConfig) -> None:
        self.config = config

    async def __aenter__(self) -> AsyncBrokerClient:
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.disconnect()

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection / session with broker."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection / logout."""
        ...

    @abstractmethod
    async def get_wallet(self) -> WalletInfo:
        """Fetch current wallet / margin balances asynchronously."""
        ...

    @abstractmethod
    async def get_positions(self) -> list[Position]:
        """Fetch current open positions asynchronously."""
        ...

    @abstractmethod
    async def submit_order(self, order: OrderRequest) -> OrderResult:
        """Submit a single order asynchronously."""
        ...

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order asynchronously."""
        ...


class AsyncDryRunBrokerClient(AsyncBrokerClient):
    """Asynchronous Dry-Run Simulator for testing execution flows.

    Simulates network latencies, fills, and wallet updates without real money.
    """

    def __init__(
        self,
        config: BrokerConfig | None = None,
        initial_cash: float = 10_000_000.0,
        simulated_latency_ms: float = 20.0,
    ) -> None:
        if config is None:
            config = BrokerConfig(provider="dry_run")
        super().__init__(config)
        self.cash = initial_cash
        self.positions: dict[str, Position] = {}
        self.submitted_orders: list[OrderResult] = []
        self.latency_sec = simulated_latency_ms / 1000.0
        self._connected = False

    async def connect(self) -> None:
        await asyncio.sleep(self.latency_sec)
        self._connected = True
        logger.info("AsyncDryRunBrokerClient connected with cash=%.2f", self.cash)

    async def disconnect(self) -> None:
        await asyncio.sleep(self.latency_sec)
        self._connected = False
        logger.info("AsyncDryRunBrokerClient disconnected")

    async def get_wallet(self) -> WalletInfo:
        await asyncio.sleep(self.latency_sec)
        return WalletInfo(cash_available=self.cash, margin_available=self.cash * 3.0)

    async def get_positions(self) -> list[Position]:
        await asyncio.sleep(self.latency_sec)
        return list(self.positions.values())

    async def submit_order(self, order: OrderRequest) -> OrderResult:
        """Simulate order execution with non-blocking async delay."""
        await asyncio.sleep(self.latency_sec)
        order_id = f"DRY_{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{order.ticker}"

        # Simulate fill
        fill_price = order.limit_price if (order.limit_price is not None and order.limit_price > 0) else 1000.0
        result = OrderResult(
            order_id=order_id,
            status=OrderStatus.FILLED,
            ticker=order.ticker,
            side=order.side,
            quantity=order.quantity,
            order_type=order.order_type,
            limit_price=fill_price,
            message="Simulated fill",
        )
        self.submitted_orders.append(result)

        # Update simulated positions with signed quantities (BUY=+, SELL=-)
        current_pos = self.positions.get(order.ticker)
        if current_pos is None:
            current_qty = 0
        else:
            current_qty = current_pos.quantity if current_pos.side == "BUY" else -current_pos.quantity

        signed_qty = order.quantity if str(order.side) == "BUY" else -order.quantity
        new_qty = current_qty + signed_qty

        if new_qty == 0:
            self.positions.pop(order.ticker, None)
        elif new_qty > 0:
            self.positions[order.ticker] = Position(
                ticker=order.ticker,
                side="BUY",
                quantity=abs(new_qty),
                price=fill_price,
            )
        else:
            self.positions[order.ticker] = Position(
                ticker=order.ticker,
                side="SELL",
                quantity=abs(new_qty),
                price=fill_price,
            )

        logger.info("Simulated order filled: %s %s x %d @ %.1f", str(order.side), order.ticker, order.quantity, fill_price)
        return result

    async def cancel_order(self, order_id: str) -> bool:
        await asyncio.sleep(self.latency_sec)
        return True
