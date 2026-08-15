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
        """Submit a single order asynchronously.

        Implementations are expected to either block internally until the order
        is filled or return SUBMITTED and support get_order_status polling.
        """
        ...

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order asynchronously."""
        ...


class AsyncThreadedBrokerClient(AsyncBrokerClient):
    """Async wrapper that runs an existing synchronous BrokerClient in a thread pool.

    This lets production-grade sync broker adapters (Kabu, Tachibana, dry-run)
    be used by the async execution FSM without rewriting their internals.
    """

    def __init__(
        self,
        sync_client: Any,
        broker_request_timeout: float = 30.0,
    ) -> None:
        # Accept either a BrokerConfig or a BrokerClient instance.
        from leadlag.broker.base import BrokerClient

        self._timeout = broker_request_timeout
        if isinstance(sync_client, BrokerClient):
            self._client = sync_client
            self._config = sync_client.config if hasattr(sync_client, "config") else BrokerConfig()
        else:
            self._config = sync_client
            from leadlag.broker import create_broker

            self._client = create_broker(self._config)
        super().__init__(self._config)

    async def connect(self) -> None:
        # No async connection handshake for threaded clients by default.
        pass

    async def disconnect(self) -> None:
        await asyncio.to_thread(self._client.close)

    async def get_wallet(self) -> WalletInfo:
        return await asyncio.wait_for(
            asyncio.to_thread(self._client.get_wallet),
            timeout=self._timeout,
        )

    async def get_positions(self) -> list[Position]:
        return await asyncio.wait_for(
            asyncio.to_thread(self._client.get_positions),
            timeout=self._timeout,
        )

    async def _wait_for_fill(self, order_result: OrderResult) -> OrderResult:
        """Optionally poll the underlying client until the order is filled.

        If get_order_status is not implemented, return the original SUBMITTED
        result and let the FSM handle it.
        """
        if order_result.status != OrderStatus.SUBMITTED or not order_result.order_id:
            return order_result

        if not hasattr(self._client, "get_order_status"):
            return order_result

        deadline = asyncio.get_running_loop().time() + self._timeout
        while asyncio.get_running_loop().time() < deadline:
            try:
                status = await asyncio.to_thread(
                    self._client.get_order_status,
                    order_result.order_id,
                )
                if status == OrderStatus.FILLED:
                    return OrderResult(
                        order_id=order_result.order_id,
                        status=OrderStatus.FILLED,
                        ticker=order_result.ticker,
                        side=order_result.side,
                        quantity=order_result.quantity,
                        order_type=order_result.order_type,
                        limit_price=order_result.limit_price,
                        margin_trade_type=order_result.margin_trade_type,
                        eigyou_day=order_result.eigyou_day,
                        message="Filled (polled)",
                    )
                if status in (OrderStatus.CANCELLED, OrderStatus.FAILED):
                    return OrderResult(
                        order_id=order_result.order_id,
                        status=status,
                        ticker=order_result.ticker,
                        side=order_result.side,
                        quantity=order_result.quantity,
                        order_type=order_result.order_type,
                        limit_price=order_result.limit_price,
                        margin_trade_type=order_result.margin_trade_type,
                        eigyou_day=order_result.eigyou_day,
                        message=f"Order ended as {status.value}",
                    )
            except Exception as e:
                logger.warning(
                    "Failed to poll order status for %s: %s",
                    order_result.order_id,
                    e,
                )
            await asyncio.sleep(1.0)

        # Timed out without fill confirmation.
        return order_result

    async def submit_order(self, order: OrderRequest) -> OrderResult:
        result = await asyncio.wait_for(
            asyncio.to_thread(self._client.submit_order, order),
            timeout=self._timeout,
        )
        if result.status == OrderStatus.SUBMITTED:
            return await self._wait_for_fill(result)
        return result

    async def cancel_order(self, order_id: str) -> bool:
        if not hasattr(self._client, "cancel_order"):
            return False
        return await asyncio.wait_for(
            asyncio.to_thread(self._client.cancel_order, order_id),
            timeout=self._timeout,
        )


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
