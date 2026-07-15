"""Broker abstraction: base classes and data models.

This module defines the BrokerClient ABC and associated data models
that all broker implementations must conform to. The abstraction
decouples the strategy engine from any specific broker's API.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from leadlag.core.types import OrderRequest, OrderResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BrokerConfig:
    """Broker connection configuration.

    Attributes:
        provider: Broker identifier ("kabu", "sbi", "dry_run", etc.)
        api_url: API base URL
        api_token: Authentication token (may be empty if password-based auth)
        api_password: Password for token issuance (optional)
        request_timeout: HTTP request timeout in seconds
        margin_trade_type: 1=制度信用, 2=一般信用(長期), 3=一般信用(デイトレ)
        account_type: 2=一般, 4=特定, 12=法人
        extra: Provider-specific additional settings
    """

    provider: str = "kabu"
    api_url: str = ""
    api_token: str = ""
    api_password: str = ""
    request_timeout: int = 10
    margin_trade_type: int = 3
    account_type: int = 4
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WalletInfo:
    """Broker wallet / account balance information.

    Attributes:
        cash_available: 現物買付可能額
        margin_available: 信用新規可能額 (0 if not a margin account)
        extra: Provider-specific fields
    """

    cash_available: float = 0.0
    margin_available: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Position:
    """An open position returned by the broker.

    This is a broker-neutral representation. Provider-specific fields
    are stored in ``extra``.

    Attributes:
        ticker: Ticker in yfinance format (e.g. "1617.T")
        side: "BUY" or "SELL"
        quantity: Number of shares (LeavesQty equivalent)
        price: Entry / average price
        exchange: Exchange code (broker-specific)
        execution_id: Execution or position ID
        margin_trade_type: Margin type if applicable
        account_type: Account type code
        extra: Additional provider-specific data
    """

    ticker: str
    side: str
    quantity: int
    price: float = 0.0
    exchange: int = 0
    execution_id: str = ""
    margin_trade_type: int | None = None
    account_type: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class BrokerClient(ABC):
    """Abstract broker client interface.

    All broker implementations (kabuステーション, SBI, 楽天, IB, dry-run, …)
    must implement this interface. The strategy engine interacts with
    brokers **exclusively** through this ABC.

    Context-manager protocol is supported for resource cleanup::

        with create_broker(config) as client:
            client.submit_order(...)
    """

    def __enter__(self) -> BrokerClient:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False

    # -- Lifecycle -----------------------------------------------------------

    def close(self) -> None:
        """Release resources (HTTP sessions, file handles, etc.)."""

    # -- Health --------------------------------------------------------------

    @abstractmethod
    def health_check(self) -> bool:
        """Check API connectivity and authentication.

        Returns:
            True if the broker API is reachable and authenticated.
        """

    # -- Wallet / Account ----------------------------------------------------

    @abstractmethod
    def get_wallet(self) -> WalletInfo:
        """Retrieve account balance information.

        Returns:
            WalletInfo with cash and margin balances.
        """

    # -- Market Data ---------------------------------------------------------

    @abstractmethod
    def fetch_open_prices(
        self,
        tickers: list[str],
        *,
        allow_missing: bool = False,
    ) -> dict[str, float]:
        """Fetch today's opening prices for the given tickers.

        Args:
            tickers: List of tickers in yfinance format (e.g. "1617.T")
            allow_missing: If True, return partial results instead of raising

        Returns:
            Dict mapping ticker → opening price
        """

    @abstractmethod
    def fetch_us_etf_returns(
        self,
        us_tickers: list[str],
    ) -> dict[str, float]:
        """Fetch US ETF close-to-close returns.

        Args:
            us_tickers: List of US ETF ticker symbols (e.g. "XLB")

        Returns:
            Dict mapping ticker → close-to-close return
        """

    @abstractmethod
    def fetch_current_prices(
        self,
        tickers: list[str],
        *,
        allow_missing: bool = False,
    ) -> dict[str, float]:
        """Fetch current real-time prices for the given tickers.

        Args:
            tickers: List of tickers in yfinance format (e.g. "1617.T")
            allow_missing: If True, return partial results instead of raising

        Returns:
            Dict mapping ticker → current price
        """

    # -- Positions -----------------------------------------------------------

    @abstractmethod
    def get_positions(
        self,
        **filters: Any,
    ) -> list[Position]:
        """Fetch current open positions.

        Args:
            **filters: Provider-specific filters (product, symbol, side, …)

        Returns:
            List of Position objects
        """

    # -- Order Execution -----------------------------------------------------

    @abstractmethod
    def submit_order(
        self,
        order: OrderRequest,
        *,
        is_close: bool = False,
        close_position_order: int = 0,
    ) -> OrderResult:
        """Submit a single order.

        Args:
            order: The order to submit
            is_close: True for position-closing orders (信用返済)
            close_position_order: Close priority (0-7)

        Returns:
            OrderResult with status and order ID
        """

    def submit_orders_batch(
        self,
        orders: list[OrderRequest],
        *,
        delay_ms: int = 50,
        is_close: bool = False,
        close_position_order: int = 0,
    ) -> list[OrderResult]:
        """Submit multiple orders with delay between each.

        Default implementation calls submit_order in a loop.
        Subclasses may override for provider-specific batching.

        Args:
            orders: List of OrderRequest objects
            delay_ms: Delay between orders in milliseconds
            is_close: True for closing orders
            close_position_order: Close priority (0-7)

        Returns:
            List of OrderResult objects
        """
        import time

        results: list[OrderResult] = []
        for i, order in enumerate(orders):
            if i > 0:
                time.sleep(delay_ms / 1000.0)
            result = self.submit_order(
                order,
                is_close=is_close,
                close_position_order=close_position_order,
            )
            results.append(result)
        return results

    # -- Position Closing ----------------------------------------------------

    def close_all_positions(
        self,
        *,
        dry_run: bool = False,
        order_type: str = "CLO",
        close_position_order: int = 0,
    ) -> list[OrderResult]:
        """Close all open margin positions.

        Default implementation fetches positions and submits opposite-side
        close orders. Subclasses may override for provider-specific behavior.

        Args:
            dry_run: Simulate without submitting
            order_type: Order type for close (default: CLO = 引成後場)
            close_position_order: Close priority (0-7)

        Returns:
            List of OrderResult for close orders
        """
        from leadlag.core.types import OrderSide, OrderStatus, OrderType

        positions = self.get_positions()
        if not positions:
            logger.info("No open positions to close")
            return []

        # Build close orders (opposite side)
        close_orders: list[OrderRequest] = []
        for pos in positions:
            if pos.quantity <= 0:
                continue
            close_side = OrderSide.SELL if pos.side == "BUY" else OrderSide.BUY
            ot = OrderType(order_type) if order_type in ("MO", "LO", "CLO") else OrderType.CLOSE
            close_orders.append(
                OrderRequest(
                    ticker=pos.ticker,
                    side=close_side,
                    quantity=pos.quantity,
                    order_type=ot,
                )
            )

        if dry_run:
            from datetime import datetime

            results = []
            for order in close_orders:
                results.append(
                    OrderResult(
                        order_id=f"SIM-CLOSE-{datetime.now().strftime('%Y%m%d%H%M%S')}-{order.ticker}",
                        status=OrderStatus.SIMULATED,
                        ticker=order.ticker,
                        side=order.side,
                        quantity=order.quantity,
                        order_type=order.order_type,
                        message="Simulated close (dry run)",
                    )
                )
            return results

        return self.submit_orders_batch(
            close_orders,
            is_close=True,
            close_position_order=close_position_order,
        )
