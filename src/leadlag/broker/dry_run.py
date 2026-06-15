"""DryRunBrokerClient — simulated broker for testing and paper trading.

All order submissions are simulated locally without any network calls.
Position and wallet queries return sensible defaults or configured values.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from leadlag.broker.base import BrokerClient, BrokerConfig, Position, WalletInfo
from leadlag.core.types import (
    OrderRequest,
    OrderResult,
    OrderStatus,
)

logger = logging.getLogger(__name__)


class DryRunBrokerClient(BrokerClient):
    """Simulated broker client for testing and paper trading.

    All orders are logged and assigned simulated order IDs.
    No network calls are made.

    Usage::

        config = BrokerConfig(provider="dry_run")
        with DryRunBrokerClient(config) as client:
            result = client.submit_order(order)
            assert result.status == OrderStatus.SIMULATED
    """

    def __init__(
        self,
        config: BrokerConfig | None = None,
        *,
        simulated_cash: float = 1_000_000.0,
        simulated_positions: list[Position] | None = None,
        simulated_open_prices: dict[str, float] | None = None,
        simulated_us_returns: dict[str, float] | None = None,
    ) -> None:
        self._config = config or BrokerConfig(provider="dry_run")
        self._cash = simulated_cash
        self._positions = simulated_positions or []
        self._open_prices = simulated_open_prices or {}
        self._us_returns = simulated_us_returns or {}
        self._order_log: list[OrderResult] = []

    @property
    def order_log(self) -> list[OrderResult]:
        """Inspection of all simulated orders submitted."""
        return list(self._order_log)

    # -- health --------------------------------------------------------------

    def health_check(self) -> bool:
        logger.info("[DRY RUN] Health check: OK (simulated)")
        return True

    # -- wallet --------------------------------------------------------------

    def get_wallet(self) -> WalletInfo:
        return WalletInfo(
            cash_available=self._cash,
            margin_available=self._cash * 3.0,  # typical 3x leverage
        )

    # -- market data ---------------------------------------------------------

    def fetch_open_prices(
        self,
        tickers: list[str],
        *,
        allow_missing: bool = False,
    ) -> dict[str, float]:
        result = {}
        missing = []
        for tk in tickers:
            if tk in self._open_prices:
                result[tk] = self._open_prices[tk]
            else:
                missing.append(tk)

        if missing:
            if allow_missing:
                logger.warning(
                    "[DRY RUN] Missing simulated open prices for: %s",
                    ", ".join(missing),
                )
            else:
                raise ValueError(f"DryRunBrokerClient has no open prices for: {', '.join(missing)}")
        return result

    def fetch_us_etf_returns(
        self,
        us_tickers: list[str],
    ) -> dict[str, float]:
        result = {}
        for tk in us_tickers:
            if tk in self._us_returns:
                result[tk] = self._us_returns[tk]
            else:
                # Default: zero return
                result[tk] = 0.0
        return result

    # -- positions -----------------------------------------------------------

    def get_positions(self, **filters: Any) -> list[Position]:
        return list(self._positions)

    # -- order execution -----------------------------------------------------

    def submit_order(
        self,
        order: OrderRequest,
        *,
        is_close: bool = False,
        close_position_order: int = 0,
    ) -> OrderResult:
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        clean_ticker = order.ticker.replace(".T", "")
        order_id = f"SIM-{ts}-{clean_ticker}"

        result = OrderResult(
            order_id=order_id,
            status=OrderStatus.SIMULATED,
            ticker=order.ticker,
            side=order.side,
            quantity=order.quantity,
            order_type=order.order_type,
            limit_price=order.limit_price,
            message=f"Simulated {'close' if is_close else 'new'} order (dry run)",
        )

        logger.info(
            "[DRY RUN] %s %s %d shares of %s (ID: %s)",
            "CLOSE" if is_close else "NEW",
            order.side.value,
            order.quantity,
            order.ticker,
            order_id,
        )

        self._order_log.append(result)
        return result
