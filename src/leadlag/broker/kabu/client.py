"""KabuBrokerClient — BrokerClient adapter for kabuステーション API.

This module wraps the existing `leadlag.broker.kabu.api.KabuClient` behind the
broker-neutral ``BrokerClient`` ABC so that the strategy engine and
production runner never import kabu-specific code directly.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from leadlag.broker.base import BrokerClient, BrokerConfig, Position, WalletInfo
from leadlag.broker.kabu.api import KabuApiError, KabuClient, KabuConfig, issue_api_token
from leadlag.core.types import (
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
)

logger = logging.getLogger(__name__)


class KabuBrokerClient(BrokerClient):
    """BrokerClient implementation backed by kabuステーション API.

    Wraps ``KabuClient`` and translates between the
    broker-neutral interface and kabu-specific types.

    Usage::

        config = BrokerConfig(provider="kabu", api_url="...", api_token="...")
        with KabuBrokerClient(config) as client:
            client.submit_order(order)
    """

    def __init__(self, config: BrokerConfig) -> None:
        self._broker_config = config
        self._kabu_config: KabuConfig | None = None
        self._client: KabuClient | None = None
        self._connect(config)

    # -- internal helpers ----------------------------------------------------

    def _connect(self, config: BrokerConfig) -> None:
        """Build and validate the internal KabuClient."""
        api_url = config.api_url
        api_token = config.api_token
        timeout = config.request_timeout

        # Auto-issue token if not provided
        if not api_token:
            api_password = config.api_password or os.environ.get("KABU_API_PASSWORD", "")
            if api_password:
                logger.info("[KabuBroker] No API token provided. Issuing via /token...")
                api_token = issue_api_token(api_url, api_password, timeout)
                os.environ["KABU_API_TOKEN"] = api_token
                logger.info("[KabuBroker] Token issued successfully")
            else:
                raise ValueError(
                    "KabuBrokerClient requires either api_token or "
                    "api_password (KABU_API_PASSWORD) to authenticate."
                )

        self._kabu_config = KabuConfig(
            api_url=api_url,
            api_token=api_token,
            request_timeout=timeout,
        )
        self._client = KabuClient(self._kabu_config)

    @property
    def inner(self) -> KabuClient:
        """Access the underlying KabuClient (escape hatch for kabu-specific features)."""
        if self._client is None:
            raise RuntimeError("KabuBrokerClient has been closed")
        return self._client

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
            logger.debug("KabuBrokerClient closed")

    # -- health --------------------------------------------------------------

    def health_check(self) -> bool:
        try:
            return self.inner.health_check()
        except Exception as e:
            logger.error("Health check failed: %s", e)
            return False

    # -- wallet --------------------------------------------------------------

    def get_wallet(self) -> WalletInfo:
        cash = self.inner.get_cash_wallet()
        try:
            margin = self.inner.get_margin_wallet()
        except Exception:
            margin = {}

        return WalletInfo(
            cash_available=float(cash.get("cash_available", 0.0)),
            margin_available=float(margin.get("margin_available", 0.0)),
            extra={
                "au_kc_cash_available": cash.get("au_kc_cash_available", 0.0),
                "au_jbn_cash_available": cash.get("au_jbn_cash_available", 0.0),
                "deposit_keep_rate": margin.get("deposit_keep_rate", 0.0),
            },
        )

    # -- market data ---------------------------------------------------------

    def fetch_open_prices(
        self,
        tickers: list[str],
        *,
        allow_missing: bool = False,
    ) -> dict[str, float]:
        return self.inner.fetch_jp_open_prices(
            tickers,
            allow_missing=allow_missing,
        )

    def fetch_us_etf_returns(
        self,
        us_tickers: list[str],
    ) -> dict[str, float]:
        return self.inner.fetch_us_etf_returns(us_tickers)

    # -- positions -----------------------------------------------------------

    def get_positions(self, **filters: Any) -> list[Position]:
        raw = self.inner.get_positions(
            product=filters.get("product", 0),
            symbol=filters.get("symbol"),
            side=filters.get("side"),
        )
        positions = []
        for pos in raw:
            # Convert ticker from kabu code to yfinance format if needed
            ticker_raw = str(pos.get("ticker", ""))
            if ticker_raw and not ticker_raw.endswith(".T") and ticker_raw.isdigit():
                ticker = f"{ticker_raw}.T"
            else:
                ticker = ticker_raw

            positions.append(
                Position(
                    ticker=ticker,
                    side=str(pos.get("side", "")),
                    quantity=int(pos.get("quantity", 0)),
                    price=float(pos.get("price", 0)),
                    exchange=int(pos.get("exchange", 0)),
                    execution_id=str(pos.get("execution_id", "")),
                    margin_trade_type=pos.get("margin_trade_type"),
                    account_type=pos.get("account_type"),
                    extra={
                        "symbol_name": pos.get("symbol_name", ""),
                        "side_raw": pos.get("side_raw", ""),
                        "hold_qty": pos.get("hold_qty", 0),
                        "leaves_qty": pos.get("leaves_qty", 0),
                        "current_price": pos.get("current_price"),
                        "valuation": pos.get("valuation"),
                        "profit_loss": pos.get("profit_loss"),
                        "profit_loss_rate": pos.get("profit_loss_rate"),
                        "expire_day": pos.get("expire_day"),
                        "execution_day": pos.get("execution_day"),
                    },
                )
            )
        return positions

    # -- order execution -----------------------------------------------------

    def submit_order(
        self,
        order: OrderRequest,
        *,
        is_close: bool = False,
        close_position_order: int = 0,
    ) -> OrderResult:
        clean_ticker = order.ticker.replace(".T", "")

        ot_map = {
            OrderType.MARKET: "MO",
            OrderType.LIMIT: "LO",
            OrderType.CLOSE: "CLO",
        }
        order_type_str = ot_map.get(order.order_type, "MO")

        try:
            result = self.inner.send_order(
                ticker=clean_ticker,
                side=order.side.value,
                quantity=order.quantity,
                order_type=order_type_str,
                limit_price=order.limit_price,
                margin_trade_type=self._broker_config.margin_trade_type,
                account_type=self._broker_config.account_type,
                is_close=is_close,
                close_position_order=close_position_order,
            )

            return OrderResult(
                order_id=str(result.get("order_id", "")),
                status=OrderStatus.SUBMITTED,
                ticker=order.ticker,
                side=order.side,
                quantity=order.quantity,
                order_type=order.order_type,
                limit_price=order.limit_price,
                margin_trade_type=result.get(
                    "margin_trade_type",
                    self._broker_config.margin_trade_type,
                ),
            )

        except (KabuApiError, ValueError) as e:
            logger.error("Order failed for %s: %s", order.ticker, e)
            return OrderResult(
                order_id="",
                status=OrderStatus.FAILED,
                ticker=order.ticker,
                side=order.side,
                quantity=order.quantity,
                order_type=order.order_type,
                limit_price=order.limit_price,
                message=str(e),
            )

    def submit_orders_batch(
        self,
        orders: list[OrderRequest],
        *,
        delay_ms: int = 50,
        is_close: bool = False,
        close_position_order: int = 0,
    ) -> list[OrderResult]:
        """Override to delegate to KabuClient's rollback-aware batch method."""
        if not orders:
            return []

        order_dicts = []
        for o in orders:
            clean_ticker = o.ticker.replace(".T", "")
            ot_map = {OrderType.MARKET: "MO", OrderType.LIMIT: "LO", OrderType.CLOSE: "CLO"}
            order_dicts.append(
                {
                    "ticker": clean_ticker,
                    "side": o.side.value,
                    "quantity": o.quantity,
                    "order_type": ot_map.get(o.order_type, "MO"),
                    "limit_price": o.limit_price,
                }
            )

        raw_results = self.inner.place_orders_batch(
            order_dicts,
            delay_ms=delay_ms,
            margin_trade_type=self._broker_config.margin_trade_type,
            account_type=self._broker_config.account_type,
            is_close=is_close,
            close_position_order=close_position_order,
        )

        results = []
        for r in raw_results:
            side = OrderSide.BUY if r.get("side") == "BUY" else OrderSide.SELL
            ticker = r.get("ticker", "")
            if ticker and not ticker.endswith(".T") and ticker.isdigit():
                ticker = f"{ticker}.T"
            results.append(
                OrderResult(
                    order_id=str(r.get("order_id", "")),
                    status=OrderStatus.SUBMITTED,
                    ticker=ticker,
                    side=side,
                    quantity=int(r.get("quantity", 0)),
                    margin_trade_type=r.get(
                        "margin_trade_type",
                        self._broker_config.margin_trade_type,
                    ),
                )
            )
        return results
