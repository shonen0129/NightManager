"""tests/unit/test_dry_run_broker.py

Unit tests for broker.dry_run.DryRunBrokerClient and broker.factory.
No network connections required — all tests are deterministic.
"""

from __future__ import annotations

import pytest

from broker.base import BrokerClient, BrokerConfig
from broker.dry_run import DryRunBrokerClient
from broker.factory import create_broker_from_args
from broker.kabu.client import KabuBrokerClient
from domain.models.types import (
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client() -> DryRunBrokerClient:
    return DryRunBrokerClient()


@pytest.fixture
def buy_order() -> OrderRequest:
    return OrderRequest(
        ticker="1617.T",
        side=OrderSide.BUY,
        quantity=100,
        order_type=OrderType.MARKET,
    )


@pytest.fixture
def sell_order() -> OrderRequest:
    return OrderRequest(
        ticker="XLB",
        side=OrderSide.SELL,
        quantity=50,
        order_type=OrderType.MARKET,
    )


# ---------------------------------------------------------------------------
# DryRunBrokerClient tests
# ---------------------------------------------------------------------------


class TestDryRunBrokerClient:
    def test_is_subclass_of_broker_client(self, client):
        assert isinstance(client, BrokerClient)

    def test_health_check_returns_true(self, client):
        assert client.health_check() is True

    def test_get_wallet_cash_positive(self, client):
        wallet = client.get_wallet()
        assert wallet.cash_available > 0

    def test_get_wallet_margin_positive(self, client):
        wallet = client.get_wallet()
        assert wallet.margin_available > 0

    def test_submit_order_returns_simulated_status(self, client, buy_order):
        result = client.submit_order(buy_order)
        assert result.status == OrderStatus.SIMULATED

    def test_submit_order_preserves_ticker(self, client, buy_order):
        result = client.submit_order(buy_order)
        assert result.ticker == buy_order.ticker

    def test_submit_order_preserves_side(self, client, buy_order):
        result = client.submit_order(buy_order)
        assert result.side == buy_order.side

    def test_submit_order_preserves_quantity(self, client, buy_order):
        result = client.submit_order(buy_order)
        assert result.quantity == buy_order.quantity

    def test_submit_order_has_order_id(self, client, buy_order):
        result = client.submit_order(buy_order)
        assert result.order_id
        assert len(result.order_id) > 0

    def test_submit_orders_batch_count_matches(self, client, buy_order, sell_order):
        orders = [buy_order, sell_order]
        results = client.submit_orders_batch(orders)
        assert len(results) == 2

    def test_submit_orders_batch_all_simulated(self, client, buy_order, sell_order):
        orders = [buy_order, sell_order]
        results = client.submit_orders_batch(orders)
        assert all(r.status == OrderStatus.SIMULATED for r in results)

    def test_get_positions_returns_list(self, client):
        positions = client.get_positions()
        assert isinstance(positions, list)

    def test_fetch_jp_open_prices_returns_dict(self, client):
        # Provide simulated prices via a fresh client instance
        tickers = ["1617.T", "1618.T"]
        priced_client = DryRunBrokerClient(
            simulated_open_prices={"1617.T": 1500.0, "1618.T": 900.0}
        )
        prices = priced_client.fetch_open_prices(tickers)
        assert isinstance(prices, dict)
        for tk in tickers:
            assert tk in prices
            assert prices[tk] > 0

    def test_fetch_open_prices_allow_missing(self, client):
        # With allow_missing=True, missing tickers are silently omitted
        tickers = ["1617.T", "1618.T"]
        prices = client.fetch_open_prices(tickers, allow_missing=True)
        assert isinstance(prices, dict)  # may be empty, but must be a dict

    def test_close_sets_order_type_to_close(self, client):
        """close() should not raise errors."""
        client.close()  # Should be a no-op (no network)


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------


class TestBrokerFactory:
    def test_dry_run_true_returns_dry_run_client(self):
        client = create_broker_from_args(
            api_url="http://localhost:18080",
            api_token=None,
            api_password=None,
            dry_run=True,
        )
        assert isinstance(client, DryRunBrokerClient)

    def test_dry_run_false_returns_kabu_client(self):
        client = create_broker_from_args(
            api_url="http://localhost:18080",
            api_token="test-token",
            api_password=None,
            dry_run=False,
        )
        assert isinstance(client, KabuBrokerClient)

    def test_both_are_broker_client_subclasses(self):
        dry = create_broker_from_args(
            api_url="http://localhost:18080",
            api_token=None,
            api_password=None,
            dry_run=True,
        )
        kabu = create_broker_from_args(
            api_url="http://localhost:18080",
            api_token="test-token",
            api_password=None,
            dry_run=False,
        )
        assert isinstance(dry, BrokerClient)
        assert isinstance(kabu, BrokerClient)
