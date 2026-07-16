"""tests/unit/test_split_large_orders.py

Unit tests for split_large_orders and 1629.T close-order splitting.
"""

from __future__ import annotations

import tempfile

import pandas as pd
from leadlag.broker.base import BrokerClient, Position
from leadlag.core.types import OrderRequest, OrderResult, OrderSide, OrderStatus, OrderType
from leadlag.execution.helpers import split_large_orders, submit_orders_via_api


def _req(ticker: str, side: OrderSide, qty: int, mtt: int | None = None, at: int | None = None) -> OrderRequest:
    return OrderRequest(
        ticker=ticker,
        side=side,
        quantity=qty,
        order_type=OrderType.MARKET,
        margin_trade_type=mtt,
        account_type=at,
    )


class _MockBroker(BrokerClient):
    """Mock that records submitted batches."""

    def __init__(self, positions: list[Position]) -> None:
        self._positions = positions
        self.batches: list[list[OrderRequest]] = []

    def get_positions(self, **filters) -> list[Position]:
        return list(self._positions)

    def get_wallet(self) -> object:
        raise NotImplementedError

    def fetch_open_prices(self, tickers, allow_missing=False) -> dict:
        return {}

    def fetch_us_etf_returns(self, us_tickers, start_date, end_date) -> object:
        raise NotImplementedError

    def fetch_current_prices(self, tickers, *, allow_missing=False) -> dict:
        return {}

    def health_check(self) -> bool:
        return True

    def submit_order(self, order: OrderRequest, *, is_close=False, close_position_order=0) -> OrderResult:
        return OrderResult(
            order_id="MOCK-001",
            status=OrderStatus.SUBMITTED,
            ticker=order.ticker,
            side=order.side,
            quantity=order.quantity,
            order_type=order.order_type,
        )

    def submit_orders_batch(self, orders, *, delay_ms=250, is_close=False, close_position_order=0) -> list[OrderResult]:
        self.batches.append(list(orders))
        return [
            OrderResult(
                order_id="MOCK-001",
                status=OrderStatus.SUBMITTED,
                ticker=o.ticker,
                side=o.side,
                quantity=o.quantity,
                order_type=o.order_type,
            )
            for o in orders
        ]

    def close(self) -> None:
        pass


class TestSplitLargeOrders:
    """Test split_large_orders function directly."""

    def test_1629_large_split(self):
        """1629.T 500 shares → 250 immediate + 250 delayed."""
        orders = [_req("1629.T", OrderSide.SELL, 500, mtt=3, at=4)]
        imm, delayed = split_large_orders(orders)
        assert len(imm) == 1
        assert len(delayed) == 1
        assert imm[0].quantity == 250
        assert delayed[0].quantity == 250
        assert imm[0].margin_trade_type == 3
        assert imm[0].account_type == 4
        assert delayed[0].margin_trade_type == 3
        assert delayed[0].account_type == 4

    def test_1629_small_no_split(self):
        """1629.T 50 shares (< 100 threshold) → no split."""
        orders = [_req("1629.T", OrderSide.SELL, 50)]
        imm, delayed = split_large_orders(orders)
        assert len(imm) == 1
        assert imm[0].quantity == 50
        assert len(delayed) == 0

    def test_1629_threshold_exact(self):
        """1629.T 100 shares (== threshold) → split into 50+50."""
        orders = [_req("1629.T", OrderSide.BUY, 100)]
        imm, delayed = split_large_orders(orders)
        assert len(imm) == 1
        assert len(delayed) == 1
        assert imm[0].quantity == 50
        assert delayed[0].quantity == 50

    def test_non_split_ticker_unchanged(self):
        """1617.T 500 shares → no split (not SPLIT_TICKER)."""
        orders = [_req("1617.T", OrderSide.SELL, 500)]
        imm, delayed = split_large_orders(orders)
        assert len(imm) == 1
        assert imm[0].quantity == 500
        assert len(delayed) == 0

    def test_mixed_orders(self):
        """Mix of 1629.T large, 1629.T small, and other tickers."""
        orders = [
            _req("1629.T", OrderSide.SELL, 300),
            _req("1629.T", OrderSide.BUY, 30),
            _req("1617.T", OrderSide.SELL, 200),
        ]
        imm, delayed = split_large_orders(orders)
        # 1629.T 300 → 150+150, 1629.T 30 → immediate, 1617.T 200 → immediate
        assert len(imm) == 3
        assert len(delayed) == 1
        assert delayed[0].ticker == "1629.T"
        assert delayed[0].quantity == 150

    def test_1629_lot_alignment(self):
        """1629.T 305 shares → 150 + 155 (lot=10, 305//2//10=15, 15*10=150)."""
        orders = [_req("1629.T", OrderSide.SELL, 305)]
        imm, delayed = split_large_orders(orders)
        assert imm[0].quantity == 150
        assert delayed[0].quantity == 155


class TestSubmitOrdersViaApiSplit:
    """Test that submit_orders_via_api splits 1629.T new orders."""

    def _make_decision_df(self, rows: list[tuple[str, str, int]]) -> pd.DataFrame:
        return pd.DataFrame(rows, columns=["ticker", "action", "quantity"])

    def test_1629_large_new_order_split_into_two_batches(self):
        """1629.T BUY 300 → 150 immediate + 150 delayed."""
        decision_df = self._make_decision_df([("1629.T", "BUY", 300)])
        client = _MockBroker([])
        import leadlag.execution.helpers as helpers_mod
        _orig_sleep = helpers_mod.time.sleep
        helpers_mod.time.sleep = lambda *a, **kw: None
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                submit_orders_via_api(decision_df, client, tmpdir)
        finally:
            helpers_mod.time.sleep = _orig_sleep
        assert len(client.batches) == 2
        assert client.batches[0][0].quantity == 150
        assert client.batches[1][0].quantity == 150

    def test_1629_first_batch_failure_skips_delayed(self):
        """If immediate batch fails, delayed batch should be skipped."""
        decision_df = self._make_decision_df([("1629.T", "BUY", 300)])
        client = _FailingMockBroker([])
        import leadlag.execution.helpers as helpers_mod
        _orig_sleep = helpers_mod.time.sleep
        helpers_mod.time.sleep = lambda *a, **kw: None
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                summary = submit_orders_via_api(decision_df, client, tmpdir)
        finally:
            helpers_mod.time.sleep = _orig_sleep
        # Only first batch attempted; delayed skipped
        assert len(client.batches) == 1
        assert len(summary["buy_results"]) == 2  # FAILED + SKIPPED
        assert summary["buy_results"][0]["status"] == "FAILED"
        assert summary["buy_results"][1]["status"] == "SKIPPED"


class _FailingMockBroker(BrokerClient):
    """Mock that fails every immediate batch order."""

    def __init__(self, positions: list[Position]) -> None:
        self._positions = positions
        self.batches: list[list[OrderRequest]] = []

    def get_positions(self, **filters) -> list[Position]:
        return list(self._positions)

    def get_wallet(self) -> object:
        raise NotImplementedError

    def fetch_open_prices(self, tickers, allow_missing=False) -> dict:
        return {}

    def fetch_us_etf_returns(self, us_tickers, start_date, end_date) -> object:
        raise NotImplementedError

    def fetch_current_prices(self, tickers, *, allow_missing=False) -> dict:
        return {}

    def health_check(self) -> bool:
        return True

    def submit_order(self, order: OrderRequest, *, is_close=False, close_position_order=0) -> OrderResult:
        return OrderResult(
            order_id="MOCK-001",
            status=OrderStatus.FAILED,
            ticker=order.ticker,
            side=order.side,
            quantity=order.quantity,
            order_type=order.order_type,
            message="Mock failure",
        )

    def submit_orders_batch(self, orders, *, delay_ms=250, is_close=False, close_position_order=0) -> list[OrderResult]:
        self.batches.append(list(orders))
        return [
            OrderResult(
                order_id="MOCK-001",
                status=OrderStatus.FAILED,
                ticker=o.ticker,
                side=o.side,
                quantity=o.quantity,
                order_type=o.order_type,
                message="Mock failure",
            )
            for o in orders
        ]

    def close(self) -> None:
        pass


class TestCloseSplitIntegration:
    """Test that close_all_positions splits 1629.T large close orders."""

    def test_1629_large_close_split_into_two_batches(self):
        """1629.T SELL x300, alpha=0 → close 300, should split into 2 batches."""
        positions = [
            Position(
                ticker="1629.T", side="SELL", quantity=300, price=250.0,
                exchange=27, execution_id="POS-001",
                margin_trade_type=3, account_type=4,
            ),
        ]
        client = _MockBroker(positions)
        with tempfile.TemporaryDirectory() as tmpdir:
            import leadlag.execution.close as close_mod
            _orig_sleep = close_mod.time_module.sleep
            close_mod.time_module.sleep = lambda *a, **kw: None
            try:
                from leadlag.execution.close import close_all_positions
                close_all_positions(
                    client, tmpdir, dry_run=False,
                    overnight_alpha_long=0.0, overnight_alpha_short=0.0,
                )
            finally:
                close_mod.time_module.sleep = _orig_sleep
        # Should have 2 batches: immediate (150) + delayed (150)
        assert len(client.batches) == 2
        assert client.batches[0][0].quantity == 150
        assert client.batches[1][0].quantity == 150

    def test_1629_small_close_single_batch(self):
        """1629.T SELL x50, alpha=0 → close 50 (< 100), no split."""
        positions = [
            Position(
                ticker="1629.T", side="SELL", quantity=50, price=250.0,
                exchange=27, execution_id="POS-001",
                margin_trade_type=3, account_type=4,
            ),
        ]
        client = _MockBroker(positions)
        with tempfile.TemporaryDirectory() as tmpdir:
            from leadlag.execution.close import close_all_positions
            close_all_positions(
                client, tmpdir, dry_run=False,
                overnight_alpha_long=0.0, overnight_alpha_short=0.0,
            )
        assert len(client.batches) == 1
        assert client.batches[0][0].quantity == 50
