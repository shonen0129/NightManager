"""tests/unit/test_close_positions.py

Unit tests for close_all_positions lot-size rounding and OrderRequest propagation.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from leadlag.broker.base import BrokerClient, Position
from leadlag.core.types import OrderRequest, OrderResult, OrderSide, OrderStatus, OrderType


# ---------------------------------------------------------------------------
# Mock broker client
# ---------------------------------------------------------------------------


class MockBrokerClient(BrokerClient):
    """Minimal mock that records submitted orders and returns canned results."""

    def __init__(self, positions: list[Position]) -> None:
        self._positions = positions
        self.submitted: list[OrderRequest] = []

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
        self.submitted.append(order)
        return OrderResult(
            order_id="MOCK-001",
            status=OrderStatus.SUBMITTED,
            ticker=order.ticker,
            side=order.side,
            quantity=order.quantity,
            order_type=order.order_type,
            margin_trade_type=order.margin_trade_type or 3,
        )

    def submit_orders_batch(self, orders, *, delay_ms=250, is_close=False, close_position_order=0) -> list[OrderResult]:
        results = []
        for order in orders:
            self.submitted.append(order)
            results.append(OrderResult(
                order_id="MOCK-001",
                status=OrderStatus.SUBMITTED,
                ticker=order.ticker,
                side=order.side,
                quantity=order.quantity,
                order_type=order.order_type,
                margin_trade_type=order.margin_trade_type or 3,
            ))
        return results

    def close(self) -> None:
        pass


class FailingBrokerClient(MockBrokerClient):
    """Mock that fails every other order."""

    def submit_orders_batch(self, orders, *, delay_ms=250, is_close=False, close_position_order=0) -> list[OrderResult]:
        results = []
        for i, order in enumerate(orders):
            self.submitted.append(order)
            if i % 2 == 1:
                results.append(OrderResult(
                    order_id="",
                    status=OrderStatus.FAILED,
                    ticker=order.ticker,
                    side=order.side,
                    quantity=order.quantity,
                    order_type=order.order_type,
                    message="Mock failure",
                ))
            else:
                results.append(OrderResult(
                    order_id="MOCK-001",
                    status=OrderStatus.SUBMITTED,
                    ticker=order.ticker,
                    side=order.side,
                    quantity=order.quantity,
                    order_type=order.order_type,
                ))
        return results


def _make_position(ticker: str, side: str, qty: int, *, margin_trade_type: int = 3, account_type: int = 4) -> Position:
    return Position(
        ticker=ticker,
        side=side,
        quantity=qty,
        price=1000.0,
        exchange=27,
        execution_id="POS-001",
        margin_trade_type=margin_trade_type,
        account_type=account_type,
    )


# ---------------------------------------------------------------------------
# Tests: lot-size rounding
# ---------------------------------------------------------------------------


class TestCloseQtyRounding:
    """Test that close_qty is rounded to the nearest lot-size multiple."""

    def test_1629_short_half_rounds_up(self):
        """1629.T SELL x330, alpha_short=0.5 -> close 170 (round 165/10=16.5->17*10)."""
        positions = [_make_position("1629.T", "SELL", 330)]
        client = MockBrokerClient(positions)
        with tempfile.TemporaryDirectory() as tmpdir:
            from leadlag.execution.close import close_all_positions
            summary = close_all_positions(
                client, tmpdir, dry_run=True,
                overnight_alpha_long=0.0, overnight_alpha_short=0.5,
            )
        # Find the 1629.T order
        close_orders = [r for r in summary["close_results"] if r["ticker"] == "1629.T"]
        assert len(close_orders) == 1
        assert close_orders[0]["quantity"] == 170

    def test_1629_long_quarter_rounds(self):
        """1629.T BUY x330, alpha_long=0.75 -> close 85 (round 82.5/10=8.25->8*10=80)."""
        positions = [_make_position("1629.T", "BUY", 330)]
        client = MockBrokerClient(positions)
        with tempfile.TemporaryDirectory() as tmpdir:
            from leadlag.execution.close import close_all_positions
            summary = close_all_positions(
                client, tmpdir, dry_run=True,
                overnight_alpha_long=0.75, overnight_alpha_short=0.0,
            )
        close_orders = [r for r in summary["close_results"] if r["ticker"] == "1629.T"]
        assert len(close_orders) == 1
        # 330 * 0.25 = 82.5, round(82.5/10) = round(8.25) = 8, 8*10 = 80
        assert close_orders[0]["quantity"] == 80

    def test_standard_lot_size_1_unchanged(self):
        """1617.T BUY x100, alpha_long=0.0 -> close 100 (lot_size=1, no rounding needed)."""
        positions = [_make_position("1617.T", "BUY", 100)]
        client = MockBrokerClient(positions)
        with tempfile.TemporaryDirectory() as tmpdir:
            from leadlag.execution.close import close_all_positions
            summary = close_all_positions(
                client, tmpdir, dry_run=True,
                overnight_alpha_long=0.0, overnight_alpha_short=0.0,
            )
        close_orders = [r for r in summary["close_results"] if r["ticker"] == "1617.T"]
        assert len(close_orders) == 1
        assert close_orders[0]["quantity"] == 100

    def test_1629_full_close(self):
        """1629.T SELL x300, alpha_short=0.0 -> close 300 (exact multiple of 10)."""
        positions = [_make_position("1629.T", "SELL", 300)]
        client = MockBrokerClient(positions)
        with tempfile.TemporaryDirectory() as tmpdir:
            from leadlag.execution.close import close_all_positions
            summary = close_all_positions(
                client, tmpdir, dry_run=True,
                overnight_alpha_long=0.0, overnight_alpha_short=0.0,
            )
        close_orders = [r for r in summary["close_results"] if r["ticker"] == "1629.T"]
        assert len(close_orders) == 1
        assert close_orders[0]["quantity"] == 300

    def test_1629_close_qty_does_not_exceed_position(self):
        """1629.T SELL x305, alpha_short=0.0 -> close 300 (305 rounded down to 300)."""
        positions = [_make_position("1629.T", "SELL", 305)]
        client = MockBrokerClient(positions)
        with tempfile.TemporaryDirectory() as tmpdir:
            from leadlag.execution.close import close_all_positions
            summary = close_all_positions(
                client, tmpdir, dry_run=True,
                overnight_alpha_long=0.0, overnight_alpha_short=0.0,
            )
        close_orders = [r for r in summary["close_results"] if r["ticker"] == "1629.T"]
        assert len(close_orders) == 1
        assert close_orders[0]["quantity"] == 300

    def test_1629_sub_lot_size_skipped(self):
        """1629.T SELL x5, alpha_short=0.0 -> close_qty=0, order skipped."""
        positions = [_make_position("1629.T", "SELL", 5)]
        client = MockBrokerClient(positions)
        with tempfile.TemporaryDirectory() as tmpdir:
            from leadlag.execution.close import close_all_positions
            summary = close_all_positions(
                client, tmpdir, dry_run=True,
                overnight_alpha_long=0.0, overnight_alpha_short=0.0,
            )
        close_orders = [r for r in summary["close_results"] if r["ticker"] == "1629.T"]
        assert len(close_orders) == 0

    def test_multiple_positions_mixed_lot_sizes(self):
        """Mix of 1629.T (lot=10) and 1617.T (lot=1) with alpha_short=0.5."""
        positions = [
            _make_position("1629.T", "SELL", 330),
            _make_position("1617.T", "SELL", 100),
        ]
        client = MockBrokerClient(positions)
        with tempfile.TemporaryDirectory() as tmpdir:
            from leadlag.execution.close import close_all_positions
            summary = close_all_positions(
                client, tmpdir, dry_run=True,
                overnight_alpha_long=0.0, overnight_alpha_short=0.5,
            )
        qty_by_ticker = {r["ticker"]: r["quantity"] for r in summary["close_results"]}
        assert qty_by_ticker.get("1629.T") == 170  # round(165/10)*10
        assert qty_by_ticker.get("1617.T") == 50   # round(50/1)*1


# ---------------------------------------------------------------------------
# Tests: OrderRequest propagation
# ---------------------------------------------------------------------------


class TestOrderRequestPropagation:
    """Test that margin_trade_type and account_type are propagated to OrderRequest."""

    def test_margin_trade_type_propagated(self):
        """Position with margin_trade_type=1 (制度) should propagate to OrderRequest.

        Note: 1629.T x300 >= 100 threshold → split into 2 batches (150+150).
        Both batches should carry margin_trade_type=1.
        """
        positions = [_make_position("1629.T", "SELL", 300, margin_trade_type=1, account_type=4)]
        client = MockBrokerClient(positions)
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
        assert len(client.submitted) == 2  # split into 2 batches
        for req in client.submitted:
            assert req.margin_trade_type == 1
            assert req.account_type == 4

    def test_margin_trade_type_default_when_none(self):
        """Position with margin_trade_type=None should not set it on OrderRequest."""
        pos = Position(
            ticker="1617.T", side="SELL", quantity=100, price=1000.0,
            exchange=27, execution_id="POS-001",
            margin_trade_type=None, account_type=None,
        )
        client = MockBrokerClient([pos])
        with tempfile.TemporaryDirectory() as tmpdir:
            from leadlag.execution.close import close_all_positions
            close_all_positions(
                client, tmpdir, dry_run=False,
                overnight_alpha_long=0.0, overnight_alpha_short=0.0,
            )
        assert len(client.submitted) == 1
        assert client.submitted[0].margin_trade_type is None
        assert client.submitted[0].account_type is None


# ---------------------------------------------------------------------------
# Tests: success count
# ---------------------------------------------------------------------------


class TestSuccessCount:
    """Test that success count excludes FAILED orders."""

    def test_failed_orders_excluded_from_success_count(self):
        """With 4 orders where 2 fail, success_count should be 2."""
        positions = [
            _make_position("1617.T", "SELL", 100),
            _make_position("1618.T", "SELL", 100),
            _make_position("1619.T", "SELL", 100),
            _make_position("1620.T", "SELL", 100),
        ]
        client = FailingBrokerClient(positions)
        with tempfile.TemporaryDirectory() as tmpdir:
            from leadlag.execution.close import close_all_positions
            summary = close_all_positions(
                client, tmpdir, dry_run=False,
                overnight_alpha_long=0.0, overnight_alpha_short=0.0,
            )
        statuses = [r["status"] for r in summary["close_results"]]
        submitted_count = sum(1 for s in statuses if s != "FAILED")
        failed_count = sum(1 for s in statuses if s == "FAILED")
        assert submitted_count == 2
        assert failed_count == 2
