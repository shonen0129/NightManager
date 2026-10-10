from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from leadlag.broker.base import BrokerClient, Position
from leadlag.core.types import OrderRequest, OrderResult, OrderSide, OrderStatus, OrderType
from leadlag.execution import close as close_module
from leadlag.execution.contracts import ExecutionPlan, report_from_records
from leadlag.execution.state_store import ExecutionRunStatus, ExecutionStateStore


def _position(
    ticker: str,
    side: str,
    quantity: int,
    *,
    execution_id: str = "position",
) -> Position:
    return Position(
        ticker=ticker,
        side=side,
        quantity=quantity,
        price=1000.0,
        exchange=27,
        execution_id=execution_id,
        margin_trade_type=3,
        account_type=4,
    )


class _BatchBroker(BrokerClient):
    def __init__(
        self,
        positions: Sequence[Position],
        batches: Sequence[Sequence[OrderResult]],
        *,
        polled_status: OrderStatus | None = None,
    ) -> None:
        self._positions = list(positions)
        self._batches = [list(batch) for batch in batches]
        self.polled_status = polled_status
        self.submitted_batches: list[list[OrderRequest]] = []

    def get_positions(self, **filters) -> list[Position]:
        del filters
        return list(self._positions)

    def get_wallet(self):
        raise NotImplementedError

    def fetch_open_prices(self, tickers, *, allow_missing=False):
        del tickers, allow_missing
        return {}

    def fetch_us_etf_returns(self, us_tickers):
        del us_tickers
        return {}

    def fetch_current_prices(self, tickers, *, allow_missing=False):
        del tickers, allow_missing
        return {}

    def health_check(self) -> bool:
        return True

    def submit_order(
        self,
        order: OrderRequest,
        *,
        is_close: bool = False,
        close_position_order: int = 0,
    ) -> OrderResult:
        del is_close, close_position_order
        return OrderResult(
            order_id="single",
            status=OrderStatus.FILLED,
            ticker=order.ticker,
            side=order.side,
            quantity=order.quantity,
            order_type=order.order_type,
        )

    def submit_orders_batch(
        self,
        orders,
        *,
        delay_ms=250,
        is_close=False,
        close_position_order=0,
    ) -> list[OrderResult]:
        del delay_ms, is_close, close_position_order
        self.submitted_batches.append(list(orders))
        if not self._batches:
            return []
        return self._batches.pop(0)

    def get_order_status(self, order_id: str) -> OrderStatus:
        del order_id
        if self.polled_status is None:
            raise NotImplementedError
        return self.polled_status

    def close(self) -> None:
        return None


def _result(
    order_id: str,
    status: OrderStatus,
    ticker: str,
    side: OrderSide,
    quantity: int,
) -> OrderResult:
    return OrderResult(
        order_id=order_id,
        status=status,
        ticker=ticker,
        side=side,
        quantity=quantity,
        order_type=OrderType.CLOSE,
    )


def test_empty_broker_response_is_explicit_and_durable(tmp_path: Path) -> None:
    store = ExecutionStateStore(tmp_path / "state.sqlite")
    broker = _BatchBroker([_position("1617.T", "BUY", 100)], [[]])

    summary = close_module.close_all_positions(
        broker,
        tmp_path,
        state_store=store,
        account_key="test",
        overnight_alpha_long=0.0,
    )

    assert summary["close_incomplete"] is True
    assert len(summary["close_results"]) == 1
    assert summary["close_results"][0]["quantity"] == 100
    assert summary["close_results"][0]["status"] == close_module.MISSING_RESPONSE_STATUS
    assert summary["execution_report"]["incomplete"] is True
    assert summary["execution_report"]["unresolved_orders"] == 1
    run = store.get_run(str(summary["run_id"]))
    assert run is not None
    assert run.status == ExecutionRunStatus.RECONCILIATION_REQUIRED
    checkpoint = store.latest_reconciliation(run.run_id)
    assert checkpoint is not None
    assert checkpoint["outcome"] == "incomplete"


def test_partial_batch_response_synthesizes_only_missing_request(tmp_path: Path) -> None:
    broker = _BatchBroker(
        [
            _position("1617.T", "BUY", 100),
            _position("1618.T", "BUY", 200),
        ],
        [[_result("a", OrderStatus.FILLED, "1617.T", OrderSide.SELL, 100)]],
    )

    summary = close_module.close_all_positions(broker, tmp_path)

    by_ticker = {row["ticker"]: row for row in summary["close_results"]}
    assert by_ticker["1617.T"]["status"] == "FILLED"
    assert by_ticker["1618.T"]["status"] == close_module.MISSING_RESPONSE_STATUS
    assert by_ticker["1618.T"]["quantity"] == 200
    assert summary["close_incomplete"] is True
    assert summary["execution_report"]["incomplete"] is True


def test_duplicate_response_is_reconciliation_error(tmp_path: Path) -> None:
    duplicate = _result("dup", OrderStatus.FILLED, "1617.T", OrderSide.SELL, 100)
    broker = _BatchBroker(
        [_position("1617.T", "BUY", 100)],
        [[duplicate, duplicate]],
    )

    summary = close_module.close_all_positions(broker, tmp_path)

    assert summary["close_incomplete"] is True
    assert any(
        "duplicate broker order response" in error
        for error in summary["plan_reconciliation_errors"]
    )
    assert any(
        "persisted plan" in error
        for error in summary["plan_reconciliation_errors"]
    )
    assert summary["execution_report"]["incomplete"] is True


def test_filled_response_without_order_id_is_not_complete(tmp_path: Path, monkeypatch) -> None:
    store = ExecutionStateStore(tmp_path / "state.sqlite")
    broker = _BatchBroker(
        [_position("1617.T", "BUY", 100)],
        [[_result("", OrderStatus.FILLED, "1617.T", OrderSide.SELL, 100)]],
    )
    monkeypatch.setattr(close_module, "fetch_fill_prices", lambda *_args, **_kwargs: None)

    summary = close_module.close_all_positions(broker, tmp_path, state_store=store, account_key="test")

    assert summary["close_incomplete"] is True
    assert summary["execution_report"]["incomplete"] is True
    assert any("missing broker order id" in error for error in summary["plan_reconciliation_errors"])
    run = store.get_run(str(summary["run_id"]))
    assert run is not None
    assert run.status == ExecutionRunStatus.RECONCILIATION_REQUIRED


@pytest.mark.parametrize(
    ("status", "expected_status"),
    [
        (OrderStatus.CANCELLED, "CANCELLED"),
        (OrderStatus.PARTIALLY_FILLED, "PARTIALLY_FILLED"),
        (OrderStatus.SUBMITTED, "SUBMITTED"),
    ],
)
def test_split_delayed_batch_stops_on_unreconciled_first_batch(
    tmp_path: Path,
    monkeypatch,
    status: OrderStatus,
    expected_status: str,
) -> None:
    broker = _BatchBroker(
        [_position("1629.T", "SELL", 330)],
        [[_result("first", status, "1629.T", OrderSide.BUY, 80)]],
    )
    monkeypatch.setattr(close_module, "_wait_for_close_fills_sync", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        close_module.time_module,
        "sleep",
        lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("delayed sleep must not run")),
    )

    summary = close_module.close_all_positions(
        broker,
        tmp_path,
        overnight_alpha_short=0.5,
    )

    assert len(broker.submitted_batches) == 1
    assert broker.submitted_batches[0][0].quantity == 80
    statuses = [row["status"] for row in summary["close_results"]]
    assert expected_status in statuses
    assert "SKIPPED" in statuses
    skipped = next(row for row in summary["close_results"] if row["status"] == "SKIPPED")
    assert "first batch was not reconciled" in skipped["message"]
    assert summary["close_incomplete"] is True
    assert summary["execution_report"]["incomplete"] is True


def test_split_delayed_batch_stops_on_missing_first_response(
    tmp_path: Path,
    monkeypatch,
) -> None:
    broker = _BatchBroker([_position("1629.T", "SELL", 330)], [[]])
    monkeypatch.setattr(
        close_module.time_module,
        "sleep",
        lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("delayed sleep must not run")),
    )

    summary = close_module.close_all_positions(
        broker,
        tmp_path,
        overnight_alpha_short=0.5,
    )

    assert len(broker.submitted_batches) == 1
    assert [row["status"] for row in summary["close_results"]] == [
        close_module.MISSING_RESPONSE_STATUS,
        "SKIPPED",
    ]
    assert summary["close_incomplete"] is True


def test_split_all_filled_reconciles_by_quantity_not_response_count(
    tmp_path: Path,
    monkeypatch,
) -> None:
    broker = _BatchBroker(
        [_position("1629.T", "SELL", 330)],
        [
            [_result("first", OrderStatus.FILLED, "1629.T", OrderSide.BUY, 80)],
            [_result("second", OrderStatus.FILLED, "1629.T", OrderSide.BUY, 80)],
        ],
    )
    monkeypatch.setattr(close_module.time_module, "sleep", lambda *_a, **_kw: None)

    summary = close_module.close_all_positions(
        broker,
        tmp_path,
        overnight_alpha_short=0.5,
    )

    assert len(broker.submitted_batches) == 2
    assert summary["planned_orders_count"] == 1
    assert len(summary["close_results"]) == 2
    assert summary["planned_close_quantity"] == 160
    assert summary["represented_close_quantity"] == 160
    assert summary["plan_reconciliation_errors"] == []
    assert summary["close_incomplete"] is False
    assert summary["execution_report"]["expected_orders"] == 1
    assert summary["execution_report"]["accepted_orders"] == 2
    assert summary["execution_report"]["incomplete"] is False


def test_same_ticker_multiple_lots_reconcile_without_count_shortcut(tmp_path: Path) -> None:
    broker = _BatchBroker(
        [
            _position("1617.T", "BUY", 40, execution_id="lot-a"),
            _position("1617.T", "BUY", 60, execution_id="lot-b"),
        ],
        [[
            _result("lot-b", OrderStatus.FILLED, "1617.T", OrderSide.SELL, 60),
            _result("lot-a", OrderStatus.FILLED, "1617.T", OrderSide.SELL, 40),
        ]],
    )

    summary = close_module.close_all_positions(broker, tmp_path)

    assert summary["planned_orders_count"] == 2
    assert summary["planned_close_quantity"] == 100
    assert summary["represented_close_quantity"] == 100
    assert summary["plan_reconciliation_errors"] == []
    assert summary["close_incomplete"] is False
    assert summary["execution_report"]["incomplete"] is False


def test_plan_reconciliation_rejects_empty_records_without_synthetic_rows() -> None:
    plan = ExecutionPlan(
        decision_id="d",
        trade_date="2026-10-10",
        close_orders=(
            OrderRequest(
                ticker="1617.T",
                side=OrderSide.SELL,
                quantity=100,
                order_type=OrderType.CLOSE,
            ),
        ),
        new_orders=(),
    )

    errors = close_module._close_plan_reconciliation_errors(plan, [])

    assert errors
    assert "persisted plan" in errors[0]


def test_missing_response_status_is_unresolved_in_execution_report() -> None:
    report = report_from_records(
        [
            {
                "order_id": "",
                "status": close_module.MISSING_RESPONSE_STATUS,
                "ticker": "1617.T",
                "side": "SELL",
                "quantity": 100,
            }
        ],
        expected_orders=1,
    )

    assert report.unresolved_orders == 1
    assert report.incomplete is True
