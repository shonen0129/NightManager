"""Regression tests for the S4 execution contracts and shared polling."""

from __future__ import annotations

import pandas as pd

from leadlag.core.types import OrderResult, OrderSide, OrderStatus, OrderType
from leadlag.execution.broker_ops import _wait_for_fills_sync, build_execution_plan
from leadlag.execution.close import _wait_for_close_fills_sync
from leadlag.execution.contracts import ExecutionPlan, OrderObservation, report_from_records


class _SequenceBroker:
    def __init__(self) -> None:
        self.calls = 0

    def get_order_status(self, _order_id: str) -> OrderStatus:
        self.calls += 1
        if self.calls % 3 == 1:
            return OrderStatus.SUBMITTED
        if self.calls % 3 == 2:
            return OrderStatus.PARTIALLY_FILLED
        return OrderStatus.FILLED


def test_shared_poll_updates_order_result_and_close_dict() -> None:
    broker = _SequenceBroker()
    order = OrderResult(
        order_id="O-1",
        status=OrderStatus.SUBMITTED,
        ticker="1617.T",
        side=OrderSide.BUY,
        quantity=100,
        order_type=OrderType.MARKET,
    )
    results, polled = _wait_for_fills_sync(
        broker, [order], timeout_seconds=1.0, poll_interval=0.0
    )
    assert polled is True
    assert results[0].status is OrderStatus.FILLED
    assert "terminal" not in results[0].message

    close_row = [{"order_id": "O-2", "status": "SUBMITTED", "ticker": "1617.T"}]
    _wait_for_close_fills_sync(broker, close_row, timeout_seconds=1.0, poll_interval=0.0)
    assert close_row[0]["status"] == OrderStatus.FILLED.value
    assert "Polled to" in close_row[0]["message"]


def test_execution_plan_is_immutable_and_tracks_close_new_split() -> None:
    frame = pd.DataFrame(
        {
            "ticker": ["1617.T", "1629.T"],
            "action": ["SELL", "BUY"],
            "quantity": [150, 100],
            "weight": [-0.5, 0.5],
        }
    )
    frame.attrs["trade_date"] = "2026-09-16"
    plan = build_execution_plan(frame, {"1617.T": 100})

    assert isinstance(plan, ExecutionPlan)
    assert plan.trade_date == "2026-09-16"
    assert [(o.ticker, o.quantity) for o in plan.close_orders] == [("1617.T", 100)]
    assert [(o.ticker, o.quantity) for o in plan.new_orders] == [
        ("1617.T", 150),
        ("1629.T", 100),
    ]
    assert plan.expected_order_count == 3
    assert plan.to_dict()["current_positions"] == {"1617.T": 100}


def test_execution_report_marks_partial_and_reconciliation_incomplete() -> None:
    report = report_from_records(
        [
            {"status": "FILLED"},
            {"status": "PARTIALLY_FILLED", "quantity": 100, "fill_quantity": 40},
        ],
        expected_orders=2,
        reconciliation_errors=["positions unavailable"],
    )
    assert report.filled_orders == 1
    assert report.partial_orders == 1
    assert report.unresolved_orders == 1
    assert report.incomplete is True
    assert isinstance(report.observations[0], OrderObservation)
    assert report.observations[1].status == "PARTIALLY_FILLED"
    assert report.observations[1].remaining_quantity == 60
    assert report.to_dict()["reconciliation_errors"] == ["positions unavailable"]


def test_execution_report_exposes_confirmed_fill_for_accounting() -> None:
    report = report_from_records(
        [
            {
                "status": "FILLED",
                "order_id": "O-3",
                "ticker": "1306.T",
                "side": "SELL",
                "quantity": 10,
                "fill_quantity": 10,
                "fill_price": 110.0,
                "fee": 2.0,
                "observed_at": "2026-09-16T15:30:00+09:00",
            }
        ],
        expected_orders=1,
    )
    fills = report.observed_fills(trade_date="2026-09-16")
    assert len(fills) == 1
    assert fills[0].price == 110.0
    assert fills[0].fee == 2.0
