"""S6a/S6b pure P&L and fill-ledger contracts."""

from __future__ import annotations

import pytest

from leadlag.core.pnl import FeeAccrual, Fill, InventoryLedger, fill_from_record
from leadlag.reporting.daily_pnl_report import compute_unrealized_pnl


def test_fifo_ledger_allocates_entry_and_exit_fees_once() -> None:
    ledger = InventoryLedger()
    ledger.apply_fill(
        Fill("2026-01-05", "1570.T", "BUY", 100, 1000.0, fee=10.0, source="observed")
    )
    records = ledger.apply_fill(
        Fill("2026-01-06", "1570.T", "SELL", 40, 1050.0, fee=4.0, source="observed")
    )
    assert len(records) == 1
    assert records[0].entry_fee == pytest.approx(4.0)
    assert records[0].exit_fee == pytest.approx(4.0)
    assert records[0].realized_pnl == pytest.approx(40 * 50.0 - 8.0)
    assert ledger.open_lots()[0].quantity == 60
    assert ledger.total_fees == pytest.approx(14.0)


def test_fifo_ledger_handles_short_and_reversal() -> None:
    ledger = InventoryLedger()
    ledger.apply_fill(Fill("2026-01-05", "1617.T", "SELL", 10, 2000.0))
    short_close = ledger.apply_fill(Fill("2026-01-06", "1617.T", "BUY", 4, 1900.0))
    assert short_close[0].realized_pnl == pytest.approx(400.0)
    assert ledger.open_lots()[0].side == "SELL"
    assert ledger.open_lots()[0].quantity == 6

    reversal = ledger.apply_fill(Fill("2026-01-07", "1617.T", "BUY", 10, 2100.0))
    assert reversal[0].quantity == 6
    assert reversal[0].realized_pnl == pytest.approx((2000.0 - 2100.0) * 6)
    assert ledger.open_lots()[0].side == "BUY"
    assert ledger.open_lots()[0].quantity == 4


def test_mark_to_market_uses_observed_inventory_and_entry_fee() -> None:
    ledger = InventoryLedger()
    ledger.apply_fill(Fill("2026-01-05", "1306.T", "BUY", 10, 100.0, fee=10.0))
    marked = ledger.mark_to_market({"1306.T": 110.0}, trade_date="2026-01-06")
    assert len(marked) == 1
    assert marked[0].unrealized_pnl == pytest.approx(90.0)


def test_report_unrealized_falls_back_only_when_snapshot_lacks_prices() -> None:
    snapshot = {
        "total_unrealized_pnl": 123.0,
        "positions": [
            {
                "ticker": "1306.T",
                "side": "BUY",
                "quantity": 10,
                "entry_price": 100.0,
                "evaluation_price": 110.0,
            }
        ],
    }
    assert compute_unrealized_pnl(snapshot) == pytest.approx(100.0)
    assert compute_unrealized_pnl({"total_unrealized_pnl": 123.0, "positions": [{"ticker": "1306.T"}]}) == pytest.approx(123.0)


def test_fill_adapter_uses_observed_price_and_explicit_fee_without_slippage() -> None:
    fill = fill_from_record(
        {
            "status": "FILLED",
            "ticker": "1306.T",
            "side": "SELL",
            "fill_quantity": 10,
            "fill_price": 110.0,
            "fill_detail": {"sBaiBaiTesuryo": 2.5},
            "trade_date": "2026-01-06",
        }
    )
    assert fill is not None
    assert fill.price == 110.0
    assert fill.fee == 2.5
    assert fill.source == "observed"
    assert fill_from_record({"status": "FILLED", "ticker": "1306.T"}) is None
    assert fill_from_record(
        {
            "status": "FILLED",
            "ticker": "1306.T",
            "side": "SELL",
            "fill_quantity": 10,
            "fill_price": 110.0,
            "trade_date": "2026-01-06",
        }
    ) is None


def test_fill_adapter_keeps_confirmed_partial_cancelled_fill() -> None:
    fill = fill_from_record(
        {
            "status": "CANCELLED",
            "ticker": "1306.T",
            "side": "BUY",
            "fill_quantity": 3,
            "fill_price": 100.0,
            "fee": 0.0,
            "trade_date": "2026-01-06",
        }
    )
    assert fill is not None
    assert fill.quantity == 3


@pytest.mark.parametrize("fee_value", [None, float("nan")])
def test_observed_fill_requires_explicit_finite_fee(fee_value: float | None) -> None:
    record = {
        "status": "FILLED",
        "ticker": "1306.T",
        "side": "BUY",
        "fill_quantity": 1,
        "fill_price": 100.0,
        "fee": fee_value,
        "trade_date": "2026-01-06",
    }
    assert fill_from_record(record, source="observed_fill") is None


def test_ledger_keeps_non_trade_fee_accruals_explicit() -> None:
    ledger = InventoryLedger()
    ledger.record_fee(
        FeeAccrual(
            trade_date="2026-01-06",
            ticker="1306.T",
            kind="financing",
            amount=3.0,
            source="broker_statement",
        )
    )
    assert ledger.total_fees == pytest.approx(3.0)
    assert ledger.fees()[0].kind == "financing"
