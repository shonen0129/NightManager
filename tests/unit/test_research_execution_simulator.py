from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from leadlag.core.types import OrderRequest, OrderSide, OrderType
from leadlag.data.tickers import JP_TICKERS
from leadlag.execution.microstructure.order_book_schema import OrderBookSnapshot
from research.execution_simulator import (
    OrderIntent,
    PassiveFillEvidence,
    PolicyConfig,
    ReplayEvent,
    assess_capture_record,
    build_order_intents,
    simulate_adaptive_policy,
    simulate_market_baseline,
)
from research.scripts.experiments.replay_execution_policy import build_audit


def _book(
    ticker: str,
    timestamp: datetime,
    *,
    bid: float = 99.0,
    ask: float = 101.0,
    bid_size: float = 100.0,
    ask_size: float = 100.0,
) -> OrderBookSnapshot:
    values: dict[str, object] = {
        "ticker": ticker,
        "timestamp": timestamp.isoformat(),
        "last_price": (bid + ask) / 2,
        "lob_available": True,
        "cost_source": "api_lob",
    }
    for level in range(1, 6):
        values[f"bid_price_{level}"] = bid - (level - 1)
        values[f"ask_price_{level}"] = ask + (level - 1)
        values[f"bid_size_{level}"] = bid_size
        values[f"ask_size_{level}"] = ask_size
    return OrderBookSnapshot(**values)  # type: ignore[arg-type]


def _event(at: datetime, *books: OrderBookSnapshot, **kwargs) -> ReplayEvent:
    return ReplayEvent(at, {book.ticker: book for book in books}, **kwargs)


def test_build_order_intents_uses_pure_market_order_plan_from_broker_ops():
    decisions = pd.DataFrame(
        [
            {"ticker": "AAA.T", "action": "BUY", "quantity": 3},
            {"ticker": "BBB.T", "action": "SELL", "quantity": 2},
            {"ticker": "CCC.T", "action": "HOLD", "quantity": 0},
        ]
    )

    intents = build_order_intents(
        decisions,
        current_positions={},
        pair_ids={"AAA.T": "pair-1", "BBB.T": "pair-1"},
    )

    assert [
        (item.request.ticker, item.request.side, item.request.order_type) for item in intents
    ] == [
        ("AAA.T", OrderSide.BUY, OrderType.MARKET),
        ("BBB.T", OrderSide.SELL, OrderType.MARKET),
    ]
    assert all(item.pair_id == "pair-1" for item in intents)


def test_market_baseline_sweeps_visible_levels_and_keeps_hidden_remainder_unresolved():
    start = datetime.fromisoformat("2026-10-07T09:10:00+09:00")
    book = _book("AAA.T", start, ask_size=1)
    intent = OrderIntent(OrderRequest("AAA.T", OrderSide.BUY, 7))

    result = simulate_market_baseline(
        [intent], [_event(start, book)], deadline=start + timedelta(seconds=30)
    )

    assert result.outcomes[0].filled_quantity == 5
    assert result.outcomes[0].unresolved_quantity == 2
    assert result.metrics()["total_shortfall_status"] == "INCOMPLETE_DATA"
    assert result.metrics()["total_shortfall_jpy"] is None


def test_adaptive_policy_waits_for_queue_evidence_then_crosses_at_deadline():
    start = datetime.fromisoformat("2026-10-07T09:10:00+09:00")
    deadline = start + timedelta(seconds=20)
    first = _book("AAA.T", start)
    passive_time = start + timedelta(seconds=5)
    passive = PassiveFillEvidence("AAA.T", OrderSide.BUY, 99.0, 1, True, "queue_replay")
    second = _book("AAA.T", passive_time)
    final = _book("AAA.T", deadline)
    intent = OrderIntent(OrderRequest("AAA.T", OrderSide.BUY, 2))
    events = [
        _event(start, first),
        _event(passive_time, second, passive_fills=(passive,), passive_evidence_complete=True),
        _event(deadline, final, passive_evidence_complete=True),
    ]

    adaptive = simulate_adaptive_policy([intent], events, deadline=deadline)
    baseline = simulate_market_baseline([intent], events, deadline=deadline)

    assert [fill.source for fill in adaptive.fills] == [
        "queue_aware_passive_evidence",
        "adaptive_cross_displayed_depth",
    ]
    assert adaptive.metrics()["total_shortfall_status"] == "COMPLETE"
    assert adaptive.metrics()["total_shortfall_jpy"] == pytest.approx(0.0)
    assert baseline.metrics()["total_shortfall_jpy"] == pytest.approx(2.0)
    assert baseline.metrics()["adverse_selection_observed_fill_count"] == 1
    assert adaptive.metrics()["adverse_selection_observed_fill_count"] == 1


def test_quote_touch_without_queue_fill_evidence_is_not_counted_as_passive_fill():
    start = datetime.fromisoformat("2026-10-07T09:10:00+09:00")
    deadline = start + timedelta(seconds=30)
    events = [
        _event(start, _book("AAA.T", start)),
        _event(start + timedelta(seconds=10), _book("AAA.T", start + timedelta(seconds=10))),
    ]
    intent = OrderIntent(OrderRequest("AAA.T", OrderSide.BUY, 2))

    result = simulate_adaptive_policy(
        [intent],
        events,
        deadline=deadline,
        config=PolicyConfig(urgent_last_seconds=5),
    )

    assert result.outcomes[0].filled_quantity == 0
    assert result.outcomes[0].unresolved_quantity == 2
    assert result.metrics()["known_unfilled_quantity"] == 0


def test_pair_exposure_cap_reduces_allowed_order_quantity_and_records_net_exposure():
    start = datetime.fromisoformat("2026-10-07T09:10:00+09:00")
    deadline = start + timedelta(seconds=1)
    event = _event(
        start,
        _book("AAA.T", start, bid=100.0, ask=100.1),
        _book("BBB.T", start, bid=100.0, ask=100.1),
    )
    intents = [
        OrderIntent(OrderRequest("AAA.T", OrderSide.BUY, 10), "pair"),
        OrderIntent(OrderRequest("BBB.T", OrderSide.SELL, 10), "pair"),
    ]

    result = simulate_adaptive_policy(intents, [event], deadline=deadline)

    assert result.metrics()["pair_quantity_adjusted_orders"] == 2
    assert result.metrics()["quantity_deferred_by_pair_cap"] > 0
    assert result.metrics()["max_abs_net_exposure_jpy"] == pytest.approx(100.05)


def test_out_of_window_snapshot_is_rejected_even_with_full_depth():
    timestamp = datetime.fromisoformat("2026-10-07T02:58:00+09:00")
    row = {
        "ticker": "1617.T",
        "lob_available": True,
        **{
            f"{side}_{kind}_{level}": 1.0
            for side in ("bid", "ask")
            for kind in ("price", "size")
            for level in range(1, 6)
        },
    }
    record = {
        "status": "OBSERVED_OUTSIDE_0910_WINDOW",
        "observed_at": timestamp.isoformat(),
        "window_valid": False,
        "rows": [row],
    }

    result = assess_capture_record(record, required_tickers=["1617.T"])

    assert result["full_five_level_depth"] is True
    assert result["eligible_for_execution_replay"] is False
    assert "not_a_valid_09:10_capture" in result["reasons"]


def test_window_flag_cannot_promote_a_timestamped_quote_outside_0910():
    observed_at = "2026-10-07T02:58:00+09:00"
    row = {
        "ticker": "1617.T",
        "timestamp": observed_at,
        "lob_available": True,
        **{
            f"{side}_{kind}_{level}": float(level)
            for side in ("bid", "ask")
            for kind in ("price", "size")
            for level in range(1, 6)
        },
    }
    record = {
        "status": "OBSERVED",
        "observed_at": observed_at,
        "window_valid": True,
        "rows": [row],
        "trades": [{}],
        "queue_events": [{}],
        "passive_evidence_complete": True,
    }

    result = assess_capture_record(record, required_tickers=["1617.T"])

    assert result["valid_09_10_capture"] is False
    assert result["eligible_for_execution_replay"] is False


@pytest.mark.parametrize("deadline_complete", [False, True])
def test_missing_passive_interval_never_becomes_a_complete_cross(deadline_complete):
    start = datetime.fromisoformat("2026-10-07T09:10:00+09:00")
    deadline = start + timedelta(seconds=30)
    middle = start + timedelta(seconds=5)
    events = [
        _event(start, _book("AAA.T", start)),
        _event(middle, _book("AAA.T", middle)),
        _event(deadline, _book("AAA.T", deadline), passive_evidence_complete=deadline_complete),
    ]
    intent = OrderIntent(OrderRequest("AAA.T", OrderSide.BUY, 2))

    result = simulate_adaptive_policy([intent], events, deadline=deadline)

    assert result.fills == []
    assert result.outcomes[0].unresolved_quantity == 2
    assert result.metrics()["total_shortfall_jpy"] is None
    assert result.metrics()["total_shortfall_status"] == "INCOMPLETE_DATA"
    assert result.metrics()["net_exposure_status"] == "INCOMPLETE_DATA"


def test_missing_passive_evidence_preserves_prior_known_fills():
    start = datetime.fromisoformat("2026-10-07T09:10:00+09:00")
    fill_time = start + timedelta(seconds=5)
    deadline = start + timedelta(seconds=30)
    intents = [OrderIntent(OrderRequest("AAA.T", OrderSide.BUY, 2))]
    events = [
        _event(start, _book("AAA.T", start)),
        _event(fill_time, _book("AAA.T", fill_time),
               passive_fills=(PassiveFillEvidence("AAA.T", OrderSide.BUY, 99, 1, True, "queue"),),
               passive_evidence_complete=True),
        _event(deadline, _book("AAA.T", deadline)),
    ]

    result = simulate_adaptive_policy(intents, events, deadline=deadline)

    assert result.outcomes[0].filled_quantity == 1
    assert result.outcomes[0].unresolved_quantity == 1
    assert all(fill.timestamp == fill_time for fill in result.fills)
    assert result.metrics()["total_shortfall_status"] == "INCOMPLETE_DATA"


def test_missing_passive_interval_stops_both_legs_before_any_deadline_cross():
    start = datetime.fromisoformat("2026-10-07T09:10:00+09:00")
    deadline = start + timedelta(seconds=30)
    intents = [
        OrderIntent(OrderRequest("AAA.T", OrderSide.BUY, 2), "pair"),
        OrderIntent(OrderRequest("BBB.T", OrderSide.SELL, 2), "pair"),
    ]
    events = [
        _event(at, _book("AAA.T", at), _book("BBB.T", at)) for at in [start, deadline]
    ]

    result = simulate_adaptive_policy(intents, events, deadline=deadline)

    assert result.fills == []
    assert [outcome.unresolved_quantity for outcome in result.outcomes] == [2, 2]


@pytest.mark.parametrize("simulate", [simulate_market_baseline, simulate_adaptive_policy])
def test_exposure_area_revalues_fills_and_carries_last_mark_to_deadline(simulate):
    start = datetime.fromisoformat("2026-10-07T09:10:00+09:00")
    events = [
        _event(at, _book("AAA.T", at, bid=mid - 0.01, ask=mid + 0.01))
        for at, mid in [(start, 100), (start + timedelta(seconds=10), 110),
                        (start + timedelta(seconds=20), 120),
                        (start + timedelta(seconds=40), 999)]
    ]
    intent = OrderIntent(OrderRequest("AAA.T", OrderSide.BUY, 1))

    result = simulate([intent], events, deadline=start + timedelta(seconds=30))

    assert result.max_abs_net_exposure_jpy == pytest.approx(120)
    assert result.net_exposure_area_jpy_seconds == pytest.approx(3300)
    assert result.metrics()["net_exposure_status"] == "COMPLETE"


@pytest.mark.parametrize("simulate", [simulate_market_baseline, simulate_adaptive_policy])
def test_exposure_peak_includes_both_sequential_legs_within_one_event(simulate):
    start = datetime.fromisoformat("2026-10-07T09:10:00+09:00")
    intents = [
        OrderIntent(OrderRequest("AAA.T", OrderSide.BUY, 1), "pair"),
        OrderIntent(OrderRequest("BBB.T", OrderSide.SELL, 1), "pair"),
    ]
    event = _event(start, *[_book(t, start, bid=99.99, ask=100.01) for t in ["AAA.T", "BBB.T"]])

    result = simulate(intents, [event], deadline=start + timedelta(seconds=30))

    assert len(result.fills) == 2
    assert result.max_abs_net_exposure_jpy == pytest.approx(100)
    assert result.net_exposure_area_jpy_seconds == 0


def _capture(at: datetime, *, complete: bool = True) -> dict:
    return {
        "observed_at": at.isoformat(),
        "status": "OBSERVED" if at.second == 0 and at.minute == 10 else "OBSERVED_OUTSIDE_0910_WINDOW",
        "window_valid": at.minute == 10 and at.second <= 30,
        "rows": [_book(ticker, at).to_dict() for ticker in JP_TICKERS],
        "trades": [{"source": "synthetic"}],
        "queue_events": [{"source": "synthetic"}],
        "passive_evidence_complete": complete,
    }


def _audit_records(tmp_path, records: list[dict], deadline: datetime) -> dict:
    import json

    quotes = tmp_path / "quotes.jsonl"
    quotes.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    orders = tmp_path / "orders.csv"
    pd.DataFrame([{"ticker": JP_TICKERS[0], "action": "BUY", "quantity": 1,
                   "trade_date": "2026-10-07"}]).to_csv(orders, index=False)
    positions = tmp_path / "positions.json"
    positions.write_text(json.dumps({"trade_date": "2026-10-07", "positions": {}}), encoding="utf-8")
    return build_audit(quote_path=quotes, output_dir=tmp_path, orders_path=orders,
                       positions_path=positions, deadline_text=deadline.isoformat())


def test_replay_cli_keeps_0911_deadline_and_later_adverse_selection_mark(tmp_path):
    start = datetime.fromisoformat("2026-10-07T09:10:00+09:00")
    deadline = start + timedelta(seconds=60)
    records = [_capture(start), _capture(deadline), _capture(deadline + timedelta(seconds=5))]

    payload = _audit_records(tmp_path, records, deadline)

    assert payload["eligible_replay_event_count"] == 3
    assert payload["eligible_arrival_record_count"] == 1
    assert payload["simulation_status"] == "COMPLETE"
    assert payload["comparison"]["adaptive"]["filled_quantity"] == 1
    assert payload["comparison"]["adaptive"]["adverse_selection_observed_fill_count"] == 1


def test_later_quote_cannot_replace_missing_0910_arrival(tmp_path):
    at = datetime.fromisoformat("2026-10-07T09:11:00+09:00")

    payload = _audit_records(tmp_path, [_capture(at)], at)

    assert payload["simulation_status"] == "NOT_EVALUABLE_NO_ELIGIBLE_ARRIVAL_BOOK"
    assert payload["comparison"] is None


def test_replay_cli_retains_missing_queue_interval_instead_of_skipping_it(tmp_path):
    start = datetime.fromisoformat("2026-10-07T09:10:00+09:00")
    deadline = start + timedelta(seconds=60)
    gap = _capture(start + timedelta(seconds=35), complete=False)
    gap.pop("queue_events")

    payload = _audit_records(tmp_path, [_capture(start), gap, _capture(deadline)], deadline)

    assert payload["eligible_replay_event_count"] == 3
    assert payload["simulation_status"] == "INCOMPLETE_DATA"
    assert payload["comparison"]["adaptive"]["unresolved_quantity"] == 1


def test_replay_cli_rejects_invalid_book_inside_resting_interval(tmp_path):
    start = datetime.fromisoformat("2026-10-07T09:10:00+09:00")
    deadline = start + timedelta(seconds=60)
    gap = _capture(start + timedelta(seconds=35))
    gap["rows"][0]["ask_price_5"] = None

    payload = _audit_records(tmp_path, [_capture(start), gap, _capture(deadline)], deadline)

    assert payload["simulation_status"] == "NOT_EVALUABLE_INVALID_REPLAY_EVENTS"
    assert payload["comparison"] is None
