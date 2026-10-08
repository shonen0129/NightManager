"""Research-only, event-replay execution simulation.

This module builds the existing market-order baseline with ``broker_ops`` but
never creates a broker client or submits/cancels/retries an order. Passive
fills require explicit queue-aware evidence; touching a quote is not treated
as a fill.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from datetime import datetime, time, timedelta
from enum import StrEnum
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from leadlag.core.types import OrderRequest, OrderSide, OrderType
from leadlag.data.tickers import lot_size_for
from leadlag.execution.broker_ops import build_execution_plan
from leadlag.execution.microstructure.order_book_cost import (
    compute_depth_jpy,
    compute_quoted_spread_bps,
)
from leadlag.execution.microstructure.order_book_schema import OrderBookSnapshot


class Route(StrEnum):
    CROSS = "CROSS"
    PASSIVE = "PASSIVE"
    BLOCKED = "BLOCKED_BY_PAIR_EXPOSURE"
    UNKNOWN = "UNRESOLVED_FILL_EVIDENCE"


@dataclass(frozen=True)
class OrderIntent:
    request: OrderRequest
    pair_id: str = "portfolio"


@dataclass(frozen=True)
class PassiveFillEvidence:
    """Queue-aware fill capacity at one resting price for one replay event."""

    ticker: str
    side: OrderSide
    price: float
    quantity: int
    queue_position_known: bool
    source: str

    def __post_init__(self) -> None:
        if self.quantity < 0 or not math.isfinite(self.price) or self.price <= 0:
            raise ValueError(
                "passive fill evidence must have a positive finite price and nonnegative quantity"
            )
        if not self.queue_position_known:
            raise ValueError("passive fill evidence requires known queue position")
        if not self.source:
            raise ValueError("passive fill evidence must identify its source")


@dataclass(frozen=True)
class ReplayEvent:
    """One book update; complete evidence covers the preceding replay interval."""

    timestamp: datetime
    books: Mapping[str, OrderBookSnapshot]
    passive_fills: tuple[PassiveFillEvidence, ...] = ()
    passive_evidence_complete: bool = False

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError("replay timestamps must be timezone-aware")
        for ticker, book in self.books.items():
            if ticker != book.ticker:
                raise ValueError(f"book key {ticker} does not match snapshot ticker {book.ticker}")


@dataclass(frozen=True)
class PolicyConfig:
    """Initial policy constants; not calibrated or promoted to production."""

    urgent_last_seconds: float = 10.0
    wide_spread_bps: float = 20.0
    max_pair_net_ratio: float = 0.05
    adverse_selection_horizon_seconds: float = 5.0

    def __post_init__(self) -> None:
        if self.urgent_last_seconds < 0 or self.wide_spread_bps < 0:
            raise ValueError("urgency and spread thresholds must be nonnegative")
        if not 0 <= self.max_pair_net_ratio <= 1:
            raise ValueError("max_pair_net_ratio must be in [0, 1]")
        if self.adverse_selection_horizon_seconds < 0:
            raise ValueError("adverse selection horizon must be nonnegative")


@dataclass
class SimulatedFill:
    ticker: str
    side: OrderSide
    quantity: int
    price: float
    timestamp: datetime
    arrival_mid: float
    fill_mid: float
    best_executable_price: float
    source: str
    pair_id: str
    adverse_selection_jpy: float | None = None
    adverse_selection_markout_seconds: float | None = None

    @property
    def sign(self) -> int:
        return 1 if self.side == OrderSide.BUY else -1

    @property
    def shortfall_jpy(self) -> float:
        return self.sign * (self.price - self.arrival_mid) * self.quantity

    @property
    def delay_jpy(self) -> float:
        return self.sign * (self.fill_mid - self.arrival_mid) * self.quantity

    @property
    def spread_and_book_jpy(self) -> float:
        return self.sign * (self.price - self.fill_mid) * self.quantity

    @property
    def book_walk_jpy(self) -> float:
        return self.sign * (self.price - self.best_executable_price) * self.quantity


@dataclass
class OrderOutcome:
    ticker: str
    side: OrderSide
    requested_quantity: int
    filled_quantity: int = 0
    known_unfilled_quantity: int = 0
    unresolved_quantity: int = 0
    quantity_deferred_by_pair_cap: int = 0
    fills: list[SimulatedFill] = field(default_factory=list)
    routes: list[str] = field(default_factory=list)


@dataclass
class SimulationResult:
    mode: str
    arrival_time: datetime
    deadline: datetime
    outcomes: list[OrderOutcome]
    fills: list[SimulatedFill]
    arrival_mid_by_ticker: dict[str, float]
    terminal_mid_by_ticker: dict[str, float]
    terminal_mark_time_by_ticker: dict[str, datetime]
    max_abs_net_exposure_jpy: float
    net_exposure_area_jpy_seconds: float
    pair_quantity_adjusted_orders: int
    total_quantity_deferred_by_pair_cap: int

    def metrics(self) -> dict[str, Any]:
        filled_shortfall = sum(fill.shortfall_jpy for fill in self.fills)
        delay = sum(fill.delay_jpy for fill in self.fills)
        spread_and_book = sum(fill.spread_and_book_jpy for fill in self.fills)
        book_walk = sum(fill.book_walk_jpy for fill in self.fills)
        adverse = [
            fill.adverse_selection_jpy
            for fill in self.fills
            if fill.adverse_selection_jpy is not None
        ]
        known_unfilled = sum(outcome.known_unfilled_quantity for outcome in self.outcomes)
        unresolved = sum(outcome.unresolved_quantity for outcome in self.outcomes)
        opportunity_cost = 0.0
        opportunity_complete = unresolved == 0
        # Known unfilled opportunity cost is marked only when an exact deadline
        # quote exists for the ticker. A stale last quote remains unresolved.
        for outcome in self.outcomes:
            ticker, side = outcome.ticker, outcome.side
            qty = outcome.known_unfilled_quantity
            if not qty:
                continue
            mark = self.terminal_mid_by_ticker.get(ticker)
            mark_time = self.terminal_mark_time_by_ticker.get(ticker)
            if mark is None or mark_time != self.deadline:
                opportunity_complete = False
                continue
            reference = self.arrival_mid_by_ticker.get(ticker)
            if reference is None:
                opportunity_complete = False
                continue
            sign = 1 if side == OrderSide.BUY else -1
            opportunity_cost += sign * (mark - reference) * qty
        total = filled_shortfall + opportunity_cost
        complete = unresolved == 0 and opportunity_complete
        reference_complete = all(
            item.ticker in self.arrival_mid_by_ticker for item in self.outcomes
        )
        reference_notional = sum(
            item.requested_quantity * self.arrival_mid_by_ticker.get(item.ticker, 0.0)
            for item in self.outcomes
        )
        return {
            "mode": self.mode,
            "requested_quantity": sum(item.requested_quantity for item in self.outcomes),
            "filled_quantity": sum(item.filled_quantity for item in self.outcomes),
            "known_unfilled_quantity": known_unfilled,
            "unresolved_quantity": unresolved,
            "filled_shortfall_jpy": filled_shortfall,
            "filled_shortfall_bps": (
                filled_shortfall / reference_notional * 10000.0
                if reference_complete and reference_notional > 0
                else None
            ),
            "delay_price_movement_jpy": delay,
            "spread_and_book_cost_jpy": spread_and_book,
            "book_walk_beyond_touch_jpy": book_walk,
            "unfilled_opportunity_cost_jpy": opportunity_cost if opportunity_complete else None,
            "total_shortfall_jpy": total if complete else None,
            "total_shortfall_bps": (
                total / reference_notional * 10000.0
                if complete and reference_complete and reference_notional > 0
                else None
            ),
            "total_shortfall_status": "COMPLETE" if complete else "INCOMPLETE_DATA",
            "adverse_selection_jpy": sum(adverse) if len(adverse) == len(self.fills) else None,
            "adverse_selection_observed_fill_count": len(adverse),
            "fill_count": len(self.fills),
            "max_abs_net_exposure_jpy": self.max_abs_net_exposure_jpy,
            "net_exposure_area_jpy_seconds": self.net_exposure_area_jpy_seconds,
            "net_exposure_status": "INCOMPLETE_DATA" if unresolved else "COMPLETE",
            "pair_quantity_adjusted_orders": self.pair_quantity_adjusted_orders,
            "quantity_deferred_by_pair_cap": self.total_quantity_deferred_by_pair_cap,
        }


def snapshot_from_row(row: Mapping[str, Any]) -> OrderBookSnapshot:
    """Parse the existing quote-capture row format without filling missing data."""
    allowed = {item.name for item in fields(OrderBookSnapshot)}
    values = {key: value for key, value in row.items() if key in allowed}
    if not values.get("ticker") or not values.get("timestamp"):
        raise ValueError("book row requires ticker and timestamp")
    return OrderBookSnapshot(**values)


def replay_event_from_capture_record(record: Mapping[str, Any]) -> ReplayEvent:
    """Convert one captured cross-section to a replay event."""
    observed_at = record.get("observed_at")
    if not isinstance(observed_at, str):
        raise ValueError("capture record has no observed_at timestamp")
    timestamp = datetime.fromisoformat(observed_at)
    rows = record.get("rows")
    if not isinstance(rows, list):
        raise ValueError("capture record has no rows list")
    books = {
        str(row["ticker"]): snapshot_from_row(row)
        for row in rows
        if isinstance(row, dict) and row.get("ticker")
    }
    if not books:
        raise ValueError("capture record contains no ticker books")
    passive_fills = tuple(
        PassiveFillEvidence(
            ticker=str(item["ticker"]),
            side=OrderSide(str(item["side"]).upper()),
            price=float(item["price"]),
            quantity=int(item["quantity"]),
            queue_position_known=bool(item.get("queue_position_known")),
            source=str(item.get("source", "")),
        )
        for item in record.get("passive_fills", [])
        if isinstance(item, dict)
    )
    return ReplayEvent(
        timestamp=timestamp,
        books=books,
        passive_fills=passive_fills,
        passive_evidence_complete=(
            record.get("passive_evidence_complete") is True
            and bool(record.get("trades"))
            and bool(record.get("queue_events"))
        ),
    )


def build_order_intents(
    decisions: pd.DataFrame,
    *,
    current_positions: Mapping[str, int],
    pair_ids: Mapping[str, str] | None = None,
) -> list[OrderIntent]:
    """Use broker_ops' pure execution-plan builder for the market baseline."""
    required = {"ticker", "action", "quantity"}
    missing = required.difference(decisions.columns)
    if missing:
        raise ValueError(f"decisions missing columns: {sorted(missing)}")
    plan = build_execution_plan(decisions.copy(), dict(current_positions))
    pair_ids = pair_ids or {}
    grouped: dict[tuple[str, OrderSide, str], OrderRequest] = {}
    for req in [*plan.close_orders, *plan.new_orders]:
        pair_id = str(pair_ids.get(req.ticker, "portfolio"))
        key = (req.ticker, req.side, pair_id)
        previous = grouped.get(key)
        grouped[key] = (
            req
            if previous is None
            else OrderRequest(
                ticker=req.ticker,
                side=req.side,
                quantity=previous.quantity + req.quantity,
                order_type=previous.order_type,
                limit_price=previous.limit_price,
                is_close=previous.is_close and req.is_close,
            )
        )
    return [
        OrderIntent(request=req, pair_id=pair_id)
        for (ticker, side, pair_id), req in grouped.items()
    ]


def assess_capture_record(
    record: Mapping[str, Any], *, required_tickers: Sequence[str]
) -> dict[str, Any]:
    """Separate arrival eligibility from subsequent same-session book updates.

    Missing trade/queue coverage stays visible to the simulator rather than
    removing the event and silently bridging an unobserved passive interval.
    """
    rows = record.get("rows")
    row_tickers = (
        {str(row.get("ticker")) for row in rows if isinstance(row, dict)}
        if isinstance(rows, list)
        else set()
    )
    full_depth = bool(rows) and all(
        isinstance(row, dict)
        and row.get("lob_available") is True
        and all(
            row.get(f"{side}_{field}_{level}") is not None
            for side in ("bid", "ask")
            for field in ("price", "size")
            for level in range(1, 6)
        )
        for row in rows
    )
    time_valid = False
    replay_time_valid = False
    observed_at = record.get("observed_at")
    if isinstance(observed_at, str):
        try:
            parsed = datetime.fromisoformat(observed_at)
            if parsed.tzinfo is not None:
                local = parsed.astimezone(ZoneInfo("Asia/Tokyo"))
                clock = local.timetz().replace(tzinfo=None)
                replay_time_valid = clock >= time(9, 10) and record.get("status") in {
                    "OBSERVED", "OBSERVED_OUTSIDE_0910_WINDOW"
                }
                time_valid = (
                    time(9, 10) <= clock <= time(9, 10, 30)
                    and record.get("window_valid") is True
                    and record.get("status") == "OBSERVED"
                )
        except ValueError:
            time_valid = False
    reasons: list[str] = []
    if not replay_time_valid:
        reasons.append("not_a_valid_09:10_capture")
    if not set(required_tickers).issubset(row_tickers):
        reasons.append("required_ticker_books_missing")
    if not full_depth:
        reasons.append("complete_five_level_depth_missing")
    if not isinstance(rows, list) or len(row_tickers) != len(rows):
        reasons.append("duplicate_or_invalid_ticker_rows")
    if isinstance(rows, list) and isinstance(observed_at, str):
        if any(not isinstance(row, dict) or row.get("timestamp") != observed_at for row in rows):
            reasons.append("row_timestamps_misaligned")
    passive_reasons: list[str] = []
    if not record.get("trades"):
        passive_reasons.append("trade_tape_missing")
    if not record.get("queue_events") or record.get("passive_evidence_complete") is not True:
        passive_reasons.append("queue_position_or_depletion_events_missing")
    return {
        "observed_at": record.get("observed_at"),
        "status": record.get("status"),
        "window_valid": bool(record.get("window_valid")),
        "valid_09_10_capture": time_valid,
        "book_ticker_count": len(row_tickers),
        "full_five_level_depth": full_depth,
        "trade_tape_present": bool(record.get("trades")),
        "queue_events_present": bool(record.get("queue_events"))
        and bool(record.get("passive_evidence_complete")),
        "eligible_for_execution_replay": not reasons,
        "eligible_for_arrival": time_valid and not reasons,
        "passive_evidence_reasons": passive_reasons,
        "reasons": reasons,
    }


def simulate_market_baseline(
    orders: Sequence[OrderIntent],
    events: Sequence[ReplayEvent],
    *,
    deadline: datetime,
    config: PolicyConfig = PolicyConfig(),
) -> SimulationResult:
    """Immediate market-order baseline, bounded by the displayed book only."""
    _validate_replay(orders, events, deadline)
    arrival = events[0]
    fills: list[SimulatedFill] = []
    outcomes: list[OrderOutcome] = []
    for intent in orders:
        req = intent.request
        outcome = OrderOutcome(req.ticker, req.side, req.quantity)
        outcomes.append(outcome)
        book = arrival.books.get(req.ticker)
        if book is None or not _has_two_sided_book(book):
            outcome.unresolved_quantity = req.quantity
            continue
        arrival_mid = _execution_mid(book)
        legs, remaining, best = _sweep(book, req.side, req.quantity)
        for qty, price in legs:
            fill = SimulatedFill(
                req.ticker,
                req.side,
                qty,
                price,
                arrival.timestamp,
                arrival_mid,
                arrival_mid,
                best or price,
                "displayed_market_depth",
                intent.pair_id,
            )
            fills.append(fill)
            outcome.fills.append(fill)
            outcome.filled_quantity += qty
        # The visible five levels do not prove that the unshown remainder did
        # not fill at deeper levels; keep it unresolved rather than impute.
        outcome.unresolved_quantity = remaining
        outcome.routes.append(Route.CROSS.value)
    max_net, area = _exposure_metrics(fills, events, deadline)
    result = SimulationResult(
        "market_baseline",
        arrival.timestamp,
        deadline,
        outcomes,
        fills,
        {
            ticker: _execution_mid(book)
            for ticker, book in arrival.books.items()
            if _has_two_sided_book(book)
        },
        {
            ticker: _execution_mid(book)
            for event in events if event.timestamp <= deadline
            for ticker, book in event.books.items()
            if _has_two_sided_book(book)
        },
        {
            ticker: event.timestamp
            for event in events if event.timestamp <= deadline
            for ticker, book in event.books.items()
            if _has_two_sided_book(book)
        },
        max_net,
        area,
        0,
        0,
    )
    _attach_adverse_selection(result, events, config.adverse_selection_horizon_seconds)
    return result


def simulate_adaptive_policy(
    orders: Sequence[OrderIntent],
    events: Sequence[ReplayEvent],
    *,
    deadline: datetime,
    config: PolicyConfig = PolicyConfig(),
) -> SimulationResult:
    """Replay the adaptive limit/market policy against timestamped book events."""
    _validate_replay(orders, events, deadline)
    arrival = events[0]
    arrival_mids = {
        ticker: _execution_mid(book)
        for ticker, book in arrival.books.items()
        if _has_two_sided_book(book)
    }
    outcomes = [OrderOutcome(i.request.ticker, i.request.side, i.request.quantity) for i in orders]
    fills: list[SimulatedFill] = []
    resting_limits: dict[int, float] = {}
    last_mids: dict[str, float] = {}
    last_mark_times: dict[str, datetime] = {}
    adjusted_orders: set[int] = set()
    unknown_pairs: set[str] = set()
    target_gross_by_pair: dict[str, float] = {}
    maximum_unit_notional_by_pair: dict[str, float] = {}
    pair_sides: dict[str, set[OrderSide]] = {}
    for intent in orders:
        mid = arrival_mids.get(intent.request.ticker)
        if mid is not None:
            target_gross_by_pair[intent.pair_id] = (
                target_gross_by_pair.get(intent.pair_id, 0.0) + mid * intent.request.quantity
            )
            pair_sides.setdefault(intent.pair_id, set()).add(intent.request.side)
            unit_notional = mid * lot_size_for(intent.request.ticker)
            maximum_unit_notional_by_pair[intent.pair_id] = max(
                maximum_unit_notional_by_pair.get(intent.pair_id, 0.0), unit_notional
            )

    for event in events:
        if event.timestamp > deadline:
            break
        # A later complete event cannot reconstruct a missing interval. Freeze
        # the affected pair before deciding either leg from an unknown net.
        for idx in resting_limits:
            if outcomes[idx].filled_quantity >= orders[idx].request.quantity:
                continue
            book = event.books.get(orders[idx].request.ticker)
            if not event.passive_evidence_complete or book is None or not _has_two_sided_book(book):
                outcomes[idx].unresolved_quantity = (
                    orders[idx].request.quantity - outcomes[idx].filled_quantity
                )
                unknown_pairs.add(orders[idx].pair_id)
                outcomes[idx].routes.append(Route.UNKNOWN.value)
        for ticker, book in event.books.items():
            if _has_two_sided_book(book):
                last_mids[ticker] = _execution_mid(book)
                last_mark_times[ticker] = event.timestamp

        # Process the leg that reduces current pair imbalance first.
        current_marks = {**arrival_mids, **last_mids}
        current_net = _pair_net_exposure(orders, outcomes, event.books, current_marks)
        ordered_indices = sorted(
            range(len(orders)),
            key=lambda idx: _order_priority(orders[idx], current_net.get(orders[idx].pair_id, 0.0)),
        )
        for idx in ordered_indices:
            intent = orders[idx]
            req = intent.request
            outcome = outcomes[idx]
            if outcome.unresolved_quantity:
                continue
            remaining_qty = req.quantity - outcome.filled_quantity
            if remaining_qty <= 0:
                continue
            if intent.pair_id in unknown_pairs:
                outcome.unresolved_quantity = remaining_qty
                outcome.routes.append(Route.UNKNOWN.value)
                continue
            book = event.books.get(req.ticker)
            if book is None or not _has_two_sided_book(book):
                continue
            mid = _execution_mid(book)
            cap = _pair_cap(
                intent.pair_id,
                target_gross_by_pair,
                maximum_unit_notional_by_pair,
                pair_sides,
                config,
            )
            net = _pair_net_exposure(orders, outcomes, event.books, current_marks).get(
                intent.pair_id, 0.0
            )
            allowed = _allowed_quantity(req.ticker, req.side, remaining_qty, mid, net, cap)
            if allowed < remaining_qty:
                adjusted_orders.add(idx)
                deferred = remaining_qty - allowed
                outcome.quantity_deferred_by_pair_cap = max(
                    outcome.quantity_deferred_by_pair_cap, deferred
                )
            if allowed <= 0:
                outcome.routes.append(Route.BLOCKED.value)
                continue

            # Explicit queue-aware evidence can fill a resting order. Quote
            # touches alone are intentionally ignored.
            limit_price = resting_limits.get(idx)
            if limit_price is not None:
                evidence_qty = sum(
                    item.quantity
                    for item in event.passive_fills
                    if item.ticker == req.ticker
                    and item.side == req.side
                    and math.isclose(item.price, limit_price, rel_tol=0.0, abs_tol=1e-9)
                    and item.queue_position_known
                )
                passive_qty = min(evidence_qty, allowed)
                lot = lot_size_for(req.ticker)
                passive_qty = (passive_qty // lot) * lot
                if passive_qty > 0:
                    arrival_mid = arrival_mids.get(req.ticker)
                    if arrival_mid is None:
                        continue
                    best_same_side = _best_same_side(book, req.side)
                    fill = SimulatedFill(
                        req.ticker,
                        req.side,
                        passive_qty,
                        limit_price,
                        event.timestamp,
                        arrival_mid,
                        mid,
                        best_same_side or limit_price,
                        "queue_aware_passive_evidence",
                        intent.pair_id,
                    )
                    fills.append(fill)
                    outcome.fills.append(fill)
                    outcome.filled_quantity += passive_qty
                    remaining_qty -= passive_qty
                    allowed -= passive_qty
                # An event is only proof of no passive fills when the feed
                # explicitly declares complete queue-aware evidence.
                if event.passive_evidence_complete and remaining_qty > 0:
                    outcome.routes.append(Route.PASSIVE.value)

            if remaining_qty <= 0 or allowed <= 0:
                continue
            remaining_seconds = max(0.0, (deadline - event.timestamp).total_seconds())
            spread = _spread_bps(book)
            depth_coverage = _depth_coverage(book, req.side, remaining_qty, mid)
            reduces_imbalance = (
                req.side == OrderSide.SELL and net > 0 or req.side == OrderSide.BUY and net < 0
            )
            urgency = remaining_seconds <= config.urgent_last_seconds
            pair_imbalance = cap is not None and abs(net) >= cap * 0.5 and reduces_imbalance
            existing_limit = resting_limits.get(idx)
            already_marketable = existing_limit is not None and _is_marketable(
                book, req.side, existing_limit
            )
            choose_passive = (
                not already_marketable
                and not urgency
                and not pair_imbalance
                and (
                    (spread is not None and spread >= config.wide_spread_bps)
                    or (depth_coverage is not None and depth_coverage < 1.0)
                )
            )
            if choose_passive:
                resting_limits[idx] = _best_same_side(book, req.side)
                outcome.routes.append(Route.PASSIVE.value)
                continue

            # A resting limit becomes a marketable limit only if the contra
            # quote actually crosses its limit; otherwise deadline urgency
            # sends a displayed-depth marketable order.
            marketable_limit = resting_limits.get(idx)
            if marketable_limit is not None and _is_marketable(book, req.side, marketable_limit):
                legs, remaining_after, best = _sweep(
                    book, req.side, allowed, limit_price=marketable_limit
                )
                fill_source = "marketable_limit_displayed_depth"
            elif urgency or pair_imbalance or not choose_passive:
                legs, remaining_after, best = _sweep(book, req.side, allowed)
                fill_source = "adaptive_cross_displayed_depth"
            else:
                continue
            arrival_mid = arrival_mids.get(req.ticker)
            if arrival_mid is None:
                continue
            for qty, price in legs:
                fill = SimulatedFill(
                    req.ticker,
                    req.side,
                    qty,
                    price,
                    event.timestamp,
                    arrival_mid,
                    mid,
                    best or price,
                    fill_source,
                    intent.pair_id,
                )
                fills.append(fill)
                outcome.fills.append(fill)
                outcome.filled_quantity += qty
            outcome.routes.append(Route.CROSS.value)
            # Do not decide that hidden depth did not fill. Preserve the
            # remainder as unresolved; subsequent snapshots may resolve intent.
            if remaining_after > 0:
                outcome.unresolved_quantity = req.quantity - outcome.filled_quantity
                unknown_pairs.add(intent.pair_id)
            resting_limits.pop(idx, None)

    # Pending passive orders are unfilled only if the feed declares queue
    # evidence complete through an exact deadline mark. Otherwise they remain
    # unresolved, never zero-filled by default.
    last_event_at_or_before_deadline = next(
        (e for e in reversed(events) if e.timestamp <= deadline), None
    )
    exact_deadline = (
        last_event_at_or_before_deadline is not None
        and last_event_at_or_before_deadline.timestamp == deadline
    )
    passive_complete_at_deadline = bool(
        last_event_at_or_before_deadline
        and last_event_at_or_before_deadline.passive_evidence_complete
    )
    for outcome in outcomes:
        residual = outcome.requested_quantity - outcome.filled_quantity
        if residual <= 0:
            continue
        if outcome.unresolved_quantity:
            continue
        if exact_deadline and passive_complete_at_deadline:
            outcome.known_unfilled_quantity = residual
        else:
            outcome.unresolved_quantity = residual

    max_net, exposure_area = _exposure_metrics(fills, events, deadline)
    result = SimulationResult(
        "adaptive_policy",
        arrival.timestamp,
        deadline,
        outcomes,
        fills,
        arrival_mids,
        last_mids,
        last_mark_times,
        max_net,
        exposure_area,
        len(adjusted_orders),
        sum(item.quantity_deferred_by_pair_cap for item in outcomes),
    )
    _attach_adverse_selection(result, events, config.adverse_selection_horizon_seconds)
    return result


def compare_results(baseline: SimulationResult, adaptive: SimulationResult) -> dict[str, Any]:
    """Return paired aggregate deltas without hiding incomplete components."""
    base = baseline.metrics()
    candidate = adaptive.metrics()
    base_total = base["total_shortfall_jpy"]
    candidate_total = candidate["total_shortfall_jpy"]
    return {
        "baseline": base,
        "adaptive": candidate,
        "adaptive_minus_baseline_shortfall_jpy": (
            candidate_total - base_total
            if candidate_total is not None and base_total is not None
            else None
        ),
        "shortfall_delta_status": (
            "COMPLETE"
            if candidate_total is not None and base_total is not None
            else "INCOMPLETE_DATA"
        ),
    }


def _validate_replay(
    orders: Sequence[OrderIntent], events: Sequence[ReplayEvent], deadline: datetime
) -> None:
    if not orders:
        raise ValueError("at least one order intent is required")
    if not events:
        raise ValueError("at least one book event is required")
    if deadline.tzinfo is None:
        raise ValueError("deadline must be timezone-aware")
    if list(events) != sorted(events, key=lambda event: event.timestamp):
        raise ValueError("replay events must be sorted by timestamp")
    if deadline < events[0].timestamp:
        raise ValueError("deadline cannot precede arrival event")
    arrival_local = events[0].timestamp.astimezone(ZoneInfo("Asia/Tokyo"))
    arrival_clock = arrival_local.timetz().replace(tzinfo=None)
    if not time(9, 10) <= arrival_clock <= time(9, 10, 30):
        raise ValueError("arrival event must be a valid 09:10:00-09:10:30 JST quote")
    session_dates = {event.timestamp.astimezone(ZoneInfo("Asia/Tokyo")).date() for event in events}
    if len(session_dates) != 1:
        raise ValueError("replay events must belong to one JST trading session")
    if deadline.astimezone(ZoneInfo("Asia/Tokyo")).date() != arrival_local.date():
        raise ValueError("deadline must be on the replay session date")
    if len({event.timestamp for event in events}) != len(events):
        raise ValueError("duplicate replay timestamps are not allowed")
    for intent in orders:
        if intent.request.quantity <= 0:
            raise ValueError("order quantities must be positive")
        if intent.request.order_type != OrderType.MARKET:
            raise ValueError("the baseline contract only accepts market-order intents")
        if intent.request.quantity % lot_size_for(intent.request.ticker):
            raise ValueError(f"order quantity violates lot size for {intent.request.ticker}")
    order_keys = [(item.request.ticker, item.request.side) for item in orders]
    if len(set(order_keys)) != len(order_keys):
        raise ValueError(
            "aggregate duplicate ticker/side orders before replay to avoid reusing displayed depth"
        )


def _sweep(
    book: OrderBookSnapshot,
    side: OrderSide,
    quantity: int,
    *,
    limit_price: float | None = None,
) -> tuple[list[tuple[int, float]], int, float | None]:
    levels: list[tuple[float, float]] = []
    previous_price: float | None = None
    for level in range(1, 6):
        if side == OrderSide.BUY:
            price = getattr(book, f"ask_price_{level}")
            size = getattr(book, f"ask_size_{level}")
        else:
            price = getattr(book, f"bid_price_{level}")
            size = getattr(book, f"bid_size_{level}")
        if price is None or size is None or price <= 0 or size <= 0:
            break
        if not float(size).is_integer():
            raise ValueError(f"fractional displayed share size is unsupported for {book.ticker}")
        if previous_price is not None:
            if side == OrderSide.BUY and price < previous_price:
                raise ValueError(f"ask levels are not ascending for {book.ticker}")
            if side == OrderSide.SELL and price > previous_price:
                raise ValueError(f"bid levels are not descending for {book.ticker}")
        if limit_price is not None and (
            price > limit_price if side == OrderSide.BUY else price < limit_price
        ):
            break
        levels.append((float(price), float(size)))
        previous_price = float(price)
    legs: list[tuple[int, float]] = []
    lot = lot_size_for(book.ticker)
    visible_capacity = sum(int(size) for _, size in levels)
    fill_capacity = min(quantity, visible_capacity)
    total_fill = (fill_capacity // lot) * lot
    remaining_to_allocate = total_fill
    best = levels[0][0] if levels else None
    for price, size in levels:
        fill_qty = min(remaining_to_allocate, int(size))
        if fill_qty > 0:
            legs.append((fill_qty, price))
            remaining_to_allocate -= fill_qty
        if remaining_to_allocate == 0:
            break
    return legs, quantity - total_fill, best


def _spread_bps(book: OrderBookSnapshot) -> float | None:
    if not _has_two_sided_book(book):
        return None
    try:
        return compute_quoted_spread_bps(book)
    except (ValueError, RuntimeError):
        return None


def _depth_coverage(
    book: OrderBookSnapshot, side: OrderSide, quantity: int, mid: float
) -> float | None:
    if not _has_two_sided_book(book):
        return None
    try:
        depth = compute_depth_jpy(book, side.value, n_levels=5)
    except (ValueError, RuntimeError):
        return None
    notional = quantity * mid
    return depth / notional if notional > 0 else None


def _best_same_side(book: OrderBookSnapshot, side: OrderSide) -> float:
    value = book.bid_price_1 if side == OrderSide.BUY else book.ask_price_1
    if value is None or value <= 0:
        raise ValueError(f"no same-side quote for passive order {book.ticker}")
    return float(value)


def _is_marketable(book: OrderBookSnapshot, side: OrderSide, limit_price: float) -> bool:
    contra = book.ask_price_1 if side == OrderSide.BUY else book.bid_price_1
    if contra is None:
        return False
    return limit_price >= contra if side == OrderSide.BUY else limit_price <= contra


def _pair_cap(
    pair_id: str,
    target_gross_by_pair: Mapping[str, float],
    maximum_unit_notional_by_pair: Mapping[str, float],
    pair_sides: Mapping[str, set[OrderSide]],
    config: PolicyConfig,
) -> float | None:
    if len(pair_sides.get(pair_id, set())) < 2:
        return None
    # A balanced pair cannot start if its permitted transient net is below
    # one legal lot on either leg; expose this unavoidable lot granularity.
    return max(
        target_gross_by_pair.get(pair_id, 0.0) * config.max_pair_net_ratio,
        maximum_unit_notional_by_pair.get(pair_id, 0.0),
    )


def _allowed_quantity(
    ticker: str, side: OrderSide, remaining: int, mid: float, net: float, cap: float | None
) -> int:
    if cap is None:
        return remaining
    max_additional = cap - net if side == OrderSide.BUY else cap + net
    if max_additional <= 0:
        return 0
    lot = lot_size_for(ticker)
    lot_adjusted = (int(math.floor(max_additional / mid)) // lot) * lot
    return min(remaining, lot_adjusted)


def _pair_net_exposure(
    orders: Sequence[OrderIntent],
    outcomes: Sequence[OrderOutcome],
    books: Mapping[str, OrderBookSnapshot],
    arrival_mids: Mapping[str, float],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for intent, outcome in zip(orders, outcomes, strict=True):
        if outcome.filled_quantity == 0:
            continue
        book = books.get(intent.request.ticker)
        mid = (
            _execution_mid(book)
            if book is not None and _has_two_sided_book(book)
            else arrival_mids.get(intent.request.ticker)
        )
        if mid is None:
            continue
        sign = 1 if intent.request.side == OrderSide.BUY else -1
        result[intent.pair_id] = (
            result.get(intent.pair_id, 0.0) + sign * outcome.filled_quantity * mid
        )
    return result


def _order_priority(intent: OrderIntent, pair_net: float) -> tuple[int, str]:
    sign = 1 if intent.request.side == OrderSide.BUY else -1
    reduces = sign * pair_net < 0
    return (0 if reduces else 1, intent.request.ticker)


def _exposure_metrics(
    fills: Sequence[SimulatedFill], events: Sequence[ReplayEvent], deadline: datetime
) -> tuple[float, float]:
    """Mark known fills identically in both modes, in simulated fill order.

    Event mids are held until the next observed mark (or the deadline). Fills
    at the same timestamp have zero elapsed time but each contributes to peak
    exposure. Unknown fills are excluded; metrics() labels that incompleteness.
    """
    quantities: dict[str, int] = {}
    mids: dict[str, float] = {}
    points: list[tuple[datetime, float]] = []
    fill_index = 0
    for event in events:
        if event.timestamp > deadline:
            break
        mids.update({ticker: _execution_mid(book) for ticker, book in event.books.items()
                     if _has_two_sided_book(book)})
        points.append((event.timestamp, sum(qty * mids[ticker] for ticker, qty in quantities.items())))
        while fill_index < len(fills) and fills[fill_index].timestamp == event.timestamp:
            fill = fills[fill_index]
            quantities[fill.ticker] = quantities.get(fill.ticker, 0) + fill.sign * fill.quantity
            points.append((event.timestamp, sum(qty * mids[ticker] for ticker, qty in quantities.items())))
            fill_index += 1
    if points and points[-1][0] < deadline:
        points.append((deadline, points[-1][1]))
    if not points:
        return 0.0, 0.0
    maximum = max(abs(value) for _, value in points)
    area = 0.0
    for (t0, v0), (t1, _v1) in zip(points, points[1:]):
        area += abs(v0) * max(0.0, (t1 - t0).total_seconds())
    return maximum, area


def _attach_adverse_selection(
    result: SimulationResult, events: Sequence[ReplayEvent], horizon_seconds: float
) -> None:
    for fill in result.fills:
        target = fill.timestamp + timedelta(seconds=horizon_seconds)
        later = next((event for event in events if event.timestamp >= target), None)
        if later is None:
            continue
        book = later.books.get(fill.ticker)
        if book is None or not _has_two_sided_book(book):
            continue
        mark = _execution_mid(book)
        fill.adverse_selection_jpy = fill.sign * (mark - fill.fill_mid) * fill.quantity
        fill.adverse_selection_markout_seconds = (later.timestamp - fill.timestamp).total_seconds()


def _has_two_sided_book(book: OrderBookSnapshot) -> bool:
    bid = book.bid_price_1
    ask = book.ask_price_1
    return bool(
        book.lob_available and bid is not None and ask is not None and bid > 0 and ask >= bid
    )


def _execution_mid(book: OrderBookSnapshot) -> float:
    if not _has_two_sided_book(book):
        raise ValueError(f"valid two-sided quote required for {book.ticker}")
    return (float(book.bid_price_1) + float(book.ask_price_1)) / 2.0
