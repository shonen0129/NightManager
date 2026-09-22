"""Typed boundaries between a portfolio decision and broker execution.

The execution pipeline historically passed a mixture of DataFrames, tuples,
and JSON dictionaries between planning, submission, polling, and reporting.
These small immutable contracts make the boundary explicit while keeping the
existing CSV/JSON representations available at the outer edges.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from leadlag.core.pnl import Fill
from leadlag.core.types import OrderRequest, OrderStatus


@dataclass(frozen=True)
class ExecutionPlan:
    """Orders derived from one target decision and one position snapshot."""

    decision_id: str
    trade_date: str
    close_orders: tuple[OrderRequest, ...]
    new_orders: tuple[OrderRequest, ...]
    current_positions: tuple[tuple[str, int], ...] = ()
    target_net_exposure: float | None = None
    target_gross_exposure: float | None = None

    @property
    def expected_order_count(self) -> int:
        return len(self.close_orders) + len(self.new_orders)

    @property
    def all_orders(self) -> tuple[OrderRequest, ...]:
        return self.close_orders + self.new_orders

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON-compatible representation used in logs."""
        return {
            "decision_id": self.decision_id,
            "trade_date": self.trade_date,
            "close_orders": [_order_to_dict(order) for order in self.close_orders],
            "new_orders": [_order_to_dict(order) for order in self.new_orders],
            "current_positions": dict(self.current_positions),
            "target_net_exposure": self.target_net_exposure,
            "target_gross_exposure": self.target_gross_exposure,
            "expected_order_count": self.expected_order_count,
        }


@dataclass(frozen=True)
class OrderObservation:
    """One broker observation kept separate from the order intent."""

    order_id: str
    ticker: str
    status: OrderStatus | str
    requested_quantity: int
    filled_quantity: int | None = None
    remaining_quantity: int | None = None
    observed_at: str | None = None
    raw_reference: str | None = None
    side: str | None = None
    fill_price: float | None = None
    fee: float | None = None
    fill_id: str | None = None
    fill_source: str = "observed"

    def to_dict(self) -> dict[str, Any]:
        status = self.status.value if isinstance(self.status, OrderStatus) else str(self.status)
        return {
            "order_id": self.order_id,
            "ticker": self.ticker,
            "status": status,
            "requested_quantity": self.requested_quantity,
            "filled_quantity": self.filled_quantity,
            "remaining_quantity": self.remaining_quantity,
            "observed_at": self.observed_at,
            "raw_reference": self.raw_reference,
            "side": self.side,
            "fill_price": self.fill_price,
            "fee": self.fee,
            "fill_id": self.fill_id,
            "fill_source": self.fill_source,
        }

    def to_fill(self, *, trade_date: str | None = None) -> Fill | None:
        """Convert a price-bearing observation into one confirmed Fill.

        Observations without a broker fill price or date remain status facts;
        they are not treated as zero-price executions.
        """
        if self.fill_price is None or self.filled_quantity is None:
            return None
        if not self.side or not trade_date:
            return None
        if self.fee is None and self.fill_source == "observed":
            return None
        try:
            return Fill(
                trade_date=trade_date,
                ticker=self.ticker,
                side=self.side,
                quantity=self.filled_quantity,
                price=self.fill_price,
                fee=self.fee or 0.0,
                source=self.fill_source,
                order_id=self.order_id,
            )
        except (TypeError, ValueError):
            return None


@dataclass(frozen=True)
class ExecutionReport:
    """Typed summary of broker outcomes and reconciliation completeness."""

    expected_orders: int
    accepted_orders: int
    filled_orders: int
    partial_orders: int
    failed_orders: int
    unresolved_orders: int
    close_failed: bool = False
    recording_errors: tuple[str, ...] = ()
    reconciliation_errors: tuple[str, ...] = ()
    observations: tuple[OrderObservation, ...] = ()

    @property
    def incomplete(self) -> bool:
        return bool(
            self.recording_errors
            or self.reconciliation_errors
            or self.close_failed
            or self.failed_orders
            or self.unresolved_orders
            or self.accepted_orders < self.expected_orders
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_orders": self.expected_orders,
            "accepted_orders": self.accepted_orders,
            "filled_orders": self.filled_orders,
            "partial_orders": self.partial_orders,
            "failed_orders": self.failed_orders,
            "unresolved_orders": self.unresolved_orders,
            "close_failed": self.close_failed,
            "recording_errors": list(self.recording_errors),
            "reconciliation_errors": list(self.reconciliation_errors),
            "observations": [observation.to_dict() for observation in self.observations],
            "incomplete": self.incomplete,
        }

    def observed_fills(self, *, trade_date: str | None = None) -> tuple[Fill, ...]:
        """Return only observations with enough data for P&L accounting."""
        fills = [
            fill
            for observation in self.observations
            if (fill := observation.to_fill(trade_date=trade_date)) is not None
        ]
        return tuple(fills)

    def with_reconciliation_errors(self, errors: Sequence[str]) -> ExecutionReport:
        """Return a copy with post-submission reconciliation errors attached."""
        return ExecutionReport(
            expected_orders=self.expected_orders,
            accepted_orders=self.accepted_orders,
            filled_orders=self.filled_orders,
            partial_orders=self.partial_orders,
            failed_orders=self.failed_orders,
            unresolved_orders=self.unresolved_orders,
            close_failed=self.close_failed,
            recording_errors=self.recording_errors,
            reconciliation_errors=tuple(errors),
            observations=self.observations,
        )


def _order_to_dict(order: OrderRequest) -> dict[str, Any]:
    return {
        "ticker": order.ticker,
        "side": order.side.value,
        "quantity": order.quantity,
        "order_type": order.order_type.value,
        "limit_price": order.limit_price,
        "margin_trade_type": order.margin_trade_type,
        "account_type": order.account_type,
        "is_close": order.is_close,
        "close_position_order": order.close_position_order,
    }


def build_decision_id(
    trade_date: Any,
    close_orders: Sequence[OrderRequest],
    new_orders: Sequence[OrderRequest],
) -> str:
    """Build a deterministic id without using process-randomized ``hash()``."""
    payload = {
        "trade_date": str(trade_date),
        "close_orders": [_order_to_dict(order) for order in close_orders],
        "new_orders": [_order_to_dict(order) for order in new_orders],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]


def report_from_records(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_orders: int,
    close_failed: bool = False,
    recording_errors: Sequence[str] = (),
    reconciliation_errors: Sequence[str] = (),
) -> ExecutionReport:
    """Create an ``ExecutionReport`` from legacy result dictionaries."""
    accepted = {"SUBMITTED", "PARTIALLY_FILLED", "FILLED", "SIMULATED"}
    filled = {"FILLED", "SIMULATED"}
    partial = {"PARTIALLY_FILLED"}
    unresolved = {"SUBMITTED", "PARTIALLY_FILLED"}
    failed = {"FAILED", "CANCELLED", "SKIPPED"}
    statuses = [str(record.get("status", "")) for record in records]
    observations = tuple(
        _observation_from_record(record)
        for record in records
    )
    return ExecutionReport(
        expected_orders=int(expected_orders),
        accepted_orders=sum(status in accepted for status in statuses),
        filled_orders=sum(status in filled for status in statuses),
        partial_orders=sum(status in partial for status in statuses),
        failed_orders=sum(status in failed for status in statuses),
        unresolved_orders=sum(status in unresolved for status in statuses),
        close_failed=bool(close_failed),
        recording_errors=tuple(recording_errors),
        reconciliation_errors=tuple(reconciliation_errors),
        observations=observations,
    )


def _observation_from_record(record: Mapping[str, Any]) -> OrderObservation:
    """Normalize one legacy result row into an immutable broker observation."""
    requested = int(record.get("quantity", 0) or 0)
    filled_raw = record.get("fill_quantity", record.get("filled_quantity"))
    filled = int(filled_raw) if filled_raw is not None else None
    remaining_raw = record.get("remaining_quantity", record.get("remaining_qty"))
    remaining = int(remaining_raw) if remaining_raw is not None else None
    if remaining is None and filled is not None:
        remaining = max(0, requested - filled)
    status = record.get("status", "")
    fill_detail = record.get("fill_detail") or {}
    if not isinstance(fill_detail, Mapping):
        fill_detail = {}
    raw_price = record.get("fill_price")
    raw_fee = fill_detail.get(
        "sBaiBaiTesuryo",
        record.get("fee", record.get("commission")),
    )
    try:
        fill_price = float(raw_price) if raw_price is not None else None
    except (TypeError, ValueError):
        fill_price = None
    try:
        fee = float(raw_fee) if raw_fee is not None else None
    except (TypeError, ValueError):
        fee = None
    return OrderObservation(
        order_id=str(record.get("order_id", "")),
        ticker=str(record.get("ticker", "")),
        status=status if isinstance(status, OrderStatus) else str(status),
        requested_quantity=requested,
        filled_quantity=filled,
        remaining_quantity=remaining,
        observed_at=(
            str(record.get("observed_at"))
            if record.get("observed_at") is not None
            else None
        ),
        raw_reference=(
            str(record.get("raw_reference"))
            if record.get("raw_reference") is not None
            else None
        ),
        side=str(record.get("side")) if record.get("side") is not None else None,
        fill_price=fill_price,
        fee=fee,
        fill_id=(
            str(record.get("fill_id", record.get("execution_id")))
            if record.get("fill_id", record.get("execution_id")) is not None
            else None
        ),
        fill_source=str(
            record.get(
                "fill_source",
                "simulated" if str(status).upper() == "SIMULATED" else "observed",
            )
        ),
    )
