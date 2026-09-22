"""Pure P&L and inventory accounting primitives.

This module has no broker, filesystem, or network dependency.  It contains
the two accounting contracts used by the project:

* :func:`simulate_daily_pnl` is the weight-based backtest model.  It keeps the
  existing daily return and cost semantics while making the calculation
  reusable outside ``BacktestEngine``.
* :class:`InventoryLedger` consumes observed or simulated fills.  It keeps
  FIFO lots, allocates explicit fees once, and reports realized and
  mark-to-market P&L without applying a second slippage assumption to an
  observed fill price.

The backtest and live report adapters are intentionally outside this module.
They translate their existing DataFrame/JSON representations into these
small contracts at the boundary.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, cast

import numpy as np
import pandas as pd


def _date_string(value: str | date | datetime | pd.Timestamp) -> str:
    """Normalize an accounting date without attaching an availability claim."""
    return cast(str, pd.Timestamp(value).date().isoformat())


def _finite_nonnegative(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative: {value!r}")
    return result


@dataclass(frozen=True)
class Fill:
    """One observed or simulated execution fill.

    ``price`` is the actual fill price for ``source='observed'``.  It already
    contains the realized market impact; callers must not apply the backtest
    slippage assumption to it again.  ``fee`` is an explicit currency amount
    charged by the broker for this fill.
    """

    trade_date: str | date | datetime | pd.Timestamp
    ticker: str
    side: str
    quantity: int
    price: float
    fee: float = 0.0
    source: str = "observed"
    order_id: str | None = None

    def __post_init__(self) -> None:
        side = str(self.side).upper()
        if side not in {"BUY", "SELL"}:
            raise ValueError(f"Unsupported fill side: {self.side!r}")
        if not str(self.ticker):
            raise ValueError("Fill ticker must not be empty")
        if int(self.quantity) <= 0:
            raise ValueError("Fill quantity must be positive")
        price = float(self.price)
        if not np.isfinite(price) or price <= 0.0:
            raise ValueError(f"Fill price must be positive and finite: {self.price!r}")
        _finite_nonnegative(self.fee, "Fill fee")
        object.__setattr__(self, "trade_date", _date_string(self.trade_date))
        object.__setattr__(self, "side", side)
        object.__setattr__(self, "quantity", int(self.quantity))
        object.__setattr__(self, "price", price)
        object.__setattr__(self, "fee", float(self.fee))
        object.__setattr__(self, "source", str(self.source))


@dataclass(frozen=True)
class FeeAccrual:
    """A fee explicitly charged or accrued against an inventory event."""

    trade_date: str | date | datetime | pd.Timestamp
    ticker: str
    kind: str
    amount: float
    source: str
    fill_order_id: str | None = None

    def __post_init__(self) -> None:
        if not str(self.ticker):
            raise ValueError("Fee ticker must not be empty")
        _finite_nonnegative(self.amount, "Fee amount")
        object.__setattr__(self, "trade_date", _date_string(self.trade_date))
        object.__setattr__(self, "amount", float(self.amount))
        object.__setattr__(self, "kind", str(self.kind))
        object.__setattr__(self, "source", str(self.source))


@dataclass(frozen=True)
class InventoryLot:
    """An open FIFO lot. ``side`` identifies long versus short inventory."""

    ticker: str
    side: str
    quantity: int
    entry_price: float
    entry_fee_per_unit: float
    opened_at: str
    source: str
    order_id: str | None = None

    def __post_init__(self) -> None:
        side = str(self.side).upper()
        if side not in {"BUY", "SELL"}:
            raise ValueError(f"Unsupported lot side: {self.side!r}")
        if int(self.quantity) <= 0:
            raise ValueError("Inventory quantity must be positive")
        if not np.isfinite(float(self.entry_price)) or float(self.entry_price) <= 0.0:
            raise ValueError("Inventory entry price must be positive and finite")
        _finite_nonnegative(self.entry_fee_per_unit, "Entry fee per unit")
        object.__setattr__(self, "side", side)
        object.__setattr__(self, "quantity", int(self.quantity))
        object.__setattr__(self, "entry_price", float(self.entry_price))
        object.__setattr__(self, "entry_fee_per_unit", float(self.entry_fee_per_unit))


@dataclass(frozen=True)
class RealizedPnlRecord:
    """One matched FIFO lot segment, including both sides' explicit fees."""

    trade_date: str
    ticker: str
    original_side: str
    close_side: str
    quantity: int
    original_price: float
    fill_price: float
    entry_fee: float
    exit_fee: float
    realized_pnl: float
    source: str
    order_id: str | None = None

    @property
    def fee(self) -> float:
        """Compatibility total fee used by the close report."""
        return self.entry_fee + self.exit_fee


@dataclass(frozen=True)
class UnrealizedPnlRecord:
    """Mark-to-market value of one open lot."""

    trade_date: str
    ticker: str
    side: str
    quantity: int
    entry_price: float
    mark_price: float
    entry_fee: float
    unrealized_pnl: float


@dataclass(frozen=True)
class DailyCostBreakdown:
    """One simulated day's costs in decimal return units.

    This is deliberately distinct from ``execution.cost_calculator.CostBreakdown``
    (basis points) and ``domain.portfolio.CostBreakdown`` (a decision-level
    decimal estimate).  The backtest's values are fractions of portfolio
    notional and are deducted from a return series.
    """

    slippage_return: float = 0.0
    financing_return: float = 0.0
    borrow_return: float = 0.0
    reverse_return: float = 0.0

    @property
    def unit(self) -> str:
        return "return_fraction"

    @property
    def total_return(self) -> float:
        return (
            self.slippage_return
            + self.financing_return
            + self.borrow_return
            + self.reverse_return
        )


def fill_from_record(
    record: Mapping[str, Any],
    *,
    source: str = "observed",
    trade_date: str | date | datetime | pd.Timestamp | None = None,
) -> Fill | None:
    """Translate a confirmed execution record into one observed :class:`Fill`.

    The outer broker/JSON formats use several aliases for quantity and fee.
    This adapter accepts those aliases once, rejects terminal rows without a
    price, and leaves benchmark slippage outside the observed fill price.
    ``trade_date`` is required either in the record or by the caller so an
    accounting row is never assigned the current wall-clock date implicitly.
    """
    status = str(record.get("status", "")).upper()
    if status in {"FAILED", "SKIPPED", "SIMULATED"}:
        return None
    raw_price = record.get("fill_price", record.get("price"))
    raw_quantity = record.get("fill_quantity", record.get("filled_quantity", record.get("quantity")))
    raw_date = record.get("trade_date") or record.get("executed_at") or trade_date
    ticker = str(record.get("ticker", ""))
    side = str(record.get("side", ""))
    if raw_price is None or raw_quantity is None or raw_date is None or not ticker or not side:
        return None
    try:
        quantity = int(raw_quantity)
        price = float(raw_price)
    except (TypeError, ValueError):
        return None
    if quantity <= 0 or not np.isfinite(price) or price <= 0.0:
        return None
    if status == "CANCELLED" and quantity <= 0:
        return None
    detail = record.get("fill_detail") or {}
    if not isinstance(detail, Mapping):
        detail = {}
    fee_provided = "sBaiBaiTesuryo" in detail or any(
        key in record for key in ("fee", "commission")
    )
    observed_source = source in {"observed", "observed_fill"}
    raw_fee = detail.get("sBaiBaiTesuryo") if "sBaiBaiTesuryo" in detail else record.get(
        "fee", record.get("commission")
    )
    if observed_source and (not fee_provided or raw_fee is None):
        # A broker fill without a fee is still an execution fact, but it is
        # not a complete realized-PnL fact.  Do not silently certify it as a
        # zero-fee trade.
        return None
    if raw_fee is None:
        raw_fee = 0.0
    try:
        fee = float(raw_fee)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(fee) or fee < 0.0:
        return None
    try:
        return Fill(
            trade_date=raw_date,
            ticker=ticker,
            side=side,
            quantity=quantity,
            price=price,
            fee=fee,
            source=source,
            order_id=str(record.get("order_id")) if record.get("order_id") else None,
        )
    except (TypeError, ValueError):
        return None


class InventoryLedger:
    """FIFO inventory ledger for observed and simulated fills.

    The ledger is deliberately in-memory.  Durable order intent and broker
    observation state belong to ``ExecutionStateStore``; this class is the
    accounting projection built from confirmed fills.
    """

    def __init__(self) -> None:
        self._lots: dict[str, list[InventoryLot]] = defaultdict(list)
        self._realized: list[RealizedPnlRecord] = []
        self._fees: list[FeeAccrual] = []

    def apply_fill(self, fill: Fill) -> tuple[RealizedPnlRecord, ...]:
        """Apply a fill and return newly realized FIFO segments.

        A fill's fee is allocated per filled unit.  When it closes an existing
        lot, both the original lot fee and the closing fee are included in the
        realized record.  Any remaining quantity becomes a new lot with only
        its proportional entry fee.  This makes partial fills and reversals
        auditable without double counting a broker fee.
        """
        self._fees.append(
            FeeAccrual(
                trade_date=cast(str, fill.trade_date),
                ticker=fill.ticker,
                kind="broker_fill_fee",
                amount=fill.fee,
                source=fill.source,
                fill_order_id=fill.order_id,
            )
        )
        lots = self._lots[fill.ticker]
        remaining = fill.quantity
        fee_per_unit = fill.fee / fill.quantity
        realized: list[RealizedPnlRecord] = []

        while remaining > 0 and lots and lots[0].side != fill.side:
            lot = lots[0]
            matched = min(remaining, lot.quantity)
            entry_fee = lot.entry_fee_per_unit * matched
            exit_fee = fee_per_unit * matched
            price_pnl = (
                (fill.price - lot.entry_price) * matched
                if lot.side == "BUY"
                else (lot.entry_price - fill.price) * matched
            )
            record = RealizedPnlRecord(
                    trade_date=cast(str, fill.trade_date),
                ticker=fill.ticker,
                original_side=lot.side,
                close_side=fill.side,
                quantity=matched,
                original_price=lot.entry_price,
                fill_price=fill.price,
                entry_fee=entry_fee,
                exit_fee=exit_fee,
                realized_pnl=price_pnl - entry_fee - exit_fee,
                source=fill.source,
                order_id=fill.order_id,
            )
            realized.append(record)
            self._realized.append(record)
            remaining -= matched
            if matched == lot.quantity:
                lots.pop(0)
            else:
                lots[0] = InventoryLot(
                    ticker=lot.ticker,
                    side=lot.side,
                    quantity=lot.quantity - matched,
                    entry_price=lot.entry_price,
                    entry_fee_per_unit=lot.entry_fee_per_unit,
                    opened_at=lot.opened_at,
                    source=lot.source,
                    order_id=lot.order_id,
                )

        if remaining > 0:
            lots.append(
                InventoryLot(
                    ticker=fill.ticker,
                    side=fill.side,
                    quantity=remaining,
                    entry_price=fill.price,
                    entry_fee_per_unit=fee_per_unit,
                    opened_at=cast(str, fill.trade_date),
                    source=fill.source,
                    order_id=fill.order_id,
                )
            )
        return tuple(realized)

    def record_fee(self, accrual: FeeAccrual) -> None:
        """Record a non-trade fee (financing/borrow/reverse or tax).

        Callers add only confirmed or explicitly estimated accruals.  Missing
        broker data is intentionally not represented as a zero amount.
        """
        self._fees.append(accrual)

    def open_lots(self) -> tuple[InventoryLot, ...]:
        """Return open lots in deterministic FIFO order."""
        return tuple(lot for ticker in sorted(self._lots) for lot in self._lots[ticker])

    def fees(self) -> tuple[FeeAccrual, ...]:
        """Return explicit broker fee events applied to the ledger."""
        return tuple(self._fees)

    def realized(self) -> tuple[RealizedPnlRecord, ...]:
        """Return all realized FIFO segments."""
        return tuple(self._realized)

    def mark_to_market(
        self,
        mark_prices: dict[str, float],
        *,
        trade_date: str | date | datetime | pd.Timestamp,
    ) -> tuple[UnrealizedPnlRecord, ...]:
        """Mark every open lot; missing/invalid prices fail closed."""
        mark_date = _date_string(trade_date)
        records: list[UnrealizedPnlRecord] = []
        for lot in self.open_lots():
            if lot.ticker not in mark_prices:
                raise ValueError(f"Missing mark price for open ticker {lot.ticker}")
            mark = float(mark_prices[lot.ticker])
            if not np.isfinite(mark) or mark <= 0.0:
                raise ValueError(f"Invalid mark price for {lot.ticker}: {mark!r}")
            entry_fee = lot.entry_fee_per_unit * lot.quantity
            price_pnl = (
                (mark - lot.entry_price) * lot.quantity
                if lot.side == "BUY"
                else (lot.entry_price - mark) * lot.quantity
            )
            records.append(
                UnrealizedPnlRecord(
                    trade_date=mark_date,
                    ticker=lot.ticker,
                    side=lot.side,
                    quantity=lot.quantity,
                    entry_price=lot.entry_price,
                    mark_price=mark,
                    entry_fee=entry_fee,
                    unrealized_pnl=price_pnl - entry_fee,
                )
            )
        return tuple(records)

    @property
    def total_realized_pnl(self) -> float:
        return float(sum(item.realized_pnl for item in self._realized))

    @property
    def total_fees(self) -> float:
        return float(sum(item.amount for item in self._fees))


def simulate_daily_pnl(
    *,
    weights: np.ndarray,
    target_returns: np.ndarray,
    gap_returns: np.ndarray,
    sim_dates: pd.DatetimeIndex,
    slip: float,
    financing_daily: float,
    borrow_daily: float,
    reverse_daily: float,
    alpha_long: float,
    alpha_short: float,
    side_leverage: float = 1.0,
    oc_returns: np.ndarray | None = None,
    alpha_masks: np.ndarray | None = None,
    calendar_days: np.ndarray | None = None,
) -> dict[str, list[float]]:
    """Run the existing daily weight-based backtest cost model.

    This is intentionally a direct extraction of the former
    ``BacktestEngine._simulate_daily_pnl`` implementation.  Inputs are copied
    at the boundary so the accounting loop cannot mutate caller-owned arrays.
    ``alpha_masks`` supplies per-asset carry fractions for research callers;
    when omitted, long/short fractions are selected from the weight sign.
    ``calendar_days`` is an explicit override for legacy research assumptions;
    production callers use calendar gaps between ``sim_dates`` by default.
    """
    weights_arr = np.array(weights, dtype=float, copy=True)
    target_arr = np.array(target_returns, dtype=float, copy=True)
    gap_arr = np.array(gap_returns, dtype=float, copy=True)
    if weights_arr.ndim != 2 or target_arr.shape != weights_arr.shape or gap_arr.shape != weights_arr.shape:
        raise ValueError("weights, target_returns, and gap_returns must have the same 2-D shape")
    if oc_returns is not None and np.asarray(oc_returns).shape != weights_arr.shape:
        raise ValueError("oc_returns must have the same shape as weights when supplied")
    alpha_masks_arr = None if alpha_masks is None else np.array(alpha_masks, dtype=float, copy=True)
    if alpha_masks_arr is not None and alpha_masks_arr.shape != weights_arr.shape:
        raise ValueError("alpha_masks must have the same shape as weights when supplied")
    calendar_days_arr = None if calendar_days is None else np.array(calendar_days, dtype=float, copy=True)
    if calendar_days_arr is not None and calendar_days_arr.shape != (len(weights_arr),):
        raise ValueError("calendar_days must have one value per weight row when supplied")
    if len(sim_dates) != len(weights_arr):
        raise ValueError("sim_dates length must match the number of weight rows")

    n_sim_days, n_j = weights_arr.shape
    w_prev = np.zeros(n_j)
    held_prev = np.zeros(n_j)
    default_calendar_days = np.ones(n_sim_days)
    sim_dates_pd = pd.to_datetime(sim_dates)
    for i in range(n_sim_days - 1):
        default_calendar_days[i] = (sim_dates_pd[i + 1] - sim_dates_pd[i]).days
    held_days = calendar_days_arr if calendar_days_arr is not None else default_calendar_days

    result: dict[str, list[float]] = {
        "gross_returns": [],
        "net_returns": [],
        "costs": [],
        "slip_costs": [],
        "financing_costs": [],
        "borrow_costs": [],
        "reverse_costs": [],
        "overnight_returns": [],
        "gross_exps": [],
        "turnover": [],
    }
    if oc_returns is not None:
        result["gross_returns_oc"] = []
        result["net_returns_oc"] = []

    for i in range(n_sim_days):
        w_t = weights_arr[i]
        r_target_t = target_arr[i]
        days_held = held_days[i]
        gross_ret = side_leverage * float(np.sum(w_t * r_target_t))
        gross_exp = float(np.sum(np.abs(w_t)))
        alpha_mask = (
            alpha_masks_arr[i]
            if alpha_masks_arr is not None
            else np.where(w_t > 0, alpha_long, np.where(w_t < 0, alpha_short, 0.0))
        )
        overnight_ret = 0.0
        carry_enabled = np.any(alpha_mask > 0) or (
            alpha_masks_arr is None and (alpha_long > 0 or alpha_short > 0)
        )
        if carry_enabled and i < n_sim_days - 1:
            overnight_ret = side_leverage * float(np.sum(alpha_mask * w_t * gap_arr[i + 1]))
        turnover = float(np.sum(np.abs(w_t - w_prev)) / 2.0)
        held_t = alpha_mask * w_t
        opening_trade = np.sum(np.abs(w_t - held_prev))
        close_trade = np.sum(np.abs(w_t - held_t))
        slip_cost = side_leverage * slip * (opening_trade + close_trade)
        held_long = float(np.sum(alpha_mask * np.maximum(w_t, 0.0)))
        held_short = float(np.sum(alpha_mask * np.maximum(-w_t, 0.0)))
        fin_cost = side_leverage * held_long * financing_daily * days_held
        borrow_cost = side_leverage * held_short * borrow_daily * days_held
        reverse_cost = side_leverage * held_short * reverse_daily * days_held
        cost_breakdown = DailyCostBreakdown(
            slippage_return=slip_cost,
            financing_return=fin_cost,
            borrow_return=borrow_cost,
            reverse_return=reverse_cost,
        )
        cost = cost_breakdown.total_return
        net_ret = gross_ret + overnight_ret - cost

        result["gross_returns"].append(gross_ret + overnight_ret)
        result["net_returns"].append(net_ret)
        result["costs"].append(cost)
        result["slip_costs"].append(slip_cost)
        result["financing_costs"].append(fin_cost)
        result["borrow_costs"].append(borrow_cost)
        result["reverse_costs"].append(reverse_cost)
        result["overnight_returns"].append(overnight_ret)
        result["gross_exps"].append(gross_exp)
        result["turnover"].append(turnover)

        if oc_returns is not None:
            gross_ret_oc = side_leverage * float(np.sum(w_t * np.asarray(oc_returns)[i]))
            result["gross_returns_oc"].append(gross_ret_oc)
            result["net_returns_oc"].append(gross_ret_oc - cost)
        w_prev = w_t.copy()
        held_prev = held_t.copy()
    return result
