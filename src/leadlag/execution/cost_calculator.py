"""Unified cost calculation module.

Consolidates entry/exit spread, slippage, financing, borrow, and reverse fee
calculations into a single interface.  Wraps the low-level functions in
``slippage_model.py`` and ``order_book_cost.py`` so that all callers
(``net_score_ranking_lob``, ``backtester``, sprint scripts) share one
consistent cost model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from leadlag.execution.microstructure.order_book_schema import OrderBookSnapshot
from leadlag.execution.microstructure.slippage_model import (
    CostSource,
    compute_entry_cost_bps,
    compute_exit_cost_bps,
    compute_financing_bps_daily,
    compute_borrow_bps_daily,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CostConfig:
    """Configuration for cost calculations.

    All defaults mirror the values used across sprint configs and
    ``ProductionV2RunConfig``.
    """
    fallback_roundtrip_bps: float = 15.0
    buy_interest_rate_annual: float = 0.025
    stock_borrow_fee_annual: float = 0.0115
    holding_days: int = 1
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> CostConfig:
        """Build CostConfig from a top-level strategy config dict.

        Reads from ``cost_aware_optimization`` section if present,
        otherwise falls back to defaults.
        """
        cost_opt = config.get("cost_aware_optimization", {})
        return cls(
            fallback_roundtrip_bps=cost_opt.get("default_spread_fallback_roundtrip_bps", 15.0),
            buy_interest_rate_annual=cost_opt.get("buy_interest_rate_annual", 0.025),
            stock_borrow_fee_annual=cost_opt.get("stock_borrow_fee_annual", 0.0115),
            holding_days=cost_opt.get("holding_days", 1),
        )


@dataclass
class CostBreakdown:
    """Detailed cost breakdown for a single position (in bps)."""
    entry_cost_bps: float
    exit_cost_bps: float
    financing_bps: float
    borrow_bps: float
    reverse_bps: float
    cost_source: CostSource

    @property
    def total_roundtrip_bps(self) -> float:
        """Total round-trip cost in bps (entry + exit + holding costs)."""
        return (
            self.entry_cost_bps
            + self.exit_cost_bps
            + self.financing_bps
            + self.borrow_bps
            + self.reverse_bps
        )

    @property
    def total_roundtrip_decimal(self) -> float:
        """Total round-trip cost as a decimal fraction of position notional."""
        return self.total_roundtrip_bps / 10000.0


class CostCalculator:
    """Unified cost calculator for all strategy components.

    Usage::

        calc = CostCalculator.from_config(cfg)
        breakdown = calc.compute_cost(
            snapshot=snap,
            side="BUY",
            order_jpy=500_000,
            reverse_fee_bps=2.0,
        )
        net_mu = signal - breakdown.total_roundtrip_decimal
    """

    def __init__(self, cost_config: CostConfig) -> None:
        self.cfg = cost_config
        self._financing_bps = compute_financing_bps_daily(
            "BUY",
            cost_config.buy_interest_rate_annual,
            cost_config.holding_days,
        )
        self._borrow_bps = compute_borrow_bps_daily(
            cost_config.stock_borrow_fee_annual,
            cost_config.holding_days,
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> CostCalculator:
        """Create from a top-level strategy config dict."""
        return cls(CostConfig.from_config(config))

    def compute_entry_cost(
        self,
        snapshot: OrderBookSnapshot | None,
        order_jpy: float,
        side: str,
    ) -> tuple[float, CostSource]:
        """One-way entry cost in bps (spread + slippage or fallback)."""
        return compute_entry_cost_bps(
            snapshot,
            order_jpy,
            side,
            self.cfg.fallback_roundtrip_bps,
        )

    def compute_exit_cost(
        self,
        snapshot: OrderBookSnapshot | None,
        order_jpy: float,
        side: str,
    ) -> tuple[float, CostSource]:
        """One-way exit cost in bps (always fallback spread at close)."""
        return compute_exit_cost_bps(
            snapshot,
            order_jpy,
            side,
            self.cfg.fallback_roundtrip_bps,
        )

    def compute_cost(
        self,
        snapshot: OrderBookSnapshot | None,
        side: str,
        order_jpy: float,
        reverse_fee_bps: float = 0.0,
    ) -> CostBreakdown:
        """Full cost breakdown for a position.

        Args:
            snapshot: LOB snapshot (or None for fallback).
            side: "BUY" (long) or "SELL" (short).
            order_jpy: Order size in JPY.
            reverse_fee_bps: Daily reverse stock lending fee in bps.

        Returns:
            CostBreakdown with all components in bps.
        """
        exit_side = "SELL" if side.upper() == "BUY" else "BUY"

        entry_bps, cost_src = self.compute_entry_cost(snapshot, order_jpy, side)
        exit_bps, _ = self.compute_exit_cost(snapshot, order_jpy, exit_side)

        if side.upper() == "BUY":
            financing = self._financing_bps
            borrow = 0.0
        else:
            financing = 0.0
            borrow = self._borrow_bps

        return CostBreakdown(
            entry_cost_bps=entry_bps,
            exit_cost_bps=exit_bps,
            financing_bps=financing,
            borrow_bps=borrow,
            reverse_bps=reverse_fee_bps,
            cost_source=cost_src,
        )

    def compute_net_signal(
        self,
        signal: float,
        snapshot: OrderBookSnapshot | None,
        side: str,
        order_jpy: float,
        reverse_fee_bps: float = 0.0,
    ) -> tuple[float, CostBreakdown]:
        """Compute net signal after deducting all costs.

        Args:
            signal: Raw signal value (expected return in decimal).
            snapshot: LOB snapshot or None.
            side: "BUY" or "SELL".
            order_jpy: Order size in JPY.
            reverse_fee_bps: Reverse stock lending fee in bps/day.

        Returns:
            Tuple of (net_signal_decimal, CostBreakdown).
        """
        breakdown = self.compute_cost(snapshot, side, order_jpy, reverse_fee_bps)
        total_cost_dec = breakdown.total_roundtrip_decimal

        if side.upper() == "BUY":
            net = signal - total_cost_dec
        else:
            net = -signal - total_cost_dec

        return net, breakdown
