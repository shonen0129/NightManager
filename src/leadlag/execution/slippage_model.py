from __future__ import annotations

import logging
from enum import Enum
from typing import Any
from leadlag.execution.order_book_schema import OrderBookSnapshot
from leadlag.execution.order_book_cost import estimate_lob_slippage_bps, compute_quoted_spread_bps, LobNotAvailable

logger = logging.getLogger(__name__)

class CostSource(str, Enum):
    LOB_SNAPSHOT = "lob_snapshot"
    FIXED_SPREAD_FALLBACK = "fixed_spread_fallback"
    NOT_CONFIGURED = "not_configured"
    API_ERROR = "api_error"


def compute_entry_cost_bps(
    snapshot: OrderBookSnapshot | None,
    order_jpy: float,
    side: str,
    fallback_roundtrip_bps: float = 15.0,
    config: dict[str, Any] | None = None
) -> tuple[float, CostSource]:
    """Computes one-way entry cost in bps.

    If LOB data is available: entry_cost = one_way_quoted_spread + slippage.
    Otherwise: entry_cost = fallback_roundtrip_bps / 2.0.
    """
    if snapshot is not None and snapshot.lob_available:
        try:
            one_way_spread = compute_quoted_spread_bps(snapshot) / 2.0
            slippage = estimate_lob_slippage_bps(snapshot, side, order_jpy)
            return one_way_spread + slippage, CostSource.LOB_SNAPSHOT
        except Exception as e:
            logger.warning(f"Error computing LOB entry cost for {snapshot.ticker}: {e}. Falling back.")
            return fallback_roundtrip_bps / 2.0, CostSource.API_ERROR

    # Fallback path
    source = CostSource.FIXED_SPREAD_FALLBACK
    if snapshot is not None:
        if snapshot.cost_source == "api_error":
            source = CostSource.API_ERROR
        elif snapshot.cost_source == "not_configured":
            source = CostSource.NOT_CONFIGURED

    return fallback_roundtrip_bps / 2.0, source


def compute_exit_cost_bps(
    snapshot: OrderBookSnapshot | None,
    order_jpy: float,
    side: str,
    fallback_roundtrip_bps: float = 15.0,
    config: dict[str, Any] | None = None
) -> tuple[float, CostSource]:
    """Computes one-way exit cost in bps.

    Since exits occur at the close, we always use the fixed fallback spread.
    """
    source = CostSource.FIXED_SPREAD_FALLBACK
    if snapshot is not None:
        if snapshot.cost_source == "api_error":
            source = CostSource.API_ERROR
        elif snapshot.cost_source == "not_configured":
            source = CostSource.NOT_CONFIGURED

    return fallback_roundtrip_bps / 2.0, source


def compute_financing_bps_daily(side: str, annual_rate: float, days: int = 1) -> float:
    """Computes daily financing cost in bps for long margin positions.

    Formula: annual_rate / 365.0 * 10000.0 * days
    """
    if side.upper() != "BUY":
        return 0.0
    return (annual_rate / 365.0) * 10000.0 * days


def compute_borrow_bps_daily(annual_rate: float, days: int = 1) -> float:
    """Computes daily borrow fee in bps for short positions.

    Formula: annual_rate / 365.0 * 10000.0 * days
    """
    return (annual_rate / 365.0) * 10000.0 * days


def compute_reverse_fee_bps(bps_per_day: float, days: int = 1) -> float:
    """Computes daily reverse stock lending fee (逆日歩) in bps."""
    return bps_per_day * days
