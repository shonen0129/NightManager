from __future__ import annotations

import logging
from typing import Any
from leadlag.execution.order_book_schema import OrderBookSnapshot

logger = logging.getLogger(__name__)

class LobNotAvailable(Exception):
    """Raised when an operation requires LOB data but it is not available."""
    pass


def compute_mid_price(snapshot: OrderBookSnapshot) -> float:
    """Computes mid price. Fallback to last_price if LOB is unavailable."""
    if snapshot.lob_available:
        if snapshot.bid_price_1 is not None and snapshot.ask_price_1 is not None:
            return (snapshot.bid_price_1 + snapshot.ask_price_1) / 2.0
    if snapshot.last_price is not None:
        return snapshot.last_price
    raise ValueError(f"No price data available to compute mid price for {snapshot.ticker}")


def compute_quoted_spread_bps(snapshot: OrderBookSnapshot) -> float:
    """Computes quoted spread in bps: (ask_1 - bid_1) / mid * 10000."""
    if not snapshot.lob_available:
        raise LobNotAvailable(f"LOB not available for {snapshot.ticker}")
    mid = compute_mid_price(snapshot)
    if mid <= 0:
        raise ValueError(f"Invalid mid price {mid} for {snapshot.ticker}")
    if snapshot.bid_price_1 is None or snapshot.ask_price_1 is None:
        raise ValueError(f"Missing bid_price_1 or ask_price_1 for {snapshot.ticker}")
    return (snapshot.ask_price_1 - snapshot.bid_price_1) / mid * 10000.0


def compute_depth_jpy(snapshot: OrderBookSnapshot, side: str, n_levels: int = 5) -> float:
    """Computes total book depth in JPY for first n_levels.

    side: "BUY" (to buy, we look at Ask side depth) or "SELL" (to sell/short, we look at Bid side depth).
    """
    if not snapshot.lob_available:
        raise LobNotAvailable(f"LOB not available for {snapshot.ticker}")

    total_depth = 0.0
    for i in range(1, n_levels + 1):
        if side.upper() == "BUY":
            price = getattr(snapshot, f"ask_price_{i}")
            size = getattr(snapshot, f"ask_size_{i}")
        else:
            price = getattr(snapshot, f"bid_price_{i}")
            size = getattr(snapshot, f"bid_size_{i}")

        if price is not None and size is not None:
            total_depth += price * size

    return total_depth


def estimate_market_order_fill_price(snapshot: OrderBookSnapshot, side: str, order_jpy: float) -> float:
    """Estimates the average execution price (VWAP) for a market order of size order_jpy.

    side: "BUY" (buy from asks) or "SELL" (sell to bids).
    """
    if not snapshot.lob_available:
        raise LobNotAvailable(f"LOB not available for {snapshot.ticker}")

    if order_jpy <= 0:
        return compute_mid_price(snapshot)

    remaining_jpy = order_jpy
    executed_qty = 0.0
    total_spent_jpy = 0.0

    last_price = None

    for i in range(1, 6):
        if side.upper() == "BUY":
            price = getattr(snapshot, f"ask_price_{i}")
            size = getattr(snapshot, f"ask_size_{i}")
        else:
            price = getattr(snapshot, f"bid_price_{i}")
            size = getattr(snapshot, f"bid_size_{i}")

        if price is None or size is None or price <= 0 or size <= 0:
            continue

        last_price = price
        level_capacity_jpy = price * size

        if remaining_jpy <= level_capacity_jpy:
            # Consume part of this level
            qty_to_exec = remaining_jpy / price
            executed_qty += qty_to_exec
            total_spent_jpy += remaining_jpy
            remaining_jpy = 0.0
            break
        else:
            # Consume entire level
            executed_qty += size
            total_spent_jpy += level_capacity_jpy
            remaining_jpy -= level_capacity_jpy

    if remaining_jpy > 0:
        # If the order is larger than the 5-level depth, execute the remainder at the 5th level price
        if last_price is None or last_price <= 0:
            last_price = compute_mid_price(snapshot)
        qty_to_exec = remaining_jpy / last_price
        executed_qty += qty_to_exec
        total_spent_jpy += remaining_jpy

    if executed_qty <= 0:
        return compute_mid_price(snapshot)

    return total_spent_jpy / executed_qty


def estimate_lob_slippage_bps(snapshot: OrderBookSnapshot, side: str, order_jpy: float) -> float:
    """Estimates one-way slippage in bps: abs(fill_price - mid) / mid * 10000."""
    if not snapshot.lob_available:
        raise LobNotAvailable(f"LOB not available for {snapshot.ticker}")

    mid = compute_mid_price(snapshot)
    if mid <= 0:
        return 0.0

    fill_price = estimate_market_order_fill_price(snapshot, side, order_jpy)
    return abs(fill_price - mid) / mid * 10000.0


def compute_order_to_depth_ratio(snapshot: OrderBookSnapshot, side: str, order_jpy: float, n_levels: int = 5) -> float:
    """Computes the ratio of the order JPY size to the total depth in JPY."""
    if not snapshot.lob_available:
        raise LobNotAvailable(f"LOB not available for {snapshot.ticker}")

    depth = compute_depth_jpy(snapshot, side, n_levels)
    if depth <= 0:
        return float('inf')
    return order_jpy / depth
