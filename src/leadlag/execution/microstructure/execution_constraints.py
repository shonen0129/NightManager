from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any
from .order_book_schema import OrderBookSnapshot
from .order_book_cost import (
    compute_quoted_spread_bps,
    estimate_lob_slippage_bps
)

logger = logging.getLogger(__name__)

@dataclass
class ExecutionDecision:
    ticker: str
    side: str
    selected: bool
    skip_reason: str | None = None
    scale_reason: str | None = None
    scale_factor: float = 1.0
    quoted_spread_bps: float | None = None
    estimated_slippage_bps: float | None = None
    depth_jpy: float | None = None
    order_depth_ratio: float | None = None


def apply_hard_rules(
    snapshot: OrderBookSnapshot | None,
    side: str,
    order_jpy: float,
    short_available: bool = True,
    reverse_fee_bps: float = 0.0,
    config: dict[str, Any] | None = None
) -> ExecutionDecision:
    """Applies execution hard rules and determines if order should be skipped or scaled.

    Rules applied:
      1. Short availability: if side is SELL and short_available is False, skip.
      2. Quoted spread cap: if LOB is available and spread > max_quoted_spread_bps, skip.
      3. Slippage cap: if LOB is available and slippage > max_estimated_slippage_bps, skip.
      4. Size-to-depth ratio: if LOB is available and order size > min_depth_ratio_scale * depth,
         scale down order weight.
    """
    config = config or {}
    exec_config = config.get("execution", {})
    max_quoted_spread = exec_config.get("max_quoted_spread_bps", 30.0)
    max_slippage = exec_config.get("max_estimated_slippage_bps", 20.0)
    min_depth_ratio_scale = exec_config.get("min_depth_ratio_scale", 1.5)
    lob_levels = exec_config.get("lob_depth_levels", 5)

    ticker = snapshot.ticker if snapshot else "UNKNOWN"
    decision = ExecutionDecision(ticker=ticker, side=side, selected=True)

    # Rule 1: Short availability
    if side.upper() == "SELL" and not short_available:
        decision.selected = False
        decision.skip_reason = "SHORT_UNAVAILABLE"
        return decision

    # If snapshot is missing or LOB is not available, we skip LOB constraints
    if snapshot is None or not snapshot.lob_available:
        return decision

    try:
        # Calculate LOB metrics
        spread = compute_quoted_spread_bps(snapshot)
        slippage = estimate_lob_slippage_bps(snapshot, side, order_jpy)

        from .order_book_cost import compute_depth_jpy
        depth = compute_depth_jpy(snapshot, side, lob_levels)
        ratio = order_jpy / depth if depth > 0 else float('inf')

        decision.quoted_spread_bps = spread
        decision.estimated_slippage_bps = slippage
        decision.depth_jpy = depth
        decision.order_depth_ratio = ratio

        # Rule 2: Spread Cap
        if spread > max_quoted_spread:
            decision.selected = False
            decision.skip_reason = f"SPREAD_EXCEEDS_CAP({spread:.1f}bps > {max_quoted_spread:.1f}bps)"
            return decision

        # Rule 3: Slippage Cap
        if slippage > max_slippage:
            decision.selected = False
            decision.skip_reason = f"SLIPPAGE_EXCEEDS_CAP({slippage:.1f}bps > {max_slippage:.1f}bps)"
            return decision

        # Rule 4: Depth Ratio Scaling
        if ratio > min_depth_ratio_scale:
            scale = min_depth_ratio_scale / ratio
            decision.scale_factor = scale
            decision.scale_reason = f"ORDER_DEPTH_RATIO_EXCEEDS_LIMIT({ratio:.2f} > {min_depth_ratio_scale})"

    except Exception as e:
        logger.warning(f"Error applying LOB rules for {ticker}: {e}")
        # On calculation error, do not fail. Proceed without scaling/skipping but log
        pass

    return decision


def replace_unavailable_short(
    selected_shorts: list[str],
    available_shorts_pool: list[str],
    max_shorts: int = 5
) -> list[str]:
    """Replaces any short candidates that are not borrowable.

    Args:
        selected_shorts: Initial ranked list of short candidate tickers.
        available_shorts_pool: The pool of all short-able tickers, in order of priority.
        max_shorts: Maximum number of shorts to select.
    """
    final_shorts = []
    # Identify which selected ones are available
    for ticker in selected_shorts:
        if ticker in available_shorts_pool:
            final_shorts.append(ticker)

    # Fill up to max_shorts from the pool
    if len(final_shorts) < max_shorts:
        for candidate in available_shorts_pool:
            if candidate not in final_shorts and candidate not in selected_shorts:
                final_shorts.append(candidate)
                if len(final_shorts) == max_shorts:
                    break

    return final_shorts[:max_shorts]
