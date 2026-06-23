from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

@dataclass
class OrderBookSnapshot:
    ticker: str
    timestamp: str
    last_price: float | None = None
    bid_price_1: float | None = None
    bid_price_2: float | None = None
    bid_price_3: float | None = None
    bid_price_4: float | None = None
    bid_price_5: float | None = None
    bid_size_1: float | None = None
    bid_size_2: float | None = None
    bid_size_3: float | None = None
    bid_size_4: float | None = None
    bid_size_5: float | None = None
    ask_price_1: float | None = None
    ask_price_2: float | None = None
    ask_price_3: float | None = None
    ask_price_4: float | None = None
    ask_price_5: float | None = None
    ask_size_1: float | None = None
    ask_size_2: float | None = None
    ask_size_3: float | None = None
    ask_size_4: float | None = None
    ask_size_5: float | None = None
    lob_available: bool = False
    cost_source: str = "not_configured"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_quote(snapshot: OrderBookSnapshot) -> bool:
    """Validates the quote snapshot.

    If lob_available is True, bid_price_1 and ask_price_1 must be present, positive,
    and bid_price_1 < ask_price_1.
    If lob_available is False, last_price must be present and positive.
    """
    if snapshot.lob_available:
        if snapshot.bid_price_1 is None or snapshot.ask_price_1 is None:
            logger.warning(f"LOB is enabled but bid/ask is missing for {snapshot.ticker}")
            return False
        if snapshot.bid_price_1 <= 0 or snapshot.ask_price_1 <= 0:
            logger.warning(f"LOB is enabled but non-positive bid/ask for {snapshot.ticker}")
            return False
        if snapshot.bid_price_1 >= snapshot.ask_price_1:
            logger.warning(f"Invalid spread: bid {snapshot.bid_price_1} >= ask {snapshot.ask_price_1} for {snapshot.ticker}")
            return False
    else:
        if snapshot.last_price is None or snapshot.last_price <= 0:
            logger.warning(f"LOB not available and missing/invalid last_price for {snapshot.ticker}")
            return False
    return True


def from_api_price_response(api_item: dict[str, Any], timestamp: str | None = None) -> OrderBookSnapshot:
    """Creates an OrderBookSnapshot from Tachibana API get_price response item.

    Tachibana API fields:
      - sIssueCode: e.g. '1617' (we map this to ticker by appending '.T' if needed)
      - pDPP: Current price (now-price)
      - pPRP: Previous close
      - pDOP: Open price
      - pDHP: High price
      - pDLP: Low price
      - pDV: Volume
      - pGAP1..5: Ask price levels 1 to 5 (売り気配値段)
      - pGAV1..5: Ask size levels 1 to 5  (売り気配数量)
      - pGBP1..5: Bid price levels 1 to 5 (買い気配値段)
      - pGBV1..5: Bid size levels 1 to 5  (買い気配数量)
    """
    if timestamp is None:
        timestamp = datetime.now().isoformat()

    raw_code = api_item.get("sIssueCode", "")
    ticker = raw_code if raw_code.endswith(".T") else f"{raw_code}.T"

    def to_float_or_none(val: Any) -> float | None:
        if val is None or val == "" or str(val).strip() in ("*", "0", "0.0", "0.0000"):
            return None
        try:
            fval = float(val)
            return fval if fval != 0.0 else None
        except ValueError:
            return None

    last_price = to_float_or_none(api_item.get("pDPP"))
    if last_price is None:
        # Fallback to Open or Previous Close
        last_price = to_float_or_none(api_item.get("pDOP")) or to_float_or_none(api_item.get("pPRP"))

    # Extract 5-level bid/ask LOB data
    bid_prices = [to_float_or_none(api_item.get(f"pGBP{i}")) for i in range(1, 6)]
    bid_sizes  = [to_float_or_none(api_item.get(f"pGBV{i}")) for i in range(1, 6)]
    ask_prices = [to_float_or_none(api_item.get(f"pGAP{i}")) for i in range(1, 6)]
    ask_sizes  = [to_float_or_none(api_item.get(f"pGAV{i}")) for i in range(1, 6)]

    # LOB is available only if the best bid and best ask are both present and positive
    lob_available = (
        bid_prices[0] is not None
        and ask_prices[0] is not None
        and bid_prices[0] > 0
        and ask_prices[0] > 0
    )

    if lob_available:
        cost_source = "api_lob"
    elif last_price is not None:
        cost_source = "fixed_spread_fallback"
    else:
        cost_source = "api_error"

    return OrderBookSnapshot(
        ticker=ticker,
        timestamp=timestamp,
        last_price=last_price,
        bid_price_1=bid_prices[0],
        bid_price_2=bid_prices[1],
        bid_price_3=bid_prices[2],
        bid_price_4=bid_prices[3],
        bid_price_5=bid_prices[4],
        bid_size_1=bid_sizes[0],
        bid_size_2=bid_sizes[1],
        bid_size_3=bid_sizes[2],
        bid_size_4=bid_sizes[3],
        bid_size_5=bid_sizes[4],
        ask_price_1=ask_prices[0],
        ask_price_2=ask_prices[1],
        ask_price_3=ask_prices[2],
        ask_price_4=ask_prices[3],
        ask_price_5=ask_prices[4],
        ask_size_1=ask_sizes[0],
        ask_size_2=ask_sizes[1],
        ask_size_3=ask_sizes[2],
        ask_size_4=ask_sizes[3],
        ask_size_5=ask_sizes[4],
        lob_available=lob_available,
        cost_source=cost_source,
    )


def to_parquet_record(snapshot: OrderBookSnapshot) -> dict[str, Any]:
    """Converts snapshot to dict format matching the parquet schema requirements."""
    return snapshot.to_dict()
