"""Pricing helpers for the execution layer.

This module was split from ``execution/helpers.py`` as part of P1-B1 to
isolate open-price resolution and fill-price fetching from broker/client,
output, and risk concerns.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime

from leadlag.broker.base import BrokerClient
from leadlag.data.tickers import JP_TICKERS, TOPIX_TICKER
from leadlag.execution.config import StrategyConfig as ProductionConfig

logger = logging.getLogger(__name__)


def resolve_daily_open_prices(
    api_client: BrokerClient | None,
    config: ProductionConfig,
    opens_csv: str | None,
    use_google_opens: bool,
) -> tuple[dict[str, float], float | None]:
    """Fetch JP open prices with API -> Google -> CSV fallback mechanism.

    Used by both ``decision.py`` and ``fast.py``.
    """
    from leadlag.data.market_data import (
        fetch_opens_from_google as _fetch_opens_from_google,
    )
    from leadlag.data.market_data import (
        load_opens_from_csv as _load_opens_from_csv,
    )
    from leadlag.data.market_data import (
        validate_manual_opens as _validate_manual_opens,
    )
    from leadlag.data.market_data import (
        validate_topix_open as _validate_topix_open,
    )

    tickers_for_opens = JP_TICKERS
    if config.signal_mode == "gap_residual":
        tickers_for_opens = JP_TICKERS + [TOPIX_TICKER]

    if api_client is not None:
        logger.info("Fetching JP opens from broker API...")
        manual_opens = api_client.fetch_open_prices(tickers_for_opens, allow_missing=True)
        missing = [tk for tk in tickers_for_opens if tk not in manual_opens]
        if missing:
            logger.warning(
                "Falling back to Google Finance for %d ticker(s): %s",
                len(missing),
                ", ".join(missing),
            )
            google_fetched = _fetch_opens_from_google(tickers=missing, allow_missing=True)
            manual_opens.update(google_fetched)
            missing_jp = [tk for tk in JP_TICKERS if tk not in manual_opens]
            if missing_jp:
                raise ValueError(
                    "Missing open prices after API + Google fallback: " + ", ".join(missing_jp)
                )
        logger.info("Resolved open prices for %d tickers", len(manual_opens))
    elif use_google_opens:
        logger.info("Fetching JP current real-time prices from Google Finance...")
        manual_opens = _fetch_opens_from_google(tickers=tickers_for_opens)
    elif opens_csv is not None:
        logger.info("Loading JP opens from CSV...")
        manual_opens = _load_opens_from_csv(opens_csv)
    else:
        raise ValueError(
            "--jp-opens-csv or --google-opens is required when API is not enabled. "
            "Either provide a CSV file, use --google-opens, or use --api-enable."
        )

    _validate_manual_opens(manual_opens)
    topix_open = None
    if config.signal_mode == "gap_residual":
        topix_open = _validate_topix_open(manual_opens)

    return manual_opens, topix_open


def fetch_fill_prices(
    api_client: BrokerClient,
    order_results: list[dict],
    *,
    wait_seconds: float = 3.0,
) -> list[dict]:
    """Fetch fill prices for submitted orders via CLMOrderListDetail.

    For each order in *order_results* (containing ``order_id``), queries the
    broker for fill details and enriches the dict with:
      - ``fill_price``: 約定単価 (float or None)
      - ``fill_quantity``: 約定株数 (int or None)
      - ``fill_status``: 約定ステータス (str)
      - ``fill_detail``: raw API response (dict)

    Args:
        api_client: BrokerClient with get_order_detail support.
        order_results: List of order result dicts from submit_orders_via_api.
        wait_seconds: Delay before fetching fills (allows exchange processing).

    Returns:
        Enriched order_results with fill information.
    """
    from leadlag.broker.tachibana.client import TachibanaBrokerClient

    if not isinstance(api_client, TachibanaBrokerClient):
        logger.debug("Fill price fetch not supported for this broker type, skipping.")
        return order_results

    if not order_results:
        return order_results

    logger.info("[HEARTBEAT] Waiting %.1f seconds before fetching fill prices", wait_seconds)
    time.sleep(wait_seconds)

    for result in order_results:
        order_id = result.get("order_id", "")
        if not order_id or result.get("status") != "SUBMITTED":
            result["fill_price"] = None
            result["fill_quantity"] = None
            result["fill_status"] = "NOT_SUBMITTED"
            continue

        eigyou_day = result.get("eigyou_day") or datetime.now().strftime("%Y%m%d")
        try:
            detail = api_client.get_order_detail(order_id, eigyou_day)
            fill_price_str = detail.get("sYakuzyouPrice", "0.0000")
            fill_qty_str = detail.get("sYakuzyouSuryou", "0")
            fill_status = detail.get("sOrderStatus", "")

            fill_price = float(fill_price_str) if fill_price_str and fill_price_str != "0.0000" else None
            fill_quantity = int(fill_qty_str) if fill_qty_str else None

            result["fill_price"] = fill_price
            result["fill_quantity"] = fill_quantity
            result["fill_status"] = fill_status
            result["fill_detail"] = {
                "sYakuzyouPrice": fill_price_str,
                "sYakuzyouSuryou": fill_qty_str,
                "sOrderStatus": fill_status,
                "sOrderYakuzyouStatus": detail.get("sOrderYakuzyouStatus"),
                "sBaiBaiDaikin": detail.get("sBaiBaiDaikin"),
                "sBaiBaiTesuryo": detail.get("sBaiBaiTesuryo"),
                "aYakuzyouSikkouList": detail.get("aYakuzyouSikkouList"),
                "aKessaiOrderTategyokuList": detail.get("aKessaiOrderTategyokuList"),
            }

            logger.info(
                "  [FILL] %s: %d shares @ %s (Order ID: %s, Status: %s)",
                result.get("ticker"),
                fill_quantity,
                fill_price,
                order_id,
                fill_status,
            )
        except Exception as e:
            logger.warning("Failed to fetch fill detail for order %s: %s", order_id, e)
            result["fill_price"] = None
            result["fill_quantity"] = None
            result["fill_status"] = "FETCH_ERROR"

    return order_results
