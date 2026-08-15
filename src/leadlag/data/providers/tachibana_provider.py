"""Broker-backed 09:10 quote provider (Tachibana / kabu-station compatible).

The provider uses a ``BrokerClient`` to fetch opening prices.  Any broker
client that implements ``fetch_open_prices`` can be passed in, making this
testable without a live Tachibana connection.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, time
from typing import Any

import pandas as pd

from leadlag.broker.base import BrokerClient
from leadlag.data.providers import DataProvider

logger = logging.getLogger(__name__)


def _is_market_open(target: datetime | date) -> bool:
    """Return True if the target time is after the JP morning open."""
    if isinstance(target, datetime):
        return target.time() >= time(9, 0)
    return True


class TachibanaProvider(DataProvider):
    """Live intraday quote provider backed by a broker client.

    This provider is intended for decision-time 09:10 (or later) quote
    retrieval.  Historical daily OHLC is not supported by the broker API and
    should be obtained from ``YFinanceProvider`` or cache.
    """

    def __init__(self, client: BrokerClient | None = None) -> None:
        self._client = client

    def source_name(self) -> str:
        return "tachibana"

    def fetch_daily_ohlc(
        self,
        tickers: list[str],
        start: date,
        end: date,
    ) -> dict[str, pd.DataFrame]:
        """Historical daily OHLC is not available from the broker API."""
        raise NotImplementedError(
            "TachibanaProvider does not provide historical daily OHLC. "
            "Use YFinanceProvider for backfill data."
        )

    def fetch_intraday_quote(
        self,
        tickers: list[str],
        at: Any,
    ) -> dict[str, float]:
        """Return opening prices for ``tickers`` at the requested time.

        The broker ``fetch_open_prices`` call returns the exchange-issued
        opening price (pDPP) when called after market open.
        """
        if self._client is None:
            raise RuntimeError("TachibanaProvider requires a BrokerClient")
        if not _is_market_open(at):
            logger.warning(
                "Requested quote before JP market open (09:00). "
                "Prices may be stale or unavailable."
            )
        try:
            prices = self._client.fetch_open_prices(tickers, allow_missing=True)
            return {tk: float(p) for tk, p in prices.items()}
        except Exception as e:
            logger.error("TachibanaProvider failed to fetch quotes: %s", e)
            raise
