"""Broker-backed 09:10 quote provider (Tachibana / kabu-station compatible).

The provider uses a ``BrokerClient`` to fetch opening prices.  Any broker
client that implements ``fetch_open_prices`` can be passed in, making this
testable without a live Tachibana connection.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from leadlag.broker.base import BrokerClient
from leadlag.data.providers import DataProvider


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
        """Reject unsupported timestamp-specific retrieval without broker I/O."""
        raise NotImplementedError(
            "TachibanaProvider cannot reconstruct quotes at a requested time. "
            "Use the frozen timestamped market snapshot capture."
        )
