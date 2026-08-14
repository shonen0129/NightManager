"""Data provider abstraction layer.

Provides an abstract base class for market-data sources so that production,
backtest, and test code can depend on an interface rather than on a concrete
yfinance/Tachibana implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class OHLC:
    """A single daily OHLC observation."""

    open: float
    high: float
    low: float
    close: float
    volume: float | None = None


class DataProvider(ABC):
    """Abstract data provider for OHLC and quote data.

    Decision-time callers use ``fetch_intraday_quote`` to obtain the 09:10
    opening prices for JP tickers.  Backtest / research callers use
    ``fetch_daily_ohlc`` to obtain historical daily OHLC series.
    """

    @abstractmethod
    def fetch_daily_ohlc(
        self,
        tickers: list[str],
        start: date,
        end: date,
    ) -> dict[str, pd.DataFrame]:
        """Return ``{ticker: DataFrame}`` indexed by date with OHLC columns."""
        ...

    @abstractmethod
    def fetch_intraday_quote(
        self,
        tickers: list[str],
        at: Any,
    ) -> dict[str, float]:
        """Return the best available price for each ticker at a given time."""
        ...

    @abstractmethod
    def source_name(self) -> str:
        """Return a human-readable provider name (for diagnostics)."""
        ...


class _DataProviderError(Exception):
    """Base exception for data provider failures."""


from leadlag.data.providers.tachibana_provider import TachibanaProvider
from leadlag.data.providers.yfinance_provider import YFinanceProvider

__all__ = ["DataProvider", "OHLC", "YFinanceProvider", "TachibanaProvider"]
