"""YFinance-backed data provider.

This is a thin, testable wrapper around yfinance.  The actual download
implementation can be injected at construction time so that unit tests run
without network access.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from leadlag.data.providers import DataProvider

logger = logging.getLogger(__name__)


def _default_yf_download(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    """Default yfinance download; isolated for test injection."""
    return yf.download(
        tickers=tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=False,
    )


class YFinanceProvider(DataProvider):
    """Data provider backed by Yahoo Finance (yfinance).

    ``download_fn`` may be supplied to override the network call.  This makes
    the provider testable without network access and lets callers use a
    pre-fetched DataFrame or a cached download function.
    """

    def __init__(
        self,
        download_fn: Callable[[list[str], date, date], pd.DataFrame] | None = None,
    ) -> None:
        self._download_fn = download_fn or _default_yf_download

    def source_name(self) -> str:
        return "yfinance"

    def fetch_daily_ohlc(
        self,
        tickers: list[str],
        start: date,
        end: date,
    ) -> dict[str, pd.DataFrame]:
        """Return OHLC DataFrames for each ticker.

        The returned DataFrame has columns ``open``, ``high``, ``low``,
        ``close`` and optional ``volume`` and is indexed by date.
        """
        raw = self._download_fn(tickers, start, end)
        if raw.empty:
            raise ValueError("yfinance returned empty data")

        result: dict[str, pd.DataFrame] = {}
        for tk in tickers:
            if raw.columns.nlevels == 2:
                df = pd.DataFrame({
                    "open": raw[("Open", tk)],
                    "high": raw[("High", tk)],
                    "low": raw[("Low", tk)],
                    "close": raw[("Close", tk)],
                    "volume": raw[("Volume", tk)] if ("Volume", tk) in raw else np.nan,
                })
            else:
                df = pd.DataFrame({
                    "open": raw["Open"],
                    "high": raw["High"],
                    "low": raw["Low"],
                    "close": raw["Close"],
                    "volume": raw.get("Volume", np.nan),
                })
            df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
            df = df.dropna()
            result[tk] = df
        return result

    def fetch_intraday_quote(
        self,
        tickers: list[str],
        at: Any,
    ) -> dict[str, float]:
        """Fetch the most recent available price at ``at``.

        For yfinance this is a best-effort call to ``yf.Ticker(ticker).history``
        for the requested period.  Production code should prefer broker APIs
        (``TachibanaProvider``/``KabuProvider``) for live 09:10 quotes.
        """
        result: dict[str, float] = {}
        for tk in tickers:
            try:
                hist = yf.Ticker(tk).history(period="1d", interval="1m")
                if not hist.empty:
                    price = float(hist["Close"].iloc[-1])
                    if np.isfinite(price):
                        result[tk] = price
            except Exception as e:
                logger.warning("YFinance intraday quote failed for %s: %s", tk, e)
        return result
