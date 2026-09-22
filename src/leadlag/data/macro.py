"""Macro price input adapter.

Network access and cache ownership live in this module.  The numerical macro
transformations remain in :mod:`leadlag.core.macro` and consume only the
returned series.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import numpy as np
import pandas as pd

from leadlag.core.macro import MACRO_NAMES, MACRO_TICKERS
from leadlag.utils.threading import run_with_timeout

_MACRO_DOWNLOAD_TIMEOUT = 30.0
_MACRO_PRICE_CACHE: dict[tuple[str | None, str | None, str], pd.DataFrame] = {}


def clear_macro_cache() -> None:
    """Clear the process-local macro price cache."""
    _MACRO_PRICE_CACHE.clear()


def load_macro_prices(
    start: str | None = None,
    end: str | None = None,
    period: str = "10y",
    timeout: float = _MACRO_DOWNLOAD_TIMEOUT,
    cache: MutableMapping[Any, Any] | None = None,
) -> pd.DataFrame:
    """Download and normalize daily close prices for the macro factors.

    The provider returns columns in the stable ``MACRO_NAMES`` order regardless
    of yfinance's column ordering.  A caller-owned cache can be supplied for
    backtests; otherwise the adapter's process-local cache is used.
    """
    active_cache = _MACRO_PRICE_CACHE if cache is None else cache
    cache_key = (start, end, period)
    if cache_key in active_cache:
        return active_cache[cache_key].copy()

    import yfinance as yf

    def _download() -> Any:
        return yf.download(
            MACRO_TICKERS,
            start=start,
            end=end,
            period=period if start is None else None,
            progress=False,
            auto_adjust=False,
        )

    try:
        raw = run_with_timeout(
            _download,
            timeout,
            label=f"yf.download(tickers={MACRO_TICKERS}, start={start}, end={end})",
        )
    except TimeoutError:
        raise
    except Exception as exc:
        raise RuntimeError(f"yfinance macro download failed: {exc}") from exc

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    elif not isinstance(raw, pd.DataFrame):
        close = raw.to_frame()
    else:
        close = raw

    existing_cols = set(close.columns)
    if existing_cols == set(MACRO_NAMES):
        close = close[MACRO_NAMES]
    elif existing_cols.issubset(set(MACRO_TICKERS)):
        close = close.reindex(columns=MACRO_TICKERS)
        close.columns = MACRO_NAMES
    else:
        raise RuntimeError(
            f"Unexpected macro close columns {close.columns.tolist()}. "
            f"Expected one of {MACRO_TICKERS} or {MACRO_NAMES}."
        )

    if close.isna().all().any():
        all_na = close.columns[close.isna().all()].tolist()
        returned = raw.columns.tolist() if hasattr(raw, "columns") else []
        raise RuntimeError(
            f"Macro download missing or all-NaN tickers in {all_na}. "
            f"Requested {MACRO_TICKERS}; returned columns {returned}."
        )

    close = close.dropna(how="all")
    active_cache[cache_key] = close.copy()
    return close


def load_macro_returns(
    start: str | None = None,
    end: str | None = None,
    period: str = "10y",
    timeout: float = _MACRO_DOWNLOAD_TIMEOUT,
    cache: MutableMapping[Any, Any] | None = None,
) -> pd.DataFrame:
    """Load macro prices and convert them to finite daily returns."""
    close = load_macro_prices(
        start=start,
        end=end,
        period=period,
        timeout=timeout,
        cache=cache,
    )
    returns = close.pct_change().replace([np.inf, -np.inf], np.nan)
    return returns.fillna(0.0)


__all__ = [
    "MACRO_NAMES",
    "MACRO_TICKERS",
    "clear_macro_cache",
    "load_macro_prices",
    "load_macro_returns",
]
