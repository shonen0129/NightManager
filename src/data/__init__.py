"""Data package — market data acquisition, caching, and preprocessing.

Public API::

    from data.ticker_registry import US_TICKERS, JP_TICKERS, TOPIX_TICKER
    from data.downloader import download_data
    from data.preprocessor import preprocess_data
    from data.cache import (
        is_decision_cache_valid,
        save_decision_cache,
        load_decision_cache,
        load_jp_close_from_cache,
    )
"""

from data.ticker_registry import (
    JP_TICKERS,
    JP_TICKERS_WITH_TOPIX,
    N_JP,
    N_JP_ASSETS,
    N_TOTAL,
    N_TOTAL_ASSETS,
    N_US,
    N_US_ASSETS,
    TOPIX_TICKER,
    US_TICKERS,
)
from data.downloader import download_data
from data.preprocessor import preprocess_data
from data.cache import (
    CACHE_TTL_HOURS,
    is_decision_cache_valid,
    is_pkl_cache_valid,
    load_decision_cache,
    load_jp_close_from_cache,
    save_decision_cache,
)

__all__ = [
    # Ticker registry
    "US_TICKERS",
    "JP_TICKERS",
    "JP_TICKERS_WITH_TOPIX",
    "TOPIX_TICKER",
    "N_US",
    "N_JP",
    "N_TOTAL",
    "N_US_ASSETS",
    "N_JP_ASSETS",
    "N_TOTAL_ASSETS",
    # Downloader
    "download_data",
    # Preprocessor
    "preprocess_data",
    # Cache
    "CACHE_TTL_HOURS",
    "is_pkl_cache_valid",
    "is_decision_cache_valid",
    "save_decision_cache",
    "load_decision_cache",
    "load_jp_close_from_cache",
]
