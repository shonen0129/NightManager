"""data_loader.py — backward-compatible facade.

This module is kept as a thin facade to avoid breaking existing imports
across the codebase. All implementation has been moved to the ``data/``
package:

    - data.ticker_registry  →  ticker definitions and conversion utils
    - data.downloader       →  download_data()
    - data.preprocessor     →  preprocess_data()
    - data.cache            →  cache I/O helpers

New code should import directly from the ``data`` package::

    from data.ticker_registry import US_TICKERS, JP_TICKERS
    from data.downloader import download_data
    from data.preprocessor import preprocess_data
    from data.cache import (
        is_decision_cache_valid,
        save_decision_cache,
        load_decision_cache,
        load_jp_close_from_cache,
    )
"""

# ---------------------------------------------------------------------------
# Re-export all public symbols from data/ for backward compatibility
# ---------------------------------------------------------------------------

from data.ticker_registry import (  # noqa: F401
    US_TICKERS,
    JP_TICKERS,
    TOPIX_TICKER,
    JP_TICKERS_WITH_TOPIX,
    N_US_ASSETS,
    N_JP_ASSETS,
    N_TOTAL_ASSETS,
)

from data.cache import (  # noqa: F401
    CACHE_TTL_HOURS,
    is_pkl_cache_valid as is_cache_valid,
    is_decision_cache_valid,
    save_decision_cache,
    load_decision_cache,
    load_jp_close_from_cache,
)

from data.downloader import download_data  # noqa: F401
from data.preprocessor import preprocess_data  # noqa: F401

# Backward-compatible alias used by services/cache_service.py
CACHE_UPDATE_LOOKBACK_DAYS = 14
