#!/usr/bin/env python3
"""Force-update the etf_data.pkl market data cache.

This script bypasses the 12-hour TTL and performs an incremental update of
US/JP ETF OHLC data from yfinance. It is intended to be run after the US
market close (JST ~05:00-06:00) so that the subsequent distribution_diagnostics
pipeline can use the latest close data.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from leadlag.core.market_calendar import is_trading_day
from leadlag.data.adr_producer import refresh_adr_features
from leadlag.data.fetcher import download_data
from leadlag.data.market_data_cache import load_df_exec_from_local_cache

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> int:
    try:
        data = download_data(start_date="2009-01-01", beta_window=60, force=True)
        logger = logging.getLogger(__name__)
        logger.info("Market data updated successfully")
        for key in ("us_close", "jp_close", "jp_open"):
            df = data.get(key)
            if df is not None:
                logger.info("%s: %d rows x %d columns, last index=%s", key, len(df), len(df.columns), df.index[-1])

        # A trading-day refresh must publish that exact date, not yesterday's row.
        df_exec = load_df_exec_from_local_cache(max_stale_bdays=1)
        today = pd.Timestamp.now(tz="Asia/Tokyo").tz_localize(None).normalize()
        required = today if is_trading_day(today.date()) else df_exec.index.max()
        refresh_adr_features(df_exec, required_trade_date=required)
        return 0
    except Exception as e:
        logging.error("Failed to update market data: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
