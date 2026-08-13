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

from leadlag.data.fetcher import download_data
from leadlag.data.market_data_cache import load_df_exec_from_local_cache
from research.scripts.experiments.build_adr_features import build_adr_features

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

        # Rebuild ADR features for the ML overlay.
        # Uses the refreshed local df_exec cache; this keeps data/adr_features.pkl
        # within a few business days of the latest trade date.
        df_exec = load_df_exec_from_local_cache(max_stale_bdays=1)
        adr_df = build_adr_features(df_exec)
        adr_output = ROOT / "data" / "adr_features.pkl"
        adr_csv = ROOT / "data" / "adr_features.csv"
        adr_output.parent.mkdir(parents=True, exist_ok=True)
        adr_df.to_pickle(adr_output)
        adr_df.to_csv(adr_csv)
        logger.info(
            "ADR features rebuilt: %d rows, last_index=%s, saved to %s",
            len(adr_df),
            adr_df.index[-1],
            adr_output,
        )
        return 0
    except Exception as e:
        logging.error("Failed to update market data: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
