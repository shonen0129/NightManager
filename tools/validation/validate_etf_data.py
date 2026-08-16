#!/usr/bin/env python3
"""Validate and optionally repair raw ETF OHLC cache.

The cache can be either the legacy ``etf_data.pkl`` or the SQLite-backed
``etf_prices.sqlite`` (the current canonical store).  This tool runs before
``preprocess_data`` and detects the all-NaN / trailing-NaN ticker issues that
otherwise cause ``df_exec`` to be truncated.
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import pandas as pd

# Allow running from repo root without installing the package.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from leadlag.config.paths import market_data
from leadlag.data.cache_store import SqliteCacheStore
from leadlag.data.market_data_cache import _ETF_CACHE_FILENAME, _ETF_RAW_KEY
from leadlag.data.validation import DataValidationError, validate_etf_raw_data

logger = logging.getLogger(__name__)


def _resolve_cache_path(path: str | None) -> Path:
    if path is None:
        return Path(market_data(_ETF_CACHE_FILENAME))
    p = Path(path)
    if p.is_absolute():
        return p
    return ROOT / p


def _load_raw_data(cache_path: Path) -> dict:
    if cache_path.suffix in (".pkl", ".pickle"):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    if cache_path.suffix in (".sqlite", ".sqlite3", ".db"):
        store = SqliteCacheStore(cache_path)
        data = store.get(_ETF_RAW_KEY)
        if data is None:
            raise FileNotFoundError(f"ETF raw cache key {_ETF_RAW_KEY!r} not found in {cache_path}")
        return data
    raise ValueError(f"Unsupported cache file extension: {cache_path}")


def _save_raw_data(cache_path: Path, data: dict) -> None:
    if cache_path.suffix in (".pkl", ".pickle"):
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)
    elif cache_path.suffix in (".sqlite", ".sqlite3", ".db"):
        store = SqliteCacheStore(cache_path)
        store.set(_ETF_RAW_KEY, data)
    else:
        raise ValueError(f"Unsupported cache file extension: {cache_path}")


def _repair_data(data: dict, max_trailing_fill_days: int | None = None) -> dict:
    """Forward-fill isolated/trailing NaN in price tables.

    Leading NaN are *not* filled because that would require future prices
    (look-ahead).  Long trailing NaN blocks are left untouched unless
    ``max_trailing_fill_days`` is large enough; this prevents stale prices from
    being propagated indefinitely for delisted / unavailable tickers.
    """
    repaired = {}
    for key in ("us_close", "jp_close", "jp_open"):
        df = data[key].copy()
        for col in df.columns:
            series = df[col]
            if series.isna().all():
                continue

            first_valid = series.first_valid_index()
            last_valid = series.last_valid_index()
            if first_valid is None or last_valid is None:
                continue

            # Isolated + trailing NaN.
            filled = series.ffill()

            if max_trailing_fill_days is not None:
                idx = series.index
                last_valid_pos = idx.get_loc(last_valid)
                n_rows = len(idx)
                trailing_len = n_rows - last_valid_pos - 1
                if trailing_len > max_trailing_fill_days:
                    # Revert the over-long trailing fill.
                    fill_start = idx[last_valid_pos + 1]
                    fill_end = idx[-1]
                    filled.loc[fill_start:fill_end] = pd.NA

            df[col] = filled
        repaired[key] = df
    return repaired


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate raw ETF OHLC cache (pkl or sqlite)."
    )
    parser.add_argument(
        "--cache-file",
        default=None,
        help="Path to etf_data.pkl or etf_prices.sqlite. Defaults to canonical SQLite store.",
    )
    parser.add_argument(
        "--repair",
        action="store_true",
        help="Forward-fill repairable NaN and write the repaired cache back.",
    )
    parser.add_argument(
        "--max-trailing-fill-days",
        type=int,
        default=None,
        help="If --repair, do not fill trailing NaN blocks longer than N days.",
    )
    parser.add_argument(
        "--min-history-days",
        type=int,
        default=30,
        help="Minimum valid observations required per ticker.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print per-ticker statistics.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    cache_path = _resolve_cache_path(args.cache_file)
    if not cache_path.exists():
        logger.error("Cache file not found: %s", cache_path)
        return 2

    try:
        data = _load_raw_data(cache_path)
    except Exception as e:
        logger.error("Failed to load cache: %s", e)
        return 2

    try:
        result = validate_etf_raw_data(data, min_history_days=args.min_history_days)
    except DataValidationError as e:
        logger.error("Validation failed: %s", e)
        return 2

    if result["fatals"]:
        print("FATAL issues detected:")
        for msg in result["fatals"]:
            print(f"  - {msg}")

    if result["alerts"]:
        print("WARNINGS:")
        for msg in result["alerts"]:
            print(f"  - {msg}")

    if args.verbose:
        print("\nPer-ticker statistics:")
        for name, stats in result["ticker_stats"].items():
            print(
                f"  {name}: n_nan={stats['n_nan']}, "
                f"first={stats['first_valid']}, last={stats['last_valid']}, "
                f"leading={stats['leading_nan']}, trailing={stats['trailing_nan']}"
            )

    if not result["ok"]:
        print(
            "\nValidation FAILED. Recommend re-running market data download "
            "(e.g. `python3 scripts/batch/_update_market_data.py --force`) or "
            "investigating the flagged tickers in the upstream data source."
        )
        return 1

    if args.repair:
        repaired = _repair_data(data, max_trailing_fill_days=args.max_trailing_fill_days)
        try:
            _save_raw_data(cache_path, repaired)
            print(f"Repaired cache written to: {cache_path}")
        except Exception as e:
            logger.error("Failed to write repaired cache: %s", e)
            return 2

    print("Validation OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
