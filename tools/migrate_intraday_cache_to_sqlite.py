"""Migrate legacy 1m/5m pkl intraday caches into etf_prices.sqlite.

One-off migration tool for ADR-0004 / T-P1-2.  After the SQLite cache store
became the default for raw OHLC, the 1m/5m pkl files were left behind.  This
script reads them and writes their contents into the canonical ETF SQLite
store with the key naming used by ``market_data_cache.save_intraday_cache``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from leadlag.data.market_data_cache import _etf_store_path, save_intraday_cache


def _find_pkl(root: Path, filename: str) -> Path | None:
    for rel in ("var/market_data", "market_data"):
        p = root / rel / filename
        if p.exists():
            return p
    return None


def migrate(intervals: list[str], dry_run: bool = False) -> None:
    for iv in intervals:
        filename = f"etf_{iv}_data.pkl"
        pkl_path = _find_pkl(ROOT, filename)
        if pkl_path is None:
            print(f"[skip] {filename} not found")
            continue

        print(f"[load] {pkl_path}")
        df = pd.read_pickle(pkl_path)
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected DataFrame, got {type(df)} in {pkl_path}")

        # Sanity check on the MultiIndex schema
        top = df.columns.get_level_values(0).unique().tolist()
        if "Open" not in top or "High" not in top or "Low" not in top or "Close" not in top:
            raise ValueError(
                f"{pkl_path} is missing OHLC columns. Found top-level: {top}"
            )

        if dry_run:
            print(f"[dry-run] would write {iv} cache ({len(df)} rows, {len(df.columns)} cols)")
            continue

        save_intraday_cache(df, iv)
        print(f"[save] intraday {iv} cache written to {_etf_store_path()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate legacy 1m/5m pkl intraday caches to SQLite."
    )
    parser.add_argument(
        "--intervals",
        nargs="+",
        default=["1m", "5m"],
        choices=["1m", "5m"],
        help="Intervals to migrate (default: 1m 5m).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    migrate(args.intervals, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
