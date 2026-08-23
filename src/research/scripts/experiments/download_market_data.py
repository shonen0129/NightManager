#!/usr/bin/env python3
"""Download and cache market data for subsector experiments."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.fetcher import download_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def main() -> int:
    data = download_data(
        start_date="2009-01-01",
        end_date="2026-12-31",
        beta_window=60,
        force=False,
    )
    print("us_close:", data["us_close"].shape)
    print("jp_close:", data["jp_close"].shape)
    print("jp_open:", data["jp_open"].shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
