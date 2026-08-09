#!/usr/bin/env python3
"""5分足・1分足キャッシュを yfinance から更新する。

既存 cache とマージし、最新 60 日（5m）/ 7 日（1m）を追加する。
yfinance は 5m データを直近 60 日間しか提供しない制限がある。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.fetcher import update_intraday_cache


def main():
    update_intraday_cache()


if __name__ == "__main__":
    main()
