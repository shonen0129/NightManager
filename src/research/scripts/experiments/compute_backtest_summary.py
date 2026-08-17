#!/usr/bin/env python
"""Compute backtest summary from daily CSVs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

results_dir = Path(sys.argv[1])
turnover = pd.read_csv(results_dir / "daily_daily_turnover.csv")
fallback = pd.read_csv(results_dir / "daily_daily_fallback.csv")

summary = {
    "mean_turnover": float(turnover["daily_turnover"].mean()),
    "fallback_count": int(fallback["daily_fallback"].astype(bool).sum()),
    "fallback_rate": float(fallback["daily_fallback"].astype(bool).mean()),
}
print(json.dumps(summary, indent=2))
