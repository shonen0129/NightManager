"""DEPRECATED: v1 fallback weights loader.

This module was used to load v1 Residual-BLPX baseline weights as a fallback
when v2 gap data was missing or audits failed. This fallback mechanism has been
DEPRECATED as of 2026-07-09 because v1 fallback would also fail when v2 fails
(circular dependency issue). Gap data missing now results in flat position (w_final=0).

Kept in archive for reference only. Do not use in production.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from leadlag.data.tickers import JP_TICKERS

logger = logging.getLogger(__name__)


def load_v1_fallback_weights(
    v1_weights_file: Path,
    date_str: str,
    n_j: int,
) -> tuple[np.ndarray, list[str]]:
    """Load v1 Residual-BLPX baseline weights as fallback.

    Args:
        v1_weights_file: Path to ``v1_baseline_weights.csv``.
        date_str: Expected trade date in the file (for alignment check).
        n_j: Number of JP tickers (determines output array length).

    Returns:
        Tuple of (w_v1 array, alerts list).
    """
    alerts: list[str] = []
    w_v1 = np.zeros(n_j)

    if not v1_weights_file.exists():
        alerts.append(f"v1 weights file not found: {v1_weights_file}. Using zero weights.")
        return w_v1, alerts

    df = pd.read_csv(v1_weights_file)
    if len(df) == 0:
        alerts.append("v1 weights file is empty. Using zero weights.")
        return w_v1, alerts

    # Verify date alignment
    file_date = str(df.iloc[0].get("trade_date", ""))
    if file_date != date_str:
        alerts.append(
            f"v1 weights date mismatch: file has {file_date}, expected {date_str}. "
            "Using stale v1 weights (caution)."
        )

    for _, row in df.iterrows():
        tk = str(row.get("ticker", ""))
        if tk in JP_TICKERS:
            idx = JP_TICKERS.index(tk)
            w_v1[idx] = float(row.get("weight", 0.0))

    if np.sum(np.abs(w_v1)) < 1e-8:
        alerts.append("v1 weights loaded but all zero. Using zero weights.")

    return w_v1, alerts
