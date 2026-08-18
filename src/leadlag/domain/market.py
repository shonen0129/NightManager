"""Domain types for market data and gap-adjusted distribution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AssetReturns:
    """Aligned US/JP return panel at a single signal date."""

    sig_date: pd.Timestamp
    us_returns: np.ndarray
    jp_returns: np.ndarray


@dataclass(frozen=True)
class GapDistribution:
    """Gap-adjusted return distribution at a single trade date."""

    trade_date: pd.Timestamp
    mu: np.ndarray
    omega: np.ndarray
    gap_override: np.ndarray | None = None
    topix_night: float | None = None
    betas: np.ndarray | None = None
