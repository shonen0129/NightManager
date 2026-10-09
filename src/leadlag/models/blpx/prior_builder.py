"""BLPX prior builder helpers."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from leadlag.core.blpx_math import compute_sector_prior
from leadlag.data.tickers import US_TO_JP_SECTOR_MAPPING

if TYPE_CHECKING:
    from leadlag.models.blpx.model import ProductionBLPXModel

logger = logging.getLogger("leadlag.models.blpx")

def _load_macro_returns(self: ProductionBLPXModel, df_exec: pd.DataFrame) -> pd.DataFrame | None:
    """Load macro factor returns aligned to df_exec index.

    Downloads macro close prices (USDJPY, CLF, TNX) via yfinance,
    aligns them to the trading dates in df_exec with forward-fill,
    then computes daily returns. This ensures that non-trading days
    (e.g. JP market open but US market closed) produce zero returns
    rather than carrying forward the previous day's return.

    If download fails or the resulting data is too short, returns None.
    """
    from leadlag.core.macro import MACRO_NAMES
    from leadlag.data import macro as macro_data

    sim_dates = df_exec.index
    start = sim_dates[0].strftime("%Y-%m-%d")
    end = sim_dates[-1].strftime("%Y-%m-%d")

    # The data adapter normalizes yfinance/network errors to RuntimeError.
    try:
        close_prices = macro_data.load_macro_prices(
            start=start,
            end=end,
            cache=self._macro_price_cache,
        )
    except (RuntimeError, TimeoutError, OSError, ValueError, TypeError, KeyError, IndexError) as e:
        logger.warning("Failed to download macro prices: %s", e)
        return None

    if close_prices is None or len(close_prices) < 30:
        logger.warning("Macro data too short (%d rows); skipping.", len(close_prices) if close_prices is not None else 0)
        return None

    # Align prices to df_exec dates, forward-fill missing values
    try:
        prices_aligned = close_prices.reindex(sim_dates, method="ffill")
        prices_aligned = prices_aligned.ffill().fillna(0.0)

        # Compute returns AFTER alignment so non-trading days get zero return
        macro_returns = prices_aligned.pct_change()
        macro_returns = macro_returns.replace([np.inf, -np.inf], np.nan)
        macro_returns = macro_returns.fillna(0.0)
        return macro_returns[MACRO_NAMES]
    except (KeyError, ValueError, TypeError, IndexError) as e:
        logger.warning("Failed to load macro data: %s", e)
        return None

_SECTOR_MAPPING_STRUCTURE = US_TO_JP_SECTOR_MAPPING


def _get_sector_prior(
    self: ProductionBLPXModel,
    current_index: int,
    all_returns: np.ndarray,
    corr: np.ndarray,
    B_blp: np.ndarray,
) -> np.ndarray:
    """Apply model-owned sector settings through the shared prior calculation."""
    return compute_sector_prior(
        corr,
        B_blp,
        self.M_sector,
        self._M_sector_fixed,
        self._sector_mapping_indices,
        n_u=self.n_u,
        n_j=self.n_j,
        sector_eta=self.sector_eta,
        sector_gamma=self.sector_gamma,
    )
