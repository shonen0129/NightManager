"""BLPX prior builder helpers."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from leadlag.data.tickers import JP_TICKERS, US_TICKERS

if TYPE_CHECKING:
    from leadlag.models.blpx.model import ProductionBLPXModel

logger = logging.getLogger("leadlag.models.blpx")

def _build_sector_prior(self: ProductionBLPXModel) -> np.ndarray:
    """Build the fixed 日米業種対応行列 M_sector of size (n_j x n_u).

    Weights are derived from _SECTOR_MAPPING_STRUCTURE with equal split,
    then column-normalized so each US ETF column sums to 1.0.
    """
    M = np.zeros((self.n_j, self.n_u))

    for u_idx, us_tk in enumerate(US_TICKERS):
        if us_tk in self._SECTOR_MAPPING_STRUCTURE:
            jp_tickers = self._SECTOR_MAPPING_STRUCTURE[us_tk]
            w = 1.0 / len(jp_tickers)
            for jp_tk in jp_tickers:
                if jp_tk in JP_TICKERS:
                    j_idx = JP_TICKERS.index(jp_tk)
                    M[j_idx, u_idx] = w

    # Column normalize (sum to 1.0)
    col_sums = np.sum(M, axis=0)
    for u_idx in range(self.n_u):
        if col_sums[u_idx] > 0:
            M[:, u_idx] /= col_sums[u_idx]

    return M

def _load_macro_returns(self: ProductionBLPXModel, df_exec: pd.DataFrame) -> pd.DataFrame | None:
    """Load macro factor returns aligned to df_exec index.

    Downloads macro close prices (USDJPY, CLF, TNX) via yfinance,
    aligns them to the trading dates in df_exec with forward-fill,
    then computes daily returns. This ensures that non-trading days
    (e.g. JP market open but US market closed) produce zero returns
    rather than carrying forward the previous day's return.

    If download fails or the resulting data is too short, returns None.
    """
    from leadlag.core.macro import MACRO_NAMES, download_macro_prices

    sim_dates = df_exec.index
    start = sim_dates[0].strftime("%Y-%m-%d")
    end = sim_dates[-1].strftime("%Y-%m-%d")

    # download_macro_prices normalizes yfinance/network errors to RuntimeError.
    try:
        close_prices = download_macro_prices(
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

# Structural mapping: which JP tickers relate to which US tickers
_SECTOR_MAPPING_STRUCTURE = {
    "XLB": ["1620.T", "1623.T"],
    "XLC": ["1626.T"],
    "XLE": ["1618.T", "1627.T"],
    "XLF": ["1631.T", "1632.T"],
    "XLI": ["1624.T", "1622.T", "1626.T"],
    "XLK": ["1626.T", "1625.T"],
    "XLP": ["1617.T", "1630.T"],
    "XLRE": ["1633.T"],
    "XLU": ["1627.T"],
    "XLV": ["1621.T"],
    "XLY": ["1630.T", "1626.T", "1622.T"],
    "MTUM": ["1625.T", "1626.T"],
    "VLUE": ["1631.T", "1632.T", "1623.T", "1622.T"],
    "IUSG": ["1626.T", "1625.T"],
    "USMV": ["1617.T", "1621.T", "1627.T"],
}

def _get_sector_prior(
    self: ProductionBLPXModel,
    current_index: int,
    all_returns: np.ndarray,
    corr: np.ndarray,
    B_blp: np.ndarray,
) -> np.ndarray:
    """Return the sector prior matrix M_sector (n_j x n_u).

    When sector_eta > 0, blends the fixed mapping with data-driven
    weights derived from the rolling cross-correlation:
      w_ji = max(0, corr(u, ji))^gamma / sum_k max(0, corr(u, jk))^gamma
      M_final = (1-eta) * M_fixed + eta * M_data

    Override in subclasses to provide a fully dynamic sector prior.
    """
    if self.sector_eta <= 0.0 or self._M_sector_fixed.shape != B_blp.shape:
        if self.M_sector.shape == B_blp.shape:
            return self.M_sector
        return np.zeros(B_blp.shape)

    if corr.shape != (self.n_u + self.n_j, self.n_u + self.n_j):
        return self._M_sector_fixed if self._M_sector_fixed.shape == B_blp.shape else np.zeros(B_blp.shape)

    c_xy = corr[: self.n_u, self.n_u:]  # (n_u, n_j) — US vs JP cross-corr

    M_data = np.zeros((self.n_j, self.n_u))
    for u_idx, j_indices in self._sector_mapping_indices.items():
        weights = []
        for j_idx in j_indices:
            raw_corr = c_xy[u_idx, j_idx]
            weights.append((j_idx, max(0.0, raw_corr) ** self.sector_gamma))
        if not weights:
            continue
        total = sum(w for _, w in weights)
        if total > 1e-10:
            for j_idx, w in weights:
                M_data[j_idx, u_idx] = w / total

    M_blended = (1.0 - self.sector_eta) * self._M_sector_fixed + self.sector_eta * M_data

    col_sums = np.sum(M_blended, axis=0)
    for u_idx in range(self.n_u):
        if col_sums[u_idx] > 1e-10:
            M_blended[:, u_idx] /= col_sums[u_idx]

    if M_blended.shape == B_blp.shape:
        return M_blended
    return np.zeros(B_blp.shape)


