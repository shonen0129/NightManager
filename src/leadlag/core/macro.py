"""Macro signal utilities — factor-specific volatility-adjusted surprise.

Provides:
- MACRO_TICKERS / MACRO_NAMES: yfinance tickers and short names for the three
  macro factors (USDJPY, crude oil futures, 10-year Treasury yield).
- MACRO_SENS_MATRIX: (n_j, n_macro) domain-knowledge sensitivity weights mapping
  each JP sector ETF to each macro factor.
- download_macro_prices: fetch daily close prices for the three macro factors.
- download_macro_data: fetch daily returns for the three macro factors (wrapper).
- compute_macro_surprise: EWMA-based volatility-adjusted surprise (z-score)
  with per-factor independent mean and variance tracking.
- compute_factor_kappa_scale: per-stock risk-scaling vector from factor-specific
  kappa values and the sensitivity matrix.

All computations are lookahead-safe: EWMA mean and variance at time t use only
data from t-1 and earlier.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pandas as pd

from leadlag.data.tickers import JP_TICKERS

logger = logging.getLogger(__name__)

# Default timeout for yfinance downloads (seconds)
_MACRO_DOWNLOAD_TIMEOUT: float = 30.0

# Module-level cache: (start, end, period) -> DataFrame of close prices
_MACRO_PRICE_CACHE: dict[tuple[str | None, str | None, str], pd.DataFrame] = {}

# ---------------------------------------------------------------------------
# Macro factor definitions
# ---------------------------------------------------------------------------

MACRO_TICKERS: list[str] = ["JPY=X", "CL=F", "^TNX"]
MACRO_NAMES: list[str] = ["USDJPY", "CLF", "TNX"]
N_MACRO: int = len(MACRO_NAMES)

# ---------------------------------------------------------------------------
# Sector sensitivity matrix (n_j x n_macro)
#
# Domain-knowledge weights mapping each JP sector ETF to macro factors.
# Values on a 0–1 scale reflecting the directional sensitivity of each
# sector's earnings / equity performance to each macro factor.
#
# USDJPY:  FX channel — higher USDJPY (weaker yen) benefits exporters.
# CLF:     Oil price channel — impacts energy, transport, utilities.
# TNX:     US 10yr yield channel — impacts financials, real estate.
# ---------------------------------------------------------------------------

MACRO_SECTOR_MAPPING: dict[str, dict[str, float]] = {
    "1617.T": {"USDJPY": 0.2, "CLF": 0.0, "TNX": 0.0},   # 食品 (defensive)
    "1618.T": {"USDJPY": 0.8, "CLF": 1.0, "TNX": 0.0},   # エネルギー資源
    "1619.T": {"USDJPY": 0.5, "CLF": 0.0, "TNX": 0.0},   # 建設・資材
    "1620.T": {"USDJPY": 0.5, "CLF": 0.2, "TNX": 0.0},   # 素材・化学
    "1621.T": {"USDJPY": 0.3, "CLF": 0.0, "TNX": 0.0},   # 医薬品 (defensive)
    "1622.T": {"USDJPY": 0.9, "CLF": 0.3, "TNX": 0.0},   # 自動車・輸送機
    "1623.T": {"USDJPY": 0.7, "CLF": 0.0, "TNX": 0.0},   # 鉄鋼・非鉄
    "1624.T": {"USDJPY": 0.7, "CLF": 0.0, "TNX": 0.0},   # 機械
    "1625.T": {"USDJPY": 0.8, "CLF": 0.0, "TNX": 0.0},   # 電機・精密
    "1626.T": {"USDJPY": 0.6, "CLF": 0.0, "TNX": 0.0},   # 情報通信・サービス
    "1627.T": {"USDJPY": 0.3, "CLF": 0.6, "TNX": 0.1},   # 電力・ガス
    "1628.T": {"USDJPY": 0.5, "CLF": 0.4, "TNX": 0.0},   # 運輸・物流
    "1629.T": {"USDJPY": 0.7, "CLF": 0.2, "TNX": 0.0},   # 商社・卸売
    "1630.T": {"USDJPY": 0.5, "CLF": 0.0, "TNX": 0.0},   # 小売
    "1631.T": {"USDJPY": 0.6, "CLF": 0.0, "TNX": 1.0},   # 銀行
    "1632.T": {"USDJPY": 0.5, "CLF": 0.0, "TNX": 0.8},   # 金融（除く銀行）
    "1633.T": {"USDJPY": 0.3, "CLF": 0.0, "TNX": 0.3},   # 不動産
}

MACRO_SENS_MATRIX: np.ndarray = np.zeros((len(JP_TICKERS), N_MACRO))
for _j_idx, _jp_tk in enumerate(JP_TICKERS):
    for _m_idx, _m_name in enumerate(MACRO_NAMES):
        MACRO_SENS_MATRIX[_j_idx, _m_idx] = MACRO_SECTOR_MAPPING.get(_jp_tk, {}).get(_m_name, 0.0)


# ---------------------------------------------------------------------------
# Macro data download
# ---------------------------------------------------------------------------

def clear_macro_cache() -> None:
    """Clear the module-level macro price cache."""
    _MACRO_PRICE_CACHE.clear()


def download_macro_prices(
    start: str | None = None,
    end: str | None = None,
    period: str = "10y",
    timeout: float = _MACRO_DOWNLOAD_TIMEOUT,
) -> pd.DataFrame:
    """Download daily close prices for the three macro factors.

    Uses a module-level cache to avoid redundant downloads within the same
    session.  If the download does not complete within *timeout* seconds,
    a TimeoutError is raised.

    Returns a DataFrame with columns MACRO_NAMES and a DatetimeIndex.
    Values are daily close prices.
    """
    cache_key = (start, end, period)
    if cache_key in _MACRO_PRICE_CACHE:
        return _MACRO_PRICE_CACHE[cache_key].copy()

    import yfinance as yf

    def _do_download() -> Any:
        return yf.download(
            MACRO_TICKERS,
            start=start,
            end=end,
            period=period if start is None else None,
            progress=False,
            auto_adjust=False,
        )

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_do_download)
            raw = future.result(timeout=timeout)
    except Exception as exc:
        if isinstance(exc, TimeoutError) or "timeout" in type(exc).__name__.lower():
            raise TimeoutError(
                f"yfinance download did not complete within {timeout}s "
                f"(tickers={MACRO_TICKERS}, start={start}, end={end})"
            ) from exc
        raise

    # Extract Close prices
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw.to_frame() if not isinstance(raw, pd.DataFrame) else raw

    close.columns = MACRO_NAMES
    close = close.dropna(how="all")

    _MACRO_PRICE_CACHE[cache_key] = close.copy()
    return close


def download_macro_data(
    start: str | None = None,
    end: str | None = None,
    period: str = "10y",
    timeout: float = _MACRO_DOWNLOAD_TIMEOUT,
) -> pd.DataFrame:
    """Download daily macro factor returns aligned to trading days.

    Returns a DataFrame with columns MACRO_NAMES and a DatetimeIndex.
    Values are daily percentage returns.
    """
    close = download_macro_prices(start=start, end=end, period=period, timeout=timeout)

    macro_returns = close.pct_change()
    macro_returns = macro_returns.replace([np.inf, -np.inf], np.nan)
    macro_returns = macro_returns.fillna(0.0)

    return macro_returns


# ---------------------------------------------------------------------------
# Volatility-adjusted surprise (per-factor, lookahead-safe)
# ---------------------------------------------------------------------------

def compute_macro_surprise(
    macro_returns: pd.DataFrame | np.ndarray,
    halflife_mean: float = 20.0,
    halflife_vol: float = 60.0,
) -> np.ndarray:
    """Compute per-factor volatility-adjusted surprise z-scores.

    Models each macro factor return as a stochastic volatility process:
        r_t ~ N(mu_t, sigma_t)

    The standardized surprise z_t = (r_t - mu_t) / sigma_t is:
    - Scale-free (comparable across regimes)
    - Gaussian-approximate
    - Independent of volatility level

    Lookahead-safe: mu_t and sigma_t at time t use only data from t-1 and earlier.

    Args:
        macro_returns: (T, n_macro) array or DataFrame of daily macro returns.
        halflife_mean: EWMA half-life for mean estimation.
        halflife_vol: EWMA half-life for variance estimation.

    Returns:
        (T, n_macro) array of standardized surprise z-scores.
    """
    if isinstance(macro_returns, pd.DataFrame):
        vals = macro_returns[MACRO_NAMES].values
    else:
        vals = np.asarray(macro_returns)
    vals = np.nan_to_num(vals, nan=0.0)

    T, n_m = vals.shape
    decay_mean = float(np.power(0.5, 1.0 / halflife_mean))
    decay_vol = float(np.power(0.5, 1.0 / halflife_vol))

    ewma_mean = np.zeros(n_m)
    ewma_var = np.zeros(n_m)
    surprise_z = np.zeros((T, n_m))

    for t in range(T):
        if t > 0:
            prev = vals[t - 1]
            ewma_mean = decay_mean * ewma_mean + (1.0 - decay_mean) * prev
            resid = prev - ewma_mean
            ewma_var = decay_vol * ewma_var + (1.0 - decay_vol) * resid ** 2

        sigma = np.sqrt(ewma_var)
        sigma_safe = np.where(sigma > 1e-8, sigma, 1.0)
        surprise_z[t] = (vals[t] - ewma_mean) / sigma_safe

    return surprise_z


# ---------------------------------------------------------------------------
# Factor-Specific Kappa risk scaling
# ---------------------------------------------------------------------------

def compute_factor_kappa_scale(
    surprise_raw: np.ndarray,
    kappas: np.ndarray | tuple[float, float, float],
    sens_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """Compute per-stock risk-scaling factor from macro surprise.

    For each stock j at time t:
        scale_j = 1 + sum_m kappa_m * |surprise_m| * |sensitivity_jm|

    When macro surprise is large for a factor that stock j is sensitive to,
    the scale increases, reducing the position size (signal / scale).

    The signal direction stays purely BLPX (preserving AR), while the risk
    scaling adapts to macro conditions per-factor.

    Args:
        surprise_raw: (T, n_macro) array of per-factor surprise z-scores.
        kappas: (n_macro,) array or tuple of factor-specific kappa values.
        sens_matrix: (n_j, n_macro) sensitivity matrix. Defaults to MACRO_SENS_MATRIX.

    Returns:
        (T, n_j) array of scaling factors (>= 1.0).
    """
    if sens_matrix is None:
        sens_matrix = MACRO_SENS_MATRIX

    kappas_arr = np.asarray(kappas, dtype=float)
    abs_sens = np.abs(sens_matrix)  # (n_j, n_macro)

    T = surprise_raw.shape[0]
    n_j = abs_sens.shape[0]
    scales = np.ones((T, n_j))

    for t in range(T):
        abs_surprise = np.abs(surprise_raw[t])  # (n_macro,)
        scales[t] = 1.0 + abs_sens @ (kappas_arr * abs_surprise)

    return scales
