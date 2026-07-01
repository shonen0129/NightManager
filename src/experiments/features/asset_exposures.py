"""src/features/asset_exposures.py — Sprint 3-B Asset-Specific Exposure Computation.

Computes per-ticker exposures to macro factors and US sectors:
  1. Rolling beta exposure (ridge-shrunk, lag-1, no look-ahead)
  2. Static sector exposure map (from YAML config)

Rolling beta:
    r_{j,t} = a_j + beta_{j,k,t} * x_{k,t} + epsilon_{j,t}
    Estimated with ridge shrinkage using t-1 data only.

Look-ahead prevention:
    - All rolling betas are computed with lag_days=1 (default).
    - Full-sample beta is PROHIBITED.
    - min_obs is enforced; if not met, NaN is returned.
    - Cross-sectional z-scoring of betas uses same-date tickers only.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rolling beta with ridge shrinkage
# ---------------------------------------------------------------------------


def rolling_ridge_beta(
    y: pd.Series,
    x: pd.Series,
    window: int,
    min_obs: int = 60,
    shrinkage_alpha: float = 10.0,
    lag_days: int = 1,
) -> pd.Series:
    """Compute rolling beta of y on x using ridge shrinkage.

    Formula:
        beta_hat = sum(x*y) / (sum(x^2) + lambda)

    where lambda = shrinkage_alpha.

    Parameters
    ----------
    y:
        Asset return series (DatetimeIndex). This is the dependent variable.
    x:
        Factor return series (DatetimeIndex). This is the independent variable.
    window:
        Rolling window in trading days.
    min_obs:
        Minimum observations required (returns NaN otherwise).
    shrinkage_alpha:
        Ridge penalty (lambda). Higher = more shrinkage toward zero.
    lag_days:
        Lag applied to x before computing beta (prevents look-ahead).
        With lag_days=1, x_{t-1} is used as predictor for y_t.

    Returns
    -------
    pd.Series
        Rolling ridge beta series (same index as y).
    """
    # Align on common index
    common = y.index.intersection(x.index)
    y_aligned = y.reindex(common)

    # Apply lag to factor (lag_days prevents look-ahead)
    x_lagged = x.reindex(common).shift(lag_days)

    # We compute beta using rolling windows: beta = cov(x_lag, y) / (var(x_lag) + lambda/n)
    # Equivalently: sum(x*y) / (sum(x^2) + alpha) where sum is over window
    # Demean x and y within window
    beta_values = np.full(len(common), np.nan)

    y_arr = y_aligned.values
    x_arr = x_lagged.values

    for i in range(window - 1, len(common)):
        start = max(0, i - window + 1)
        y_win = y_arr[start:i + 1]
        x_win = x_arr[start:i + 1]

        # Drop pairs where either is NaN
        valid = ~(np.isnan(y_win) | np.isnan(x_win))
        n_valid = valid.sum()

        if n_valid < min_obs:
            continue

        y_v = y_win[valid]
        x_v = x_win[valid]

        # Demean
        x_m = x_v - x_v.mean()
        y_m = y_v - y_v.mean()

        # Ridge beta: sum(x_m * y_m) / (sum(x_m^2) + alpha)
        num = np.dot(x_m, y_m)
        denom = np.dot(x_m, x_m) + shrinkage_alpha
        if denom > 1e-12:
            beta_values[i] = num / denom

    result = pd.Series(beta_values, index=common, name=f"beta_{y.name}_{x.name}_w{window}")
    return result


def compute_rolling_betas_panel(
    asset_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    windows: Sequence[int] = (120, 252),
    min_obs: int = 60,
    shrinkage_alpha: float = 10.0,
    lag_days: int = 1,
    standardize_cross_sectionally: bool = True,
) -> pd.DataFrame:
    """Compute rolling betas for all asset-factor pairs across windows.

    Parameters
    ----------
    asset_returns:
        Wide panel (dates × tickers) of asset returns.
    factor_returns:
        Wide panel (dates × factors) of factor returns.
    windows:
        List of rolling window sizes.
    min_obs:
        Minimum observations required.
    shrinkage_alpha:
        Ridge shrinkage parameter.
    lag_days:
        Lag applied to factor (default 1 = no look-ahead).
    standardize_cross_sectionally:
        If True, cross-sectionally z-score betas per date.

    Returns
    -------
    pd.DataFrame
        MultiIndex panel with columns like:
        beta_{ticker}_{factor}_w{window}
        Indexed by date, wide-format (one column per asset-factor-window).
    """
    results: dict[str, pd.Series] = {}

    tickers = asset_returns.columns.tolist()
    factors = factor_returns.columns.tolist()

    logger.info(
        "Computing rolling betas: %d tickers × %d factors × %d windows",
        len(tickers), len(factors), len(windows)
    )

    for factor_col in factors:
        if factor_col not in factor_returns.columns:
            continue
        x_series = factor_returns[factor_col]

        for ticker in tickers:
            if ticker not in asset_returns.columns:
                continue
            y_series = asset_returns[ticker]

            for w in windows:
                beta_series = rolling_ridge_beta(
                    y=y_series,
                    x=x_series,
                    window=w,
                    min_obs=min_obs,
                    shrinkage_alpha=shrinkage_alpha,
                    lag_days=lag_days,
                )
                col_name = f"beta_{ticker}_{factor_col}_w{w}"
                results[col_name] = beta_series

    if not results:
        return pd.DataFrame()

    # Align all series to common date index
    beta_df = pd.DataFrame(results)

    if standardize_cross_sectionally:
        beta_df = _cross_sectional_zscore_betas(beta_df, tickers, factors, windows)

    logger.info(
        "Rolling beta panel computed: %d dates × %d columns",
        beta_df.shape[0], beta_df.shape[1]
    )
    return beta_df


def _cross_sectional_zscore_betas(
    beta_df: pd.DataFrame,
    tickers: list[str],
    factors: list[str],
    windows: Sequence[int],
) -> pd.DataFrame:
    """Cross-sectionally z-score betas per date, per factor, per window.

    For each (factor, window) combination:
        beta_zscore_{j,t} = (beta_{j,t} - mean_{k}(beta_{k,t})) / std_{k}(beta_{k,t})

    Adds columns with _cs suffix alongside original columns.
    """
    cs_results: dict[str, pd.Series] = {}

    for factor_col in factors:
        for w in windows:
            # Get all tickers' betas for this factor/window
            cols = [f"beta_{tk}_{factor_col}_w{w}" for tk in tickers
                    if f"beta_{tk}_{factor_col}_w{w}" in beta_df.columns]

            if not cols:
                continue

            sub = beta_df[cols]  # dates × tickers
            # Cross-sectional z-score per date
            mean_cs = sub.mean(axis=1)
            std_cs = sub.std(axis=1).replace(0, np.nan)

            for col in cols:
                ticker = col.split("_")[1]
                cs_col = f"beta_cs_{ticker}_{factor_col}_w{w}"
                cs_results[cs_col] = (beta_df[col] - mean_cs) / std_cs

    if cs_results:
        cs_df = pd.DataFrame(cs_results)
        beta_df = pd.concat([beta_df, cs_df], axis=1)

    return beta_df


# ---------------------------------------------------------------------------
# Static sector exposure
# ---------------------------------------------------------------------------


def load_static_sector_map(
    config_path: str | Path,
    tickers: list[str],
    sector_columns: list[str],
    normalize_rows: bool = False,
) -> pd.DataFrame:
    """Load static sector exposure map from YAML config.

    Parameters
    ----------
    config_path:
        Path to sector_exposure_map.yaml.
    tickers:
        List of JP ticker strings (e.g., ['1610.T', '1613.T', ...]).
    sector_columns:
        List of sector names to extract (e.g., ['us_tech', 'us_energy', ...]).
    normalize_rows:
        If True, normalize each row so it sums to 1.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by ticker, columns = sector names. Shape: (n_tickers, n_sectors).
    """
    config_path = Path(config_path)
    if not config_path.exists():
        logger.warning(
            "Sector exposure map not found at %s. Using zero exposures.", config_path
        )
        return pd.DataFrame(0.0, index=tickers, columns=sector_columns)

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    exposures_raw = raw.get("exposures", {})

    records = []
    for tk in tickers:
        row = {"ticker": tk}
        if tk in exposures_raw:
            src = exposures_raw[tk]
            for col in sector_columns:
                row[col] = float(src.get(col, 0.0))
        else:
            # Unknown ticker: zero exposure
            for col in sector_columns:
                row[col] = 0.0
        records.append(row)

    df = pd.DataFrame(records).set_index("ticker")
    df = df[sector_columns]  # ensure column order

    if normalize_rows:
        row_sums = df.abs().sum(axis=1)
        row_sums = row_sums.replace(0, np.nan)
        df = df.div(row_sums, axis=0).fillna(0.0)

    logger.info(
        "Static sector exposure map loaded: %d tickers × %d sectors. "
        "Non-zero exposure rate: %.1f%%",
        df.shape[0], df.shape[1],
        float((df != 0).values.mean() * 100),
    )
    return df


# ---------------------------------------------------------------------------
# Build asset-specific exposure panel
# ---------------------------------------------------------------------------


def build_asset_exposure_panel(
    dates: pd.DatetimeIndex,
    tickers: list[str],
    asset_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    sector_exposure_map: pd.DataFrame | None,
    sector_columns: list[str],
    beta_windows: Sequence[int] = (120, 252),
    min_obs: int = 60,
    shrinkage_alpha: float = 10.0,
    lag_days: int = 1,
    standardize_cross_sectionally: bool = True,
) -> dict[str, pd.DataFrame]:
    """Build the full asset exposure panel for Sprint 3-B.

    Returns a dict with keys:
    - 'rolling_beta': DataFrame (dates × [beta_{ticker}_{factor}_w{window}])
    - 'static_sector': DataFrame (tickers × sector_columns), time-invariant

    Parameters
    ----------
    dates:
        DatetimeIndex of trading dates.
    tickers:
        List of asset tickers.
    asset_returns:
        Wide panel (dates × tickers) of asset intraday returns.
    factor_returns:
        Wide panel (dates × factors) of macro/sector factor returns.
    sector_exposure_map:
        Static sector exposure (tickers × sectors). If None, zero-filled.
    sector_columns:
        Sector column names to use.
    beta_windows, min_obs, shrinkage_alpha, lag_days, standardize_cross_sectionally:
        Rolling beta parameters.

    Returns
    -------
    dict with 'rolling_beta' and 'static_sector' DataFrames.
    """
    result = {}

    # Rolling beta panel
    if factor_returns is not None and not factor_returns.empty:
        rolling_beta_df = compute_rolling_betas_panel(
            asset_returns=asset_returns.reindex(dates),
            factor_returns=factor_returns.reindex(dates),
            windows=beta_windows,
            min_obs=min_obs,
            shrinkage_alpha=shrinkage_alpha,
            lag_days=lag_days,
            standardize_cross_sectionally=standardize_cross_sectionally,
        )
        result["rolling_beta"] = rolling_beta_df
    else:
        logger.warning("No factor returns provided. Rolling beta panel will be empty.")
        result["rolling_beta"] = pd.DataFrame(index=dates)

    # Static sector exposure
    if sector_exposure_map is not None:
        result["static_sector"] = sector_exposure_map
    else:
        result["static_sector"] = pd.DataFrame(
            0.0, index=tickers, columns=sector_columns
        )

    return result
