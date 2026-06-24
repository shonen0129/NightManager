"""src/features/hinge_features.py — Sprint 3-A Hinge Feature Generation.

Implements rolling z-score standardization (t-1 only, no look-ahead) and
positive/negative hinge transformations for US sector, macro, gap, and regime
indicators.

Look-ahead prevention rules:
- rolling mean/std at date t uses only data up to date t-1 (shift(1)).
- full-sample zscore is PROHIBITED.
- std == 0 or NaN → feature is NaN for that date.
- NaN imputation only within train window.

Output column naming:
    hinge_pos_{feature}_k{threshold_str}
    hinge_neg_{feature}_k{threshold_str}

where threshold_str replaces "." with "_" (e.g. 1.5 → "1_5").
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rolling z-score (look-ahead free)
# ---------------------------------------------------------------------------


def rolling_zscore_lag1(
    series: pd.Series,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    """Compute rolling z-score using only t-1 data (no look-ahead).

    z_{k,t} = (x_{k,t} - mu_{k,t-1}^roll) / sigma_{k,t-1}^roll

    Parameters
    ----------
    series:
        Raw feature time series (DatetimeIndex).
    window:
        Rolling window length in trading days.
    min_periods:
        Minimum observations required (defaults to window // 2).

    Returns
    -------
    pd.Series
        z-score series; NaN when std is zero or insufficient data.
    """
    if min_periods is None:
        min_periods = max(2, window // 2)

    # Shift by 1: rolling statistics are computed from data up to t-1
    roll_mean = series.shift(1).rolling(window=window, min_periods=min_periods).mean()
    roll_std = series.shift(1).rolling(window=window, min_periods=min_periods).std()

    # z = (x_t - mu_{t-1}) / sigma_{t-1}
    z = (series - roll_mean) / roll_std

    # Replace inf/–inf arising from zero std
    z = z.replace([np.inf, -np.inf], np.nan)

    return z


def rolling_zscore_panel_lag1(
    df: pd.DataFrame,
    window: int,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Apply rolling_zscore_lag1 to every column of df independently.

    Parameters
    ----------
    df:
        DataFrame where columns are features, rows are dates.
    window:
        Rolling window length.
    min_periods:
        Minimum observations (defaults to window // 2).

    Returns
    -------
    pd.DataFrame
        z-score panel, same shape as df.
    """
    return df.apply(
        lambda col: rolling_zscore_lag1(col, window=window, min_periods=min_periods)
    )


# ---------------------------------------------------------------------------
# Hinge transformation
# ---------------------------------------------------------------------------


def positive_hinge(z: pd.Series | pd.DataFrame, kappa: float) -> pd.Series | pd.DataFrame:
    """Compute positive hinge: max(0, z - kappa)."""
    return np.maximum(0.0, z - kappa)


def negative_hinge(z: pd.Series | pd.DataFrame, kappa: float) -> pd.Series | pd.DataFrame:
    """Compute negative hinge: max(0, -z - kappa)."""
    return np.maximum(0.0, -z - kappa)


def _threshold_to_colname(kappa: float) -> str:
    """Convert threshold float to column name suffix (1.5 → '1_5')."""
    return str(kappa).replace(".", "_")


# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------


def build_hinge_features(
    df_raw: pd.DataFrame,
    feature_columns: list[str],
    thresholds: Sequence[float] = (1.0, 1.5, 2.0),
    directions: Sequence[str] = ("positive", "negative"),
    zscore_window: int = 120,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Build hinge features from raw feature columns.

    Applies rolling z-score (lag-1, no look-ahead) and then hinge transforms.
    Missing raw columns are skipped with a warning.

    Parameters
    ----------
    df_raw:
        Raw feature panel (rows = dates, columns include feature_columns).
    feature_columns:
        List of raw feature column names to transform.
    thresholds:
        List of kappa values for hinge.
    directions:
        Subset of {"positive", "negative"}.
    zscore_window:
        Rolling window for z-score computation.
    min_periods:
        Minimum periods for rolling stats.

    Returns
    -------
    pd.DataFrame
        Hinge feature panel. Shape: (n_dates, n_features * n_thresholds * n_directions).
        All NaN-producing rows are preserved as NaN (no imputation done here).
    """
    available_cols = [c for c in feature_columns if c in df_raw.columns]
    missing_cols = [c for c in feature_columns if c not in df_raw.columns]
    if missing_cols:
        logger.warning(
            "Hinge feature builder: %d columns not found in df_raw, skipping: %s",
            len(missing_cols),
            missing_cols,
        )

    if not available_cols:
        logger.warning("No valid feature columns found. Returning empty DataFrame.")
        return pd.DataFrame(index=df_raw.index)

    raw_features = df_raw[available_cols].copy()

    # Compute rolling z-scores (lag-1, no look-ahead)
    z_scores = rolling_zscore_panel_lag1(raw_features, window=zscore_window, min_periods=min_periods)

    hinge_frames: list[pd.DataFrame] = []

    for kappa in thresholds:
        kappa_str = _threshold_to_colname(kappa)

        for direction in directions:
            if direction == "positive":
                h = positive_hinge(z_scores, kappa)
                prefix = "hinge_pos"
            elif direction == "negative":
                h = negative_hinge(z_scores, kappa)
                prefix = "hinge_neg"
            else:
                raise ValueError(f"Unknown direction '{direction}'. Expected 'positive' or 'negative'.")

            # Rename columns: hinge_pos_{feature}_k{kappa_str}
            h.columns = [f"{prefix}_{col}_k{kappa_str}" for col in available_cols]
            hinge_frames.append(h)

    if not hinge_frames:
        return pd.DataFrame(index=df_raw.index)

    result = pd.concat(hinge_frames, axis=1)
    logger.info(
        "Built %d hinge features from %d raw columns (thresholds=%s, directions=%s, window=%d).",
        result.shape[1],
        len(available_cols),
        list(thresholds),
        list(directions),
        zscore_window,
    )
    return result


# ---------------------------------------------------------------------------
# Feature registry: derive available raw columns from df_exec
# ---------------------------------------------------------------------------


def get_available_feature_columns(
    df_exec: pd.DataFrame,
    feature_groups: dict,
) -> list[str]:
    """Return list of raw feature columns that exist in df_exec.

    Parameters
    ----------
    df_exec:
        Execution DataFrame from local cache.
    feature_groups:
        Dict mapping group names to {"enabled": bool, "columns": [...]}.

    Returns
    -------
    list[str]
        Available column names in df_exec.
    """
    requested: list[str] = []
    for group_name, group_cfg in feature_groups.items():
        if not group_cfg.get("enabled", True):
            continue
        requested.extend(group_cfg.get("columns", []))

    available = [c for c in requested if c in df_exec.columns]
    missing = [c for c in requested if c not in df_exec.columns]

    if missing:
        logger.warning(
            "Feature group column resolution: %d columns not found in df_exec "
            "(will skip): %s",
            len(missing),
            missing,
        )

    return available


# ---------------------------------------------------------------------------
# Proxy column builder: derive common macro/US proxies from df_exec columns
# ---------------------------------------------------------------------------


def derive_proxy_features(
    df_exec: pd.DataFrame,
    macro_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Derive macro and US sector proxy features from available df_exec columns.

    Constructs the following derived columns (if source data is present):
    - vix_return: daily log-return of VIX (from macro_df)
    - usd_jpy_return: daily log-return of USD/JPY (from macro_df)
    - us10y_change: daily change in US 10yr yield (from macro_df)
    - topix_futures_return: already in df_exec as topix_cc_trade (proxy)
    - realized_vol_20d: 20-day rolling std of topix_oc_return (shifted by 1)
    - cross_sectional_dispersion: daily std across JP ETF intraday returns (shifted by 1)
    - topix_intraday_vol: rolling 20d std of topix_oc_return (shifted by 1)

    For US sector returns, these require separate macro data.
    If macro_df is None, only basic proxies from df_exec are built.

    Parameters
    ----------
    df_exec:
        Execution DataFrame (DatetimeIndex).
    macro_df:
        Optional macro DataFrame with columns like '^VIX', 'USDJPY=X', '^TNX'.

    Returns
    -------
    pd.DataFrame
        DataFrame of derived proxy features.
    """
    from leadlag.data.tickers import JP_TICKERS

    proxies: dict[str, pd.Series] = {}

    # --- Macro proxies from df_exec ---
    if "topix_oc_return" in df_exec.columns:
        proxies["topix_futures_return"] = df_exec["topix_oc_return"]
        # realized_vol_20d: rolling std shifted by 1 (no look-ahead)
        proxies["realized_vol_20d"] = (
            df_exec["topix_oc_return"]
            .rolling(20, min_periods=5)
            .std()
            .shift(1)
        )
        proxies["topix_intraday_vol"] = proxies["realized_vol_20d"]

    if "topix_cc_trade" in df_exec.columns:
        proxies["nikkei_futures_return"] = df_exec["topix_cc_trade"]

    # Cross-sectional dispersion of JP ETF intraday returns (shifted 1 day)
    oc_cols = [f"jp_oc_{tk}" for tk in JP_TICKERS if f"jp_oc_{tk}" in df_exec.columns]
    if oc_cols:
        oc_panel = df_exec[oc_cols].copy()
        proxies["cross_sectional_dispersion"] = (
            oc_panel.std(axis=1).shift(1)
        )

    # Gap proxies
    gap_cols = [f"jp_gap_{tk}" for tk in JP_TICKERS if f"jp_gap_{tk}" in df_exec.columns]
    if gap_cols:
        gap_panel = df_exec[gap_cols].copy()
        proxies["jp_open_gap"] = gap_panel.mean(axis=1)
        proxies["gap_surprise"] = gap_panel.std(axis=1)

    # Entry gap: mean of gap columns (proxy for gap_adjustment_term)
    if gap_cols and oc_cols:
        proxies["entry_gap"] = proxies.get("jp_open_gap", pd.Series(0.0, index=df_exec.index))
        proxies["gap_adjustment_term"] = proxies.get("jp_open_gap", pd.Series(0.0, index=df_exec.index))

    # --- Macro proxies from macro_df ---
    if macro_df is not None:
        macro_aligned = macro_df.reindex(df_exec.index).ffill()

        if "^VIX" in macro_aligned.columns:
            vix = macro_aligned["^VIX"].replace(0, np.nan)
            proxies["vix_return"] = np.log(vix / vix.shift(1))

        if "USDJPY=X" in macro_aligned.columns:
            usdjpy = macro_aligned["USDJPY=X"].replace(0, np.nan)
            proxies["usd_jpy_return"] = np.log(usdjpy / usdjpy.shift(1))

        if "^TNX" in macro_aligned.columns:
            proxies["us10y_change"] = macro_aligned["^TNX"].diff()

        # US sector proxies: map ETF return columns to sector names
        sector_map = {
            "QQQ": "us_tech_return",
            "SOXX": "us_semiconductor_return",
            "XLE": "us_energy_return",
            "XLF": "us_financial_return",
            "XLI": "us_industrial_return",
            "XLV": "us_healthcare_return",
            "IWM": "us_smallcap_return",
        }
        for src_col, dst_col in sector_map.items():
            if src_col in macro_aligned.columns:
                p = macro_aligned[src_col].replace(0, np.nan)
                proxies[dst_col] = np.log(p / p.shift(1))

    result = pd.DataFrame(proxies, index=df_exec.index)
    result = result.replace([np.inf, -np.inf], np.nan)

    logger.info(
        "Derived %d proxy feature columns: %s",
        result.shape[1],
        list(result.columns),
    )
    return result


# ---------------------------------------------------------------------------
# Build full feature panel
# ---------------------------------------------------------------------------


def build_full_feature_panel(
    df_exec: pd.DataFrame,
    feature_groups: dict,
    thresholds: Sequence[float] = (1.0, 1.5, 2.0),
    directions: Sequence[str] = ("positive", "negative"),
    zscore_window: int = 120,
    macro_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build complete hinge feature panel combining all feature groups.

    Parameters
    ----------
    df_exec:
        Execution DataFrame.
    feature_groups:
        Config dict of feature groups (from YAML).
    thresholds:
        Hinge kappa values.
    directions:
        Hinge directions to compute.
    zscore_window:
        Rolling z-score window.
    macro_df:
        Optional macro data for US sector / macro features.

    Returns
    -------
    (raw_features_df, hinge_features_df):
        raw_features_df: panel of raw z-scored features (before hinge).
        hinge_features_df: panel of hinge-transformed features.
    """
    # Build proxy features first
    proxy_df = derive_proxy_features(df_exec, macro_df=macro_df)

    # Combine with df_exec to get full source
    combined = pd.concat([df_exec, proxy_df], axis=1)
    # Drop duplicate columns (df_exec columns take precedence)
    combined = combined.loc[:, ~combined.columns.duplicated(keep="first")]

    # Collect requested raw columns
    requested_cols = get_available_feature_columns(combined, feature_groups)

    if not requested_cols:
        logger.warning("No feature columns available. Returning empty panels.")
        empty = pd.DataFrame(index=df_exec.index)
        return empty, empty

    # Build hinge features
    hinge_df = build_hinge_features(
        df_raw=combined,
        feature_columns=requested_cols,
        thresholds=thresholds,
        directions=directions,
        zscore_window=zscore_window,
    )

    # Raw z-scores panel (for diagnostics)
    raw_z_df = rolling_zscore_panel_lag1(
        combined[requested_cols], window=zscore_window
    )

    return raw_z_df, hinge_df
