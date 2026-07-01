"""src/features/hinge_interactions.py — Sprint 3-B Asset-specific Hinge Interaction Features.

Implements asset-specific interaction features:
    feature_{j,t} = hinge(z_{k,t}) × exposure_{j,k,t-1}

Interaction groups:
  G1/G2: macro_hinge_x_asset_beta    — hinge(macro_z) × rolling_beta_{j,macro}
  G3/G4: us_sector_x_sector_exposure — hinge(sector_z) × sector_exposure_{j,sector}
  G5/G6: regime_x_base_signal        — hinge(regime_z) × signal_{j,t}
  G7/G8: gap_asset_specific          — hinge(gap_{j,t}) (asset-specific by nature)

Key sprint 3-B guarantee:
  All interaction features must be ASSET-SPECIFIC (vary across tickers on the same date).
  within-date cross-sectional std > 0 is checked and enforced.

Look-ahead prevention:
  - Hinge z-scores use rolling stats from t-1 only (via shift(1)).
  - Rolling betas use lag_days=1.
  - Static sector exposures are pre-specified (no data contamination).
  - Gap features from per-ticker columns are asset-specific.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

from features.hinge_features import (
    rolling_zscore_lag1,
    positive_hinge,
    negative_hinge,
    _threshold_to_colname,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Within-date cross-sectional std QA
# ---------------------------------------------------------------------------


def compute_within_date_cs_std(
    feature_panel_long: pd.DataFrame,
    feature_cols: list[str],
    date_col: str = "date",
    ticker_col: str = "ticker",
) -> pd.DataFrame:
    """Compute within-date cross-sectional std for each interaction feature.

    Used to enforce that interaction features actually vary across tickers.

    Parameters
    ----------
    feature_panel_long:
        Long-format panel with [date, ticker, feature_cols...].
    feature_cols:
        Columns to evaluate.
    date_col, ticker_col:
        Column names.

    Returns
    -------
    pd.DataFrame
        Per-feature stats: mean_within_date_std, p10, p50, p90, nonzero_date_ratio, use_for_model.
    """
    if feature_panel_long.empty or not feature_cols:
        return pd.DataFrame()

    records = []
    for col in feature_cols:
        if col not in feature_panel_long.columns:
            continue
        daily_std = feature_panel_long.groupby(date_col)[col].std(ddof=1)
        daily_std = daily_std.dropna()

        if len(daily_std) == 0:
            records.append({
                "feature": col,
                "mean_within_date_std": 0.0,
                "p10_within_date_std": 0.0,
                "p50_within_date_std": 0.0,
                "p90_within_date_std": 0.0,
                "nonzero_date_ratio": 0.0,
                "use_for_model": False,
            })
            continue

        mean_std = float(daily_std.mean())
        p10 = float(daily_std.quantile(0.10))
        p50 = float(daily_std.quantile(0.50))
        p90 = float(daily_std.quantile(0.90))
        nonzero_ratio = float((daily_std > 1e-8).mean())

        records.append({
            "feature": col,
            "mean_within_date_std": mean_std,
            "p10_within_date_std": p10,
            "p50_within_date_std": p50,
            "p90_within_date_std": p90,
            "nonzero_date_ratio": nonzero_ratio,
            "use_for_model": (mean_std > 1e-8) and (nonzero_ratio >= 0.10),
        })

    result = pd.DataFrame(records)
    logger.info(
        "Within-date CS std QA: %d features, %d pass (mean_std > 0), %d fail.",
        len(result),
        int(result["use_for_model"].sum()) if not result.empty else 0,
        int((~result["use_for_model"]).sum()) if not result.empty else 0,
    )
    return result


# ---------------------------------------------------------------------------
# G1/G2: Macro hinge × asset rolling beta
# ---------------------------------------------------------------------------


def build_macro_hinge_x_asset_beta(
    macro_z_df: pd.DataFrame,
    rolling_beta_df: pd.DataFrame,
    tickers: list[str],
    macro_cols: list[str],
    beta_windows: Sequence[int],
    thresholds: Sequence[float] = (1.0, 1.5, 2.0),
    directions: Sequence[str] = ("positive", "negative"),
) -> pd.DataFrame:
    """Build macro hinge × asset rolling beta interaction features.

    feature_{j,t} = hinge(z_{macro_k,t}) × beta_{j,k,t-1}

    Result is long-format: (date, ticker) → feature columns.

    Parameters
    ----------
    macro_z_df:
        Rolling z-scores of macro factors (dates × macro_cols). Already lag-1.
    rolling_beta_df:
        Rolling beta panel (dates × [beta_{ticker}_{factor}_w{window}...]).
        Betas are already lag-1.
    tickers:
        List of asset ticker strings.
    macro_cols:
        Macro factor column names present in macro_z_df.
    beta_windows:
        Windows used for rolling beta.
    thresholds:
        Hinge kappa values.
    directions:
        Hinge directions.

    Returns
    -------
    pd.DataFrame
        Long-format panel with columns [date, ticker, feature_cols...].
    """
    available_macro = [c for c in macro_cols if c in macro_z_df.columns]
    if not available_macro:
        logger.warning("No macro columns found in macro_z_df. Returning empty.")
        return pd.DataFrame()

    all_dates = macro_z_df.index
    feature_series_list = []

    # Pre-compute hinge values for all macro cols (date-level)
    hinge_values: dict[tuple, pd.Series] = {}
    for kappa in thresholds:
        for direction in directions:
            for macro_col in available_macro:
                if direction == "positive":
                    h = positive_hinge(macro_z_df[macro_col], kappa)
                else:
                    h = negative_hinge(macro_z_df[macro_col], kappa)
                hinge_values[(kappa, direction, macro_col)] = h

    for kappa in thresholds:
        kappa_str = _threshold_to_colname(kappa)
        for direction in directions:
            dir_prefix = "pos" if direction == "positive" else "neg"
            for macro_col in available_macro:
                h_series = hinge_values.get((kappa, direction, macro_col), pd.Series(np.nan, index=all_dates))
                for w in beta_windows:
                    fname = (
                        f"int_macro_beta_hinge_{dir_prefix}_{macro_col}"
                        f"_k{kappa_str}_beta{w}"
                    )
                    # Get rolling beta columns for all tickers on this macro / window
                    cols = [f"beta_{tk}_{macro_col}_w{w}" for tk in tickers]
                    # Filter only columns present in rolling_beta_df
                    valid_cols = [c for c in cols if c in rolling_beta_df.columns]
                    if not valid_cols:
                        continue
                    
                    beta_wide = rolling_beta_df.reindex(all_dates)[valid_cols]
                    # Map back to tickers
                    col_to_ticker = {f"beta_{tk}_{macro_col}_w{w}": tk for tk in tickers}
                    beta_wide = beta_wide.rename(columns=col_to_ticker)
                    beta_wide.columns.name = "ticker"
                    beta_wide.index.name = "date"
                    
                    # Multiply by hinge
                    feat_wide = beta_wide.mul(h_series, axis=0)
                    feat_wide.index.name = "date"
                    feat_wide.columns.name = "ticker"
                    # Stack to long format
                    feat_series = feat_wide.stack(dropna=False).rename(fname)
                    feature_series_list.append(feat_series)

    if not feature_series_list:
        return pd.DataFrame()

    # Concat all series along columns axis
    result_df = pd.concat(feature_series_list, axis=1).reset_index()
    # Filter to tickers
    result_df = result_df[result_df["ticker"].isin(tickers)]
    logger.info(
        "Macro hinge × asset beta: %d rows, %d interaction features",
        len(result_df), len(feature_series_list)
    )
    return result_df


# ---------------------------------------------------------------------------
# G3/G4: US Sector hinge × sector exposure
# ---------------------------------------------------------------------------


def build_sector_hinge_x_sector_exposure(
    sector_z_df: pd.DataFrame,
    static_sector_map: pd.DataFrame,
    rolling_beta_df: pd.DataFrame | None,
    tickers: list[str],
    sector_cols: list[str],
    beta_windows: Sequence[int] = (120, 252),
    thresholds: Sequence[float] = (1.0, 1.5, 2.0),
    directions: Sequence[str] = ("positive", "negative"),
    use_rolling_beta: bool = True,
    use_static: bool = True,
) -> pd.DataFrame:
    """Build US sector hinge × sector exposure interaction features.

    feature_{j,t} = hinge(z_{sector_k,t}) × exposure_{j,k,t-1}

    Exposure comes from:
    - static_sector_map: time-invariant JP ETF sector weights
    - rolling_beta_df: rolling beta of asset on sector factor (lag-1)

    Parameters
    ----------
    sector_z_df:
        Rolling z-scores of US sector factors (dates × sector_cols).
    static_sector_map:
        Static sector exposure (tickers × sector names).
    rolling_beta_df:
        Rolling beta panel. May be None if not computed.
    tickers:
        List of asset tickers.
    sector_cols:
        Sector factor column names in sector_z_df.
    beta_windows:
        Windows for rolling beta.
    thresholds, directions:
        Hinge parameters.
    use_rolling_beta, use_static:
        Which exposure types to include.

    Returns
    -------
    pd.DataFrame
        Long-format panel [date, ticker, interaction_feature_cols...].
    """
    available_sector = [c for c in sector_cols if c in sector_z_df.columns]
    if not available_sector:
        logger.warning("No sector columns found in sector_z_df.")
        return pd.DataFrame()

    all_dates = sector_z_df.index
    feature_series_list = []

    # Pre-compute hinge values (date-level)
    hinge_values: dict[tuple, pd.Series] = {}
    for kappa in thresholds:
        for direction in directions:
            for sc in available_sector:
                if direction == "positive":
                    h = positive_hinge(sector_z_df[sc], kappa)
                else:
                    h = negative_hinge(sector_z_df[sc], kappa)
                hinge_values[(kappa, direction, sc)] = h

    # Normalize sector names for matching
    sector_to_map_col = {}
    if static_sector_map is not None:
        for sc in available_sector:
            map_key = sc.replace("_return", "")
            if map_key in static_sector_map.columns:
                sector_to_map_col[sc] = map_key

    for kappa in thresholds:
        kappa_str = _threshold_to_colname(kappa)
        for direction in directions:
            dir_prefix = "pos" if direction == "positive" else "neg"
            for sc in available_sector:
                h_series = hinge_values.get((kappa, direction, sc), pd.Series(np.nan, index=all_dates))

                # Static exposure
                if use_static and sc in sector_to_map_col:
                    map_col = sector_to_map_col[sc]
                    fname = f"int_sector_static_hinge_{dir_prefix}_{sc}_k{kappa_str}"
                    
                    # Create wide DataFrame where each column has the static exposure of that ticker
                    exposures = [float(static_sector_map.loc[tk, map_col]) if tk in static_sector_map.index else 0.0 for tk in tickers]
                    static_wide = pd.DataFrame(
                        np.repeat(np.array(exposures)[np.newaxis, :], len(all_dates), axis=0),
                        index=all_dates,
                        columns=tickers
                    )
                    static_wide.columns.name = "ticker"
                    static_wide.index.name = "date"
                    
                    feat_wide = static_wide.mul(h_series, axis=0)
                    feat_wide.index.name = "date"
                    feat_wide.columns.name = "ticker"
                    feat_series = feat_wide.stack(dropna=False).rename(fname)
                    feature_series_list.append(feat_series)

                # Rolling beta exposure
                if use_rolling_beta and rolling_beta_df is not None:
                    for w in beta_windows:
                        fname = f"int_sector_beta_hinge_{dir_prefix}_{sc}_k{kappa_str}_beta{w}"
                        cols = [f"beta_{tk}_{sc}_w{w}" for tk in tickers]
                        valid_cols = [c for c in cols if c in rolling_beta_df.columns]
                        if not valid_cols:
                            continue
                        
                        beta_wide = rolling_beta_df.reindex(all_dates)[valid_cols]
                        col_to_ticker = {f"beta_{tk}_{sc}_w{w}": tk for tk in tickers}
                        beta_wide = beta_wide.rename(columns=col_to_ticker)
                        beta_wide.columns.name = "ticker"
                        beta_wide.index.name = "date"
                        
                        feat_wide = beta_wide.mul(h_series, axis=0)
                        feat_wide.index.name = "date"
                        feat_wide.columns.name = "ticker"
                        feat_series = feat_wide.stack(dropna=False).rename(fname)
                        feature_series_list.append(feat_series)

    if not feature_series_list:
        return pd.DataFrame()

    result_df = pd.concat(feature_series_list, axis=1).reset_index()
    result_df = result_df[result_df["ticker"].isin(tickers)]
    logger.info(
        "Sector hinge × sector exposure: %d rows, %d interaction features",
        len(result_df), len(feature_series_list)
    )
    return result_df


# ---------------------------------------------------------------------------
# G5/G6: Regime hinge × base signal
# ---------------------------------------------------------------------------


def build_regime_hinge_x_base_signal(
    regime_z_df: pd.DataFrame,
    signal_panel: pd.DataFrame,
    tickers: list[str],
    regime_cols: list[str],
    thresholds: Sequence[float] = (1.0, 1.5, 2.0),
    directions: Sequence[str] = ("positive", "negative"),
) -> pd.DataFrame:
    """Build regime hinge × base signal interaction features.

    feature_{j,t} = hinge(z_{regime_k,t}) × signal_{j,t}

    This creates asset-specific features because signal_{j,t} varies by ticker.

    Parameters
    ----------
    regime_z_df:
        Rolling z-scores of regime indicators (dates × regime_cols).
    signal_panel:
        Wide panel of baseline signals (dates × tickers).
    tickers:
        List of asset tickers.
    regime_cols:
        Regime column names present in regime_z_df.
    thresholds, directions:
        Hinge parameters.

    Returns
    -------
    pd.DataFrame
        Long-format panel [date, ticker, interaction_feature_cols...].
    """
    available_regime = [c for c in regime_cols if c in regime_z_df.columns]
    if not available_regime:
        logger.warning("No regime columns found in regime_z_df.")
        return pd.DataFrame()

    all_dates = regime_z_df.index.intersection(signal_panel.index)
    feature_series_list = []

    # Pre-compute hinge values (date-level)
    hinge_values: dict[tuple, pd.Series] = {}
    for kappa in thresholds:
        for direction in directions:
            for rc in available_regime:
                if direction == "positive":
                    h = positive_hinge(regime_z_df[rc], kappa)
                else:
                    h = negative_hinge(regime_z_df[rc], kappa)
                hinge_values[(kappa, direction, rc)] = h

    # Ensure it's reindexed
    sig_wide = signal_panel.reindex(index=all_dates, columns=tickers)
    sig_wide.columns.name = "ticker"
    sig_wide.index.name = "date"

    for kappa in thresholds:
        kappa_str = _threshold_to_colname(kappa)
        for direction in directions:
            dir_prefix = "pos" if direction == "positive" else "neg"
            for rc in available_regime:
                h_series = hinge_values.get((kappa, direction, rc), pd.Series(np.nan, index=all_dates))
                fname = f"int_regime_signal_hinge_{dir_prefix}_{rc}_k{kappa_str}"
                
                feat_wide = sig_wide.mul(h_series, axis=0)
                feat_wide.index.name = "date"
                feat_wide.columns.name = "ticker"
                feat_series = feat_wide.stack(dropna=False).rename(fname)
                feature_series_list.append(feat_series)

    if not feature_series_list:
        return pd.DataFrame()

    result_df = pd.concat(feature_series_list, axis=1).reset_index()
    result_df = result_df[result_df["ticker"].isin(tickers)]
    logger.info(
        "Regime hinge × base signal: %d rows, %d interaction features",
        len(result_df), len(feature_series_list)
    )
    return result_df


# ---------------------------------------------------------------------------
# G7/G8: Gap asset-specific hinge
# ---------------------------------------------------------------------------


def build_gap_asset_specific_hinge(
    gap_panel_long: pd.DataFrame,
    tickers: list[str],
    gap_cols: list[str],
    zscore_window: int = 120,
    thresholds: Sequence[float] = (1.0, 1.5, 2.0),
    directions: Sequence[str] = ("positive", "negative"),
) -> pd.DataFrame:
    """Build asset-specific gap hinge features.

    For gap columns that are asset-specific (vary by ticker),
    computes rolling z-score per ticker and applies hinge.

    feature_{j,t} = hinge(z_{gap_{j,t}})

    Parameters
    ----------
    gap_panel_long:
        Long-format panel with [date, ticker, gap_cols...].
        Gap values must be per-ticker (asset-specific).
    tickers:
        List of asset tickers.
    gap_cols:
        Gap column names to process.
    zscore_window:
        Rolling window for z-score (per ticker).
    thresholds, directions:
        Hinge parameters.

    Returns
    -------
    pd.DataFrame
        Long-format panel [date, ticker, interaction_feature_cols...].
    """
    available_gap = [c for c in gap_cols if c in gap_panel_long.columns]
    if not available_gap:
        logger.warning("No gap columns found in gap_panel_long.")
        return pd.DataFrame()

    # Verify asset-specificity: columns must vary across tickers per date
    cs_std_check = compute_within_date_cs_std(
        gap_panel_long, available_gap, date_col="date", ticker_col="ticker"
    )
    if not cs_std_check.empty:
        non_asset_specific = cs_std_check[~cs_std_check["use_for_model"]]["feature"].tolist()
        if non_asset_specific:
            logger.warning(
                "Gap columns with zero cross-sectional std (NOT asset-specific), skipping: %s",
                non_asset_specific
            )
            available_gap = [c for c in available_gap if c not in non_asset_specific]

    if not available_gap:
        return pd.DataFrame()

    feature_series_list = []

    for gap_col in available_gap:
        # Pivot to wide: dates × tickers
        try:
            wide = gap_panel_long.pivot(index="date", columns="ticker", values=gap_col)
        except Exception:
            wide = gap_panel_long.pivot_table(
                index="date", columns="ticker", values=gap_col, aggfunc="first"
            )

        # Rolling z-score per ticker (lag-1 within each ticker column)
        min_periods = max(2, zscore_window // 2)
        roll_mean = wide.shift(1).rolling(window=zscore_window, min_periods=min_periods).mean()
        roll_std = wide.shift(1).rolling(window=zscore_window, min_periods=min_periods).std()
        z_wide = (wide - roll_mean) / roll_std.replace(0, np.nan)
        z_wide = z_wide.replace([np.inf, -np.inf], np.nan)

        z_val = z_wide.values

        for kappa in thresholds:
            kappa_str = _threshold_to_colname(kappa)
            for direction in directions:
                dir_prefix = "pos" if direction == "positive" else "neg"
                fname = f"int_gap_asset_hinge_{dir_prefix}_{gap_col}_k{kappa_str}"

                if direction == "positive":
                    h_val = np.where(np.isnan(z_val), np.nan, np.maximum(0.0, z_val - kappa))
                else:
                    h_val = np.where(np.isnan(z_val), np.nan, np.maximum(0.0, -z_val - kappa))

                h_wide = pd.DataFrame(h_val, index=z_wide.index, columns=z_wide.columns)
                h_wide.columns.name = "ticker"
                h_wide.index.name = "date"
                feat_series = h_wide.stack(dropna=False).rename(fname)
                feature_series_list.append(feat_series)

    if not feature_series_list:
        return pd.DataFrame()

    result_df = pd.concat(feature_series_list, axis=1).reset_index()
    result_df = result_df[result_df["ticker"].isin(tickers)]
    logger.info(
        "Gap asset-specific hinge: %d rows, %d interaction features",
        len(result_df), len(feature_series_list)
    )
    return result_df


# ---------------------------------------------------------------------------
# Combine all interaction groups
# ---------------------------------------------------------------------------


def build_all_interaction_features(
    macro_interactions: pd.DataFrame | None,
    sector_interactions: pd.DataFrame | None,
    regime_interactions: pd.DataFrame | None,
    gap_interactions: pd.DataFrame | None,
    max_raw_features: int = 120,
) -> pd.DataFrame:
    """Merge all interaction feature groups into a single long-format panel.

    Parameters
    ----------
    macro_interactions:
        Output of build_macro_hinge_x_asset_beta.
    sector_interactions:
        Output of build_sector_hinge_x_sector_exposure.
    regime_interactions:
        Output of build_regime_hinge_x_base_signal.
    gap_interactions:
        Output of build_gap_asset_specific_hinge.
    max_raw_features:
        Cap on total features (after this, randomly sample if exceeded).

    Returns
    -------
    pd.DataFrame
        Long-format panel [date, ticker, all_feature_cols...].
    """
    frames = []
    for frame in [macro_interactions, sector_interactions, regime_interactions, gap_interactions]:
        if frame is not None and not frame.empty and "date" in frame.columns:
            frames.append(frame)

    if not frames:
        logger.warning("No interaction feature groups available.")
        return pd.DataFrame()

    # Merge all on (date, ticker)
    combined = frames[0]
    for frame in frames[1:]:
        feature_cols = [c for c in frame.columns if c not in ["date", "ticker"]]
        combined = combined.merge(frame[["date", "ticker"] + feature_cols], on=["date", "ticker"], how="outer")

    # Cap features if needed
    feature_cols = [c for c in combined.columns if c not in ["date", "ticker"]]
    if len(feature_cols) > max_raw_features:
        logger.warning(
            "Interaction features capped: %d → %d", len(feature_cols), max_raw_features
        )
        import random
        random.seed(42)
        # Sort first to ensure deterministic random sampling regardless of input ordering
        feature_cols_sorted = sorted(feature_cols)
        feature_cols = random.sample(feature_cols_sorted, max_raw_features)
        combined = combined[["date", "ticker"] + feature_cols]

    logger.info(
        "Combined interaction feature panel: %d rows × %d features",
        len(combined), len(feature_cols)
    )
    return combined
