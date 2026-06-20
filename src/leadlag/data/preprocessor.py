"""Data preprocessor — builds the execution DataFrame from raw OHLC data.

Transforms the raw ``{"us_close", "jp_close", "jp_open"}`` dict returned by
`leadlag.data.fetcher.download_data()` into the ``df_exec`` DataFrame used by
the strategy engine and backtesting framework.

The output DataFrame ``df_exec``:
- Index: ``trade_date`` (JP trading day on which the order is executed)
- Columns: ``sig_date``, ``us_cc_*``, ``jp_cc_*``, ``jp_oc_*``, ``jp_gap_*``,
  ``jp_close_sig_*``, ``jp_open_trade_*``, ``topix_night_return``,
  ``jp_beta_*``
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from leadlag.data.tickers import JP_TICKERS, TOPIX_TICKER, US_TICKERS

logger = logging.getLogger(__name__)


def preprocess_data(data: dict, beta_window: int = 60) -> pd.DataFrame:
    """Align raw OHLC data and build the execution DataFrame.

    Steps:
    1. Strip timezone, find joint US/JP trading dates
    2. Proxy early returns for XLC (XLK+XLY avg) and XLRE (XLF)
    3. Map each joint date to the next JP trading day (trade_date)
    4. Build df_exec with signal and return columns
    5. Append TOPIX overnight return and rolling JP betas

    Args:
        data: Dict with keys "us_close", "jp_close", "jp_open" (DataFrames)
        beta_window: Rolling window for beta computation (default 60 days)

    Returns:
        df_exec DataFrame indexed by trade_date
    """
    us_c = data["us_close"].copy()
    jp_c = data["jp_close"].copy()
    jp_o = data["jp_open"].copy()

    # Normalize indices to tz-naive daily dates
    us_c.index = pd.to_datetime(us_c.index).tz_localize(None).normalize()
    jp_c.index = pd.to_datetime(jp_c.index).tz_localize(None).normalize()
    jp_o.index = pd.to_datetime(jp_o.index).tz_localize(None).normalize()

    # Separate TOPIX proxy from sector ETFs
    topix_close = jp_c[TOPIX_TICKER].copy() if TOPIX_TICKER in jp_c.columns else None
    topix_open = jp_o[TOPIX_TICKER].copy() if TOPIX_TICKER in jp_o.columns else None
    if TOPIX_TICKER in jp_c.columns:
        jp_c = jp_c[JP_TICKERS].copy()
    if TOPIX_TICKER in jp_o.columns:
        jp_o = jp_o[JP_TICKERS].copy()

    # Joint dates: days where both US and JP have valid data
    us_valid_dates = us_c.dropna(subset=["XLB"]).index
    jp_valid_dates = jp_c.dropna(subset=["1617.T"]).index
    joint_dates = us_valid_dates.intersection(jp_valid_dates).sort_values()

    us_c_joint = us_c.loc[joint_dates]
    jp_c_joint = jp_c.loc[joint_dates]

    # Close-to-close returns
    ret_us_cc = us_c_joint.pct_change(fill_method=None)
    ret_jp_cc = jp_c_joint.pct_change(fill_method=None)

    # Proxy returns for ETFs with limited history
    if "XLC" in ret_us_cc.columns and ret_us_cc["XLC"].isna().any():
        logger.info("Proxying XLC returns with average of XLK and XLY")
        ret_us_cc["XLC"] = ret_us_cc["XLC"].fillna((ret_us_cc["XLK"] + ret_us_cc["XLY"]) / 2)
    if "XLRE" in ret_us_cc.columns and ret_us_cc["XLRE"].isna().any():
        logger.info("Proxying XLRE returns with XLF")
        ret_us_cc["XLRE"] = ret_us_cc["XLRE"].fillna(ret_us_cc["XLF"])
    if (
        "MTUM" in ret_us_cc.columns
        and "IUSG" in ret_us_cc.columns
        and ret_us_cc["MTUM"].isna().any()
    ):
        logger.info("Proxying MTUM returns with IUSG")
        ret_us_cc["MTUM"] = ret_us_cc["MTUM"].fillna(ret_us_cc["IUSG"])
    if "VLUE" in ret_us_cc.columns and ret_us_cc["VLUE"].isna().any():
        logger.info("Proxying VLUE returns with XLF")
        ret_us_cc["VLUE"] = ret_us_cc["VLUE"].fillna(ret_us_cc["XLF"])
    if "USMV" in ret_us_cc.columns and ret_us_cc["USMV"].isna().any():
        logger.info("Proxying USMV returns with average of XLP and XLV")
        ret_us_cc["USMV"] = ret_us_cc["USMV"].fillna((ret_us_cc["XLP"] + ret_us_cc["XLV"]) / 2)

    # Map each joint date T to the next JP trading day (trade_date)
    trade_targets: dict = {}
    for t in joint_dates:
        future_jp_dates = jp_valid_dates[jp_valid_dates > t]
        if len(future_jp_dates) > 0:
            trade_targets[t] = future_jp_dates[0]

    # OC and gap returns for JP
    ret_jp_oc = jp_c / jp_o - 1.0
    ret_jp_gap = jp_o / jp_c.shift(1) - 1.0

    # TOPIX overnight and rolling betas
    topix_night = None
    beta_df = None
    if topix_close is not None and topix_open is not None:
        topix_close.index = pd.to_datetime(topix_close.index).tz_localize(None).normalize()
        topix_open.index = pd.to_datetime(topix_open.index).tz_localize(None).normalize()
        topix_night = topix_open / topix_close.shift(1) - 1.0
        topix_var = topix_night.rolling(beta_window).var()

        betas: dict[str, pd.Series] = {}
        for tk in JP_TICKERS:
            cov = ret_jp_gap[tk].rolling(beta_window).cov(topix_night)
            betas[tk] = cov / topix_var
        beta_df = pd.DataFrame(betas)

    # Build execution records
    records = []
    for sig_date in joint_dates:
        if sig_date not in trade_targets:
            continue
        trade_date = trade_targets[sig_date]
        if trade_date not in ret_jp_oc.index:
            continue

        r_us = ret_us_cc.loc[sig_date]
        r_jp = ret_jp_cc.loc[sig_date]
        r_oc = ret_jp_oc.loc[trade_date]
        r_gap = ret_jp_gap.loc[trade_date]
        jp_close_sig = jp_c_joint.loc[sig_date]
        jp_open_trade = jp_o.loc[trade_date]

        if (
            r_us.isna().any()
            or r_jp.isna().any()
            or r_oc.isna().any()
            or r_gap.isna().any()
            or jp_close_sig.isna().any()
            or jp_open_trade.isna().any()
        ):
            continue

        record: dict = {"trade_date": trade_date, "sig_date": sig_date}
        for tk in US_TICKERS:
            record[f"us_cc_{tk}"] = r_us[tk]
        for tk in JP_TICKERS:
            record[f"jp_cc_{tk}"] = r_jp[tk]
            record[f"jp_oc_{tk}"] = r_oc[tk]
            record[f"jp_gap_{tk}"] = r_gap[tk]
            record[f"jp_close_sig_{tk}"] = jp_close_sig[tk]
            record[f"jp_open_trade_{tk}"] = jp_open_trade[tk]

        records.append(record)

    df_exec = pd.DataFrame(records).set_index("trade_date").sort_index()
    logger.info("Total valid trading days constructed: %d", len(df_exec))

    # Append TOPIX night return
    df_exec["topix_night_return"] = np.nan
    if topix_night is not None:
        df_exec["topix_night_return"] = topix_night.reindex(df_exec.index).values

    # Append TOPIX oc return and cc trade return
    df_exec["topix_oc_return"] = np.nan
    df_exec["topix_cc_trade"] = np.nan
    if topix_close is not None and topix_open is not None:
        r_topix_oc = topix_close / topix_open - 1.0
        df_exec["topix_oc_return"] = r_topix_oc.reindex(df_exec.index).values
        df_exec["topix_cc_trade"] = (1.0 + df_exec["topix_night_return"]) * (
            1.0 + df_exec["topix_oc_return"]
        ) - 1.0

    # Append rolling JP betas
    for tk in JP_TICKERS:
        beta_col = f"jp_beta_{tk}"
        if beta_df is not None and tk in beta_df.columns:
            df_exec[beta_col] = beta_df[tk].reindex(df_exec.index).values
        else:
            df_exec[beta_col] = np.nan

    return df_exec


def compute_us_residualized_returns(
    us_returns: np.ndarray,
    spy_returns: np.ndarray,
    beta_window: int = 60,
    gamma: float = 0.5,
) -> np.ndarray:
    """Compute rolling US residualized returns using SPY as benchmark.

    beta_us[u, t-1] is estimated on [t-beta_window, ..., t-1].
    r_us_adj[u, t] = r_us[u, t] - gamma * beta_us[u, t-1] * r_mkt[t]
    """
    # Replace any NaNs or Infs with 0.0 at the very beginning to avoid propagation
    us_returns = np.nan_to_num(us_returns, nan=0.0, posinf=0.0, neginf=0.0)
    spy_returns = np.nan_to_num(spy_returns, nan=0.0, posinf=0.0, neginf=0.0)

    T, n_u = us_returns.shape
    
    us_df = pd.DataFrame(us_returns)
    spy_series = pd.Series(spy_returns)
    
    cov_rolling = us_df.rolling(beta_window).cov(spy_series)
    var_rolling = spy_series.rolling(beta_window).var()
    
    var_mask = var_rolling > 1e-12
    betas_raw = cov_rolling.divide(var_rolling.where(var_mask, np.nan), axis=0)
    betas_shifted = betas_raw.shift(1)
    
    # Any non-finite values (NaN/inf) should be treated as NaN to be filled
    betas_shifted = betas_shifted.where(np.isfinite(betas_shifted), np.nan)
    
    betas_val = betas_shifted.values
    
    # Also ensure first beta_window rows are 0.0
    betas_val[:beta_window] = 0.0
    
    # If there are no NaNs in betas_val (common case for clean data), we can skip the loop
    if np.isnan(betas_val).any():
        for t in range(beta_window, T):
            row = betas_val[t]
            prev_row = betas_val[t - 1]
            
            is_finite_prev = np.isfinite(prev_row).all() if t > beta_window else False
            
            nan_mask = np.isnan(row)
            if np.any(nan_mask):
                if is_finite_prev:
                    betas_val[t, nan_mask] = prev_row[nan_mask]
                else:
                    betas_val[t, nan_mask] = 1.0
                    
    r_us_adj = us_returns - gamma * betas_val * spy_returns[:, np.newaxis]

    # Final fallback check to guarantee no NaNs/infs
    if not np.isfinite(r_us_adj).all():
        bad_mask = ~np.isfinite(r_us_adj)
        r_us_adj[bad_mask] = us_returns[bad_mask]

    return r_us_adj

