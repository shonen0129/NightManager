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
from typing import cast

import numpy as np
import pandas as pd

from leadlag.core.market_calendar import next_trading_day
from leadlag.data.tickers import JP_TICKERS, TOPIX_TICKER, US_TICKERS
from leadlag.data.validation import (
    DataValidationError,
    validate_exec_record,
    validate_raw_data_sources,
)

logger = logging.getLogger(__name__)


def _winsorize_rolling(
    series: pd.Series | pd.DataFrame,
    window: int,
    n_sigma: float = 3.0,
) -> pd.Series | pd.DataFrame:
    """Rolling winsorize: clip values beyond n_sigma from rolling mean.

    Args:
        series: Series or DataFrame to winsorize
        window: Rolling window size
        n_sigma: Number of standard deviations for clipping

    Returns:
        Winsorized Series or DataFrame
    """
    # Shift bounds by 1 so the current observation does not influence its own
    # clipping threshold.
    rolling_mean = series.rolling(window, min_periods=window).mean().shift(1)
    rolling_std = series.rolling(window, min_periods=window).std().shift(1)
    lower = rolling_mean - n_sigma * rolling_std
    upper = rolling_mean + n_sigma * rolling_std
    return series.clip(lower=lower, upper=upper)


def _compute_ewma_betas(
    ret_jp_gap: pd.DataFrame,
    topix_night: pd.Series,
    beta_window: int,
    ewma_halflife: float,
) -> pd.DataFrame:
    """Compute EWMA-weighted rolling betas for JP gap returns vs TOPIX night.

    Uses exponentially weighted covariance and variance within a fixed rolling
    window of ``beta_window`` days. Recent observations receive higher weight
    (controlled by ``ewma_halflife``) while old data is fully discarded after
    the window expires — ensuring lookahead-safe behavior identical to rolling
    OLS, but with smoother and more responsive estimates.

    Args:
        ret_jp_gap: DataFrame of JP gap returns (per ticker columns)
        topix_night: Series of TOPIX overnight returns
        beta_window: Rolling window size (trading days)
        ewma_halflife: EWMA half-life (trading days)

    Returns:
        DataFrame of rolling betas, same shape as ret_jp_gap
    """
    n = len(ret_jp_gap)
    decay = 0.5 ** (1.0 / float(ewma_halflife))
    weights = np.power(decay, np.arange(beta_window - 1, -1, -1))
    weights = weights / np.sum(weights)

    topix_arr = topix_night.values
    gap_arr = ret_jp_gap.values

    betas_arr = np.full((n, ret_jp_gap.shape[1]), np.nan)

    for t in range(beta_window, n):
        w_gap = gap_arr[t - beta_window : t]
        w_topix = topix_arr[t - beta_window : t]

        if np.any(~np.isfinite(w_topix)) or np.any(~np.isfinite(w_gap)):
            continue

        w_mean_topix = np.sum(weights * w_topix)
        w_mean_gap = np.sum(weights[:, None] * w_gap, axis=0)

        w_var_topix = np.sum(weights * (w_topix - w_mean_topix) ** 2)
        if w_var_topix < 1e-16:
            continue

        w_cov = np.sum(
            weights[:, None] * (w_gap - w_mean_gap) * (w_topix - w_mean_topix)[:, None],
            axis=0,
        )
        betas_arr[t] = w_cov / w_var_topix

    beta_df = pd.DataFrame(betas_arr, index=ret_jp_gap.index, columns=ret_jp_gap.columns)
    beta_df = beta_df.replace([np.inf, -np.inf], np.nan)

    return beta_df


def _apply_beta_shrinkage(beta_df: pd.DataFrame, shrinkage: float) -> pd.DataFrame:
    """Apply Bayesian shrinkage of betas toward 1.0.

    Shrinks each beta estimate toward the prior mean of 1.0 (the theoretical
    expectation for sector ETFs vs the market). The shrinkage intensity
    controls the blend: 0.0 = no shrinkage, 1.0 = full shrink to 1.0.

    Args:
        beta_df: DataFrame of raw beta estimates
        shrinkage: Shrinkage intensity in [0, 1]

    Returns:
        DataFrame of shrunk beta estimates
    """
    s = float(np.clip(shrinkage, 0.0, 1.0))
    if s == 0.0:
        return beta_df
    return beta_df * (1.0 - s) + 1.0 * s


def preprocess_data(
    data: dict,
    beta_window: int = 60,
    beta_ewma_halflife: float | None = None,
    beta_shrinkage: float = 0.0,
    beta_winsor_sigma: float | None = None,
    strict_validation: bool = False,
) -> pd.DataFrame:
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
        beta_ewma_halflife: If set, use EWMA-weighted beta estimation with this
            half-life (in trading days). When None, falls back to equal-weight
            rolling cov/var (legacy behavior).
        beta_shrinkage: Bayesian shrinkage intensity toward 1.0 (0.0 = no shrink,
            1.0 = full shrink to 1.0). Applied after EWMA or rolling estimation.
        beta_winsor_sigma: If set, winsorize gap and TOPIX returns at this many
            rolling standard deviations before beta estimation (e.g. 3.0).
            When None, no winsorization is applied.
        strict_validation: If True, raise ``DataValidationError`` as soon as a
            source-key check or an execution-record check fails. If False (the
            default for backward compatibility), invalid records are skipped with
            a warning.

    Returns:
        df_exec DataFrame indexed by trade_date

    Raises:
        DataValidationError: When ``strict_validation=True`` and data quality
            invariants are violated.
    """
    raw_alerts = validate_raw_data_sources(data)
    if raw_alerts and strict_validation:
        raise DataValidationError("; ".join(raw_alerts))
    for alert in raw_alerts:
        logger.warning("preprocess_data source validation: %s", alert)

    us_c = data["us_close"].copy()
    if isinstance(us_c, pd.DataFrame):
        us_c = us_c[US_TICKERS].copy()
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

    # Close-to-close returns. pct_change can produce inf when the previous
    # close is zero; treat those as missing so the record-level validator can
    # reject the affected days.
    with np.errstate(divide="ignore", invalid="ignore"):
        ret_us_cc = us_c_joint.pct_change()
        ret_jp_cc = jp_c_joint.pct_change()
    ret_us_cc = ret_us_cc.replace([np.inf, -np.inf], np.nan)
    ret_jp_cc = ret_jp_cc.replace([np.inf, -np.inf], np.nan)

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

    # If the most recent joint date has no following JP trading day in the cache
    # (e.g., yfinance has not yet published the next day's bar at 08:15 JST),
    # project the next business day as a provisional trade_date. This keeps the
    # Step 1 panel fresh and lets compute_gap_adjusted_distribution overwrite the
    # placeholder gap values with Tachibana 9:10 prices.
    if len(joint_dates) > 0:
        last_joint = joint_dates[-1]
        if last_joint not in trade_targets:
            next_trade_date = pd.Timestamp(
                next_trading_day(last_joint.to_pydatetime())
            ).normalize()
            if next_trade_date > last_joint:
                trade_targets[last_joint] = next_trade_date

    # OC and gap returns for JP. Guard against zero open/close prices (data
    # errors / split-adjusted historical prices) so we do not propagate inf
    # into the execution frame and downstream target computations.
    with np.errstate(divide="ignore", invalid="ignore"):
        ret_jp_oc = jp_c / jp_o - 1.0
        ret_jp_gap = jp_o / jp_c.shift(1) - 1.0
    ret_jp_oc = ret_jp_oc.replace([np.inf, -np.inf], np.nan)
    ret_jp_gap = ret_jp_gap.replace([np.inf, -np.inf], np.nan)

    # TOPIX overnight and rolling betas
    topix_night = None
    beta_df = None
    if topix_close is not None and topix_open is not None:
        topix_close.index = pd.to_datetime(topix_close.index).tz_localize(None).normalize()
        topix_open.index = pd.to_datetime(topix_open.index).tz_localize(None).normalize()
        with np.errstate(divide="ignore", invalid="ignore"):
            topix_night = topix_open / topix_close.shift(1) - 1.0
        topix_night = topix_night.replace([np.inf, -np.inf], np.nan)

        gap_for_beta = ret_jp_gap
        topix_for_beta = topix_night
        if beta_winsor_sigma is not None and beta_winsor_sigma > 0:
            gap_for_beta = _winsorize_rolling(ret_jp_gap, beta_window, beta_winsor_sigma)
            topix_for_beta = _winsorize_rolling(topix_night, beta_window, beta_winsor_sigma)

        if beta_ewma_halflife is not None and beta_ewma_halflife > 0:
            beta_df = _compute_ewma_betas(
                gap_for_beta, topix_for_beta, beta_window, beta_ewma_halflife
            )
        else:
            # Shift by 1 so the beta for row t is estimated only from strictly
            # historical observations (up to t-1).
            topix_var = topix_for_beta.rolling(beta_window).var().shift(1)
            betas: dict[str, pd.Series] = {}
            for tk in JP_TICKERS:
                cov = gap_for_beta[tk].rolling(beta_window).cov(topix_for_beta).shift(1)
                betas[tk] = cov / topix_var
            beta_df = pd.DataFrame(betas)

        if beta_shrinkage > 0.0:
            beta_df = _apply_beta_shrinkage(beta_df, beta_shrinkage)

    # Build execution records
    records = []
    for sig_date in joint_dates:
        if sig_date not in trade_targets:
            continue
        trade_date = trade_targets[sig_date]

        r_us = ret_us_cc.loc[sig_date]
        r_jp = ret_jp_cc.loc[sig_date]
        jp_close_sig = jp_c_joint.loc[sig_date]

        # ret_jp_oc / ret_jp_gap / jp_open may not have the trade_date yet
        # (Japanese market has not opened or yfinance has not published the
        # daily bar). Use NaN placeholders; these are filled with 0.0 and
        # marked as provisional below. compute_gap_adjusted_distribution later
        # overwrites the gap/open values with Tachibana 9:10 prices when
        # --use-tachibana-prices is enabled.
        if trade_date in ret_jp_oc.index:
            r_oc = ret_jp_oc.loc[trade_date]
        else:
            r_oc = pd.Series(np.nan, index=JP_TICKERS)

        if trade_date in ret_jp_gap.index:
            r_gap = ret_jp_gap.loc[trade_date]
        else:
            r_gap = pd.Series(np.nan, index=JP_TICKERS)

        if trade_date in jp_o.index:
            jp_open_trade = jp_o.loc[trade_date]
        else:
            jp_open_trade = pd.Series(np.nan, index=JP_TICKERS)

        if r_us.isna().any() or r_jp.isna().any() or jp_close_sig.isna().any():
            if strict_validation:
                raise DataValidationError(
                    f"NaN in required columns for trade_date={trade_date}; "
                    f"strict_validation is enabled, so this record is rejected"
                )
            logger.warning(
                "Skipping trade_date=%s due to NaN in required columns",
                trade_date,
            )
            continue

        # r_oc (target return), r_gap, and jp_open_trade may be NaN for today
        # because the Japanese market has not opened yet or the 9:10 real-time
        # prices have not been injected. Keep the row with 0.0 placeholders and
        # mark it as provisional so consumers (backtest, PnL aggregation) can
        # exclude it. Step 1 distribution diagnostics uses r_gap only as a
        # placeholder (np.nan_to_num to 0.0), and compute_gap_adjusted_distribution
        # overwrites these values with Tachibana real-time prices at 9:10 JST.
        # Collapse any remaining infinities before filling placeholders.
        r_oc = r_oc.replace([np.inf, -np.inf], np.nan)
        r_gap = r_gap.replace([np.inf, -np.inf], np.nan)
        jp_open_trade = jp_open_trade.replace([np.inf, -np.inf], np.nan)

        is_provisional = bool(
            r_oc.isna().any()
            or r_gap.isna().any()
            or jp_open_trade.isna().any()
        )
        if is_provisional:
            r_oc = r_oc.fillna(0.0)
            r_gap = r_gap.fillna(0.0)
            jp_open_trade = jp_open_trade.fillna(0.0)

        # Only allow a zero/non-positive open for the placeholder today row,
        # which is appended beyond the raw data and will be overwritten by
        # Tachibana real-time prices.  Historical zero opens are data errors.
        is_today_placeholder = last_joint is not None and trade_date > last_joint

        record: dict = {"trade_date": trade_date, "sig_date": sig_date, "is_provisional": is_provisional}
        for tk in US_TICKERS:
            record[f"us_cc_{tk}"] = r_us[tk]
        for tk in JP_TICKERS:
            record[f"jp_cc_{tk}"] = r_jp[tk]
            record[f"jp_oc_{tk}"] = r_oc[tk]
            record[f"jp_gap_{tk}"] = r_gap[tk]
            record[f"jp_close_sig_{tk}"] = jp_close_sig[tk]
            record[f"jp_open_trade_{tk}"] = jp_open_trade[tk]

        # Historical non-finite or non-positive TOPIX open/close prices (data
        # quality) must not enter df_exec, as they break TOPIX OC returns and
        # beta estimation. Allow only the placeholder today row.
        if topix_open is not None and trade_date in topix_open.index:
            topix_o_t = topix_open.loc[trade_date]
            if (
                pd.notna(topix_o_t)
                and (not np.isfinite(float(topix_o_t)) or float(topix_o_t) <= 0.0)
                and not (is_today_placeholder and is_provisional)
            ):
                msg = f"topix_open non-positive on {trade_date}"
                if strict_validation:
                    raise DataValidationError(msg)
                logger.warning("preprocess_data record validation: %s", msg)
                continue

        if topix_close is not None and trade_date in topix_close.index:
            topix_c_t = topix_close.loc[trade_date]
            if (
                pd.notna(topix_c_t)
                and (not np.isfinite(float(topix_c_t)) or float(topix_c_t) <= 0.0)
                and not (is_today_placeholder and is_provisional)
            ):
                msg = f"topix_close non-positive on {trade_date}"
                if strict_validation:
                    raise DataValidationError(msg)
                logger.warning("preprocess_data record validation: %s", msg)
                continue

        record_alerts = validate_exec_record(record)
        if record_alerts:
            # Allow provisional placeholder today rows to pass even if they have
            # a zero open and a missing close_sig (today's close is not yet known).
            # These will be overwritten by Tachibana real-time prices (see P1-002).
            allowed_for_today = all(
                ("jp_open_trade" in a and "non-positive" in a) or "jp_close_sig" in a
                for a in record_alerts
            )
            if is_provisional and allowed_for_today and is_today_placeholder:
                logger.warning("Keeping provisional row with non-positive open: %s", record_alerts)
            else:
                if strict_validation:
                    raise DataValidationError("; ".join(record_alerts))
                for alert in record_alerts:
                    logger.warning("preprocess_data record validation: %s", alert)
                continue

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
        with np.errstate(divide="ignore", invalid="ignore"):
            r_topix_oc = topix_close / topix_open - 1.0
        r_topix_oc = r_topix_oc.replace([np.inf, -np.inf], np.nan)
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

    betas_val = betas_shifted.values.copy()

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

    return cast(np.ndarray, r_us_adj)

def build_5m_910_prices(
    df_exec: pd.DataFrame,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Build a trade-date × ticker DataFrame of 09:10 midpoint prices from 5m cache.

    The 5-minute intraday cache is keyed by bar timestamps. For each date present
    in the cache, the 09:10 bar's (High+Low)/2 is used as the 9:10 execution price
    ``p_910``.  If the 09:10 bar is not present for a date or a ticker, the cell
    remains NaN.

    Args:
        df_exec: Execution DataFrame whose index provides the trade-date grid.
        tickers: List of tickers to extract. Defaults to ``JP_TICKERS``.

    Returns:
        DataFrame indexed by ``df_exec.index``, columns = ``tickers``.
    """
    if tickers is None:
        tickers = list(JP_TICKERS)

    from leadlag.data.cache import load_intraday_cache

    p_910 = pd.DataFrame(np.nan, index=df_exec.index, columns=tickers, dtype=float)
    df_5m = load_intraday_cache("5m")
    if df_5m is None or df_5m.empty:
        return p_910

    for dt in pd.Series(df_5m.index.date).unique():
        dt_ts = pd.Timestamp(dt).normalize()
        if dt_ts not in p_910.index:
            continue
        day_data = df_5m[df_5m.index.date == dt]

        idx_910 = pd.Timestamp(f"{dt} 09:10:00")
        if idx_910 not in day_data.index:
            continue
        row_910 = day_data.loc[idx_910]

        for ticker in tickers:
            high = row_910.get(("High", ticker))
            low = row_910.get(("Low", ticker))
            close = row_910.get(("Close", ticker))
            val = (high + low) / 2 if (pd.notna(high) and pd.notna(low)) else close
            if pd.notna(val) and np.isfinite(val):
                p_910.loc[dt_ts, ticker] = float(val)

    return p_910


def _compute_jp_target_returns_h1_legacy(
    df_exec: pd.DataFrame, jp_tickers: list[str]
) -> np.ndarray:
    """Legacy h=1 9:10-to-close target computation preserved for exact backward compat.

    This path is kept for callers (``backtester.py``, ``ml_order_overlay.py``,
    and the baseline h=1 setup in ``compute_gap_adjusted_distribution``) that rely
    on the exact historical definition.
    """
    jp_oc = df_exec[[f"jp_oc_{tk}" for tk in jp_tickers]].values
    y_jp_target = jp_oc.copy()

    from leadlag.data.cache import load_intraday_cache
    df_5m = load_intraday_cache("5m")
    if df_5m is not None and not df_5m.empty:
        dates_5m = pd.Series(df_5m.index.date).unique()
        r_open_910_dict = {}
        for dt in dates_5m:
            dt_ts = pd.Timestamp(dt).normalize()
            day_data = df_5m[df_5m.index.date == dt]

            idx_910 = pd.Timestamp(f"{dt} 09:10:00")
            row_910 = day_data.loc[idx_910] if idx_910 in day_data.index else None

            ticker_returns = {}
            for ticker in jp_tickers:
                p_910 = np.nan
                if row_910 is not None:
                    high = row_910.get(("High", ticker))
                    low = row_910.get(("Low", ticker))
                    close = row_910.get(("Close", ticker))
                    if pd.notna(high) and pd.notna(low) and np.isfinite(high) and np.isfinite(low):
                        p_910 = (high + low) / 2
                    elif pd.notna(close) and np.isfinite(close):
                        p_910 = close

                p_open_5m = np.nan
                for time_str in ["09:00:00", "09:05:00", "09:10:00"]:
                    idx_time = pd.Timestamp(f"{dt} {time_str}")
                    if idx_time in day_data.index:
                        row_time = day_data.loc[idx_time]
                        op = row_time.get(("Open", ticker))
                        cl = row_time.get(("Close", ticker))
                        val = op if pd.notna(op) else cl
                        if pd.notna(val) and np.isfinite(val):
                            p_open_5m = val
                            break

                ret_open_910 = 0.0
                if (
                    pd.notna(p_910)
                    and pd.notna(p_open_5m)
                    and np.isfinite(p_910)
                    and np.isfinite(p_open_5m)
                    and p_910 > 0
                    and p_open_5m > 0
                ):
                    ret_open_910 = float(p_910 / p_open_5m - 1.0)
                ticker_returns[ticker] = ret_open_910
            r_open_910_dict[dt_ts] = ticker_returns

        for idx, date in enumerate(df_exec.index):
            date_ts = pd.Timestamp(date).normalize()
            if date_ts in r_open_910_dict:
                ticker_returns = r_open_910_dict[date_ts]
                for t_idx, ticker in enumerate(jp_tickers):
                    ret_oc = jp_oc[idx, t_idx]
                    ret_open_910 = ticker_returns.get(ticker, 0.0)
                    y_jp_target[idx, t_idx] = (1.0 + ret_oc) / (1.0 + ret_open_910) - 1.0
    return cast(np.ndarray, y_jp_target)


def _compute_jp_target_returns_h(
    df_exec: pd.DataFrame,
    jp_tickers: list[str],
    horizon: int,
    p_910_df: pd.DataFrame | None,
) -> np.ndarray:
    """Compute the h-day 9:10-to-close target return.

    For each row ``i`` (trade date) and ticker, the target is defined as:

        y_h[i, tk] = close_i / p_910_{i-h+1} - 1

    where ``close_i`` is derived from the h=1 open-to-close return and open price,
    and ``p_910_{i-h+1}`` is the 9:10 midpoint price on the starting day of the
    h-day window.  If ``p_910`` is unavailable for the start day, the open price
    on the start day is used, producing the h-day open-to-close return.

    The first ``horizon - 1`` rows are NaN because the window is not yet complete.
    """
    n = len(df_exec)
    m = len(jp_tickers)
    open_cols = [f"jp_open_trade_{tk}" for tk in jp_tickers]
    oc_cols = [f"jp_oc_{tk}" for tk in jp_tickers]

    open_arr = df_exec[open_cols].values.astype(float)
    oc_arr = df_exec[oc_cols].values.astype(float)
    close_arr = (1.0 + oc_arr) * open_arr

    p_910_arr = np.full((n, m), np.nan)
    if p_910_df is not None and not p_910_df.empty:
        aligned = p_910_df.reindex(df_exec.index)
        p_910_arr = aligned.values.astype(float)

    # start-day arrays, shifted by (horizon - 1) rows
    p_start = np.full((n, m), np.nan)
    open_start = np.full((n, m), np.nan)
    if n >= horizon:
        p_start[horizon - 1 :] = p_910_arr[: n - horizon + 1]
        open_start[horizon - 1 :] = open_arr[: n - horizon + 1]

    # Use p_910 when available and valid; otherwise fall back to open.
    p_use = np.where(
        np.isfinite(p_start) & (p_start > 0),
        p_start,
        open_start,
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        y = close_arr / p_use - 1.0

    # Guard against invalid / zero denominators, NaN/Inf close values, and
    # non-positive open prices (which make both close and the target undefined).
    valid = (
        np.isfinite(p_use)
        & (p_use > 0)
        & np.isfinite(close_arr)
        & np.isfinite(open_arr)
        & (open_arr > 0)
        & np.isfinite(y)
    )
    y = np.where(valid, y, 0.0)

    # First (horizon - 1) rows have incomplete windows.
    if horizon > 1:
        y[: horizon - 1] = np.nan

    return cast(np.ndarray, y)


def compute_jp_target_returns(
    df_exec: pd.DataFrame,
    jp_tickers: list[str],
    horizon: int = 1,
    p_910_df: pd.DataFrame | None = None,
) -> np.ndarray:
    """Compute 9:10-to-close returns for JP assets, with Open-to-Close as fallback.

    Args:
        df_exec: Execution DataFrame with ``jp_oc_*`` and ``jp_open_trade_*``.
        jp_tickers: JP tickers to compute targets for.
        horizon: Number of trading days in the target window.  Defaults to 1,
            which preserves the legacy h=1 definition for callers that do not
            pass ``p_910_df``.
        p_910_df: Optional pre-built 9:10 midpoint prices (date × ticker).  When
            ``horizon > 1`` this is used to compute the start-day 9:10 price.  When
            both ``horizon == 1`` and ``p_910_df is None``, the legacy h=1 path is
            used to guarantee backward compatibility.

    Returns:
        Array of target returns, shape (n_rows, n_tickers).  Leading ``horizon - 1``
        rows are NaN for ``horizon > 1``.
    """
    if horizon == 1 and p_910_df is None:
        return _compute_jp_target_returns_h1_legacy(df_exec, jp_tickers)
    return _compute_jp_target_returns_h(df_exec, jp_tickers, horizon, p_910_df)
