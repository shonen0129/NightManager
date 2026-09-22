"""Research-side input assembly for gap-adjusted distribution diagnostics.

The calculation itself lives in :mod:`leadlag.pipeline.gap_distribution`.  This
module keeps model-specific input preparation (including horizon target
overrides) at the research boundary so the production pipeline does not need
to know about the diagnostic command's orchestration.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from leadlag.broker.tachibana.session_cache import save_current_prices_cache
from leadlag.data.decision_cache import is_decision_cache_valid, load_decision_cache
from leadlag.data.fetcher import download_data
from leadlag.data.preprocessor import preprocess_data
from leadlag.data.tickers import TOPIX_TICKER, US_TICKERS
from leadlag.domain.inputs import HistoricalInputs
from leadlag.utils.timestamps import normalize_jst_date

logger = logging.getLogger(__name__)


def inject_tachibana_realtime_prices(
    df_exec: pd.DataFrame,
    raw_data: dict[str, Any],
    today: pd.Timestamp,
    api_client: Any | None = None,
) -> tuple[pd.DataFrame, Any]:
    """Inject or override today's row in df_exec using Tachibana API real-time prices.

    Fetches current prices (pDPP) for all JP tickers + TOPIX at ~9:10 JST,
    computes gap returns against previous JP close, and either appends a new
    row or overrides the gap values if today's row already exists (e.g. from
    yfinance 9:00 open).

    The correct sig_date (most recent US trading day before today) and US
    close-to-close returns are computed from raw_data, not carried over from
    the previous row.

    Args:
        df_exec: Execution DataFrame from preprocess_data().
        raw_data: Raw data dict with jp_close, jp_open, us_close.
        today: Today's date (tz-naive, normalized).
        api_client: Optional pre-built broker client. If provided, it is not closed.

    Returns:
        (df_exec with today's row added or updated, api_client used)
    """
    today = pd.Timestamp(today).tz_localize(None).normalize()

    logger.info("Injecting Tachibana real-time prices for %s...", today.date())

    from leadlag.data.tickers import JP_TICKERS, TOPIX_TICKER
    from leadlag.execution.broker_ops import build_api_client

    # --- 1. Fetch current prices from Tachibana API ---
    own_client = api_client is None
    if own_client:
        api_client = build_api_client(api_url=None, api_token=None, api_dry_run=False)
    if api_client is None:
        raise RuntimeError("build_api_client returned None")

    tickers_to_fetch = JP_TICKERS + [TOPIX_TICKER]
    current_prices = api_client.fetch_current_prices(tickers_to_fetch, allow_missing=True)

    if not current_prices:
        logger.error("Failed to fetch any prices from Tachibana API.")
        if own_client:
            try:
                api_client.close()
            except Exception:
                pass
        return df_exec, api_client

    logger.info(
        "Fetched %d/%d prices from Tachibana API.", len(current_prices), len(tickers_to_fetch)
    )
    save_current_prices_cache(
        {tk: price for tk, price in current_prices.items() if tk != TOPIX_TICKER},
        current_prices.get(TOPIX_TICKER),
        today.strftime("%Y%m%d"),
    )

    # --- 2. Determine sig_date: most recent US trading day before today ---
    us_close = raw_data["us_close"].copy()
    us_close.index = pd.to_datetime(us_close.index, format="ISO8601").tz_localize(None).normalize()
    if isinstance(us_close, pd.DataFrame):
        us_close = us_close[US_TICKERS]

    us_dates_before_today = us_close.index[us_close.index < today]
    if len(us_dates_before_today) == 0:
        logger.error("No US trading data before %s.", today.date())
        if own_client:
            try:
                api_client.close()
            except Exception:
                pass
        return df_exec, api_client
    sig_date = us_dates_before_today[-1]
    logger.info("sig_date=%s (most recent US trading day before %s)", sig_date.date(), today.date())

    # Compute US close-to-close returns for sig_date
    sig_idx = us_close.index.get_loc(sig_date)
    if sig_idx > 0:
        us_returns = us_close.iloc[sig_idx] / us_close.iloc[sig_idx - 1] - 1.0
    else:
        us_returns = pd.Series(0.0, index=us_close.columns)

    # --- 3. Determine previous JP close date ---
    jp_close = raw_data["jp_close"].copy()
    jp_close.index = pd.to_datetime(jp_close.index, format="ISO8601").tz_localize(None).normalize()

    prev_dates = jp_close.index[jp_close.index < today]
    if len(prev_dates) == 0:
        logger.error("No historical JP close data before %s.", today.date())
        if own_client:
            try:
                api_client.close()
            except Exception:
                pass
        return df_exec, api_client
    prev_date = prev_dates[-1]

    # JP close on sig_date (for jp_close_sig): use the closest JP trading day
    # on or before sig_date
    jp_dates_on_sig = jp_close.index[jp_close.index <= sig_date]
    jp_close_sig_date = jp_dates_on_sig[-1] if len(jp_dates_on_sig) > 0 else prev_date

    # --- 4. Get beta from last row (carried over) ---
    last_row = df_exec.iloc[-1]

    # --- 5. Build the record ---
    record: dict[str, Any] = {
        "trade_date": today,
        "sig_date": sig_date,
        "is_provisional": True,
    }

    for tk in JP_TICKERS:
        prev_close = float(jp_close.loc[prev_date, tk]) if tk in jp_close.columns else np.nan
        curr_price = current_prices.get(tk, np.nan)

        if pd.notna(prev_close) and prev_close > 0 and pd.notna(curr_price) and curr_price > 0:
            gap_ret = curr_price / prev_close - 1.0
        else:
            gap_ret = 0.0

        record[f"jp_gap_{tk}"] = gap_ret
        record[f"jp_open_trade_{tk}"] = curr_price if pd.notna(curr_price) else 0.0

        # jp_close_sig: JP close on sig_date (or nearest JP trading day)
        sig_close = float(jp_close.loc[jp_close_sig_date, tk]) if tk in jp_close.columns else np.nan
        record[f"jp_close_sig_{tk}"] = sig_close if pd.notna(sig_close) else 0.0

        record[f"jp_cc_{tk}"] = 0.0  # not available yet
        record[f"jp_oc_{tk}"] = 0.0  # not available yet (close unknown)

        # Carry over beta from last row
        beta_col = f"jp_beta_{tk}"
        if beta_col in df_exec.columns:
            record[beta_col] = last_row[beta_col]

    # US returns for sig_date
    for tk in US_TICKERS:
        col = f"us_cc_{tk}"
        if col in df_exec.columns:
            record[col] = float(us_returns.get(tk, 0.0))

    # TOPIX overnight return
    topix_prev_close = (
        float(jp_close.loc[prev_date, TOPIX_TICKER]) if TOPIX_TICKER in jp_close.columns else np.nan
    )
    topix_curr = current_prices.get(TOPIX_TICKER, np.nan)
    if (
        pd.notna(topix_prev_close)
        and topix_prev_close > 0
        and pd.notna(topix_curr)
        and topix_curr > 0
    ):
        topix_night = topix_curr / topix_prev_close - 1.0
    else:
        topix_night = 0.0

    record["topix_night_return"] = topix_night
    record["topix_oc_return"] = 0.0  # not available yet
    record["topix_cc_trade"] = topix_night  # approximate

    # --- 6. Add or override today's row ---
    new_row = pd.DataFrame([record])
    new_row = new_row.set_index("trade_date")
    new_row.index = pd.to_datetime(new_row.index, format="ISO8601").tz_localize(None).normalize()

    if today in df_exec.index:
        logger.info(
            "Today (%s) already in df_exec, overriding gap values with Tachibana prices.",
            today.date(),
        )
        df_exec = df_exec.drop(index=today)
        df_exec = pd.concat([df_exec, new_row])
    else:
        df_exec = pd.concat([df_exec, new_row])

    df_exec = df_exec.sort_index()

    logger.info(
        "Injected row for %s: sig_date=%s, topix_night=%.4f, %d gap returns computed.",
        today.date(),
        sig_date.date(),
        topix_night,
        sum(1 for tk in JP_TICKERS if record.get(f"jp_gap_{tk}", 0.0) != 0.0),
    )

    # Close API client only if we created it
    if own_client:
        try:
            api_client.close()
        except Exception:
            pass

    return df_exec, api_client


@dataclass(frozen=True)
class GapModelInputs:
    """Named view of the common model inputs consumed by the gap calculation."""

    y_jp_target: Any
    jp_gap: Any
    jp_beta: Any
    topix_night: Any
    jp_res_returns_p3: Any
    c_full_p3: Any
    v0_static: Any

    def __getitem__(self, key: str) -> Any:
        """Provide mapping-style compatibility for the existing worker code."""
        return getattr(self, key)

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> GapModelInputs:
        """Build the stable contract from ``_prepare_common_inputs`` output."""
        return cls(
            y_jp_target=values["y_jp_target"],
            jp_gap=values["jp_gap"],
            jp_beta=values["jp_beta"],
            topix_night=values["topix_night"],
            jp_res_returns_p3=values["jp_res_returns_p3"],
            c_full_p3=values["c_full_p3"],
            v0_static=values["v0_static"],
        )


@dataclass(frozen=True)
class GapExecutionInputs:
    """Raw and preprocessed market inputs owned by the research entrypoint."""

    raw_data: dict[str, Any]
    df_exec: pd.DataFrame


GAP_LABEL_AVAILABILITY: dict[str, str] = {
    # These are session-relative rules because one absolute timestamp cannot
    # describe every row in a full historical frame.
    "jp_oc_*": "trade_date+15:30",
    "jp_cc_*": "trade_date+15:30",
    "topix_oc_return": "trade_date+15:30",
    "topix_cc_trade": "trade_date+15:30",
    "target_*": "trade_date+15:30",
    "y_jp_*": "trade_date+15:30",
}


def build_gap_historical_inputs(
    df_exec: pd.DataFrame,
    *,
    open_910_returns: pd.DataFrame | None = None,
    horizon: int = 1,
    source: str = "gap_generation",
    observed_at: Mapping[str, str] | None = None,
    observed_at_by_date: Mapping[str, Mapping[str, str]] | None = None,
) -> HistoricalInputs:
    """Create the run-owned PIT history used by every gap horizon.

    The realized target remains outside this contract.  It is supplied to
    reporting as an :class:`EvaluationInputs`-equivalent array after the
    prediction is calculated.  The calculation view therefore cannot expose a
    same-session close label to a 09:10 signal.
    """
    if observed_at_by_date is None:
        observed_at_by_date = {
            normalize_jst_date(value).strftime("%Y-%m-%d"): {
                "us_returns": f"{normalize_jst_date(value).date()} 09:00",
                "jp_gap_returns": f"{normalize_jst_date(value).date()} 09:10",
                "jp_betas": f"{normalize_jst_date(value).date()} 09:10",
                "topix_night_return": f"{normalize_jst_date(value).date()} 09:10",
                "open_910_returns": f"{normalize_jst_date(value).date()} 09:10",
            }
            for value in df_exec.index
        }
    return HistoricalInputs(
        df_exec,
        horizon=horizon,
        label_available_at=GAP_LABEL_AVAILABILITY,
        observed_at=observed_at,
        observed_at_by_date=observed_at_by_date,
        open_910_returns=open_910_returns,
        source=source,
    )


def mask_future_jp_labels(
    returns: np.ndarray,
    index: pd.Index,
    as_of: pd.Timestamp | str,
    *,
    n_u: int,
    label_close_time: str = "15:30",
) -> np.ndarray:
    """Mask JP target rows whose session close was not known at ``as_of``.

    BLPX already excludes the current row from its rolling window.  Masking
    here makes that invariant explicit for the full matrix and protects future
    callers that inspect more than the rolling slice.
    """
    frame = np.asarray(returns, dtype=float).copy()
    dates = pd.DatetimeIndex(pd.to_datetime(index, format="ISO8601"))
    if dates.tz is not None:
        dates = dates.tz_convert("Asia/Tokyo").tz_localize(None)
    dates = dates.normalize()
    cutoff = pd.Timestamp(as_of)
    if cutoff.tzinfo is not None:
        cutoff = cutoff.tz_convert("Asia/Tokyo").tz_localize(None)
    hours, minutes = (int(part) for part in label_close_time.split(":", 1))
    available = dates + pd.Timedelta(hours=hours, minutes=minutes) <= cutoff
    frame[~available, n_u:] = np.nan
    return frame


def load_gap_execution_inputs(*, beta_window: int = 60) -> GapExecutionInputs:
    """Load raw market data and reuse the validated Step 1 decision cache."""
    raw_data = download_data(beta_window=beta_window)
    if is_decision_cache_valid():
        df_exec = load_decision_cache()
    else:
        df_exec = preprocess_data(raw_data, beta_window=beta_window)
    return GapExecutionInputs(raw_data=raw_data, df_exec=df_exec)


def attach_topix_trade_returns(
    df_exec: pd.DataFrame,
    raw_data: dict[str, Any],
    *,
    preserve_today_placeholder: bool = False,
) -> pd.DataFrame:
    """Attach TOPIX open-to-close and overnight-to-close trade returns."""
    frame = df_exec.copy()
    topix_close = raw_data["jp_close"][TOPIX_TICKER].copy()
    topix_open = raw_data["jp_open"][TOPIX_TICKER].copy()
    topix_close.index = (
        pd.to_datetime(topix_close.index, format="ISO8601").tz_localize(None).normalize()
    )
    topix_open.index = (
        pd.to_datetime(topix_open.index, format="ISO8601").tz_localize(None).normalize()
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        r_topix_oc = topix_close / topix_open - 1.0
    r_topix_oc = r_topix_oc.replace([np.inf, -np.inf], np.nan)
    frame["topix_oc_return"] = r_topix_oc.reindex(frame.index).values
    if preserve_today_placeholder:
        today = pd.Timestamp.now().tz_localize(None).normalize()
        if today in frame.index and pd.isna(frame.loc[today, "topix_oc_return"]):
            frame.loc[today, "topix_oc_return"] = 0.0
    frame["topix_cc_trade"] = (1.0 + frame["topix_night_return"]) * (
        1.0 + frame["topix_oc_return"]
    ) - 1.0
    return frame


def prepare_gap_model_inputs(
    model: Any,
    df_exec: pd.DataFrame,
    *,
    y_jp_target: Any | None = None,
    horizon: int = 1,
    p_910_df: pd.DataFrame | None = None,
    open_910_returns: pd.DataFrame | None = None,
    historical_inputs: HistoricalInputs | None = None,
    as_of: pd.Timestamp | str | None = None,
) -> GapModelInputs:
    """Prepare the common inputs required by one gap diagnostic horizon.

    ``y_jp_target`` is passed only for multi-horizon runs, where the target is
    computed from the original one-day execution frame.  Omitting it retains
    the model's existing target computation and cache behavior.
    """
    if historical_inputs is not None:
        if not isinstance(historical_inputs, HistoricalInputs):
            raise TypeError("historical_inputs must be a HistoricalInputs instance")
        # Keep the contract on the research boundary even though the common
        # input builder still receives the complete frame for vectorization.
        # The per-date worker masks the target rows before the model consumes
        # them; this call rejects a missing trade date or an invalid cutoff.
        historical_inputs.calculation_frame(as_of) if as_of is not None else historical_inputs.calculation_frame()
    kwargs: dict[str, Any] = {}
    if horizon != 1:
        kwargs["horizon"] = horizon
    if p_910_df is not None:
        kwargs["p_910_df"] = p_910_df
    if open_910_returns is not None:
        kwargs["open_910_returns"] = open_910_returns
    if y_jp_target is not None:
        kwargs["y_jp_target"] = y_jp_target
    values = model._prepare_common_inputs(df_exec, **kwargs)
    return GapModelInputs.from_mapping(values)


__all__ = [
    "GAP_LABEL_AVAILABILITY",
    "GapExecutionInputs",
    "GapModelInputs",
    "attach_topix_trade_returns",
    "build_gap_historical_inputs",
    "inject_tachibana_realtime_prices",
    "load_gap_execution_inputs",
    "mask_future_jp_labels",
    "prepare_gap_model_inputs",
]
