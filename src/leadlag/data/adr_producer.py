"""Operational ADR return generation; missing observations remain missing."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from leadlag.core.timeouts import YFINANCE_DOWNLOAD, YFINANCE_TICKER_HISTORY
from leadlag.data.adr_features import DEFAULT_ADR_FEATURES_PATH, publish_adr_features
from leadlag.data.tickers import ADR_SECTOR_MAP, ADR_TICKERS, JP_TICKERS
from leadlag.utils.threading import run_with_timeout
from leadlag.utils.timestamps import normalize_jst_index

logger = logging.getLogger(__name__)


def build_adr_features(df_exec: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    """Align actual US close returns to the supplied US/JP business-day mapping.

    An unmapped JP sector is a structural zero. A mapped sector requires at
    least one observed ADR return; missing prices are never forward-filled.
    Coverage counts are stored alongside each feature for publication.
    """
    if df_exec.empty or "sig_date" not in df_exec:
        raise ValueError("ADR generation requires nonempty trade/signal dates")
    trade_dates = normalize_jst_index(df_exec.index)
    signal_dates = normalize_jst_index(df_exec["sig_date"])
    if (
        trade_dates.hasnans
        or signal_dates.hasnans
        or trade_dates.has_duplicates
        or not trade_dates.is_monotonic_increasing
        or (signal_dates >= trade_dates).any()
    ):
        raise ValueError("ADR signal dates must precede unique ordered JP trade dates")
    prices = close.reindex(columns=ADR_TICKERS).astype(float).copy()
    prices.index = pd.DatetimeIndex(pd.to_datetime(prices.index)).tz_localize(None).normalize()
    if (
        prices.index.hasnans
        or prices.index.has_duplicates
        or not prices.index.is_monotonic_increasing
    ):
        raise ValueError("ADR close dates must be unique and ordered")
    prices = prices.where(np.isfinite(prices) & (prices > 0))
    returns = prices.pct_change(fill_method=None).reindex(signal_dates)
    returns.index = trade_dates
    frame = pd.DataFrame({"sig_date": signal_dates}, index=trade_dates)
    for ticker in JP_TICKERS:
        adrs = ADR_SECTOR_MAP[ticker]
        if not adrs:
            frame[f"adr_{ticker}"] = 0.0
            frame[f"coverage_{ticker}"] = 0
        else:
            observations = returns[adrs].where(np.isfinite(returns[adrs]))
            frame[f"adr_{ticker}"] = observations.mean(axis=1)
            frame[f"coverage_{ticker}"] = observations.notna().sum(axis=1)
    frame.index.name = "trade_date"
    return frame


def refresh_adr_features(
    df_exec: pd.DataFrame,
    *,
    required_trade_date: pd.Timestamp | str,
    path: Path | str = DEFAULT_ADR_FEATURES_PATH,
) -> dict:
    """Fetch and publish a complete required row, or retain the previous bundle.

    The scheduled caller supplies a whole-process deadline in addition to the
    network/wait bounds here. No late download thread performs publication.
    """
    dates = normalize_jst_index(df_exec["sig_date"])
    if dates.empty or dates.hasnans:
        raise ValueError("ADR download requires valid signal dates")
    raw = run_with_timeout(
        lambda: yf.download(
            tickers=" ".join(ADR_TICKERS),
            start=(dates.min() - pd.Timedelta(days=30)).strftime("%Y-%m-%d"),
            end=(dates.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
            threads=False,
            group_by="ticker",
            timeout=YFINANCE_TICKER_HISTORY,
        ),
        YFINANCE_DOWNLOAD,
        label="ADR close download",
    )
    if not isinstance(raw.columns, pd.MultiIndex):
        raise ValueError("ADR download did not return ticker/price columns")
    close = raw.xs("Close", level=1, axis=1)
    frame = build_adr_features(df_exec, close)
    manifest = publish_adr_features(frame, path, required_trade_date=required_trade_date)
    logger.info(
        "ADR published: latest=%s coverage=%s",
        manifest["latest_trade_date"],
        manifest["latest_coverage"],
    )
    return manifest


def main() -> int:
    """ADR-only recovery entry, run under job_guard/phase_deadline by operators."""
    import argparse

    from leadlag.data.market_data_cache import load_df_exec_from_local_cache

    parser = argparse.ArgumentParser(description="Publish observed ADR features for an exact trade date")
    parser.add_argument("--trade-date", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_ADR_FEATURES_PATH)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    try:
        refresh_adr_features(load_df_exec_from_local_cache(max_stale_bdays=1),
                             required_trade_date=args.trade_date, path=args.output)
    except Exception as exc:
        logger.error("ADR refresh failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
