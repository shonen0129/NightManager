"""PIT market-cap panel construction for subsector weighting and aggregation.

Builds `cap_i,t = close_i,t * shares_i,t` per ticker, where `shares_i,t` is
approximated from the latest shares outstanding and historical split ratio
inferred from close/adj_close.

All outputs are written under `var/research/subsector/`.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger("research.subsector.market_cap")


def load_raw_ohlc(ticker: str, subsector_dir: Path) -> pd.DataFrame | None:
    """Load raw OHLC parquet for one ticker."""
    path = subsector_dir / "raw_ohlc" / f"{ticker.replace('.T', '')}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def fetch_latest_shares(ticker: str, timeout: int = 30) -> float | None:
    """Fetch latest shares outstanding from yfinance info."""
    try:
        t = yf.Ticker(ticker)
        info = t.info
        shares = info.get("sharesOutstanding") or info.get("shares")
        if shares is not None and shares > 0:
            return float(shares)
    except Exception as e:
        logger.warning("Failed to fetch shares for %s: %s", ticker, e)
    return None


def build_ticker_cap(
    ticker: str,
    subsector_dir: Path,
    shares_now: float | None = None,
) -> pd.Series | None:
    """Build PIT market-cap series for one ticker.

    Uses `close / adj_close` as a cumulative split/dividend adjustment factor.
    Since market cap should not be dividend-adjusted, we approximate:
        shares_t = shares_now * (close_t / adj_close_t)
    This is a known approximation: it ignores new-share/buyback drift and
    treats the close/adj_close ratio as a pure split factor.
    """
    df = load_raw_ohlc(ticker, subsector_dir)
    if df is None or df.empty:
        return None
    if shares_now is None:
        shares_now = fetch_latest_shares(ticker)
        if shares_now is None:
            return None

    close = df["close"].astype(float)
    adj_close = df["adj_close"].astype(float)
    # Cumulative adjustment factor.  Replace zeros/NaNs safely.
    adj_close = adj_close.replace(0.0, np.nan).ffill()
    split_factor = close / adj_close
    split_factor = split_factor.replace([np.inf, -np.inf], np.nan).ffill().fillna(1.0)
    split_factor = np.maximum(split_factor, 1e-9)

    shares_t = shares_now * split_factor
    cap = close * shares_t
    cap = cap.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    cap = cap[cap > 0]
    cap.name = ticker
    return cap


def build_market_cap_panel(
    tickers: list[str],
    subsector_dir: Path,
    output_path: Path | None = None,
    chunk_size: int = 50,
    timeout_per_ticker: int = 30,
) -> pd.DataFrame:
    """Build a date x ticker market-cap panel for a list of tickers.

    Args:
        tickers: yfinance-style tickers (e.g. '7203.T').
        subsector_dir: `var/research/subsector/` directory.
        output_path: optional path to write the parquet.
        chunk_size: number of tickers per chunk to avoid rate limiting.
        timeout_per_ticker: per-ticker yfinance timeout.

    Returns:
        DataFrame with DatetimeIndex and one column per ticker.
    """
    all_caps: dict[str, pd.Series] = {}
    failed: list[str] = []

    for i, tk in enumerate(tickers):
        if (i + 1) % chunk_size == 0:
            logger.info("Market-cap build progress: %d / %d", i + 1, len(tickers))
        cap = build_ticker_cap(tk, subsector_dir)
        if cap is not None and not cap.empty:
            all_caps[tk] = cap
        else:
            failed.append(tk)

    if not all_caps:
        raise RuntimeError("No market-cap series built.")

    panel = pd.DataFrame(all_caps)
    panel = panel.sort_index()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(output_path)
        meta = {
            "tickers": list(all_caps.keys()),
            "n_tickers": len(all_caps),
            "failed": failed,
            "n_failed": len(failed),
            "output_path": str(output_path),
        }
        with open(output_path.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.info(
        "Market-cap panel built: %d tickers, %d dates, %d failed",
        len(all_caps),
        len(panel),
        len(failed),
    )
    return panel
