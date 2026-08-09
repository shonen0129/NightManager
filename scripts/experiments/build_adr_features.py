#!/usr/bin/env python3
"""Build ADR close-to-close return features aligned with df_exec trade dates.

This script fetches Japanese ADR data from Yahoo Finance and constructs
per-ticker ADR return features that can be consumed by the ML order overlay.
The ADR close-to-close return for a US business day is used as a signal
for the corresponding Japanese trade day (the next JP trading day after
that US close).  No look-ahead is used.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.cache import load_df_exec_from_local_cache
from leadlag.data.tickers import JP_TICKERS
from leadlag.utils.threading import run_with_timeout

logger = logging.getLogger("BuildADRFeatures")

# Mapping from JP sector ETF to ADR tickers (liquid / NYSE where possible)
ADR_MAP: dict[str, list[str]] = {
    "1617.T": [],                  # Food
    "1618.T": [],                  # Energy resources
    "1619.T": [],                  # Construction / materials
    "1620.T": [],                  # Materials / chemicals
    "1621.T": ["TAK"],             # Pharma
    "1622.T": ["TM", "HMC"],       # Auto / transportation
    "1623.T": [],                  # Steel / non-ferrous
    "1624.T": ["KMTUY"],           # Machinery (Komatsu)
    "1625.T": ["SONY", "KYOCY"],   # Electric / precision
    "1626.T": ["SFTBY", "NTDOY"],  # IT / services
    "1627.T": [],                  # Electric power / gas
    "1628.T": [],                  # Transport / logistics
    "1629.T": ["MITSY"],           # Trading / wholesale
    "1630.T": [],                  # Retail
    "1631.T": ["MUFG", "MFG", "SMFG"],  # Banks
    "1632.T": ["NMR", "IX"],       # Finance (ex-banks)
    "1633.T": [],                  # Real estate
}

ALL_ADR_TICKERS = sorted({t for tickers in ADR_MAP.values() for t in tickers})


def _yf_download(tickers: list[str], start: str, end: str, timeout: int = 120) -> pd.DataFrame:
    """Download ADR close prices with a timeout guard."""
    tickers_str = " ".join(tickers)
    logger.info("Downloading ADR data for %d tickers from %s to %s", len(tickers), start, end)

    def _download():
        return yf.download(
            tickers=tickers_str,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=False,
            group_by="ticker",
        )

    return run_with_timeout(_download, timeout, label=f"yfinance ADR download {tickers_str}")


def build_adr_features(
    df_exec: pd.DataFrame,
    start_pad: int = 30,
    end_pad: int = 5,
) -> pd.DataFrame:
    """Construct per-ticker ADR return features aligned with df_exec index."""
    sig_dates = pd.DatetimeIndex(pd.to_datetime(df_exec["sig_date"].dropna().unique())).sort_values()
    start = (sig_dates.min() - pd.Timedelta(days=start_pad)).strftime("%Y-%m-%d")
    end = (sig_dates.max() + pd.Timedelta(days=end_pad)).strftime("%Y-%m-%d")

    if not ALL_ADR_TICKERS:
        raise ValueError("No ADR tickers configured")

    raw = _yf_download(ALL_ADR_TICKERS, start, end)

    # yf.download with multiple tickers may return MultiIndex columns or a flat DataFrame
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw.xs("Close", level=1, axis=1)
    elif "Close" in raw.columns:
        # Single ticker: yf.download returns a flat DataFrame with columns like ['Open', ..., 'Close']
        close = pd.DataFrame({ALL_ADR_TICKERS[0]: raw["Close"].astype(float)})
    else:
        # Fallback: assume raw is already a Series/DataFrame of closes
        close = raw

    close = close.astype(float)
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()

    # Drop tickers that were fully delisted / no data
    available = [t for t in ALL_ADR_TICKERS if t in close.columns and close[t].notna().any()]
    if not available:
        raise ValueError("No ADR tickers returned valid data")
    close = close[available]
    logger.info("Available ADR tickers after download: %s", available)

    # Compute close-to-close returns
    adr_rets = close.pct_change(fill_method=None)
    # Replace NaNs on days where an ADR did not trade (or delisted) with 0
    adr_rets = adr_rets.fillna(0.0).replace([np.inf, -np.inf], 0.0)

    # Build per-ticker average ADR return
    records: list[dict] = []
    for sig_date, trade_date in zip(df_exec["sig_date"], df_exec.index):
        rec: dict = {"sig_date": sig_date}
        for tk in JP_TICKERS:
            adrs = [a for a in ADR_MAP.get(tk, []) if a in available]
            if not adrs:
                rec[f"adr_{tk}"] = 0.0
            else:
                vals = adr_rets.loc[sig_date, adrs] if sig_date in adr_rets.index else pd.Series(dtype=float)
                rec[f"adr_{tk}"] = float(vals.mean()) if not vals.empty else 0.0
        records.append(rec)

    adr_df = pd.DataFrame(records, index=df_exec.index)
    adr_df.index.name = "trade_date"
    return adr_df


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    p = argparse.ArgumentParser(description="Build ADR features for ML overlay")
    p.add_argument("--output", default=str(ROOT / "data" / "adr_features.pkl"))
    p.add_argument("--output-csv", default=str(ROOT / "data" / "adr_features.csv"))
    p.add_argument("--start-pad", type=int, default=30)
    p.add_argument("--end-pad", type=int, default=5)
    args = p.parse_args()

    logger.info("Loading df_exec...")
    df_exec = load_df_exec_from_local_cache()
    logger.info("df_exec loaded: %d rows", len(df_exec))

    adr_df = build_adr_features(df_exec, start_pad=args.start_pad, end_pad=args.end_pad)

    output_pkl = Path(args.output)
    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    adr_df.to_pickle(output_pkl)
    logger.info("Saved ADR features to %s", output_pkl)

    output_csv = Path(args.output_csv)
    adr_df.to_csv(output_csv)
    logger.info("Saved ADR features CSV to %s", output_csv)

    # Basic diagnostics
    numeric_df = adr_df.select_dtypes(include=[np.number])
    non_zero = (numeric_df.abs() > 0).sum().sum()
    total = numeric_df.size
    logger.info("Non-zero ADR entries: %d / %d (%.2f%%)", non_zero, total, 100 * non_zero / total)

    return 0


if __name__ == "__main__":
    sys.exit(main())
