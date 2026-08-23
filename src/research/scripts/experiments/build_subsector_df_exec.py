#!/usr/bin/env python3
"""Build a 79-dimensional df_exec using subsector VW returns.

This creates a df_exec with the same columns layout as the canonical 17-ETF
version but where each JP_* ticker is replaced by a subsector name.  It is
intended for the subsector BLPX experiments in Phase 2.

Outputs:
  var/research/subsector/df_exec_subsector_vw.parquet
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.fetcher import download_data
from leadlag.data.tickers import TOPIX_TICKER, US_TICKERS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _reconstruct_prices(cc: pd.DataFrame, oc: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconstruct close/open price levels from close-to-close and open-to-close returns."""
    # Forward fill cc returns then cumprod from initial close=1.0
    cc_filled = cc.copy()
    close = pd.DataFrame(np.nan, index=cc_filled.index, columns=cc_filled.columns)
    for col in close.columns:
        s = cc_filled[col].copy()
        s_filled = s.fillna(0.0)
        close[col] = (1.0 + s_filled).cumprod()
        # Reset to 1.0 at first valid cc observation for easier interpretation
        first_valid = s.first_valid_index()
        if first_valid is not None:
            close.loc[first_valid:, col] = (1.0 + s_filled.loc[first_valid:]).cumprod()
    open_ = close / (1.0 + oc)
    open_ = open_.replace([np.inf, -np.inf], np.nan)
    return close, open_


def main() -> int:
    # Load subsector panels
    oc = pd.read_parquet(ROOT / "var" / "research" / "subsector" / "panel_subsector_oc_vw.parquet")
    cc = pd.read_parquet(ROOT / "var" / "research" / "subsector" / "panel_subsector_cc_vw.parquet")
    logger.info("Loaded oc=%s cc=%s", oc.shape, cc.shape)

    # Load market data for US and TOPIX
    data = download_data(start_date="2009-01-01", end_date="2026-12-31", force=False)
    us_close = data["us_close"][US_TICKERS].copy()
    us_close.index = pd.to_datetime(us_close.index).tz_localize(None).normalize()
    us_cc = us_close.pct_change().replace([np.inf, -np.inf], np.nan)

    # TOPIX close/open for topix_night and topix_oc
    jp_close = data["jp_close"].copy()
    jp_open = data["jp_open"].copy()
    jp_close.index = pd.to_datetime(jp_close.index).tz_localize(None).normalize()
    jp_open.index = pd.to_datetime(jp_open.index).tz_localize(None).normalize()
    topix_close = jp_close[TOPIX_TICKER]
    topix_open = jp_open[TOPIX_TICKER]
    topix_night = topix_open / topix_close.shift(1) - 1.0
    topix_oc = topix_open / topix_close - 1.0

    # Reconstruct subsector open/close prices
    close_sub, open_sub = _reconstruct_prices(cc, oc)

    # Align to common dates (subsector dates, US dates, topix dates)
    common_index = oc.index.intersection(us_cc.index).intersection(topix_close.index).sort_values()
    logger.info("Common dates: %s to %s (%d)", common_index[0].date(), common_index[-1].date(), len(common_index))

    us_cc = us_cc.loc[common_index]
    oc = oc.loc[common_index]
    cc = cc.loc[common_index]
    close_sub = close_sub.loc[common_index]
    open_sub = open_sub.loc[common_index]
    topix_night = topix_night.loc[common_index]
    topix_oc = topix_oc.loc[common_index]

    # Compute jp_cc, jp_oc, jp_gap, jp_open_trade, jp_close_sig, jp_beta
    records: list[dict] = []
    subsectors = list(oc.columns)
    n_sub = len(subsectors)
    close_sub_filled = close_sub.ffill().fillna(1.0)
    open_sub_filled = open_sub.ffill().fillna(1.0)

    # JP beta: rolling cov(gap, topix_night) / var(topix_night) with shift(1)
    beta_window = 60
    gap = open_sub_filled / close_sub_filled.shift(1) - 1.0
    topix_night_s = topix_night.rename("topix")
    topix_var = topix_night_s.rolling(beta_window).var().shift(1)
    beta_df = pd.DataFrame(index=common_index, columns=subsectors, dtype=float)
    for sub in subsectors:
        cov = gap[sub].rolling(beta_window).cov(topix_night_s).shift(1)
        beta_df[sub] = cov / topix_var
    beta_df = beta_df.replace([np.inf, -np.inf], np.nan)

    for dt in common_index:
        record: dict = {"trade_date": dt, "sig_date": dt}
        for tk in US_TICKERS:
            record[f"us_cc_{tk}"] = float(us_cc.loc[dt, tk])
        for sub in subsectors:
            record[f"jp_cc_{sub}"] = float(cc.loc[dt, sub])
            record[f"jp_oc_{sub}"] = float(oc.loc[dt, sub])
            record[f"jp_gap_{sub}"] = float(gap.loc[dt, sub])
            record[f"jp_open_trade_{sub}"] = float(open_sub_filled.loc[dt, sub])
            record[f"jp_close_sig_{sub}"] = float(close_sub_filled.loc[dt, sub])
            record[f"jp_beta_{sub}"] = float(beta_df.loc[dt, sub])
        record["topix_night_return"] = float(topix_night.loc[dt])
        record["topix_oc_return"] = float(topix_oc.loc[dt])
        records.append(record)

    df_exec = pd.DataFrame(records).set_index("trade_date")
    df_exec.index.name = "trade_date"

    out_path = ROOT / "var" / "research" / "subsector" / "df_exec_subsector_vw.parquet"
    df_exec.to_parquet(out_path)
    logger.info("Saved df_exec_subsector_vw: %s", df_exec.shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
