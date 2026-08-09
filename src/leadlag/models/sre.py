"""JP target return computation utilities.

The ``SectorRelativeEnsembleModel`` class has been moved to
``archive/legacy_src/models/sre.py`` as part of the V1 model cleanup.
This module now only retains ``compute_jp_target_returns`` which is still
used by the V2 backtest pipeline.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_jp_target_returns(df_exec: pd.DataFrame, jp_tickers: list[str]) -> np.ndarray:
    """Compute 9:10-to-close returns for JP assets, with Open-to-Close as fallback."""
    jp_oc = df_exec[[f"jp_oc_{tk}" for tk in jp_tickers]].values
    y_jp_target = jp_oc.copy()

    from leadlag.data.cache import load_intraday_cache
    df_5m = load_intraday_cache("5m")
    if df_5m is not None and not df_5m.empty:
        dates_5m = pd.Series(df_5m.index.date).unique()
        r_open_910_dict = {}
        for dt in dates_5m:
            dt_ts = pd.Timestamp(dt)
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
                    p_910 = (high + low) / 2 if (pd.notna(high) and pd.notna(low)) else close

                p_open_5m = np.nan
                for time_str in ["09:00:00", "09:05:00", "09:10:00"]:
                    idx_time = pd.Timestamp(f"{dt} {time_str}")
                    if idx_time in day_data.index:
                        row_time = day_data.loc[idx_time]
                        op = row_time.get(("Open", ticker))
                        cl = row_time.get(("Close", ticker))
                        val = op if pd.notna(op) else cl
                        if pd.notna(val):
                            p_open_5m = val
                            break

                ret_open_910 = 0.0
                if pd.notna(p_910) and pd.notna(p_open_5m) and p_open_5m > 0:
                    ret_open_910 = float(p_910 / p_open_5m - 1.0)
                ticker_returns[ticker] = ret_open_910
            r_open_910_dict[dt_ts] = ticker_returns

        for idx, date in enumerate(df_exec.index):
            date_ts = pd.Timestamp(date)
            if date_ts in r_open_910_dict:
                ticker_returns = r_open_910_dict[date_ts]
                for t_idx, ticker in enumerate(jp_tickers):
                    ret_oc = jp_oc[idx, t_idx]
                    ret_open_910 = ticker_returns.get(ticker, 0.0)
                    y_jp_target[idx, t_idx] = (1.0 + ret_oc) / (1.0 + ret_open_910) - 1.0
    return y_jp_target
