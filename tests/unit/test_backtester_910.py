import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from leadlag.execution.backtester import BacktestEngine
from leadlag.models.base import BaseModel
from leadlag.data.tickers import JP_TICKERS


class DummyModel(BaseModel):
    def __init__(self):
        self.n_j = len(JP_TICKERS)
        self.corr_window = 0
        self.slippage_bps = 5.0

    def predict_signals(self, df_exec):
        T = len(df_exec)
        sigs = pd.DataFrame(np.zeros((T, self.n_j)), index=df_exec.index, columns=JP_TICKERS)
        y_jp_oc_df = df_exec[[f"jp_oc_{tk}" for tk in JP_TICKERS]].rename(
            columns=lambda c: c.replace("jp_oc_", "")
        )
        return {
            "signals": sigs,
            "raw_pca_signals": sigs,
            "residual_pca_signals": sigs,
            "p4_signals": sigs,
            "normalized_signals": sigs,
            "y_jp_oc_df": y_jp_oc_df,
        }

    def build_weights(self, signal, q=None, Sigma_YY=None):
        w = np.zeros(self.n_j)
        w[0] = 1.0
        w[-1] = -1.0
        return w


def test_backtester_910_adjustment():
    # Setup mock df_exec
    dates = pd.date_range("2026-03-03", periods=3)
    records = []
    for dt in dates:
        rec = {"trade_date": dt, "sig_date": dt - pd.Timedelta(days=1)}
        for tk in JP_TICKERS:
            rec[f"jp_oc_{tk}"] = 0.02
        records.append(rec)
    df_exec = pd.DataFrame(records).set_index("trade_date")

    # Setup mock 5m data
    cols = pd.MultiIndex.from_product(
        [["High", "Low", "Close", "Open"], JP_TICKERS], names=["Price", "Ticker"]
    )
    times = []
    for dt in dates:
        for t_str in ["09:00:00", "09:05:00", "09:10:00"]:
            times.append(pd.Timestamp(f"{dt.date()} {t_str}"))
    df_5m = pd.DataFrame(np.nan, index=times, columns=cols)

    # Fill in prices for 2026-03-03 (first date)
    dt1 = dates[0].date()
    df_5m.loc[pd.Timestamp(f"{dt1} 09:00:00"), ("Open", JP_TICKERS[0])] = 100.0
    df_5m.loc[pd.Timestamp(f"{dt1} 09:10:00"), ("High", JP_TICKERS[0])] = 101.0
    df_5m.loc[pd.Timestamp(f"{dt1} 09:10:00"), ("Low", JP_TICKERS[0])] = 101.0

    df_5m.loc[pd.Timestamp(f"{dt1} 09:00:00"), ("Open", JP_TICKERS[-1])] = 100.0
    df_5m.loc[pd.Timestamp(f"{dt1} 09:10:00"), ("High", JP_TICKERS[-1])] = 99.0
    df_5m.loc[pd.Timestamp(f"{dt1} 09:10:00"), ("Low", JP_TICKERS[-1])] = 99.0

    # Run backtest with patch
    model = DummyModel()
    with patch("leadlag.data.cache.load_intraday_cache", return_value=df_5m):
        results = BacktestEngine.run_backtest(model, df_exec, start_date="2026-03-03")

    # Verify results
    # Asset 0: ret_oc = 2%, ret_open_910 = 1% -> ret_910_close = (1.02)/(1.01) - 1 = 0.990099%
    # Asset -1: ret_oc = 2%, ret_open_910 = -1% -> ret_910_close = (1.02)/(0.99) - 1 = 3.030303%
    # w[0] = 1.0, w[-1] = -1.0
    expected_gross = 1.0 * ((1.02 / 1.01) - 1.0) - 1.0 * ((1.02 / 0.99) - 1.0)
    assert np.isclose(results["daily_returns_gross"].iloc[0], expected_gross)
    assert np.isclose(results["daily_returns_gross_oc"].iloc[0], 0.0)

    # For other dates, missing 5m data makes it fall back to Open-to-Close (gross=0)
    assert np.isclose(results["daily_returns_gross"].iloc[1], 0.0)
    assert np.isclose(results["daily_returns_gross_oc"].iloc[1], 0.0)
