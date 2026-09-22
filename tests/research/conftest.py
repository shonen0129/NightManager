"""Deterministic I/O fixtures for the maintained legacy diagnostic tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sprint_market_inputs(sample_df_exec, monkeypatch, tmp_path):
    """Keep the actual diagnostic math, without a live cache or network."""
    from leadlag.data import intraday_inputs, macro
    from leadlag.data.tickers import JP_TICKERS
    from research.diagnostics import sprint0, sprint0_qa, sprint1_experiments

    frame, raw = sample_df_exec
    close = raw["jp_close"].loc[:, JP_TICKERS]
    daily = pd.concat({"Close": close, "Volume": close * 0 + 100_000.0}, axis=1)
    # Cover both actual 09:10 bars and the documented missing-bar proxy.
    opens = raw["jp_open"].loc["2026-05-01":, JP_TICKERS].iloc[::2]
    bars = pd.concat({"Open": opens, "Close": opens * 1.001,
                      "High": opens * 1.002, "Low": opens}, axis=1)
    bars.index = bars.index + pd.Timedelta(hours=9, minutes=10)
    diagnostics = tmp_path / "portfolio_gap_distribution_diagnostics.csv"
    pd.DataFrame({"trade_date": frame.index, "gross_exposure": 2.0,
                  "pred_ir_gap_exante_cost": np.random.default_rng(81).uniform(0.01, 0.2, len(frame)),
                  "pit_bin": "Medium"}).to_csv(diagnostics, index=False)

    def prices(*, tickers, start, end, **_kwargs):
        dates = daily.index[(daily.index >= pd.Timestamp(start)) & (daily.index <= pd.Timestamp(end))]
        return daily.loc[dates, pd.IndexSlice[:, tickers]].copy()

    for module in (sprint0, sprint0_qa):
        monkeypatch.setattr(module, "load_df_exec_from_local_cache", lambda: frame.copy())
        monkeypatch.setattr(module, "_yf_download_with_timeout", prices)
    for module in (sprint0, sprint0_qa, sprint1_experiments, intraday_inputs):
        monkeypatch.setattr(module, "load_intraday_cache", lambda *_a, **_kw: bars.copy())
    for module in (sprint0, sprint0_qa, sprint1_experiments):
        monkeypatch.setattr(module, "find_latest_distribution_diagnostics", lambda: str(diagnostics))
    macro_prices = pd.DataFrame({"USDJPY=X": 140 + np.arange(len(frame)) * 0.001,
                                 "CL=F": 70 + np.sin(np.arange(len(frame)) / 20),
                                 "^TNX": 4 + np.cos(np.arange(len(frame)) / 30) * 0.1}, index=frame.index)
    monkeypatch.setattr(macro, "load_macro_prices", lambda **_kw: macro_prices.copy())
    return frame.copy()
