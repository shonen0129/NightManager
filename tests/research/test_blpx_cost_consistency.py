"""Cost consistency check for Residual-BLPX using the legacy V1 backtester."""

from __future__ import annotations

import numpy as np
import pandas as pd

from leadlag.data.tickers import JP_TICKERS
from research.backtest_v1 import run_v1_backtest
from research.models.sector_relative_ensemble_blp_enhanced import (
    SectorRelativeEnsembleBLPEnhancedModel,
)


def test_cost_consistency(residual_blpx_prod_config, sample_df_exec, monkeypatch):
    """Check costs using an explicit daily-open fallback, never a terminal cache."""
    df_exec, _ = sample_df_exec
    model = SectorRelativeEnsembleBLPEnhancedModel(residual_blpx_prod_config)
    start_str = df_exec.index[-20].strftime("%Y-%m-%d")
    open_910_returns = pd.DataFrame(np.nan, index=df_exec.index, columns=JP_TICKERS)

    def fail_on_implicit_intraday_io(*_args, **_kwargs):
        raise AssertionError("cost regression must not read the intraday cache")

    monkeypatch.setattr(
        "leadlag.execution.backtester.build_open_910_returns",
        fail_on_implicit_intraday_io,
    )
    results = run_v1_backtest(
        model,
        df_exec,
        start_date=start_str,
        open_910_returns=open_910_returns,
    )
    r_gross = results["daily_returns_gross"]
    r_net = results["daily_returns"]
    costs = results["daily_costs"]

    assert np.allclose(r_gross - costs, r_net, atol=1e-15)
