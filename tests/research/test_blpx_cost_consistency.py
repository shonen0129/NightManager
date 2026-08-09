"""Cost consistency check for Residual-BLPX using the legacy V1 backtester."""

from __future__ import annotations

import numpy as np

from research.backtest_v1 import run_v1_backtest
from research.models.sector_relative_ensemble_blp_enhanced import (
    SectorRelativeEnsembleBLPEnhancedModel,
)


def test_cost_consistency(residual_blpx_prod_config, sample_df_exec):
    """Check that cost function subtraction is algebraically consistent."""
    df_exec, _ = sample_df_exec
    model = SectorRelativeEnsembleBLPEnhancedModel(residual_blpx_prod_config)
    start_str = df_exec.index[-20].strftime("%Y-%m-%d")

    results = run_v1_backtest(model, df_exec, start_date=start_str)
    r_gross = results["daily_returns_gross"]
    r_net = results["daily_returns"]
    costs = results["daily_costs"]

    assert np.allclose(r_gross - costs, r_net, atol=1e-15)
