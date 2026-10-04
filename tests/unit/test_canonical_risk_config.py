"""Risk thresholds must come from AppConfig.risk throughout execution."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from leadlag.config.schemas import AppConfig, RiskConfig
from leadlag.execution import post_decision


@pytest.mark.parametrize("daily_stop, blocked", [(0.002, True), (0.02, False)])
def test_post_decision_uses_explicit_risk_thresholds(daily_stop, blocked, tmp_path, monkeypatch):
    app = AppConfig(risk=RiskConfig(daily_loss_stop=daily_stop))
    decision = {
        "trade_date": "2026-10-02", "tickers": ["1617.T", "1618.T"],
        "signal": np.array([0.01, -0.01]), "weight": np.array([0.5, -0.5]),
        "action": ["BUY", "SELL"],
    }
    returns = pd.Series(
        [0.0] * 249 + [-0.005],
        index=pd.date_range(end="2026-10-01", periods=250, freq="B"),
    )
    writes = []
    reports = []
    monkeypatch.setattr(post_decision, "_print_risk_report", reports.append)
    monkeypatch.setattr(post_decision, "_write_decision_output_and_submit", lambda **kwargs: writes.append(kwargs) or "dry.csv")
    args = dict(
        decision=decision, config=app.strategy, risk_config=app.risk,
        manual_opens={"1617.T": 1000.0, "1618.T": 1000.0},
        max_capital=100_000.0, hist_returns=returns, output_dir=tmp_path,
    )
    if blocked:
        with pytest.raises(RuntimeError, match="Risk stop threshold breached"):
            post_decision.execute_post_decision_flow(**args)
        assert any("DailyLoss=" in breach for breach in reports[0]["stop_breaches"])
        assert writes == []
    else:
        assert post_decision.execute_post_decision_flow(**args) == "dry.csv"
        assert not reports[0]["is_blocked"]
        assert len(writes) == 1
