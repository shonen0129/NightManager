from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from leadlag.config.schemas import AppConfig
from leadlag.data.tickers import JP_TICKERS
from leadlag.domain.inputs import HistoricalInputs
from leadlag.domain.portfolio import PortfolioDecision
from leadlag.execution import backtester


@pytest.mark.parametrize("invalid", [False, True])
def test_research_transform_changes_actual_backtest_weights_and_is_audited(monkeypatch, invalid):
    cfg = AppConfig()
    original = PortfolioDecision(
        w_final=np.array([0.5, -0.5] + [0.0] * 15), scores=np.arange(17, dtype=float),
        mu_gap=np.zeros(17), sigma_gap=np.ones(17), Omega_gap=np.eye(17),
        fallback={"gap_data_missing": False, "audit_failure": False}, pit_binning={"multiplier": 1.0},
        leakage={"status": "PASSED"}, numerical={"status": "PASSED"}, alerts=[], summary={}, run_config=cfg.v2,
    )
    monkeypatch.setattr(backtester, "build_v2_model_bundle", lambda *a, **kw: SimpleNamespace(
        decision_model=SimpleNamespace(decide=lambda **_: original), overlay_enabled=False,
    ))
    dates = pd.DatetimeIndex(["2026-09-16"])
    frame = pd.DataFrame({"sig_date": ["2026-09-15"]}, index=dates)
    calls = []

    def transform(date, decision):
        calls.append(date)
        weights = decision.w_final * 0.5
        if invalid:
            weights[0] = 5.0
        return replace(decision, w_final=weights)

    weights, fallback, _ = backtester.BacktestEngine._generate_v2_weights(
        frame, cfg, None, dates, 17, None, None, 1, transform,
    )
    assert calls == ["2026-09-16"]
    if invalid:
        assert fallback.tolist() == [True]
        np.testing.assert_array_equal(weights, np.zeros((1, 17)))
    else:
        assert fallback.tolist() == [False]
        np.testing.assert_array_equal(weights[0], original.w_final * 0.5)


def test_backtest_builds_run_owned_typed_inputs(monkeypatch, tmp_path):
    cfg = AppConfig()
    captured = {}
    original = PortfolioDecision(
        w_final=np.array([0.5, -0.5] + [0.0] * 15), scores=np.arange(17, dtype=float),
        mu_gap=np.zeros(17), sigma_gap=np.ones(17), Omega_gap=np.eye(17),
        fallback={"gap_data_missing": False, "audit_failure": False}, pit_binning={"multiplier": 1.0},
        leakage={"status": "PASSED"}, numerical={"status": "PASSED"}, alerts=[], summary={}, run_config=cfg.v2,
    )

    def decide(**kwargs):
        captured["inputs"] = kwargs["inputs"]
        return original

    monkeypatch.setattr(backtester, "build_v2_model_bundle", lambda *a, **kw: SimpleNamespace(
        decision_model=SimpleNamespace(decide=decide), overlay_enabled=False, run_config=cfg.v2,
    ))
    dates = pd.DatetimeIndex(["2026-09-16"])
    frame = pd.DataFrame({"sig_date": ["2026-09-15"]}, index=dates)
    open_910 = pd.DataFrame(0.01, index=dates, columns=JP_TICKERS)
    monkeypatch.setattr(backtester, "build_open_910_returns", lambda *_a, **_kw: open_910.copy())

    weights, fallback, _ = backtester.BacktestEngine._generate_v2_weights(
        frame, cfg, tmp_path, dates, 17, None, None, 1,
    )

    inputs = captured["inputs"]
    assert inputs.known.source == "backtest_pit"
    assert inputs.historical.open_910_returns is not None
    pd.testing.assert_frame_equal(inputs.historical.open_910_returns, open_910)
    assert inputs.historical.pit_ir_history_for(dates[0]) is not None
    assert fallback.tolist() == [False]
    np.testing.assert_array_equal(weights[0], original.w_final)


def test_run_v2_backtest_rejects_historical_frame_mismatch() -> None:
    frame = pd.DataFrame(
        {"sig_date": [pd.Timestamp("2026-09-15")], "feature": [1.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-09-16")]),
    )
    mismatched = frame.copy()
    mismatched.loc[:, "feature"] = 2.0

    with pytest.raises(ValueError, match="historical_inputs frame must match df_exec"):
        backtester.BacktestEngine.run_v2_backtest(
            AppConfig(),
            gap_input_dir=None,
            df_exec=frame,
            start_date="2026-09-16",
            end_date="2026-09-16",
            historical_inputs=HistoricalInputs(mismatched, source="test"),
        )
