"""Observed 09:10 prices and missing-observation open fallback across entry points."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from leadlag.config.schemas import AppConfig
from leadlag.data.intraday_inputs import resolve_execution_prices
from leadlag.data.pit_lake import PITDataLake
from leadlag.data.tickers import JP_TICKERS, US_TICKERS
from leadlag.domain.inputs import HistoricalInputs
from leadlag.domain.portfolio import PortfolioDecision
from leadlag.execution import backtester
from research.experiments.ml_overlay_training import _collect_training_data


def inputs():
    date = pd.Timestamp("2026-08-14")
    frame = pd.DataFrame({
        "sig_date": ["2026-08-13"], "topix_night_return": [0.0],
        **{f"us_cc_{ticker}": [0.001] for ticker in US_TICKERS},
        **{f"jp_open_trade_{ticker}": [100.0] for ticker in JP_TICKERS},
        **{f"jp_close_sig_{ticker}": [98.0] for ticker in JP_TICKERS},
        **{f"jp_gap_{ticker}": [100.0 / 98.0 - 1.0] for ticker in JP_TICKERS},
        **{f"jp_beta_{ticker}": [1.0] for ticker in JP_TICKERS},
        **{f"jp_oc_{ticker}": [0.03] for ticker in JP_TICKERS},
    }, index=pd.DatetimeIndex([date]))
    intraday = pd.DataFrame(0.02, index=frame.index, columns=JP_TICKERS)
    intraday.loc[date, JP_TICKERS[-1]] = np.nan
    return date, frame, intraday


def decision(config):
    return PortfolioDecision(
        w_final=np.array([0.5, -0.5] + [0.0] * 15), scores=np.ones(17),
        mu_gap=np.ones(17) * 0.01, sigma_gap=np.ones(17) * 0.1, Omega_gap=np.eye(17),
        fallback={"gap_data_missing": False}, pit_binning={}, leakage={}, numerical={},
        alerts=[], summary={}, run_config=config.v2,
    )


def test_snapshot_uses_observation_and_identifies_open_fallback():
    date, frame, intraday = inputs()
    snapshot = PITDataLake(frame).get_execution_snapshot(date + pd.Timedelta(minutes=550), intraday)
    assert snapshot.current_prices[JP_TICKERS[0]] == 102.0
    assert snapshot.current_prices[JP_TICKERS[-1]] == 100.0
    assert snapshot.price_sources[JP_TICKERS[0]] == "observed_0910"
    assert snapshot.price_sources[JP_TICKERS[-1]] == "daily_open_fallback"
    np.testing.assert_allclose(snapshot.jp_gap_returns[:-1], 102.0 / 98.0 - 1.0)
    assert snapshot.jp_gap_returns[-1] == pytest.approx(100.0 / 98.0 - 1.0)
    known = snapshot.to_known_inputs()
    with pytest.raises(TypeError):
        known.price_sources[JP_TICKERS[0]] = "daily_open_fallback"
    assert replace(known, price_sources={}).fingerprint != known.fingerprint
    assert np.isnan(intraday.loc[date, JP_TICKERS[-1]])


@pytest.mark.parametrize("invalid", [np.inf, -np.inf, -1.0, -2.0])
def test_corrupt_observation_is_not_an_open_fallback(invalid):
    date, frame, intraday = inputs()
    intraday.loc[date, JP_TICKERS[0]] = invalid
    with pytest.raises(ValueError, match="observation"):
        resolve_execution_prices(frame, intraday)


def test_bt_and_training_share_prices_gaps_and_source(monkeypatch, tmp_path):
    date, frame, intraday = inputs()
    config = AppConfig()
    history = HistoricalInputs(frame, open_910_returns=intraday)
    captured = []

    def decide(**kwargs):
        captured.append(kwargs["inputs"])
        return decision(config)

    model = SimpleNamespace(decide=decide)
    monkeypatch.setattr(backtester, "build_v2_model_bundle", lambda *a, **kw: SimpleNamespace(
        decision_model=model, run_config=config.v2, overlay_enabled=False,
    ))
    backtester.BacktestEngine._generate_v2_weights(
        frame, config, tmp_path, frame.index, 17, None, None, 1,
        historical_inputs=history,
    )
    training = _collect_training_data(
        frame.index, frame, np.ones((1, 17)) * 0.02, tmp_path, config.v2,
        pd.DataFrame(0.01, index=frame.index, columns=JP_TICKERS),
        historical_inputs=history, open_910_returns=intraday, decision_model=model,
    )
    assert len(captured) == 2 and len(training) == 17
    for supplied in captured:
        assert supplied.known.current_prices[JP_TICKERS[0]] == 102.0
        assert supplied.known.current_prices[JP_TICKERS[-1]] == 100.0
        assert supplied.known.price_sources[JP_TICKERS[-1]] == "daily_open_fallback"
    expected = np.array([102.0] * 16 + [100.0]) / 98.0 - 1.0
    np.testing.assert_allclose(training["gap"].to_numpy(), expected)
    np.testing.assert_allclose(training["score_x_gap"].to_numpy(), expected)
