from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from leadlag.config import AppConfig
from leadlag.data.gap_store import GapStore
from leadlag.data.quote_snapshot import FrozenQuoteSnapshot
from leadlag.data.tickers import JP_TICKERS, TOPIX_TICKER
from leadlag.pipeline.gap_publisher import (
    build_market_snapshot,
    compute_rank_reversal_signal,
    publish_gap_cache,
)


def test_rank_reversal_signal_ignores_current_day_close_label() -> None:
    dates = pd.date_range("2026-09-24", periods=4, freq="B")
    columns = [f"jp_oc_{ticker}" for ticker in JP_TICKERS]
    values = np.vstack(
        [
            np.arange(len(JP_TICKERS), dtype=float),
            np.arange(len(JP_TICKERS), dtype=float)[::-1],
            np.roll(np.arange(len(JP_TICKERS), dtype=float), 3),
            np.linspace(-100.0, 100.0, len(JP_TICKERS)),
        ]
    )
    frame = pd.DataFrame(values, index=dates, columns=columns)
    changed = frame.copy()
    changed.loc[dates[-1], columns] = np.linspace(1000.0, -1000.0, len(JP_TICKERS))

    first = compute_rank_reversal_signal(frame, dates[-1].date().isoformat())
    second = compute_rank_reversal_signal(changed, dates[-1].date().isoformat())

    np.testing.assert_array_equal(first, second)
    assert np.isfinite(first).all()


@pytest.fixture
def publication_inputs(sample_df_exec):
    frame, raw = sample_df_exec
    frame = frame.loc[:raw["jp_close"].index[-1]].tail(10).copy()
    day = frame.index[-1]
    prior_date = raw["jp_close"].index[raw["jp_close"].index < day][-1]
    prices = {ticker: float(frame.iloc[-1][f"jp_close_sig_{ticker}"]) for ticker in JP_TICKERS}
    prices[TOPIX_TICKER] = float(raw["jp_close"].loc[prior_date, TOPIX_TICKER]) * 1.01
    as_of = day.tz_localize("Asia/Tokyo") + pd.Timedelta(hours=9, minutes=10, seconds=2)
    frozen = FrozenQuoteSnapshot(
        trade_date=day.date().isoformat(), as_of=as_of,
        request_started_at=as_of - pd.Timedelta(seconds=1), snapshot_id="synthetic",
        prices=prices, observed_at={ticker: as_of.isoformat() for ticker in prices},
        quote_rows={}, source="tachibana:CLMMfdsGetMarketPrice", timestamp_source="local_response_receipt",
    )
    return frame, frozen


def test_market_snapshot_has_prior_topix_from_preprocessed_inputs(publication_inputs):
    frame, frozen = publication_inputs
    snapshot, _, _ = build_market_snapshot(frame, frozen, frozen.trade_date)
    assert snapshot.topix_night_return == pytest.approx(0.01)
    assert snapshot.prev_closes[TOPIX_TICKER] == pytest.approx(frozen.prices[TOPIX_TICKER] / 1.01)


def test_market_snapshot_rejects_quote_date_mismatch(publication_inputs):
    frame, frozen = publication_inputs
    with pytest.raises(ValueError, match="quote.*trade_date"):
        build_market_snapshot(frame, frozen, frame.index[-2].date().isoformat())


@pytest.mark.parametrize("fail_horizon", [None, 3])
def test_publish_gap_cache_uses_real_sources_and_sqlite(publication_inputs, tmp_path, monkeypatch, fail_horizon):
    frame, frozen = publication_inputs
    config = AppConfig.model_validate({"v2": {
        "mh_blend_enabled": True, "mh_horizons": [1, 3, 5],
        "cs_overlay_enabled": False, "shadow_ondemand_validation": False,
    }})
    path = tmp_path / "gap.sqlite"
    monkeypatch.setattr("leadlag.pipeline.gap_publisher.build_blpx_model", lambda *args, **kwargs: object())
    monkeypatch.setattr("leadlag.pipeline.gap_publisher.build_open_910_returns", lambda df, tickers: pd.DataFrame(0.0, index=df.index, columns=tickers))

    def compute(model, *, horizon, **kwargs):
        if horizon == fail_horizon:
            raise ValueError("synthetic computation failure")
        return np.full(len(JP_TICKERS), horizon * 0.001), np.eye(len(JP_TICKERS)) * 0.01

    monkeypatch.setattr("leadlag.models.v2.distribution_source._compute_ondemand", compute)
    args = dict(app_config=config, df_exec=frame, frozen_snapshot=frozen, gap_store=path, trade_date=frozen.trade_date)
    if fail_horizon is not None:
        with pytest.raises(RuntimeError, match="h=3 on-demand gap computation failed"):
            publish_gap_cache(**args)
        assert not path.exists()
        return
    result = publish_gap_cache(**args)
    assert result.horizons == (1, 3, 5)
    for horizon in result.horizons:
        mu, omega, metadata, _ = GapStore(path).load_horizon_bundle(frozen.trade_date, horizon=None if horizon == 1 else horizon)
        np.testing.assert_array_equal(mu, np.full(len(JP_TICKERS), horizon * 0.001))
        np.testing.assert_array_equal(omega, np.eye(len(JP_TICKERS)) * 0.01)
        assert metadata["quote_snapshot_id"] == frozen.snapshot_id
        assert result.verification[horizon]["source"] == "file_cache"
