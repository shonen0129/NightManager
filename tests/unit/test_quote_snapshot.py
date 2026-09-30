from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from leadlag.data.quote_snapshot import freeze_quote_snapshot, load_frozen_quote_snapshot
from leadlag.data.tickers import JP_TICKERS, JP_TICKERS_WITH_TOPIX, US_TICKERS
from leadlag.domain.inputs import KnownMarketInputs
from research.diagnostics import gap_inputs


def _snapshot_record(response_time: str = "2026-09-28T09:10:07+09:00") -> dict:
    return {
        "status": "OBSERVED",
        "window_valid": True,
        "request_started_at": "2026-09-28T09:10:02+09:00",
        "response_received_at": response_time,
        "observed_at": response_time,
        "timestamp_source": "local_response_receipt",
        "source": "tachibana:CLMMfdsGetMarketPrice",
        "rows": [
            {
                "ticker": ticker,
                "status": "OBSERVED",
                "request_started_at": "2026-09-28T09:10:02+09:00",
                "response_received_at": response_time,
                "observed_at": response_time,
                "timestamp_source": "local_response_receipt",
                "source": "tachibana:CLMMfdsGetMarketPrice",
                "bid_price_1": 100.0,
                "ask_price_1": 102.0,
                "quote_mid_price": 101.0,
                "valid_two_sided_quote": True,
            }
            for ticker in JP_TICKERS_WITH_TOPIX
        ],
    }


def test_frozen_quote_snapshot_is_complete_timestamped_and_immutable(tmp_path):
    first = _snapshot_record()
    path = freeze_quote_snapshot(
        first,
        tmp_path,
        trade_date="2026-09-28",
    )
    first_payload = json.loads(path.read_text(encoding="utf-8"))

    second = _snapshot_record("2026-09-28T09:10:15+09:00")
    second["rows"][0].update(bid_price_1=998.0, ask_price_1=1000.0, quote_mid_price=999.0)
    freeze_quote_snapshot(second, tmp_path, trade_date="2026-09-28")
    frozen = load_frozen_quote_snapshot(tmp_path, trade_date="2026-09-28")

    assert frozen.snapshot_id == first_payload["snapshot_id"]
    assert frozen.as_of == pd.Timestamp("2026-09-28T09:10:07+09:00")
    assert frozen.prices[JP_TICKERS_WITH_TOPIX[0]] == 101.0
    assert set(frozen.prices) == set(JP_TICKERS_WITH_TOPIX)
    assert all(
        value == "2026-09-28T09:10:07+09:00"
        for value in frozen.observed_at.values()
    )


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda record: record.update(response_received_at="2026-09-28T09:10:31+09:00"), "window"),
        (lambda record: record.update(timestamp_source="provider_guess"), "receipt time"),
        (lambda record: record["rows"].pop(), "missing required tickers"),
        (lambda record: record["rows"][0].update(ask_price_1=99.0), "invalid two-sided quote"),
    ],
)
def test_invalid_quote_snapshot_is_rejected(tmp_path, mutate, message):
    record = _snapshot_record()
    mutate(record)

    with pytest.raises(ValueError, match=message):
        freeze_quote_snapshot(record, tmp_path, trade_date="2026-09-28")


def test_typed_known_inputs_fingerprint_quote_id_and_reject_late_prices():
    as_of = pd.Timestamp("2026-09-28T09:10:07+09:00")
    timestamp = as_of.isoformat()
    prices = {ticker: 101.0 for ticker in JP_TICKERS_WITH_TOPIX[:-1]}
    source = {ticker: "tachibana:CLMMfdsGetMarketPrice:quote_mid" for ticker in prices}
    observed = {ticker: timestamp for ticker in prices}
    kwargs = {
        "trade_date": date(2026, 9, 28),
        "as_of": as_of,
        "ticker_order": tuple(JP_TICKERS_WITH_TOPIX[:-1]),
        "us_returns": np.zeros(15),
        "jp_gap_returns": np.zeros(17),
        "jp_betas": np.zeros(17),
        "topix_night_return": 0.0,
        "current_prices": prices,
        "prev_closes": prices,
        "source": "v2_bridge_live",
        "price_sources": source,
        "price_observed_at": observed,
        "quote_snapshot_id": "sha256:quote-a",
    }
    known = KnownMarketInputs(**kwargs)
    assert known.price_observed_at == observed
    assert known.quote_snapshot_id == "sha256:quote-a"
    assert KnownMarketInputs(**{**kwargs, "quote_snapshot_id": "sha256:quote-b"}).fingerprint != known.fingerprint

    late = dict(observed)
    late[JP_TICKERS_WITH_TOPIX[0]] = "2026-09-28T09:10:08+09:00"
    with pytest.raises(ValueError, match="later than as_of"):
        KnownMarketInputs(**{**kwargs, "price_observed_at": late})


def test_gap_generation_uses_frozen_quote_prices_without_last_price_cache(tmp_path, monkeypatch):
    dates = pd.to_datetime(["2026-09-25", "2026-09-28"])
    jp_close = pd.DataFrame(
        [
            {ticker: 100.0 for ticker in JP_TICKERS_WITH_TOPIX},
            {ticker: 100.0 for ticker in JP_TICKERS_WITH_TOPIX},
        ],
        index=dates,
    )
    us_close = pd.DataFrame(
        [{ticker: 100.0 for ticker in US_TICKERS}, {ticker: 101.0 for ticker in US_TICKERS}],
        index=dates,
    )
    previous_frame = pd.DataFrame(
        [{f"jp_beta_{ticker}": 0.5 for ticker in JP_TICKERS}],
        index=[pd.Timestamp("2026-09-28")],
    )
    frozen_prices = {ticker: 101.0 for ticker in JP_TICKERS_WITH_TOPIX}
    monkeypatch.setattr(
        gap_inputs,
        "save_current_prices_cache",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("frozen quote input must not be converted to an untimestamped cache")
        ),
    )

    result, client = gap_inputs.inject_tachibana_realtime_prices(
        previous_frame,
        {"jp_close": jp_close, "us_close": us_close},
        pd.Timestamp("2026-09-29"),
        frozen_prices=frozen_prices,
        quote_snapshot_id="quote-a",
        quote_observed_at="2026-09-29T09:10:07+09:00",
    )

    assert client is None
    assert result.loc[pd.Timestamp("2026-09-29"), "jp_open_trade_1617.T"] == 101.0
    assert result.loc[pd.Timestamp("2026-09-29"), "jp_gap_1617.T"] == pytest.approx(0.01)
    assert result.attrs["quote_snapshot_id"] == "quote-a"
    assert result.attrs["quote_snapshot_as_of"] == "2026-09-29T09:10:07+09:00"
