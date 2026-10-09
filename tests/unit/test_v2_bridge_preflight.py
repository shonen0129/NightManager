from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from leadlag.data.pit_lake import MarketSnapshot, PITDataLake
from leadlag.data.quote_snapshot import FrozenQuoteSnapshot
from leadlag.data.tickers import JP_TICKERS
from leadlag.execution.v2_bridge import CurrentPricePreflight, QuotePreflight

JST = ZoneInfo("Asia/Tokyo")


def _frozen_quote() -> FrozenQuoteSnapshot:
    return FrozenQuoteSnapshot(
        trade_date="2026-10-07",
        as_of=datetime(2026, 10, 7, 9, 10, 4, tzinfo=JST),
        request_started_at=datetime(2026, 10, 7, 9, 10, 3, tzinfo=JST),
        snapshot_id="quote-1",
        prices={ticker: 100.0 for ticker in JP_TICKERS},
        observed_at={ticker: "2026-10-07T09:10:04+09:00" for ticker in JP_TICKERS},
        quote_rows={},
        source="tachibana:CLMMfdsGetMarketPrice",
        timestamp_source="local_response_receipt",
    )


def _quote_preflight(price_input: CurrentPricePreflight) -> QuotePreflight:
    market_snapshot = MarketSnapshot(
        as_of=pd.Timestamp("2026-10-07 09:10"),
        trade_date="2026-10-07",
        us_returns=np.zeros(15),
        jp_gap_returns=np.zeros(17),
        jp_betas=np.zeros(17),
        topix_night_return=0.0,
        current_prices={},
        prev_closes={},
    )
    return QuotePreflight(
        current_prices={"1301.T": 100.0},
        data_lake=PITDataLake(
            pd.DataFrame(
                {"marker": [0]},
                index=pd.DatetimeIndex([pd.Timestamp("2026-10-07")]),
            )
        ),
        market_snapshot=market_snapshot,
        decision_as_of=pd.Timestamp("2026-10-07 09:10"),
        price_input=price_input,
    )


def test_current_price_preflight_owns_immutable_price_selection() -> None:
    prices = {"1301.T": 100.0}
    preflight = CurrentPricePreflight(prices=prices, source="broker_current")

    prices["1301.T"] = 101.0

    assert preflight.prices["1301.T"] == 100.0
    with pytest.raises(TypeError):
        preflight.prices["1301.T"] = 102.0


def test_frozen_quote_source_requires_its_snapshot() -> None:
    with pytest.raises(ValueError, match="must be supplied together"):
        CurrentPricePreflight(prices={}, source="frozen_quote")


@pytest.mark.parametrize(
    ("price_input", "required"),
    [
        (CurrentPricePreflight(prices={}, source="broker_current"), False),
        (CurrentPricePreflight(prices={}, source="previous_close_placeholder"), False),
        (
            CurrentPricePreflight(
                prices={ticker: 100.0 for ticker in JP_TICKERS},
                source="frozen_quote",
                frozen_snapshot=_frozen_quote(),
            ),
            True,
        ),
    ],
)
def test_quote_preflight_carries_actual_account_risk_requirement(
    price_input: CurrentPricePreflight, required: bool
) -> None:
    quote_preflight = _quote_preflight(price_input)

    assert quote_preflight.requires_actual_account_risk is required
