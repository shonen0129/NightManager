"""Tests for the data provider abstraction."""

from __future__ import annotations

from datetime import date, datetime
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from leadlag.data.providers import TachibanaProvider, YFinanceProvider


def _fake_yf_download(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    """Return a fake multi-ticker yfinance-like DataFrame."""
    idx = pd.date_range(start=start.strftime("%Y-%m-%d"), periods=3, freq="B")
    data = {
        ("Open", "1617.T"): [100.0, 101.0, 102.0],
        ("High", "1617.T"): [101.0, 102.0, 103.0],
        ("Low", "1617.T"): [99.0, 100.0, 101.0],
        ("Close", "1617.T"): [100.5, 101.5, 102.5],
        ("Volume", "1617.T"): [1000, 2000, 3000],
        ("Open", "1321.T"): [200.0, 201.0, 202.0],
        ("High", "1321.T"): [201.0, 202.0, 203.0],
        ("Low", "1321.T"): [199.0, 200.0, 201.0],
        ("Close", "1321.T"): [200.5, 201.5, 202.5],
        ("Volume", "1321.T"): [500, 600, 700],
    }
    return pd.DataFrame(data, index=idx)


class TestYFinanceProvider:
    def test_fetch_daily_ohlc_no_network(self) -> None:
        """YFinanceProvider can be tested without network using a download stub."""
        provider = YFinanceProvider(download_fn=_fake_yf_download)
        result = provider.fetch_daily_ohlc(["1617.T", "1321.T"], date(2025, 1, 1), date(2025, 1, 7))

        assert set(result.keys()) == {"1617.T", "1321.T"}
        assert list(result["1617.T"].columns) == ["open", "high", "low", "close", "volume"]
        assert np.isclose(result["1617.T"]["close"].iloc[-1], 102.5)
        assert np.isclose(result["1321.T"]["open"].iloc[0], 200.0)

    def test_source_name(self) -> None:
        provider = YFinanceProvider(download_fn=_fake_yf_download)
        assert provider.source_name() == "yfinance"


class TestTachibanaProvider:
    def test_fetch_intraday_quote(self) -> None:
        client = MagicMock()
        client.fetch_open_prices.return_value = {"1617.T": 100.0, "1321.T": 200.0}
        provider = TachibanaProvider(client=client)
        at = datetime(2025, 1, 6, 9, 10)
        result = provider.fetch_intraday_quote(["1617.T", "1321.T"], at)

        assert result == {"1617.T": 100.0, "1321.T": 200.0}
        client.fetch_open_prices.assert_called_once_with(["1617.T", "1321.T"], allow_missing=True)

    def test_fetch_daily_ohlc_not_supported(self) -> None:
        provider = TachibanaProvider(client=MagicMock())
        with pytest.raises(NotImplementedError):
            provider.fetch_daily_ohlc(["1617.T"], date(2025, 1, 1), date(2025, 1, 7))
