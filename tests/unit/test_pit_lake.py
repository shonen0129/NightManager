"""Unit tests for Point-in-Time (PIT) Data Lake."""

import numpy as np
import pandas as pd
import pytest

from leadlag.data.pit_lake import MarketSnapshot, PITDataLake, PITLookaheadError
from leadlag.data.tickers import JP_TICKERS, US_TICKERS


@pytest.fixture
def sample_pit_df() -> pd.DataFrame:
    """Create a synthetic df_exec for PIT data lake testing."""
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    data = {}

    # US columns
    for tk in US_TICKERS:
        data[f"us_cc_{tk}"] = np.random.randn(10) * 0.01

    # JP columns
    for tk in JP_TICKERS:
        data[f"jp_open_trade_{tk}"] = 1000.0 + np.random.randn(10) * 10
        data[f"jp_close_sig_{tk}"] = 1000.0 + np.random.randn(10) * 10
        data[f"jp_beta_{tk}"] = 1.0 + np.random.randn(10) * 0.1
        data[f"jp_gap_{tk}"] = np.random.randn(10) * 0.005
        # Leakage-probe column: same day close return
        data[f"jp_oc_{tk}"] = np.random.randn(10) * 0.01

    data["topix_night_return"] = np.random.randn(10) * 0.005
    return pd.DataFrame(data, index=dates)


def test_pit_lake_get_snapshot_integrity(sample_pit_df: pd.DataFrame):
    """Test that get_snapshot extracts exactly the expected dimensions."""
    lake = PITDataLake(sample_pit_df)

    test_date = sample_pit_df.index[3]
    snapshot = lake.get_snapshot(test_date)

    assert isinstance(snapshot, MarketSnapshot)
    assert snapshot.as_of == test_date
    assert len(snapshot.us_returns) == len(US_TICKERS)
    assert len(snapshot.jp_gap_returns) == len(JP_TICKERS)
    assert len(snapshot.jp_betas) == len(JP_TICKERS)
    assert len(snapshot.current_prices) == len(JP_TICKERS)
    assert len(snapshot.prev_closes) == len(JP_TICKERS)
    assert isinstance(snapshot.topix_night_return, float)


def test_pit_lake_lookahead_prevention(sample_pit_df: pd.DataFrame):
    """Test that snapshots never contain same-day intraday/close realization returns."""
    lake = PITDataLake(sample_pit_df)

    test_date = sample_pit_df.index[5]
    snapshot = lake.get_snapshot(test_date)

    # Must pass lookahead audit
    assert lake.validate_no_lookahead(test_date)

    # Snapshot must only contain allowed, pre-as-of attributes even when
    # same-day close columns exist in the source df_exec.
    assert not hasattr(snapshot, "jp_oc_returns")
    assert not hasattr(snapshot, "close_prices")


def test_pit_lake_future_date_handling(sample_pit_df: pd.DataFrame):
    """Test accessing out-of-range dates raises PITLookaheadError."""
    lake = PITDataLake(sample_pit_df)

    early_date = "2020-01-01"
    with pytest.raises(PITLookaheadError):
        lake.get_snapshot(early_date)

    future_date = sample_pit_df.index[-1] + pd.Timedelta(days=10)
    with pytest.raises(PITLookaheadError):
        lake.get_snapshot(future_date)
