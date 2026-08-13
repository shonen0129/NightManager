"""Tests for the execution DataFrame local cache helpers."""

import pandas as pd
import pytest

from leadlag.data import market_data_cache


def test_load_df_exec_rejects_stale_fallback_when_max_stale_set(tmp_path, monkeypatch):
    """When max_stale_bdays is set, stale df_exec cache must not be used as fallback."""
    store_path = tmp_path / "df_exec.sqlite"
    monkeypatch.setattr(market_data_cache, "_df_exec_store_path", lambda: store_path)
    monkeypatch.setattr(
        market_data_cache, "_etf_store_path", lambda: tmp_path / "nonexistent.sqlite"
    )

    yesterday = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
    index = pd.DatetimeIndex([yesterday], name="trade_date")
    df_exec = pd.DataFrame({"topix_night_return": [0.01]}, index=index)
    market_data_cache.save_df_exec_to_local_cache(df_exec)

    with pytest.raises(RuntimeError, match="stale cache fallback is disabled"):
        market_data_cache.load_df_exec_from_local_cache(max_stale_bdays=0)


def test_load_df_exec_allows_stale_fallback_when_max_stale_unset(tmp_path, monkeypatch):
    """When max_stale_bdays is None, stale df_exec cache may be used as fallback."""
    store_path = tmp_path / "df_exec.sqlite"
    monkeypatch.setattr(market_data_cache, "_df_exec_store_path", lambda: store_path)
    monkeypatch.setattr(
        market_data_cache, "_etf_store_path", lambda: tmp_path / "nonexistent.sqlite"
    )

    yesterday = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
    index = pd.DatetimeIndex([yesterday], name="trade_date")
    df_exec = pd.DataFrame({"topix_night_return": [0.01]}, index=index)
    market_data_cache.save_df_exec_to_local_cache(df_exec)

    loaded = market_data_cache.load_df_exec_from_local_cache(max_stale_bdays=None)
    pd.testing.assert_frame_equal(loaded, df_exec)
