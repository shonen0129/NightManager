"""Tests for the execution DataFrame local cache helpers."""

import pandas as pd
import pytest

from leadlag.data import market_data_cache
from leadlag.data.intraday_inputs import build_5m_910_prices


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


def test_load_intraday_cache_adjusts_1629_prices_without_rewriting_raw_cache(
    tmp_path, monkeypatch
):
    """Default 5m reads align prices while explicit raw reads preserve source bars."""
    store_path = tmp_path / "etf_prices.sqlite"
    monkeypatch.setattr(market_data_cache, "_etf_store_path", lambda: store_path)

    index = pd.DatetimeIndex(
        ["2026-03-27 09:10:00", "2026-03-30 09:10:00"]
    )
    raw_bars = pd.DataFrame(
        {
            ("Open", "1629.T"): [142_000.0, 284.0],
            ("High", "1629.T"): [143_000.0, 286.0],
            ("Low", "1629.T"): [141_000.0, 282.0],
            ("Close", "1629.T"): [142_500.0, 285.0],
            ("Adj Close", "1629.T"): [142_500.0, 285.0],
            ("Volume", "1629.T"): [22.0, 12_710.0],
        },
        index=index,
    )
    market_data_cache.save_intraday_cache(raw_bars, "5m")

    adjusted = market_data_cache.load_intraday_cache("5m")
    raw_read = market_data_cache.load_intraday_cache("5m", adjust_split_basis=False)
    execution_frame = pd.DataFrame(index=pd.DatetimeIndex([index[0].normalize()]))
    p_910 = build_5m_910_prices(execution_frame, ["1629.T"])

    assert adjusted is not None
    assert adjusted.loc[index[0], ("Close", "1629.T")] == 285.0
    assert adjusted.loc[index[1], ("Close", "1629.T")] == 285.0
    assert adjusted.loc[index[0], ("Adj Close", "1629.T")] == 285.0
    assert adjusted.loc[index[0], ("Volume", "1629.T")] == 22.0
    assert p_910.loc[index[0].normalize(), "1629.T"] == 284.0
    pd.testing.assert_frame_equal(raw_read, raw_bars)


def test_obsolete_proxy_cache_cannot_bypass_rebuild_via_stale_fallback(tmp_path, monkeypatch):
    from leadlag.data.cache_store import SqliteCacheStore

    cache_path = tmp_path / 'df_exec.sqlite'
    monkeypatch.setattr(market_data_cache, '_df_exec_store_path', lambda: cache_path)
    monkeypatch.setattr(market_data_cache, '_etf_store_path', lambda: tmp_path / 'absent.sqlite')
    frame = pd.DataFrame({'topix_night_return': [0.]}, index=pd.DatetimeIndex(['2026-10-01']))
    store = SqliteCacheStore(cache_path)
    store.set('df_exec', frame)
    store.set('df_exec_meta', {'columns': list(frame.columns)})
    assert not market_data_cache.is_df_exec_cache_valid()
    with pytest.raises(RuntimeError, match='no fallback'):
        market_data_cache.load_df_exec_from_local_cache()
