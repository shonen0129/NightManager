"""Direct regressions for the used bps cost functions after removing dead wrappers."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from leadlag.execution.microstructure.slippage_model import (
    CostSource,
    compute_borrow_bps_daily,
    compute_entry_cost_bps,
    compute_exit_cost_bps,
    compute_financing_bps_daily,
)


@pytest.mark.parametrize('side', ['BUY', 'SELL'])
def test_fixed_spread_roundtrip_is_sum_of_two_one_way_costs(side):
    entry, source = compute_entry_cost_bps(None, 100_000, side, fallback_roundtrip_bps=20)
    exit_cost, exit_source = compute_exit_cost_bps(None, 100_000, side, fallback_roundtrip_bps=20)
    assert entry == exit_cost == 10
    assert source == exit_source == CostSource.FIXED_SPREAD_FALLBACK
    assert (entry + exit_cost)/10000 == .002


@pytest.mark.parametrize('days', [1, 3, 5])
def test_holding_costs_use_calendar_days_and_explicit_bps(days):
    assert compute_financing_bps_daily('BUY', .025, days) == pytest.approx(.025/365*10000*days)
    assert compute_financing_bps_daily('SELL', .025, days) == 0
    assert compute_borrow_bps_daily(.0115, days) == pytest.approx(.0115/365*10000*days)


def test_real_order_book_path_and_close_fallback_remain_distinct():
    snapshot = SimpleNamespace(lob_available=True, ticker='7203', last_price=2501., cost_source='lob_snapshot')
    for level in range(1, 6):
        setattr(snapshot, f'bid_price_{level}', 2500. - (level-1)*2)
        setattr(snapshot, f'ask_price_{level}', 2502. + (level-1)*2)
        setattr(snapshot, f'bid_size_{level}', 1000)
        setattr(snapshot, f'ask_size_{level}', 1000)
    entry, source = compute_entry_cost_bps(snapshot, 100_000, 'BUY')
    assert source == CostSource.LOB_SNAPSHOT and entry > 0
    exit_cost, exit_source = compute_exit_cost_bps(snapshot, 100_000, 'SELL')
    assert exit_cost == 7.5 and exit_source == CostSource.FIXED_SPREAD_FALLBACK


@pytest.mark.parametrize('kind, source', [('api_error', CostSource.API_ERROR), ('not_configured', CostSource.NOT_CONFIGURED)])
def test_failed_or_unconfigured_book_preserves_cost_source(kind, source):
    snapshot = SimpleNamespace(lob_available=False, cost_source=kind)
    assert compute_entry_cost_bps(snapshot, 100_000, 'BUY') == (7.5, source)
    assert compute_exit_cost_bps(snapshot, 100_000, 'SELL') == (7.5, source)
