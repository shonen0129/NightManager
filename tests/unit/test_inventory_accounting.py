from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from leadlag.core.pnl import simulate_daily_pnl


def _run(weights, dates=None, **kwargs):
    weights = np.asarray(weights, dtype=float)
    args = dict(
        weights=weights,
        target_returns=np.zeros_like(weights),
        gap_returns=np.zeros_like(weights),
        sim_dates=pd.DatetimeIndex(dates or ["2026-10-02", "2026-10-05"]),
        slip=0.0,
        financing_daily=0.0,
        borrow_daily=0.0,
        reverse_daily=0.0,
        alpha_long=0.75,
        alpha_short=0.5,
    )
    args.update(kwargs)
    return simulate_daily_pnl(**args)


@pytest.mark.parametrize(
    "dates,days", [(["2026-10-02", "2026-10-05"], 3), (["2026-10-09", "2026-10-13"], 4)]
)
def test_carry_marks_quantities_and_accrues_incoming_calendar_fees(dates, days):
    result = _run(
        [[1.0, -1.0], [0.0, 0.0]],
        dates,
        target_returns=np.array([[0.2, -0.1], [0.0, 0.0]]),
        gap_returns=np.array([[0.0, 0.0], [0.1, -0.1]]),
        open_910_returns=np.array([[0.0, 0.0], [-0.1, 0.1]]),
        financing_daily=0.001,
        borrow_daily=0.002,
        reverse_daily=0.003,
    )
    # Close: long .75*1.2=.9, short -.5*.9=-.45, NAV=1.3.
    # Gap PnL .09+.045=.135, morning -.099-.0405=-.1395.
    np.testing.assert_allclose(result["carry_gap_returns"], [0.0, 0.135 / 1.3])
    np.testing.assert_allclose(result["carry_open_910_returns"], [0.0, -0.1395 / 1.3])
    np.testing.assert_allclose(result["financing_costs"], [0.0, 0.9 * 0.001 * days / 1.3])
    np.testing.assert_allclose(result["borrow_costs"], [0.0, 0.45 * 0.002 * days / 1.3])
    np.testing.assert_allclose(result["reverse_costs"], [0.0, 0.45 * 0.003 * days / 1.3])
    np.testing.assert_allclose(result["opening_volume"][1], (0.891 + 0.4455) / 1.3)
    assert result["terminal_inventory"]["equity"] == pytest.approx(
        1.2955 - (0.0009 + 0.00225) * days
    )
    assert result["terminal_inventory"]["holdings"] == [0.0, 0.0]
    np.testing.assert_allclose(
        result["gross_returns"], np.asarray(result["net_returns"]) + result["costs"]
    )


@pytest.mark.parametrize(
    "policy,fee,holdings,cash",
    [("liquidate", 0.0021, [0.0], 1.0979), ("open_inventory", 0.001, [1.1], -0.001)],
)
def test_array_terminal_inventory_cash_and_cost(policy, fee, holdings, cash):
    result = _run(
        [[1.0]],
        ["2026-10-02"],
        target_returns=np.array([[0.1]]),
        alpha_long=1.0,
        slip=0.001,
        financing_daily=0.1,
        terminal_policy=policy,
    )
    terminal = result["terminal_inventory"]
    assert result["slip_costs"] == pytest.approx([fee])
    assert result["financing_costs"] == [0.0]  # no fabricated next-day accrual
    assert terminal["holdings"] == pytest.approx(holdings)
    assert terminal["cash"] == pytest.approx(cash)
    assert terminal["equity"] == pytest.approx(cash + sum(holdings))
    assert terminal["mark_time"] == "TSE_close"
    assert terminal["mark_date"] == "2026-10-02"


def test_initial_holdings_are_marked_before_flat_rebalance():
    result = _run(
        [[0.0, 0.0]],
        ["2026-10-05"],
        initial_holdings=np.array([0.75, -0.5]),
        initial_cash=0.75,
        initial_mark_date="2026-10-02",
        gap_returns=np.array([[0.1, -0.1]]),
        open_910_returns=np.array([[-0.1, 0.1]]),
        slip=0.001,
        financing_daily=0.001,
    )
    assert result["gross_returns"] == pytest.approx([-0.0025])
    assert result["opening_volume"] == pytest.approx([1.2375])
    assert result["net_returns"] == pytest.approx([-0.0025 - 0.0012375 - 0.00225])
    assert result["terminal_inventory"]["cash"] == pytest.approx(0.9940125)


def test_artifact_boundary_state_transfer_matches_single_inventory_run():
    dates = pd.DatetimeIndex(["2026-10-02", "2026-10-05"])
    weights = np.array([[1.0, -1.0], [0.5, -0.5]])
    target = np.array([[0.1, -0.1], [-0.05, 0.03]])
    gap = np.array([[0.0, 0.0], [0.02, 0.03]])
    common = dict(
        slip=0.001,
        financing_daily=0.001,
        borrow_daily=0.002,
        reverse_daily=0.003,
        side_leverage=1.3,
    )
    full = _run(weights, list(dates), target_returns=target, gap_returns=gap, **common)
    first = _run(
        weights[:1],
        list(dates[:1]),
        target_returns=target[:1],
        gap_returns=gap[:1],
        terminal_policy="open_inventory",
        **common,
    )
    state = first["terminal_inventory"]
    second = _run(
        weights[1:],
        list(dates[1:]),
        target_returns=target[1:],
        gap_returns=gap[1:],
        initial_holdings=state["holdings"],
        initial_cash=state["cash"],
        initial_mark_date=state["mark_date"],
        initial_target_weights=state["target_weights"],
        **common,
    )
    for key in (
        "net_returns",
        "costs",
        "turnover",
        "target_weight_turnover",
        "cash",
        "equity",
        "holdings",
    ):
        np.testing.assert_allclose(full[key], first[key] + second[key], atol=1e-14)
    assert full["terminal_inventory"] == second["terminal_inventory"]


def test_execution_volume_marks_close_price_and_distinguishes_target_turnover():
    result = _run(
        [[1.0, -1.0]],
        ["2026-10-02"],
        target_returns=np.array([[0.1, -0.1]]),
        side_leverage=1.3,
        alpha_long=0.0,
        alpha_short=0.0,
        slip=0.001,
    )
    assert result["opening_volume"] == pytest.approx([2.6])
    assert result["closing_volume"] == pytest.approx([2.6])
    assert result["execution_volume"] == pytest.approx([5.2])
    assert result["turnover"] == pytest.approx([2.6])
    assert result["target_weight_turnover"] == [1.0]
    assert result["slip_costs"] == pytest.approx([0.0052])


def test_zero_effective_inventory_ignores_missing_price_labels():
    result = _run([[1.]], ["2026-10-02"], side_leverage=0., target_returns=np.array([[np.nan]]),
                  gap_returns=np.array([[np.nan]]), oc_returns=np.array([[np.nan]]))
    assert result["net_returns"] == [0.]
    assert result["gross_returns_oc"] == [0.]
    assert result["execution_volume"] == [0.]
    assert result["terminal_inventory"]["cash"] == 1.


@pytest.mark.parametrize(
    "kwargs",
    [
        {"initial_holdings": [0.1]},
        {"terminal_policy": "unknown"},
        {"alpha_long": 1.01},
        {"side_leverage": -1.0},
        {"initial_mark_date": "2026-10-06"},
    ],
)
def test_invalid_accounting_boundaries_rejected(kwargs):
    with pytest.raises(ValueError):
        _run([[0.0]], ["2026-10-05"], **kwargs)
