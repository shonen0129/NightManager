import numpy as np

from research.experiments.order_economics import (
    OrderCostSchedule,
    estimate_order_execution_costs,
    expected_gross_returns_from_overlay,
    select_partial_orders,
)


def test_restore_known_fixed_cost_from_directional_lgbm_target():
    expected = expected_gross_returns_from_overlay(
        np.array([0.001, 0.001, 0.001]),
        np.array([1.0, -1.0, 0.0]),
        fixed_round_trip_cost=0.001,
    )

    np.testing.assert_allclose(expected, [0.002, -0.002, -0.002])


def test_order_cost_breakdown_uses_half_spread_and_separate_components():
    costs = estimate_order_execution_costs(
        delta_weights=np.array([0.1]),
        target_weights=np.array([0.1]),
        tickers=["1617.T"],
        side_leverage=1.5,
        alpha_long=0.75,
        alpha_short=0.5,
        schedule=OrderCostSchedule(
            full_spread_bps=25.0,
            commission_bps_per_side=2.0,
            impact_bps_per_side=3.0,
        ),
    )

    row = costs.iloc[0]
    assert row["spread_bps_per_side"] == 12.5
    assert np.isclose(row["spread_cost_return"], 0.000234375)
    assert np.isclose(row["commission_cost_return"], 0.0000375)
    assert np.isclose(row["impact_cost_return"], 0.00005625)
    assert np.isclose(row["cost_return"], 0.000328125)


def test_order_cost_schedule_accepts_ticker_specific_spreads():
    costs = estimate_order_execution_costs(
        delta_weights=np.array([0.1, 0.1]),
        target_weights=np.array([0.1, -0.1]),
        tickers=["1617.T", "1618.T"],
        side_leverage=1.0,
        alpha_long=0.75,
        alpha_short=0.5,
        schedule=OrderCostSchedule(
            full_spread_bps={"1617.T": 20.0, "1618.T": 40.0},
            impact_bps_per_side={"1617.T": 1.0, "1618.T": 2.0},
        ),
    )

    np.testing.assert_allclose(costs["spread_bps_per_side"], [10.0, 20.0])
    np.testing.assert_allclose(costs["impact_bps_per_side"], [1.0, 2.0])


def test_partial_order_selector_keeps_net_and_gross_constraints():
    target = np.array([0.5, 0.5, -0.5, -0.5])
    previous = np.zeros(4)
    returns = np.array([0.01, -0.01, -0.01, 0.01])
    chosen, diagnostics = select_partial_orders(
        target_weights=target,
        previous_weights=previous,
        expected_returns=returns,
        tickers=["a", "b", "c", "d"],
        side_leverage=1.0,
        alpha_long=0.75,
        alpha_short=0.5,
        schedule=OrderCostSchedule(full_spread_bps=2.0),
        max_gross=2.0,
        max_abs_net=0.05,
    )

    assert abs(float(chosen.sum())) <= 0.05 + 1e-12
    assert float(np.abs(chosen).sum()) <= 2.0 + 1e-12
    assert chosen[0] > 0.0
    assert chosen[2] < 0.0
    assert chosen[1] == 0.0
    assert chosen[3] == 0.0
    assert diagnostics["eligible"].sum() == 2


def test_high_order_cost_rejects_all_candidate_moves():
    chosen, diagnostics = select_partial_orders(
        target_weights=np.array([0.5, -0.5]),
        previous_weights=np.zeros(2),
        expected_returns=np.array([0.001, -0.001]),
        tickers=["long", "short"],
        side_leverage=1.0,
        alpha_long=0.75,
        alpha_short=0.5,
        schedule=OrderCostSchedule(full_spread_bps=1000.0),
    )

    np.testing.assert_allclose(chosen, [0.0, 0.0])
    assert not diagnostics["eligible"].any()
