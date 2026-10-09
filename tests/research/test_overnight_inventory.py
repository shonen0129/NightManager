from __future__ import annotations

import numpy as np
import pandas as pd

from leadlag.core.pnl import simulate_daily_pnl
from research.overnight_inventory import (
    AdaptiveCarryConfig,
    adaptive_carry_masks,
    attribute_inventory_transitions,
)


def _masks(weights: np.ndarray, gaps: np.ndarray, borrow: float = 0.0115) -> np.ndarray:
    masks, _ = adaptive_carry_masks(
        weights=weights,
        gap_returns=gaps,
        sim_dates=pd.date_range("2025-01-06", periods=len(weights), freq="B"),
        slip=0.0005,
        financing_annual=0.025,
        borrow_annual=borrow,
        reverse_fee_bps=2.0,
        config=AdaptiveCarryConfig(lookback_sessions=3),
    )
    return masks


def test_adaptive_carry_prediction_ignores_future_weights_and_marks() -> None:
    weights = np.array([[1.0], [1.0], [0.8], [-1.0], [-0.7], [1.0]])
    gaps = np.array([[0.001], [0.002], [-0.001], [0.0], [0.003], [-0.02]])
    original = _masks(weights, gaps)

    changed_weights = weights.copy()
    changed_weights[5:, 0] = -4.0
    changed_gaps = gaps.copy()
    changed_gaps[5:, 0] = 0.5
    perturbed = _masks(changed_weights, changed_gaps)

    np.testing.assert_allclose(original[:5], perturbed[:5])
    assert original[-1, 0] == 0.0


def test_higher_short_borrow_cost_reduces_short_carry() -> None:
    weights = -np.ones((12, 1), dtype=float)
    gaps = np.zeros_like(weights)
    base = _masks(weights, gaps, borrow=0.0115)
    stress = _masks(weights, gaps, borrow=0.30)

    assert float(np.mean(stress[:-1])) < float(np.mean(base[:-1]))


def test_transition_and_overlap_diagnostics_are_ticker_specific() -> None:
    weights = np.array(
        [
            [1.0, -1.0],
            [1.0, 1.0],
            [0.5, -0.8],
            [-1.0, -0.7],
            [-0.8, 0.5],
        ]
    )
    gaps = np.zeros_like(weights)
    masks, diagnostics = adaptive_carry_masks(
        weights=weights,
        gap_returns=gaps,
        sim_dates=pd.date_range("2025-01-06", periods=len(weights), freq="B"),
        slip=0.0005,
        financing_annual=0.025,
        borrow_annual=0.0115,
        reverse_fee_bps=2.0,
        config=AdaptiveCarryConfig(lookback_sessions=4),
    )

    assert masks.shape == weights.shape
    assert len(diagnostics) == weights.size
    assert set(diagnostics["ticker_index"]) == {0, 1}
    assert np.isfinite(masks).all()
    assert ((masks >= 0) & (masks <= 1)).all()


def test_ticker_transition_attribution_reconciles_inventory_ledger() -> None:
    dates = pd.DatetimeIndex(["2025-01-06", "2025-01-07", "2025-01-08"])
    weights = np.array([[1.0, -1.0], [0.8, 0.5], [0.5, 0.0]])
    masks = np.array([[0.75, 0.5], [0.5, 0.25], [0.0, 0.0]])
    target = np.array([[0.01, -0.02], [0.03, -0.01], [0.0, 0.0]])
    gaps = np.array([[0.0, 0.0], [0.02, 0.01], [-0.01, 0.0]])
    days = np.array([1.0, 1.0, 0.0])
    attributed = attribute_inventory_transitions(
        weights=weights,
        target_returns=target,
        gap_returns=gaps,
        alpha_masks=masks,
        sim_dates=dates,
        calendar_days=days,
        slip=0.0005,
        financing_daily=0.025 / 365,
        borrow_daily=0.0115 / 365,
        reverse_daily=2.0 / 10_000,
        side_leverage=1.3,
    )
    pnl = simulate_daily_pnl(
        weights=weights,
        target_returns=target,
        gap_returns=gaps,
        sim_dates=dates,
        slip=0.0005,
        financing_daily=0.025 / 365,
        borrow_daily=0.0115 / 365,
        reverse_daily=2.0 / 10_000,
        alpha_long=0.0,
        alpha_short=0.0,
        alpha_masks=masks,
        calendar_days=days,
        side_leverage=1.3,
    )

    np.testing.assert_allclose(
        attributed.groupby("date")["gross_pnl"].sum().to_numpy(), pnl["gross_returns"]
    )
    np.testing.assert_allclose(
        attributed.groupby("date")["total_cost"].sum().to_numpy(), pnl["costs"]
    )
    np.testing.assert_allclose(
        attributed.groupby("date")["net_pnl"].sum().to_numpy(), pnl["net_returns"]
    )
    assert (attributed["transition_class"] == "next_signal_reversal").sum() == 1
    reversal = attributed.loc[
        (attributed["date"] == dates[0]) & (attributed["ticker_index"] == 1)
    ].iloc[0]
    np.testing.assert_allclose(reversal["close_slippage_delta_vs_flat"], -0.000325)
    np.testing.assert_allclose(reversal["next_open_slippage_delta_vs_flat"], 0.000325)
    np.testing.assert_allclose(reversal["transition_slippage_delta_vs_flat"], 0.0)
