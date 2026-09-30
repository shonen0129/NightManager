"""Research helpers for gross-return forecasts and order-level decisions.

This module intentionally stays outside the production decision path.  The
current overlay is trained on directional return less a fixed round-trip cost;
the forecast adapter restores that known constant so the execution decision
can compare gross expected return with an independent order-cost estimate.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linprog


@dataclass(frozen=True)
class OrderCostSchedule:
    """Per-side execution cost assumptions for one scenario.

    ``full_spread_bps`` is the quoted bid/ask width.  Crossing from midpoint
    is charged half of that width per side.  Commission and impact are
    additional per-side assumptions and must not include spread again.
    """

    full_spread_bps: float | Mapping[str, float]
    commission_bps_per_side: float | Mapping[str, float] = 0.0
    impact_bps_per_side: float | Mapping[str, float] = 0.0

    def __post_init__(self) -> None:
        for name in (
            "full_spread_bps",
            "commission_bps_per_side",
            "impact_bps_per_side",
        ):
            raw_value = getattr(self, name)
            values = raw_value.values() if isinstance(raw_value, Mapping) else (raw_value,)
            for value in values:
                numeric = float(value)
                if not np.isfinite(numeric) or numeric < 0.0:
                    raise ValueError(f"{name} must be finite and non-negative")


def _cost_vector(
    value: float | Mapping[str, float], tickers: list[str], name: str
) -> np.ndarray:
    if isinstance(value, Mapping):
        missing = [ticker for ticker in tickers if ticker not in value]
        if missing:
            raise ValueError(f"{name} missing ticker costs: {missing}")
        result = np.asarray([value[ticker] for ticker in tickers], dtype=float)
    else:
        result = np.full(len(tickers), float(value), dtype=float)
    if not np.isfinite(result).all() or np.any(result < 0.0):
        raise ValueError(f"{name} must be finite and non-negative for every ticker")
    return result


def expected_gross_returns_from_overlay(
    raw_lgbm_prediction: np.ndarray,
    score_direction: np.ndarray,
    fixed_round_trip_cost: float,
) -> np.ndarray:
    """Recover a raw-return forecast from the current directional net label.

    The current raw target is ``sign(score) * realized_return - fixed_cost``.
    Therefore the regressor's output plus the known fixed cost estimates the
    gross return in the signal direction.  Multiplying by that same direction
    returns a per-ticker raw return forecast in portfolio weight coordinates.
    """

    prediction = np.asarray(raw_lgbm_prediction, dtype=float)
    direction = np.asarray(score_direction, dtype=float)
    if prediction.shape != direction.shape:
        raise ValueError("prediction and score_direction must have the same shape")
    if not np.isfinite(prediction).all() or not np.isfinite(direction).all():
        raise ValueError("prediction and score_direction must be finite")
    cost = float(fixed_round_trip_cost)
    if not np.isfinite(cost) or cost < 0.0:
        raise ValueError("fixed_round_trip_cost must be finite and non-negative")
    # This mirrors the training collector's ``1 if score > 0 else -1`` rule,
    # including the exact-zero boundary.
    direction = np.where(direction > 0.0, 1.0, -1.0)
    return direction * (prediction + cost)


def estimate_order_execution_costs(
    *,
    delta_weights: np.ndarray,
    target_weights: np.ndarray,
    tickers: list[str] | tuple[str, ...],
    side_leverage: float,
    alpha_long: float,
    alpha_short: float,
    schedule: OrderCostSchedule,
) -> pd.DataFrame:
    """Estimate each proposed order's spread, fee, and impact cost.

    Weights and returned ``cost_return`` are fractions of portfolio NAV.  The
    order's notional is ``abs(delta_weight) * side_leverage``.  The modeled
    execution sides match the existing daily convention: one entry plus the
    expected intraday close fraction for a non-flat target, or one side for a
    full exit.  Carry is excluded here because it is charged separately in
    portfolio P&L and is not an execution fee.
    """

    delta = np.asarray(delta_weights, dtype=float)
    target = np.asarray(target_weights, dtype=float)
    names = list(tickers)
    if delta.ndim != 1 or target.shape != delta.shape or len(names) != len(delta):
        raise ValueError("weights and tickers must be aligned one-dimensional arrays")
    if not np.isfinite(delta).all() or not np.isfinite(target).all():
        raise ValueError("weights must be finite")
    leverage = float(side_leverage)
    long_alpha = float(alpha_long)
    short_alpha = float(alpha_short)
    if not np.isfinite(leverage) or leverage <= 0.0:
        raise ValueError("side_leverage must be finite and positive")
    if not (0.0 <= long_alpha <= 1.0 and 0.0 <= short_alpha <= 1.0):
        raise ValueError("overnight alpha values must be in [0, 1]")

    close_fraction = np.where(target > 1e-12, 1.0 - long_alpha,
                              np.where(target < -1e-12, 1.0 - short_alpha, 0.0))
    execution_sides = 1.0 + close_fraction
    execution_sides[np.abs(target) <= 1e-12] = 1.0
    order_notional = np.abs(delta) * leverage
    full_spread_bps = _cost_vector(schedule.full_spread_bps, names, "full_spread_bps")
    spread_bps_per_side = full_spread_bps / 2.0
    commission_bps_per_side = _cost_vector(
        schedule.commission_bps_per_side, names, "commission_bps_per_side"
    )
    impact_bps_per_side = _cost_vector(
        schedule.impact_bps_per_side, names, "impact_bps_per_side"
    )
    components = {
        "spread_cost_return": order_notional * execution_sides * spread_bps_per_side / 10000.0,
        "commission_cost_return": order_notional * execution_sides * commission_bps_per_side / 10000.0,
        "impact_cost_return": order_notional * execution_sides * impact_bps_per_side / 10000.0,
    }
    frame = pd.DataFrame(
        {
            "ticker": names,
            "delta_weight": delta,
            "order_notional_nav": order_notional,
            "execution_sides": execution_sides,
            "full_spread_bps": full_spread_bps,
            "spread_bps_per_side": spread_bps_per_side,
            "commission_bps_per_side": commission_bps_per_side,
            "impact_bps_per_side": impact_bps_per_side,
            **components,
        }
    )
    frame["cost_return"] = frame[
        ["spread_cost_return", "commission_cost_return", "impact_cost_return"]
    ].sum(axis=1)
    return frame


def select_partial_orders(
    *,
    target_weights: np.ndarray,
    previous_weights: np.ndarray,
    expected_returns: np.ndarray,
    tickers: list[str] | tuple[str, ...],
    side_leverage: float,
    alpha_long: float,
    alpha_short: float,
    schedule: OrderCostSchedule,
    max_gross: float = 2.0,
    max_abs_net: float = 0.05,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Move only into positive expected-net-value orders under portfolio limits.

    The optimizer chooses a fraction in [0, 1] of each day's move toward the
    canonical target.  Candidate order value is change in weight times the
    LightGBM gross-return forecast; the separately estimated order cost is
    deducted once.  The LP enforces the model net and gross constraints.
    """

    target = np.asarray(target_weights, dtype=float)
    previous = np.asarray(previous_weights, dtype=float)
    mu = np.asarray(expected_returns, dtype=float)
    if target.ndim != 1 or previous.shape != target.shape or mu.shape != target.shape:
        raise ValueError("target, previous, and expected_returns must be aligned vectors")
    if not np.isfinite(target).all() or not np.isfinite(previous).all() or not np.isfinite(mu).all():
        raise ValueError("weights and expected returns must be finite")
    if len(tickers) != len(target):
        raise ValueError("tickers must align with weights")
    if float(max_gross) <= 0.0 or float(max_abs_net) < 0.0:
        raise ValueError("portfolio limits are invalid")

    delta = target - previous
    order_costs = estimate_order_execution_costs(
        delta_weights=delta,
        target_weights=target,
        tickers=tickers,
        side_leverage=side_leverage,
        alpha_long=alpha_long,
        alpha_short=alpha_short,
        schedule=schedule,
    )
    expected_edge = float(side_leverage) * delta * mu
    margin = expected_edge - order_costs["cost_return"].to_numpy(dtype=float)
    eligible = (np.abs(delta) > 1e-12) & (margin > 0.0)

    n = len(target)
    objective = np.zeros(2 * n, dtype=float)
    objective[:n] = -np.where(eligible, margin, 0.0)
    a_ub: list[np.ndarray] = []
    b_ub: list[float] = []
    for i in range(n):
        # absolute value of resulting model weight
        row_pos = np.zeros(2 * n, dtype=float)
        row_pos[i] = delta[i]
        row_pos[n + i] = -1.0
        a_ub.append(row_pos)
        b_ub.append(-previous[i])
        row_neg = np.zeros(2 * n, dtype=float)
        row_neg[i] = -delta[i]
        row_neg[n + i] = -1.0
        a_ub.append(row_neg)
        b_ub.append(previous[i])

    gross_row = np.zeros(2 * n, dtype=float)
    gross_row[n:] = 1.0
    a_ub.append(gross_row)
    b_ub.append(float(max_gross))
    net_positive = np.zeros(2 * n, dtype=float)
    net_positive[:n] = delta
    a_ub.append(net_positive)
    b_ub.append(float(max_abs_net) - float(previous.sum()))
    net_negative = np.zeros(2 * n, dtype=float)
    net_negative[:n] = -delta
    a_ub.append(net_negative)
    b_ub.append(float(max_abs_net) + float(previous.sum()))
    lp = linprog(
        objective,
        A_ub=np.asarray(a_ub),
        b_ub=np.asarray(b_ub),
        bounds=[(0.0, 1.0 if eligible[i] else 0.0) for i in range(n)]
        + [(0.0, None) for _ in range(n)],
        method="highs",
    )
    if not lp.success:
        # If previous inventory is outside the requested constraint, a
        # no-trade solution may be infeasible.  Do not silently repair it.
        raise RuntimeError(f"order-selection LP failed: {lp.message}")
    fractions = np.clip(lp.x[:n], 0.0, 1.0)
    chosen = previous + fractions * delta
    chosen[np.abs(chosen) < 1e-13] = 0.0
    if float(np.abs(chosen).sum()) > float(max_gross) + 1e-7:
        raise AssertionError("selected weights exceed gross limit")
    if float(abs(chosen.sum())) > float(max_abs_net) + 1e-7:
        raise AssertionError("selected weights exceed net limit")

    order_costs["expected_return"] = mu
    order_costs["expected_edge_return"] = expected_edge
    order_costs["estimated_net_margin"] = margin
    order_costs["eligible"] = eligible
    order_costs["selected_fraction"] = fractions
    order_costs["selected_delta_weight"] = fractions * delta
    return chosen, order_costs


__all__ = [
    "OrderCostSchedule",
    "estimate_order_execution_costs",
    "expected_gross_returns_from_overlay",
    "select_partial_orders",
]
