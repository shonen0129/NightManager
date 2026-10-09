"""Point-in-time research sizing for overnight inventory.

The policy uses only signal weights and realized overnight marks available by
the current close. The following session's signal and return are deliberately
not accepted by the sizing function; they are reserved for ex-post replay and
inventory-reuse diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AdaptiveCarryConfig:
    """Fixed research-policy assumptions, chosen before the OOS replay."""

    lookback_sessions: int = 252
    transition_prior_same: float = 1.0
    transition_prior_reverse: float = 1.0
    transition_prior_flat: float = 1.0
    overlap_prior_alpha: float = 1.0
    overlap_prior_beta: float = 1.0
    reversal_exit_cost_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if self.lookback_sessions < 1:
            raise ValueError("lookback_sessions must be positive")
        values = (
            self.transition_prior_same,
            self.transition_prior_reverse,
            self.transition_prior_flat,
            self.overlap_prior_alpha,
            self.overlap_prior_beta,
            self.reversal_exit_cost_multiplier,
        )
        if not np.isfinite(values).all() or any(value <= 0 for value in values):
            raise ValueError("prior parameters and reversal cost multiplier must be positive and finite")


def adaptive_carry_masks(
    *,
    weights: np.ndarray,
    gap_returns: np.ndarray,
    sim_dates: pd.DatetimeIndex,
    slip: float,
    financing_annual: float,
    borrow_annual: float,
    reverse_fee_bps: float,
    config: AdaptiveCarryConfig = AdaptiveCarryConfig(),
) -> tuple[np.ndarray, pd.DataFrame]:
    """Return per-date/per-ticker carry fractions and their PIT diagnostics.

    At row ``i``, transition probabilities use only historical pairs whose
    destination is at or before ``i - 1``. The current weight sign is known by
    the current close. Gap-return history includes row ``i`` (the current
    session's close-to-09:10 mark, already observed that morning). No target
    return, next-session weight, or next-session gap enters this function.

    The expected-value score is

        p_same * overlap * (close + repurchase slippage)
        + expected signed overnight return
        - calendar-day financing/borrow/reverse fees
        - expected next-session exit slippage,

    divided by the two-sided slippage saved by a fully reused unit and clipped
    to [0, 1]. The reversal term charges one carry unwind leg on an expected
    reversal; opening the new opposite target remains in the replay's actual
    execution-volume/slippage ledger.
    """
    weight_array = np.asarray(weights, dtype=float)
    gap_array = np.asarray(gap_returns, dtype=float)
    dates = pd.DatetimeIndex(sim_dates).normalize()
    if weight_array.ndim != 2 or gap_array.shape != weight_array.shape:
        raise ValueError("weights and gap_returns must have the same 2-D shape")
    if len(dates) != len(weight_array):
        raise ValueError("sim_dates length must match the number of weight rows")
    if not np.isfinite(weight_array).all():
        raise ValueError("weights must be finite")
    if not np.isfinite(slip) or slip <= 0:
        raise ValueError("slip must be positive and finite")
    rates = (financing_annual, borrow_annual, reverse_fee_bps)
    if not np.isfinite(rates).all() or any(rate < 0 for rate in rates):
        raise ValueError("annual rates and reverse fees must be finite and non-negative")

    n_days, n_assets = weight_array.shape
    masks = np.zeros_like(weight_array)
    diagnostics: list[dict[str, object]] = []
    prior_total = (
        config.transition_prior_same
        + config.transition_prior_reverse
        + config.transition_prior_flat
    )
    reverse_daily = reverse_fee_bps / 10_000.0

    for i in range(n_days):
        # Only pairs ending by i-1 are known at the current close.
        first_origin = max(0, i - config.lookback_sessions - 1)
        origin_rows = np.arange(first_origin, max(first_origin, i - 1), dtype=int)
        destination_rows = origin_rows + 1
        days_to_next = (
            int((dates[i + 1] - dates[i]).days) if i + 1 < n_days else 0
        )

        for j in range(n_assets):
            current_weight = float(weight_array[i, j])
            current_direction = float(np.sign(current_weight))
            row: dict[str, object] = {
                "date": dates[i],
                "ticker_index": j,
                "current_weight": current_weight,
                "lookback_sessions": config.lookback_sessions,
                "transition_observations": 0,
                "same_probability": np.nan,
                "reverse_probability": np.nan,
                "flat_probability": np.nan,
                "expected_overlap_fraction": np.nan,
                "expected_repurchase_cost_saved": 0.0,
                "expected_overnight_pnl_per_notional": 0.0,
                "expected_holding_cost_per_notional": 0.0,
                "expected_reversal_exit_cost_per_notional": 0.0,
                "expected_flat_exit_cost_per_notional": 0.0,
                "carry_alpha": 0.0,
            }
            if current_direction == 0.0 or i == n_days - 1:
                diagnostics.append(row)
                continue

            origins = weight_array[origin_rows, j]
            destinations = weight_array[destination_rows, j]
            relevant = np.sign(origins) == current_direction
            historical_origins = origins[relevant]
            historical_destinations = destinations[relevant]
            destination_signs = np.sign(historical_destinations)
            same_count = int(np.sum(destination_signs == current_direction))
            reverse_count = int(np.sum(destination_signs == -current_direction))
            flat_count = int(np.sum(destination_signs == 0.0))
            observations = same_count + reverse_count + flat_count
            denominator = observations + prior_total
            p_same = (same_count + config.transition_prior_same) / denominator
            p_reverse = (reverse_count + config.transition_prior_reverse) / denominator
            p_flat = (flat_count + config.transition_prior_flat) / denominator

            same_mask = destination_signs == current_direction
            if same_count:
                source_size = np.abs(historical_origins[same_mask])
                target_size = np.abs(historical_destinations[same_mask])
                overlap_values = np.minimum(source_size, target_size) / source_size
                overlap_fraction = float(
                    (overlap_values.sum() + config.overlap_prior_alpha)
                    / (same_count + config.overlap_prior_alpha + config.overlap_prior_beta)
                )
            else:
                overlap_fraction = config.overlap_prior_alpha / (
                    config.overlap_prior_alpha + config.overlap_prior_beta
                )

            gap_start = max(0, i - config.lookback_sessions + 1)
            historical_gaps = gap_array[gap_start : i + 1, j]
            finite_gaps = historical_gaps[np.isfinite(historical_gaps)]
            mean_gap = float(finite_gaps.mean()) if len(finite_gaps) else 0.0
            expected_overnight_pnl = current_direction * mean_gap
            if current_direction > 0:
                holding_cost = financing_annual / 365.0 * days_to_next
            else:
                holding_cost = (borrow_annual / 365.0 + reverse_daily) * days_to_next
            reversal_exit_cost = (
                p_reverse * slip * config.reversal_exit_cost_multiplier
            )
            flat_exit_cost = p_flat * slip
            expected_repurchase_saved = p_same * overlap_fraction * 2.0 * slip
            edge = (
                expected_repurchase_saved
                + expected_overnight_pnl
                - holding_cost
                - reversal_exit_cost
                - flat_exit_cost
            )
            alpha = float(np.clip(edge / (2.0 * slip), 0.0, 1.0))
            masks[i, j] = alpha
            row.update(
                {
                    "transition_observations": observations,
                    "same_probability": p_same,
                    "reverse_probability": p_reverse,
                    "flat_probability": p_flat,
                    "expected_overlap_fraction": overlap_fraction,
                    "expected_repurchase_cost_saved": expected_repurchase_saved,
                    "expected_overnight_pnl_per_notional": expected_overnight_pnl,
                    "expected_holding_cost_per_notional": holding_cost,
                    "expected_reversal_exit_cost_per_notional": reversal_exit_cost,
                    "expected_flat_exit_cost_per_notional": flat_exit_cost,
                    "carry_alpha": alpha,
                }
            )
            diagnostics.append(row)

    return masks, pd.DataFrame(diagnostics)


def realized_inventory_reuse(
    *,
    weights: np.ndarray,
    alpha_masks: np.ndarray,
) -> dict[str, float]:
    """Measure ex-post same-direction carried notional used next session."""
    weight_array = np.asarray(weights, dtype=float)
    masks = np.asarray(alpha_masks, dtype=float)
    if weight_array.ndim != 2 or masks.shape != weight_array.shape:
        raise ValueError("weights and alpha_masks must have matching 2-D shapes")
    if len(weight_array) < 2:
        return {"carried_notional": 0.0, "reused_notional": 0.0, "reuse_ratio": 0.0}
    carried = masks[:-1] * weight_array[:-1]
    next_weights = weight_array[1:]
    same_direction = np.sign(carried) == np.sign(next_weights)
    reused = np.where(
        same_direction,
        np.minimum(np.abs(carried), np.abs(next_weights)),
        0.0,
    )
    carried_notional = float(np.abs(carried).sum())
    reused_notional = float(reused.sum())
    return {
        "carried_notional": carried_notional,
        "reused_notional": reused_notional,
        "reuse_ratio": reused_notional / carried_notional if carried_notional else 0.0,
    }


def attribute_inventory_transitions(
    *,
    weights: np.ndarray,
    target_returns: np.ndarray,
    gap_returns: np.ndarray,
    alpha_masks: np.ndarray,
    sim_dates: pd.DatetimeIndex,
    calendar_days: np.ndarray,
    slip: float,
    financing_daily: float,
    borrow_daily: float,
    reverse_daily: float,
    side_leverage: float,
) -> pd.DataFrame:
    """Build an additive ticker/day attribution of the corrected inventory ledger.

    The returned rows classify the current ticker's next-session signal as a
    continuation, reversal, or flat transition. Every return and cost column
    is in portfolio return-fraction units after side leverage. Summing each
    numeric ledger column over ticker/day must reconcile to the corresponding
    ``simulate_daily_pnl`` series.
    """
    weight_array = np.asarray(weights, dtype=float)
    target_array = np.asarray(target_returns, dtype=float)
    gap_array = np.asarray(gap_returns, dtype=float)
    alpha_array = np.asarray(alpha_masks, dtype=float)
    days_array = np.asarray(calendar_days, dtype=float)
    dates = pd.DatetimeIndex(sim_dates).normalize()
    if (
        weight_array.ndim != 2
        or target_array.shape != weight_array.shape
        or gap_array.shape != weight_array.shape
        or alpha_array.shape != weight_array.shape
    ):
        raise ValueError("weights, returns, and alpha_masks must have matching 2-D shapes")
    if len(dates) != len(weight_array) or days_array.shape != (len(weight_array),):
        raise ValueError("date and calendar arrays must align with the weight rows")
    if not np.isfinite(weight_array).all() or not np.isfinite(alpha_array).all():
        raise ValueError("weights and alpha_masks must be finite")
    if np.any((alpha_array < 0.0) | (alpha_array > 1.0)):
        raise ValueError("alpha_masks must be within [0, 1]")
    if not np.isfinite(days_array).all() or np.any(days_array < 0.0):
        raise ValueError("calendar_days must be finite and non-negative")
    if not dates.is_monotonic_increasing or dates.has_duplicates:
        raise ValueError("sim_dates must be strictly increasing and unique")
    scalar_costs = (slip, financing_daily, borrow_daily, reverse_daily, side_leverage)
    if not np.isfinite(scalar_costs).all() or any(value < 0 for value in scalar_costs):
        raise ValueError("cost rates and side_leverage must be finite and non-negative")

    n_days, n_assets = weight_array.shape
    held = alpha_array * weight_array
    held_previous = np.zeros_like(held)
    if n_days > 1:
        held_previous[1:] = held[:-1]
    opening_flow = np.abs(weight_array - held_previous)
    closing_flow = np.abs(weight_array - held)
    execution_volume = side_leverage * (opening_flow + closing_flow)
    slip_cost = slip * execution_volume
    close_slippage_delta_vs_flat = slip * side_leverage * (
        closing_flow - np.abs(weight_array)
    )
    next_open_slippage_delta_vs_flat = np.zeros_like(weight_array)
    if n_days > 1:
        next_open_flow = np.abs(weight_array[1:] - held[:-1])
        next_open_slippage_delta_vs_flat[:-1] = slip * side_leverage * (
            next_open_flow - np.abs(weight_array[1:])
        )
    transition_slippage_delta_vs_flat = (
        close_slippage_delta_vs_flat + next_open_slippage_delta_vs_flat
    )
    financing_cost = (
        side_leverage
        * alpha_array
        * np.maximum(weight_array, 0.0)
        * financing_daily
        * days_array[:, None]
    )
    borrow_cost = (
        side_leverage
        * alpha_array
        * np.maximum(-weight_array, 0.0)
        * borrow_daily
        * days_array[:, None]
    )
    reverse_cost = (
        side_leverage
        * alpha_array
        * np.maximum(-weight_array, 0.0)
        * reverse_daily
        * days_array[:, None]
    )
    intraday_gross = side_leverage * np.where(
        weight_array != 0.0, weight_array * target_array, 0.0
    )
    overnight = np.zeros_like(weight_array)
    if n_days > 1:
        next_gap = gap_array[1:]
        carry = held[:-1]
        overnight[:-1] = side_leverage * np.where(carry != 0.0, carry * next_gap, 0.0)
    gross = intraday_gross + overnight
    total_cost = slip_cost + financing_cost + borrow_cost + reverse_cost
    next_weights = np.zeros_like(weight_array)
    if n_days > 1:
        next_weights[:-1] = weight_array[1:]
    current_direction = np.sign(weight_array)
    next_direction = np.sign(next_weights)
    classes = np.full(weight_array.shape, "current_flat", dtype=object)
    if n_days:
        classes[-1] = "terminal_final_close"
    active = current_direction[:-1] != 0.0
    classes[:-1][active & (next_direction[:-1] == -current_direction[:-1])] = "next_signal_reversal"
    classes[:-1][active & (next_direction[:-1] == current_direction[:-1])] = "next_signal_same_direction"
    classes[:-1][active & (next_direction[:-1] == 0.0)] = "next_signal_flat"
    reused_notional = np.where(
        (current_direction != 0.0) & (current_direction == next_direction),
        np.minimum(np.abs(held), np.abs(next_weights)),
        0.0,
    )

    frame = pd.DataFrame(
        {
            "date": np.repeat(dates.to_numpy(), n_assets),
            "ticker_index": np.tile(np.arange(n_assets), n_days),
            "current_weight": weight_array.reshape(-1),
            "next_weight": next_weights.reshape(-1),
            "transition_class": classes.reshape(-1),
            "intraday_gross": intraday_gross.reshape(-1),
            "overnight_pnl": overnight.reshape(-1),
            "gross_pnl": gross.reshape(-1),
            "slippage_cost": slip_cost.reshape(-1),
            "financing_cost": financing_cost.reshape(-1),
            "borrow_cost": borrow_cost.reshape(-1),
            "reverse_cost": reverse_cost.reshape(-1),
            "total_cost": total_cost.reshape(-1),
            "net_pnl": (gross - total_cost).reshape(-1),
            "close_slippage_delta_vs_flat": close_slippage_delta_vs_flat.reshape(-1),
            "next_open_slippage_delta_vs_flat": next_open_slippage_delta_vs_flat.reshape(-1),
            "transition_slippage_delta_vs_flat": transition_slippage_delta_vs_flat.reshape(-1),
            "execution_volume": execution_volume.reshape(-1),
            "carried_notional": np.abs(held).reshape(-1),
            "reused_notional": reused_notional.reshape(-1),
        }
    )
    numeric = frame.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("ticker/day attribution contains a non-finite value")
    return frame
