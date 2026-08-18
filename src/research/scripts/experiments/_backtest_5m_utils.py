#!/usr/bin/env python3
"""Shared helpers for 5-minute intraday V2 backtest experiment modes.

These were previously duplicated across ``run_v2_backtest_pessimistic.py``,
``run_v2_backtest_realistic.py`` and ``run_v2_backtest_lot_rounding.py``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

JP_TICKERS = [
    "1617.T",
    "1618.T",
    "1619.T",
    "1620.T",
    "1621.T",
    "1622.T",
    "1623.T",
    "1624.T",
    "1625.T",
    "1626.T",
    "1627.T",
    "1628.T",
    "1629.T",
    "1630.T",
    "1631.T",
    "1632.T",
    "1633.T",
]


def _bar_value(bar: pd.Series, field: str, ticker: str) -> float | None:
    """Return a single field/ticker value from a 5m bar, None if missing/NaN."""
    if bar is None:
        return None
    val = bar.get((field, ticker))
    if val is None or (isinstance(val, float) and not np.isfinite(val)) or not pd.notna(val):
        return None
    return float(val)


def _find_bar(
    df_5m: pd.DataFrame,
    date: pd.Timestamp,
    ticker: str,
    time_strs: list[str] | None = None,
    latest_before: str | None = None,
    required_fields: tuple[str, ...] = ("Close", "High", "Low"),
) -> pd.Series | None:
    """Find a 5m bar for a specific ticker.

    - time_strs: iterate candidate times and pick the first bar with a valid
      field (default Close/High/Low, configurable via ``required_fields``).
    - latest_before: pick the latest bar at/before the time with a valid field.
    """
    day_data = df_5m[df_5m.index.date == date.date()]
    if day_data.empty:
        return None

    if time_strs:
        for t in time_strs:
            idx = pd.Timestamp(f"{date.date()} {t}")
            if idx in day_data.index:
                bar = day_data.loc[idx]
                for field in required_fields:
                    if pd.notna(bar.get((field, ticker))):
                        return bar
        return None

    if latest_before:
        cutoff = pd.Timestamp(f"{date.date()} {latest_before}")
        day_data = day_data[day_data.index <= cutoff]
        if day_data.empty:
            return None
        for i in range(len(day_data) - 1, -1, -1):
            bar = day_data.iloc[i]
            for field in required_fields:
                if pd.notna(bar.get((field, ticker))):
                    return bar
        return None

    return day_data.iloc[0]


def _lot_size(ticker: str) -> int:
    from leadlag.data.tickers import lot_size_for

    return lot_size_for(ticker)


def _allocate_lots(
    target_notional: np.ndarray,
    side: np.ndarray,
    price: np.ndarray,
    lot: np.ndarray,
    eff_capital: float,
    gross_limit_mult: float = 2.0,
    rounding: str = "floor",
    reallocate: bool = False,
) -> np.ndarray:
    """Allocate integer lots to approximate target notional weights.

    Parameters:
        target_notional: signed target notional (long positive, short negative)
        side: sign array (long +1, short -1)
        price: execution price per share
        lot: lot size per ticker
        eff_capital: effective capital (capital * side_leverage)
        gross_limit_mult: gross exposure limit as multiple of eff_capital
        rounding: 'floor' or 'nearest'
        reallocate: redistribute residual to other tickers of the same side

    Returns:
        integer share quantities (signed)
    """
    n = len(target_notional)
    q = np.zeros(n, dtype=int)
    gross_limit_notional = gross_limit_mult * eff_capital

    for i in range(n):
        if side[i] == 0 or price[i] <= 0 or lot[i] <= 0:
            continue
        target = abs(target_notional[i])
        lot_price = price[i] * lot[i]
        if rounding == "nearest":
            q_i = int(round(target / lot_price)) * lot[i]
        else:
            q_i = int(target / lot_price) * lot[i]
        q_i = q_i if side[i] > 0 else -q_i
        q[i] = q_i

    def _gross(q_arr: np.ndarray) -> float:
        return float(np.sum(np.abs(q_arr * price)))

    current_gross = _gross(q)
    if current_gross > gross_limit_notional:
        scale = gross_limit_notional / current_gross
        for i in range(n):
            if side[i] == 0 or price[i] <= 0 or lot[i] <= 0:
                continue
            scaled = int(round(abs(q[i]) * scale / lot[i])) * lot[i]
            q[i] = scaled if side[i] > 0 else -scaled

    if reallocate:
        for _ in range(50):
            current_gross = _gross(q)
            capacity = gross_limit_notional - current_gross
            if capacity < 1e-6:
                break
            best_idx = -1
            best_residual = 0.0
            for i in range(n):
                if side[i] == 0 or price[i] <= 0 or lot[i] <= 0:
                    continue
                target = abs(target_notional[i])
                actual = abs(q[i]) * price[i]
                residual = target - actual
                if residual <= 0:
                    continue
                lot_price = price[i] * lot[i]
                if lot_price > capacity:
                    continue
                if residual > best_residual:
                    best_residual = residual
                    best_idx = i
            if best_idx < 0:
                break
            q[best_idx] += lot[best_idx] if side[best_idx] > 0 else -lot[best_idx]

    return q


def price_from_mode(bar: pd.Series, ticker: str, side: int, mode: str) -> float | None:
    """Return a price from the 5m bar based on mode.

    side=+1 long, side=-1 short. mode: 'adverse', 'midpoint', 'close', 'open'.
    """
    high = _bar_value(bar, "High", ticker)
    low = _bar_value(bar, "Low", ticker)
    close = _bar_value(bar, "Close", ticker)
    op = _bar_value(bar, "Open", ticker)

    if mode == "adverse":
        if side > 0:
            return low if low is not None else close
        else:
            return high if high is not None else close
    elif mode == "midpoint":
        if high is not None and low is not None:
            return (high + low) / 2.0
        return close
    elif mode == "close":
        return close
    elif mode == "open":
        return op
    else:
        raise ValueError(f"Unknown price mode: {mode}")


def _load_5m_cost_params(config_path: Path) -> dict[str, float]:
    """Load the V2 cost / financing parameters used by the 5m backtest modes."""
    import yaml

    cfg = yaml.safe_load(open(config_path))
    costs = cfg.get("costs", {})
    side_leverage = float(
        cfg.get("execution", {}).get(
            "side_leverage", cfg.get("portfolio", {}).get("side_leverage", 1.5)
        )
    )
    return {
        "alpha_long": float(costs.get("overnight_alpha_long", 0.75)),
        "alpha_short": float(costs.get("overnight_alpha_short", 0.5)),
        "fin_annual": float(costs.get("buy_interest_annual", 0.025)),
        "borrow_annual": float(costs.get("borrow_fee_annual", 0.0115)),
        "rev_bps": float(costs.get("reverse_fee_bps", 2.0)),
        "slip_bps": float(costs.get("slippage_bps_per_side", 5.0)),
        "side_leverage": side_leverage,
    }
