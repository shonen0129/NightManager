"""Capital allocation: convert weights to actual quantities."""

from __future__ import annotations

import logging

import numpy as np

from leadlag.core.types import CapitalAllocation

logger = logging.getLogger(__name__)
DEFAULT_SIDE_LEVERAGE = 1.5

LOT_SIZE_BY_TICKER = {
    "1629.T": 10,
    "1629": 10,
}


def _lot_size_for_ticker(ticker: str) -> int:
    lot_size = LOT_SIZE_BY_TICKER.get(ticker)
    if lot_size is None and ticker.endswith(".T"):
        lot_size = LOT_SIZE_BY_TICKER.get(ticker.replace(".T", ""))
    if lot_size is None or lot_size < 1:
        return 1
    return int(lot_size)


def allocate_capital(
    weights: np.ndarray,
    tickers: list[str],
    open_prices: dict[str, float],
    max_capital: float,
    max_net_exposure: float | None = None,
    side_leverage: float = DEFAULT_SIDE_LEVERAGE,
) -> CapitalAllocation:
    """Allocate capital to both BUY and SELL positions from target weights.

    Gross allocation budget is split by absolute side weights.
    Capital is interpreted as equity: gross budget scales by gross_abs_sum
    and side leverage. With the default side leverage 1.5,
    1x long + 1x short implies ~3x gross notional.
    Returns quantities and allocated amounts for each ticker.

    Args:
        weights: Target weight array of shape (N_J,)
        tickers: List of ticker strings
        open_prices: Dict mapping ticker -> opening price
        max_capital: Total available capital in JPY
        max_net_exposure: Optional absolute net-notional cap as a ratio of equity

    Returns:
        CapitalAllocation with quantities, allocated amounts, and budgets
    """
    n = len(tickers)
    quantities = np.zeros(n, dtype=int)
    allocated = np.zeros(n, dtype=float)

    if max_capital <= 0:
        return CapitalAllocation(
            quantities=quantities,
            allocated_amounts=allocated,
            buy_budget=0.0,
            sell_budget=0.0,
        )

    buy_indices = np.where(weights > 1e-12)[0]
    sell_indices = np.where(weights < -1e-12)[0]

    buy_abs_sum = float(np.sum(weights[buy_indices])) if len(buy_indices) > 0 else 0.0
    sell_abs_sum = float(np.sum(-weights[sell_indices])) if len(sell_indices) > 0 else 0.0
    gross_abs_sum = buy_abs_sum + sell_abs_sum

    if gross_abs_sum <= 0:
        return CapitalAllocation(
            quantities=quantities,
            allocated_amounts=allocated,
            buy_budget=0.0,
            sell_budget=0.0,
        )

    leverage = float(side_leverage)
    if not np.isfinite(leverage) or leverage < 0.0:
        leverage = 0.0

    gross_budget = float(max_capital) * float(gross_abs_sum) * leverage
    buy_budget = gross_budget * (buy_abs_sum / gross_abs_sum)
    sell_budget = gross_budget * (sell_abs_sum / gross_abs_sum)

    def _allocate_side(indices: np.ndarray, abs_weights: np.ndarray, side_budget: float):
        if len(indices) == 0 or side_budget <= 0:
            return

        side_sum = float(np.sum(abs_weights))
        if side_sum <= 0:
            return

        idx_sorted = indices[np.argsort(-abs_weights)]
        for idx in idx_sorted:
            tk = tickers[idx]
            price = float(open_prices.get(tk, 0))
            if price <= 0:
                logger.warning(
                    f"Skipping {tk}: open price={price} is non-positive "
                    "(possible data fetch failure)"
                )
                continue

            target_alloc = side_budget * (abs(weights[idx]) / side_sum)
            lot_size = _lot_size_for_ticker(tk)
            lot_price = price * lot_size
            if lot_price <= 0:
                continue
            q_lots = int(np.floor(target_alloc / lot_price))
            q = max(0, q_lots * lot_size)
            quantities[idx] = q
            allocated[idx] = float(q * price)

    _allocate_side(buy_indices, np.abs(weights[buy_indices]), buy_budget)
    _allocate_side(sell_indices, np.abs(weights[sell_indices]), sell_budget)

    # Balance allocated net exposure drift caused by integer share rounding.
    if max_net_exposure is not None and float(max_capital) > 0:
        net_limit = float(max_net_exposure) * float(max_capital)
        if net_limit < 0:
            net_limit = 0.0

        def _side_totals() -> tuple[float, float]:
            buy_total = float(np.sum(allocated[buy_indices])) if len(buy_indices) else 0.0
            sell_total = float(np.sum(allocated[sell_indices])) if len(sell_indices) else 0.0
            return buy_total, sell_total

        def _price_at(idx: int) -> float:
            tk = tickers[idx]
            return float(open_prices.get(tk, 0.0))

        def _lot_size_at(idx: int) -> int:
            return _lot_size_for_ticker(tickers[idx])

        def _lot_price_at(idx: int) -> float:
            return _price_at(idx) * _lot_size_at(idx)

        buy_total, sell_total = _side_totals()
        net = buy_total - sell_total

        remaining_buy = float(buy_budget - buy_total)
        remaining_sell = float(sell_budget - sell_total)

        max_steps = 500
        steps = 0
        while abs(net) > net_limit + 1e-9 and steps < max_steps:
            steps += 1
            best_move = None  # (abs_new_net, kind, idx, price)

            if net > 0:
                for idx in sell_indices:
                    price = _lot_price_at(int(idx))
                    if price <= 0:
                        continue
                    if price <= remaining_sell + 1e-9:
                        new_net = net - price
                        cand = (abs(new_net), "add_sell", int(idx), price)
                        if best_move is None or cand < best_move:
                            best_move = cand
                if best_move is None:
                    for idx in buy_indices:
                        idx = int(idx)
                        lot_size = _lot_size_at(idx)
                        if quantities[idx] < lot_size:
                            continue
                        price = _lot_price_at(idx)
                        if price <= 0:
                            continue
                        new_net = net - price
                        cand = (abs(new_net), "remove_buy", idx, price)
                        if best_move is None or cand < best_move:
                            best_move = cand
            else:
                for idx in buy_indices:
                    price = _lot_price_at(int(idx))
                    if price <= 0:
                        continue
                    if price <= remaining_buy + 1e-9:
                        new_net = net + price
                        cand = (abs(new_net), "add_buy", int(idx), price)
                        if best_move is None or cand < best_move:
                            best_move = cand
                if best_move is None:
                    for idx in sell_indices:
                        idx = int(idx)
                        lot_size = _lot_size_at(idx)
                        if quantities[idx] < lot_size:
                            continue
                        price = _lot_price_at(idx)
                        if price <= 0:
                            continue
                        new_net = net + price
                        cand = (abs(new_net), "remove_sell", idx, price)
                        if best_move is None or cand < best_move:
                            best_move = cand

            if best_move is None:
                break

            _, kind, idx, price = best_move
            lot_size = _lot_size_at(idx)
            if kind == "add_sell":
                quantities[idx] += lot_size
                allocated[idx] += price
                remaining_sell -= price
            elif kind == "remove_buy":
                quantities[idx] -= lot_size
                allocated[idx] -= price
                remaining_buy += price
            elif kind == "add_buy":
                quantities[idx] += lot_size
                allocated[idx] += price
                remaining_buy -= price
            elif kind == "remove_sell":
                quantities[idx] -= lot_size
                allocated[idx] -= price
                remaining_sell += price

            buy_total, sell_total = _side_totals()
            net = buy_total - sell_total

    return CapitalAllocation(
        quantities=quantities.astype(int),
        allocated_amounts=allocated,
        buy_budget=float(buy_budget),
        sell_budget=float(sell_budget),
        gross_budget=float(gross_budget),
    )
