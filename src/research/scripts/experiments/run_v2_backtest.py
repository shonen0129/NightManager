#!/usr/bin/env python3
"""Consolidated V2 backtest experiment runner.

Replaces the previous ``run_v2_backtest_exact_production.py``,
``run_v2_backtest_pessimistic.py``, ``run_v2_backtest_realistic.py``,
``run_v2_backtest_theoretical_5m.py`` and ``run_v2_backtest_lot_rounding.py``
with a single entry point selected by ``--mode``.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.cache import load_df_exec_from_local_cache, load_intraday_cache
from leadlag.data.preprocessor import compute_jp_target_returns
from leadlag.execution.backtester import BacktestEngine
from leadlag.execution.config import load_config_from_yaml
from research.experiment_registry import Decision
from research.experiment_utils import record_backtest_experiment
from research.scripts.experiments._backtest_5m_utils import (
    JP_TICKERS,
    _allocate_lots,
    _bar_value,
    _find_bar,
    _load_5m_cost_params,
    _lot_size,
    price_from_mode,
)


def _default_input_dir() -> Path:
    return ROOT / "var" / "results" / "v2_backtest_exact_production_20260729"


def _run_exact(args: argparse.Namespace) -> None:
    """Run the V2 exact-production backtest with optional overlay."""
    config_path = ROOT / args.config
    gap_dir = ROOT / args.gap_dir
    overlay_model_dir = ROOT / args.overlay_model_dir
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    app_config = load_config_from_yaml(config_path)

    df_exec = load_df_exec_from_local_cache()
    logger.info("df_exec: %s rows", len(df_exec))

    results = BacktestEngine.run_v2_backtest(
        cfg=app_config,
        gap_input_dir=gap_dir,
        df_exec=df_exec,
        start_date=args.start_date,
        end_date=args.end_date,
        side_leverage=args.side_leverage,
        overlay_model_dir=overlay_model_dir,
        n_jobs=args.n_jobs,
    )

    results["weights"].to_csv(output_dir / "daily_weights.csv")
    results["daily_returns"].to_csv(output_dir / "daily_net_returns.csv", header=["net_return"])
    results["daily_returns_gross"].to_csv(
        output_dir / "daily_gross_returns.csv", header=["gross_return"]
    )
    results["equity_curve"].to_csv(output_dir / "daily_equity_curve.csv", header=["equity"])
    results["drawdown"].to_csv(output_dir / "daily_drawdown.csv", header=["drawdown"])
    results["daily_turnover"].to_csv(output_dir / "daily_turnover.csv", header=["turnover"])
    results["daily_gross_exps"].to_csv(output_dir / "daily_gross.csv", header=["gross"])
    results["daily_costs"].to_csv(output_dir / "daily_costs_total.csv", header=["cost"])
    results["daily_slip_costs"].to_csv(output_dir / "daily_costs_slip.csv", header=["slip_cost"])
    results["daily_financing_costs"].to_csv(
        output_dir / "daily_costs_financing.csv", header=["financing_cost"]
    )
    results["daily_borrow_costs"].to_csv(
        output_dir / "daily_costs_borrow.csv", header=["borrow_cost"]
    )
    results["daily_reverse_costs"].to_csv(
        output_dir / "daily_costs_reverse.csv", header=["reverse_cost"]
    )
    results["daily_overnight_returns"].to_csv(
        output_dir / "daily_overnight_returns.csv", header=["overnight_return"]
    )
    results["daily_fallback"].to_csv(output_dir / "daily_fallback.csv", header=["fallback"])

    returns = results["daily_returns"]
    fb = results["daily_fallback"]
    valid = returns[~fb]
    n_valid = len(valid)
    n_fallback = int(fb.sum())
    total_ret = float(np.sum(valid))
    mean_ret = float(np.mean(valid)) if n_valid > 0 else 0.0
    std_ret = float(np.std(valid, ddof=1)) if n_valid > 1 else 0.0
    sharpe = mean_ret / std_ret * np.sqrt(252) if std_ret > 1e-8 else 0.0
    wealth = results["equity_curve"]
    mdd = float(results["drawdown"].min()) if len(results["drawdown"]) > 0 else 0.0
    avg_turnover = float(np.mean(results["daily_turnover"][~fb])) if n_valid > 0 else 0.0
    avg_gross = float(np.mean(results["daily_gross_exps"][~fb])) if n_valid > 0 else 0.0
    fb_rate = n_fallback / len(returns) * 100 if len(returns) > 0 else 0.0
    final_wealth = float(wealth.iloc[-1]) if len(wealth) > 0 else 1.0
    cagr = (final_wealth ** (252 / n_valid) - 1) * 100 if n_valid > 0 else 0.0

    print("\n" + "=" * 60)
    print("=== V2 Exact-Production Backtest (with overlay) ===")
    print("=" * 60)
    print(f"  Period:        {returns.index[0].date()} -> {returns.index[-1].date()}")
    print(f"  Total days:    {len(returns)}")
    print(f"  Success:       {n_valid}")
    print(f"  Fallback:      {n_fallback} ({fb_rate:.1f}%)")
    print(f"  Sharpe:        {sharpe:.4f}")
    print(f"  Total Return:  {total_ret*100:.2f}%")
    print(f"  CAGR:          {cagr:.2f}%")
    print(f"  Final Wealth:  {final_wealth:,.2f}x")
    print(f"  AR (ann):      {mean_ret*252*100:.2f}%")
    print(f"  Vol (ann):     {std_ret*np.sqrt(252)*100:.2f}%")
    print(f"  Max DD:        {mdd*100:.2f}%")
    print(f"  Avg Turnover:  {avg_turnover:.4f}")
    print(f"  Avg Gross:     {avg_gross:.4f}")
    if n_valid > 0:
        print(f"  Avg Cost/day:  {float(np.mean(results['daily_costs'][~fb]))*10000:.2f} bps")
    print("=" * 60)

    summary = {
        "period": f"{returns.index[0].date()} -> {returns.index[-1].date()}",
        "total_days": int(len(returns)),
        "fallback_days": n_fallback,
        "fallback_rate_pct": fb_rate,
        "sharpe": float(sharpe),
        "total_return_pct": float(total_ret * 100),
        "cagr_pct": float(cagr),
        "final_wealth": float(final_wealth),
        "annualized_return_pct": float(mean_ret * 252 * 100),
        "annualized_volatility_pct": float(std_ret * np.sqrt(252) * 100),
        "max_drawdown_pct": float(mdd * 100),
        "avg_turnover": float(avg_turnover),
        "avg_gross_exposure": float(avg_gross),
        "side_leverage": args.side_leverage,
        "overlay_model_dir": str(overlay_model_dir),
        "gap_dir": str(gap_dir),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    record_backtest_experiment(
        name=Path(__file__).stem,
        hypothesis="V2 exact-production backtest (overlay enabled) reproduces live production behavior.",
        app_config=app_config,
        results=results,
        decision=Decision.PENDING,
    )


def _run_pessimistic(args: argparse.Namespace) -> None:
    """Run the 5m pessimistic (adverse price + integer lots) backtest."""
    config_path = ROOT / args.config
    params = _load_5m_cost_params(config_path)
    alpha_long = params["alpha_long"]
    alpha_short = params["alpha_short"]
    fin_annual = params["fin_annual"]
    borrow_annual = params["borrow_annual"]
    rev_bps = params["rev_bps"]
    slip_bps = params["slip_bps"]
    side_leverage = params["side_leverage"]
    initial_capital = float(args.capital)

    financing_daily = fin_annual / 365.0
    borrow_daily = borrow_annual / 365.0
    reverse_daily = rev_bps / 10000.0
    slip = slip_bps / 10000.0

    bt_dir = ROOT / args.input_dir
    w_df, bt_returns, bt_equity = _load_base_backtest(bt_dir)

    df_5m = load_intraday_cache("5m")
    if df_5m is None or df_5m.empty:
        raise FileNotFoundError("5m intraday cache not found")

    five_dates = sorted(pd.Series(df_5m.index.date).unique())
    five_dates = pd.to_datetime(five_dates)
    overlap = w_df.index.intersection(five_dates)
    if len(overlap) == 0:
        raise ValueError("No overlapping dates between theoretical weights and 5m data")

    records = []
    capital = initial_capital
    w_prev = np.zeros(len(JP_TICKERS))
    equity = 1.0

    required_fields = ("Close", "Open", "High", "Low")

    for i, trade_dt in enumerate(overlap):
        w_t = w_df.loc[trade_dt].values.astype(float)

        if np.abs(w_t).sum() < 1e-8:
            records.append({
                "trade_date": trade_dt.strftime("%Y-%m-%d"),
                "gross": 0.0,
                "net": 0.0,
                "gross_ret": 0.0,
                "overnight_ret": 0.0,
                "slip_cost": 0.0,
                "fin_cost": 0.0,
                "borrow_cost": 0.0,
                "reverse_cost": 0.0,
                "cost": 0.0,
                "net_ret": 0.0,
                "equity": equity,
                "capital": capital,
                "turnover": 0.0,
                "positions_count": 0,
            })
            w_prev = np.zeros(len(JP_TICKERS))
            continue

        eff_capital = capital * side_leverage

        quantities = np.zeros(len(JP_TICKERS), dtype=int)
        notionals = np.zeros(len(JP_TICKERS))
        prices_entry = np.zeros(len(JP_TICKERS))
        prices_close = np.zeros(len(JP_TICKERS))
        valid_tk = []

        for j, tk in enumerate(JP_TICKERS):
            side = int(np.sign(w_t[j]))
            if side == 0:
                continue

            entry_bar = _find_bar(
                df_5m,
                trade_dt,
                tk,
                time_strs=["09:10:00", "09:05:00", "09:00:00"],
                required_fields=required_fields,
            )
            close_bar = _find_bar(
                df_5m, trade_dt, tk, latest_before="15:00:00", required_fields=required_fields
            )
            if entry_bar is None or close_bar is None:
                continue

            lot = _lot_size(tk)
            if side > 0:
                p_entry = _bar_value(entry_bar, "High", tk)
                p_close = _bar_value(close_bar, "Low", tk)
            else:
                p_entry = _bar_value(entry_bar, "Low", tk)
                p_close = _bar_value(close_bar, "High", tk)

            if p_entry is None or p_close is None or p_entry <= 0 or p_close <= 0:
                continue

            target_notional = abs(w_t[j]) * eff_capital
            q = int(target_notional / (p_entry * lot)) * lot
            if q < lot:
                continue
            q = q if side > 0 else -q

            quantities[j] = q
            notionals[j] = q * p_entry
            prices_entry[j] = p_entry
            prices_close[j] = p_close
            valid_tk.append(j)

        w_actual = np.where(notionals != 0, notionals / eff_capital, 0.0)

        r_target = np.zeros(len(JP_TICKERS))
        for j in valid_tk:
            side = int(np.sign(w_t[j]))
            p_entry = prices_entry[j]
            p_close = prices_close[j]
            if side > 0:
                r_target[j] = (p_close / p_entry) - 1.0
            else:
                r_target[j] = (p_entry / p_close) - 1.0

        overnight_ret = 0.0
        if i + 1 < len(overlap):
            next_dt = overlap[i + 1]
            r_gap = np.zeros(len(JP_TICKERS))
            for j in valid_tk:
                tk = JP_TICKERS[j]
                side = int(np.sign(w_t[j]))
                next_bar = _find_bar(
                    df_5m,
                    next_dt,
                    tk,
                    time_strs=["09:00:00", "09:05:00", "09:10:00"],
                    required_fields=required_fields,
                )
                if next_bar is None:
                    continue
                if side > 0:
                    p_next = _bar_value(next_bar, "High", tk)
                else:
                    p_next = _bar_value(next_bar, "Low", tk)
                p_prev_close = prices_close[j]
                if p_next is None or p_prev_close is None or p_prev_close <= 0:
                    continue
                r_gap[j] = (
                    (p_next / p_prev_close) - 1.0
                    if side > 0
                    else (1.0 - (p_next / p_prev_close))
                )

            alpha_mask = np.where(w_actual > 0, alpha_long, np.where(w_actual < 0, alpha_short, 0.0))
            overnight_ret = side_leverage * float(np.sum(alpha_mask * w_actual * r_gap))

        calendar_days = 1
        if i + 1 < len(overlap):
            calendar_days = (overlap[i + 1] - trade_dt).days

        gross_ret = side_leverage * float(np.sum(w_actual * r_target))
        alpha_mask = np.where(w_actual > 0, alpha_long, np.where(w_actual < 0, alpha_short, 0.0))

        turnover = float(np.sum(np.abs(w_actual - w_prev)) / 2.0)
        slip_cost = side_leverage * slip * (
            2.0 * np.sum((1.0 - alpha_mask) * np.abs(w_actual))
            + np.sum(alpha_mask * np.abs(w_actual - w_prev) / 2.0)
        )
        held_long = float(np.sum(alpha_mask * np.maximum(w_actual, 0.0)))
        held_short = float(np.sum(alpha_mask * np.maximum(-w_actual, 0.0)))
        fin_cost = side_leverage * held_long * financing_daily * calendar_days
        borrow_cost = side_leverage * held_short * borrow_daily * calendar_days
        reverse_cost = side_leverage * held_short * reverse_daily * calendar_days
        cost = slip_cost + fin_cost + borrow_cost + reverse_cost

        net_ret = gross_ret + overnight_ret - cost
        equity *= 1.0 + net_ret
        capital *= 1.0 + net_ret

        records.append({
            "trade_date": trade_dt.strftime("%Y-%m-%d"),
            "gross": float(np.sum(np.abs(w_actual))),
            "net": float(np.sum(w_actual)),
            "gross_ret": gross_ret,
            "overnight_ret": overnight_ret,
            "slip_cost": slip_cost,
            "fin_cost": fin_cost,
            "borrow_cost": borrow_cost,
            "reverse_cost": reverse_cost,
            "cost": cost,
            "net_ret": net_ret,
            "equity": equity,
            "capital": capital,
            "turnover": turnover,
            "positions_count": np.sum(quantities != 0),
        })

        w_prev = w_actual

    df = pd.DataFrame(records)
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    df.set_index("trade_date").to_csv(out_dir / "daily_results.csv")

    theory_slice = bt_returns.loc[overlap]
    theory_equity_slice = bt_equity.loc[overlap]
    theory_cumret = theory_equity_slice.iloc[-1] / theory_equity_slice.iloc[0] - 1.0

    pess_cumret = equity - 1.0
    pess_returns = pd.Series(df["net_ret"].values, index=overlap)
    pess_mean = float(pess_returns.mean())
    pess_std = float(pess_returns.std(ddof=1))
    pess_sharpe = pess_mean / pess_std * np.sqrt(252) if pess_std > 1e-8 else 0.0
    pess_ar = pess_mean * 252 * 100
    pess_vol = pess_std * np.sqrt(252) * 100
    pess_mdd = float(
        (pd.Series(df["equity"].values, index=overlap)
         / pd.Series(df["equity"].values, index=overlap).cummax() - 1.0).min()
        * 100
    )
    pess_avg_turn = float(df["turnover"].mean())
    pess_avg_gross = float(df["gross"].mean())
    pess_avg_cost = float(df["cost"].mean()) * 10000
    pess_avg_pos = float(df["positions_count"].mean())

    theory_mean = float(theory_slice.mean())
    theory_std = float(theory_slice.std(ddof=1))
    theory_sharpe = theory_mean / theory_std * np.sqrt(252) if theory_std > 1e-8 else 0.0
    theory_ar = theory_mean * 252 * 100
    theory_vol = theory_std * np.sqrt(252) * 100
    theory_mdd = float(
        (theory_equity_slice / theory_equity_slice.cummax() - 1.0).min() * 100
    )
    theory_avg_cost = float(
        pd.read_csv(bt_dir / "daily_costs_total.csv", index_col=0, parse_dates=True)
        .loc[overlap, "cost"]
        .mean()
    ) * 10000

    print("\n" + "=" * 60)
    print("=== V2 Pessimistic Backtest (adverse 5m + integer lots) ===")
    print("=" * 60)
    print(f"  Period: {overlap[0].date()} -> {overlap[-1].date()}")
    print(f"  Days:   {len(df)}")
    print(f"  Theoretical window return: {theory_cumret*100:.2f}%")
    print(f"  Pessimistic window return: {pess_cumret*100:.2f}%")
    print(f"  AR (pessimistic): {pess_ar:.2f}%")
    print(f"  Vol (pessimistic): {pess_vol:.2f}%")
    print(f"  Sharpe (pessimistic): {pess_sharpe:.2f}")
    print(f"  Max DD (pessimistic): {pess_mdd:.2f}%")
    print(f"  Avg Turnover (pessimistic): {pess_avg_turn:.4f}")
    print(f"  Avg Gross (pessimistic): {pess_avg_gross:.4f}")
    print(f"  Avg Cost/day (pessimistic): {pess_avg_cost:.2f} bps")
    print(f"  Avg Positions/day: {pess_avg_pos:.1f}")
    print("=" * 60)

    summary = {
        "period_start": str(overlap[0].date()),
        "period_end": str(overlap[-1].date()),
        "days": int(len(df)),
        "initial_capital_jpy": initial_capital,
        "final_capital_jpy": float(capital),
        "pessimistic_cumret_pct": float(pess_cumret * 100),
        "theoretical_cumret_pct": float(theory_cumret * 100),
        "pessimistic_ar_pct": float(pess_ar),
        "pessimistic_vol_pct": float(pess_vol),
        "pessimistic_sharpe": float(pess_sharpe),
        "pessimistic_mdd_pct": float(pess_mdd),
        "pessimistic_avg_turnover": float(pess_avg_turn),
        "pessimistic_avg_gross": float(pess_avg_gross),
        "pessimistic_avg_cost_bps": float(pess_avg_cost),
        "pessimistic_avg_positions": float(pess_avg_pos),
        "theoretical_ar_pct": float(theory_ar),
        "theoretical_vol_pct": float(theory_vol),
        "theoretical_sharpe": float(theory_sharpe),
        "theoretical_mdd_pct": float(theory_mdd),
        "theoretical_avg_cost_bps": float(theory_avg_cost),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Realistic and lot-rounding share the same high-level flow and differ only
# in how they obtain target returns and the exact price-mode arguments.
# ---------------------------------------------------------------------------

def _collect_prices(
    df_5m: pd.DataFrame,
    trade_dt: pd.Timestamp,
    w_t: np.ndarray,
    entry_mode: str,
    close_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collect entry price, close price, lot and target notional per ticker."""
    p_entry_arr = np.zeros(len(JP_TICKERS))
    p_close_arr = np.zeros(len(JP_TICKERS))
    lot_arr = np.zeros(len(JP_TICKERS), dtype=int)
    side_arr = np.zeros(len(JP_TICKERS), dtype=int)
    target_arr = np.zeros(len(JP_TICKERS))
    valid_mask = np.zeros(len(JP_TICKERS), dtype=bool)

    for j, tk in enumerate(JP_TICKERS):
        side = int(np.sign(w_t[j]))
        if side == 0:
            continue

        entry_bar = _find_bar(df_5m, trade_dt, tk, time_strs=["09:10:00", "09:05:00", "09:00:00"])
        close_bar = _find_bar(df_5m, trade_dt, tk, latest_before="15:00:00")
        if entry_bar is None or close_bar is None:
            continue

        p_entry = price_from_mode(entry_bar, tk, side, entry_mode)
        p_close = price_from_mode(close_bar, tk, side, close_mode)

        if p_entry is None or p_close is None or p_entry <= 0 or p_close <= 0:
            continue

        p_entry_arr[j] = p_entry
        p_close_arr[j] = p_close
        lot_arr[j] = _lot_size(tk)
        side_arr[j] = side
        valid_mask[j] = True

    return p_entry_arr, p_close_arr, lot_arr, side_arr, target_arr, valid_mask


def _run_realistic(args: argparse.Namespace) -> None:
    """Run the 5m realistic backtest with configurable price modes."""
    config_path = ROOT / args.config
    params = _load_5m_cost_params(config_path)
    alpha_long = params["alpha_long"]
    alpha_short = params["alpha_short"]
    fin_annual = params["fin_annual"]
    borrow_annual = params["borrow_annual"]
    rev_bps = params["rev_bps"]
    slip_bps = params["slip_bps"]
    side_leverage = params["side_leverage"]
    initial_capital = float(args.capital)

    financing_daily = fin_annual / 365.0
    borrow_daily = borrow_annual / 365.0
    reverse_daily = rev_bps / 10000.0
    slip = slip_bps / 10000.0

    bt_dir = ROOT / args.input_dir
    w_df, bt_returns, bt_equity = _load_base_backtest(bt_dir)

    df_5m = load_intraday_cache("5m")
    if df_5m is None or df_5m.empty:
        raise FileNotFoundError("5m intraday cache not found")

    five_dates = sorted(pd.Series(df_5m.index.date).unique())
    five_dates = pd.to_datetime(five_dates)
    overlap = w_df.index.intersection(five_dates)
    if len(overlap) == 0:
        raise ValueError("No overlapping dates between theoretical weights and 5m data")

    records = []
    capital = initial_capital
    w_prev = np.zeros(len(JP_TICKERS))
    equity = 1.0

    for i, trade_dt in enumerate(overlap):
        w_t = w_df.loc[trade_dt].values.astype(float)

        if np.abs(w_t).sum() < 1e-8:
            records.append({
                "trade_date": trade_dt.strftime("%Y-%m-%d"),
                "gross": 0.0,
                "net": 0.0,
                "gross_ret": 0.0,
                "overnight_ret": 0.0,
                "slip_cost": 0.0,
                "fin_cost": 0.0,
                "borrow_cost": 0.0,
                "reverse_cost": 0.0,
                "cost": 0.0,
                "net_ret": 0.0,
                "equity": equity,
                "capital": capital,
                "turnover": 0.0,
                "positions_count": 0,
            })
            w_prev = np.zeros(len(JP_TICKERS))
            continue

        eff_capital = capital * side_leverage

        p_entry_arr, p_close_arr, lot_arr, side_arr, target_arr, valid_mask = _collect_prices(
            df_5m, trade_dt, w_t, args.entry_mode, args.close_mode
        )

        for j, tk in enumerate(JP_TICKERS):
            if valid_mask[j]:
                target_arr[j] = w_t[j] * eff_capital

        quantities = _allocate_lots(
            target_notional=target_arr,
            side=side_arr,
            price=p_entry_arr,
            lot=lot_arr,
            eff_capital=eff_capital,
            gross_limit_mult=args.gross_limit,
            rounding=args.rounding_mode,
            reallocate=args.reallocate_unaffordable,
        )
        notionals = np.where(quantities != 0, quantities * p_entry_arr, 0.0)
        prices_entry = np.where(valid_mask, p_entry_arr, 0.0)
        prices_close = np.where(valid_mask, p_close_arr, 0.0)
        valid_tk = np.where(quantities != 0)[0].tolist()

        w_actual = np.where(notionals != 0, notionals / eff_capital, 0.0)

        r_target = np.zeros(len(JP_TICKERS))
        for j in valid_tk:
            side = int(np.sign(w_t[j]))
            p_entry = prices_entry[j]
            p_close = prices_close[j]
            if side > 0:
                r_target[j] = (p_close / p_entry) - 1.0
            else:
                r_target[j] = (p_entry / p_close) - 1.0

        overnight_ret = 0.0
        if i + 1 < len(overlap):
            next_dt = overlap[i + 1]
            r_gap = np.zeros(len(JP_TICKERS))
            for j in valid_tk:
                tk = JP_TICKERS[j]
                side = int(np.sign(w_t[j]))
                next_bar = _find_bar(df_5m, next_dt, tk, time_strs=["09:00:00", "09:05:00", "09:10:00"])
                if next_bar is None:
                    continue
                p_next = price_from_mode(next_bar, tk, side, args.overnight_mode)
                p_prev_close = prices_close[j]
                if p_next is None or p_prev_close is None or p_prev_close <= 0:
                    continue
                r_gap[j] = (
                    (p_next / p_prev_close) - 1.0
                    if side > 0
                    else (1.0 - (p_next / p_prev_close))
                )

            alpha_mask = np.where(w_actual > 0, alpha_long, np.where(w_actual < 0, alpha_short, 0.0))
            overnight_ret = side_leverage * float(np.sum(alpha_mask * w_actual * r_gap))

        calendar_days = 1
        if i + 1 < len(overlap):
            calendar_days = (overlap[i + 1] - trade_dt).days

        gross_ret = side_leverage * float(np.sum(w_actual * r_target))
        alpha_mask = np.where(w_actual > 0, alpha_long, np.where(w_actual < 0, alpha_short, 0.0))

        turnover = float(np.sum(np.abs(w_actual - w_prev)) / 2.0)
        slip_cost = side_leverage * slip * (
            2.0 * np.sum((1.0 - alpha_mask) * np.abs(w_actual))
            + np.sum(alpha_mask * np.abs(w_actual - w_prev) / 2.0)
        )
        held_long = float(np.sum(alpha_mask * np.maximum(w_actual, 0.0)))
        held_short = float(np.sum(alpha_mask * np.maximum(-w_actual, 0.0)))
        fin_cost = side_leverage * held_long * financing_daily * calendar_days
        borrow_cost = side_leverage * held_short * borrow_daily * calendar_days
        reverse_cost = side_leverage * held_short * reverse_daily * calendar_days
        cost = slip_cost + fin_cost + borrow_cost + reverse_cost

        net_ret = gross_ret + overnight_ret - cost
        equity *= 1.0 + net_ret
        capital *= 1.0 + net_ret

        records.append({
            "trade_date": trade_dt.strftime("%Y-%m-%d"),
            "gross": float(np.sum(np.abs(w_actual))),
            "net": float(np.sum(w_actual)),
            "gross_ret": gross_ret,
            "overnight_ret": overnight_ret,
            "slip_cost": slip_cost,
            "fin_cost": fin_cost,
            "borrow_cost": borrow_cost,
            "reverse_cost": reverse_cost,
            "cost": cost,
            "net_ret": net_ret,
            "equity": equity,
            "capital": capital,
            "turnover": turnover,
            "positions_count": np.sum(quantities != 0),
        })

        w_prev = w_actual

    df = pd.DataFrame(records)

    out_suffix = f"{args.entry_mode}_{args.close_mode}_{args.overnight_mode}"
    if args.output_dir is None:
        out_dir = ROOT / "var" / "results" / f"v2_backtest_realistic_5m_{out_suffix}"
    else:
        out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    df.set_index("trade_date").to_csv(out_dir / "daily_results.csv")

    theory_slice = bt_returns.loc[overlap]
    theory_equity_slice = bt_equity.loc[overlap]
    theory_cumret = theory_equity_slice.iloc[-1] / theory_equity_slice.iloc[0] - 1.0

    pess_cumret = equity - 1.0
    pess_returns = pd.Series(df["net_ret"].values, index=overlap)
    pess_mean = float(pess_returns.mean())
    pess_std = float(pess_returns.std(ddof=1))
    pess_sharpe = pess_mean / pess_std * np.sqrt(252) if pess_std > 1e-8 else 0.0
    pess_ar = pess_mean * 252 * 100
    pess_vol = pess_std * np.sqrt(252) * 100
    pess_mdd = float(
        (pd.Series(df["equity"].values, index=overlap)
         / pd.Series(df["equity"].values, index=overlap).cummax() - 1.0).min()
        * 100
    )
    pess_avg_turn = float(df["turnover"].mean())
    pess_avg_gross = float(df["gross"].mean())
    pess_avg_cost = float(df["cost"].mean()) * 10000
    pess_avg_pos = float(df["positions_count"].mean())

    theory_mean = float(theory_slice.mean())
    theory_std = float(theory_slice.std(ddof=1))
    theory_sharpe = theory_mean / theory_std * np.sqrt(252) if theory_std > 1e-8 else 0.0
    theory_ar = theory_mean * 252 * 100
    theory_vol = theory_std * np.sqrt(252) * 100
    theory_mdd = float(
        (theory_equity_slice / theory_equity_slice.cummax() - 1.0).min() * 100
    )
    theory_avg_cost = float(
        pd.read_csv(bt_dir / "daily_costs_total.csv", index_col=0, parse_dates=True)
        .loc[overlap, "cost"]
        .mean()
    ) * 10000

    print("\n" + "=" * 60)
    print(
        f"=== V2 Realistic Backtest "
        f"(entry={args.entry_mode}, close={args.close_mode}, overnight={args.overnight_mode}) ==="
    )
    print("=" * 60)
    print(f"  Period: {overlap[0].date()} -> {overlap[-1].date()}")
    print(f"  Days:   {len(df)}")
    print(f"  Theoretical window return: {theory_cumret*100:.2f}%")
    print(f"  Realistic window return:   {pess_cumret*100:.2f}%")
    print(f"  AR: {pess_ar:.2f}%")
    print(f"  Vol: {pess_vol:.2f}%")
    print(f"  Sharpe: {pess_sharpe:.2f}")
    print(f"  Max DD: {pess_mdd:.2f}%")
    print(f"  Avg Turnover: {pess_avg_turn:.4f}")
    print(f"  Avg Gross: {pess_avg_gross:.4f}")
    print(f"  Avg Cost/day: {pess_avg_cost:.2f} bps")
    print(f"  Avg Positions/day: {pess_avg_pos:.1f}")
    print("=" * 60)

    summary = {
        "period_start": str(overlap[0].date()),
        "period_end": str(overlap[-1].date()),
        "days": int(len(df)),
        "initial_capital_jpy": initial_capital,
        "final_capital_jpy": float(capital),
        "realistic_cumret_pct": float(pess_cumret * 100),
        "theoretical_cumret_pct": float(theory_cumret * 100),
        "realistic_ar_pct": float(pess_ar),
        "realistic_vol_pct": float(pess_vol),
        "realistic_sharpe": float(pess_sharpe),
        "realistic_mdd_pct": float(pess_mdd),
        "realistic_avg_turnover": float(pess_avg_turn),
        "realistic_avg_gross": float(pess_avg_gross),
        "realistic_avg_cost_bps": float(pess_avg_cost),
        "realistic_avg_positions": float(pess_avg_pos),
        "theoretical_ar_pct": float(theory_ar),
        "theoretical_vol_pct": float(theory_vol),
        "theoretical_sharpe": float(theory_sharpe),
        "theoretical_mdd_pct": float(theory_mdd),
        "theoretical_avg_cost_bps": float(theory_avg_cost),
        "entry_mode": args.entry_mode,
        "close_mode": args.close_mode,
        "overnight_mode": args.overnight_mode,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)


def _run_lot_rounding(args: argparse.Namespace) -> None:
    """Run the 5m lot-rounding backtest using theoretical returns."""
    config_path = ROOT / args.config
    params = _load_5m_cost_params(config_path)
    alpha_long = params["alpha_long"]
    alpha_short = params["alpha_short"]
    fin_annual = params["fin_annual"]
    borrow_annual = params["borrow_annual"]
    rev_bps = params["rev_bps"]
    slip_bps = params["slip_bps"]
    side_leverage = params["side_leverage"]
    initial_capital = float(args.capital)

    financing_daily = fin_annual / 365.0
    borrow_daily = borrow_annual / 365.0
    reverse_daily = rev_bps / 10000.0
    slip = slip_bps / 10000.0

    bt_dir = ROOT / args.input_dir
    w_df, bt_returns, bt_equity = _load_base_backtest(bt_dir)

    df_exec = load_df_exec_from_local_cache()
    y_target = compute_jp_target_returns(df_exec, JP_TICKERS)
    y_target_df = pd.DataFrame(y_target, index=df_exec.index, columns=JP_TICKERS)
    gap_df = df_exec[[f"jp_gap_{tk}" for tk in JP_TICKERS]]

    df_5m = load_intraday_cache("5m")
    if df_5m is None or df_5m.empty:
        raise FileNotFoundError("5m intraday cache not found")

    five_dates = sorted(pd.Series(df_5m.index.date).unique())
    five_dates = pd.to_datetime(five_dates)
    overlap = w_df.index.intersection(five_dates)
    if len(overlap) == 0:
        raise ValueError("No overlapping dates between theoretical weights and 5m data")

    records = []
    capital = initial_capital
    w_prev = np.zeros(len(JP_TICKERS))
    equity = 1.0

    for i, trade_dt in enumerate(overlap):
        w_t = w_df.loc[trade_dt].values.astype(float)
        r_target = y_target_df.loc[trade_dt].values.astype(float)

        if np.abs(w_t).sum() < 1e-8:
            records.append({
                "trade_date": trade_dt.strftime("%Y-%m-%d"),
                "gross": 0.0,
                "net": 0.0,
                "gross_ret": 0.0,
                "overnight_ret": 0.0,
                "slip_cost": 0.0,
                "fin_cost": 0.0,
                "borrow_cost": 0.0,
                "reverse_cost": 0.0,
                "cost": 0.0,
                "net_ret": 0.0,
                "equity": equity,
                "capital": capital,
                "turnover": 0.0,
                "positions_count": 0,
            })
            w_prev = np.zeros(len(JP_TICKERS))
            continue

        eff_capital = capital * side_leverage

        p_entry_arr = np.zeros(len(JP_TICKERS))
        lot_arr = np.zeros(len(JP_TICKERS), dtype=int)
        side_arr = np.zeros(len(JP_TICKERS), dtype=int)
        target_arr = np.zeros(len(JP_TICKERS))
        valid_mask = np.zeros(len(JP_TICKERS), dtype=bool)

        for j, tk in enumerate(JP_TICKERS):
            side = int(np.sign(w_t[j]))
            if side == 0:
                continue

            entry_bar = _find_bar(df_5m, trade_dt, tk, time_strs=["09:10:00", "09:05:00", "09:00:00"])
            if entry_bar is None:
                continue

            p_entry = price_from_mode(entry_bar, tk, side, args.entry_mode)
            if p_entry is None or p_entry <= 0:
                continue

            p_entry_arr[j] = p_entry
            lot_arr[j] = _lot_size(tk)
            side_arr[j] = side
            target_arr[j] = w_t[j] * eff_capital
            valid_mask[j] = True

        quantities = _allocate_lots(
            target_notional=target_arr,
            side=side_arr,
            price=p_entry_arr,
            lot=lot_arr,
            eff_capital=eff_capital,
            gross_limit_mult=args.gross_limit,
            rounding=args.rounding_mode,
            reallocate=args.reallocate_unaffordable,
        )
        notionals = np.where(quantities != 0, quantities * p_entry_arr, 0.0)
        w_actual = np.where(notionals != 0, notionals / eff_capital, 0.0)

        overnight_ret = 0.0
        if i + 1 < len(overlap):
            next_dt = overlap[i + 1]
            r_gap = gap_df.loc[next_dt].values.astype(float)
            alpha_mask = np.where(w_actual > 0, alpha_long, np.where(w_actual < 0, alpha_short, 0.0))
            overnight_ret = side_leverage * float(np.sum(alpha_mask * w_actual * r_gap))

        calendar_days = 1
        if i + 1 < len(overlap):
            calendar_days = (overlap[i + 1] - trade_dt).days

        gross_ret = side_leverage * float(np.sum(w_actual * r_target))
        alpha_mask = np.where(w_actual > 0, alpha_long, np.where(w_actual < 0, alpha_short, 0.0))

        turnover = float(np.sum(np.abs(w_actual - w_prev)) / 2.0)
        slip_cost = side_leverage * slip * (
            2.0 * np.sum((1.0 - alpha_mask) * np.abs(w_actual))
            + np.sum(alpha_mask * np.abs(w_actual - w_prev) / 2.0)
        )
        held_long = float(np.sum(alpha_mask * np.maximum(w_actual, 0.0)))
        held_short = float(np.sum(alpha_mask * np.maximum(-w_actual, 0.0)))
        fin_cost = side_leverage * held_long * financing_daily * calendar_days
        borrow_cost = side_leverage * held_short * borrow_daily * calendar_days
        reverse_cost = side_leverage * held_short * reverse_daily * calendar_days
        cost = slip_cost + fin_cost + borrow_cost + reverse_cost

        net_ret = gross_ret + overnight_ret - cost
        equity *= 1.0 + net_ret
        capital *= 1.0 + net_ret

        records.append({
            "trade_date": trade_dt.strftime("%Y-%m-%d"),
            "gross": float(np.sum(np.abs(w_actual))),
            "net": float(np.sum(w_actual)),
            "gross_ret": gross_ret,
            "overnight_ret": overnight_ret,
            "slip_cost": slip_cost,
            "fin_cost": fin_cost,
            "borrow_cost": borrow_cost,
            "reverse_cost": reverse_cost,
            "cost": cost,
            "net_ret": net_ret,
            "equity": equity,
            "capital": capital,
            "turnover": turnover,
            "positions_count": np.sum(quantities != 0),
        })

        w_prev = w_actual

    df = pd.DataFrame(records)

    out_suffix = f"{args.entry_mode}_{args.rounding_mode}"
    if args.reallocate_unaffordable:
        out_suffix += "_realloc"
    if args.output_dir is None:
        out_dir = ROOT / "var" / "results" / f"v2_backtest_lot_rounding_{out_suffix}"
    else:
        out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    df.set_index("trade_date").to_csv(out_dir / "daily_results.csv")

    theory_slice = bt_returns.loc[overlap]
    theory_equity_slice = bt_equity.loc[overlap]
    theory_cumret = theory_equity_slice.iloc[-1] / theory_equity_slice.iloc[0] - 1.0

    pess_cumret = equity - 1.0
    pess_returns = pd.Series(df["net_ret"].values, index=overlap)
    pess_mean = float(pess_returns.mean())
    pess_std = float(pess_returns.std(ddof=1))
    pess_sharpe = pess_mean / pess_std * np.sqrt(252) if pess_std > 1e-8 else 0.0
    pess_ar = pess_mean * 252 * 100
    pess_vol = pess_std * np.sqrt(252) * 100
    pess_mdd = float(
        (pd.Series(df["equity"].values, index=overlap)
         / pd.Series(df["equity"].values, index=overlap).cummax() - 1.0).min()
        * 100
    )
    pess_avg_turn = float(df["turnover"].mean())
    pess_avg_gross = float(df["gross"].mean())
    pess_avg_cost = float(df["cost"].mean()) * 10000
    pess_avg_pos = float(df["positions_count"].mean())

    theory_mean = float(theory_slice.mean())
    theory_std = float(theory_slice.std(ddof=1))
    theory_sharpe = theory_mean / theory_std * np.sqrt(252) if theory_std > 1e-8 else 0.0
    theory_ar = theory_mean * 252 * 100
    theory_vol = theory_std * np.sqrt(252) * 100
    theory_mdd = float(
        (theory_equity_slice / theory_equity_slice.cummax() - 1.0).min() * 100
    )
    theory_avg_cost = float(
        pd.read_csv(bt_dir / "daily_costs_total.csv", index_col=0, parse_dates=True)
        .loc[overlap, "cost"]
        .mean()
    ) * 10000

    print("\n" + "=" * 60)
    print(
        f"=== V2 Lot Rounding Backtest "
        f"(entry={args.entry_mode}, rounding={args.rounding_mode}, "
        f"realloc={args.reallocate_unaffordable}, capital={initial_capital:,.0f}) ==="
    )
    print("=" * 60)
    print(f"  Period: {overlap[0].date()} -> {overlap[-1].date()}")
    print(f"  Days:   {len(df)}")
    print(f"  Theoretical window return: {theory_cumret*100:.2f}%")
    print(f"  Lot rounding window return: {pess_cumret*100:.2f}%")
    print(f"  AR: {pess_ar:.2f}%")
    print(f"  Vol: {pess_vol:.2f}%")
    print(f"  Sharpe: {pess_sharpe:.2f}")
    print(f"  Max DD: {pess_mdd:.2f}%")
    print(f"  Avg Turnover: {pess_avg_turn:.4f}")
    print(f"  Avg Gross: {pess_avg_gross:.4f}")
    print(f"  Avg Cost/day: {pess_avg_cost:.2f} bps")
    print(f"  Avg Positions/day: {pess_avg_pos:.1f}")
    print("=" * 60)

    summary = {
        "period_start": str(overlap[0].date()),
        "period_end": str(overlap[-1].date()),
        "days": int(len(df)),
        "initial_capital_jpy": initial_capital,
        "final_capital_jpy": float(capital),
        "lot_rounding_cumret_pct": float(pess_cumret * 100),
        "theoretical_cumret_pct": float(theory_cumret * 100),
        "lot_rounding_ar_pct": float(pess_ar),
        "lot_rounding_vol_pct": float(pess_vol),
        "lot_rounding_sharpe": float(pess_sharpe),
        "lot_rounding_mdd_pct": float(pess_mdd),
        "lot_rounding_avg_turnover": float(pess_avg_turn),
        "lot_rounding_avg_gross": float(pess_avg_gross),
        "lot_rounding_avg_cost_bps": float(pess_avg_cost),
        "lot_rounding_avg_positions": float(pess_avg_pos),
        "theoretical_ar_pct": float(theory_ar),
        "theoretical_vol_pct": float(theory_vol),
        "theoretical_sharpe": float(theory_sharpe),
        "theoretical_mdd_pct": float(theory_mdd),
        "theoretical_avg_cost_bps": float(theory_avg_cost),
        "entry_mode": args.entry_mode,
        "rounding_mode": args.rounding_mode,
        "reallocate_unaffordable": args.reallocate_unaffordable,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)


def _run_theoretical_5m(args: argparse.Namespace) -> None:
    """Extract the exact backtest to the 5m window used by pessimistic."""
    bt_dir = ROOT / args.input_dir
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    daily_returns = pd.read_csv(
        bt_dir / "daily_net_returns.csv", index_col=0, parse_dates=True
    )["net_return"]
    equity = pd.read_csv(
        bt_dir / "daily_equity_curve.csv", index_col=0, parse_dates=True
    )["equity"]
    weights = pd.read_csv(bt_dir / "daily_weights.csv", index_col=0, parse_dates=True)
    costs = pd.read_csv(
        bt_dir / "daily_costs_total.csv", index_col=0, parse_dates=True
    )["cost"]
    turnover = pd.read_csv(
        bt_dir / "daily_turnover.csv", index_col=0, parse_dates=True
    )["turnover"]
    gross = pd.read_csv(
        bt_dir / "daily_gross.csv", index_col=0, parse_dates=True
    )["gross"]
    fallback = pd.read_csv(
        bt_dir / "daily_fallback.csv", index_col=0, parse_dates=True
    )["fallback"]

    pess_dir = ROOT / "var" / "results" / "v2_backtest_pessimistic_5m"
    if (pess_dir / "daily_results.csv").exists():
        pess_dates = pd.read_csv(pess_dir / "daily_results.csv", parse_dates=["trade_date"])["trade_date"]
    elif (pess_dir / "summary.json").exists():
        with open(pess_dir / "summary.json") as f:
            pess_summary = json.load(f)
        start = pd.to_datetime(pess_summary["period_start"])
        end = pd.to_datetime(pess_summary["period_end"])
        pess_dates = pd.date_range(start, end, freq="B")
    else:
        start = pd.to_datetime("2026-03-03")
        end = pd.to_datetime("2026-06-01")
        pess_dates = pd.date_range(start, end, freq="B")

    start = pess_dates.min()
    end = pess_dates.max()

    mask = daily_returns.index.isin(pess_dates)
    ret_slice = daily_returns.loc[mask]
    eq_slice = equity.loc[mask]
    w_slice = weights.loc[mask]
    cost_slice = costs.loc[mask]
    turn_slice = turnover.loc[mask]
    gross_slice = gross.loc[mask]
    fb_slice = fallback.loc[mask]

    ret_slice.to_csv(out_dir / "daily_net_returns.csv", header=["net_return"])
    eq_slice.to_csv(out_dir / "daily_equity_curve.csv", header=["equity"])
    w_slice.to_csv(out_dir / "daily_weights.csv")
    cost_slice.to_csv(out_dir / "daily_costs_total.csv", header=["cost"])
    turn_slice.to_csv(out_dir / "daily_turnover.csv", header=["turnover"])
    gross_slice.to_csv(out_dir / "daily_gross.csv", header=["gross"])
    fb_slice.to_csv(out_dir / "daily_fallback.csv", header=["fallback"])

    mean_ret = ret_slice.mean()
    std_ret = ret_slice.std(ddof=1)
    sharpe = mean_ret / std_ret * np.sqrt(252) if std_ret > 1e-8 else 0.0
    ar = mean_ret * 252 * 100
    vol = std_ret * np.sqrt(252) * 100
    mdd = float((eq_slice / eq_slice.cummax() - 1.0).min() * 100)
    cumret = eq_slice.iloc[-1] / eq_slice.iloc[0] - 1.0
    avg_cost = float(cost_slice.mean() * 10000)
    avg_turn = float(turn_slice.mean())
    avg_gross = float(gross_slice.mean())
    avg_fb = float(fb_slice.mean())

    print("=" * 60)
    print("=== V2 Theoretical Backtest (5m window) ===")
    print("=" * 60)
    print(f"  Period: {start.date()} -> {end.date()}")
    print(f"  Days:   {len(ret_slice)}")
    print(f"  Cumulative Return: {cumret*100:.2f}%")
    print(f"  AR: {ar:.2f}%")
    print(f"  Vol: {vol:.2f}%")
    print(f"  Sharpe: {sharpe:.2f}")
    print(f"  Max DD: {mdd:.2f}%")
    print(f"  Avg Cost/day: {avg_cost:.2f} bps")
    print(f"  Avg Turnover: {avg_turn:.4f}")
    print(f"  Avg Gross: {avg_gross:.4f}")
    print(f"  Avg Fallback: {avg_fb:.2%}")
    print("=" * 60)

    summary = {
        "period_start": str(start.date()),
        "period_end": str(end.date()),
        "days": int(len(ret_slice)),
        "cumulative_return_pct": float(cumret * 100),
        "ar_pct": float(ar),
        "vol_pct": float(vol),
        "sharpe": float(sharpe),
        "mdd_pct": float(mdd),
        "avg_cost_bps": float(avg_cost),
        "avg_turnover": float(avg_turn),
        "avg_gross": float(avg_gross),
        "avg_fallback": float(avg_fb),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    report_lines = [
        "# V2 理論値バックテスト（5分足対象期間）\n\n",
        f"期間: {start.date()} 〜 {end.date()}\n",
        f"日数: {len(ret_slice)}\n\n",
        "## 主要指標\n\n",
        f"- 累積リターン: {cumret*100:.2f}%\n",
        f"- AR: {ar:.2f}%\n",
        f"- Volatility: {vol:.2f}%\n",
        f"- Sharpe: {sharpe:.2f}\n",
        f"- Max DD: {mdd:.2f}%\n",
        f"- 平均コスト/日: {avg_cost:.2f} bps\n",
        f"- 平均ターンオーバー: {avg_turn:.4f}\n",
        f"- 平均グロス: {avg_gross:.4f}\n",
        f"- 平均フォールバック率: {avg_fb:.2%}\n\n",
        "## 説明\n\n",
        f"これは `{bt_dir}` の理論値結果を、\n",
        "5分足キャッシュが存在する期間に切り出したもの。\n",
        "悲観的バックテスト `results/v2_backtest_pessimistic_5m` と同一期間で比較可能。\n",
    ]
    (out_dir / "report.md").write_text("".join(report_lines), encoding="utf-8")
    print(f"\nTheoretical 5m window outputs written to {out_dir}")


def _load_base_backtest(bt_dir: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    w_df = pd.read_csv(bt_dir / "daily_weights.csv", index_col=0, parse_dates=True)
    bt_returns = pd.read_csv(
        bt_dir / "daily_net_returns.csv", index_col=0, parse_dates=True
    )["net_return"]
    bt_equity = pd.read_csv(
        bt_dir / "daily_equity_curve.csv", index_col=0, parse_dates=True
    )["equity"]
    return w_df, bt_returns, bt_equity


def main() -> None:
    parser = argparse.ArgumentParser(description="Consolidated V2 backtest experiment runner")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["exact", "realistic", "pessimistic", "theoretical_5m", "lot_rounding"],
        help="Backtest mode to run",
    )

    # Exact-production arguments
    parser.add_argument("--config", default="configs/production/production.yaml")
    parser.add_argument(
        "--gap-dir", default="var/live/pipeline_data/gap_adjusted_distribution/20260731_024303"
    )
    parser.add_argument("--start-date", default="2020-01-06")
    parser.add_argument("--end-date", default="latest")
    parser.add_argument("--overlay-model-dir", default="models/ml_order_overlay/phase2_8")
    parser.add_argument("--side-leverage", type=float, default=1.5)
    parser.add_argument(
        "--output-dir",
        default="var/results/v2_backtest_exact_production",
        help="Output directory for exact mode",
    )
    parser.add_argument("--n-jobs", type=int, default=1)

    # 5m modes arguments
    parser.add_argument(
        "--input-dir",
        default=str(_default_input_dir()),
        help="Directory containing the exact backtest CSVs (weights, net returns, equity)",
    )
    parser.add_argument("--entry-mode", default="midpoint", choices=["adverse", "midpoint", "close", "open"])
    parser.add_argument("--close-mode", default="adverse", choices=["adverse", "midpoint", "close", "open"])
    parser.add_argument(
        "--overnight-mode", default="adverse", choices=["adverse", "midpoint", "close", "open"]
    )
    parser.add_argument("--rounding-mode", default="floor", choices=["floor", "nearest"])
    parser.add_argument("--reallocate-unaffordable", action="store_true")
    parser.add_argument("--gross-limit", type=float, default=2.0)
    parser.add_argument("--capital", type=float, default=324_280.0)

    args = parser.parse_args()

    if args.mode == "exact":
        _run_exact(args)
    elif args.mode == "pessimistic":
        _run_pessimistic(args)
    elif args.mode == "realistic":
        _run_realistic(args)
    elif args.mode == "lot_rounding":
        _run_lot_rounding(args)
    elif args.mode == "theoretical_5m":
        if args.output_dir == "var/results/v2_backtest_exact_production":
            # The default was intended for exact mode; switch to the theoretical output.
            args.output_dir = "var/results/v2_backtest_theoretical_5m"
        _run_theoretical_5m(args)
    else:
        parser.error(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
