"""Legacy V1 backtest orchestration for research experiments.

This module contains the generic ``BaseModel`` backtest that used to live in
``leadlag.execution.backtester.BacktestEngine.run_backtest``.  It is intentionally
moved to the ``research`` package because the production path is now V2
(``BacktestEngine.run_v2_backtest`` / ``generate_v2_production_portfolio``).

Research experiments that still need the legacy ``BaseModel`` path should import
``run_v1_backtest`` from here instead of ``BacktestEngine.run_backtest``.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
import pandas as pd

from leadlag.data.tickers import JP_TICKERS
from leadlag.execution.backtester import BacktestEngine
from research.models.base import BaseModel

logger = logging.getLogger(__name__)


def _resolve_run_backtest_cost_params(
    model: BaseModel,
    slippage_bps: float | None,
    overnight_alpha: float | None,
    overnight_alpha_long: float | None,
    overnight_alpha_short: float | None,
    buy_interest_annual: float | None,
    borrow_fee_annual: float | None,
    reverse_fee_bps: float | None,
) -> dict:
    """Resolve cost/financing parameters for run_v1_backtest."""
    slip_bps = slippage_bps if slippage_bps is not None else getattr(model, "slippage_bps", 5.0)
    if overnight_alpha is not None:
        alpha_long = overnight_alpha
        alpha_short = overnight_alpha
    else:
        alpha_long = (
            overnight_alpha_long
            if overnight_alpha_long is not None
            else getattr(model, "overnight_alpha_long", 0.0)
        )
        alpha_short = (
            overnight_alpha_short
            if overnight_alpha_short is not None
            else getattr(model, "overnight_alpha_short", 0.0)
        )
    fin_annual = (
        buy_interest_annual
        if buy_interest_annual is not None
        else getattr(model, "buy_interest_annual", 0.025)
    )
    borrow_annual = (
        borrow_fee_annual
        if borrow_fee_annual is not None
        else getattr(model, "borrow_fee_annual", 0.0115)
    )
    rev_bps = reverse_fee_bps if reverse_fee_bps is not None else getattr(model, "reverse_fee_bps", 2.0)
    return {
        "slip_bps": slip_bps,
        "alpha_long": alpha_long,
        "alpha_short": alpha_short,
        "fin_annual": fin_annual,
        "borrow_annual": borrow_annual,
        "rev_bps": rev_bps,
    }


def _predict_signals_and_weights_run_backtest(
    model: BaseModel,
    df_exec: pd.DataFrame,
    sim_dates: pd.DatetimeIndex,
    start_idx: int,
    end_idx: int,
    sim_dates_slice: pd.DatetimeIndex,
    n_jobs: int,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Predict signals and build the SRE weight matrix."""
    T = len(df_exec)

    # Predict signals for the entire dataset
    pred: dict[str, Any] = model.predict_signals(df_exec, n_jobs=n_jobs)
    sre_signals_df = pred["signals"]

    # Generate weights
    n_j = cast(int, getattr(model, "n_j"))
    sre_weights = np.zeros((T, n_j))
    sigma_yy_array = pred.get("sigma_yy", None)
    for i in range(start_idx, end_idx + 1):
        sigma_yy_i = sigma_yy_array[i] if sigma_yy_array is not None else None
        sre_weights[i] = model.build_weights(sre_signals_df.iloc[i].values, Sigma_YY=sigma_yy_i)

    sre_weights_df = pd.DataFrame(
        sre_weights[start_idx : end_idx + 1], index=sim_dates_slice, columns=JP_TICKERS
    )

    y_jp_oc_df = pred["y_jp_oc_df"]
    return pred, sre_weights_df, y_jp_oc_df


def _assemble_run_backtest_results(
    pnl: dict,
    pred: dict,
    sre_weights_df: pd.DataFrame,
    sim_dates_slice: pd.DatetimeIndex,
    alpha_long: float,
    alpha_short: float,
) -> dict:
    """Assemble the output dict for run_v1_backtest."""
    sre_signals_df = pred["signals"]

    daily_returns_gross = pd.Series(pnl["gross_returns"], index=sim_dates_slice)
    daily_returns_net = pd.Series(pnl["net_returns"], index=sim_dates_slice)
    daily_returns_gross_oc = pd.Series(pnl["gross_returns_oc"], index=sim_dates_slice)
    daily_returns_net_oc = pd.Series(pnl["net_returns_oc"], index=sim_dates_slice)
    daily_costs = pd.Series(pnl["costs"], index=sim_dates_slice)
    daily_slip_costs = pd.Series(pnl["slip_costs"], index=sim_dates_slice)
    daily_financing_costs = pd.Series(pnl["financing_costs"], index=sim_dates_slice)
    daily_borrow_costs = pd.Series(pnl["borrow_costs"], index=sim_dates_slice)
    daily_reverse_costs = pd.Series(pnl["reverse_costs"], index=sim_dates_slice)
    daily_overnight_returns = pd.Series(pnl["overnight_returns"], index=sim_dates_slice)
    daily_gross_exps = pd.Series(pnl["gross_exps"], index=sim_dates_slice)
    daily_turnover = pd.Series(pnl["turnover"], index=sim_dates_slice)

    wealth = (1.0 + daily_returns_net).cumprod()
    running_max = wealth.cummax()
    drawdown = (wealth / running_max) - 1.0

    out_res = {
        "raw_pca_signals": pred["raw_pca_signals"].loc[sim_dates_slice],
        "residual_pca_signals": pred["residual_pca_signals"].loc[sim_dates_slice],
        "p4_signals": pred["p4_signals"].loc[sim_dates_slice],
        "signals": sre_signals_df.loc[sim_dates_slice],
        "normalized_signals": pred["normalized_signals"].loc[sim_dates_slice],
        "weights": sre_weights_df,
        "daily_returns_gross": daily_returns_gross,
        "daily_returns": daily_returns_net,
        "daily_returns_gross_oc": daily_returns_gross_oc,
        "daily_returns_net_oc": daily_returns_net_oc,
        "daily_costs": daily_costs,
        "daily_slip_costs": daily_slip_costs,
        "daily_financing_costs": daily_financing_costs,
        "daily_borrow_costs": daily_borrow_costs,
        "daily_reverse_costs": daily_reverse_costs,
        "daily_overnight_returns": daily_overnight_returns,
        "daily_gross_exps": daily_gross_exps,
        "daily_turnover": daily_turnover,
        "overnight_alpha_long": alpha_long,
        "overnight_alpha_short": alpha_short,
        "equity_curve": wealth,
        "drawdown": drawdown,
    }
    if "prior_info" in pred:
        out_res["prior_info"] = pred["prior_info"]

    return out_res


def run_v1_backtest(
    model: BaseModel,
    df_exec: pd.DataFrame,
    start_date: str = "2015-01-05",
    end_date: str = "latest",
    slippage_bps: float | None = None,
    overnight_alpha: float | None = None,
    overnight_alpha_long: float | None = None,
    overnight_alpha_short: float | None = None,
    buy_interest_annual: float | None = None,
    borrow_fee_annual: float | None = None,
    reverse_fee_bps: float | None = None,
    n_jobs: int = 1,
) -> dict:
    """Run a historical backtest of the model on the execution dataset.

    This is the legacy V1 generic ``BaseModel`` backtest, moved to the research
    package for backward compatibility of existing experiments.

    Args:
        model: Pure model implementing BaseModel.
        df_exec: Execution DataFrame.
        start_date: Backtest start date.
        end_date: Backtest end date.
        slippage_bps: Slippage bps one-way to override defaults.
        overnight_alpha: Uniform alpha for both long and short (backward compat).
            If specified, overrides overnight_alpha_long/short.
        overnight_alpha_long: Alpha for long positions (0=full close, 1=full hold).
        overnight_alpha_short: Alpha for short positions (0=full close, 1=full hold).
        buy_interest_annual: Annual financing rate for long positions.
        borrow_fee_annual: Annual stock borrow fee for short positions.
        reverse_fee_bps: Daily reverse stock lending fee (bps).
        n_jobs: Number of parallel workers for signal computation. 1 = sequential.

    Returns:
        Dict containing backtest results and metrics.
    """
    cost_params = _resolve_run_backtest_cost_params(
        model,
        slippage_bps,
        overnight_alpha,
        overnight_alpha_long,
        overnight_alpha_short,
        buy_interest_annual,
        borrow_fee_annual,
        reverse_fee_bps,
    )
    slip_bps = cost_params["slip_bps"]
    alpha_long = cost_params["alpha_long"]
    alpha_short = cost_params["alpha_short"]
    fin_annual = cost_params["fin_annual"]
    borrow_annual = cost_params["borrow_annual"]
    rev_bps = cost_params["rev_bps"]

    logger.info(
        "Starting V1 generic backtest: start=%s, slippage=%s bps, "
        "alpha_long=%s, alpha_short=%s, financing=%s%% ann, "
        "borrow=%s%% ann, reverse=%s bps/day",
        start_date,
        slip_bps,
        alpha_long,
        alpha_short,
        fin_annual * 100,
        borrow_annual * 100,
        rev_bps,
    )

    sim_dates, start_idx, end_idx = BacktestEngine._resolve_sim_dates(
        df_exec, start_date, end_date, getattr(model, "corr_window", 60)
    )
    sim_dates_slice = cast(pd.DatetimeIndex, sim_dates[start_idx : end_idx + 1])

    pred, sre_weights_df, y_jp_oc_df = _predict_signals_and_weights_run_backtest(
        model,
        df_exec,
        sim_dates,
        start_idx,
        end_idx,
        sim_dates_slice,
        n_jobs,
    )

    y_jp_target_arr, gap_returns_arr = BacktestEngine._compute_target_and_gap_returns(
        df_exec, sim_dates, sim_dates_slice
    )

    # Cost parameters
    slip = slip_bps / 10000.0
    financing_daily = fin_annual / 365.0
    borrow_daily = borrow_annual / 365.0
    reverse_daily = rev_bps / 10000.0

    sre_weights_arr = sre_weights_df.values
    y_jp_oc_arr = y_jp_oc_df.loc[sim_dates_slice].values

    pnl = BacktestEngine._simulate_daily_pnl(
        weights=sre_weights_arr,
        target_returns=y_jp_target_arr,
        gap_returns=gap_returns_arr,
        sim_dates=sim_dates_slice,
        slip=slip,
        financing_daily=financing_daily,
        borrow_daily=borrow_daily,
        reverse_daily=reverse_daily,
        alpha_long=alpha_long,
        alpha_short=alpha_short,
        side_leverage=1.0,
        oc_returns=y_jp_oc_arr,
    )

    return _assemble_run_backtest_results(
        pnl, pred, sre_weights_df, sim_dates_slice, alpha_long, alpha_short
    )
