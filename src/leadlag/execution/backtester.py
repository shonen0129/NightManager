"""BacktestEngine — generic engine for running historical backtests on pure strategy models."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from leadlag.data.tickers import JP_TICKERS
from leadlag.models.base import BaseModel

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Generic engine for executing historical backtests on strategy models."""

    @classmethod
    def run_backtest(
        cls,
        model: BaseModel,
        df_exec: pd.DataFrame,
        start_date: str = "2015-01-05",
        end_date: str = "latest",
        slippage_bps: float | None = None,
    ) -> dict:
        """Run a historical backtest of the model on the execution dataset.

        Args:
            model: Pure model implementing BaseModel.
            df_exec: Execution DataFrame.
            start_date: Backtest start date.
            end_date: Backtest end date.
            slippage_bps: Slippage bps one-way to override defaults.

        Returns:
            Dict containing backtest results and metrics.
        """
        slip_bps = slippage_bps if slippage_bps is not None else getattr(model, "slippage_bps", 5.0)
        logger.info(f"Starting generic backtest: start={start_date}, slippage={slip_bps} bps")

        T = len(df_exec)
        sim_dates = df_exec.index

        # Predict signals for the entire dataset
        pred = model.predict_signals(df_exec)
        sre_signals_df = pred["signals"]

        # Setup simulation indexes
        start_dt = pd.to_datetime(start_date)
        start_idx = max(df_exec.index.searchsorted(start_dt), getattr(model, "corr_window", 60))

        if end_date != "latest":
            end_dt = pd.to_datetime(end_date)
            end_idx = min(df_exec.index.searchsorted(end_dt), T - 1)
        else:
            end_idx = T - 1

        sim_dates_slice = sim_dates[start_idx : end_idx + 1]

        # Generate weights
        sre_weights = np.zeros((T, model.n_j))
        for i in range(start_idx, end_idx + 1):
            sre_weights[i] = model.build_weights(sre_signals_df.iloc[i].values)

        sre_weights_df = pd.DataFrame(
            sre_weights[start_idx : end_idx + 1], index=sim_dates_slice, columns=JP_TICKERS
        )

        y_jp_oc_df = pred["y_jp_oc_df"]

        # Compute 9:10-to-close target returns for JP assets
        from leadlag.models.sre import compute_jp_target_returns
        y_jp_target = compute_jp_target_returns(df_exec, JP_TICKERS)
        y_jp_target_df = pd.DataFrame(y_jp_target, index=sim_dates, columns=JP_TICKERS)

        # Returns and Cost drag calculations
        gross_returns_list = []
        net_returns_list = []
        gross_returns_oc_list = []
        net_returns_oc_list = []
        cost_list = []
        gross_exp_list = []
        turnover_list = []

        w_prev = np.zeros(model.n_j)
        for date in sim_dates_slice:
            w_t = sre_weights_df.loc[date].values
            r_target_t = y_jp_target_df.loc[date].values
            r_oc_t = y_jp_oc_df.loc[date].values

            # Primary (9:10-to-Close)
            gross_ret = float(np.sum(w_t * r_target_t))
            gross_exp = float(np.sum(np.abs(w_t)))

            cost = 2.0 * (slip_bps / 10000.0) * gross_exp
            net_ret = gross_ret - cost

            # Auxiliary (Open-to-Close)
            gross_ret_oc = float(np.sum(w_t * r_oc_t))
            net_ret_oc = gross_ret_oc - cost

            turnover = float(np.sum(np.abs(w_t - w_prev)) / 2.0)

            gross_returns_list.append(gross_ret)
            net_returns_list.append(net_ret)
            gross_returns_oc_list.append(gross_ret_oc)
            net_returns_oc_list.append(net_ret_oc)
            cost_list.append(cost)
            gross_exp_list.append(gross_exp)
            turnover_list.append(turnover)

            w_prev = w_t

        daily_returns_gross = pd.Series(gross_returns_list, index=sim_dates_slice)
        daily_returns_net = pd.Series(net_returns_list, index=sim_dates_slice)
        daily_returns_gross_oc = pd.Series(gross_returns_oc_list, index=sim_dates_slice)
        daily_returns_net_oc = pd.Series(net_returns_oc_list, index=sim_dates_slice)
        daily_costs = pd.Series(cost_list, index=sim_dates_slice)
        daily_gross_exps = pd.Series(gross_exp_list, index=sim_dates_slice)
        daily_turnover = pd.Series(turnover_list, index=sim_dates_slice)

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
            "daily_gross_exps": daily_gross_exps,
            "daily_turnover": daily_turnover,
            "equity_curve": wealth,
            "drawdown": drawdown,
        }
        if "prior_info" in pred:
            out_res["prior_info"] = pred["prior_info"]

        return out_res
