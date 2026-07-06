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
        overnight_alpha: float | None = None,
        overnight_alpha_long: float | None = None,
        overnight_alpha_short: float | None = None,
        buy_interest_annual: float | None = None,
        borrow_fee_annual: float | None = None,
        reverse_fee_bps: float | None = None,
    ) -> dict:
        """Run a historical backtest of the model on the execution dataset.

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

        Returns:
            Dict containing backtest results and metrics.
        """
        slip_bps = slippage_bps if slippage_bps is not None else getattr(model, "slippage_bps", 5.0)
        # Resolve alpha: uniform overnight_alpha takes precedence for backward compat
        if overnight_alpha is not None:
            alpha_long = overnight_alpha
            alpha_short = overnight_alpha
        else:
            alpha_long = overnight_alpha_long if overnight_alpha_long is not None else getattr(model, "overnight_alpha_long", 0.0)
            alpha_short = overnight_alpha_short if overnight_alpha_short is not None else getattr(model, "overnight_alpha_short", 0.0)
        fin_annual = buy_interest_annual if buy_interest_annual is not None else getattr(model, "buy_interest_annual", 0.025)
        borrow_annual = borrow_fee_annual if borrow_fee_annual is not None else getattr(model, "borrow_fee_annual", 0.0115)
        rev_bps = reverse_fee_bps if reverse_fee_bps is not None else getattr(model, "reverse_fee_bps", 2.0)
        logger.info(
            f"Starting generic backtest: start={start_date}, slippage={slip_bps} bps, "
            f"alpha_long={alpha_long}, alpha_short={alpha_short}, "
            f"financing={fin_annual*100:.2f}% ann, "
            f"borrow={borrow_annual*100:.2f}% ann, reverse={rev_bps:.1f} bps/day"
        )

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
        sigma_yy_array = pred.get("sigma_yy", None)
        for i in range(start_idx, end_idx + 1):
            sigma_yy_i = sigma_yy_array[i] if sigma_yy_array is not None else None
            sre_weights[i] = model.build_weights(sre_signals_df.iloc[i].values, Sigma_YY=sigma_yy_i)

        sre_weights_df = pd.DataFrame(
            sre_weights[start_idx : end_idx + 1], index=sim_dates_slice, columns=JP_TICKERS
        )

        y_jp_oc_df = pred["y_jp_oc_df"]

        # Compute 9:10-to-close target returns for JP assets
        from leadlag.models.sre import compute_jp_target_returns
        y_jp_target = compute_jp_target_returns(df_exec, JP_TICKERS)
        y_jp_target_df = pd.DataFrame(y_jp_target, index=sim_dates, columns=JP_TICKERS)

        # Overnight gap returns: gap(t) = open(t)/close(t-1) - 1
        gap_cols = [f"jp_gap_{tk}" for tk in JP_TICKERS]
        if all(c in df_exec.columns for c in gap_cols):
            gap_returns_df = df_exec[gap_cols].copy()
            gap_returns_df.columns = JP_TICKERS
        else:
            gap_returns_df = pd.DataFrame(
                0.0, index=sim_dates, columns=JP_TICKERS
            )

        # Cost parameters
        slip = slip_bps / 10000.0
        financing_daily = fin_annual / 365.0
        borrow_daily = borrow_annual / 365.0
        reverse_daily = rev_bps / 10000.0

        # Returns and Cost drag calculations
        gross_returns_list = []
        net_returns_list = []
        gross_returns_oc_list = []
        net_returns_oc_list = []
        cost_list = []
        slip_cost_list = []
        financing_cost_list = []
        borrow_cost_list = []
        reverse_cost_list = []
        overnight_ret_list = []
        gross_exp_list = []
        turnover_list = []

        w_prev = np.zeros(model.n_j)
        dates_list = list(sim_dates_slice)
        for i, date in enumerate(dates_list):
            w_t = sre_weights_df.loc[date].values
            r_target_t = y_jp_target_df.loc[date].values
            r_oc_t = y_jp_oc_df.loc[date].values

            # Intraday return (9:10-to-Close) — same for all alpha
            gross_ret = float(np.sum(w_t * r_target_t))
            gross_exp = float(np.sum(np.abs(w_t)))
            long_exp = float(np.sum(np.maximum(w_t, 0.0)))
            short_exp = float(np.sum(np.maximum(-w_t, 0.0)))

            # Per-asset alpha mask: long positions use alpha_long, short uses alpha_short
            alpha_mask = np.where(w_t > 0, alpha_long, np.where(w_t < 0, alpha_short, 0.0))

            # Overnight return: sum over assets of alpha_mask[j] * w_t[j] * gap(t+1)[j]
            overnight_ret = 0.0
            if (alpha_long > 0 or alpha_short > 0) and i < len(dates_list) - 1:
                next_date = dates_list[i + 1]
                if next_date in gap_returns_df.index:
                    r_gap_next = gap_returns_df.loc[next_date].values
                    overnight_ret = float(np.sum(alpha_mask * w_t * r_gap_next))

            # Cost model:
            # (1-alpha_mask[j]) fraction: full round-trip (close at 15:00, reopen at 9:10)
            # alpha_mask[j] fraction: only rebalance cost (hold overnight, adjust at 9:10)
            turnover = float(np.sum(np.abs(w_t - w_prev)) / 2.0)

            slip_cost = slip * (2.0 * np.sum((1.0 - alpha_mask) * np.abs(w_t)) + np.sum(alpha_mask * np.abs(w_t - w_prev) / 2.0))
            held_long = float(np.sum(alpha_mask * np.maximum(w_t, 0.0)))
            held_short = float(np.sum(alpha_mask * np.maximum(-w_t, 0.0)))
            fin_cost = held_long * financing_daily
            borrow_cost = held_short * borrow_daily
            reverse_cost = held_short * reverse_daily
            cost = slip_cost + fin_cost + borrow_cost + reverse_cost

            # Net return = intraday + overnight - total cost
            net_ret = gross_ret + overnight_ret - cost

            # Auxiliary (Open-to-Close) — no overnight component for OC measure
            gross_ret_oc = float(np.sum(w_t * r_oc_t))
            net_ret_oc = gross_ret_oc - cost

            gross_returns_list.append(gross_ret + overnight_ret)
            net_returns_list.append(net_ret)
            gross_returns_oc_list.append(gross_ret_oc)
            net_returns_oc_list.append(net_ret_oc)
            cost_list.append(cost)
            slip_cost_list.append(slip_cost)
            financing_cost_list.append(fin_cost)
            borrow_cost_list.append(borrow_cost)
            reverse_cost_list.append(reverse_cost)
            overnight_ret_list.append(overnight_ret)
            gross_exp_list.append(gross_exp)
            turnover_list.append(turnover)

            w_prev = w_t

        daily_returns_gross = pd.Series(gross_returns_list, index=sim_dates_slice)
        daily_returns_net = pd.Series(net_returns_list, index=sim_dates_slice)
        daily_returns_gross_oc = pd.Series(gross_returns_oc_list, index=sim_dates_slice)
        daily_returns_net_oc = pd.Series(net_returns_oc_list, index=sim_dates_slice)
        daily_costs = pd.Series(cost_list, index=sim_dates_slice)
        daily_slip_costs = pd.Series(slip_cost_list, index=sim_dates_slice)
        daily_financing_costs = pd.Series(financing_cost_list, index=sim_dates_slice)
        daily_borrow_costs = pd.Series(borrow_cost_list, index=sim_dates_slice)
        daily_reverse_costs = pd.Series(reverse_cost_list, index=sim_dates_slice)
        daily_overnight_returns = pd.Series(overnight_ret_list, index=sim_dates_slice)
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
