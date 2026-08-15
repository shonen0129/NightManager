"""Experiment Step 1: Lambda Risk Sensitivity Analysis for Convex Optimizer.

Evaluates lambda_risk in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0] across full 2015-2026 backtest.
Investigates how relaxing risk penalty restores alpha concentration and Sharpe ratio.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from leadlag.core.convex_optimizer import (
    ConvexOptimizerConfig,
    optimize_portfolio_convex,
)
from leadlag.core.portfolio import get_rolling_pit_bin
from leadlag.data.market_data_cache import load_df_exec_from_local_cache
from leadlag.data.tickers import JP_TICKERS
from leadlag.execution.backtester import BacktestEngine
from leadlag.execution.config import load_config_from_yaml
from leadlag.models.blpx import ProductionBLPXModel
from leadlag.models.production_v2 import (
    ProductionV2Model,
    _build_current_prices_from_df_exec,
)
from leadlag.reporting.metrics import calculate_metrics


def run_sweep(df_exec: pd.DataFrame, lambda_list: list[float]) -> dict:
    app_config = load_config_from_yaml("configs/production/production.yaml")
    cfg = app_config.v2
    blpx_model = ProductionBLPXModel(app_config.model_dump())
    v2_model = ProductionV2Model(cfg, blpx_model=blpx_model)

    sim_dates, start_idx, end_idx = BacktestEngine._resolve_sim_dates(
        df_exec, "2015-01-05", "latest", min_start_idx=250
    )
    sim_dates_slice = sim_dates[start_idx : end_idx + 1]
    T_sim = len(sim_dates_slice)
    n_j = len(JP_TICKERS)

    y_jp_target_arr, gap_returns_arr = BacktestEngine._compute_target_and_gap_returns(
        df_exec, sim_dates, sim_dates_slice
    )

    print("Pre-computing daily on-demand distributions...")
    mu_list = []
    omega_list = []
    pit_ir_history = []
    gross_mult_list = []

    for t_idx, sim_dt in enumerate(sim_dates_slice):
        trade_date_str = str(sim_dt)
        current_prices = _build_current_prices_from_df_exec(df_exec, trade_date_str)
        try:
            mu_gap, omega_gap = v2_model._compute_ondemand(
                trade_date=trade_date_str,
                df_exec=df_exec,
                current_prices=current_prices,
                horizon=1,
            )
        except Exception:
            mu_gap, omega_gap = np.zeros(n_j), np.eye(n_j)

        sigma_gap = np.sqrt(np.maximum(np.diag(omega_gap), 1e-8))
        raw_ir = float(np.mean(mu_gap / sigma_gap)) if len(mu_gap) > 0 else 0.0

        if len(pit_ir_history) >= cfg.pit_rolling_window:
            _, _, _, gross_mult = get_rolling_pit_bin(
                history_ir=np.array(pit_ir_history),
                current_ir=raw_ir,
                rolling_window=cfg.pit_rolling_window,
                mult_low=cfg.mult_low,
                mult_mid=cfg.mult_mid,
                mult_high=cfg.mult_high,
            )
        else:
            gross_mult = cfg.fallback_multiplier

        pit_ir_history.append(raw_ir)
        mu_list.append(mu_gap)
        omega_list.append(omega_gap)
        gross_mult_list.append(gross_mult)

        if (t_idx + 1) % 500 == 0 or (t_idx + 1) == T_sim:
            print(f"  Distribution prepared {t_idx + 1}/{T_sim} days...")

    sweep_results = {}
    strat_cfg = app_config.strategy

    for l_risk in lambda_list:
        print(f"\n--- Running Convex Optimization with lambda_risk = {l_risk} ---")
        weights = np.zeros((T_sim, n_j))
        w_prev = np.zeros(n_j)
        opt_config = ConvexOptimizerConfig(
            lambda_risk=l_risk,
            cost_bps=5.0,
            turnover_penalty=0.0001,
            max_single_weight=0.25,
            gross_target=cfg.baseline_gross,
        )

        for t_idx in range(T_sim):
            res = optimize_portfolio_convex(
                mu_gap=mu_list[t_idx],
                omega_gap=omega_list[t_idx],
                w_prev=w_prev,
                config=opt_config,
                gross_multiplier=gross_mult_list[t_idx],
            )
            weights[t_idx] = res.weights
            w_prev = res.weights.copy()

        pnl = BacktestEngine._simulate_daily_pnl(
            weights=weights,
            target_returns=y_jp_target_arr,
            gap_returns=gap_returns_arr,
            sim_dates=sim_dates_slice,
            slip=strat_cfg.slippage_bps * 1e-4,
            financing_daily=strat_cfg.buy_interest_annual / 365.0,
            borrow_daily=strat_cfg.borrow_fee_annual / 365.0,
            reverse_daily=strat_cfg.reverse_fee_bps * 1e-4,
            alpha_long=strat_cfg.overnight_alpha_long,
            alpha_short=strat_cfg.overnight_alpha_short,
            side_leverage=strat_cfg.side_leverage,
        )

        net_series = pd.Series(pnl["net_returns"], index=sim_dates_slice)
        gross_series = pd.Series(pnl["gross_returns"], index=sim_dates_slice)
        net_m = calculate_metrics(net_series)
        gross_m = calculate_metrics(gross_series)

        sweep_results[l_risk] = {
            "net_sharpe": net_m.get("Sharpe", 0.0),
            "gross_sharpe": gross_m.get("Sharpe", 0.0),
            "ar": net_m.get("AR", 0.0),
            "vol": net_m.get("RISK", 0.0),
            "mdd": net_m.get("MDD", 0.0),
            "turnover": float(np.mean(pnl["turnover"])),
            "gross_exp": float(np.mean(pnl["gross_exps"])),
            "total_slip": float(np.sum(pnl["slip_costs"])),
        }
        print(f"  Result (lambda={l_risk}): Net Sharpe = {net_m.get('Sharpe', 0.0):.4f}, AR = {net_m.get('AR', 0.0)*100:.2f}%, Vol = {net_m.get('RISK', 0.0)*100:.2f}%, MDD = {net_m.get('MDD', 0.0)*100:.2f}%, Turnover = {np.mean(pnl['turnover']):.4f}")

    return sweep_results


def main() -> None:
    print("=== Step 1: Lambda Risk Sensitivity Analysis for Convex Optimizer ===")
    df_exec = load_df_exec_from_local_cache()
    if df_exec is None:
        print("df_exec not found.")
        return

    lambda_list = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    results = run_sweep(df_exec, lambda_list)

    print("\n" + "=" * 80)
    print(f"{'lambda_risk':<12} | {'Net Sharpe':<12} | {'Gross Sharpe':<14} | {'Annual Return':<14} | {'Annual Vol':<12} | {'MDD':<10} | {'Turnover':<10}")
    print("=" * 80)
    for l_risk, res in results.items():
        print(f"{l_risk:<12.1f} | {res['net_sharpe']:>12.4f} | {res['gross_sharpe']:>14.4f} | {res['ar']*100:>13.2f}% | {res['vol']*100:>11.2f}% | {res['mdd']*100:>9.2f}% | {res['turnover']:>10.4f}")
    print("=" * 80)

if __name__ == "__main__":
    main()
