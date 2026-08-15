"""Experiment Step 3: Max Single Weight & Alpha Utility Scaling for Convex Optimizer.

Investigates whether increasing max_single_weight (from 0.25 to 0.30, 0.35, 0.40, 0.50)
and scaling alpha utility enables the Convex Optimizer to fully capture strong sector signals
and bridge the return gap to 199%+ (Sharpe 4.10+).
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
from leadlag.models.signal_enhancement import apply_rank_reversal_overlay
from leadlag.reporting.metrics import calculate_metrics


def run_weight_cap_sweep(df_exec: pd.DataFrame) -> dict:
    app_config = load_config_from_yaml("configs/production/production.yaml")
    cfg = app_config.v2
    blpx_model = ProductionBLPXModel(cfg.model_dump())
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

    print(f"Pre-computing full Multi-Horizon signals for {T_sim} days...")
    omega_h1_list = []
    full_score_list = []
    pit_ir_history = []
    gross_mult_list = []

    for t_idx, sim_dt in enumerate(sim_dates_slice):
        trade_date_str = str(sim_dt)
        current_prices = _build_current_prices_from_df_exec(df_exec, trade_date_str)

        try:
            mu1, om1 = v2_model._compute_ondemand(
                trade_date=trade_date_str,
                df_exec=df_exec,
                current_prices=current_prices,
                horizon=1,
            )
        except Exception as e:
            print(f"Warning: distribution computation failed for {trade_date_str}: {e}. Skipping day.")
            continue

        sig1 = np.sqrt(np.maximum(np.diag(om1), 1e-8))
        score1 = mu1 / sig1

        try:
            mu_mh, om_mh, score_mh = v2_model._multi_horizon_scores(
                trade_date=trade_date_str,
                df_exec=df_exec,
                current_prices=current_prices,
                use_file_cache=True,
            )
        except Exception as e:
            print(f"Warning: multi-horizon scores failed for {trade_date_str}: {e}. Falling back to single horizon.")
            _, _, score_mh = mu1, om1, score1

        if score_mh is None:
            score_mh = score1
        if om_mh is None:
            om_mh = om1

        score_full, _ = apply_rank_reversal_overlay(
            scores=score_mh,
            gap_input_dir=Path("var/live/pipeline_data/gap_adjusted_distribution/latest"),
            date_str=trade_date_str.replace("-", ""),
            weight=0.05,
        )

        raw_ir = float(np.mean(score1))
        if len(pit_ir_history) >= cfg.pit_rolling_window:
            _, _, _, gross_mult = get_rolling_pit_bin(
                history_ir=np.array(pit_ir_history),
                current_ir=raw_ir,
                rolling_window=cfg.pit_rolling_window,
                low_pct=cfg.tertile_low_pct,
                high_pct=cfg.tertile_high_pct,
                mult_low=cfg.mult_low,
                mult_mid=cfg.mult_mid,
                mult_high=cfg.mult_high,
            )
        else:
            gross_mult = cfg.fallback_multiplier

        pit_ir_history.append(raw_ir)
        gross_mult_list.append(gross_mult)
        omega_h1_list.append(om_mh)
        full_score_list.append(score_full)

    max_weight_options = [0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    strat_cfg = app_config.strategy
    results = {}

    for max_w in max_weight_options:
        print(f"\n--- Testing Max Single Weight = {max_w:.2f} ---")
        weights = np.zeros((T_sim, n_j))
        w_prev = np.zeros(n_j)
        opt_config = ConvexOptimizerConfig(
            lambda_risk=1.0,
            cost_bps=5.0,
            turnover_penalty=0.00005,
            max_single_weight=max_w,
            gross_target=cfg.baseline_gross,
        )

        for t_idx in range(T_sim):
            score_vec = full_score_list[t_idx]
            sig_diag = np.sqrt(np.maximum(np.diag(omega_h1_list[t_idx]), 1e-8))
            alpha_vec = score_vec * sig_diag

            res = optimize_portfolio_convex(
                mu_gap=alpha_vec,
                omega_gap=omega_h1_list[t_idx],
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

        results[max_w] = {
            "net_sharpe": net_m.get("Sharpe", 0.0),
            "gross_sharpe": gross_m.get("Sharpe", 0.0),
            "ar": net_m.get("AR", 0.0),
            "vol": net_m.get("RISK", 0.0),
            "mdd": net_m.get("MDD", 0.0),
            "turnover": float(np.mean(pnl["turnover"])),
        }
        print(f"  Result (max_w={max_w:.2f}): Net Sharpe = {net_m.get('Sharpe', 0.0):.4f}, AR = {net_m.get('AR', 0.0)*100:.2f}%, Vol = {net_m.get('RISK', 0.0)*100:.2f}%, MDD = {net_m.get('MDD', 0.0)*100:.2f}%, Turnover = {np.mean(pnl['turnover']):.4f}")

    return results


def main() -> None:
    print("=== Step 3: Max Single Weight Sensitivity for Convex Optimizer ===")
    df_exec = load_df_exec_from_local_cache()
    if df_exec is None:
        print("df_exec not found.")
        return

    results = run_weight_cap_sweep(df_exec)

    print("\n" + "=" * 85)
    print(f"{'max_single_weight':<18} | {'Net Sharpe':<12} | {'Annual Return':<14} | {'Annual Vol':<12} | {'MDD':<10} | {'Turnover':<10}")
    print("=" * 85)
    for max_w, res in results.items():
        print(f"{max_w:<18.2f} | {res['net_sharpe']:>12.4f} | {res['ar']*100:>13.2f}% | {res['vol']*100:>11.2f}% | {res['mdd']*100:>9.2f}% | {res['turnover']:>10.4f}")
    print("=" * 85)
    print("Baseline Production V2 Benchmark: Net Sharpe = 4.0883, AR = 199.14%, Vol = 48.71%, MDD = -8.62%")

if __name__ == "__main__":
    main()
