"""Experiment Step 2: Score Formulation & Signal Feature Comparison for Convex Optimizer.

Compares 5 signal input formulations into Convex Optimizer across 2015-2026:
  1. Raw mu_gap (h=1)
  2. Risk-adjusted mu_over_sigma (h=1)
  3. Multi-horizon Raw mu (h=1, 3, 5)
  4. Multi-horizon Risk-adjusted score (h=1, 3, 5)
  5. Full Baseline features (Multi-horizon + Rank Reversal) into Convex Opt

Identifies the exact driver that brings Net Sharpe from 3.55 to 4.10+.
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


def run_formulation_sweep(df_exec: pd.DataFrame) -> dict:
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

    print(f"Pre-computing Multi-Horizon & Baseline signals for {T_sim} days...")
    raw_mu_h1 = []
    omega_h1 = []
    mu_over_sigma_h1 = []
    mh_score_list = []
    mh_raw_mu_list = []
    full_feature_score_list = []
    pit_ir_history = []
    gross_mult_list = []

    for t_idx, sim_dt in enumerate(sim_dates_slice):
        trade_date_str = str(sim_dt)
        current_prices = _build_current_prices_from_df_exec(df_exec, trade_date_str)

        # 1. Horizon 1
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

        # 2. Multi-horizon blend (h=1, 3, 5)
        # On-demand multi-horizon calculation
        try:
            mu_mh, om_mh, score_mh = v2_model._multi_horizon_scores(
                trade_date=trade_date_str,
                df_exec=df_exec,
                current_prices=current_prices,
                use_file_cache=True,
            )
        except Exception as e:
            print(f"Warning: multi-horizon scores failed for {trade_date_str}: {e}. Falling back to single horizon.")
            mu_mh, om_mh, score_mh = mu1, om1, score1

        if score_mh is None:
            score_mh = score1
        if mu_mh is None:
            mu_mh = mu1
        if om_mh is None:
            om_mh = om1

        # 3. Apply Rank Reversal Overlay
        score_full, _ = apply_rank_reversal_overlay(
            scores=score_mh,
            gap_input_dir=Path("var/live/pipeline_data/gap_adjusted_distribution/latest"),
            date_str=trade_date_str.replace("-", ""),
            weight=cfg.rank_reversal_weight if hasattr(cfg, "rank_reversal_weight") else 0.05,
        )

        # RuleD PIT binning
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

        raw_mu_h1.append(mu1)
        omega_h1.append(om1)
        mu_over_sigma_h1.append(score1)
        mh_score_list.append(score_mh)
        mh_raw_mu_list.append(mu_mh)
        full_feature_score_list.append(score_full)

        if (t_idx + 1) % 500 == 0 or (t_idx + 1) == T_sim:
            print(f"  Signal features prepared {t_idx + 1}/{T_sim} days...")

    # Define test formulations
    formulations = {
        "1. Raw mu_gap (h=1)": raw_mu_h1,
        "2. Risk-Adjusted mu/sigma (h=1)": mu_over_sigma_h1,
        "3. Multi-Horizon Raw mu (h=1,3,5)": mh_raw_mu_list,
        "4. Multi-Horizon Risk-Adjusted": mh_score_list,
        "5. Full Features (MH + Rank Reversal)": full_feature_score_list,
    }

    strat_cfg = app_config.strategy
    results = {}

    for name, alpha_inputs in formulations.items():
        print(f"\n--- Testing Formulation: {name} ---")
        weights = np.zeros((T_sim, n_j))
        w_prev = np.zeros(n_j)
        opt_config = ConvexOptimizerConfig(
            lambda_risk=2.0,
            cost_bps=5.0,
            turnover_penalty=0.0001,
            max_single_weight=0.25,
            gross_target=cfg.baseline_gross,
        )

        for t_idx in range(T_sim):
            # Scale score appropriately if using mu_over_sigma (normalize to expected return scale)
            alpha_vec = alpha_inputs[t_idx]
            if "Risk-Adjusted" in name or "Features" in name:
                # mu/sigma score converted to scaled expected return
                alpha_vec = alpha_vec * np.sqrt(np.maximum(np.diag(omega_h1[t_idx]), 1e-8))

            res = optimize_portfolio_convex(
                mu_gap=alpha_vec,
                omega_gap=omega_h1[t_idx],
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

        results[name] = {
            "net_sharpe": net_m.get("Sharpe", 0.0),
            "gross_sharpe": gross_m.get("Sharpe", 0.0),
            "ar": net_m.get("AR", 0.0),
            "vol": net_m.get("RISK", 0.0),
            "mdd": net_m.get("MDD", 0.0),
            "turnover": float(np.mean(pnl["turnover"])),
        }
        print(f"  Result: Net Sharpe = {net_m.get('Sharpe', 0.0):.4f}, AR = {net_m.get('AR', 0.0)*100:.2f}%, Vol = {net_m.get('RISK', 0.0)*100:.2f}%, MDD = {net_m.get('MDD', 0.0)*100:.2f}%, Turnover = {np.mean(pnl['turnover']):.4f}")

    return results


def main() -> None:
    print("=== Step 2: Signal Formulation Comparison for Convex Optimizer ===")
    df_exec = load_df_exec_from_local_cache()
    if df_exec is None:
        print("df_exec not found.")
        return

    results = run_formulation_sweep(df_exec)

    print("\n" + "=" * 95)
    print(f"{'Formulation':<40} | {'Net Sharpe':<12} | {'Annual Return':<14} | {'Annual Vol':<12} | {'MDD':<10} | {'Turnover':<10}")
    print("=" * 95)
    for name, res in results.items():
        print(f"{name:<40} | {res['net_sharpe']:>12.4f} | {res['ar']*100:>13.2f}% | {res['vol']*100:>11.2f}% | {res['mdd']*100:>9.2f}% | {res['turnover']:>10.4f}")
    print("=" * 95)
    print("Baseline Production V2 Benchmark: Net Sharpe = 4.0883, AR = 199.14%, Vol = 48.71%, MDD = -8.62%")

if __name__ == "__main__":
    main()
