"""Experiment: Signal Dispersion Adaptive Sizing + Power Scaling.

Tests scaling portfolio gross exposure based on Cross-Sectional Score Dispersion:
  Confidence = (std(scores_t) / rolling_mean(std(scores)))^beta
High dispersion (clear lead-lag signal) -> Scale up gross + Power Scaling
Low dispersion (noisy / ambiguous day) -> Scale down gross to protect capital

Evaluates on full 2015-2026 backtest.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

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


def solve_power_scaled_baseline(
    scores: np.ndarray,
    gamma: float = 1.4,
    top_k: int = 5,
    baseline_gross: float = 2.0,
) -> np.ndarray:
    n = len(scores)
    w = np.zeros(n)
    med_score = np.median(scores)
    scores_centered = scores - med_score

    order = np.argsort(scores)
    short_idx = order[:top_k]
    long_idx = order[-top_k:]

    long_raw = np.maximum(scores_centered[long_idx], 1e-12) ** gamma
    long_denom = np.sum(long_raw)
    if long_denom > 0:
        w[long_idx] = (baseline_gross / 2.0) * (long_raw / long_denom)

    short_raw = np.maximum(-scores_centered[short_idx], 1e-12) ** gamma
    short_denom = np.sum(short_raw)
    if short_denom > 0:
        w[short_idx] = -(baseline_gross / 2.0) * (short_raw / short_denom)

    return w


def run_dispersion_experiment(df_exec: pd.DataFrame) -> dict:
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

    print(f"Pre-computing Multi-Horizon & Baseline signals for {T_sim} days...")
    full_score_list = []
    pit_ir_history = []
    gross_mult_list = []
    score_std_list = []

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
        except Exception:
            mu1, om1 = np.zeros(n_j), np.eye(n_j)

        sig1 = np.sqrt(np.maximum(np.diag(om1), 1e-8))
        score1 = mu1 / sig1

        try:
            mu_mh, om_mh, score_mh = v2_model._multi_horizon_scores(
                trade_date=trade_date_str,
                df_exec=df_exec,
                current_prices=current_prices,
                use_file_cache=True,
            )
        except Exception:
            mu_mh, om_mh, score_mh = mu1, om1, score1

        if score_mh is None:
            score_mh = score1

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
                mult_low=cfg.mult_low,
                mult_mid=cfg.mult_mid,
                mult_high=cfg.mult_high,
            )
        else:
            gross_mult = cfg.fallback_multiplier

        pit_ir_history.append(raw_ir)
        gross_mult_list.append(gross_mult)
        full_score_list.append(score_full)
        score_std_list.append(float(np.std(score_full)))

    # Compute rolling score dispersion
    score_std_arr = np.array(score_std_list)
    rolling_disp_mean = pd.Series(score_std_arr).rolling(60, min_periods=20).mean().bfill().values
    disp_ratio = score_std_arr / np.maximum(rolling_disp_mean, 1e-6)

    strat_cfg = app_config.strategy
    methods = {}

    # 1. Baseline V2
    weights_base = np.zeros((T_sim, n_j))
    for t_idx in range(T_sim):
        target_g = cfg.baseline_gross * gross_mult_list[t_idx]
        w = solve_power_scaled_baseline(full_score_list[t_idx], gamma=1.0, top_k=5, baseline_gross=target_g)
        weights_base[t_idx] = w
    methods["Baseline V2 (Benchmark)"] = weights_base

    # 2. Power Scaling only (gamma=1.4)
    weights_p14 = np.zeros((T_sim, n_j))
    for t_idx in range(T_sim):
        target_g = cfg.baseline_gross * gross_mult_list[t_idx]
        w = solve_power_scaled_baseline(full_score_list[t_idx], gamma=1.4, top_k=5, baseline_gross=target_g)
        weights_p14[t_idx] = w
    methods["Power Scaling (gamma=1.4)"] = weights_p14

    # 3. Dispersion Adaptive Scaling + Power Scaling
    for clip_low, clip_high in [(0.70, 1.30), (0.80, 1.20), (0.60, 1.40)]:
        weights_disp = np.zeros((T_sim, n_j))
        for t_idx in range(T_sim):
            # Scale gross dynamically with score dispersion
            scale_disp = float(np.clip(disp_ratio[t_idx], clip_low, clip_high))
            target_g = cfg.baseline_gross * gross_mult_list[t_idx] * scale_disp
            w = solve_power_scaled_baseline(full_score_list[t_idx], gamma=1.4, top_k=5, baseline_gross=target_g)
            weights_disp[t_idx] = w
        methods[f"Dispersion Sizing [{clip_low:.1f}-{clip_high:.1f}] + Gamma 1.4"] = weights_disp

    # Evaluate
    results = {}
    print("\n" + "=" * 105)
    print(f"{'Strategy / Enhancement':<48} | {'Net Sharpe':<12} | {'Annual Return':<14} | {'Annual Vol':<12} | {'MDD':<10} | {'Turnover':<10}")
    print("=" * 105)

    for name, w_mat in methods.items():
        pnl = BacktestEngine._simulate_daily_pnl(
            weights=w_mat,
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

        res = {
            "net_sharpe": net_m.get("Sharpe", 0.0),
            "gross_sharpe": gross_m.get("Sharpe", 0.0),
            "ar": net_m.get("AR", 0.0),
            "vol": net_m.get("RISK", 0.0),
            "mdd": net_m.get("MDD", 0.0),
            "turnover": float(np.mean(pnl["turnover"])),
        }
        results[name] = res
        print(f"{name:<48} | {res['net_sharpe']:>12.4f} | {res['ar']*100:>13.2f}% | {res['vol']*100:>11.2f}% | {res['mdd']*100:>9.2f}% | {res['turnover']:>10.4f}")

    print("=" * 105)
    return results


def main() -> None:
    print("=== Testing Signal Dispersion Adaptive Sizing ===")
    df_exec = load_df_exec_from_local_cache()
    if df_exec is None:
        print("df_exec not found.")
        return

    run_dispersion_experiment(df_exec)


if __name__ == "__main__":
    main()
