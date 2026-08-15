"""Experiment: Surpassing Baseline V2 (Higher Return + Lower Risk).

Tests 4 advanced quantitative enhancements over Baseline V2:
  1. Non-linear Power Scaling (gamma in [1.2, 1.4, 1.6, 1.8, 2.0])
  2. Adaptive Z-score Thresholding (Dynamic K selection)
  3. Risk-Parity Normalized Alpha (score / sigma_i^p)
  4. Non-linear Convex Optimization (Power-scaled Alpha + Covariance Penalty)

Goal: Achieve Annual Return > 199.14%, Volatility < 48.71%, Net Sharpe > 4.10+.
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


def solve_power_scaled_baseline(
    scores: np.ndarray,
    gamma: float = 1.5,
    top_k: int = 5,
    baseline_gross: float = 2.0,
) -> np.ndarray:
    """Non-linear power scaling: emphasizes high-conviction alpha signals."""
    n = len(scores)
    w = np.zeros(n)
    med_score = np.median(scores)
    scores_centered = scores - med_score

    order = np.argsort(scores)
    short_idx = order[:top_k]
    long_idx = order[-top_k:]

    # Long side power scaling
    long_raw = np.maximum(scores_centered[long_idx], 1e-12) ** gamma
    long_denom = np.sum(long_raw)
    if long_denom > 0:
        w[long_idx] = (baseline_gross / 2.0) * (long_raw / long_denom)

    # Short side power scaling
    short_raw = np.maximum(-scores_centered[short_idx], 1e-12) ** gamma
    short_denom = np.sum(short_raw)
    if short_denom > 0:
        w[short_idx] = -(baseline_gross / 2.0) * (short_raw / short_denom)

    return w


def solve_vol_parity_baseline(
    scores: np.ndarray,
    vols: np.ndarray,
    top_k: int = 5,
    baseline_gross: float = 2.0,
) -> np.ndarray:
    """Risk-parity normalized alpha: score / sigma_i prevents high-vol sector overexposure."""
    n = len(scores)
    w = np.zeros(n)
    norm_scores = scores / np.maximum(vols, 1e-4)
    med_score = np.median(norm_scores)
    scores_centered = norm_scores - med_score

    order = np.argsort(norm_scores)
    short_idx = order[:top_k]
    long_idx = order[-top_k:]

    long_raw = np.maximum(scores_centered[long_idx], 1e-12)
    long_denom = np.sum(long_raw)
    if long_denom > 0:
        w[long_idx] = (baseline_gross / 2.0) * (long_raw / long_denom)

    short_raw = np.maximum(-scores_centered[short_idx], 1e-12)
    short_denom = np.sum(short_raw)
    if short_denom > 0:
        w[short_idx] = -(baseline_gross / 2.0) * (short_raw / short_denom)

    return w


def run_experiment(df_exec: pd.DataFrame) -> dict:
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

    strat_cfg = app_config.strategy
    methods = {}

    # 1. Baseline V2
    weights_base = np.zeros((T_sim, n_j))
    for t_idx in range(T_sim):
        scores = full_score_list[t_idx]
        order = np.argsort(scores)
        short_idx = order[:5]
        long_idx = order[-5:]
        med_score = np.median(scores)
        sc = scores - med_score
        w = np.zeros(n_j)
        l_raw = np.maximum(sc[long_idx], 1e-12)
        s_raw = np.maximum(-sc[short_idx], 1e-12)
        target_g = cfg.baseline_gross * gross_mult_list[t_idx]
        w[long_idx] = (target_g / 2.0) * (l_raw / np.sum(l_raw))
        w[short_idx] = -(target_g / 2.0) * (s_raw / np.sum(s_raw))
        weights_base[t_idx] = w
    methods["Baseline V2 (Benchmark)"] = weights_base

    # 2. Power Scaling (gamma in 1.2, 1.4, 1.6, 1.8, 2.0)
    for gamma in [1.2, 1.4, 1.5, 1.6, 1.8, 2.0]:
        weights_p = np.zeros((T_sim, n_j))
        for t_idx in range(T_sim):
            target_g = cfg.baseline_gross * gross_mult_list[t_idx]
            w = solve_power_scaled_baseline(
                scores=full_score_list[t_idx],
                gamma=gamma,
                top_k=5,
                baseline_gross=target_g,
            )
            weights_p[t_idx] = w
        methods[f"Power Scaling (gamma={gamma:.1f})"] = weights_p

    # 3. Vol Parity Normalized Alpha
    weights_vp = np.zeros((T_sim, n_j))
    for t_idx in range(T_sim):
        target_g = cfg.baseline_gross * gross_mult_list[t_idx]
        vols = np.sqrt(np.maximum(np.diag(omega_h1_list[t_idx]), 1e-8))
        w = solve_vol_parity_baseline(
            scores=full_score_list[t_idx],
            vols=vols,
            top_k=5,
            baseline_gross=target_g,
        )
        weights_vp[t_idx] = w
    methods["Vol-Parity Normalized Alpha"] = weights_vp

    # 4. Combined: Power Scaling + Vol Parity
    for gamma in [1.3, 1.5, 1.7]:
        weights_comb = np.zeros((T_sim, n_j))
        for t_idx in range(T_sim):
            target_g = cfg.baseline_gross * gross_mult_list[t_idx]
            vols = np.sqrt(np.maximum(np.diag(omega_h1_list[t_idx]), 1e-8))
            norm_scores = full_score_list[t_idx] / np.maximum(vols, 1e-4)
            w = solve_power_scaled_baseline(
                scores=norm_scores,
                gamma=gamma,
                top_k=5,
                baseline_gross=target_g,
            )
            weights_comb[t_idx] = w
        methods[f"Combined (Vol-Parity + gamma={gamma:.1f})"] = weights_comb

    # Evaluate all
    results = {}
    print("\n" + "=" * 95)
    print(f"{'Strategy / Method':<38} | {'Net Sharpe':<12} | {'Annual Return':<14} | {'Annual Vol':<12} | {'MDD':<10} | {'Turnover':<10}")
    print("=" * 95)

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
        print(f"{name:<38} | {res['net_sharpe']:>12.4f} | {res['ar']*100:>13.2f}% | {res['vol']*100:>11.2f}% | {res['mdd']*100:>9.2f}% | {res['turnover']:>10.4f}")

    print("=" * 95)
    return results


def main() -> None:
    print("=== Testing Methods to Beat Baseline V2 (Higher Return + Lower Risk) ===")
    df_exec = load_df_exec_from_local_cache()
    if df_exec is None:
        print("df_exec not found.")
        return

    run_experiment(df_exec)


if __name__ == "__main__":
    main()
