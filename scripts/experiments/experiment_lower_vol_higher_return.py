"""Experiment: True Alpha Enhancement (Higher Return + STRICTLY Lower Volatility).

Evaluates methods on the FULL Production V2 Pipeline (with MinVar, Macro Kappa, BLPX Prior):
  1. Baseline Production V2 (Benchmark: AR ~199.14%, Vol ~48.71%, Sharpe ~4.088)
  2. Volatility-Targeted Scaling (Ex-ante Vol normalization to 40% / 35%)
  3. MinVar-Constrained Power Scaling (Alpha concentrated on hedged MinVar manifold)
  4. Dynamic Macro Volatility Dampener

Goal: Achieve Annual Return >= 199.14% AND Volatility < 48.71% (Sharpe > 4.10+).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from leadlag.data.market_data_cache import load_df_exec_from_local_cache
from leadlag.data.tickers import JP_TICKERS
from leadlag.execution.backtester import BacktestEngine
from leadlag.execution.config import load_config_from_yaml
from leadlag.reporting.metrics import calculate_metrics


def run_full_pipeline_test(df_exec: pd.DataFrame) -> dict:
    app_config = load_config_from_yaml("configs/production/production.yaml")
    gap_dir = "var/live/pipeline_data/gap_adjusted_distribution/latest"

    print("--- 1. Running Full Baseline Production V2 Backtest ---")
    base_results = BacktestEngine.run_v2_backtest(
        cfg=app_config,
        gap_input_dir=gap_dir,
        df_exec=df_exec,
        start_date="2015-01-05",
        end_date="latest",
    )
    base_net_m = calculate_metrics(base_results["daily_returns"])
    print(f"Baseline V2: Net Sharpe = {base_net_m['Sharpe']:.4f}, AR = {base_net_m['AR']*100:.2f}%, Vol = {base_net_m['RISK']*100:.2f}%, MDD = {base_net_m['MDD']*100:.2f}%")

    sim_dates, start_idx, end_idx = BacktestEngine._resolve_sim_dates(df_exec, "2015-01-05", "latest", 250)
    sim_dates_slice = sim_dates[start_idx : end_idx + 1]
    T_sim = len(sim_dates_slice)
    n_j = len(JP_TICKERS)
    strat_cfg = app_config.strategy

    y_jp_target_arr, gap_returns_arr = BacktestEngine._compute_target_and_gap_returns(
        df_exec, sim_dates, sim_dates_slice
    )

    # Extract base weights
    base_weights = base_results["weights"].values  # shape (T_sim, n_j)

    methods = {
        "Baseline Production V2 (Benchmark)": base_weights,
    }

    # 2. Volatility Targeting (Normalize daily portfolio ex-ante volatility)
    # Estimate realized rolling volatility and scale gross to target vol
    base_ret_series = base_results["daily_returns"]
    rolling_vol = base_ret_series.rolling(21, min_periods=10).std() * np.sqrt(245)
    rolling_vol_clean = rolling_vol.bfill().values

    target_vols = [0.40, 0.42, 0.45]
    for tv in target_vols:
        vol_scale = np.clip(tv / np.maximum(rolling_vol_clean, 0.10), 0.5, 1.5)
        w_vt = base_weights * vol_scale[:, None]
        methods[f"Volatility Targeting (Target Vol = {tv*100:.0f}%)"] = w_vt

    # 3. Non-linear Tail Amplification on Full Baseline Weights
    # Amplify top positions while reducing middle positions proportionally
    for p in [1.15, 1.25, 1.35]:
        w_tail = np.zeros_like(base_weights)
        for t in range(T_sim):
            w_t = base_weights[t]
            gross_t = np.sum(np.abs(w_t))
            if gross_t < 1e-6:
                continue
            # Apply power scaling preserving exact signs and gross
            pos_mask = w_t > 0
            neg_mask = w_t < 0
            w_new = np.zeros(n_j)
            if np.sum(pos_mask) > 0:
                pos_raw = (w_t[pos_mask]) ** p
                w_new[pos_mask] = (gross_t / 2.0) * (pos_raw / np.sum(pos_raw))
            if np.sum(neg_mask) > 0:
                neg_raw = (-w_t[neg_mask]) ** p
                w_new[neg_mask] = -(gross_t / 2.0) * (neg_raw / np.sum(neg_raw))
            w_tail[t] = w_new
        methods[f"Tail Amplification (p={p:.2f})"] = w_tail

    # 4. Tail Amplification + Volatility Targeting Combined
    for p, tv in [(1.20, 0.45), (1.25, 0.44), (1.30, 0.42)]:
        w_comb = np.zeros_like(base_weights)
        for t in range(T_sim):
            w_t = base_weights[t]
            gross_t = np.sum(np.abs(w_t))
            if gross_t < 1e-6:
                continue
            pos_mask = w_t > 0
            neg_mask = w_t < 0
            w_new = np.zeros(n_j)
            if np.sum(pos_mask) > 0:
                pos_raw = (w_t[pos_mask]) ** p
                w_new[pos_mask] = (gross_t / 2.0) * (pos_raw / np.sum(pos_raw))
            if np.sum(neg_mask) > 0:
                neg_raw = (-w_t[neg_mask]) ** p
                w_new[neg_mask] = -(gross_t / 2.0) * (neg_raw / np.sum(neg_raw))
            # Apply vol scale
            vs = float(np.clip(tv / max(rolling_vol_clean[t], 0.10), 0.5, 1.5))
            w_comb[t] = w_new * vs
        methods[f"Combined (Tail p={p:.2f} + TargetVol {tv*100:.0f}%)"] = w_comb

    # Evaluate
    results = {}
    print("\n" + "=" * 105)
    print(f"{'Strategy / Method':<45} | {'Net Sharpe':<12} | {'Annual Return':<14} | {'Annual Vol':<12} | {'MDD':<10} | {'Turnover':<10}")
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
        print(f"{name:<45} | {res['net_sharpe']:>12.4f} | {res['ar']*100:>13.2f}% | {res['vol']*100:>11.2f}% | {res['mdd']*100:>9.2f}% | {res['turnover']:>10.4f}")

    print("=" * 105)
    return results


def main() -> None:
    print("=== Testing True Alpha Enhancement (Higher Return + Lower Vol) ===")
    df_exec = load_df_exec_from_local_cache()
    if df_exec is None:
        print("df_exec not found.")
        return

    run_full_pipeline_test(df_exec)


if __name__ == "__main__":
    main()
