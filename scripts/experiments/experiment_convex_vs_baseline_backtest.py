"""Experiment: Compare Baseline Production V2 vs Unified Convex Optimization Backtest.

Period: 2015-01-05 to latest (full historical backtest).
Models compared:
  1. Baseline V2: Heuristic mu_over_sigma ranking + Top-5 Long / Bottom-5 Short + RuleD
  2. Convex V2: Single-stage Convex Portfolio Optimization (QP/SLSQP) + RuleD dynamic gross

Evaluates:
  - Net Sharpe Ratio (after slippage + financing + borrow + reverse fees)
  - Total Net Return & Annualized Return
  - Max Drawdown (MDD)
  - Daily Turnover
  - Ex-ante vs Realized metrics
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from leadlag.config.schemas import ProductionV2RunConfig
from leadlag.core.convex_optimizer import (
    ConvexOptimizerConfig,
    optimize_portfolio_convex,
)
from leadlag.core.portfolio import get_rolling_pit_bin
from leadlag.data.market_data_cache import load_df_exec_from_local_cache
from leadlag.data.preprocessor import compute_jp_target_returns
from leadlag.data.tickers import JP_TICKERS
from leadlag.execution.backtester import BacktestEngine
from leadlag.execution.config import load_config_from_yaml
from leadlag.models.blpx import ProductionBLPXModel
from leadlag.models.production_v2 import (
    ProductionV2Model,
    _build_current_prices_from_df_exec,
)


def run_convex_backtest(
    df_exec: pd.DataFrame,
    start_date: str = "2015-01-05",
    lambda_risk: float = 5.0,
    cost_bps: float = 5.0,
    turnover_penalty: float = 0.0001,
    max_single_weight: float = 0.25,
) -> dict:
    """Run full historical backtest using Convex Portfolio Optimizer."""
    # 1. Setup V2 model for distribution computation
    app_config = load_config_from_yaml("configs/production/production.yaml")
    cfg = app_config.v2
    blpx_model = ProductionBLPXModel(app_config.model_dump())
    v2_model = ProductionV2Model(cfg, blpx_model=blpx_model)

    # 2. Setup dates and targets
    sim_dates, start_idx, end_idx = BacktestEngine._resolve_sim_dates(
        df_exec, start_date, "latest", min_start_idx=250
    )
    sim_dates_slice = sim_dates[start_idx : end_idx + 1]
    T_sim = len(sim_dates_slice)
    n_j = len(JP_TICKERS)

    y_jp_target_arr, gap_returns_arr = BacktestEngine._compute_target_and_gap_returns(
        df_exec, sim_dates, sim_dates_slice
    )

    # 3. Simulate daily weights
    weights = np.zeros((T_sim, n_j))
    pit_ir_history = []
    opt_config = ConvexOptimizerConfig(
        lambda_risk=lambda_risk,
        cost_bps=cost_bps,
        turnover_penalty=turnover_penalty,
        max_single_weight=max_single_weight,
        gross_target=cfg.baseline_gross,
    )

    print(f"Running Convex Optimization Simulation over {T_sim} trading days...")
    w_prev = np.zeros(n_j)

    for t_idx, sim_dt in enumerate(sim_dates_slice):
        trade_date_str = str(sim_dt)
        current_prices = _build_current_prices_from_df_exec(df_exec, trade_date_str)

        # On-demand distribution
        try:
            mu_gap, omega_gap = v2_model._compute_ondemand(
                trade_date=trade_date_str,
                df_exec=df_exec,
                current_prices=current_prices,
                horizon=1,
            )
        except Exception:
            mu_gap, omega_gap = np.zeros(n_j), np.eye(n_j)

        # Ex-ante IR for RuleD PIT binning
        sigma_gap = np.sqrt(np.maximum(np.diag(omega_gap), 1e-8))
        raw_ir = float(np.mean(mu_gap / sigma_gap)) if len(mu_gap) > 0 else 0.0

        # RuleD dynamic gross scaling (using strictly historical PIT history)
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

        # Solve convex optimization
        res = optimize_portfolio_convex(
            mu_gap=mu_gap,
            omega_gap=omega_gap,
            w_prev=w_prev,
            config=opt_config,
            gross_multiplier=gross_mult,
        )

        weights[t_idx] = res.weights
        w_prev = res.weights.copy()

        if (t_idx + 1) % 500 == 0 or (t_idx + 1) == T_sim:
            print(f"  Processed {t_idx + 1}/{T_sim} days...")

    # 4. Simulate PnL using standard BacktestEngine logic
    strat_cfg = app_config.strategy
    pnl_results = BacktestEngine._simulate_daily_pnl(
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

    return pnl_results


def main() -> None:
    print("=== Backtest Comparison: Baseline Production V2 vs Unified Convex Optimization ===")
    
    df_exec = load_df_exec_from_local_cache()
    if df_exec is None:
        print("df_exec not found in local cache.")
        return

    # 1. Run Baseline V2 Backtest
    print("\n--- 1. Running Baseline Production V2 Backtest ---")
    app_config = load_config_from_yaml("configs/production/production.yaml")
    gap_dir = "var/live/pipeline_data/gap_adjusted_distribution/latest"
    base_results = BacktestEngine.run_v2_backtest(
        cfg=app_config,
        gap_input_dir=gap_dir,
        df_exec=df_exec,
        start_date="2015-01-05",
        end_date="latest",
    )

    # 2. Run Convex Optimization Backtest
    print("\n--- 2. Running Next-Gen Convex Optimization Backtest ---")
    convex_pnl = run_convex_backtest(
        df_exec=df_exec,
        start_date="2015-01-05",
        lambda_risk=5.0,
        cost_bps=5.0,
        turnover_penalty=0.0001,
        max_single_weight=0.25,
    )

    from leadlag.reporting.metrics import calculate_metrics

    base_net_metrics = calculate_metrics(base_results["daily_returns"])
    base_gross_metrics = calculate_metrics(base_results["daily_returns_gross"])
    
    sim_dates, start_idx, end_idx = BacktestEngine._resolve_sim_dates(df_exec, "2015-01-05", "latest", 250)
    sim_dates_slice = sim_dates[start_idx : end_idx + 1]
    
    conv_net_series = pd.Series(convex_pnl["net_returns"], index=sim_dates_slice)
    conv_gross_series = pd.Series(convex_pnl["gross_returns"], index=sim_dates_slice)
    conv_net_metrics = calculate_metrics(conv_net_series)
    conv_gross_metrics = calculate_metrics(conv_gross_series)

    # 3. Print Comparison Metrics
    metrics_summary = [
        ("Net Sharpe Ratio", f"{base_net_metrics.get('Sharpe', 0.0):.4f}", f"{conv_net_metrics.get('Sharpe', 0.0):.4f}"),
        ("Gross Sharpe Ratio", f"{base_gross_metrics.get('Sharpe', 0.0):.4f}", f"{conv_gross_metrics.get('Sharpe', 0.0):.4f}"),
        ("Annualized Net Return", f"{base_net_metrics.get('AR', 0.0)*100:.2f}%", f"{conv_net_metrics.get('AR', 0.0)*100:.2f}%"),
        ("Annualized Net Vol", f"{base_net_metrics.get('RISK', 0.0)*100:.2f}%", f"{conv_net_metrics.get('RISK', 0.0)*100:.2f}%"),
        ("Return / Risk (R/R)", f"{base_net_metrics.get('R/R', 0.0):.4f}", f"{conv_net_metrics.get('R/R', 0.0):.4f}"),
        ("Total Net Return", f"{base_net_metrics.get('Total Return', 0.0)*100:.2f}%", f"{conv_net_metrics.get('Total Return', 0.0)*100:.2f}%"),
        ("Max Drawdown (MDD)", f"{base_net_metrics.get('MDD', 0.0)*100:.2f}%", f"{conv_net_metrics.get('MDD', 0.0)*100:.2f}%"),
        ("Average Daily Turnover", f"{base_results['daily_turnover'].mean():.4f}", f"{np.mean(convex_pnl['turnover']):.4f}"),
        ("Average Gross Exposure", f"{base_results['daily_gross_exps'].mean():.4f}", f"{np.mean(convex_pnl['gross_exps']):.4f}"),
        ("Total Slippage Cost", f"{base_results['daily_slip_costs'].sum()*100:.2f}%", f"{np.sum(convex_pnl['slip_costs'])*100:.2f}%"),
        ("Total Financing Cost", f"{base_results['daily_financing_costs'].sum()*100:.2f}%", f"{np.sum(convex_pnl['financing_costs'])*100:.2f}%"),
        ("Total Borrow Cost", f"{base_results['daily_borrow_costs'].sum()*100:.2f}%", f"{np.sum(convex_pnl['borrow_costs'])*100:.2f}%"),
        ("Total Reverse Fee", f"{base_results['daily_reverse_costs'].sum()*100:.2f}%", f"{np.sum(convex_pnl['reverse_costs'])*100:.2f}%"),
    ]

    print("\n" + "=" * 70)
    print(f"{'Performance Metric':<26} | {'Baseline Production V2':<20} | {'Next-Gen Convex Opt':<20}")
    print("=" * 70)
    for name, base_val, conv_val in metrics_summary:
        print(f"{name:<26} | {base_val:>20} | {conv_val:>20}")
    print("=" * 70)

    # Save summary report
    out_dir = Path("reports/nextgen_convex_backtest")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "convex_vs_baseline_comparison.md"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Next-Gen Convex Optimization vs Baseline Production V2 Backtest Report\n\n")
        f.write(f"**Period**: 2015-01-05 to {df_exec.index[-1].strftime('%Y-%m-%d')}\n\n")
        f.write("| Performance Metric | Baseline Production V2 | Next-Gen Convex Opt |\n")
        f.write("|---|---|---|\n")
        for name, base_val, conv_val in metrics_summary:
            f.write(f"| {name} | {base_val} | {conv_val} |\n")
    print(f"\nReport saved to: {report_path}")

if __name__ == "__main__":
    main()
