import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

# Import system modules
import data.ticker_registry as registry
import data.preprocessor as preprocessor
import strategy as sys_strategy
from runner.config import ProductionConfig
from domain.signals import lead_lag as signals
from performance import calculate_metrics

# Load approved sensitivities
from search_optimal_style_combination import make_custom_build_v3_static, BASE_US_TICKERS

import backtest.runner as runner

def run_backtest_for_comb(active_styles):
    active_us_tickers = BASE_US_TICKERS + active_styles
    n_us = len(active_us_tickers)

    # Monkeypatch registry and all targets
    registry.US_TICKERS = active_us_tickers
    registry.N_US = n_us
    registry.N_TOTAL = n_us + registry.N_JP
    registry.N_US_ASSETS = n_us
    registry.N_TOTAL_ASSETS = registry.N_TOTAL

    preprocessor.US_TICKERS = active_us_tickers
    sys_config_module = sys.modules.get("config")
    if sys_config_module:
        sys_config_module.N_US_ASSETS = n_us
        sys_config_module.N_TOTAL_ASSETS = registry.N_TOTAL

    sys_strategy.N_US_ASSETS = n_us
    sys_strategy.N_TOTAL_ASSETS = registry.N_TOTAL

    runner.N_US_ASSETS = n_us

    # Monkeypatch the build_v3_static function
    signals.build_v3_static = make_custom_build_v3_static(active_styles)

    # Force load raw cache and slice
    raw_data = pd.read_pickle(ROOT / "data" / "etf_data.pkl")
    us_close_sliced = raw_data["us_close"][active_us_tickers].copy()
    jp_close_sliced = raw_data["jp_close"].copy()
    jp_open_sliced = raw_data["jp_open"].copy()

    sliced_data = {
        "us_close": us_close_sliced,
        "jp_close": jp_close_sliced,
        "jp_open": jp_open_sliced
    }

    # Preprocess
    config = ProductionConfig(start_date="2015-01-01")
    df_exec = preprocessor.preprocess_data(sliced_data, beta_window=config.beta_window)

    # Run backtest
    strategy = sys_strategy.LeadLagStrategy(
        df_exec=df_exec,
        K=config.k,
        lambda_reg=config.lambda_reg,
        q=config.q,
        weight_mode=config.weight_mode,
        dispersion_filter=config.dispersion_filter,
        v3_mode=config.v3_mode,
        ewma_half_life=config.ewma_half_life,
        lambda_lw=config.lambda_lw,
        lw_target=config.lw_target,
        corr_window=config.corr_window,
        include_v4_prior=config.include_v4_prior,
        signal_mode=config.signal_mode,
        gap_open_coef=config.gap_open_coef,
        topix_beta_coef=config.topix_beta_coef,
        beta_window=config.beta_window,
        gamma=config.gamma,
    )
    results = strategy.run_backtest(start_date=config.start_date)
    return results["daily_return"]

def get_yearly_metrics(daily_returns: pd.Series) -> pd.DataFrame:
    years = daily_returns.index.year.unique()
    records = []
    for y in sorted(years):
        ret_y = daily_returns[daily_returns.index.year == y]
        n_days = len(ret_y)
        if n_days == 0:
            continue
        # Annualized Return (geometric mean)
        compounded = (1.0 + ret_y).prod()
        ar = compounded ** (252.0 / n_days) - 1.0
        # Volatility
        vol = ret_y.std() * np.sqrt(252.0)
        # R/R
        rr = ar / vol if vol > 0 else np.nan
        # Max Drawdown (MDD)
        cum = (1.0 + ret_y).cumprod()
        running_max = cum.cummax()
        drawdowns = (cum - running_max) / running_max
        mdd = drawdowns.min()
        
        records.append({
            "Year": y,
            "AR": ar,
            "Vol": vol,
            "R/R": rr,
            "MDD": mdd,
            "Wealth": compounded
        })
    return pd.DataFrame(records).set_index("Year")

def main():
    print("Running Baseline backtest...")
    baseline_returns = run_backtest_for_comb([])
    
    print("Running Optimal Combination (MTUM, VLUE, IUSG, USMV) backtest...")
    optimal_styles = ["MTUM", "VLUE", "IUSG", "USMV"]
    optimal_returns = run_backtest_for_comb(optimal_styles)
    
    print("\nComputing yearly metrics...")
    df_base = get_yearly_metrics(baseline_returns)
    df_opt = get_yearly_metrics(optimal_returns)
    
    # Merge and print comparison
    df_base = df_base.rename(columns=lambda x: f"Base_{x}")
    df_opt = df_opt.rename(columns=lambda x: f"Opt_{x}")
    df_compare = pd.concat([df_base, df_opt], axis=1)
    
    # Calculate AR and R/R differences
    df_compare["Diff_AR"] = df_compare["Opt_AR"] - df_compare["Base_AR"]
    df_compare["Diff_R/R"] = df_compare["Opt_R/R"] - df_compare["Base_R/R"]
    
    print("\n=== Yearly Performance Comparison ===")
    print("Baseline (11 US Sectors) vs Optimal (11 Sectors + MTUM, VLUE, IUSG, USMV)")
    print("======================================")
    
    # Print markdown table
    print("\n| Year | Base AR | Opt AR | AR Diff | Base R/R | Opt R/R | R/R Diff | Base MDD | Opt MDD |")
    print("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for year, row in df_compare.iterrows():
        base_ar = f"{row['Base_AR']*100:.2f}%"
        opt_ar = f"{row['Opt_AR']*100:.2f}%"
        diff_ar = f"{row['Diff_AR']*100:+.2f}%"
        base_rr = f"{row['Base_R/R']:.4f}"
        opt_rr = f"{row['Opt_R/R']:.4f}"
        diff_rr = f"{row['Diff_R/R']:.4f}"
        if row['Diff_R/R'] > 0:
            diff_rr = f"+{diff_rr}"
        base_mdd = f"{row['Base_MDD']*100:.2f}%"
        opt_mdd = f"{row['Opt_MDD']*100:.2f}%"
        print(f"| {year} | {base_ar} | {opt_ar} | {diff_ar} | {base_rr} | {opt_rr} | {diff_rr} | {base_mdd} | {opt_mdd} |")
        
    print("\nSummary statistics:")
    print(f"Total Years where Optimal AR > Baseline AR: {sum(df_compare['Diff_AR'] > 0)} / {len(df_compare)}")
    print(f"Total Years where Optimal R/R > Baseline R/R: {sum(df_compare['Diff_R/R'] > 0)} / {len(df_compare)}")

if __name__ == "__main__":
    main()
