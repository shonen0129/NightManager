import sys
import itertools
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure src on path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

# Import system modules
import data.ticker_registry as registry
import data.preprocessor as preprocessor
import data.downloader as downloader
import data_loader
import config as sys_config
import strategy as sys_strategy
import backtest.runner as runner
from runner.config import ProductionConfig
from domain.signals import lead_lag as signals
from performance import calculate_metrics

# Define the approved sensitivities for the style ETFs
NEW_SENSITIVITIES = {
    "MTUM": {"w3": 0.0, "w4": 0.3, "w5": 0.0, "w6": -0.1},
    "VLUE": {"w3": 0.6, "w4": 0.1, "w5": 0.4, "w6": 0.5},
    "IUSG": {"w3": -0.2, "w4": 0.6, "w5": -0.1, "w6": -0.4},
    "IJR":  {"w3": 0.4, "w4": -0.2, "w5": 0.1, "w6": 0.1},
    "USMV": {"w3": -0.7, "w4": -0.3, "w5": -0.3, "w6": -0.3}
}

STYLE_ORDER = ["MTUM", "VLUE", "IUSG", "IJR", "USMV"]
BASE_US_TICKERS = [
    "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"
]

def make_custom_build_v3_static(active_styles):
    active_sens = [NEW_SENSITIVITIES[s] for s in active_styles]
    
    def custom_build_v3_static(n_u, n_j, include_v4=True, w6_override=None):
        base_vectors = signals.build_base_vectors(n_u, n_j)
        v1, v2 = base_vectors["v1"], base_vectors["v2"]

        # w3 Base
        us_w3 = [1.0, 0.3, 0.2, 0.8, 0.9, 0.7, -1.0, 0.4, -0.9, -0.8, 1.0]
        jp_w3 = [-0.9, 0.3, 0.6, 0.9, -0.9, 1.0, 1.0, 0.9, 0.8, -0.3, -1.0, -0.4, 0.7, -0.5, 0.8, 0.6, 0.5]
        for sens in active_sens:
            us_w3.append(sens['w3'])
        w3 = np.array(us_w3 + jp_w3, dtype=float)
        v3 = signals._orthogonalize_and_normalize(w3, [v1, v2])

        if not include_v4:
            return np.column_stack([v1, v2, v3])

        # w4 Base
        us_w4 = [0.4, 0.0, 0.1, 0.2, 0.7, 0.8, -0.5, -0.4, -0.7, -0.4, 0.6]
        jp_w4 = [-0.6, 0.2, 0.2, 0.5, -0.2, 1.0, 0.6, 0.8, 1.0, -0.2, -0.8, -0.4, 0.8, -0.7, 0.3, 0.0, -0.9]
        for sens in active_sens:
            us_w4.append(sens['w4'])
        w4 = np.array(us_w4 + jp_w4, dtype=float)

        # w5 Base
        us_w5 = [0.4, 0.0, 1.0, 0.0, 0.2, 0.0, -0.3, 0.0, -0.8, 0.0, -0.3]
        jp_w5 = [-0.3, 1.0, -0.1, 0.3, 0.0, -0.2, 0.2, 0.0, 0.0, 0.0, -0.9, -0.1, 0.7, -0.2, 0.0, 0.0, 0.0]
        for sens in active_sens:
            us_w5.append(sens['w5'])
        w5 = np.array(us_w5 + jp_w5, dtype=float)

        # w6 Base
        us_w6 = [0.8, -0.3, 1.0, 0.3, 0.3, -0.5, -0.2, 0.4, -0.7, -0.2, -0.4]
        jp_w6 = [-0.4, 1.0, 0.3, 0.7, -0.2, -0.1, 0.6, 0.2, -0.3, -0.3, -0.8, -0.3, 0.8, -0.5, 0.2, 0.1, 0.3]
        for sens in active_sens:
            us_w6.append(sens['w6'])
        w6 = np.array(us_w6 + jp_w6, dtype=float)

        if w6_override is not None:
            w6_arr = np.asarray(w6_override, dtype=float).reshape(-1)
            w6 = w6_arr

        v4 = signals._orthogonalize_and_normalize(w4, [v1, v2, v3])
        v5 = signals._orthogonalize_and_normalize(w5, [v1, v2, v3, v4])
        v6 = signals._orthogonalize_and_normalize(w6, [v1, v2, v3, v4, v5])

        return np.column_stack([v1, v2, v3, v4, v5, v6])
        
    return custom_build_v3_static

def run_backtest_for_combination(active_styles):
    active_us_tickers = BASE_US_TICKERS + active_styles
    n_us = len(active_us_tickers)

    # Monkeypatch sizes and lists globally on registry and all targets
    registry.US_TICKERS = active_us_tickers
    registry.N_US = n_us
    registry.N_TOTAL = n_us + registry.N_JP
    registry.N_US_ASSETS = n_us
    registry.N_TOTAL_ASSETS = registry.N_TOTAL

    preprocessor.US_TICKERS = active_us_tickers
    downloader.US_TICKERS = active_us_tickers

    sys_config.N_US_ASSETS = n_us
    sys_config.N_TOTAL_ASSETS = registry.N_TOTAL

    sys_strategy.N_US_ASSETS = n_us
    sys_strategy.N_TOTAL_ASSETS = registry.N_TOTAL

    runner.N_US_ASSETS = n_us

    # Monkeypatch the build_v3_static function
    signals.build_v3_static = make_custom_build_v3_static(active_styles)

    # Force load raw cache and slice columns accordingly to prevent download
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
    metrics = calculate_metrics(results["daily_return"])
    
    # Store returns for cumulative plotting
    cum_returns = (1.0 + results["daily_return"]).cumprod()
    
    return metrics, cum_returns

def main():
    # Generate all subsets
    combinations_to_test = []
    for r in range(6):
        for comb in itertools.combinations(STYLE_ORDER, r):
            combinations_to_test.append(list(comb))

    print(f"Total combinations to test: {len(combinations_to_test)}")

    all_results = []
    cum_curves = {}

    for idx, comb in enumerate(combinations_to_test):
        comb_str = ", ".join(comb) if comb else "None (Baseline)"
        print(f"Testing combination {idx+1}/{len(combinations_to_test)}: [{comb_str}]")
        try:
            metrics, cum_returns = run_backtest_for_combination(comb)
            all_results.append({
                "combination": comb,
                "combination_str": comb_str,
                "AR": metrics["AR"],
                "RISK": metrics["RISK"],
                "R/R": metrics["R/R"],
                "MDD": metrics["MDD"],
                "FinalWealth": metrics["Total Return"] + 1.0,
                "cum_returns": cum_returns
            })
            # Save all curves, key by string
            cum_curves[comb_str] = cum_returns
        except Exception as e:
            print(f"Failed to test combination {comb_str}: {e}")

    # Convert to DataFrame for easier sorting
    df_results = pd.DataFrame(all_results)
    df_results_sorted = df_results.sort_values(by="R/R", ascending=False).reset_index(drop=True)

    # Output directory
    output_dir = ROOT / "results" / "us_noise_filter"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build report
    report_lines = [
        "# Optimal Style ETF Combination Search Report",
        f"\nTested all {len(combinations_to_test)} subsets of MTUM, VLUE, IUSG, IJR, USMV using approved macro sensitivities.",
        "\n### Performance Rankings (Sorted by Risk/Return Ratio)",
        "\n| Rank | Combination | Annualized Return (AR) | Annualized Risk (Vol) | R/R Ratio | Max Drawdown (MDD) | Final Wealth |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: |"
    ]

    for rank, row in df_results_sorted.iterrows():
        ar_pct = f"{row['AR']*100:.2f}%"
        risk_pct = f"{row['RISK']*100:.2f}%"
        rr_val = f"{row['R/R']:.4f}"
        mdd_pct = f"{row['MDD']*100:.2f}%"
        final_w = f"{row['FinalWealth']:.4f}x"
        
        # Highlight top 3 and baseline/all 5
        comb_name = row['combination_str']
        if rank == 0:
            comb_name = f"**{comb_name} (BEST)**"
        
        report_lines.append(f"| {rank+1} | {comb_name} | {ar_pct} | {risk_pct} | {rr_val} | {mdd_pct} | {final_w} |")

    report_text = "\n".join(report_lines)
    print("\n\n" + report_text)

    # Write report
    with open(output_dir / "optimal_style_report.md", "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\nSaved optimal combination report to {output_dir / 'optimal_style_report.md'}")

    # Plot baseline, all 5, and top 3 best combinations
    plt.figure(figsize=(12, 7))
    
    # Baseline
    plt.plot(cum_curves["None (Baseline)"].index, cum_curves["None (Baseline)"].values, label="Baseline (No Styles)", linewidth=1.5, color="black", linestyle="--")
    
    # All 5
    all_5_str = ", ".join(STYLE_ORDER)
    plt.plot(cum_curves[all_5_str].index, cum_curves[all_5_str].values, label="All 5 Styles", linewidth=1.5, color="gray")

    # Top 3 Best (excluding if they are baseline or all 5, to avoid double plotting)
    plotted_count = 0
    for idx, row in df_results_sorted.iterrows():
        comb_str = row['combination_str']
        if comb_str in ["None (Baseline)", all_5_str]:
            continue
        plt.plot(row['cum_returns'].index, row['cum_returns'].values, label=f"Rank {idx+1}: {comb_str}", linewidth=2.0)
        plotted_count += 1
        if plotted_count >= 3:
            break

    plt.title("Cumulative Return Comparison: Optimal Style ETF Combinations")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Wealth (x)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plot_path = output_dir / "optimal_style_curves.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Saved plot to {plot_path}")

if __name__ == "__main__":
    main()
