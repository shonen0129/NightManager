"""
Comprehensive comparison of three signal modes:
- Mode A: baseline (close-to-close signal)
- Mode B: gap_residual (signal adjusted for gap open)
- Mode C: gap_tolerant (limit order execution with gap tolerance)
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backtest_config import create_timestamped_output_dir, STRATEGY_DEFAULTS
from data_loader import download_data, preprocess_data
from performance import calculate_metrics
from strategy import LeadLagStrategy


def run_mode_a_baseline():
    """Mode A: baseline signal execution"""
    params = {
        "K": 4,
        "lambda_reg": 0.75,
        "q": 5 / 17,
        "weight_mode": "signal",
        "dispersion_filter": True,
        "v3_mode": "static",
        "ewma_half_life": 45,
        "lambda_lw": 0.50,
        "lw_target": "equicorrelation",
        "corr_window": 60,
        "include_v4_prior": False,
        "signal_mode": "baseline",
    }
    return params


def run_mode_b_gap_residual():
    """Mode B: gap_residual signal execution"""
    params = run_mode_a_baseline()
    params["signal_mode"] = "gap_residual"
    params["gap_open_coef"] = 1.0
    return params


def run_mode_c_gap_tolerant(gamma):
    """Mode C: gap_tolerant limit order execution"""
    params = run_mode_a_baseline()
    params["signal_mode"] = "gap_tolerant"
    params["gamma"] = gamma
    return params


def calc_period_based_metrics(result_df, start_date, end_date):
    """Calculate metrics for a specific period"""
    period_df = result_df[
        (result_df.index >= start_date) & (result_df.index < end_date)
    ]
    if len(period_df) == 0:
        return None
    return calculate_metrics(period_df["daily_return"])


def analyze_execution_stats(result_df):
    """Analyze execution statistics for gap_tolerant mode"""
    if "long_executed" not in result_df.columns:
        return None

    # Days with trading
    trading_days = result_df[result_df["daily_return"] != 0].shape[0]
    # Days with execution
    execution_days = result_df[result_df["long_executed"] > 0].shape[0]

    execution_rate = execution_days / len(result_df) if len(result_df) > 0 else 0.0
    avg_long_exec = (
        result_df["long_executed"].mean()
        if "long_executed" in result_df.columns
        else 0.0
    )
    avg_short_exec = (
        result_df["short_executed"].mean()
        if "short_executed" in result_df.columns
        else 0.0
    )

    return {
        "execution_rate": execution_rate,
        "avg_long_exec": avg_long_exec,
        "avg_short_exec": avg_short_exec,
        "trading_days": trading_days,
    }


def main():
    print("=" * 80)
    print("THREE-MODE SIGNAL COMPARISON: Baseline vs Gap-Residual vs Gap-Tolerant")
    print("=" * 80)

    # Load data
    print("\n[1/5] Loading and preprocessing data...")
    data = download_data()
    df_exec = preprocess_data(data)
    print(f"Loaded {len(df_exec)} trading days")

    # Create output directory
    output_dir = create_timestamped_output_dir("gap_tolerant_mode_compare")

    start_date = "2015-01-01"
    all_results = {}
    all_metrics = {}

    # Mode A: Baseline
    print("\n[2/5] Running Mode A (baseline)...")
    params_a = run_mode_a_baseline()
    strategy_a = LeadLagStrategy(df_exec, **params_a)
    result_a = strategy_a.run_backtest(start_date=start_date)
    all_results["A_baseline"] = result_a
    all_metrics["A_baseline"] = calculate_metrics(result_a["daily_return"])

    # Mode B: Gap-Residual
    print("         Running Mode B (gap_residual)...")
    params_b = run_mode_b_gap_residual()
    strategy_b = LeadLagStrategy(df_exec, **params_b)
    result_b = strategy_b.run_backtest(start_date=start_date)
    all_results["B_gap_residual"] = result_b
    all_metrics["B_gap_residual"] = calculate_metrics(result_b["daily_return"])

    # Mode C: Gap-Tolerant with multiple gamma values
    print("         Running Mode C (gap_tolerant) with γ = {0.3, 0.5, 0.7, 1.0}...")
    gamma_values = [0.3, 0.5, 0.7, 1.0]
    mode_c_results = {}

    for gamma in gamma_values:
        params_c = run_mode_c_gap_tolerant(gamma)
        strategy_c = LeadLagStrategy(df_exec, **params_c)
        result_c = strategy_c.run_backtest(start_date=start_date)
        key = f"C_gap_tolerant_γ{gamma}"
        all_results[key] = result_c
        all_metrics[key] = calculate_metrics(result_c["daily_return"])
        mode_c_results[gamma] = result_c
        print(f"           γ={gamma} completed")

    # Print comprehensive metrics table
    print("\n" + "=" * 120)
    print("COMPREHENSIVE METRICS COMPARISON (全期間 2015-近況)")
    print("=" * 120)

    metrics_records = []
    for mode_key, metrics in all_metrics.items():
        record = {"Mode": mode_key}
        record.update({k: v for k, v in metrics.items()})
        metrics_records.append(record)

    metrics_df = pd.DataFrame(metrics_records)
    metrics_path = os.path.join(output_dir, "03_all_modes_metrics_overall.csv")
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    print(metrics_df.to_string(index=False))

    # Period-based analysis
    print("\n" + "=" * 120)
    print("PERIOD-BASED ANALYSIS")
    print("=" * 120)

    periods = [
        ("2015-01-01", "2020-01-01", "2015-2019"),
        ("2020-01-01", "2023-01-01", "2020-2022"),
        ("2023-01-01", None, "2023-recent"),
    ]

    period_metrics_records = []
    for start, end, period_label in periods:
        end_date = end or "2099-12-31"
        print(f"\n--- Period: {period_label} ---")
        for mode_key in all_results.keys():
            result = all_results[mode_key]
            period_metrics = calc_period_based_metrics(result, start, end_date)
            if period_metrics is not None:
                record = {"Mode": mode_key, "Period": period_label, **period_metrics}
                period_metrics_records.append(record)
                print(
                    f"{mode_key:30s} | AR={period_metrics['AR']*100:7.2f}% | Risk={period_metrics['RISK']*100:7.2f}% | R/R={period_metrics['R/R']:7.2f} | MDD={period_metrics['MDD']*100:7.2f}%"
                )

    period_df = pd.DataFrame(period_metrics_records)
    period_path = os.path.join(output_dir, "04_period_based_metrics.csv")
    period_df.to_csv(period_path, index=False, encoding="utf-8-sig")

    # Gap-tolerant execution statistics
    print("\n" + "=" * 120)
    print("MODE C (GAP-TOLERANT) EXECUTION STATISTICS BY γ")
    print("=" * 120)

    exec_stats_records = []
    for gamma, result_c in mode_c_results.items():
        exec_stats = analyze_execution_stats(result_c)
        if exec_stats:
            record = {
                "gamma": gamma,
                "execution_rate": exec_stats["execution_rate"],
                "avg_long_exec": exec_stats["avg_long_exec"],
                "avg_short_exec": exec_stats["avg_short_exec"],
                "trading_days": exec_stats["trading_days"],
            }
            exec_stats_records.append(record)
            print(
                f"γ={gamma}: execution_rate={exec_stats['execution_rate']*100:.1f}% | "
                f"avg_long={exec_stats['avg_long_exec']:.2f} | "
                f"avg_short={exec_stats['avg_short_exec']:.2f} | "
                f"trading_days={exec_stats['trading_days']}"
            )

    exec_df = pd.DataFrame(exec_stats_records)
    exec_path = os.path.join(output_dir, "05_gap_tolerant_execution_stats.csv")
    exec_df.to_csv(exec_path, index=False, encoding="utf-8-sig")

    # Save daily returns for all modes
    print("\n[3/5] Saving daily returns comparison...")
    daily_returns_df = pd.DataFrame()
    for mode_key, result in all_results.items():
        daily_returns_df[mode_key] = result["daily_return"]

    daily_path = os.path.join(output_dir, "02_all_modes_daily_return.csv")
    daily_returns_df.to_csv(daily_path, encoding="utf-8-sig")

    # Cumulative returns plot
    print("[4/5] Generating cumulative returns chart...")
    plt.figure(figsize=(14, 8))

    for mode_key, result in all_results.items():
        cumulative = (1 + result["daily_return"]).cumprod()
        plt.plot(cumulative.index, cumulative.values, label=mode_key, linewidth=2)

    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Cumulative Return (Base = 1.0)", fontsize=12)
    plt.title(
        "Strategy Cumulative Returns: Baseline vs Gap-Residual vs Gap-Tolerant Modes",
        fontsize=14,
        fontweight="bold",
    )
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    chart_path = os.path.join(output_dir, "01_cumulative_returns_comparison.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    print(f"Chart saved: {chart_path}")
    plt.close()

    # Save summary text
    print("[5/5] Generating summary report...")
    summary_path = os.path.join(output_dir, "00_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("THREE-MODE SIGNAL COMPARISON REPORT\n")
        f.write("=" * 100 + "\n\n")

        f.write("共通設定:\n")
        f.write("  - ユニバース: 米国11業種ETF（情報源）、日本17業種ETF（投資対象）\n")
        f.write("  - 期間: 2015-01-01〜直近\n")
        f.write(
            "  - PCA: K=4, λ_reg=0.75, λ_lw=0.50, lw_target=equicorrelation, EWMA h=45, L=60\n"
        )
        f.write("  - C_full推定期間: 2010/1/1〜2014/12/31\n")
        f.write(
            "  - 事前部分空間: v1(グローバル), v2(国スプレッド), v3(シクリカル/ディフェンシブ)\n"
        )
        f.write(
            "  - ウェイト: シグナル加重（中央値センタリング、ロング上位5/ショート下位5）\n"
        )
        f.write("  - Dispersion Filter: 有効（D_t指標、P10/P25閾値）\n")
        f.write("  - 価格取得: auto_adjust=False\n\n")

        f.write("【Mode A: baseline】\n")
        f.write("  - 寄付き成行で約定と仮定\n")
        f.write("  - 約定価格 = P^open_{j,t+1}\n")
        f.write("  - 決済 = 大引け P^close_{j,t+1}\n")
        f.write("  - 戦略リターン: r = P^close / P^open - 1\n\n")

        f.write("【Mode B: gap_residual（既存）】\n")
        f.write("  - シグナル補正: alpha_j = s_j - 1.0 × GapOpen_j\n")
        f.write("  - 補正後シグナルで銘柄選定・ウェイト算出\n")
        f.write("  - 約定: P^open_{j,t+1}（理想約定）\n\n")

        f.write("【Mode C: gap_tolerant（新規提案）】\n")
        f.write("  - 前日終値基準で指値を設定\n")
        f.write("    - ロング: 指値 = P^close_{j,t} × (1 + γ × s_j × σ_j)\n")
        f.write("    - ショート: 指値 = P^close_{j,t} × (1 - γ × |s_j| × σ_j)\n")
        f.write(
            "  - 約定判定と再正規化により、約定銘柄数が片側で2未満なら全取引見送り\n"
        )
        f.write("  - γ: {0.3, 0.5, 0.7, 1.0} の4水準で比較\n\n")

        f.write("=" * 100 + "\n")
        f.write("OVERALL METRICS (全期間)\n")
        f.write("=" * 100 + "\n")
        f.write(metrics_df.to_string(index=False))
        f.write("\n\n")

        f.write("=" * 100 + "\n")
        f.write("OUTPUT FILES\n")
        f.write("=" * 100 + "\n")
        f.write(f"01_cumulative_returns_comparison.png - 累積リターン折れ線グラフ\n")
        f.write(f"02_all_modes_daily_return.csv - 全モードの日次リターン\n")
        f.write(
            f"03_all_modes_metrics_overall.csv - 全モードのメトリクス一覧表（全期間）\n"
        )
        f.write(
            f"04_period_based_metrics.csv - 期間別メトリクス（2015-2019, 2020-2022, 2023-近況）\n"
        )
        f.write(
            f"05_gap_tolerant_execution_stats.csv - Mode Cのγ別約定率・約定銘柄数統計\n"
        )

    print(f"\nAll outputs saved to: {output_dir}")
    print("\n" + "=" * 80)
    print("分析完了!")


if __name__ == "__main__":
    main()
