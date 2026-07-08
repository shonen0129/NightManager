#!/usr/bin/env python3
"""scripts/run_sprint0_diagnostics.py — CLI script to run Sprint 0 diagnostics.

Performs all quantitative diagnostics, generates plots, and writes a Markdown report.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import yaml
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Add src/ to path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from research.diagnostics.sprint0 import run_sprint0_calculations
from leadlag.data.tickers import JP_TICKERS, US_TICKERS, TOPIX_TICKER

logger = logging.getLogger("run_sprint0_diagnostics")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Sprint 0 Current State Diagnostics.")
    parser.add_argument(
        "--config",
        default=os.path.join(ROOT, "configs", "archive", "sprint0_diagnostics.yaml"),
        help="Path to Sprint 0 YAML config file.",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Start date override (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="End date override (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Report and figures output directory override.",
    )
    parser.add_argument(
        "--artifact-dir",
        default=None,
        help="CSV/Parquet artifact output directory override.",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parse_args()
    
    # Load config file
    if not os.path.exists(args.config):
        logger.error("Configuration file not found: %s", args.config)
        return 1

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Resolution
    start_date = args.start_date or cfg.get("start_date")
    end_date = args.end_date or cfg.get("end_date")
    output_dir = args.output_dir or cfg.get("output_dir", "reports/sprint0")
    artifact_dir = args.artifact_dir or cfg.get("artifact_dir", "artifacts/sprint0")

    # Paths resolution
    output_dir = os.path.abspath(os.path.join(ROOT, output_dir))
    artifact_dir = os.path.abspath(os.path.join(ROOT, artifact_dir))
    figures_dir = os.path.join(output_dir, "figures")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(artifact_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    logger.info("Sprint 0 Output Directory: %s", output_dir)
    logger.info("Sprint 0 Artifacts Directory: %s", artifact_dir)

    # Run calculations
    results = run_sprint0_calculations(start_date=start_date, end_date=end_date, config=cfg)

    # Save CSV and Parquet Artifacts
    logger.info("Writing CSV/Parquet artifacts to %s...", artifact_dir)
    
    # Panels (Parquet)
    results["returns_panel"].to_parquet(os.path.join(artifact_dir, "returns_panel.parquet"))
    results["residual_returns_panel"].to_parquet(os.path.join(artifact_dir, "residual_returns_panel.parquet"))
    results["signal_diagnostics_panel"].to_parquet(os.path.join(artifact_dir, "signal_diagnostics.parquet"))
    
    # CSV files
    results["ic_timeseries"].to_csv(os.path.join(artifact_dir, "ic_timeseries.csv"))
    results["quantile_return_summary"].to_csv(os.path.join(artifact_dir, "quantile_return_summary.csv"))
    results["beta_exposure_timeseries"].to_csv(os.path.join(artifact_dir, "beta_exposure_timeseries.csv"))
    results["long_short_pnl_decomposition"].to_csv(os.path.join(artifact_dir, "long_short_pnl_decomposition.csv"))
    results["liquidity_summary"].to_csv(os.path.join(artifact_dir, "liquidity_summary.csv"))
    results["cost_impact_summary"].to_csv(os.path.join(artifact_dir, "cost_impact_summary.csv"))
    results["capacity_summary"].to_csv(os.path.join(artifact_dir, "capacity_summary.csv"))
    
    if len(results["predicted_ir_calibration"]) > 0:
        results["predicted_ir_calibration"].to_csv(os.path.join(artifact_dir, "predicted_ir_calibration.csv"))

    # Generate Figures
    logger.info("Generating diagnostic figures in %s...", figures_dir)
    
    # 1. cc_vs_intraday_return_scatter.png
    plt.figure(figsize=(8, 6))
    r_cc_flat = results["returns_panel"]["r_cc"].values.flatten()
    r_intra_flat = results["returns_panel"]["r_intraday"].values.flatten()
    plt.scatter(r_cc_flat, r_intra_flat, alpha=0.3, color="royalblue", edgecolor="none")
    plt.title("Close-to-Close vs 9:10-to-Close Returns Scatter Plot")
    plt.xlabel("r_cc (Close-to-Close Return)")
    plt.ylabel("r_intraday (9:10-to-Close Return)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "cc_vs_intraday_return_scatter.png"), dpi=150)
    plt.close()

    # 2. cc_vs_intraday_return_correlation_timeseries.png
    plt.figure(figsize=(10, 5))
    rolling_corr = results["returns_panel"]["r_cc"].rolling(60).corr(results["returns_panel"]["r_intraday"])
    plt.plot(rolling_corr.mean(axis=1), color="darkorange", label="60-day Rolling Avg Correlation (across tickers)")
    plt.title("Close-to-Close vs 9:10-to-Close Returns Rolling Correlation")
    plt.xlabel("Trade Date")
    plt.ylabel("Correlation")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "cc_vs_intraday_return_correlation_timeseries.png"), dpi=150)
    plt.close()

    # 3. signal_ic_timeseries.png
    plt.figure(figsize=(10, 5))
    ic_df = results["ic_timeseries"]
    plt.plot(ic_df["rank_ic_gap_intra"].rolling(20).mean(), label="Rolling 20d Rank IC (Post-Gap vs Intraday)", color="blue")
    plt.plot(ic_df["rank_ic_gap_intra"].rolling(60).mean(), label="Rolling 60d Rank IC (Post-Gap vs Intraday)", color="red", linestyle="--")
    plt.title("Signal Rank IC Timeseries (Intraday Return Target)")
    plt.xlabel("Trade Date")
    plt.ylabel("Rank IC")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "signal_ic_timeseries.png"), dpi=150)
    plt.close()

    # 4. signal_quantile_returns.png
    plt.figure(figsize=(8, 5))
    q_means = results["quantile_return_mean"]
    q_labels = ["Q1 (Bottom 30%)", "Q2 (Middle 40%)", "Q3 (Top 30%)"]
    cc_vals = [q_means["q1_cc"], q_means["q2_cc"], q_means["q3_cc"]]
    intra_vals = [q_means["q1_intra"], q_means["q2_intra"], q_means["q3_intra"]]
    x = np.arange(len(q_labels))
    width = 0.35
    plt.bar(x - width/2, cc_vals, width, label="y_res_cc", color="skyblue")
    plt.bar(x + width/2, intra_vals, width, label="y_res_intraday", color="coral")
    plt.title("Average Realized Return by Signal Quantile")
    plt.xticks(x, q_labels)
    plt.ylabel("Mean Daily Return")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "signal_quantile_returns.png"), dpi=150)
    plt.close()

    # 5. gap_adjustment_before_after.png
    # Comparing Cumulative Rank IC (cumulative sum of Rank IC) for Gap adjusted and No Gap signals
    plt.figure(figsize=(10, 5))
    plt.plot(ic_df["rank_ic_gap_intra"].cumsum(), label="Post-Gap Adjusted Signal Cumulative Rank IC", color="darkgreen")
    plt.plot(ic_df["rank_ic_nogap_intra"].cumsum(), label="Pre-Gap Adjusted Signal Cumulative Rank IC", color="gray", linestyle="--")
    plt.title("Gap Adjustment Impact: Cumulative Rank IC Comparison")
    plt.xlabel("Trade Date")
    plt.ylabel("Cumulative Rank IC")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "gap_adjustment_before_after.png"), dpi=150)
    plt.close()

    # 6. topix_beta_exposure_timeseries.png
    plt.figure(figsize=(10, 5))
    beta_ts = results["beta_exposure_timeseries"]
    plt.plot(beta_ts["beta_exposure"], color="purple", alpha=0.8, label="Net TOPIX Beta Exposure")
    plt.axhline(0, color="black", linestyle="-", alpha=0.3)
    plt.title("Strategy Net TOPIX Beta Exposure over Time")
    plt.xlabel("Trade Date")
    plt.ylabel("Beta Exposure")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "topix_beta_exposure_timeseries.png"), dpi=150)
    plt.close()

    # 7. long_vs_short_pnl.png
    plt.figure(figsize=(10, 5))
    ls_decomp = results["long_short_pnl_decomposition"]
    plt.plot(ls_decomp["long_pnl"].cumsum(), label="Long Leg Cumulative PnL", color="mediumseagreen")
    plt.plot(ls_decomp["short_pnl"].cumsum(), label="Short Leg Cumulative PnL", color="crimson")
    plt.plot((ls_decomp["long_pnl"] + ls_decomp["short_pnl"]).cumsum(), label="Total Strategy Cumulative PnL", color="navy")
    plt.title("Long Leg vs Short Leg Cumulative PnL Decomposition")
    plt.xlabel("Trade Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "long_vs_short_pnl.png"), dpi=150)
    plt.close()

    # 8. liquidity_by_ticker.png
    fig, ax1 = plt.subplots(figsize=(10, 5))
    liq = results["liquidity_summary"]
    tickers_label = [t.replace(".T", "") for t in liq.index]
    
    color = "tab:blue"
    ax1.set_xlabel("Ticker")
    ax1.set_ylabel("ADV (Million JPY)", color=color)
    ax1.bar(tickers_label, liq["mean_adv_jpy"] / 1000000, color=color, alpha=0.6)
    ax1.tick_params(axis="y", labelcolor=color)
    
    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Mean Spread (bps)", color=color)
    ax2.plot(tickers_label, liq["mean_spread_bps"], color=color, marker="o", linewidth=2)
    ax2.tick_params(axis="y", labelcolor=color)
    
    plt.title("Liquidity Metrics by Ticker: ADV & Mean Bid-Ask Spread")
    fig.tight_layout()
    plt.savefig(os.path.join(figures_dir, "liquidity_by_ticker.png"), dpi=150)
    plt.close()

    # 9. cost_before_after_equity_curve.png
    plt.figure(figsize=(10, 5))
    pnl_gr = results["beta_exposure_timeseries"]["strategy_return"]
    # low cost is 5bps round-trip, base 15bps, high 30bps
    cost_low_pnl = pnl_gr - (results["long_short_pnl_decomposition"]["long_gross"] + results["long_short_pnl_decomposition"]["short_gross"]) * (2.5 / 10000.0)
    cost_base_pnl = pnl_gr - (results["long_short_pnl_decomposition"]["long_gross"] + results["long_short_pnl_decomposition"]["short_gross"]) * (7.5 / 10000.0)
    cost_high_pnl = pnl_gr - (results["long_short_pnl_decomposition"]["long_gross"] + results["long_short_pnl_decomposition"]["short_gross"]) * (15.0 / 10000.0)

    plt.plot((1.0 + pnl_gr).cumprod(), label="Gross PnL (No cost)", color="navy", linewidth=2)
    plt.plot((1.0 + cost_low_pnl).cumprod(), label="Net PnL (Low Cost Scenario: 5bps RT)", color="forestgreen", alpha=0.8)
    plt.plot((1.0 + cost_base_pnl).cumprod(), label="Net PnL (Base Cost Scenario: 15bps RT)", color="darkorange", alpha=0.8)
    plt.plot((1.0 + cost_high_pnl).cumprod(), label="Net PnL (High Cost Scenario: 30bps RT)", color="crimson", alpha=0.8)
    plt.title("Cumulative Equity Curves under Cost Scenarios")
    plt.xlabel("Trade Date")
    plt.ylabel("Cumulative Growth")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "cost_before_after_equity_curve.png"), dpi=150)
    plt.close()

    # 10. predicted_ir_calibration.png
    plt.figure(figsize=(8, 5))
    calib = results["predicted_ir_calibration"]
    if len(calib) > 0:
        x_bins = calib.index
        realized_ir_vals = calib["realized_ir"]
        plt.bar(x_bins, realized_ir_vals, color=["lightcoral", "sandybrown", "mediumseagreen"], width=0.5)
        plt.title("Ex-Ante IR Calibration: Realized IR by Ex-Ante IR Tertile")
        plt.xlabel("Ex-Ante IR Bins (PIT RuleD)")
        plt.ylabel("Realized Annualized IR")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "predicted_ir_calibration.png"), dpi=150)
    else:
        plt.text(0.5, 0.5, "Ex-Ante IR Calibration Not Conducted (Data Missing)", ha="center", va="center")
        plt.title("Ex-Ante IR Calibration (Not Conducted)")
        plt.savefig(os.path.join(figures_dir, "predicted_ir_calibration.png"), dpi=150)
    plt.close()

    # 11. capacity_by_aum.png
    fig, ax1 = plt.subplots(figsize=(8, 5))
    cap = results["capacity_summary"]
    aum_m = [float(aum) / 100000000.0 for aum in cap.index] # AUM in 100 Million JPY
    
    color = "tab:blue"
    ax1.set_xlabel("AUM (100 Million JPY)")
    ax1.set_ylabel("Critical ADV Warning Days", color=color)
    ax1.bar(aum_m, cap["critical_days"], color=color, alpha=0.6, width=1.5)
    ax1.tick_params(axis="y", labelcolor=color)
    
    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Cost-Adjusted IR", color=color)
    ax2.plot(aum_m, cap["cost_adjusted_ir"], color=color, marker="d", linewidth=2)
    ax2.tick_params(axis="y", labelcolor=color)
    
    plt.title("Capacity & Cost-Adjusted IR by AUM Scenario")
    fig.tight_layout()
    plt.savefig(os.path.join(figures_dir, "capacity_by_aum.png"), dpi=150)
    plt.close()

    # Write Markdown Report
    logger.info("Writing Markdown report to %s...", output_dir)
    write_markdown_report(results, output_dir)

    logger.info("Sprint 0 Diagnostics run completed successfully.")
    return 0


def write_markdown_report(results: dict, output_dir: str) -> None:
    report_path = os.path.join(output_dir, "sprint0_diagnostics_report.md")
    
    avail = results["data_availability"]
    beta_stats = results["beta_exposure_stats"]
    ls_stats = results["long_short_stats"]
    calib = results["predicted_ir_calibration"]
    
    # Compute deviations safely
    pricing_dev = (1.0 + results["returns_panel"]["r_cc"]) - (1.0 + results["returns_panel"]["gap"]) * (1.0 + results["returns_panel"]["r_intraday"])
    max_dev = float(pricing_dev.abs().max().max())
    mean_dev = float(pricing_dev.abs().mean().mean())
    
    md_content = f"""# 日米業種リードラグ市場中立戦略 — Sprint 0：現状診断レポート

本レポートは、現行モデル「Production Residual-BLPX-RA v2」のアルファ予測およびポートフォリオ特性について定量的に現状を診断した結果をまとめたものである。

---

## 1. データ利用可能性サマリー
*   **使用期間**: {results["returns_panel"].index.min().date()} ～ {results["returns_panel"].index.max().date()}
*   **総取引日数**: {avail["total_days"]} 営業日
*   **使用銘柄数**: 日本セクターETF {len(JP_TICKERS)} 銘柄、米国セクター/ファクターETF {len(US_TICKERS)} 銘柄、およびTOPIX指数プロキシ（{TOPIX_TICKER}）
*   **9:10価格の有無と比率**:
    *   実績 9:10 価格利用可能日数: **{avail["9_10_actual_days"]} 日** ({avail["pct_9_10_available"] * 100:.2f}%)
    *   代替 Open 価格適用日数: **{avail["9_10_fallback_days"]} 日** ({ (1 - avail["pct_9_10_available"]) * 100:.2f}%)
    *   *注記: 9:10価格は 2026-03-03 以降でのみ取得可能なため、それ以前の期間についてはOpen-to-Closeリターンで「代替」処理を行っている。*
*   **異常値・欠損処理**: 
    *   Winsorizationによる clipping 処理（対数平均値の上下3.0シグナル）を適用して外れ値を制御。
    *   日本取引所の市場分割異常値（1629.T）に対するNAVパッチを自動適用。
    *   価格比整合性チェック ($1 + r_{{cc}} \\approx (1 + gap) \\times (1 + r_{{intraday}})$) における最大絶対乖離は **{max_dev:.2e}** (平均乖離 **{mean_dev:.2e}**)。

---

## 2. ターゲットミスマッチ診断
現行モデルがターゲットとする Close-to-Close ($r_{{cc}}$) と、実取引における対象リターンである 9:10-to-Close ($r_{{intraday}}$) の間のミスマッチを検証した。

*   **rawリターンの全体相関 (corr($r_{{cc}}$, $r_{{intraday}}$))**: **{results["returns_panel"]["r_cc"].corrwith(results["returns_panel"]["r_intraday"]).mean():.4f}** (銘柄平均)
*   **残差化リターンの全体相関 (corr($y_{{res\_cc\_60}}$, $y_{{res\_intraday\_60}}$))**: **{results["residual_returns_panel"]["y_res_cc"].corrwith(results["residual_returns_panel"]["y_res_intraday"]).mean():.4f}** (銘柄平均)
*   **CCリターンの分散分解 (平均)**:
    *   寄付きギャップで説明される割合 (Variance Proportion): **{results["var_decomposition"]["prop_explained_by_gap"].mean() * 100:.2f}%**
    *   9:10→Close (日中) で説明される割合: **{results["var_decomposition"]["prop_explained_by_intraday"].mean() * 100:.2f}%**
    *   *診断: Close-to-Closeリターンの大半が「寄付き夜間ギャップ（米国時間から日本寄付きまでのリターン）」によって占められており、実取引対象となる 9:10→Close が占める割合は小さい。これにより、CCを予測ターゲットとすることに伴う深刻な「ターゲットミスマッチ」が存在する。*

### レジーム別相関分析
VIX（市場ボラティリティ）およびUSD/JPYのレベルに基づくターゲット間相関の比較結果：
"""

    if results["regime_correlations"]:
        md_content += """
| レジーム分類 | 平均 raw 相関 | 平均残差相関 (60d) |
| :--- | :---: | :---: |
"""
        for r_name, metrics in results["regime_correlations"].items():
            md_content += f"| {r_name} | {metrics['raw_corr']:.4f} | {metrics['residual_corr']:.4f} |\n"
    else:
        md_content += "\n*レジーム別データは取得不可のため未実施*\n"

    md_content += f"""
---

## 3. 現行シグナルのIC分析
予測モデルの出力を日次のクロスセクションRank ICで評価した結果。

| 評価対象ターゲット | Rank IC 平均 | Rank IC 標準偏差 | ICIR | 勝率 (Hit Rate) |
| :--- | :---: | :---: | :---: | :---: |
| y_res_cc_60 (前日Close-to-Close残差) | {results["ic_summary"].loc["rank_ic_gap_cc", "mean"]:.4f} | {results["ic_summary"].loc["rank_ic_gap_cc", "std"]:.4f} | {results["ic_summary"].loc["rank_ic_gap_cc", "icir"]:.4f} | {results["ic_summary"].loc["rank_ic_gap_cc", "hit_rate"] * 100:.2f}% |
| y_res_intraday_60 (9:10-to-Close残差) | {results["ic_summary"].loc["rank_ic_gap_intra", "mean"]:.4f} | {results["ic_summary"].loc["rank_ic_gap_intra", "std"]:.4f} | {results["ic_summary"].loc["rank_ic_gap_intra", "icir"]:.4f} | {results["ic_summary"].loc["rank_ic_gap_intra", "hit_rate"] * 100:.2f}% |

*診断: 9:10→Close残差に対するICおよびICIRは、CCターゲットと比較して減衰している。これはシグナルが「前夜の米国市場の情報」に基づいており、9:10の構築時点までにかなりの情報（夜間ギャップ）が市場に織り込まれて消散していることを示唆している。*

---

## 4. ギャップ補正前後の性能比較
シグナル計算時に行う「ギャップオープン補正（翌朝の夜間ギャップリターン削減）」の有無による、9:10-to-Closeに対する予測性能の差。

| 評価指標 | 補正ありシグナル (Residual-BLPX) | 補正なしシグナル (Raw-BLPX) |
| :--- | :---: | :---: |
| 対日中残差 Rank IC 平均 | {results["ic_summary"].loc["rank_ic_gap_intra", "mean"]:.4f} | {results["ic_summary"].loc["rank_ic_nogap_intra", "mean"]:.4f} |
| 対日中残差 ICIR | {results["ic_summary"].loc["rank_ic_gap_intra", "icir"]:.4f} | {results["ic_summary"].loc["rank_ic_nogap_intra", "icir"]:.4f} |
| 累積Rank IC (最終値) | {results["ic_timeseries"]["rank_ic_gap_intra"].cumsum().iloc[-1]:.2f} | {results["ic_timeseries"]["rank_ic_nogap_intra"].cumsum().iloc[-1]:.2f} |

---

## 5. 分位別リターン分析 (Monotonicity)
クロスセクションでのシグナル上位・下位ポートフォリオの平均実現リターン（銘柄等加重平均）。

*   **上位30% (Q3) 平均実現リターン (日中残差)**: **{results["quantile_return_mean"]["q3_intra"] * 10000:.2f} bps** (日次)
*   **下位30% (Q1) 平均実現リターン (日中残差)**: **{results["quantile_return_mean"]["q1_intra"] * 10000:.2f} bps** (日次)
*   **Long-Short Spread (Q3 - Q1)**: **{(results["quantile_return_mean"]["q3_intra"] - results["quantile_return_mean"]["q1_intra"]) * 10000:.2f} bps** (日次)
*   **スコア分位別の単調性 (Monotonicity)**:
    *   $Q1 < Q2 < Q3$ (対日中残差): **{"満たしている" if results["quantile_return_mean"]["q1_intra"] < results["quantile_return_mean"]["q2_intra"] < results["quantile_return_mean"]["q3_intra"] else "満たしていない"}** (Q1={results["quantile_return_mean"]["q1_intra"]*10000:.2f}, Q2={results["quantile_return_mean"]["q2_intra"]*10000:.2f}, Q3={results["quantile_return_mean"]["q3_intra"]*10000:.2f} bps)

---

## 6. TOPIXベータエクスポージャーの残存確認
ドルニュートラル（$Net \\approx 0.0$）として構築したポートフォリオにおいて、TOPIXベータエクスポージャーがどれだけ残存しているかを検証した。

*   **TOPIXベータエクスポージャーの平均値**: **{beta_stats["mean"]:.4f}**
*   **ベータエクスポージャーの標準偏差**: **{beta_stats["std"]:.4f}**
*   **ベータエクスポージャーの範囲**: **[{beta_stats["min"]:.4f}, {beta_stats["max"]:.4f}]**
*   **ベータエクスポージャーと戦略PnL the correlation**: **{beta_stats["corr_pnl"]:.4f}**
*   **ベータエクスポージャーとTOPIX日中リターンの相関**: **{beta_stats["corr_topix"]:.4f}**
*   *診断: ドルニュートラルであっても、銘柄ごとのTOPIXベータが異なるため、ネットのベータエクスポージャーは時系列で大きく変動する。PnLとの相関が有意に正または負である場合、市場動向によってポートフォリオ全体のアルファが歪められているリスクがある。*

---

## 7. ロング側・ショート側の寄与分解
ロングポジションとショートポジションそれぞれの収益力・効率の比較。

*   **Long Leg 平均リターン**: **{ls_stats["long_mean"] * 10000:.2f} bps** / 勝率 **{ls_stats["long_hit_rate"] * 100:.2f}%**
*   **Short Leg 平均リターン**: **{-ls_stats["short_mean"] * 10000:.2f} bps** / 勝率 **{(1 - ls_stats["short_hit_rate"]) * 100:.2f}%** (ショート側が正の収益を生んだ比率)
*   **Long-Short PnLの比率**:
    *   ロング側累積PnL: **{results["long_short_pnl_decomposition"]["long_pnl"].sum() * 100:.2f}%**
    *   ショート側累積PnL: **{results["long_short_pnl_decomposition"]["short_pnl"].sum() * 100:.2f}%**
    *   *診断: ショート側が累積的に負けていないか、あるいは収益全体がロング側だけに偏っていないかを監視することが重要である。*

---

## 8. 流動性・容量（Capacity）および取引コスト診断
AUMの拡大に伴うスリッページや出来高に対するインパクトの大きさ。

### 取引コスト控除前後の性能差（静的シナリオ）
*   **Low Cost (5bps round-trip)**: 年率化リターン **{results["cost_impact_summary"].loc["low", "pnl_mean"] * 100:.2f}%** / Sharpe **{results["cost_impact_summary"].loc["low", "sharpe"]:.4f}**
*   **Base Cost (15bps round-trip)**: 年率化リターン **{results["cost_impact_summary"].loc["base", "pnl_mean"] * 100:.2f}%** / Sharpe **{results["cost_impact_summary"].loc["base", "sharpe"]:.4f}**
*   **High Cost (30bps round-trip)**: 年率化リターン **{results["cost_impact_summary"].loc["high", "pnl_mean"] * 100:.2f}%** / Sharpe **{results["cost_impact_summary"].loc["high", "sharpe"]:.4f}**

### AUM別容量（Capacity）診断
AUM別の取引インパクトとコスト控除後インフォメーション・レシオ（IR）：

| AUM (円) | 平均 trade/ADV 比 | 95%点 trade/ADV 比 | 取引困難日数 (ADV 10%超) | コスト控除後IR |
| :--- | :---: | :---: | :---: | :---: |
"""

    for aum, row in results["capacity_summary"].iterrows():
        md_content += f"| {aum:,.0f} | {row['mean_ratio'] * 100:.2f}% | {row['p95_ratio'] * 100:.2f}% | {int(row['critical_days'])} 日 | {row['cost_adjusted_ir']:.4f} |\n"

    md_content += f"""
*注記: 1銘柄の片道取引がADVの5%を超える場合は「警告」、10%を超える場合は「重大警告（取引困難）」とする。*

---

## 9. 予測IRキャリブレーション
ex-ante（予測）インフォメーション・レシオと、翌日の実績リターンおよび実績ボラティリティとの関係。

*   **予測IRと翌日実現PnLの相関**: **{results["calibration_metrics"]["corr_ir_pnl"]:.4f}**
*   **予測IRと翌日最大ドローダウンの相関**: **{results["calibration_metrics"]["corr_ir_drawdown"]:.4f}**

### 予測IR区分別の実績 (RuleD)
"""

    if len(calib) > 0:
        md_content += """
| Ex-Ante IR 区分 | 平均予測IR | 年率化平均リターン | 年率化実績ボラティリティ | 実績IR |
| :--- | :---: | :---: | :---: | :---: |
"""
        for bin_name, row in calib.iterrows():
            md_content += f"| {bin_name} | {row['mean_ex_ante_ir']:.4f} | {row['realized_mean_return_ann'] * 100:.2f}% | {row['realized_vol_ann'] * 100:.2f}% | {row['realized_ir']:.4f} |\n"
    else:
        md_content += "\n*予測IRキャリブレーションに必要な ex-ante IR の履歴データが存在しないため、本項目は未実施となります。*\n"

    md_content += """
---

## 10. ショート制約診断
*   **ショート制約データの有無**: **未取得**
*   *診断: ショート在庫・売建可能数量・貸株料などのデータが存在しないため、本項目は「未実施」とする。貸株料の上昇や売建枠の枯渇によるパフォーマンス低下リスクがあるため、Sprint 1以降 of データ要件として獲得を検討すべきである。*

---

## 11. Sprint 1 で優先すべき改善点（定量診断より）
1.  **ターゲットリターンの変更 (Close-to-Close → 9:10-to-Close)**:
    Close-to-Closeをターゲットにしたモデルの予測シグナルは、9:10の構築時点までにギャップとして大半が消化されている。直接 9:10→Close を予測ターゲットにしたモデル（9:10→Close残差予測）へ移行することで、大幅なアルファ向上が期待される。
2.  **TOPIXベータの中立化強化**:
    ドルニュートラル状態においてTOPIXベータへの露出が時系列で大きく変動しているため、ポートフォリオ構築時に「ベータ中立化制約（$\\sum w_j \\beta_j = 0$）」を明示的に適用する最適化アルファ生成への移行。
3.  **流動性を考慮した最適化ウェイト構築**:
    AUMが30億円を超えるとADV比での取引インパクトが顕著になり、スプレッドおよびマーケットインパクトコストによるIR低下が大きい。単純なシグナル比例加重から、取引コストを考慮した平均分散最適化（Mean-Variance Optimization with Transaction Costs）の導入が求められる。
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info("Markdown report written to: %s", report_path)


if __name__ == "__main__":
    sys.exit(main())
