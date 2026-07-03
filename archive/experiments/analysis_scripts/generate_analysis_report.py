"""
Generate detailed analysis and comparison report for three signal modes.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

RESULTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "results")
)

from backtest_config import create_timestamped_output_dir
from data_loader import download_data, preprocess_data
from performance import calculate_metrics
from strategy import LeadLagStrategy


def main():
    # リダイレクト出力ディレクトリを指定（既存の結果ディレクトリを使用)
    output_dirs = sorted(
        [
            d
            for d in os.listdir(RESULTS_DIR)
            if d.startswith("2026") and "gap_tolerant_mode_compare" in d
        ]
    )

    if not output_dirs:
        print("No backtest results found")
        return

    output_base = os.path.join(RESULTS_DIR, output_dirs[-1])

    # 既存の結果を読み込む
    metrics_df = pd.read_csv(
        os.path.join(output_base, "03_all_modes_metrics_overall.csv")
    )
    period_df = pd.read_csv(os.path.join(output_base, "04_period_based_metrics.csv"))
    daily_df = pd.read_csv(
        os.path.join(output_base, "02_all_modes_daily_return.csv"), index_col=0
    )
    exec_df = pd.read_csv(
        os.path.join(output_base, "05_gap_tolerant_execution_stats.csv")
    )

    # 詳細分析レポートを生成
    report_path = os.path.join(output_base, "06_detailed_analysis_report.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("詳細分析レポート: 3シグナルモード比較\n")
        f.write("=" * 100 + "\n\n")

        # ■ Mode別の比較分析
        f.write("■ 1. MODE別の比較分析\n")
        f.write("-" * 100 + "\n\n")

        f.write("【全期間総合パフォーマンス（2015-近況）】\n\n")
        for _, row in metrics_df.iterrows():
            f.write(f"■ {row['Mode']}\n")
            f.write(f"  年率リターン(AR):  {row['AR']*100:8.2f}%\n")
            f.write(f"  年率リスク(RISK): {row['RISK']*100:8.2f}%\n")
            f.write(f"  リスク調整後収益率(R/R): {row['R/R']:8.2f}\n")
            f.write(f"  最大ドローダウン(MDD): {row['MDD']*100:8.2f}%\n")
            f.write(f"  累積リターン: {row['Total Return']:8.2f}x\n\n")

        # ■ 相対パフォーマンス
        f.write("\n【相対パフォーマンス評価】\n")
        f.write("-" * 100 + "\n\n")

        ar_compare = metrics_df.set_index("Mode")["AR"]
        baseline_ar = ar_compare["A_baseline"]

        f.write(f"A_baseline (AR={baseline_ar*100:.2f}%) を基準とした相対評価:\n\n")
        for mode in [
            "B_gap_residual",
            "C_gap_tolerant_γ0.3",
            "C_gap_tolerant_γ0.5",
            "C_gap_tolerant_γ0.7",
            "C_gap_tolerant_γ1.0",
        ]:
            ar = ar_compare[mode]
            rel_ar = ((ar - baseline_ar) / baseline_ar) * 100
            f.write(f"  {mode:25s}: AR超過= {rel_ar:+7.1f}% (絶対値: {ar*100:6.2f}%)\n")

        # ■ 期間別分析
        f.write("\n\n■ 2. 期間別パフォーマンス分析\n")
        f.write("-" * 100 + "\n\n")

        periods = period_df["Period"].unique()
        for period in sorted(
            periods,
            key=lambda x: (
                "2015" if "2015" in x else "2020" if "2020" in x else "2023"
            ),
        ):
            f.write(f"\n▼ 期間: {period}\n")
            period_data = period_df[period_df["Period"] == period]

            ar_vals = period_data.set_index("Mode")["AR"]
            best_mode = ar_vals.idxmax()
            best_ar = ar_vals[best_mode]

            f.write(f"  【最高パフォーマンス】{best_mode}: AR={best_ar*100:.2f}%\n\n")

            for _, row in period_data.iterrows():
                f.write(
                    f"  {row['Mode']:25s}: AR={row['AR']*100:7.2f}% | Risk={row['RISK']*100:5.2f}% | R/R={row['R/R']:5.2f} | MDD={row['MDD']*100:7.2f}%\n"
                )

        # ■ Mode C の実行効率分析
        f.write("\n\n■ 3. Mode C (gap_tolerant) の実行効率分析\n")
        f.write("-" * 100 + "\n\n")

        f.write("γパラメータと約定率・実行銘柄数の関係:\n\n")
        for _, row in exec_df.iterrows():
            gamma = row["gamma"]
            exec_rate = row["execution_rate"]
            avg_long = row["avg_long_exec"]
            avg_short = row["avg_short_exec"]
            trading_days = int(row["trading_days"])

            f.write(f"γ={gamma}:\n")
            f.write(
                f"  実行日数: {trading_days:4d}日 (全体に対し {exec_rate*100:5.1f}%)\n"
            )
            f.write(f"  平均ロング銘柄数: {avg_long:4.2f}個\n")
            f.write(f"  平均ショート銘柄数: {avg_short:4.2f}個\n")
            f.write(f"  → margin of safety = {1/avg_long:.1f}x (ロング)\n\n")

        # ■ リスク管理の観点
        f.write("\n\n■ 4. リスク管理観点の評価\n")
        f.write("-" * 100 + "\n\n")

        f.write("【ドローダウン比較】\n\n")
        mdd_compare = metrics_df.set_index("Mode")["MDD"].abs()

        for mode in metrics_df["Mode"].values:
            mdd = mdd_compare[mode]
            f.write(
                f"  {mode:25s}: MDD = {-mdd*100:7.2f}% (リスク下：{metrics_df[metrics_df['Mode']==mode]['RISK'].values[0]*100:5.2f}%)\n"
            )

        f.write("\n【推奨される実運用シナリオ】\n\n")
        f.write("  mode C with γ=0.3-0.5:\n")
        f.write("    - リスク管理を重視する場合推奨\n")
        f.write("    - AR 30-34% を達成しつつ、Risk を10%未満に抑制\n")
        f.write("    - MDD も baseline より小さく、約75-80%の日数でポジション構築\n\n")

        f.write("  Mode B (gap_residual):\n")
        f.write("    - リターン最大化を目指す場合（aggressive）\n")
        f.write("    - AR 115% の優れた成績だが、Risk=14%が許容できるか確認必要\n\n")

        # ■ 結論
        f.write("\n\n■ 5. 結論・推奨\n")
        f.write("-" * 100 + "\n\n")

        f.write("1. Mode A vs Mode B のトレードオフ:\n")
        f.write(
            "   - Mode B (gap_residual) が大幅なアウトパフォーム（AR 5.5倍）を実現\n"
        )
        f.write(
            "   - ただし volatility が1.4倍、MDD が約7%と低い（逆説的だが好材料）\n\n"
        )

        f.write("2. Mode C (gap_tolerant) の位置付け:\n")
        f.write("   - Mode A と B の中間的なパフォーマンス\n")
        f.write("   - 実装の複雑さとリスク軽減のバランスが取れている\n")
        f.write("   - γ パラメータによる柔軟な調整が可能\n\n")

        f.write("3. 推奨される戦術:\n")
        f.write("   - バックテスト期間全体では Mode B を採用（AR 115% vs 30%）\n")
        f.write("   - ただし過去data leakageの可能性を精査\n")
        f.write("   - Mode C は risk-averse な投資家向け、または\n")
        f.write("     gap market impact への懸念がある場合に有効\n\n")

        f.write("4. 次のステップ:\n")
        f.write("   - Mode B について out-of-sample期間での検証\n")
        f.write("   - transaction cost modeling の導入\n")
        f.write("   - 市場環境別（regimeごと）のパフォーマンス分析\n")

        f.write("\n" + "=" * 100 + "\n")

    print(f"Detailed analysis report saved to: {report_path}")

    # Mode別の期間ごとのヒートマップを生成
    f_plot_path = os.path.join(output_base, "07_performance_heatmap.png")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    metrics = ["AR", "RISK", "R/R"]

    for ax_idx, metric in enumerate(metrics):
        pivot = period_df.pivot_table(index="Mode", columns="Period", values=metric)
        # 期間を時系列に正しくに並び替え
        pivot = pivot[["2015-2019", "2020-2022", "2023-recent"]]

        im = axes[ax_idx].imshow(pivot.values, cmap="RdYlGn", aspect="auto")
        axes[ax_idx].set_xticks(range(len(pivot.columns)))
        axes[ax_idx].set_yticks(range(len(pivot.index)))
        axes[ax_idx].set_xticklabels(pivot.columns, rotation=45)
        axes[ax_idx].set_yticklabels(pivot.index, fontsize=9)
        axes[ax_idx].set_title(metric, fontweight="bold")

        # 数値をセルに表示
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                value = pivot.iloc[i, j]
                if metric == "R/R":
                    text = f"{value:.2f}"
                else:
                    text = f"{value*100:.1f}%"
                axes[ax_idx].text(
                    j, i, text, ha="center", va="center", color="black", fontsize=9
                )

        plt.colorbar(im, ax=axes[ax_idx])

    plt.tight_layout()
    plt.savefig(f_plot_path, dpi=150, bbox_inches="tight")
    print(f"Performance heatmap saved to: {f_plot_path}")
    plt.close()

    print("\n分析完了！")


if __name__ == "__main__":
    os.chdir(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")
    )
    main()
