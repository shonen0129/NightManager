"""compare_linear_vs_corrected.py – ベンチマーク比較レポート出力スクリプト.

walk_forward_correction.py が生成した OOS リターン CSV を読み込み、
詳細な比較レポート（テキスト + CSV + チャート）を出力します。

単独実行例:
    python scripts/compare_linear_vs_corrected.py \
        --input results/correction_walkforward/oos_net_returns.csv \
        --n_trials 10

walk_forward_correction.py のあとに自動的に呼び出すことも可能です。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent
SRC_DIR = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from domain.correction.evaluation import (
    CostModel,
    PerformanceMetrics,
    compute_net_returns,
    compute_performance_metrics,
    compute_signal_ic,
    deflated_sharpe_ratio,
    evaluate_correction_adoption,
)


def _monthly_returns(s: pd.Series) -> pd.Series:
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index)
    return (1.0 + s).groupby(s.index.to_period("M")).prod() - 1.0


def generate_report(
    df: pd.DataFrame,
    n_trials: int,
    significance_level: float,
    output_dir: Path,
) -> None:
    """Generate comparison report from pre-computed net return columns.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns 'net_linear' and 'net_corrected' with DatetimeIndex.
    n_trials : int
        Number of hyperparameter configurations evaluated (for DSR).
    significance_level : float
        DSR significance threshold.
    output_dir : Path
        Where to write reports.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    lin = df["net_linear"].dropna()
    cor = df["net_corrected"].dropna()

    m_lin = compute_performance_metrics(lin, label="Linear Baseline", n_trials=1)
    m_cor = compute_performance_metrics(cor, label="Linear + GBT Correction", n_trials=n_trials)

    decision = evaluate_correction_adoption(m_lin, m_cor, significance_level)

    # ── Console report ───────────────────────────────────────────────────
    sep = "=" * 72
    print(f"\n{sep}")
    print("  非線形補正層 vs 線形ベンチマーク 比較レポート")
    print(sep)
    print(f"\n{'指標':<28} {'線形ベンチマーク':>18} {'補正層あり':>18} {'差分':>10}")
    print("-" * 76)

    def _fmt_pct(v): return f"{v*100:.2f}%" if np.isfinite(v) else "N/A"
    def _fmt(v): return f"{v:.3f}" if np.isfinite(v) else "N/A"

    rows = [
        ("年率リターン (AR, net)",    m_lin.ar,    m_cor.ar,    _fmt_pct),
        ("年率リスク (RISK)",          m_lin.risk,  m_cor.risk,  _fmt_pct),
        ("R/R (AR / RISK)",           m_lin.rr,    m_cor.rr,    _fmt),
        ("最大DD (MDD)",              m_lin.mdd,   m_cor.mdd,   _fmt_pct),
        ("Sharpe Ratio",              m_lin.sharpe, m_cor.sharpe, _fmt),
        ("IC 平均 (Spearman)",        m_lin.ic_mean, m_cor.ic_mean, _fmt),
        ("DSR p値 (多重比較補正後)", m_lin.dsr_pvalue, m_cor.dsr_pvalue, _fmt),
        ("SR* (H0下の期待最大SR)",   m_lin.sr_star,   m_cor.sr_star, _fmt),
        ("OOS 日数",                  float(m_lin.n_obs), float(m_cor.n_obs), _fmt),
        ("試行ハイパーパラメータ数", float(m_lin.n_trials), float(m_cor.n_trials), _fmt),
    ]

    for label, v_lin, v_cor, fmt in rows:
        diff = v_cor - v_lin
        diff_str = ("+" if diff > 0 else "") + fmt(diff)
        print(f"{label:<28} {fmt(v_lin):>18} {fmt(v_cor):>18} {diff_str:>10}")

    print(sep)
    print(str(decision))

    # ── Monthly breakdown ────────────────────────────────────────────────
    m_lin_monthly = _monthly_returns(lin)
    m_cor_monthly = _monthly_returns(cor)
    monthly_df = pd.DataFrame({
        "linear": m_lin_monthly,
        "corrected": m_cor_monthly,
        "diff": m_cor_monthly - m_lin_monthly,
    })
    monthly_path = output_dir / "monthly_comparison.csv"
    monthly_df.to_csv(monthly_path, encoding="utf-8-sig")
    print(f"\n月次リターン比較: {monthly_path}")

    # ── Metrics CSV ──────────────────────────────────────────────────────
    metrics_rows = []
    for m in [m_lin, m_cor]:
        metrics_rows.append({
            "label": m.label,
            "AR": m.ar,
            "RISK": m.risk,
            "R/R": m.rr,
            "MDD": m.mdd,
            "Sharpe": m.sharpe,
            "IC_mean": m.ic_mean,
            "DSR_pvalue": m.dsr_pvalue,
            "SR_star": m.sr_star,
            "n_obs": m.n_obs,
            "n_trials": m.n_trials,
        })
    pd.DataFrame(metrics_rows).to_csv(
        output_dir / "metrics_comparison.csv", index=False, encoding="utf-8-sig"
    )

    # ── Decision JSON ────────────────────────────────────────────────────
    with open(output_dir / "adoption_decision.json", "w", encoding="utf-8") as f:
        json.dump({
            "decision": decision.decision,
            "reason": decision.reason,
            "n_trials": n_trials,
            "significance_level": significance_level,
        }, f, indent=2, ensure_ascii=False)

    # ── Charts ───────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker

        fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True)

        # 1. Cumulative returns
        (1 + lin).cumprod().plot(ax=axes[0], label="Linear", linewidth=1.3, color="steelblue")
        (1 + cor).cumprod().plot(ax=axes[0], label="Linear + GBT", linewidth=1.3,
                                 color="darkorange", linestyle="--")
        axes[0].set_title("OOS 累積ネットリターン（取引コスト控除後）", fontsize=12)
        axes[0].set_ylabel("Wealth")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # 2. Drawdowns
        def _dd(s): return ((1 + s).cumprod() / (1 + s).cumprod().cummax()) - 1
        _dd(lin).plot(ax=axes[1], label="Linear", linewidth=1.0, color="steelblue")
        _dd(cor).plot(ax=axes[1], label="Linear + GBT", linewidth=1.0,
                      color="darkorange", linestyle="--")
        axes[1].set_title("ドローダウン比較", fontsize=12)
        axes[1].set_ylabel("Drawdown")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

        # 3. Monthly IC / returns bar
        excess = m_cor_monthly - m_lin_monthly
        colors = ["green" if x >= 0 else "red" for x in excess]
        axes[2].bar(range(len(excess)), excess.values, color=colors, alpha=0.7)
        axes[2].set_title("月次超過リターン（補正層 − 線形）", fontsize=12)
        axes[2].set_ylabel("Excess Return")
        axes[2].axhline(0, color="black", linewidth=0.8)
        axes[2].grid(alpha=0.3, axis="y")

        plt.tight_layout()
        chart_path = output_dir / "linear_vs_corrected.png"
        plt.savefig(chart_path, dpi=150)
        plt.close()
        print(f"チャート: {chart_path}")
    except Exception as e:
        print(f"[警告] チャート生成失敗: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="線形 vs 補正層 比較レポート")
    parser.add_argument(
        "--input",
        default="results/correction_walkforward/oos_net_returns.csv",
        help="OOS ネットリターン CSV (walk_forward_correction.py の出力)",
    )
    parser.add_argument("--n_trials", type=int, default=1, help="試行 HP 数 (DSR 補正用)")
    parser.add_argument(
        "--significance_level", type=float, default=0.05, help="DSR 有意水準"
    )
    parser.add_argument(
        "--output_dir",
        default="results/correction_report",
        help="レポート出力ディレクトリ",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        print("先に walk_forward_correction.py を実行してください。")
        sys.exit(1)

    df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    output_dir = Path(args.output_dir)

    generate_report(
        df=df,
        n_trials=args.n_trials,
        significance_level=args.significance_level,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
