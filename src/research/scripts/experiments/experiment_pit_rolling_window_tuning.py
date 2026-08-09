#!/usr/bin/env python3
"""PIT IR ローリング窓（pit_rolling_window）のチューニング実験。

RuleD 動的グロス調整で使う PIT 三分位ビニングのローリング窓を変化させ、
net Sharpe / MDD / turnover / bin 分布の影響を測定する。
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("PITRollingWindowTuning")

ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.cache import load_df_exec_from_local_cache
from leadlag.execution.backtester import BacktestEngine
from leadlag.execution.config import load_config_from_yaml
from research.experiment_registry import Decision
from research.experiment_utils import record_backtest_experiment


def _metrics(results: dict, window: int, label: str) -> dict:
    returns = results["daily_returns"]
    fb = results["daily_fallback"]
    valid = returns[~fb]
    n_valid = len(valid)
    n_fallback = int(fb.sum())
    mean = float(np.mean(valid)) if n_valid > 0 else 0.0
    std = float(np.std(valid, ddof=1)) if n_valid > 1 else 0.0
    sharpe = mean / std * np.sqrt(252) if std > 1e-8 else 0.0
    mdd = float(results["drawdown"].min()) if len(results["drawdown"]) > 0 else 0.0
    turnover = float(np.mean(results["daily_turnover"][~fb])) if n_valid > 0 else 0.0
    gross = float(np.mean(results["daily_gross_exps"][~fb])) if n_valid > 0 else 0.0
    fb_rate = n_fallback / len(returns) * 100 if len(returns) > 0 else 0.0

    # PIT bin distribution from v2_summaries
    bins = [s.get("pit_bin", "Unknown") for s in results["v2_summaries"] if s]
    bin_counts = pd.Series(bins).value_counts().to_dict()

    # Weighted average multiplier
    mults = [s.get("gross_multiplier", np.nan) for s in results["v2_summaries"] if s]
    avg_mult = float(np.nanmean(mults)) if mults else np.nan

    return {
        "pit_rolling_window": window,
        "label": label,
        "period": f"{returns.index[0].date()} -> {returns.index[-1].date()}",
        "days": int(len(returns)),
        "valid_days": n_valid,
        "fallback_days": n_fallback,
        "fallback_rate_pct": fb_rate,
        "sharpe": sharpe,
        "cagr_pct": (
            float((results["equity_curve"].iloc[-1] ** (252 / n_valid) - 1) * 100)
            if n_valid > 0
            else 0.0
        ),
        "annualized_return_pct": mean * 252 * 100,
        "annualized_volatility_pct": std * np.sqrt(252) * 100,
        "max_drawdown_pct": mdd * 100,
        "avg_turnover": turnover,
        "avg_gross_exposure": gross,
        "avg_multiplier": avg_mult,
        "low_count": int(bin_counts.get("Low", 0)),
        "medium_count": int(bin_counts.get("Medium", 0)),
        "high_count": int(bin_counts.get("High", 0)),
    }


def _run_one(args: tuple) -> dict:
    """Worker: tune pit_rolling_window and run a V2 backtest."""
    import sys

    sys.path.insert(0, str(ROOT / "src"))

    window, app_config, gap_input_dir, start_date, end_date, output_dir = args
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    v2_for_run = app_config.v2.model_copy(update={"pit_rolling_window": window})
    app_config_for_run = app_config.model_copy(update={"v2": v2_for_run})

    df_exec = load_df_exec_from_local_cache()

    results = BacktestEngine.run_v2_backtest(
        cfg=app_config_for_run,
        gap_input_dir=Path(gap_input_dir),
        df_exec=df_exec,
        start_date=start_date,
        end_date=end_date,
        n_jobs=1,
    )

    results["daily_returns"].to_csv(output_dir / "daily_net_returns.csv", header=["net_return"])
    results["daily_gross_exps"].to_csv(output_dir / "daily_gross.csv", header=["gross"])
    results["daily_turnover"].to_csv(output_dir / "daily_turnover.csv", header=["turnover"])
    results["equity_curve"].to_csv(output_dir / "daily_equity_curve.csv", header=["equity"])
    results["drawdown"].to_csv(output_dir / "daily_drawdown.csv", header=["drawdown"])
    results["weights"].to_csv(output_dir / "daily_weights.csv")

    label = f"w{window}"
    record_backtest_experiment(
        name=f"{Path(__file__).stem}_{label}",
        hypothesis=f"PIT rolling window tuning (window={window}).",
        app_config=app_config,
        results=results,
        extra_metrics={"pit_rolling_window": window},
        decision=Decision.PENDING,
    )
    return _metrics(results, window, label)


def _plot_equity_curves(df: pd.DataFrame, results_root: Path, report_dir: Path) -> Path:
    plt.figure(figsize=(10, 6))
    for _, row in df.iterrows():
        label = row["label"]
        equity_path = results_root / label / "daily_equity_curve.csv"
        if not equity_path.exists():
            continue
        eq = pd.read_csv(equity_path, index_col=0, parse_dates=True)
        if eq.empty:
            continue
        cum_ret = eq["equity"] / eq["equity"].iloc[0] - 1.0
        plt.plot(eq.index, cum_ret * 100, label=f"{label} (w={row['pit_rolling_window']})")

    plt.title("PIT Rolling Window Tuning: Cumulative Net Return")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Net Return (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = report_dir / "equity_curves.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    return plot_path


def _build_report(df: pd.DataFrame, start_date: str, end_date: str, gap_input_dir: str) -> str:
    headers = ["Window", "net Sharpe", "CAGR (%)", "AR (%)", "Vol (%)", "MDD (%)", "Avg Gross", "Turnover", "Avg Mult", "Low", "Medium", "High"]
    table = "| " + " | ".join(headers) + " |\n"
    table += "| " + " | ".join(["---" for _ in headers]) + " |\n"
    for _, row in df.iterrows():
        table += (
            f"| {int(row['pit_rolling_window'])} | {row['sharpe']:.4f} | "
            f"{row['cagr_pct']:.2f} | {row['annualized_return_pct']:.2f} | "
            f"{row['annualized_volatility_pct']:.2f} | {row['max_drawdown_pct']:.2f} | "
            f"{row['avg_gross_exposure']:.3f} | {row['avg_turnover']:.3f} | "
            f"{row['avg_multiplier']:.3f} | {int(row['low_count'])} | "
            f"{int(row['medium_count'])} | {int(row['high_count'])} |\n"
        )

    best = df.loc[df["sharpe"].idxmax()]
    worst = df.loc[df["sharpe"].idxmin()]

    lines = [
        "# PIT ローリング窓（pit_rolling_window）チューニングレポート",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d')}",
        f"**Period**: {start_date} -> {end_date}",
        f"**Gap input**: `{gap_input_dir}`",
        "**Status**: tuning",
        "",
        "## 1. Hypothesis",
        "",
        "RuleD 動的グロス調整の PIT 三分位ビニングでは、過去 IR のローリング窓",
        "（`pit_rolling_window`）が閾値計算に影響する。",
        "窓が短すぎるとノイズに過敏になり、長すぎると制度変更等の非定常性を取り込みすぎる。",
        "252 営業日（約 1 年）を前後する窓で、リスク調整後リターンを最大化する。",
        "",
        "## 2. Methods",
        "",
        "- `cfg['gross_scaling']['pit_rolling_window']` を変更。",
        "- 比較対象: 63, 126, 189, 252, 378, 504, 756, 1008 営業日。",
        "- バックテスト: `BacktestEngine.run_v2_backtest`, overlay なし。",
        "- PIT 履歴は unlimited（canonical `full_history_diagnostics.csv`）。",
        "- 指標: net Sharpe, CAGR, MDD, average gross, turnover, average multiplier, PIT bin 分布。",
        "",
        "## 3. Results",
        "",
        table,
        "",
        "![Cumulative Net Return](equity_curves.png)",
        "",
        "## 4. Analysis",
        "",
        f"- 最高 net Sharpe: **{best['label']}** (window={best['pit_rolling_window']}) = {best['sharpe']:.4f}",
        f"- 最低 net Sharpe: **{worst['label']}** (window={worst['pit_rolling_window']}) = {worst['sharpe']:.4f}",
        "",
        "- PIT bin 分布:",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"  - {row['label']} (w={row['pit_rolling_window']}): "
            f"Low={row['low_count']}, Mid={row['medium_count']}, High={row['high_count']}, "
            f"avg mult={row['avg_multiplier']:.3f}"
        )

    lines += [
        "",
        "## 5. Conclusion",
        "",
        "（本レポートは実行後に数値を確認して判定を記述してください。）",
        "",
        "---",
        "",
        "## 6. Files",
        "",
        "- 実験スクリプト: `scripts/experiments/experiment_pit_rolling_window_tuning.py`",
        f"- サマリー: `reports/pit_rolling_window_tuning_{datetime.now().strftime('%Y%m%d')}/summary.csv`",
        f"- プロット: `reports/pit_rolling_window_tuning_{datetime.now().strftime('%Y%m%d')}/equity_curves.png`",
        f"- 日次データ: `results/pit_rolling_window_tuning_{datetime.now().strftime('%Y%m%d')}/w*/`",
        "",
    ]

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="PIT rolling window tuning")
    parser.add_argument("--start-date", default="2020-01-06")
    parser.add_argument("--end-date", default="2026-07-29")
    parser.add_argument("--gap-input-dir", default="var/live/pipeline_data/gap_adjusted_distribution/20260731_024303")
    parser.add_argument("--results-dir", default=f"var/results/pit_rolling_window_tuning_{datetime.now().strftime('%Y%m%d')}")
    parser.add_argument("--report-dir", default=f"reports/pit_rolling_window_tuning_{datetime.now().strftime('%Y%m%d')}")
    parser.add_argument("--window-list", default="63,126,189,252,378,504,756,1008")
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()

    results_root = ROOT / args.results_dir
    results_root.mkdir(parents=True, exist_ok=True)
    report_dir = ROOT / args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    app_config = load_config_from_yaml(ROOT / "configs/production/production.yaml")

    windows = [int(x) for x in args.window_list.split(",")]
    gap_input_dir = ROOT / args.gap_input_dir

    run_args = []
    for window in windows:
        label = f"w{window}"
        out = results_root / label
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True, exist_ok=True)
        run_args.append(
            (window, app_config, str(gap_input_dir), args.start_date, args.end_date, str(out))
        )

    metrics = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(_run_one, a): a[0] for a in run_args}
        for future in as_completed(futures):
            window = futures[future]
            try:
                m = future.result()
                metrics.append(m)
                logger.info(
                    "Done window=%d: Sharpe=%.4f, MDD=%.2f%%, AR=%.2f%%",
                    m["pit_rolling_window"],
                    m["sharpe"],
                    m["max_drawdown_pct"],
                    m["annualized_return_pct"],
                )
            except Exception as e:
                logger.error("window=%d failed: %s", window, e, exc_info=True)

    df = pd.DataFrame(metrics).sort_values("pit_rolling_window")
    df.to_csv(report_dir / "summary.csv", index=False)

    plot_path = _plot_equity_curves(df, results_root, report_dir)
    report = _build_report(df, args.start_date, args.end_date, args.gap_input_dir)
    with open(report_dir / "report.md", "w") as f:
        f.write(report)

    with open(results_root / "summary.json", "w") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2, default=str)

    print("\n" + "=" * 90)
    print(
        df[
            [
                "pit_rolling_window",
                "sharpe",
                "cagr_pct",
                "max_drawdown_pct",
                "avg_gross_exposure",
                "avg_multiplier",
                "low_count",
                "medium_count",
                "high_count",
            ]
        ].to_string(index=False)
    )
    print("=" * 90)
    print(f"\nReport: {report_dir / 'report.md'}")
    print(f"Plot:   {plot_path}")
    print(f"Results (ignored by git): {results_root}")


if __name__ == "__main__":
    main()
