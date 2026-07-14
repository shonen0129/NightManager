#!/usr/bin/env python
"""A5: BayesianBLPX Kalman修正の検証実験.

修正内容:
  1. R3リーク修正: y_jp_target[i] → y_jp_target[i-1]（1日ラグ化）
  2. Q推定修正: 非重複サブサンプリングでautocorrelation biasを除去
  3. R推定修正: 理論値 (Var(ΔB̂) - Q) * window / 2 でbias-correct

比較:
  - Baseline (BLPEnhanced, no Bayesian)
  - Bayesian ic mode (修正後)
  - Bayesian kalman mode (修正後)
  - Bayesian cs_var mode (修正後)

注意: 本番configはBLPEnhancedを使用するため、本実験は研究目的。
BayesianBLPXが本番に勝たなければ、kalman/cs_varモードは削除候補。

Usage:
  python3 scripts/experiments/experiment_a5_bayesian_kalman_fix.py \
    --start-date 2015-01-05 --output-dir reports/sprint_a5_bayesian_fix
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.cache import load_df_exec_from_local_cache
from leadlag.data.tickers import JP_TICKERS
from leadlag.models.sre import compute_jp_target_returns
from leadlag.models.sector_relative_ensemble_blp_enhanced import SectorRelativeEnsembleBLPEnhancedModel
from leadlag.models.bayesian_blpx import BayesianBLPXModel
from leadlag.execution.backtester import BacktestEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TRADING_DAYS = 245


def compute_metrics(daily_returns: pd.Series, name: str | None = None) -> dict:
    dr = daily_returns.dropna()
    ar = float(dr.mean() * TRADING_DAYS)
    vol = float(dr.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe = ar / vol if vol > 0 else np.nan
    wealth = (1.0 + dr).cumprod()
    mdd = float(((wealth / wealth.cummax()) - 1.0).min())
    m = {"Sharpe_net": sharpe, "AR_net": ar, "Vol_net": vol, "MDD": mdd, "n_days": len(dr)}
    if name:
        m["variant"] = name
    return m


def run_backtest(model, df_exec, start_date, slippage_bps=5.0):
    return BacktestEngine.run_backtest(
        model, df_exec, start_date=start_date, slippage_bps=slippage_bps,
        overnight_alpha_long=0.75, overnight_alpha_short=0.5,
        buy_interest_annual=0.025, borrow_fee_annual=0.0115,
        reverse_fee_bps=2.0,
    )


def main():
    parser = argparse.ArgumentParser(description="A5: Bayesian Kalman Fix Validation")
    parser.add_argument("--start-date", default="2015-01-05")
    parser.add_argument("--output-dir", default="reports/sprint_a5_bayesian_fix")
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--eta-base", type=float, default=0.3)
    args = parser.parse_args()

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    import yaml
    with open(ROOT / "configs" / "production" / "production.yaml") as f:
        cfg_base = yaml.safe_load(f)

    logger.info("Loading df_exec...")
    df_exec = load_df_exec_from_local_cache()
    T = len(df_exec)
    y_target = compute_jp_target_returns(df_exec, JP_TICKERS)
    sim_dates = df_exec.index
    start_idx = max(df_exec.index.searchsorted(pd.to_datetime(args.start_date)), 60)

    all_results = {}

    # 1. Baseline: BLPEnhanced (no Bayesian)
    logger.info("=== Baseline: BLPEnhanced (no Bayesian) ===")
    model_base = SectorRelativeEnsembleBLPEnhancedModel(copy.deepcopy(cfg_base))
    model_base._start_date = args.start_date
    t0 = time.perf_counter()
    results_base = run_backtest(model_base, df_exec, args.start_date, args.slippage_bps)
    elapsed = time.perf_counter() - t0
    metrics_base = compute_metrics(results_base["daily_returns"], "baseline_blp_enhanced")
    all_results["baseline"] = metrics_base
    logger.info("Baseline: Sharpe=%.4f MDD=%.2f%% (%.1fs)",
                metrics_base["Sharpe_net"], metrics_base["MDD"] * 100, elapsed)

    # 2. Bayesian modes (with fixes applied)
    modes = ["ic", "kalman", "cs_var"]
    for mode in modes:
        logger.info("=== Bayesian mode=%s (post-fix) ===", mode)
        cfg_bayes = copy.deepcopy(cfg_base)
        cfg_bayes["bayesian_enabled"] = True
        cfg_bayes["bayesian_mode"] = mode
        cfg_bayes["bayesian_eta_base"] = args.eta_base
        cfg_bayes["bayesian_ic_window"] = 63
        cfg_bayes["bayesian_ic_amplifier"] = 5.0
        cfg_bayes["bayesian_eta_min"] = 0.05
        cfg_bayes["bayesian_eta_max"] = 0.80
        cfg_bayes["bayesian_kalman_window"] = 63
        cfg_bayes["bayesian_kalman_q_scale"] = 1.0

        model_bayes = BayesianBLPXModel(cfg_bayes)
        model_bayes._start_date = args.start_date
        t0 = time.perf_counter()
        results_bayes = run_backtest(model_bayes, df_exec, args.start_date, args.slippage_bps)
        elapsed = time.perf_counter() - t0
        metrics_bayes = compute_metrics(results_bayes["daily_returns"], f"bayesian_{mode}_postfix")
        all_results[f"bayesian_{mode}"] = metrics_bayes
        logger.info("Bayesian[%s]: Sharpe=%.4f MDD=%.2f%% (%.1fs)",
                    mode, metrics_bayes["Sharpe_net"], metrics_bayes["MDD"] * 100, elapsed)

        # Save daily returns
        results_bayes["daily_returns"].to_csv(out_dir / f"daily_returns_bayesian_{mode}.csv")

        # Save diagnostics if available
        pred = model_bayes.predict_signals(df_exec)
        if "bayesian_diagnostics" in pred and len(pred["bayesian_diagnostics"]) > 0:
            pred["bayesian_diagnostics"].to_csv(out_dir / f"bayesian_diagnostics_{mode}.csv")

    # 3. Summary
    results_df = pd.DataFrame(list(all_results.values()))
    results_df.to_csv(out_dir / "metrics_comparison.csv", index=False)

    # 4. Report
    report_lines = [
        "# A5: BayesianBLPX Kalman Fix Report\n",
        f"## Data: {T} rows, start={args.start_date}\n",
        f"\n## Fixes Applied\n",
        f"1. **R3 leak fix**: `y_jp_target[i]` → `y_jp_target[i-1]` (1-day lag)\n",
        f"2. **Q estimation**: non-overlapping subsampling (every blp_window steps)\n",
        f"3. **R estimation**: bias-corrected theoretical value `(Var(ΔB̂) - Q) * window / 2`\n",
        f"\n## Metrics Comparison\n",
        results_df.to_string(index=False),
        f"\n\n## Verdict\n",
    ]

    base_sharpe = all_results["baseline"]["Sharpe_net"]
    for mode in modes:
        mode_sharpe = all_results[f"bayesian_{mode}"]["Sharpe_net"]
        diff = (mode_sharpe - base_sharpe) / base_sharpe * 100
        report_lines.append(f"- Bayesian[{mode}]: Sharpe {mode_sharpe:.4f} vs baseline {base_sharpe:.4f} ({diff:+.1f}%)\n")

    best_bayes = max(all_results[f"bayesian_{m}"]["Sharpe_net"] for m in modes)
    if best_bayes > base_sharpe * 1.01:
        report_lines.append(f"\nBest Bayesian mode improves Sharpe by >1%.\n")
        report_lines.append("Recommend: investigate further but note BayesianBLPX is not in production config.\n")
    else:
        report_lines.append(f"\nNo Bayesian mode improves over baseline BLPEnhanced.\n")
        report_lines.append("Recommend: deprecate BayesianBLPX (kalman/cs_var/ic modes) to reduce code complexity.\n")
        report_lines.append("Note: R3 leak fix is still correct regardless of performance impact.\n")

    report_text = "\n".join(report_lines)
    (out_dir / "a5_bayesian_fix_report.md").write_text(report_text)
    logger.info("Report saved to %s/a5_bayesian_fix_report.md", out_dir)
    print("\n" + report_text)


if __name__ == "__main__":
    main()
