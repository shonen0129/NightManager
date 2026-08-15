"""Shadow Run Tool: Compare Production V2 vs Next-Gen Pipeline.

Runs both Production V2 (Baseline Heuristic) and Next-Gen Pipeline (Convex Optimization)
in parallel on identical market data, auditing signal divergence, portfolio weights,
and ex-ante risk-adjusted metrics.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.broker.async_base import AsyncDryRunBrokerClient
from leadlag.data.market_data_cache import load_df_exec_from_local_cache
from leadlag.data.pit_lake import PITDataLake
from leadlag.data.tickers import JP_TICKERS
from leadlag.execution.config import load_config_from_yaml
from leadlag.execution.nextgen_pipeline import NextGenDecisionResult, NextGenPipeline
from leadlag.models.blpx import ProductionBLPXModel
from leadlag.models.production_v2 import (
    ProductionV2Model,
    _build_current_prices_from_df_exec,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_shadow_comparison(
    trade_date: str = "latest",
    config_path: str = "configs/production/production.yaml",
    capital: float = 1_000_000.0,
    pit_ir_history_path: str = "var/shadow/nextgen_pit_ir_history.csv",
) -> dict[str, Any]:
    """Run parallel shadow execution and comparison."""
    # 1. Load data & config
    df_exec = load_df_exec_from_local_cache()
    if df_exec is None:
        raise RuntimeError("df_exec cache not found.")

    lake = PITDataLake(df_exec)
    if trade_date == "latest":
        trade_date = str(lake.end_date.strftime("%Y-%m-%d"))

    app_config = load_config_from_yaml(config_path)

    # 2. Run Baseline Production V2
    print("\n=======================================================")
    print(f"  SHADOW RUN EXECUTION COMPARISON: {trade_date}")
    print("=======================================================")
    print("\n[1/2] Running Baseline Production V2...")
    t0_base = datetime.now()
    # ProductionBLPXModel expects the v2 config so nested blpx/costs settings resolve.
    blpx_model = ProductionBLPXModel(app_config.v2.model_dump())
    v2_model = ProductionV2Model(app_config.v2, blpx_model=blpx_model)
    current_prices = _build_current_prices_from_df_exec(df_exec, trade_date)

    base_decision = v2_model.decide(
        trade_date=trade_date,
        df_exec=df_exec,
        current_prices=current_prices,
        use_file_cache=False,  # On-demand comparison
    )
    w_base = base_decision["w_final"]
    t_base_elapsed = (datetime.now() - t0_base).total_seconds()

    # 3. Run Next-Gen Pipeline
    print("[2/2] Running Next-Gen Convex Pipeline...")
    t0_next = datetime.now()
    nextgen = NextGenPipeline(app_config, pit_ir_history_path=pit_ir_history_path)

    async def _run_nextgen() -> NextGenDecisionResult:
        async with AsyncDryRunBrokerClient(simulated_latency_ms=10.0) as broker:
            return await nextgen.run_daily_decision(
                trade_date=trade_date,
                lake=lake,
                broker=broker,
                capital=capital,
                submit_orders=True,
            )

    next_res = asyncio.run(_run_nextgen())
    w_next = next_res.raw_weights_array
    t_next_elapsed = (datetime.now() - t0_next).total_seconds()

    # 4. Metrics Comparison
    dot_prod = float(np.dot(w_base, w_next))
    norm_base = float(np.linalg.norm(w_base))
    norm_next = float(np.linalg.norm(w_next))
    cosine_sim = dot_prod / (norm_base * norm_next) if norm_base > 0 and norm_next > 0 else 0.0

    corr = float(np.corrcoef(w_base, w_next)[0, 1]) if norm_base > 0 and norm_next > 0 else 0.0

    # Selection overlap
    base_longs = set([JP_TICKERS[i] for i in np.where(w_base > 0.01)[0]])
    base_shorts = set([JP_TICKERS[i] for i in np.where(w_base < -0.01)[0]])
    next_longs = set([JP_TICKERS[i] for i in np.where(w_next > 0.01)[0]])
    next_shorts = set([JP_TICKERS[i] for i in np.where(w_next < -0.01)[0]])

    long_overlap = len(base_longs & next_longs) / max(len(base_longs | next_longs), 1)
    short_overlap = len(base_shorts & next_shorts) / max(len(base_shorts | next_shorts), 1)

    # Print Weight Comparison Table
    print(f"\n{'Ticker':<10} | {'Baseline Weight':<18} | {'Next-Gen Weight':<18} | {'Delta':<12}")
    print("-" * 65)
    for i, tk in enumerate(JP_TICKERS):
        delta = w_next[i] - w_base[i]
        print(f"{tk:<10} | {w_base[i]:>18.4f} | {w_next[i]:>18.4f} | {delta:>+12.4f}")
    print("-" * 65)

    base_summary = base_decision.get("summary", {})
    base_ex_ante_ret = float(base_summary.get("predicted_portfolio_mean", 0.0))
    base_ex_ante_vol = float(base_summary.get("predicted_portfolio_vol", 0.0))
    base_ex_ante_ir = float(base_summary.get("predicted_portfolio_ir", 0.0))

    summary_metrics = [
        ("Trade Date", trade_date, trade_date),
        ("Net Exposure", f"{np.sum(w_base):.6f}", f"{np.sum(w_next):.6f}"),
        ("Gross Exposure", f"{np.sum(np.abs(w_base)):.4f}", f"{np.sum(np.abs(w_next)):.4f}"),
        ("Weight Cosine Similarity", "1.0000 (Ref)", f"{cosine_sim:.4f}"),
        ("Weight Correlation", "1.0000 (Ref)", f"{corr:.4f}"),
        ("Long Selection Overlap", "100.0% (Ref)", f"{long_overlap*100:.1f}%"),
        ("Short Selection Overlap", "100.0% (Ref)", f"{short_overlap*100:.1f}%"),
        ("Active Long Positions", f"{len(base_longs)}", f"{len(next_longs)}"),
        ("Active Short Positions", f"{len(base_shorts)}", f"{len(next_shorts)}"),
        ("Ex-ante Return", f"{base_ex_ante_ret*10000:.2f} bps", f"{next_res.opt_result.ex_ante_return*10000:.2f} bps"),
        ("Ex-ante Volatility", f"{base_ex_ante_vol*10000:.2f} bps", f"{next_res.opt_result.ex_ante_vol*10000:.2f} bps"),
        ("Ex-ante IR", f"{base_ex_ante_ir:.4f}", f"{next_res.opt_result.ex_ante_ir:.4f}"),
        ("Execution Time", f"{t_base_elapsed:.3f}s", f"{t_next_elapsed:.3f}s"),
    ]

    print("\n" + "=" * 65)
    print(f"{'Metric':<26} | {'Baseline V2':<16} | {'Next-Gen Convex':<16}")
    print("=" * 65)
    for name, b_val, n_val in summary_metrics:
        print(f"{name:<26} | {b_val:>16} | {n_val:>16}")
    print("=" * 65)

    # Save Markdown Report
    out_dir = ROOT / "reports/nextgen_shadow_run"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / f"shadow_comparison_{trade_date.replace('-', '')}.md"

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"# Next-Gen Shadow Run Comparison Report: {trade_date}\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("| Metric | Baseline V2 | Next-Gen Convex |\n")
        f.write("|---|---|---|\n")
        for name, b_val, n_val in summary_metrics:
            f.write(f"| {name} | {b_val} | {n_val} |\n")
        f.write("\n\n### Per-Ticker Weight Comparison\n\n")
        f.write("| Ticker | Baseline Weight | Next-Gen Weight | Delta |\n")
        f.write("|---|---|---|---|\n")
        for i, tk in enumerate(JP_TICKERS):
            f.write(f"| {tk} | {w_base[i]:.4f} | {w_next[i]:.4f} | {w_next[i]-w_base[i]:+.4f} |\n")

    print(f"\n[SUCCESS] Shadow Run Report generated: {report_file}")
    return {"trade_date": trade_date, "cosine_sim": cosine_sim, "correlation": corr}


def main() -> None:
    parser = argparse.ArgumentParser(description="Next-Gen Shadow Run Comparison Tool")
    parser.add_argument("--trade-date", default="latest", help="Trade date (YYYY-MM-DD or latest)")
    parser.add_argument("--config", default="configs/production/production.yaml", help="YAML config path")
    parser.add_argument("--capital", type=float, default=1_000_000.0, help="Capital in JPY")
    args = parser.parse_args()

    # Market holiday check for explicit trade dates (skip non-trading days)
    if args.trade_date != "latest":
        from leadlag.core.market_calendar import is_market_closed

        check_date = datetime.strptime(args.trade_date, "%Y-%m-%d").date()
        if is_market_closed(check_date):
            logger.info("Market closed on %s. Skipping nextgen-shadow.", check_date)
            return

    run_shadow_comparison(
        trade_date=args.trade_date,
        config_path=args.config,
        capital=args.capital,
    )


if __name__ == "__main__":
    main()
