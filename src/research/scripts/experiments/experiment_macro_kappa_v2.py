#!/usr/bin/env python
"""Macro Kappa v2 integration experiment.

Tests enabling macro factor-kappa (Omega_gap inflation) in the v2 production
pipeline by running BacktestEngine.run_v2_backtest with a modified config.
No production source files are changed; the experiment is isolated in this
script.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Pre-download and cache macro prices once to avoid repeated yfinance calls
# inside generate_v2_production_portfolio (which requests a 2-year window per date).
from leadlag.core import macro as _macro_module

logger.info("Pre-downloading macro prices for caching...")
_FULL_MACRO_PRICES = _macro_module.download_macro_prices(
    start="2018-01-01", end="2026-12-31"
)


def _cached_download_macro_prices(
    start: str | None = None,
    end: str | None = None,
    period: str = "10y",
    timeout: float = 30.0,
) -> pd.DataFrame:
    """Return macro prices sliced up to *end* (inclusive) to avoid lookahead.

    The full cached DataFrame extends to 2026, so we must not return future
    prices for historical backtest dates. We slice by the requested end date
    and require at least 30 rows for compute_macro_surprise to run.
    """
    df = _FULL_MACRO_PRICES.copy()
    if end is not None:
        end_ts = pd.Timestamp(end)
        df = df.loc[df.index <= end_ts]
    if start is not None:
        start_ts = pd.Timestamp(start)
        df = df.loc[df.index >= start_ts]
    return df


_macro_module.download_macro_prices = _cached_download_macro_prices

from leadlag.config.schemas import AppConfig
from leadlag.data.cache import load_df_exec_from_local_cache
from leadlag.execution.backtester import BacktestEngine
from leadlag.execution.config import load_config_from_yaml
from leadlag.reporting.metrics import calculate_metrics
from research.experiment_registry import Decision
from research.experiment_utils import record_backtest_experiment


def load_production_config() -> AppConfig:
    """Load canonical production.yaml as a Pydantic AppConfig."""
    prod_path = ROOT / "configs" / "production" / "production.yaml"
    return load_config_from_yaml(prod_path)


def build_cfg_with_macro(
    app_config: AppConfig, enabled: bool, kappas: list[float]
) -> AppConfig:
    """Return an AppConfig with the requested V2 macro kappa settings."""
    v2 = app_config.v2.model_copy(
        update={
            "macro_kappa_enabled": enabled,
            "macro_kappas": tuple(kappas),
            "macro_surprise_halflife_mean": app_config.v2.macro_surprise_halflife_mean,
            "macro_surprise_halflife_vol": app_config.v2.macro_surprise_halflife_vol,
        }
    )
    return app_config.model_copy(update={"v2": v2})


def run_v2_variant(
    app_config: AppConfig,
    df_exec: pd.DataFrame,
    gap_input_dir: Path,
    start_date: str,
    end_date: str,
    slippage_bps: float,
    overnight_alpha_long: float,
    overnight_alpha_short: float,
    buy_interest_annual: float,
    borrow_fee_annual: float,
    reverse_fee_bps: float,
    side_leverage: float,
    n_jobs: int = 1,
) -> dict[str, Any]:
    """Run a single v2 backtest variant and return flat metrics."""
    results = BacktestEngine.run_v2_backtest(
        app_config,
        gap_input_dir=gap_input_dir,
        df_exec=df_exec,
        start_date=start_date,
        end_date=end_date,
        slippage_bps=slippage_bps,
        overnight_alpha_long=overnight_alpha_long,
        overnight_alpha_short=overnight_alpha_short,
        buy_interest_annual=buy_interest_annual,
        borrow_fee_annual=borrow_fee_annual,
        reverse_fee_bps=reverse_fee_bps,
        side_leverage=side_leverage,
        n_jobs=n_jobs,
    )
    dr = results["daily_returns"]
    m = calculate_metrics(dr)
    avg_turnover = float(results["daily_turnover"].mean())
    avg_gross = float(results["daily_gross_exps"].mean())
    fallback_rate = float(results["daily_fallback"].mean())
    mdd = float(results["drawdown"].min())

    if app_config.v2.macro_kappa_enabled:
        kappa_str = "_".join(f"{k:.2f}" for k in app_config.v2.macro_kappas)
        rec_name = f"{Path(__file__).stem}_k{kappa_str}"
        rec_hypothesis = (
            f"Macro Kappa v2 enabled with kappas USDJPY={app_config.v2.macro_kappas[0]:.2f} "
            f"CLF={app_config.v2.macro_kappas[1]:.2f} TNX={app_config.v2.macro_kappas[2]:.2f}."
        )
    else:
        rec_name = f"{Path(__file__).stem}_baseline"
        rec_hypothesis = "Macro Kappa v2 baseline (disabled)."

    record_backtest_experiment(
        name=rec_name,
        hypothesis=rec_hypothesis,
        app_config=app_config,
        results=results,
        extra_metrics={
            "AR": m.get("AR", 0.0),
            "Sharpe": m.get("Sharpe", 0.0),
            "MDD": mdd,
            "Turnover": avg_turnover,
            "GrossExp": avg_gross,
            "FallbackRate": fallback_rate,
        },
        decision=Decision.PENDING,
    )

    return {
        "AR": m.get("AR", 0.0),
        "Sharpe": m.get("Sharpe", 0.0),
        "MDD": mdd,
        "Turnover": avg_turnover,
        "GrossExp": avg_gross,
        "FallbackRate": fallback_rate,
        "n_days": len(dr),
    }


def print_results_table(results: list[dict], title: str) -> None:
    """Print results list as a clean text table."""
    print(f"\n=== {title} ===")
    headers = [
        "Variant",
        "Sharpe",
        "AR (%)",
        "MDD (%)",
        "Turnover",
        "GrossExp",
        "Fallback (%)",
    ]
    print(
        f"{headers[0]:<45} | {headers[1]:>8} | {headers[2]:>8} | {headers[3]:>8} | "
        f"{headers[4]:>8} | {headers[5]:>8} | {headers[6]:>12}"
    )
    print("-" * 115)
    for r in results:
        print(
            f"{r['Variant']:<45} | "
            f"{r['Sharpe']:>8.2f} | "
            f"{r['AR'] * 100:>8.2f} | "
            f"{r['MDD'] * 100:>8.2f} | "
            f"{r['Turnover']:>8.3f} | "
            f"{r['GrossExp']:>8.3f} | "
            f"{r['FallbackRate'] * 100:>12.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Macro Kappa v2 Integration Experiment"
    )
    parser.add_argument(
        "--gap-input-dir",
        default=str(ROOT / "results" / "gap_adjusted_distribution" / "20260615_004202"),
        help="Directory with pre-computed gap matrices",
    )
    parser.add_argument(
        "--start-date", default="2020-01-06", help="Backtest start date"
    )
    parser.add_argument(
        "--end-date", default="2022-12-31", help="Backtest end date"
    )
    parser.add_argument(
        "--slippage-bps", type=float, default=5.0, help="Slippage bps per side"
    )
    parser.add_argument(
        "--production-costs",
        action="store_true",
        help="Use overnight alpha/long/short costs from production.yaml "
        "(default: 0.0/0.0 to match macro direction report)",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=1, help="Parallel workers for v2 backtest"
    )
    parser.add_argument(
        "--kappas",
        type=str,
        default=None,
        help="Comma-separated kappa triples to test (e.g. '3.0:0.5:0.5;1.5:0.5:0.5')",
    )
    args = parser.parse_args()

    base_app_config = load_production_config()
    costs_cfg = base_app_config.strategy

    if args.production_costs:
        alpha_long = costs_cfg.overnight_alpha_long
        alpha_short = costs_cfg.overnight_alpha_short
    else:
        alpha_long = 0.0
        alpha_short = 0.0

    slippage_bps = args.slippage_bps
    buy_interest = costs_cfg.buy_interest_annual
    borrow_fee = costs_cfg.borrow_fee_annual
    reverse_fee = costs_cfg.reverse_fee_bps
    side_leverage = 1.5

    gap_dir = Path(args.gap_input_dir)
    if not gap_dir.exists():
        logger.error("Gap input directory does not exist: %s", gap_dir)
        sys.exit(1)

    logger.info("Loading df_exec from local cache...")
    df_exec = load_df_exec_from_local_cache()

    all_results = []

    # Baseline: macro kappa disabled
    logger.info("Running baseline (macro kappa disabled)...")
    baseline_cfg = build_cfg_with_macro(base_app_config, enabled=False, kappas=[3.0, 0.5, 0.5])
    baseline_metrics = run_v2_variant(
        baseline_cfg,
        df_exec,
        gap_dir,
        args.start_date,
        args.end_date,
        slippage_bps,
        alpha_long,
        alpha_short,
        buy_interest,
        borrow_fee,
        reverse_fee,
        side_leverage,
        args.n_jobs,
    )
    baseline_metrics["Variant"] = "Baseline (macro kappa OFF)"
    all_results.append(baseline_metrics)

    # Kappa variants
    if args.kappas:
        kappa_triples = [
            [float(x) for x in triple.split(":")]
            for triple in args.kappas.split(";")
        ]
    else:
        kappa_triples = [
            [3.0, 0.5, 0.5],
            [1.5, 0.5, 0.5],
            [6.0, 1.0, 1.0],
            [3.0, 0.0, 0.0],
        ]

    for kappas in kappa_triples:
        variant_name = f"Kappa USDJPY={kappas[0]:.2f} CLF={kappas[1]:.2f} TNX={kappas[2]:.2f}"
        logger.info("Running %s...", variant_name)
        cfg = build_cfg_with_macro(base_app_config, enabled=True, kappas=kappas)
        m = run_v2_variant(
            cfg,
            df_exec,
            gap_dir,
            args.start_date,
            args.end_date,
            slippage_bps,
            alpha_long,
            alpha_short,
            buy_interest,
            borrow_fee,
            reverse_fee,
            side_leverage,
            args.n_jobs,
        )
        m["Variant"] = variant_name
        all_results.append(m)

    # Save and print
    out_dir = ROOT / "artifacts" / "macro_kappa_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(out_dir / "macro_kappa_v2_results.csv", index=False)

    print_results_table(all_results, "Macro Kappa v2 Integration Experiment")

    logger.info("Results saved to %s", out_dir / "macro_kappa_v2_results.csv")


if __name__ == "__main__":
    main()
