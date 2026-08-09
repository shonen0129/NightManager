#!/usr/bin/env python
"""Fractional diff weight-normalization experiment.

Tests how fixing the truncation bias (via weight normalization) affects
signal quality and a simplified long-short portfolio.

The full v2 production backtest uses pre-computed gap-adjusted distribution
matrices.  Those matrices are produced by compute_gap_adjusted_distribution.py,
which itself uses the fractional-diff-transformed US returns in
build_common_inputs.  Re-running the full Step-2 pipeline for every variant
would be expensive and requires Step-1 diagnostics inputs that are not
persisted in this repo.  As a proxy we therefore evaluate:

1. Signal quality (Rank IC, ICIR, hit rate) of the residual-BLPX ensemble signal.
2. A simplified long-short quintile portfolio (top/bottom 5 names, equal weight,
   daily rebal, 5 bps one-way cost).

This is a fast proxy; any large degradation here strongly predicts a
production backtest degradation.
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

from leadlag.data.cache import load_df_exec_from_local_cache
from leadlag.models.sector_relative_ensemble_blp_enhanced import (
    SectorRelativeEnsembleBLPEnhancedModel,
)
from leadlag.models.sre import compute_jp_target_returns


def load_production_config() -> dict:
    """Load canonical production.yaml."""
    prod_path = ROOT / "configs" / "production" / "production.yaml"
    with open(prod_path) as f:
        return yaml.safe_load(f)


def make_cfg_variant(base_cfg: dict, normalize: str | None, window: int | None = None) -> dict:
    """Return a deep copy with fractional_diff normalize/window overridden."""
    cfg = copy.deepcopy(base_cfg)
    fd = cfg.setdefault("features", {}).setdefault("fractional_diff", {})
    fd["enabled"] = True
    fd["d"] = 0.1
    fd["threshold"] = 1e-5
    if normalize is not None:
        fd["normalize"] = normalize
    else:
        fd.pop("normalize", None)
    if window is not None:
        fd["window"] = window
    return cfg


def compute_signal_quality(signal: pd.DataFrame, target: pd.DataFrame) -> dict[str, float]:
    """Compute Rank IC, ICIR, hit rate, and long-short spread."""
    rank_ics = []
    hit = 0
    spread_bps = []
    valid_days = 0

    for date in signal.index:
        if date not in target.index:
            continue
        s = signal.loc[date].dropna()
        t = target.loc[date].dropna()
        common = s.index.intersection(t.index)
        if len(common) < 6:
            continue
        s = s[common]
        t = t[common]

        # Rank IC
        r_s = s.rank()
        r_t = t.rank()
        rank_ic = r_s.corr(r_t, method="pearson")
        if np.isfinite(rank_ic):
            rank_ics.append(rank_ic)

        # Hit rate (top/bottom half agreement sign)
        s_sign = np.sign(s - s.median())
        t_sign = np.sign(t - t.median())
        hit += (s_sign == t_sign).mean()

        # Long-short spread (top/bottom quintile)
        n = len(common)
        q = max(1, n // 5)
        long_idx = s.sort_values(ascending=False).index[:q]
        short_idx = s.sort_values(ascending=True).index[:q]
        long_ret = t[long_idx].mean()
        short_ret = t[short_idx].mean()
        spread_bps.append((long_ret - short_ret) * 10000)

        valid_days += 1

    rank_ics = pd.Series(rank_ics)
    rank_ic_mean = rank_ics.mean()
    rank_ic_std = rank_ics.std(ddof=1)
    icir = rank_ic_mean / rank_ic_std if rank_ic_std > 1e-12 else 0.0
    hit_rate = hit / valid_days if valid_days > 0 else 0.0
    spread = pd.Series(spread_bps)
    return {
        "RankIC": rank_ic_mean,
        "RankIC_std": rank_ic_std,
        "ICIR": icir,
        "HitRate": hit_rate,
        "LSSpread_bps": spread.mean(),
        "LSSpread_std_bps": spread.std(ddof=1),
    }


def compute_simple_ls_backtest(signal: pd.DataFrame, target: pd.DataFrame, cost_bps: float = 5.0) -> dict[str, float]:
    """Simple daily long-short quintile portfolio with one-way cost.

    Returns annualized return, Sharpe, MDD, and average daily turnover computed
    directly from the daily PnL series.  We avoid calculate_metrics here because
    the simple quintile series has a few very strong days that make monthly
    compounding unstable for this proxy exercise.
    """
    daily_pnl = []
    turnover = []
    prev_long = set()
    prev_short = set()

    for date in signal.index:
        if date not in target.index:
            continue
        s = signal.loc[date].dropna()
        t = target.loc[date].dropna()
        common = s.index.intersection(t.index)
        if len(common) < 6:
            daily_pnl.append(0.0)
            continue
        s = s[common]
        t = t[common]

        n = len(common)
        q = max(1, n // 5)
        long_idx = s.sort_values(ascending=False).index[:q]
        short_idx = s.sort_values(ascending=True).index[:q]

        long_ret = t[long_idx].mean()
        short_ret = t[short_idx].mean()
        gross_pnl = long_ret - short_ret

        # cost: 5 bps per leg, both sides turn over
        long_turn = len(long_idx.difference(prev_long)) / len(long_idx) if long_idx.size else 0.0
        short_turn = len(short_idx.difference(prev_short)) / len(short_idx) if short_idx.size else 0.0
        # each turnover leg pays cost_bps per side
        cost = (long_turn + short_turn) * 2 * cost_bps / 10000.0

        net_pnl = gross_pnl - cost
        daily_pnl.append(net_pnl)
        turnover.append((long_turn + short_turn) / 2.0)
        prev_long = set(long_idx)
        prev_short = set(short_idx)

    daily_pnl = pd.Series(daily_pnl, index=signal.index[:len(daily_pnl)]).dropna()
    turnover = pd.Series(turnover).dropna()

    if len(daily_pnl) == 0 or daily_pnl.std(ddof=1) < 1e-12:
        return {"Sharpe": 0.0, "AR": 0.0, "MDD": 0.0, "Turnover": 0.0}

    ann_factor = 252.0
    daily_mean = daily_pnl.mean()
    daily_std = daily_pnl.std(ddof=1)
    sharpe = (daily_mean / daily_std) * np.sqrt(ann_factor)
    ann_ret = daily_mean * ann_factor

    cum = daily_pnl.cumsum()
    running_max = cum.cummax()
    drawdown = cum - running_max
    mdd = float(drawdown.min())

    return {
        "Sharpe": sharpe,
        "AR": ann_ret,
        "MDD": mdd,
        "Turnover": float(turnover.mean()) if len(turnover) > 0 else 0.0,
    }


def run_variant(cfg: dict, df_exec: pd.DataFrame, start_date: str, end_date: str, n_jobs: int = 1) -> dict[str, Any]:
    """Run one fractional-diff variant and return signal quality + simple LS metrics."""
    from leadlag.data.tickers import JP_TICKERS

    # predict_signals needs the full history (including baseline 2010-2014) to
    # build c_full.  We slice the results after signal generation.
    model = SectorRelativeEnsembleBLPEnhancedModel(cfg)
    signals = model.predict_signals(df_exec, n_jobs=n_jobs)

    combined = signals.get("signals")
    if combined is None:
        raise KeyError(f"Expected 'signals' in predict_signals output; got {list(signals.keys())}")

    signal_df = combined.copy()
    if not isinstance(signal_df, pd.DataFrame):
        signal_df = pd.DataFrame(signal_df, index=df_exec.index, columns=JP_TICKERS)
    target_arr = compute_jp_target_returns(df_exec, JP_TICKERS)
    target_df = pd.DataFrame(target_arr, index=df_exec.index, columns=JP_TICKERS)

    mask = (df_exec.index >= pd.to_datetime(start_date)) & (df_exec.index <= pd.to_datetime(end_date))
    signal_df = signal_df.loc[mask]
    target_df = target_df.loc[mask]
    if len(signal_df) == 0:
        raise ValueError(f"No rows between {start_date} and {end_date}")

    sq = compute_signal_quality(signal_df, target_df)
    ls = compute_simple_ls_backtest(signal_df, target_df)

    return {
        "SignalQuality": sq,
        "SimpleLS": ls,
        "n_days": len(signal_df),
    }


def print_results_table(results: list[dict]) -> None:
    """Print results as a clean table."""
    print("\n=== Fractional Diff Normalization Experiment ===")
    hdr = ["Variant", "RankIC", "ICIR", "HitRate", "LSSpread_bps", "Sharpe", "AR(%)", "MDD(%)", "Turnover"]
    print(
        f"{hdr[0]:<35} | {hdr[1]:>8} | {hdr[2]:>8} | {hdr[3]:>8} | {hdr[4]:>12} | "
        f"{hdr[5]:>8} | {hdr[6]:>8} | {hdr[7]:>8} | {hdr[8]:>10}"
    )
    print("-" * 120)
    for r in results:
        sq = r["SignalQuality"]
        ls = r["SimpleLS"]
        print(
            f"{r['Variant']:<35} | "
            f"{sq['RankIC']:>8.4f} | "
            f"{sq['ICIR']:>8.4f} | "
            f"{sq['HitRate']:>8.4f} | "
            f"{sq['LSSpread_bps']:>12.2f} | "
            f"{ls['Sharpe']:>8.2f} | "
            f"{ls['AR']*100:>8.2f} | "
            f"{ls['MDD']*100:>8.2f} | "
            f"{ls['Turnover']:>10.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fractional Diff Weight Normalization Experiment"
    )
    parser.add_argument("--start-date", default="2015-01-05", help="Start date")
    parser.add_argument("--end-date", default="2024-12-31", help="End date")
    parser.add_argument(
        "--variants",
        type=str,
        default="baseline,zero,unit,window252,window504",
        help="Comma-separated variants to test",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=1, help="Parallel workers for predict_signals"
    )
    args = parser.parse_args()

    base_cfg = load_production_config()

    logger.info("Loading df_exec from local cache...")
    df_exec = load_df_exec_from_local_cache()

    variants = []
    variant_cfgs = {
        "baseline": make_cfg_variant(base_cfg, None),
        "zero": make_cfg_variant(base_cfg, "zero"),
        "unit": make_cfg_variant(base_cfg, "unit"),
        "window252": make_cfg_variant(base_cfg, None, window=252),
        "window504": make_cfg_variant(base_cfg, None, window=504),
    }
    for name in args.variants.split(","):
        name = name.strip()
        if name in variant_cfgs:
            variants.append((name, variant_cfgs[name]))
        else:
            logger.warning("Unknown variant %s, skipping", name)

    results = []
    for name, cfg in variants:
        logger.info("Running variant %s...", name)
        try:
            metrics = run_variant(cfg, df_exec, args.start_date, args.end_date, n_jobs=args.n_jobs)
            metrics["Variant"] = name
            results.append(metrics)
        except Exception as e:
            logger.exception("Variant %s failed: %s", name, e)

    if not results:
        logger.error("No variants completed successfully")
        sys.exit(1)

    print_results_table(results)

    # Save
    out_dir = ROOT / "artifacts" / "fractional_diff_normalization"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in results:
        sq = r["SignalQuality"]
        ls = r["SimpleLS"]
        rows.append({
            "Variant": r["Variant"],
            "n_days": r["n_days"],
            "RankIC": sq["RankIC"],
            "RankIC_std": sq["RankIC_std"],
            "ICIR": sq["ICIR"],
            "HitRate": sq["HitRate"],
            "LSSpread_bps": sq["LSSpread_bps"],
            "LSSpread_std_bps": sq["LSSpread_std_bps"],
            "Sharpe": ls["Sharpe"],
            "AR": ls["AR"],
            "MDD": ls["MDD"],
            "Turnover": ls["Turnover"],
        })
    pd.DataFrame(rows).to_csv(out_dir / "fractional_diff_normalization_results.csv", index=False)
    logger.info("Results saved to %s", out_dir / "fractional_diff_normalization_results.csv")


if __name__ == "__main__":
    main()
