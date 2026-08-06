#!/usr/bin/env python3
"""VIX regime overlay experiment: scale gross based on US vs Japan VIX leadership.

Hypothesis:
- US VIX high -> US-led shock -> maintain/trust BLPX signal.
- JP VIX high while US VIX low -> Japan-led shock -> reduce gross because US
  information is less predictive.

This script downloads US VIX (^VIX) and Nikkei VI (^NKVI.OS), computes
rolling z-scores and a spread, defines regimes, and runs backtests with a
gross-scaling overlay on top of the production BLPX residual signal.
"""
from __future__ import annotations

import copy
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import yfinance as yf

from leadlag.data.cache import load_df_exec_from_local_cache
from leadlag.execution.backtester import BacktestEngine
from leadlag.models.sector_relative_ensemble_blp_enhanced import (
    SectorRelativeEnsembleBLPEnhancedModel,
)
from leadlag.reporting.metrics import calculate_metrics
from leadlag.utils.threading import run_with_timeout

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("vix_regime_overlay")


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _normalize_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Strip timezone and normalize to midnight."""
    dti = pd.to_datetime(idx)
    if dti.tz is not None:
        dti = dti.tz_localize(None)
    return dti.normalize()


def _download_yf_close(ticker: str, start: str, end: str) -> pd.Series:
    """Download adjusted close from yfinance with a timeout guard."""
    def fetch():
        t = yf.Ticker(ticker)
        hist = t.history(start=start, end=end, auto_adjust=False)
        if hist.empty:
            return pd.Series(dtype=float)
        close = hist["Close"].copy()
        close.index = _normalize_index(close.index)
        return close

    return run_with_timeout(fetch, timeout=120, label=f"yfinance {ticker}")


def load_or_download_vix(
    cache_path: Path,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Load VIX cache or download ^VIX and ^NKVI.OS from yfinance."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        logger.info("Loading VIX cache from %s", cache_path)
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    logger.info("Downloading VIX data...")
    us_vix = _download_yf_close("^VIX", start_date, end_date)
    jp_vix = _download_yf_close("^NKVI.OS", start_date, end_date)

    df = pd.DataFrame({"us_vix": us_vix, "jp_vix": jp_vix})
    df = df.sort_index()
    df.to_csv(cache_path)
    logger.info("Saved VIX cache to %s", cache_path)
    return df


def prepare_regime_multipliers(
    df_exec: pd.DataFrame,
    vix_df: pd.DataFrame,
    windows: list[int] = (60, 120),
    spread_thresh: float = 0.8,
    high_vix_z: float = 0.5,
    jp_led_mult: float = 0.5,
    global_stress_mult: float = 0.5,
    normal_mult: float = 1.0,
    us_led_mult: float = 1.0,
) -> dict[str, pd.Series]:
    """Compute VIX z-scores and return dict of overlay multiplier series.

    Keys:
      - "baseline": constant 1.0
      - "discrete_jp_led_cut": step function based on US/JP VIX z and spread
      - "continuous_spread": linear scaling with spread_z
      - "discrete_global_stress": step function with global stress cut
    """
    # Align VIX to df_exec business days
    vix_reindexed = vix_df.reindex(df_exec.index)
    vix_reindexed["us_vix"] = vix_reindexed["us_vix"].ffill().bfill()
    vix_reindexed["jp_vix"] = vix_reindexed["jp_vix"].ffill().bfill()

    # Use log VIX for z-scores to reduce right skew
    us_log = np.log(vix_reindexed["us_vix"])
    jp_log = np.log(vix_reindexed["jp_vix"])

    out: dict[str, pd.Series] = {}
    out["baseline"] = pd.Series(1.0, index=df_exec.index)

    for w in windows:
        suf = f"_w{w}"
        roll = us_log.rolling(window=w, min_periods=max(20, w // 3))
        us_mean = roll.mean()
        us_std = roll.std()
        us_z = (us_log - us_mean) / us_std

        roll_jp = jp_log.rolling(window=w, min_periods=max(20, w // 3))
        jp_mean = roll_jp.mean()
        jp_std = roll_jp.std()
        jp_z = (jp_log - jp_mean) / jp_std

        spread_z = jp_z - us_z

        # Regime classification
        us_high = us_z > high_vix_z
        jp_high = jp_z > high_vix_z
        us_low = us_z <= 0.0
        jp_led = us_low & (spread_z > spread_thresh)
        global_stress = us_high & jp_high

        mult = pd.Series(normal_mult, index=df_exec.index)
        mult.loc[us_high & ~global_stress] = us_led_mult
        mult.loc[jp_led] = jp_led_mult
        mult.loc[global_stress] = global_stress_mult

        out[f"discrete_jp_led{suf}"] = mult

        # Global stress variant: cut both high (no jp-led extra cut)
        mult2 = pd.Series(normal_mult, index=df_exec.index)
        mult2.loc[us_high & ~jp_high] = us_led_mult
        mult2.loc[us_low & jp_high] = jp_led_mult
        mult2.loc[global_stress] = global_stress_mult
        out[f"discrete_global_stress{suf}"] = mult2

        # Continuous scaling: reduce as spread increases
        # mult = 1 - 0.25 * spread_z, clipped
        cont = pd.Series(np.clip(1.0 - 0.25 * spread_z, 0.5, 1.0), index=df_exec.index)
        out[f"continuous_spread{suf}"] = cont

    return out


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class VixOverlayBLPEnhancedModel(SectorRelativeEnsembleBLPEnhancedModel):
    """BLPEnhancedModel that scales build_weights by a per-day VIX multiplier."""

    def __init__(self, config: dict, overlay_mult: pd.Series):
        super().__init__(config)
        self._overlay_mult = overlay_mult
        self._overlay_idx: int | None = None

    def set_overlay_idx(self, idx: int) -> None:
        """Set the index (row position in df_exec) for the next build_weights call."""
        self._overlay_idx = idx

    def build_weights(
        self,
        signal: np.ndarray,
        q: float | None = None,
        Sigma_YY: np.ndarray | None = None,
    ) -> np.ndarray:
        w = super().build_weights(signal, q=q, Sigma_YY=Sigma_YY)
        if self._overlay_idx is not None and 0 <= self._overlay_idx < len(self._overlay_mult):
            mult = float(self._overlay_mult.iloc[self._overlay_idx])
            w = w * mult
            self._overlay_idx += 1
        return w


# ---------------------------------------------------------------------------
# Backtest helpers
# ---------------------------------------------------------------------------

def run_variant_backtest(
    model: SectorRelativeEnsembleBLPEnhancedModel,
    df_exec: pd.DataFrame,
    start_date: str,
    end_date: str,
    slippage_bps: float,
    overnight_alpha_long: float,
    overnight_alpha_short: float,
    n_jobs: int,
) -> dict[str, Any]:
    """Run BacktestEngine and return results + metrics."""
    # Align overlay index if needed
    start_dt = pd.to_datetime(start_date)
    start_idx = max(df_exec.index.searchsorted(start_dt), getattr(model, "corr_window", 60))

    # If overlay model, reset the counter to the first simulated row
    if isinstance(model, VixOverlayBLPEnhancedModel):
        model.set_overlay_idx(start_idx)

    results = BacktestEngine.run_backtest(
        model,
        df_exec=df_exec,
        start_date=start_date,
        end_date=end_date,
        slippage_bps=slippage_bps,
        overnight_alpha_long=overnight_alpha_long,
        overnight_alpha_short=overnight_alpha_short,
        n_jobs=n_jobs,
    )
    metrics = calculate_metrics(results["daily_returns"])
    results["metrics"] = metrics
    return results


def summarize_results(
    name: str,
    results: dict[str, Any],
) -> dict[str, Any]:
    """Build a compact summary dict for the report."""
    m = results["metrics"]
    dr = results["daily_returns"]
    exp = results["daily_gross_exps"]
    turnover = results["daily_turnover"]
    costs = results["daily_costs"]
    return {
        "name": name,
        "AR": m["AR"],
        "RISK": m["RISK"],
        "Sharpe": float(m["Sharpe"]),
        "MDD": float(m["MDD"]),
        "Total Return": m["Total Return"],
        "mean_gross_exp": float(exp.mean()),
        "median_gross_exp": float(exp.median()),
        "mean_turnover": float(turnover.mean()),
        "mean_cost": float(costs.mean()) if len(costs) > 0 else 0.0,
        "n_days": int(len(dr)),
    }


def _save_variant_csvs(name: str, results: dict[str, Any], out_dir: Path) -> None:
    """Persist all daily result series for a variant."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results["daily_returns"].to_csv(
        out_dir / f"daily_returns_{name}.csv", header=["net_return"]
    )
    if "daily_returns_gross" in results:
        results["daily_returns_gross"].to_csv(
            out_dir / f"daily_gross_returns_{name}.csv", header=["gross_return"]
        )
    if "daily_gross_exps" in results:
        results["daily_gross_exps"].to_csv(
            out_dir / f"daily_gross_exposures_{name}.csv", header=["gross_exposure"]
        )
    if "daily_turnover" in results:
        results["daily_turnover"].to_csv(
            out_dir / f"daily_turnover_{name}.csv", header=["turnover"]
        )
    for key in ["daily_costs", "daily_slip_costs", "daily_financing_costs", "daily_borrow_costs", "daily_reverse_costs"]:
        if key in results and results[key] is not None:
            col = key.replace("daily_", "").replace("_cost", "_cost")
            results[key].to_csv(out_dir / f"{key}_{name}.csv", header=[col])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = _make_arg_parser()
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load config (deep copy for safety; no mutating original)
    with open(ROOT / args.config) as f:
        base_cfg = yaml.safe_load(f)

    # Load execution data
    logger.info("Loading df_exec...")
    df_exec = load_df_exec_from_local_cache()

    # Load VIX data
    cache_path = ROOT / "market_data" / "vix_regime_overlay" / "vix_cache.csv"
    vix_end = (pd.Timestamp.now() + pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    vix_df = load_or_download_vix(cache_path, "2010-01-01", vix_end)

    # Prepare overlays
    logger.info("Computing VIX regime multipliers...")
    mult_dict = prepare_regime_multipliers(
        df_exec,
        vix_df,
        windows=[args.window],
        spread_thresh=args.spread_thresh,
        high_vix_z=args.high_vix_z,
        jp_led_mult=args.jp_led_mult,
        global_stress_mult=args.global_stress_mult,
        normal_mult=1.0,
        us_led_mult=1.0,
    )

    # Build base model
    base_model = SectorRelativeEnsembleBLPEnhancedModel(base_cfg)

    # Run baseline
    logger.info("Running baseline backtest...")
    baseline_results = run_variant_backtest(
        base_model,
        df_exec,
        args.start_date,
        args.end_date,
        args.slippage_bps,
        args.overnight_alpha_long,
        args.overnight_alpha_short,
        args.n_jobs,
    )
    _save_variant_csvs("baseline", baseline_results, out_dir)

    summaries = [summarize_results("baseline", baseline_results)]
    variant_results: dict[str, pd.Series] = {"baseline": baseline_results["daily_returns"]}

    # Run variants
    for name, mult in mult_dict.items():
        if name == "baseline":
            continue
        logger.info("Running variant: %s", name)
        cfg = copy.deepcopy(base_cfg)
        model = VixOverlayBLPEnhancedModel(cfg, mult)
        res = run_variant_backtest(
            model,
            df_exec,
            args.start_date,
            args.end_date,
            args.slippage_bps,
            args.overnight_alpha_long,
            args.overnight_alpha_short,
            args.n_jobs,
        )
        _save_variant_csvs(name, res, out_dir)
        variant_results[name] = res["daily_returns"]
        summaries.append(summarize_results(name, res))

    # Save summary table
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summaries, f, indent=2, default=str)

    # Walk-forward: year-by-year Sharpe for each variant
    logger.info("Computing walk-forward year-by-year metrics...")
    wf_records = []
    for year in range(
        int(args.start_date[:4]),
        int(args.end_date[:4]) + 1,
    ):
        yr_start = f"{year}-01-01"
        yr_end = f"{year}-12-31"
        for name, rets in variant_results.items():
            mask = (rets.index >= yr_start) & (rets.index <= yr_end)
            if mask.sum() < 30:
                continue
            sub = rets[mask]
            m = calculate_metrics(sub)
            wf_records.append(
                {
                    "year": year,
                    "variant": name,
                    "sharpe": float(m["Sharpe"]),
                    "mdd": float(m["MDD"]),
                    "ar": float(m["AR"]),
                    "n_days": int(mask.sum()),
                }
            )
    wf_df = pd.DataFrame(wf_records)
    wf_df.to_csv(out_dir / "walkforward_sharpe.csv", index=False)

    # Generate markdown report
    report_path = out_dir / "report.md"
    _write_report(report_path, summary_df, wf_df, args)

    logger.info("Experiment complete. Artifacts in %s", out_dir)
    print("\n=== VIX Regime Overlay Summary ===")
    print(summary_df.to_string(index=False))


def _make_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description="VIX regime overlay experiment")
    parser.add_argument("--config", default="configs/production/production.yaml", help="Config YAML")
    parser.add_argument("--start-date", default="2018-04-01", help="Backtest start")
    parser.add_argument("--end-date", default="2024-12-31", help="Backtest end")
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--overnight-alpha-long", type=float, default=0.75)
    parser.add_argument("--overnight-alpha-short", type=float, default=0.5)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--window", type=int, default=60, help="Rolling VIX z-score window")
    parser.add_argument("--spread-thresh", type=float, default=0.8, help="JP-led spread z threshold")
    parser.add_argument("--high-vix-z", type=float, default=0.5, help="High VIX z-score threshold")
    parser.add_argument("--jp-led-mult", type=float, default=0.5, help="Gross multiplier in JP-led regime")
    parser.add_argument("--global-stress-mult", type=float, default=0.5, help="Gross multiplier in global stress")
    parser.add_argument("--output-dir", default="results/vix_regime_overlay", help="Output directory")
    return parser


def _write_report(path: Path, summary_df: pd.DataFrame, wf_df: pd.DataFrame, args) -> None:
    lines = [
        "# VIX Regime Overlay Experiment",
        "",
        f"- Period: {args.start_date} ~ {args.end_date}",
        f"- VIX z-score window: {args.window} days",
        f"- JP-led spread threshold (z): {args.spread_thresh}",
        f"- High VIX threshold (z): {args.high_vix_z}",
        f"- JP-led gross multiplier: {args.jp_led_mult}",
        f"- Global stress gross multiplier: {args.global_stress_mult}",
        "",
        "## Aggregate Summary",
        "",
    ]
    # Summary table
    header = summary_df.columns.tolist()
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for _, row in summary_df.iterrows():
        vals = []
        for col in header:
            v = row[col]
            if col in ("AR", "RISK", "MDD"):
                vals.append(f"{float(v) * 100:.2f}%")
            elif col in ("Sharpe", "Total Return", "mean_gross_exp", "median_gross_exp", "mean_turnover", "mean_cost"):
                vals.append(f"{float(v):.4f}")
            else:
                vals.append(f"{v}")
        lines.append("| " + " | ".join(vals) + " |")

    lines.extend(["", "## Walk-Forward Year-by-Year Sharpe", ""])
    if not wf_df.empty:
        pivot = wf_df.pivot(index="year", columns="variant", values="sharpe")
        cols = ["year"] + pivot.columns.tolist()
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| --- | " + " | ".join(["---"] * (len(cols) - 1)) + " |")
        for year, row in pivot.iterrows():
            vals = [str(year)]
            for col in pivot.columns:
                vals.append(f"{row[col]:.2f}" if not pd.isna(row[col]) else "")
            lines.append("| " + " | ".join(vals) + " |")
    else:
        lines.append("(no walk-forward data)")
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
