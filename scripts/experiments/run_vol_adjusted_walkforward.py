#!/usr/bin/env python3
"""Walk-forward validation for blpx.vol_adjusted_target (false vs true).

Runs full-period V2 backtests on two gap-adjusted distribution directories,
then extracts yearly OOS metrics (net Sharpe, total return, max DD, turnover,
fallback rate).  Computes a Deflated Sharpe correction for the 2-trial comparison.

Usage:
    python3 scripts/experiments/run_vol_adjusted_walkforward.py \
        --gap-false live/pipeline_data/gap_adjusted_distribution/20260731_024303 \
        --gap-true  live/pipeline_data/gap_adjusted_distribution/20260730_091010 \
        --start 2015-01-05 --end latest \
        --output-dir outputs/experiments/vol_adjusted_walkforward
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.cache import load_df_exec_from_local_cache
from leadlag.execution.backtester import BacktestEngine
from leadlag.models.sre import compute_jp_target_returns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

TRADING_DAYS = 245


def _sharpe(s: pd.Series, ann_factor: float = np.sqrt(TRADING_DAYS)) -> float:
    if s.std(ddof=1) < 1e-12:
        return 0.0
    return float(s.mean() / s.std(ddof=1) * ann_factor)


def _max_dd(ec: pd.Series) -> float:
    return float((ec / ec.cummax() - 1.0).min())


def _total_ret(s: pd.Series) -> float:
    return float((1.0 + s).prod() - 1.0)


def _compute_yearly_metrics(
    daily_returns: pd.Series,
    equity_curve: pd.Series,
    turnover: pd.Series,
    fallback: pd.Series,
) -> pd.DataFrame:
    """Return a DataFrame of yearly metrics."""
    rows = []
    for yr, group in daily_returns.groupby(daily_returns.index.year):
        if len(group) < 40:
            continue
        ec = equity_curve.loc[group.index]
        to = turnover.loc[group.index]
        fb = fallback.loc[group.index]
        rows.append(
            {
                "year": int(yr),
                "n_days": len(group),
                "net_total": _total_ret(group) * 100,
                "net_sharpe": _sharpe(group),
                "max_dd": _max_dd(ec) * 100,
                "turnover": float(to.mean()),
                "fallback_rate": float(fb.mean()),
            }
        )
    return pd.DataFrame(rows)


def _deflated_sharpe_params(returns: pd.Series) -> tuple:
    """Return (sharpe, skew, kurt, n) for Deflated Sharpe formula."""
    r = returns.to_numpy()
    sharpe = _sharpe(returns)
    n = len(r)
    if n < 10:
        return sharpe, 0.0, 3.0, n
    skew = float(stats.skew(r, bias=False))
    kurt = float(stats.kurtosis(r, fisher=False, bias=False))
    return sharpe, skew, kurt, n


def _deflated_sharpe(
    sharpe: float,
    skew: float,
    kurt: float,
    n: int,
    v: float,
) -> float:
    """López de Prado (2018) Deflated Sharpe.

    v: number of independent trials (variance across strategies).
    Clamp v to >= 2.0 to avoid the approximation breaking down for
    highly correlated strategies (v close to 1.0).
    """
    if n < 10 or v < 1:
        return np.nan
    v = max(float(v), 2.0)
    gamma_e = 0.5772156649015329
    phi_inv_max = stats.norm.ppf(1.0 - 1.0 / v)
    phi_inv_min = stats.norm.ppf(1.0 - 1.0 / v * np.exp(-1.0))
    e_max_sr = np.sqrt(v) * ((1.0 - gamma_e) * phi_inv_max + gamma_e * phi_inv_min)

    # Standard deviation of the (non-null) Sharpe estimate
    sigma_sr = np.sqrt(
        (1.0 - skew * sharpe + (kurt - 1.0) / 4.0 * sharpe**2) / (n - 1)
    )
    if sigma_sr < 1e-12:
        return np.nan
    dsr = (sharpe - e_max_sr) / sigma_sr
    return float(dsr)


def _estimate_v(
    returns_true: pd.Series,
    returns_false: pd.Series,
    k: int = 2,
) -> float:
    """Effective number of independent trials: V = k / (1 + (k-1) * rho)."""
    common = returns_true.index.intersection(returns_false.index)
    if len(common) < 10:
        return float(k)
    t = returns_true.loc[common].to_numpy()
    f = returns_false.loc[common].to_numpy()
    if t.std(ddof=1) < 1e-12 or f.std(ddof=1) < 1e-12:
        return float(k)
    rho = float(np.corrcoef(t, f)[0, 1])
    if np.isnan(rho):
        return float(k)
    rho = np.clip(rho, -0.99, 0.99)
    return float(k / (1.0 + (k - 1.0) * rho))


def _run_single_backtest(
    cfg: dict,
    gap_dir: Path,
    df_exec: pd.DataFrame,
    start_date: str,
    end_date: str,
    label: str,
    n_jobs: int = -1,
) -> dict:
    logger.info("Running V2 backtest for %s", label)
    logger.info("  gap_dir: %s", gap_dir)
    logger.info("  date range: %s to %s", start_date, end_date)
    t0 = pd.Timestamp.now()
    res = BacktestEngine.run_v2_backtest(
        cfg=cfg,
        gap_input_dir=gap_dir,
        df_exec=df_exec,
        start_date=start_date,
        end_date=end_date,
        n_jobs=n_jobs,
    )
    elapsed = (pd.Timestamp.now() - t0).total_seconds()
    logger.info("  %s backtest completed in %.1fs", label, elapsed)
    return res


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "production" / "production.yaml"))
    p.add_argument("--gap-false", required=True, type=Path)
    p.add_argument("--gap-true", required=True, type=Path)
    p.add_argument("--start", default="2015-01-05")
    p.add_argument("--end", default="latest")
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--output-dir", default=str(ROOT / "outputs" / "experiments" / "vol_adjusted_walkforward"))
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    df_exec = load_df_exec_from_local_cache()

    res_false = _run_single_backtest(cfg, args.gap_false, df_exec, args.start, args.end, "vol_adjusted_target=false", args.n_jobs)
    res_true = _run_single_backtest(cfg, args.gap_true, df_exec, args.start, args.end, "vol_adjusted_target=true", args.n_jobs)

    # Full-period metrics
    def full_metrics(res: dict, label: str) -> dict:
        dr_net = res["daily_returns"]
        dr_gross = res["daily_returns_gross"]
        ec = res["equity_curve"]
        to = res["daily_turnover"]
        fb = res["daily_fallback"]
        sharpe, skew, kurt, n = _deflated_sharpe_params(dr_net)
        gross_sharpe = _sharpe(dr_gross)
        return {
            "label": label,
            "n_days": len(dr_net),
            "net_total": _total_ret(dr_net) * 100,
            "gross_total": _total_ret(dr_gross) * 100,
            "net_sharpe": sharpe,
            "gross_sharpe": gross_sharpe,
            "max_dd": _max_dd(ec) * 100,
            "turnover": float(to.mean()),
            "fallback_rate": float(fb.mean()),
            "skew": skew,
            "kurt": kurt,
        }

    false_full = full_metrics(res_false, "vol_adjusted_target=false")
    true_full = full_metrics(res_true, "vol_adjusted_target=true")

    v = _estimate_v(res_true["daily_returns"], res_false["daily_returns"], k=2)
    false_full["v_independent_trials"] = v
    true_full["v_independent_trials"] = v
    false_full["deflated_sharpe"] = _deflated_sharpe(
        false_full["net_sharpe"], false_full["skew"], false_full["kurt"], false_full["n_days"], v
    )
    true_full["deflated_sharpe"] = _deflated_sharpe(
        true_full["net_sharpe"], true_full["skew"], true_full["kurt"], true_full["n_days"], v
    )

    # Cost breakdown (annualized totals)
    def cost_breakdown(res: dict) -> dict:
        n = len(res["daily_returns"])
        return {
            "slippage_cost_total": float(res["daily_slip_costs"].sum()) * 100,
            "financing_cost_total": float(res["daily_financing_costs"].sum()) * 100,
            "borrow_cost_total": float(res["daily_borrow_costs"].sum()) * 100,
            "reverse_cost_total": float(res["daily_reverse_costs"].sum()) * 100,
            "total_cost": float(res["daily_costs"].sum()) * 100,
            "avg_daily_cost": float(res["daily_costs"].mean()) * 100,
        }

    false_costs = cost_breakdown(res_false)
    true_costs = cost_breakdown(res_true)

    # Paired t-test and bootstrap
    common = res_false["daily_returns"].index.intersection(res_true["daily_returns"].index)
    f_ret = res_false["daily_returns"].loc[common].to_numpy()
    t_ret = res_true["daily_returns"].loc[common].to_numpy()
    delta = f_ret - t_ret
    t_stat, t_p = stats.ttest_rel(f_ret, t_ret)
    win_days = int(np.sum(delta > 0))

    np.random.seed(42)
    boot_sharpe_diff = []
    for _ in range(5000):
        idx = np.random.choice(len(delta), size=len(delta), replace=True)
        s_false = f_ret[idx].mean() / f_ret[idx].std(ddof=1) * np.sqrt(TRADING_DAYS)
        s_true = t_ret[idx].mean() / t_ret[idx].std(ddof=1) * np.sqrt(TRADING_DAYS)
        boot_sharpe_diff.append(float(s_false - s_true))
    boot_sharpe_diff = np.array(boot_sharpe_diff)
    ci_low, ci_high = np.percentile(boot_sharpe_diff, [2.5, 97.5])
    p_delta_gt_0 = float(np.mean(boot_sharpe_diff > 0))

    # Yearly metrics
    false_yearly = _compute_yearly_metrics(
        res_false["daily_returns"], res_false["equity_curve"], res_false["daily_turnover"], res_false["daily_fallback"]
    )
    true_yearly = _compute_yearly_metrics(
        res_true["daily_returns"], res_true["equity_curve"], res_true["daily_turnover"], res_true["daily_fallback"]
    )
    false_yearly["variant"] = "false"
    true_yearly["variant"] = "true"
    yearly = pd.concat([false_yearly, true_yearly], ignore_index=True)
    yearly = yearly[["variant", "year", "n_days", "net_total", "net_sharpe", "max_dd", "turnover", "fallback_rate"]]

    # Build a side-by-side year table
    false_yearly.set_index("year", inplace=True)
    true_yearly.set_index("year", inplace=True)
    side_by_side = pd.DataFrame(
        {
            "year": false_yearly.index,
            "false_sharpe": false_yearly["net_sharpe"],
            "true_sharpe": true_yearly["net_sharpe"],
            "false_total": false_yearly["net_total"],
            "true_total": true_yearly["net_total"],
            "false_mdd": false_yearly["max_dd"],
            "true_mdd": true_yearly["max_dd"],
            "false_turnover": false_yearly["turnover"],
            "true_turnover": true_yearly["turnover"],
            "false_fallback": false_yearly["fallback_rate"],
            "true_fallback": true_yearly["fallback_rate"],
        }
    )
    side_by_side["sharpe_delta"] = side_by_side["false_sharpe"] - side_by_side["true_sharpe"]
    side_by_side["wins"] = side_by_side["sharpe_delta"] > 0

    # Save
    yearly.to_csv(out_dir / "yearly_metrics.csv", index=False)
    side_by_side.to_csv(out_dir / "yearly_side_by_side.csv", index=False)
    with open(out_dir / "full_period_metrics.json", "w") as f:
        json.dump(
            {
                "false": false_full,
                "true": true_full,
                "false_costs": false_costs,
                "true_costs": true_costs,
                "paired_t_test": {
                    "t_stat": float(t_stat),
                    "p_value": float(t_p),
                    "win_days_false": win_days,
                    "total_days": len(common),
                    "win_rate_false": win_days / len(common),
                },
                "bootstrap_sharpe_diff": {
                    "mean": float(boot_sharpe_diff.mean()),
                    "ci_low": float(ci_low),
                    "ci_high": float(ci_high),
                    "p_delta_gt_0": p_delta_gt_0,
                },
            },
            f,
            indent=2,
            default=str,
        )

    # Print summary
    print("\n=== Full-Period Metrics ===")
    print(f"{'Metric':<20} {'vol_adjusted=false':>22} {'vol_adjusted=true':>22}")
    print("-" * 70)
    for k in ["net_total", "gross_total", "net_sharpe", "gross_sharpe", "max_dd", "turnover", "fallback_rate"]:
        print(f"{k:<20} {false_full[k]:>22.4f} {true_full[k]:>22.4f}")
    print(f"{'deflated_sharpe':<20} {false_full['deflated_sharpe']:>22.4f} {true_full['deflated_sharpe']:>22.4f}")
    print(f"{'v_independent_trials':<20} {v:>22.2f}")

    print("\n=== Yearly OOS ===")
    print(side_by_side[["year", "false_sharpe", "true_sharpe", "sharpe_delta", "wins"]].to_string(index=False))
    wins = int(side_by_side["wins"].sum())
    n_years = len(side_by_side)
    print(f"\nfalse wins: {wins}/{n_years} years ({100.0 * wins / n_years:.1f}%)")
    print(f"avg yearly delta (false - true): {side_by_side['sharpe_delta'].mean():.4f}")

    print("\n=== Statistical Tests ===")
    print(f"Paired t-test: t={t_stat:.4f}, p={t_p:.4f}")
    print(f"False win days: {win_days}/{len(common)} ({100.0 * win_days / len(common):.1f}%)")
    print(f"Bootstrap Sharpe diff: mean={boot_sharpe_diff.mean():.4f}, CI=[{ci_low:.4f}, {ci_high:.4f}], P(diff>0)={p_delta_gt_0:.1%}")

    # Report path
    print(f"\nResults saved to: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
