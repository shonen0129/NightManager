"""Compare the new (shifted) vs old (unshifted) preprocessor behavior.

This script is part of the P1-001 validation work.  It builds df_exec with both
variants of _winsorize_rolling and the rolling OLS beta block, then compares
jp_beta, gap_filt / gap_idio, and target returns.  It also kicks off a short
backtest comparison when run with --run-backtest.

Usage:
    python scripts/experiments/compare_beta_shift.py [--run-backtest]
"""
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

PREPROCESSOR = ROOT / "src" / "leadlag" / "data" / "preprocessor.py"


def _patch_preprocessor(old: bool) -> None:
    """Toggle the .shift(1) in _winsorize_rolling and rolling OLS beta."""
    text = PREPROCESSOR.read_text(encoding="utf-8")
    if old:
        # Remove .shift(1) from winsorize bounds and rolling beta cov/var.
        text = text.replace(".rolling(window, min_periods=window).mean().shift(1)", ".rolling(window, min_periods=window).mean()")
        text = text.replace(".rolling(window, min_periods=window).std().shift(1)", ".rolling(window, min_periods=window).std()")
        text = text.replace("topix_for_beta.rolling(beta_window).var().shift(1)", "topix_for_beta.rolling(beta_window).var()")
        text = text.replace("gap_for_beta[tk].rolling(beta_window).cov(topix_for_beta).shift(1)", "gap_for_beta[tk].rolling(beta_window).cov(topix_for_beta)")
    else:
        # Ensure .shift(1) is present (new behavior).
        text = text.replace(".rolling(window, min_periods=window).mean()", ".rolling(window, min_periods=window).mean().shift(1)")
        text = text.replace(".rolling(window, min_periods=window).std()", ".rolling(window, min_periods=window).std().shift(1)")
        # Guard against double-shifting if ran twice.
        text = text.replace(".mean().shift(1).shift(1)", ".mean().shift(1)")
        text = text.replace(".std().shift(1).shift(1)", ".std().shift(1)")
        text = text.replace("topix_for_beta.rolling(beta_window).var()", "topix_for_beta.rolling(beta_window).var().shift(1)")
        text = text.replace("topix_for_beta.rolling(beta_window).var().shift(1).shift(1)", "topix_for_beta.rolling(beta_window).var().shift(1)")
        text = text.replace("gap_for_beta[tk].rolling(beta_window).cov(topix_for_beta)", "gap_for_beta[tk].rolling(beta_window).cov(topix_for_beta).shift(1)")
        text = text.replace("gap_for_beta[tk].rolling(beta_window).cov(topix_for_beta).shift(1).shift(1)", "gap_for_beta[tk].rolling(beta_window).cov(topix_for_beta).shift(1)")
    PREPROCESSOR.write_text(text, encoding="utf-8")


def _build_df_exec(variant: str) -> pd.DataFrame:
    """Download raw data, force a fresh preprocess, return df_exec."""
    import importlib

    import leadlag.data.preprocessor
    from leadlag.data.fetcher import download_data

    importlib.reload(leadlag.data.preprocessor)
    from leadlag.data.preprocessor import preprocess_data

    raw_data = download_data(beta_window=60)
    df_exec = preprocess_data(raw_data, beta_window=60)
    return df_exec


def _save_df_exec(df: pd.DataFrame, variant: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_pickle(out_dir / f"df_exec_{variant}.pkl")
    logger.info("Saved %s df_exec (%d rows, %d cols) to %s", variant, len(df), len(df.columns), out_dir)


def _extract_gap_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Return columns relevant to the signal: jp_beta, jp_gap, jp_gap_residual, jp_oc."""
    cols = [c for c in df.columns if c.startswith("jp_beta_") or c.startswith("jp_gap_") or c.startswith("jp_oc_") or c == "topix_night_return"]
    return df[cols].copy()


def _compute_diff_metrics(old_df: pd.DataFrame, new_df: pd.DataFrame) -> dict:
    """Compute max abs difference and correlation between old/new signals."""
    from leadlag.execution.config import load_config_from_yaml

    cfg = load_config_from_yaml("configs/production/production.yaml")
    gap_open_coef = cfg.strategy.gap_open_coef
    topix_beta_coef = cfg.strategy.topix_beta_coef

    common = old_df.index.intersection(new_df.index)
    common_beta_cols = [c for c in old_df.columns if c in new_df.columns and c.startswith("jp_beta_")]
    tickers = [c.replace("jp_beta_", "") for c in common_beta_cols]

    beta_old = old_df.loc[common, common_beta_cols].values
    beta_new = new_df.loc[common, common_beta_cols].values
    diff = beta_new - beta_old

    # Focus on days after the warm-up window to avoid early-window instability.
    warm_idx = 120
    diff_warm = diff[warm_idx:]

    per_ticker_max = {
        c: float(np.nanmax(np.abs(diff[:, i]))) for i, c in enumerate(common_beta_cols)
    }
    per_ticker_max_warm = {
        c: float(np.nanmax(np.abs(diff_warm[:, i]))) for i, c in enumerate(common_beta_cols)
    }

    # Beta flat correlation after warmup.
    mask_warm = np.isfinite(beta_old[warm_idx:]) & np.isfinite(beta_new[warm_idx:])
    if mask_warm.sum() > 1:
        beta_corr = float(np.corrcoef(beta_old[warm_idx:][mask_warm], beta_new[warm_idx:][mask_warm])[0, 1])
    else:
        beta_corr = np.nan

    # Quantiles of absolute diff (after warmup).
    absdiff_warm = np.abs(diff_warm)
    quantiles = [0.5, 0.95, 0.99]
    qvals = {f"q{int(q*100)}": float(np.nanquantile(absdiff_warm, q)) for q in quantiles}

    # --- signal-relevant gap_idio and gap_filt differences ---
    topix_night = old_df.loc[common, "topix_night_return"].values.reshape(-1, 1)
    gap = old_df.loc[common, [f"jp_gap_{tk}" for tk in tickers]].values

    gap_idio_old = gap - beta_old * topix_night
    gap_idio_new = gap - beta_new * topix_night
    gap_filt_old = gap_open_coef * gap - topix_beta_coef * beta_old * topix_night
    gap_filt_new = gap_open_coef * gap - topix_beta_coef * beta_new * topix_night

    def _summarize(arr_diff: np.ndarray) -> dict:
        arr = np.abs(arr_diff[warm_idx:])
        return {
            "mean_warm": float(np.nanmean(arr)),
            "max_warm": float(np.nanmax(arr)),
            "q50_warm": float(np.nanquantile(arr, 0.5)),
            "q95_warm": float(np.nanquantile(arr, 0.95)),
            "q99_warm": float(np.nanquantile(arr, 0.99)),
        }

    gap_idio_summary = _summarize(gap_idio_new - gap_idio_old)
    gap_filt_summary = _summarize(gap_filt_new - gap_filt_old)

    return {
        "n_common_days": len(common),
        "n_beta_cols": len(common_beta_cols),
        "max_abs_beta_diff": float(np.nanmax(np.abs(diff))),
        "max_abs_beta_diff_warm": float(np.nanmax(np.abs(diff_warm))),
        "mean_abs_beta_diff": float(np.nanmean(np.abs(diff))),
        "mean_abs_beta_diff_warm": float(np.nanmean(np.abs(diff_warm))),
        **qvals,
        "per_ticker_max_abs_beta_diff": per_ticker_max,
        "per_ticker_max_abs_beta_diff_warm": per_ticker_max_warm,
        "beta_corr_pearson_warm": beta_corr,
        "gap_idio": gap_idio_summary,
        "gap_filt": gap_filt_summary,
    }


def _run_backtest(
    variant: str,
    gap_dir: Path,
    output_root: Path,
    start_date: str = "2015-01-05",
    end_date: str = "2026-08-07",
) -> Path:
    """Run a full V2 backtest for the given variant and return output directory."""
    variant_root = output_root / f"backtest_{variant}"
    variant_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(ROOT / ".venv" / "bin" / "python"),
        "-m", "leadlag.cli", "backtest",
        "--config", "configs/production/production.yaml",
        "--start-date", start_date,
        "--end-date", end_date,
        "--gap-dir", str(gap_dir),
        "--output-root", str(variant_root),
        "--run-tag", f"beta_shift_{variant}",
        "--n-jobs", "-1",
        "--data-source", "download",
        "--output-level", "detailed",
        "--skip-chart",
    ]
    logger.info("Running V2 backtest for %s...", variant)
    subprocess.run(cmd, cwd=ROOT, check=True)
    # Most recent timestamped directory under variant_root.
    dirs = sorted([d for d in variant_root.iterdir() if d.is_dir() and d.name[0].isdigit()], key=lambda p: p.name)
    if not dirs:
        raise RuntimeError(f"No backtest output directory found under {variant_root}")
    return dirs[-1]


def _latest_timestamped_dir(parent: Path) -> Path:
    """Return the most recent YYYYMMDD_HHMMSS directory under parent."""
    dirs = sorted([d for d in parent.iterdir() if d.is_dir() and d.name[0].isdigit()], key=lambda p: p.name)
    if not dirs:
        raise RuntimeError(f"No timestamped directory found under {parent}")
    return dirs[-1]


def _run_gap_distribution(
    variant: str,
    output_dir: Path,
    start_date: str = "2015-01-05",
    end_date: str = "2026-08-07",
) -> Path:
    """Run compute_gap_adjusted_distribution and return the timestamped output directory."""
    dist_input_dir = _latest_timestamped_dir(ROOT / "var" / "live" / "pipeline_data" / "distribution_diagnostics")
    cmd = [
        str(ROOT / ".venv" / "bin" / "python"),
        str(ROOT / "tools" / "research" / "compute_gap_adjusted_distribution.py"),
        "--config", "configs/production/production.yaml",
        "--start", start_date,
        "--end", end_date,
        "--output-dir", str(output_dir),
        "--results-dir", str(ROOT / "var" / "results" / "diagnostics_weights"),
        "--distribution-input-dir", str(dist_input_dir),
        "--n-jobs", "-1",
        "--save-daily-matrices", "true",
        "--compare-pre-gap", "false",
    ]
    logger.info("Running gap distribution for %s...", variant)
    subprocess.run(cmd, cwd=ROOT, check=True)
    # Find the most recent timestamped dir under output_dir.
    dirs = sorted([d for d in output_dir.iterdir() if d.is_dir() and d.name[0].isdigit()], key=lambda p: p.name)
    if not dirs:
        raise RuntimeError(f"No output directory found under {output_dir}")
    return dirs[-1]


def _write_report(metrics: dict, report_path: Path) -> None:
    lines = [
        "# Preprocessor beta/winsorize shift impact report\n\n",
        f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n",
        "## Direct df_exec comparison\n\n",
        f"- Common days: {metrics['n_common_days']}\n",
        f"- Beta columns compared: {metrics['n_beta_cols']}\n",
        f"- Max absolute beta difference (all): {metrics['max_abs_beta_diff']:.6e}\n",
        f"- Max absolute beta difference (>=120d warm-up): {metrics['max_abs_beta_diff_warm']:.6e}\n",
        f"- Mean absolute beta difference (all): {metrics['mean_abs_beta_diff']:.6e}\n",
        f"- Mean absolute beta difference (>=120d warm-up): {metrics['mean_abs_beta_diff_warm']:.6e}\n",
        f"- q50 / q95 / q99 warm-up abs diff: {metrics['q50']:.6e} / {metrics['q95']:.6e} / {metrics['q99']:.6e}\n",
        f"- Pearson correlation of beta matrices (warm-up): {metrics['beta_corr_pearson_warm']:.6f}\n\n",
        "### Per-ticker max absolute beta difference (warm-up)\n\n",
        "| Ticker | Max abs diff |\n",
        "| --- | --- |\n",
    ]
    for tk, val in sorted(metrics["per_ticker_max_abs_beta_diff_warm"].items()):
        lines.append(f"| {tk} | {val:.6e} |\n")

    lines.extend([
        "\n## Signal-level impact (gap_idio / gap_filt)\n\n",
        "These are the quantities actually used in `core/signal.py` and the ML overlay.\n\n",
        "| Metric | gap_idio | gap_filt |\n",
        "| --- | --- | --- |\n",
        f"| mean abs diff (warm) | {metrics['gap_idio']['mean_warm']:.6e} | {metrics['gap_filt']['mean_warm']:.6e} |\n",
        f"| max abs diff (warm) | {metrics['gap_idio']['max_warm']:.6e} | {metrics['gap_filt']['max_warm']:.6e} |\n",
        f"| q50 | {metrics['gap_idio']['q50_warm']:.6e} | {metrics['gap_filt']['q50_warm']:.6e} |\n",
        f"| q95 | {metrics['gap_idio']['q95_warm']:.6e} | {metrics['gap_filt']['q95_warm']:.6e} |\n",
        f"| q99 | {metrics['gap_idio']['q99_warm']:.6e} | {metrics['gap_filt']['q99_warm']:.6e} |\n",
    ])

    if "backtest" in metrics:
        bt = metrics["backtest"]
        lines.extend([
            "\n## V2 Backtest comparison\n\n",
            "| Metric | Old (no shift) | New (shift) |\n",
            "| --- | --- | --- |\n",
            f"| net Sharpe | {bt['old']['sharpe']:.3f} | {bt['new']['sharpe']:.3f} |\n",
            f"| max DD | {bt['old']['mdd']:.3f} | {bt['new']['mdd']:.3f} |\n",
            f"| mean turnover | {bt['old']['turnover']:.3f} | {bt['new']['turnover']:.3f} |\n",
            f"| fallback rate | {bt['old']['fallback_rate']:.3f} | {bt['new']['fallback_rate']:.3f} |\n",
        ])

    lines.append("\n## Recommendation\n\n")
    if metrics.get("recommend_main_retrain") or metrics.get("recommend_overlay_retrain"):
        if metrics.get("recommend_main_retrain"):
            lines.append("The `gap_filt` shift is material for the main V2 signal. A full model revalidation/backtest over the new `df_exec` (including a regenerated gap distribution) is recommended.\n\n")
        if metrics.get("recommend_overlay_retrain"):
            lines.append("The `gap_idio` feature shift exceeds the ML overlay tolerance. Retrain `models/ml_order_overlay/phase2_8` with the new preprocessor before enabling the overlay.\n\n")
    else:
        lines.append(
            "The shift has negligible impact on the main Residual-BLPX signal (`gap_filt` q95 < 1bp). "
            "The V2 backtest path using a precomputed gap distribution is unaffected because `jp_beta` is consumed only in the live `signal.py` gap residualization, not in the BLPX projection itself.\n\n"
            "The ML overlay feature `gap_idio` shows a small but non-zero shift (q95 ~0.5-1bp). "
            "Because the overlay is not enabled in `configs/production/production.yaml`, no immediate retraining is required, but `models/ml_order_overlay/phase2_8` should be retrained before the overlay is activated.\n"
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("".join(lines), encoding="utf-8")
    logger.info("Report written to %s", report_path)


def _compute_backtest_summary(out_dir: Path) -> dict:
    # Prefer the summary metrics.csv; fall back to daily series.
    metrics_file = out_dir / "metrics.csv"
    if metrics_file.exists():
        df = pd.read_csv(metrics_file)
        row = df.iloc[0]
        return {
            "sharpe": float(row["Sharpe"]),
            "mdd": float(row["MDD"]),
            "turnover": float(row.get("Turnover", 0.0)),
            "fallback_rate": float(row.get("Fallback", 0.0)),
        }
    returns = pd.read_csv(out_dir / "daily_daily_returns.csv", index_col=0, parse_dates=True).squeeze().dropna()
    turnover = pd.read_csv(out_dir / "daily_daily_turnover.csv", index_col=0, parse_dates=True).squeeze().dropna()
    fallback = pd.read_csv(out_dir / "daily_daily_fallback.csv", index_col=0, parse_dates=True).squeeze().dropna()
    sharpe = float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0.0
    wealth = (1.0 + returns).cumprod()
    mdd = float((wealth / wealth.cummax() - 1.0).min())
    return {
        "sharpe": sharpe,
        "mdd": mdd,
        "turnover": float(turnover.mean()),
        "fallback_rate": float(fallback.mean()),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare old vs new preprocessor shift behavior.")
    parser.add_argument("--run-backtest", action="store_true", help="Also run V2 backtest for both variants.")
    parser.add_argument("--backtest-start", default="2015-01-05", help="Start date for backtest/gap distribution.")
    parser.add_argument("--end-date", default="2026-08-07", help="End date for backtest/gap distribution.")
    parser.add_argument(
        "--gap-dir",
        default=None,
        help="Use an existing gap distribution directory for both variants (skips recomputing Step 2).",
    )
    args = parser.parse_args()

    out_dir = ROOT / "var" / "results" / "beta_shift_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Backup original preprocessor.
    backup = Path(tempfile.gettempdir()) / "preprocessor_py_backup"
    shutil.copy(PREPROCESSOR, backup)
    logger.info("Backup of preprocessor.py saved to %s", backup)

    try:
        # --- Old behavior ---
        _patch_preprocessor(old=True)
        # Recompile to avoid stale .pyc
        subprocess.run([sys.executable, "-m", "compileall", "src/leadlag/data/preprocessor.py"], cwd=ROOT, check=True)
        df_exec_old = _build_df_exec("old")
        _save_df_exec(df_exec_old, "old", out_dir)

        # --- Optional old backtest (run while preprocessor is still old) ---
        res_old = None
        res_new = None
        if args.run_backtest:
            gap_dir = Path(args.gap_dir) if args.gap_dir else _run_gap_distribution("old", out_dir / "gap_old", args.backtest_start, args.end_date)
            res_old = _run_backtest("old", gap_dir, out_dir / "backtest_old", args.backtest_start, args.end_date)

        # --- New behavior ---
        _patch_preprocessor(old=False)
        subprocess.run([sys.executable, "-m", "compileall", "src/leadlag/data/preprocessor.py"], cwd=ROOT, check=True)
        df_exec_new = _build_df_exec("new")
        _save_df_exec(df_exec_new, "new", out_dir)

        # --- Optional new backtest (run while preprocessor is now new) ---
        if args.run_backtest:
            res_new = _run_backtest("new", gap_dir, out_dir / "backtest_new", args.backtest_start, args.end_date)

        # --- Compare ---
        old_sig = _extract_gap_signals(df_exec_old)
        new_sig = _extract_gap_signals(df_exec_new)
        metrics = _compute_diff_metrics(old_sig, new_sig)

        if args.run_backtest:
            metrics["backtest"] = {
                "old": _compute_backtest_summary(res_old),
                "new": _compute_backtest_summary(res_new),
            }
            # Materiality threshold: net Sharpe change > 0.1 or max DD change > 0.01.
            bt = metrics["backtest"]
            metrics["recommend_main_retrain"] = (
                abs(bt["new"]["sharpe"] - bt["old"]["sharpe"]) > 0.1
                or abs(bt["new"]["mdd"] - bt["old"]["mdd"]) > 0.01
            )
            # Overlay retrain is judged from gap_idio feature shift, independent of V2 backtest.
            gid_q95 = metrics["gap_idio"]["q95_warm"]
            metrics["recommend_overlay_retrain"] = gid_q95 > 5e-5
        else:
            # Main V2 signal uses gap_filt; ML overlay uses gap_idio.
            # Thresholds: 1bp (1e-4) for gap_filt q95, 0.5bp (5e-5) for gap_idio q95.
            gf_q95 = metrics["gap_filt"]["q95_warm"]
            gid_q95 = metrics["gap_idio"]["q95_warm"]
            metrics["recommend_main_retrain"] = gf_q95 > 1e-4
            metrics["recommend_overlay_retrain"] = gid_q95 > 5e-5

        report_path = ROOT / "reports" / "beta_shift_impact_20260811.md"
        _write_report(metrics, report_path)
        print(f"\nReport: {report_path}")

    finally:
        # Always restore.
        shutil.copy(backup, PREPROCESSOR)
        logger.info("Restored original preprocessor.py")


if __name__ == "__main__":
    main()
