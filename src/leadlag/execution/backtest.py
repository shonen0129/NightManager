"""runner/backtest.py — full backtesting runner.

Provides ``run_production()`` which downloads/loads data, runs the V2
production strategy over the requested date range, and saves performance
artifacts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from leadlag.config.paths import gap_distribution_latest
from leadlag.config.schemas import AppConfig
from leadlag.core.risk import compute_var_es
from leadlag.data.backtest_store import BacktestResultStore
from leadlag.data.cache import load_df_exec_from_local_cache
from leadlag.data.fetcher import download_data
from leadlag.data.preprocessor import preprocess_data
from leadlag.execution.backtester import BacktestEngine
from leadlag.execution.config import load_config_from_yaml
from leadlag.execution.output_ops import build_output_dir, save_summary_files
from leadlag.reporting.metrics import calculate_metrics, generate_report

logger = logging.getLogger(__name__)


def _resolve_config_path(config_path: str | Path | None) -> tuple[Path, Path]:
    """Return project root and resolved config path."""
    project_root = Path(__file__).resolve().parents[3]
    if config_path is None:
        resolved = project_root / "configs" / "production" / "production.yaml"
    else:
        resolved = Path(config_path)
        if not resolved.is_absolute():
            resolved = project_root / resolved
    return project_root, resolved


def _resolve_gap_input_dir(
    gap_input_dir: str | Path | None,
    app_config: AppConfig,
    project_root: Path,
) -> Path | None:
    """Resolve V2 gap input directory, returning None if it does not exist."""
    if gap_input_dir is None:
        gap_input_dir = app_config.gap_distribution_dir or str(gap_distribution_latest())
    gap_dir = Path(gap_input_dir)
    if not gap_dir.is_absolute():
        gap_dir = project_root / gap_dir
    if not gap_dir.exists():
        logger.warning(
            "Gap input dir not found: %s. V2 backtest will fall back to flat positions.",
            gap_dir,
        )
        return None
    return gap_dir


def _load_df_exec(app_config: AppConfig, data_source: str) -> pd.DataFrame:
    """Load or build df_exec according to *data_source*.

    - ``download``: download/refresh raw data and preprocess.
    - ``cache``: load from local decision/etf cache if valid; otherwise fall
      back to ``download`` with a warning.
    """
    residual_cfg = app_config.strategy
    beta_window = residual_cfg.beta_window
    beta_ewma_halflife = residual_cfg.beta_ewma_halflife
    beta_shrinkage = residual_cfg.beta_shrinkage
    beta_winsor_sigma = residual_cfg.beta_winsor_sigma

    if data_source == "cache":
        try:
            return load_df_exec_from_local_cache()
        except Exception as exc:
            logger.warning("Failed to load df_exec from cache (%s); falling back to download", exc)

    raw_data = download_data(beta_window=beta_window)
    return preprocess_data(
        raw_data,
        beta_window=beta_window,
        beta_ewma_halflife=beta_ewma_halflife,
        beta_shrinkage=beta_shrinkage,
        beta_winsor_sigma=beta_winsor_sigma,
    )


def _save_detailed_backtest_results(results: dict[str, Any], output_dir: Path) -> None:
    """Write all backtest time series and weights to CSV.

    Mirrors the output produced by the legacy ``run_production_backtest.py``
    entry point so the CLI ``backtest`` subcommand is a drop-in replacement.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    def _save_series(name: str, series: pd.Series) -> None:
        if name in results and results[name] is not None:
            results[name].to_csv(output_dir / f"daily_{name}.csv", header=[name])

    _save_series("daily_returns", results.get("daily_returns"))
    _save_series("daily_returns_gross", results.get("daily_returns_gross"))
    _save_series("equity_curve", results.get("equity_curve"))
    _save_series("drawdown", results.get("drawdown"))
    _save_series("daily_turnover", results.get("daily_turnover"))
    _save_series("daily_gross_exps", results.get("daily_gross_exps"))
    _save_series("daily_costs", results.get("daily_costs"))
    _save_series("daily_slip_costs", results.get("daily_slip_costs"))
    _save_series("daily_financing_costs", results.get("daily_financing_costs"))
    _save_series("daily_borrow_costs", results.get("daily_borrow_costs"))
    _save_series("daily_reverse_costs", results.get("daily_reverse_costs"))
    _save_series("daily_fallback", results.get("daily_fallback"))

    if "weights" in results and results["weights"] is not None:
        results["weights"].to_csv(output_dir / "daily_weights.csv")


def run_production(
    start_date: str,
    output_root: str,
    run_tag: str | None,
    skip_chart: bool,
    slippage_bps: float | None = None,
    n_jobs: int = 1,
    config_path: str | Path | None = None,
    gap_input_dir: str | Path | None = None,
    data_source: str = "download",
    end_date: str = "latest",
    side_leverage: float | None = None,
    output_level: str = "detailed",
    overlay_model_dir: str | Path | None = None,
    gap_store_path: str | Path | None = None,
) -> Path:
    """Run the full V2 production backtest and save performance artifacts.

    Args:
        start_date: Backtest start date.
        output_root: Directory root where outputs are written.
        run_tag: Optional run tag.
        skip_chart: Skip chart generation if True.
        slippage_bps: Override slippage bps. If None, use YAML default.
        n_jobs: Parallel workers for V2 weight generation.
        config_path: Path to V2 production YAML config.
        gap_input_dir: Directory containing mu_gap/omega_gap .npy files.
        gap_store_path: Optional path to a ``.sqlite`` gap store that
            overrides *gap_input_dir* for matrix loading.
        data_source: ``download`` to fetch/refresh data, ``cache`` to use the
            local fast-mode cache (``load_df_exec_from_local_cache``).
        end_date: Backtest end date; ``latest`` for the last available date.
        side_leverage: Notional side leverage (defaults to config value).
        output_level: ``minimal`` for summary + charts, ``detailed`` for full
            daily time-series CSVs and weights.
        overlay_model_dir: Optional path to an ML order-overlay model.

    Returns:
        Path to the output directory
    """
    if output_level not in ("minimal", "detailed"):
        raise ValueError("output_level must be 'minimal' or 'detailed'")

    project_root, resolved = _resolve_config_path(config_path)
    app_config = load_config_from_yaml(resolved, strict=True)

    output_dir = build_output_dir(output_root, run_tag, run_name="production_backtest")
    output_dir_path = Path(output_dir)

    logger.info("[1/4] Downloading/loading market data (source=%s)...", data_source)
    df_exec = _load_df_exec(app_config, data_source)

    logger.info("[2/4] Running V2 production backtest...")
    resolved_slippage = slippage_bps if slippage_bps is not None else app_config.strategy.slippage_bps
    logger.info(
        "Slippage: %.1f bps one-way (round-trip = 2 x %.1f bps x gross_exposure/day)",
        resolved_slippage,
        resolved_slippage,
    )

    gap_dir = _resolve_gap_input_dir(gap_input_dir, app_config, project_root)

    results = BacktestEngine.run_v2_backtest(
        cfg=app_config,
        gap_input_dir=gap_dir,
        df_exec=df_exec,
        start_date=start_date,
        end_date=end_date,
        slippage_bps=resolved_slippage,
        side_leverage=side_leverage,
        n_jobs=n_jobs,
        overlay_model_dir=overlay_model_dir,
        gap_store_path=gap_store_path,
    )

    valid_returns = results["daily_returns"]
    if "daily_fallback" in results:
        valid_returns = valid_returns[~results["daily_fallback"]]

    metrics = calculate_metrics(valid_returns)
    risk_cfg = app_config.risk
    var_es_result = compute_var_es(
        results["daily_returns"],
        confidence=risk_cfg.var_confidence,
        window=risk_cfg.var_window,
        var_method=risk_cfg.var_method,
    )

    if not skip_chart:
        # Generate chart report using df structured for graphing
        graph_df = pd.DataFrame(
            {"daily_return": results["daily_returns"]}, index=results["daily_returns"].index
        )
        generate_report(graph_df, output_dir)

    logger.info("[3/4] Writing production artifacts...")
    # Wrap results for save_summary_files
    summary_results_df = pd.DataFrame(
        {"daily_return": results["daily_returns"]}, index=results["daily_returns"].index
    )
    save_summary_files(summary_results_df, metrics, app_config.strategy, output_dir)

    if output_level == "detailed":
        _save_detailed_backtest_results(results, output_dir_path)

    # Persist the full results dict to the backtest store (CSV files above
    # remain the user-visible artifacts).  Save both the detailed daily tables
    # (save_run) and the full cached results dict (save_results).
    try:
        store = BacktestResultStore(output_dir_path / "backtest_store.sqlite")
        run_id = store.save_run(results, config=app_config)
        if run_id is not None:
            store.save_results(results, run_id=run_id)
    except Exception as e:
        logger.warning("Failed to save full results to BacktestResultStore: %s", e)

    # Print summary metrics to log
    print("=== Backtest Performance Metrics ===")
    for key, v in metrics.items():
        if key in ["AR", "RISK", "MDD", "Total Return"]:
            logger.info("  %s: %.2f%%", key, v * 100)
        elif key == "Sharpe":
            logger.info("  %s: %.4f", key, v)
        else:
            logger.info("  %s: %.2f", key, v)

    if var_es_result.available:
        logger.info(
            "VaR/ES snapshot (99%%,250d): VaR=%.4f%%, ES=%.4f%%",
            var_es_result.var_loss * 100,
            var_es_result.es_loss * 100,
        )
    else:
        logger.info(
            "VaR/ES snapshot skipped: history=%d < window=%d",
            var_es_result.samples,
            var_es_result.window,
        )

    logger.info("[4/4] Done. Artifacts saved in: %s", output_dir_path)
    return output_dir_path
