#!/usr/bin/env python3
"""Paired full V2 backtest for a revised qualitative w6 sensitivity prior."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import sqlite3
import sys
import time
import traceback
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.config.frozen import safe_config_copy  # noqa: E402
from leadlag.data import adr_features as adr_data  # noqa: E402
from leadlag.data import horizon_returns  # noqa: E402
from leadlag.data import macro as macro_data  # noqa: E402
from leadlag.data.intraday_inputs import (  # noqa: E402
    build_open_910_returns,
    compute_jp_target_returns,
)
from leadlag.data.market_data_cache import load_df_exec_from_local_cache  # noqa: E402
from leadlag.data.pit_lake import PITDataLake  # noqa: E402
from leadlag.data.rank_reversal import load_rank_reversal_frame  # noqa: E402
from leadlag.data.tickers import JP_TICKERS, SENSITIVITY_LABELS, US_TICKERS  # noqa: E402
from leadlag.domain.distribution import (  # noqa: E402
    DistributionReason,
    DistributionResult,
    DistributionStatus,
)
from leadlag.domain.inputs import HistoricalInputs  # noqa: E402
from leadlag.execution.backtester import BacktestEngine  # noqa: E402
from leadlag.execution.config import load_config_from_yaml  # noqa: E402
from leadlag.experiment_registry import (  # noqa: E402
    Decision,
    ExperimentRecord,
    ExperimentRegistry,
    compute_deflated_sharpe,
)
from leadlag.features import fractional_diff as frac_diff_module  # noqa: E402
from leadlag.models.production_v2 import ProductionV2Model  # noqa: E402
from leadlag.models.v2 import gap_io  # noqa: E402
from leadlag.models.v2.distribution_source import (  # noqa: E402
    FileCacheDistributionSource,
    OnDemandDistributionSource,
)
from leadlag.models.v2.pit import load_pit_ir_history  # noqa: E402
from leadlag.reporting.metrics import (  # noqa: E402
    MetricsSpec,
    calculate_metrics,
    compute_drawdown_series,
)
from leadlag.runner.model_factory import (  # noqa: E402
    build_blpx_model,
    model_config_fingerprint,
    resolve_overlay_settings,
)
from leadlag.utils.dataframe_fingerprint import dataframe_fingerprint  # noqa: E402

CONFIG_PATH = ROOT / "configs/research/sensitivity_rate_prior_v2_20260924.yaml"
REPORT_DIR = ROOT / "reports/20260924_sensitivity_rate_prior_v2"
RESULTS_DIR = ROOT / "var/results/20260924_sensitivity_rate_prior_v2"
REGISTRY_PATH = ROOT / "var/experiments/registry.jsonl"
BASELINE_NAME = "current_w6"
PRIMARY_NAME = "rate_prior"
VARIANT_NAMES = (PRIMARY_NAME, "rate_prior_attenuated", "rate_prior_amplified")
TRADING_DAYS = 245
GRID = (0.0, 0.3, 0.6, 1.0)
LOG = logging.getLogger("sensitivity_rate_prior_v2")
CURRENT_HORIZON = 1


def _load_config() -> dict[str, Any]:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("Research config must be a mapping")
    return raw


def _variant_w6_maps(config: dict[str, Any]) -> dict[str, dict[str, float]]:
    candidate = {str(k): float(v) for k, v in config["w6_candidate"].items()}
    tickers = list(US_TICKERS) + list(JP_TICKERS)
    if set(candidate) != set(tickers):
        missing = sorted(set(tickers) - set(candidate))
        extra = sorted(set(candidate) - set(tickers))
        raise ValueError(f"w6 candidate ticker mismatch; missing={missing}, extra={extra}")
    allowed = set(float(x) for x in config["sensitivity_diagnostics"]["grid"])
    invalid = {key: value for key, value in candidate.items() if value not in allowed | {-x for x in allowed}}
    if invalid:
        raise ValueError(f"Candidate has values outside the declared grid: {invalid}")

    def move(value: float, *, away: bool) -> float:
        sign = 1.0 if value > 0 else -1.0 if value < 0 else 0.0
        if sign == 0.0:
            return 0.0
        magnitudes = list(GRID)
        position = magnitudes.index(abs(value))
        if away:
            position = min(len(magnitudes) - 1, position + 1)
        else:
            position = max(0, position - 1)
        return sign * magnitudes[position]

    return {
        PRIMARY_NAME: candidate,
        "rate_prior_attenuated": {tk: move(v, away=False) for tk, v in candidate.items()},
        "rate_prior_amplified": {tk: move(v, away=True) for tk, v in candidate.items()},
    }


def _load_owned_inputs(
    app_config: Any,
    start_date: str,
    end_date: str,
) -> tuple[pd.DataFrame, pd.DatetimeIndex, HistoricalInputs, dict[str, Any]]:
    df_full = load_df_exec_from_local_cache(max_stale_bdays=None)
    frame = df_full.loc[
        (df_full.index >= pd.Timestamp("2010-01-01"))
        & (df_full.index <= pd.Timestamp(end_date))
    ].copy()
    if frame.empty or frame.index.max() < pd.Timestamp(end_date):
        raise ValueError(f"df_exec does not reach fixed evaluation end date {end_date}")
    if "is_provisional" in frame.columns and bool(frame["is_provisional"].iloc[-1]):
        raise ValueError("Fixed evaluation end row is provisional")

    lake = PITDataLake(frame)
    sim_dates, start_idx, end_idx = BacktestEngine._resolve_sim_dates(
        frame, start_date, end_date, 0
    )
    eval_dates = pd.DatetimeIndex(sim_dates[start_idx : end_idx + 1])
    if len(eval_dates) < 100 or eval_dates.max() > pd.Timestamp(end_date):
        raise ValueError(f"Invalid fixed evaluation dates: n={len(eval_dates)}")

    open_910 = build_open_910_returns(frame, JP_TICKERS)
    macro_prices = None
    run_config = app_config.v2
    if run_config.macro_kappa_enabled or run_config.macro_direction_enabled:
        macro_prices = macro_data.load_macro_prices(
            start=frame.index.min().strftime("%Y-%m-%d"),
            end=(frame.index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            period="max",
        )
        macro_prices = macro_prices.loc[macro_prices.index <= pd.Timestamp(end_date)].copy()
        if macro_prices.empty or macro_prices.index.max() > pd.Timestamp(end_date):
            raise ValueError("Macro history is empty or extends beyond the fixed evaluation end")
        if not set(macro_data.MACRO_NAMES).issubset(macro_prices.columns):
            raise ValueError("Macro history does not contain all configured factors")

    overlay_enabled, overlay_path = resolve_overlay_settings(app_config)
    if overlay_enabled and (overlay_path is None or not overlay_path.exists()):
        raise FileNotFoundError(f"Configured production overlay is missing: {overlay_path}")
    adr_features = adr_data.load_adr_features() if overlay_enabled else None

    gap_path = Path(run_config.gap_input_dir)
    if not gap_path.is_absolute():
        gap_path = ROOT / gap_path
    if not gap_path.exists():
        raise FileNotFoundError(f"Configured gap store is missing: {gap_path}")

    # The canonical loader hashes the diagnostics file on each call to guard
    # against in-place rewrites. Read one strictly-prior superset at the last
    # evaluation date, then take date-strict prefixes from this immutable run
    # snapshot. This is equivalent to calling the loader for every date while
    # avoiding thousands of repeated full-file reads.
    latest_history, _alerts, latest_history_dates = load_pit_ir_history(
        gap_path, eval_dates[-1].strftime("%Y-%m-%d")
    )
    pit_ir_history: dict[str, np.ndarray] = {}
    pit_history_trade_dates: dict[str, np.ndarray] = {}
    for dt in eval_dates:
        date_key = dt.strftime("%Y-%m-%d")
        prior_mask = latest_history_dates < np.datetime64(dt)
        pit_ir_history[date_key] = latest_history[prior_mask].copy()
        pit_history_trade_dates[date_key] = latest_history_dates[prior_mask].copy()

    rank_reversal = None
    cached_rank_dates: set[str] | None = None
    cached_rank_rows = 0
    cached_dates = 0
    cache_rows = 0
    uri = f"file:{gap_path.resolve().as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True, timeout=5.0) as conn:
        cached_dates = int(
            conn.execute(
                "SELECT count(DISTINCT trade_date) FROM gap_matrices "
                "WHERE matrix_type='mu' AND trade_date BETWEEN ? AND ?",
                (start_date, end_date),
            ).fetchone()[0]
        )
        cache_rows = int(
            conn.execute(
                "SELECT count(*) FROM gap_matrices WHERE trade_date BETWEEN ? AND ?",
                (start_date, end_date),
            ).fetchone()[0]
        )
        rank_rows = conn.execute(
            "SELECT trade_date FROM gap_matrices WHERE matrix_type='rank_reversal' "
            "AND trade_date BETWEEN ? AND ?",
            (start_date, end_date),
        ).fetchall()
        cached_rank_dates = {str(row[0]) for row in rank_rows}
        cached_rank_rows = len(cached_rank_dates)
    if run_config.cs_overlay_enabled:
        rank_eval_dates = (
            [dt for dt in eval_dates if dt.strftime("%Y-%m-%d") in cached_rank_dates]
            if cached_rank_dates is not None
            else eval_dates
        )
        rank_reversal = load_rank_reversal_frame(
            gap_path,
            rank_eval_dates,
            file_pattern=run_config.cs_rank_reversal_file_pattern,
        )

    observed_by_date = {
        dt.strftime("%Y-%m-%d"): {
            "open_910_returns": f"{dt.date()} 09:10",
            "macro_prices": f"{dt.date()} 09:00",
            "adr_features": f"{dt.date()} 09:00",
            "rank_reversal_signals": f"{dt.date()} 09:00",
            "pit_ir_history": f"{dt.date()} 09:10",
        }
        for dt in eval_dates
    }
    historical = HistoricalInputs(
        frame,
        source="sensitivity_rate_prior_v2_backtest",
        open_910_returns=open_910,
        macro_prices=macro_prices,
        adr_features_frame=adr_features,
        pit_ir_history=pit_ir_history,
        pit_history_trade_dates=pit_history_trade_dates,
        rank_reversal_signals=rank_reversal,
        observed_at_by_date=observed_by_date,
    )
    if dataframe_fingerprint(lake.df_exec) != dataframe_fingerprint(historical.to_frame()):
        raise ValueError("Run-owned HistoricalInputs failed the df_exec identity check")

    finite_open_910 = np.isfinite(open_910.loc[eval_dates].to_numpy(dtype=float))
    input_manifest = {
        "df_exec_rows": int(len(frame)),
        "df_exec_start": frame.index.min().strftime("%Y-%m-%d"),
        "df_exec_end": frame.index.max().strftime("%Y-%m-%d"),
        "df_exec_fingerprint": dataframe_fingerprint(frame),
        "open_910_fingerprint": dataframe_fingerprint(open_910),
        "open_910_finite_cells_in_evaluation": int(finite_open_910.sum()),
        "open_910_total_cells_in_evaluation": int(finite_open_910.size),
        "open_910_dates_with_any_finite_price": int(finite_open_910.any(axis=1).sum()),
        "open_910_dates_complete_for_17_etfs": int(finite_open_910.all(axis=1).sum()),
        "macro_rows": int(0 if macro_prices is None else len(macro_prices)),
        "macro_start": None if macro_prices is None else macro_prices.index.min().strftime("%Y-%m-%d"),
        "macro_end": None if macro_prices is None else macro_prices.index.max().strftime("%Y-%m-%d"),
        "macro_fingerprint": None if macro_prices is None else dataframe_fingerprint(macro_prices),
        "adr_rows": int(0 if adr_features is None else len(adr_features)),
        "rank_reversal_rows": int(0 if rank_reversal is None else len(rank_reversal)),
        "pit_dates": int(len(pit_ir_history)),
        "gap_store_cached_mu_dates_in_evaluation": cached_dates,
        "gap_store_rows_in_evaluation": cache_rows,
        "gap_store_rank_reversal_dates_in_evaluation": cached_rank_rows,
        "gap_store_path": str(gap_path),
        "overlay_enabled": bool(overlay_enabled),
        "overlay_path": None if overlay_path is None else str(overlay_path),
        "overlay_path_exists": bool(overlay_path and overlay_path.exists()),
        "simulation_dates": int(len(eval_dates)),
        "simulation_start": eval_dates.min().strftime("%Y-%m-%d"),
        "simulation_end": eval_dates.max().strftime("%Y-%m-%d"),
    }
    return frame, eval_dates, historical, input_manifest


def _build_transform_cache(
    frame: pd.DataFrame,
    app_config: Any,
    eval_dates: pd.DatetimeIndex,
) -> dict[str, Any]:
    """Precompute causal rolling transforms once, retaining V2 point-in-time values.

    Cumulative returns and fractional differences are row-causal: a value at
    date t depends only on rows through t. Slicing a full-history transform at
    t therefore matches recomputing it on the as-of frame, after restoring the
    current-row close-label mask.
    """
    cumulative_fn = horizon_returns.compute_cumulative_returns
    horizons = sorted({1, *(int(h) for h in app_config.v2.mh_horizons)})
    cumulative_by_horizon: dict[int, pd.DataFrame] = {}
    source_us_by_horizon: dict[int, pd.DataFrame] = {}
    fractional_by_horizon: dict[int, pd.DataFrame] = {}
    blpx_cfg = app_config.v2.blpx
    frac_enabled = bool(blpx_cfg.frac_diff_enabled and blpx_cfg.frac_diff_d > 0.0)
    us_columns = [f"us_cc_{ticker}" for ticker in US_TICKERS]
    for horizon in horizons:
        transformed = (
            frame.copy(deep=True)
            if horizon == 1
            else cumulative_fn(frame, horizon=horizon)
        )
        cumulative_by_horizon[horizon] = transformed
        us_frame = transformed[us_columns].copy(deep=True)
        source_us_by_horizon[horizon] = us_frame
        if frac_enabled:
            fractional_by_horizon[horizon] = frac_diff_module.fractional_diff_df(
                us_frame,
                d=float(blpx_cfg.frac_diff_d),
                threshold=float(blpx_cfg.frac_diff_threshold),
                window=int(blpx_cfg.frac_diff_window),
                normalize=blpx_cfg.frac_diff_normalize,
            )

    # Compare the cached slice with the canonical rolling functions at the
    # first, middle, and final evaluation dates for every configured horizon.
    # The final-row mask mirrors HistoricalInputs.calculation_frame(09:10).
    check_positions = sorted({0, len(eval_dates) // 2, len(eval_dates) - 1})
    close_label_prefixes = ("jp_oc_", "jp_cc_", "topix_oc", "topix_cc_trade", "target_", "y_jp_")
    equivalence_checks = 0
    for position in check_positions:
        date = eval_dates[position]
        prefix = frame.loc[:date].copy(deep=True)
        for column in prefix.columns:
            if str(column).startswith(close_label_prefixes):
                prefix.loc[date, column] = np.nan
        for horizon in horizons:
            canonical = cumulative_fn(prefix, horizon=horizon)
            cached = cumulative_by_horizon[horizon].loc[prefix.index].copy(deep=True)
            missing_current = prefix.iloc[-1].isna()
            cached.loc[date, missing_current[missing_current].index] = np.nan
            try:
                pd.testing.assert_frame_equal(
                    canonical,
                    cached,
                    check_exact=False,
                    check_dtype=True,
                    rtol=0.0,
                    atol=1e-14,
                )
            except AssertionError:
                raise ValueError(f"Cumulative-transform cache differs at h={horizon}, date={date.date()}")
            equivalence_checks += 1
            if frac_enabled:
                canonical_frac = frac_diff_module.fractional_diff_df(
                    canonical[us_columns],
                    d=float(blpx_cfg.frac_diff_d),
                    threshold=float(blpx_cfg.frac_diff_threshold),
                    window=int(blpx_cfg.frac_diff_window),
                    normalize=blpx_cfg.frac_diff_normalize,
                )
                cached_frac = fractional_by_horizon[horizon].loc[prefix.index]
                if not np.allclose(
                    canonical_frac.to_numpy(dtype=float),
                    cached_frac.to_numpy(dtype=float),
                    rtol=0.0,
                    atol=1e-14,
                    equal_nan=True,
                ):
                    raise ValueError(f"Fractional-difference cache differs at h={horizon}, date={date.date()}")
                equivalence_checks += 1
    return {
        "horizons": tuple(horizons),
        "cumulative_by_horizon": cumulative_by_horizon,
        "source_us_by_horizon": source_us_by_horizon,
        "fractional_by_horizon": fractional_by_horizon,
        "frac_enabled": frac_enabled,
        "frac_parameters": {
            "d": float(blpx_cfg.frac_diff_d),
            "threshold": float(blpx_cfg.frac_diff_threshold),
            "window": int(blpx_cfg.frac_diff_window),
            "normalize": blpx_cfg.frac_diff_normalize,
        },
        "equivalence_checks": equivalence_checks,
        "df_exec_fingerprint": dataframe_fingerprint(frame),
    }


def _install_measurement_hooks(
    transform_cache: dict[str, Any], variant_name: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bypass cached μ/Ω and retain resolution/audit counters for one variant."""
    counters: dict[str, Any] = {
        "cache_bypass_by_horizon": Counter(),
        "on_demand_by_horizon_status": Counter(),
        "leakage_status": Counter(),
        "numerical_status": Counter(),
        "decisions": 0,
        "variant": variant_name,
        "transform_cache_hits": Counter(),
    }
    originals = {
        "cache_resolve": FileCacheDistributionSource.resolve,
        "ondemand_resolve": OnDemandDistributionSource.resolve,
        "decide": ProductionV2Model.decide,
        "cumulative_returns": gap_io.compute_cumulative_returns,
        "fractional_diff_df": frac_diff_module.fractional_diff_df,
    }

    def cached_cumulative_returns(
        df_exec: pd.DataFrame,
        horizon: int,
        *,
        method: str = "cumprod",
    ) -> pd.DataFrame:
        horizon = int(horizon)
        if horizon == 1:
            return originals["cumulative_returns"](df_exec, horizon, method=method)
        if method != "cumprod" or horizon not in transform_cache["cumulative_by_horizon"]:
            return originals["cumulative_returns"](df_exec, horizon, method=method)
        full = transform_cache["cumulative_by_horizon"][horizon]
        if not df_exec.index.isin(full.index).all() or tuple(df_exec.columns) != tuple(full.columns):
            raise ValueError("As-of frame does not match the frozen transform-cache schema")
        result = full.loc[df_exec.index].copy(deep=True)
        if not df_exec.empty:
            date = df_exec.index[-1]
            missing_columns = df_exec.iloc[-1].index[df_exec.iloc[-1].isna()]
            if len(missing_columns):
                result.loc[date, missing_columns] = np.nan
        counters["transform_cache_hits"][f"cumulative_h{horizon}"] += 1
        return result

    def cached_fractional_diff_df(
        df: pd.DataFrame,
        d: float = 0.5,
        threshold: float = 1e-5,
        window: int = 100,
        normalize: str | None = None,
    ) -> pd.DataFrame:
        horizon = int(CURRENT_HORIZON)
        parameters = transform_cache["frac_parameters"]
        if (
            not transform_cache["frac_enabled"]
            or horizon not in transform_cache["fractional_by_horizon"]
            or float(d) != parameters["d"]
            or float(threshold) != parameters["threshold"]
            or int(window) != parameters["window"]
            or normalize != parameters["normalize"]
            or tuple(df.columns) != tuple(transform_cache["fractional_by_horizon"][horizon].columns)
            or not df.index.isin(transform_cache["fractional_by_horizon"][horizon].index).all()
        ):
            return originals["fractional_diff_df"](
                df, d=d, threshold=threshold, window=window, normalize=normalize
            )
        expected_source = transform_cache["source_us_by_horizon"][horizon].loc[df.index]
        if not np.allclose(
            df.to_numpy(dtype=float),
            expected_source.to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-14,
            equal_nan=True,
        ):
            raise ValueError(f"Fractional-difference cache source differs at h={horizon}")
        counters["transform_cache_hits"][f"fractional_h{horizon}"] += 1
        return transform_cache["fractional_by_horizon"][horizon].loc[df.index].copy(deep=True)

    def cache_miss(self: Any, trade_date: str, df_exec: Any, current_prices: Any, *, horizon: int = 1, **kwargs: Any) -> DistributionResult:
        counters["cache_bypass_by_horizon"][int(horizon)] += 1
        return DistributionResult(
            source="file_cache",
            alerts=["Research comparison recomputes the sensitivity-dependent distribution on-demand."],
            status=DistributionStatus.UNAVAILABLE,
            reason=DistributionReason.CACHE_MISSING,
            horizon=horizon,
        )

    def count_ondemand(self: Any, trade_date: str, df_exec: Any, current_prices: Any, *, horizon: int = 1, **kwargs: Any) -> DistributionResult:
        global CURRENT_HORIZON
        previous_horizon = CURRENT_HORIZON
        CURRENT_HORIZON = int(horizon)
        try:
            result = originals["ondemand_resolve"](
                self, trade_date, df_exec, current_prices, horizon=horizon, **kwargs
            )
        finally:
            CURRENT_HORIZON = previous_horizon
        if result.mu_gap is not None and result.Omega_gap is not None:
            counters[f"distribution_finite_h{int(horizon)}"] = {
                "mu_finite": int(np.isfinite(result.mu_gap).sum()),
                "mu_total": int(np.asarray(result.mu_gap).size),
                "omega_finite": int(np.isfinite(result.Omega_gap).sum()),
                "omega_total": int(np.asarray(result.Omega_gap).size),
            }
        counters["on_demand_by_horizon_status"][(int(horizon), str(result.status))] += 1
        return result

    def count_decide(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            decision = originals["decide"](self, *args, **kwargs)
        except Exception as exc:
            counters["decide_exception"] = f"{type(exc).__name__}: {exc}"
            if counters.get("variant") == "preflight_current_w6":
                traceback.print_exc()
            raise
        counters["decisions"] += 1
        counters["leakage_status"][str((decision.leakage or {}).get("status", "missing"))] += 1
        counters["numerical_status"][str((decision.numerical or {}).get("status", "missing"))] += 1
        if counters["decisions"] == 1 or counters["decisions"] % 50 == 0:
            print(
                f"progress {counters.get('variant', 'variant')}: "
                f"{counters['decisions']} decisions, "
                f"through {decision.summary.get('trade_date', 'unknown date')}",
                flush=True,
            )
        return decision

    gap_io.compute_cumulative_returns = cached_cumulative_returns  # type: ignore[method-assign]
    frac_diff_module.fractional_diff_df = cached_fractional_diff_df  # type: ignore[method-assign]
    FileCacheDistributionSource.resolve = cache_miss  # type: ignore[method-assign]
    OnDemandDistributionSource.resolve = count_ondemand  # type: ignore[method-assign]
    ProductionV2Model.decide = count_decide  # type: ignore[method-assign]
    return counters, originals


def _restore_measurement_hooks(originals: dict[str, Any]) -> None:
    FileCacheDistributionSource.resolve = originals["cache_resolve"]  # type: ignore[method-assign]
    OnDemandDistributionSource.resolve = originals["ondemand_resolve"]  # type: ignore[method-assign]
    ProductionV2Model.decide = originals["decide"]  # type: ignore[method-assign]
    gap_io.compute_cumulative_returns = originals["cumulative_returns"]  # type: ignore[method-assign]
    frac_diff_module.fractional_diff_df = originals["fractional_diff_df"]  # type: ignore[method-assign]


def _install_w6(values: dict[str, float]) -> dict[str, dict[str, float]]:
    previous = copy.deepcopy(SENSITIVITY_LABELS)
    for ticker, value in values.items():
        SENSITIVITY_LABELS[ticker]["w6"] = float(value)
    return previous


def _restore_w6(previous: dict[str, dict[str, float]]) -> None:
    for ticker, row in previous.items():
        SENSITIVITY_LABELS[ticker].clear()
        SENSITIVITY_LABELS[ticker].update(row)


def _baseline_prior_diagnostic(
    app_config: Any,
    historical: HistoricalInputs,
    eval_date: pd.Timestamp,
) -> dict[str, Any]:
    """Inspect the fixed 2010–2014 prior used by the first production decision."""
    as_of = pd.Timestamp(eval_date) + pd.Timedelta(hours=9, minutes=10)
    view = historical.calculation_frame(as_of)
    target = compute_jp_target_returns(
        view,
        JP_TICKERS,
        horizon=1,
        open_910_returns=historical.open_910_returns,
        allow_implicit_io=False,
        required_index=[eval_date],
    )
    blpx = build_blpx_model(app_config)
    inputs = blpx._prepare_common_inputs(
        view,
        horizon=1,
        y_jp_target=target,
        open_910_returns=historical.open_910_returns,
        allow_implicit_io=False,
    )
    dates = pd.DatetimeIndex(view.index)
    baseline_mask = (dates >= pd.Timestamp("2010-01-01")) & (dates <= pd.Timestamp("2014-12-31"))
    n_us = len(US_TICKERS)
    n_jp = len(JP_TICKERS)

    def matrix_stats(name: str) -> dict[str, Any]:
        matrix = np.asarray(inputs[name], dtype=float)
        finite = np.isfinite(matrix)
        return {
            "shape": list(matrix.shape),
            "finite": int(finite.sum()),
            "total": int(finite.size),
            "nonfinite": int((~finite).sum()),
            "us_us_finite": int(finite[:n_us, :n_us].sum()),
            "us_us_total": int(n_us * n_us),
            "us_jp_finite": int(finite[:n_us, n_us : n_us + n_jp].sum()),
            "us_jp_total": int(n_us * n_jp),
            "jp_jp_finite": int(finite[n_us : n_us + n_jp, n_us : n_us + n_jp].sum()),
            "jp_jp_total": int(n_jp * n_jp),
        }

    residuals = np.asarray(inputs["jp_res_returns_p3"], dtype=float)[baseline_mask]
    raw = np.asarray(inputs["all_returns_raw"], dtype=float)[baseline_mask]
    return {
        "decision_date": pd.Timestamp(eval_date).strftime("%Y-%m-%d"),
        "decision_as_of": as_of.isoformat(),
        "fixed_baseline_start": "2010-01-01",
        "fixed_baseline_end": "2014-12-31",
        "baseline_rows": int(baseline_mask.sum()),
        "raw_returns_finite": int(np.isfinite(raw).sum()),
        "raw_returns_total": int(raw.size),
        "residual_returns_finite": int(np.isfinite(residuals).sum()),
        "residual_returns_total": int(residuals.size),
        "residual_returns_finite_rows": int(np.isfinite(residuals).all(axis=1).sum()),
        "residual_returns_rows": int(residuals.shape[0]),
        "c_full": matrix_stats("c_full"),
        "c_full_p3": matrix_stats("c_full_p3"),
    }


def _run_one_day_preflight(
    *,
    app_config: Any,
    frame: pd.DataFrame,
    eval_date: pd.Timestamp,
    historical: HistoricalInputs,
    input_manifest: dict[str, Any],
    transform_cache: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    previous = _install_w6({tk: float(row["w6"]) for tk, row in SENSITIVITY_LABELS.items()})
    counters, originals = _install_measurement_hooks(transform_cache, "preflight_current_w6")
    started = time.monotonic()
    try:
        result = BacktestEngine.run_v2_backtest(
            cfg=app_config,
            gap_input_dir=input_manifest["gap_store_path"],
            df_exec=frame,
            start_date=eval_date.strftime("%Y-%m-%d"),
            end_date=eval_date.strftime("%Y-%m-%d"),
            n_jobs=1,
            historical_inputs=historical,
        )
    finally:
        _restore_measurement_hooks(originals)
        _restore_w6(previous)
    output = {
        "preflight_date": eval_date.strftime("%Y-%m-%d"),
        "elapsed_seconds": time.monotonic() - started,
        "fallback": bool(result["daily_fallback"].iloc[0]),
        "net_return": float(result["daily_returns"].iloc[0]),
        "summary": result["v2_summaries"][0],
        "decisions": counters["decisions"],
        "leakage_status": dict(counters["leakage_status"]),
        "numerical_status": dict(counters["numerical_status"]),
        "decide_exception": counters.get("decide_exception"),
        "distribution_finite": {
            key: value for key, value in counters.items()
            if str(key).startswith("distribution_finite_")
        },
        "on_demand": {
            f"h{h}:{status}": count
            for (h, status), count in counters["on_demand_by_horizon_status"].items()
        },
        "transform_cache_hits": dict(counters["transform_cache_hits"]),
        "equivalence_checks": transform_cache["equivalence_checks"],
    }
    return output, counters


def _write_blocked_preflight(
    *,
    config: dict[str, Any],
    app_config: Any,
    baseline_labels: dict[str, dict[str, float]],
    variant_maps: dict[str, dict[str, float]],
    input_manifest: dict[str, Any],
    prior_diagnostic: dict[str, Any],
    preflight: dict[str, Any],
    started_at: datetime,
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    omega_has_nonfinite = any(
        item["omega_finite"] != item["omega_total"]
        for item in preflight["distribution_finite"].values()
    )
    reasons = [
        "最初のV2 decisionがfallbackまたは例外になり、全期間比較へ進めない",
        str(preflight.get("decide_exception") or "preflight fallback without a model decision"),
    ]
    if omega_has_nonfinite:
        reasons.append("on-demand Omegaに非有限値が含まれる")
    diagnostic = {
        "experiment_id": config["experiment_id"],
        "status": "blocked_by_preflight_failure",
        "decision": Decision.PENDING.value,
        "decision_reasons": reasons,
        "created_at": started_at.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "effective_config_fingerprint": model_config_fingerprint(app_config),
        "research_config_fingerprint": hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest(),
        "input_manifest": input_manifest,
        "baseline_prior_diagnostic": prior_diagnostic,
        "preflight": preflight,
        "current_w6": {tk: float(row["w6"]) for tk, row in baseline_labels.items()},
        "candidate_w6": variant_maps,
        "performance_metrics": None,
        "performance_trial_count_increment": 0,
    }
    (RESULTS_DIR / "preflight.json").write_text(
        json.dumps(diagnostic, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    finite_prior = prior_diagnostic["c_full_p3"]["finite"]
    prior_total = prior_diagnostic["c_full_p3"]["total"]
    baseline_mapping = _mapping_markdown(baseline_labels, variant_maps[PRIMARY_NAME])
    rows = [
        "# w6感応度見直しバックテスト：事前チェックで停止",
        "",
        "## 判定",
        "",
        f"**保留。長期バックテストは実行していない。** 最初の評価日で preflight が失敗した (`{preflight.get('decide_exception') or 'fallback without a model decision'}`)。フラットfallbackのリターンをモデル成績として扱うことはできないため、Sharpe・DD・DSR等は算出していない。",
        "",
        "## 確認できた原因",
        "",
        f"- 最初の評価日: {preflight['preflight_date']}。decision例外: `{preflight['decide_exception']}`。fallback={preflight['fallback']}、実行されたモデルdecision={preflight['decisions']}。",
        f"- 固定prior期間: {prior_diagnostic['fixed_baseline_start']}〜{prior_diagnostic['fixed_baseline_end']}、{prior_diagnostic['baseline_rows']}行。`c_full_p3` は {finite_prior}/{prior_total} 要素が有限。US×JPブロックは {prior_diagnostic['c_full_p3']['us_jp_finite']}/{prior_diagnostic['c_full_p3']['us_jp_total']} 要素が有限。",
        f"- 入力残差は固定期間で {prior_diagnostic['residual_returns_finite']}/{prior_diagnostic['residual_returns_total']} 要素が有限、全列有限の行は {prior_diagnostic['residual_returns_finite_rows']}/{prior_diagnostic['residual_returns_rows']}。残差betaの初期NaNを含む配列が baseline相関計算へ渡り、非有限priorに伝播している。",
        f"- on-demand出力の有限要素数: `{json.dumps(preflight['distribution_finite'], ensure_ascii=False, default=_json_default)}`。READYは供給元の状態を示すが、Ωが数値的に使えることを保証せず、その後の固有値計算で例外になった。",
        "- 例外がportfolio生成・監査前に発生したため、モデルウェイトと実効エクスポージャーは未生成。leakage監査・数値監査も未実施で、PASSとは扱わない。",
        "- baselineは規約どおり2010–2014固定。評価開始行や後年データへ差し替える回避策は使っていない。",
        "",
        "## 感応度候補",
        "",
        "候補は定性的な名目金利prior。銀行を正、不動産・公益・長期成長を負にし、w3〜w5は固定した。これはBLPX静的priorだけの候補で、macro kappa/directionの `MACRO_SENS_MATRIX` は対象外。候補値と現行値は以下。",
        "",
        baseline_mapping,
        "",
        "候補・弱め・強めの三つは事前定義したが、バックテストへ適用していないため性能試行には数えない。現行感応度のcontrol runも全期間では実行していない。採用・不採用の判断はせず、保留とした。",
        "",
        "## 未実施の評価",
        "",
        "予定していた比較は、全評価日を含むNet Sharpe/最大DD/片道turnover/fallback率、コスト4内訳、paired 20営業日非循環block bootstrap (5,000 resamples, seed=20260924)、試行数5を使うDSRである。採用基準はNet Sharpe +0.05、日次Net差CI下限>0、DD悪化≤1.0pp、turnover増加≤10%、fallback非増加、モデルgross≤2・abs(net)≤0.05。いずれも事前定義だけで、結果は算出していない。OOS区間成績、費用、監査値、Sharpe/DSRは未取得。",
        "",
        "## データと再現情報",
        "",
        f"- 評価予定期間: {input_manifest['simulation_start']}〜{input_manifest['simulation_end']}、{input_manifest['simulation_dates']}営業日。df_execは {input_manifest['df_exec_rows']}行 ({input_manifest['df_exec_start']}〜{input_manifest['df_exec_end']})、fingerprint `{input_manifest['df_exec_fingerprint']}`。",
        f"- 9:10価格の実測: {input_manifest['open_910_dates_with_any_finite_price']}日で一部あり、17 ETF全て揃う日は {input_manifest['open_910_dates_complete_for_17_etfs']}日。実測セル {input_manifest['open_910_finite_cells_in_evaluation']}/{input_manifest['open_910_total_cells_in_evaluation']}。",
        f"- macroは {input_manifest['macro_start']}〜{input_manifest['macro_end']} ({input_manifest['macro_rows']}行, `{input_manifest['macro_fingerprint']}`)、ADR {input_manifest['adr_rows']}行。gap store評価期間のcached μ={input_manifest['gap_store_cached_mu_dates_in_evaluation']}日、rank-reversal={input_manifest['gap_store_rank_reversal_dates_in_evaluation']}日。cached μ/Ωを避けてon-demandを使う計画だった。",
        f"- 因果変換cacheの標準関数との一致確認: {input_manifest['causal_transform_equivalence_checks']} checks。",
        f"- research effective config fingerprint `{model_config_fingerprint(app_config)}`、production effective config fingerprint `{input_manifest['production_effective_config_fingerprint']}`。設定 `{CONFIG_PATH.relative_to(ROOT)}`。実行スクリプト `{Path(__file__).relative_to(ROOT)}`。",
        "- 実行コマンド: `/usr/bin/timeout -k 30s 900s .venv/bin/python -u src/research/scripts/experiments/experiment_sensitivity_rate_prior_v2_20260924.py`。",
        f"- 診断JSON: `{RESULTS_DIR.relative_to(ROOT)}/preflight.json`。本番設定・`tickers.py`・gap storeは変更していない。",
        "",
        "## 次に必要な修正",
        "",
        "固定基準相関の計算で、欠損を含む残差をどう扱うかをデータ仕様に沿って修正し、回帰テストとリーク監査を通した後にこの感応度比較を再実行する必要がある。今回、基準期間を緩めたり監査を回避したりする変更はしていない。",
        "",
    ]
    (REPORT_DIR / "report.md").write_text("\n".join(rows), encoding="utf-8")

    registry = ExperimentRegistry(REGISTRY_PATH)
    registry.record(
        ExperimentRecord(
            name="20260924_sensitivity_rate_prior_v2_preflight_blocked",
            hypothesis=str(config["hypothesis"]),
            start_time=started_at,
            end_time=datetime.now(UTC),
            parameters={
                "research_config": str(CONFIG_PATH.relative_to(ROOT)),
                "candidate_w6": variant_maps,
                "w3_w4_w5": "unchanged from production tickers.py",
                "backtest_start": config["evaluation"]["start_date"],
                "backtest_end": config["evaluation"]["end_date"],
                "effective_config_fingerprint": model_config_fingerprint(app_config),
                "df_exec_fingerprint": input_manifest["df_exec_fingerprint"],
                "gap_cache_policy": "all mu/Omega reads bypassed; on-demand BLPX preflight",
            },
            metrics={
                "status": "blocked_by_preflight_failure",
                "preflight_date": preflight["preflight_date"],
                "decide_exception": preflight["decide_exception"],
                "baseline_prior_diagnostic": prior_diagnostic,
                "distribution_finite": preflight["distribution_finite"],
                "performance_metrics": None,
                "performance_trial_count_increment": 0,
            },
            decision=Decision.PENDING,
            report_path=str((REPORT_DIR / "report.md").relative_to(ROOT)),
            related_records=[
                "20260924_jpx33_sensitivity_prior",
                "20260924_jpx33_sensitivity_open_long_correction_1",
            ],
        )
    )


def _safe_sharpe(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        return float("nan")
    sigma = float(np.std(values, ddof=1))
    return float(np.mean(values) / sigma * np.sqrt(TRADING_DAYS)) if sigma > 1e-16 else float("nan")


def _metrics(result: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    returns = result["daily_returns"].astype(float)
    gross_returns = result["daily_returns_gross"].astype(float)
    weights = result["weights"].astype(float)
    model_gross = weights.abs().sum(axis=1)
    model_net = weights.sum(axis=1)
    side_leverage = float(result["side_leverage"])
    spec = MetricsSpec(annualization_periods=TRADING_DAYS, include_flat_days=True)
    net_stats = calculate_metrics(returns, spec=spec)
    gross_stats = calculate_metrics(gross_returns, spec=spec)
    daily = pd.DataFrame(index=returns.index)
    daily["net_return"] = returns
    daily["gross_return"] = gross_returns
    daily["cost"] = result["daily_costs"].astype(float)
    daily["slippage_cost"] = result["daily_slip_costs"].astype(float)
    daily["financing_cost"] = result["daily_financing_costs"].astype(float)
    daily["borrow_cost"] = result["daily_borrow_costs"].astype(float)
    daily["reverse_cost"] = result["daily_reverse_costs"].astype(float)
    daily["turnover_one_way"] = result["daily_turnover"].astype(float)
    daily["model_gross"] = model_gross
    daily["model_net"] = model_net
    daily["effective_gross"] = model_gross * side_leverage
    daily["effective_net"] = model_net * side_leverage
    daily["fallback"] = result["daily_fallback"].astype(bool)
    daily["year"] = daily.index.year

    component_sum = sum(
        result[key].astype(float)
        for key in (
            "daily_slip_costs",
            "daily_financing_costs",
            "daily_borrow_costs",
            "daily_reverse_costs",
        )
    )
    gross_net_gap = (gross_returns - returns - result["daily_costs"].astype(float)).abs()
    cost_reconciliation_gap = (component_sum - result["daily_costs"].astype(float)).abs()
    summary = {
        "n_observations": int(len(returns)),
        "net_sharpe": float(net_stats.get("Sharpe", np.nan)),
        "gross_sharpe": float(gross_stats.get("Sharpe", np.nan)),
        "net_total_return": float(net_stats.get("Total Return", np.nan)),
        "gross_total_return": float(gross_stats.get("Total Return", np.nan)),
        "net_annualized_return": float(net_stats.get("AR", np.nan)),
        "net_annualized_risk": float(net_stats.get("RISK", np.nan)),
        "max_drawdown": float(net_stats.get("MDD", np.nan)),
        "mean_daily_turnover_one_way": float(daily["turnover_one_way"].mean()),
        "total_turnover_one_way": float(daily["turnover_one_way"].sum()),
        "fallback_days": int(daily["fallback"].sum()),
        "fallback_rate": float(daily["fallback"].mean()),
        "mean_model_gross": float(daily["model_gross"].mean()),
        "max_model_gross": float(daily["model_gross"].max()),
        "max_abs_model_net": float(daily["model_net"].abs().max()),
        "mean_effective_gross": float(daily["effective_gross"].mean()),
        "max_effective_gross": float(daily["effective_gross"].max()),
        "max_abs_effective_net": float(daily["effective_net"].abs().max()),
        "side_leverage": side_leverage,
        "costs_total_return_fraction": float(daily["cost"].sum()),
        "costs_total_bps": float(daily["cost"].sum() * 10000.0),
        "mean_daily_cost_bps": float(daily["cost"].mean() * 10000.0),
        "slippage_total_bps": float(daily["slippage_cost"].sum() * 10000.0),
        "financing_total_bps": float(daily["financing_cost"].sum() * 10000.0),
        "borrow_total_bps": float(daily["borrow_cost"].sum() * 10000.0),
        "reverse_total_bps": float(daily["reverse_cost"].sum() * 10000.0),
        "max_gross_net_reconciliation_error": float(gross_net_gap.max()),
        "max_cost_component_reconciliation_error": float(cost_reconciliation_gap.max()),
        "leakage_status_counts": {},
        "numerical_status_counts": {},
    }
    return summary, daily


def _moving_block_ci(
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    block_length: int,
    resamples: int,
    seed: int,
) -> dict[str, list[float]]:
    if baseline.shape != candidate.shape or baseline.ndim != 1 or len(baseline) < block_length:
        raise ValueError("Paired bootstrap series have incompatible lengths")
    n = len(baseline)
    blocks_needed = int(np.ceil(n / block_length))
    rng = np.random.default_rng(seed)
    mean_deltas = np.empty(resamples, dtype=float)
    sharpe_deltas = np.empty(resamples, dtype=float)
    max_start = n - block_length
    for sample_id in range(resamples):
        starts = rng.integers(0, max_start + 1, size=blocks_needed)
        indices = np.concatenate(
            [np.arange(start, start + block_length, dtype=int) for start in starts]
        )[:n]
        base_sample = baseline[indices]
        candidate_sample = candidate[indices]
        mean_deltas[sample_id] = float(np.mean(candidate_sample - base_sample))
        sharpe_deltas[sample_id] = _safe_sharpe(candidate_sample) - _safe_sharpe(base_sample)
    mean_ci = np.nanpercentile(mean_deltas, [2.5, 97.5]).tolist()
    sharpe_ci = np.nanpercentile(sharpe_deltas, [2.5, 97.5]).tolist()
    return {
        "mean_daily_net_return_delta": [float(x) for x in mean_ci],
        "annualized_net_sharpe_delta": [float(x) for x in sharpe_ci],
    }


def _annual_metrics(variants: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant_name, daily in variants.items():
        for year, group in daily.groupby("year", sort=True):
            returns = group["net_return"].to_numpy(dtype=float)
            rows.append(
                {
                    "variant": variant_name,
                    "year": int(year),
                    "dates": int(len(group)),
                    "net_compounded_return": float(np.prod(1.0 + returns) - 1.0),
                    "net_sharpe": _safe_sharpe(returns),
                    "max_drawdown": float(compute_drawdown_series(pd.Series(returns)).min()),
                    "mean_daily_turnover_one_way": float(group["turnover_one_way"].mean()),
                    "fallback_rate": float(group["fallback"].mean()),
                }
            )
    result = pd.DataFrame(rows)
    base_annual = result[result["variant"] == BASELINE_NAME].set_index("year")
    candidate_annual = result[result["variant"] == PRIMARY_NAME].set_index("year")
    result["primary_minus_baseline_net_return"] = result.apply(
        lambda row: (
            candidate_annual.loc[row["year"], "net_compounded_return"]
            - base_annual.loc[row["year"], "net_compounded_return"]
            if row["variant"] == PRIMARY_NAME and row["year"] in base_annual.index
            else np.nan
        ),
        axis=1,
    )
    return result


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _mapping_markdown(base: dict[str, dict[str, float]], candidate: dict[str, float]) -> str:
    lines = ["| 対象 | 現行 w6 | 改訂 w6 |", "|---|---:|---:|"]
    for ticker in list(US_TICKERS) + list(JP_TICKERS):
        lines.append(f"| {ticker} | {base[ticker]['w6']:+.1f} | {candidate[ticker]:+.1f} |")
    return "\n".join(lines)


def _decision(
    baseline: dict[str, Any],
    primary: dict[str, Any],
    paired_ci: dict[str, list[float]],
    thresholds: dict[str, Any],
) -> tuple[Decision, list[str]]:
    reasons: list[str] = []
    sharpe_delta = primary["net_sharpe"] - baseline["net_sharpe"]
    dd_worsening_pp = (baseline["max_drawdown"] - primary["max_drawdown"]) * 100.0
    turnover_increase = (
        primary["total_turnover_one_way"] / baseline["total_turnover_one_way"] - 1.0
        if baseline["total_turnover_one_way"] > 0
        else float("inf")
    )
    limits_failed = (
        primary["max_model_gross"] > float(thresholds["model_gross_max"]) + 1e-8
        or primary["max_abs_model_net"] > float(thresholds["model_abs_net_max"]) + 1e-8
    )
    severe_regression = (
        sharpe_delta <= -0.05
        or dd_worsening_pp > 3.0
        or turnover_increase > 0.25
    )
    if limits_failed:
        reasons.append("市場中立またはgrossの事前制約を外れた")
    if severe_regression:
        reasons.append("Sharpe・DD・turnoverのいずれかで明確な悪化を確認した")
    if limits_failed or severe_regression:
        return Decision.REJECTED, reasons

    conditions = {
        "net Sharpe improvement": sharpe_delta >= float(thresholds["min_net_sharpe_improvement"]),
        "paired daily net return 95% CI": paired_ci["mean_daily_net_return_delta"][0]
        > float(thresholds["paired_net_return_ci_lower_gt"]),
        "max drawdown": dd_worsening_pp <= float(thresholds["max_drawdown_worsening_percentage_points"]),
        "turnover": turnover_increase <= float(thresholds["max_turnover_increase_fraction"]),
        "fallback rate": primary["fallback_rate"] <= baseline["fallback_rate"] + 1e-12,
    }
    reasons.extend(f"未達: {name}" for name, passed in conditions.items() if not passed)
    reasons.append("評価期間を使い切った過去比較のため、新規未使用OOSでの確認が残る")
    return Decision.PENDING, reasons


def _write_outputs(
    *,
    config: dict[str, Any],
    app_config: Any,
    frame: pd.DataFrame,
    eval_dates: pd.DatetimeIndex,
    historical: HistoricalInputs,
    input_manifest: dict[str, Any],
    variant_maps: dict[str, dict[str, float]],
    baseline_labels: dict[str, dict[str, float]],
    results: dict[str, dict[str, Any]],
    counters_by_variant: dict[str, dict[str, Any]],
    elapsed_by_variant: dict[str, float],
    started_at: datetime,
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_by_variant: dict[str, dict[str, Any]] = {}
    daily_by_variant: dict[str, pd.DataFrame] = {}
    for name, result in results.items():
        metrics, daily = _metrics(result)
        counter = counters_by_variant[name]
        metrics["leakage_status_counts"] = dict(counter["leakage_status"])
        metrics["numerical_status_counts"] = dict(counter["numerical_status"])
        metrics["cache_bypass_by_horizon"] = {
            str(k): int(v) for k, v in counter["cache_bypass_by_horizon"].items()
        }
        metrics["on_demand_by_horizon_status"] = {
            f"h{h}:{status}": int(v)
            for (h, status), v in counter["on_demand_by_horizon_status"].items()
        }
        metrics["transform_cache_hits"] = {
            str(k): int(v) for k, v in counter["transform_cache_hits"].items()
        }
        metrics["decisions"] = int(counter["decisions"])
        metrics_by_variant[name] = metrics
        daily_by_variant[name] = daily
        daily.drop(columns=["year"]).to_csv(RESULTS_DIR / f"daily_{name}.csv", index_label="trade_date")
        result["weights"].to_csv(RESULTS_DIR / f"weights_{name}.csv", index_label="trade_date")

    base = metrics_by_variant[BASELINE_NAME]
    primary = metrics_by_variant[PRIMARY_NAME]
    evaluation = config["evaluation"]
    bootstrap = evaluation["bootstrap"]
    paired_ci: dict[str, dict[str, list[float]]] = {}
    base_r = daily_by_variant[BASELINE_NAME]["net_return"].to_numpy(dtype=float)
    for name in VARIANT_NAMES:
        candidate_r = daily_by_variant[name]["net_return"].to_numpy(dtype=float)
        if len(candidate_r) != len(base_r):
            raise ValueError(f"Variant {name} has unpaired dates")
        paired_ci[name] = _moving_block_ci(
            base_r,
            candidate_r,
            block_length=int(bootstrap["block_length_trading_dates"]),
            resamples=int(bootstrap["resamples"]),
            seed=int(bootstrap["seed"]),
        )

    decision, decision_reasons = _decision(
        base,
        primary,
        paired_ci[PRIMARY_NAME],
        evaluation["adoption_thresholds"],
    )
    annual = _annual_metrics(daily_by_variant)
    annual.to_csv(RESULTS_DIR / "annual_metrics.csv", index=False)
    pd.concat(
        [daily.drop(columns=["year"]).assign(variant=name) for name, daily in daily_by_variant.items()]
    ).to_csv(RESULTS_DIR / "daily_paired_results.csv", index_label="trade_date")
    if historical.open_910_returns is not None:
        historical.open_910_returns.to_csv(RESULTS_DIR / "open_910_returns.csv", index_label="trade_date")
    if historical.macro_prices is not None:
        historical.macro_prices.to_csv(RESULTS_DIR / "macro_prices.csv", index_label="date")

    # Trial count includes the two prior completed sensitivity comparisons and
    # the three prespecified candidate vectors in this experiment. The older
    # malformed metric run is excluded; its correction remains in the registry.
    trial_count = int(config["comparison"]["dsr_trial_count"])
    current_candidate_sharpes = [metrics_by_variant[name]["net_sharpe"] for name in VARIANT_NAMES]
    dsr_metrics = {
        "net_sharpe": primary["net_sharpe"],
        "net_sharpe_frequency": "annual",
        "trading_days_per_year": TRADING_DAYS,
        "trials": trial_count,
        "n_observations": int(primary["n_observations"]),
        "returns": daily_by_variant[PRIMARY_NAME]["net_return"].tolist(),
        "trial_sharpes": current_candidate_sharpes,
    }
    dsr = compute_deflated_sharpe(dsr_metrics)

    mapping_md = _mapping_markdown(baseline_labels, variant_maps[PRIMARY_NAME])
    primary_sharpe_delta = primary["net_sharpe"] - base["net_sharpe"]
    primary_return_delta_bps_day = (
        float((daily_by_variant[PRIMARY_NAME]["net_return"] - daily_by_variant[BASELINE_NAME]["net_return"]).mean())
        * 10000.0
    )
    rows = [
        "# w6感応度を見直したV2長期バックテスト",
        "",
        "## 判定",
        "",
        f"**{decision.value}。** " + "。".join(decision_reasons) + "。",
        "",
        "改訂候補は研究用の比較だけに適用し、本番設定と `tickers.py` は変更していない。",
        "採用可否の一次判定は「現行比で年率Net Sharpe +0.05以上、paired日次Net差の20営業日ブロックbootstrap 95% CI下限が0超、最大DD悪化1.0pp以内、turnover増加10%以内、fallback率非増加、市場中立・gross制約内」。いずれも期間全体の過去比較であり、未使用OOSの代替にはしない。",
        "",
        "## 感応度の見直し",
        "",
        "現行w6は「インフレまたは金利の上昇から恩恵を受ける」という複合解釈になっており、エネルギー資源・銀行・不動産の意味が混ざる。候補は、w6の読みを名目金利上昇に寄せ、銀行を強い正、金融を弱い正、不動産・公益・長期成長を負にした。XLEと商社は金利因子へ混ぜない。w3〜w5は固定した。",
        "このw6は回帰係数として推定したものではなく、符号・大きさとも定性的な事前判断である。各評価対象に対して値合わせをしていない。macro kappa/directionが参照する独立の `MACRO_SENS_MATRIX` は変更せず、今回の差はBLPX静的priorに限られる。",
        "",
        mapping_md,
        "",
        "## 期間・データ・実行条件",
        "",
        f"- 評価日: {input_manifest['simulation_start']}〜{input_manifest['simulation_end']}、{input_manifest['simulation_dates']}営業日。両系列は全評価日でpaired。",
        f"- `df_exec`: {input_manifest['df_exec_rows']}行、{input_manifest['df_exec_start']}〜{input_manifest['df_exec_end']}。入力fingerprint `{input_manifest['df_exec_fingerprint']}`。",
        f"- 9:10入力: 評価期間で実測価格あり {input_manifest['open_910_dates_with_any_finite_price']}日、17 ETF全て揃う日は {input_manifest['open_910_dates_complete_for_17_etfs']}日。実測価格セル {input_manifest['open_910_finite_cells_in_evaluation']}/{input_manifest['open_910_total_cells_in_evaluation']}。残りは現行target fallbackに従い、始値→大引けを使う。",
        f"- マクロ価格: {input_manifest['macro_start']}〜{input_manifest['macro_end']}、{input_manifest['macro_rows']}行、fingerprint `{input_manifest['macro_fingerprint']}`。ADR行数 {input_manifest['adr_rows']}、rank-reversal行数 {input_manifest['rank_reversal_rows']}。",
        f"- 設定: `configs/production/production.yaml` を継承解決。歴史評価ではML overlay={input_manifest['overlay_enabled']} (`{input_manifest['overlay_path']}`)。本番 overlay={input_manifest['production_overlay_enabled']}。本番artifactは2024-12-20まで学習済みのため全候補共通で無効化し、未来学習済みモデルの過去適用を避けた。固定gap storeに評価期間内のμ日付 {input_manifest['gap_store_cached_mu_dates_in_evaluation']}日を確認。",
        "- cached μ/Ωはprior依存なので全評価日・全horizonで読まず、V2標準のon-demand BLPXへ再計算を強制した。gap store自体はPIT IR履歴・rank-reversal読込のみに使い、書き換えていない。",
        f"- 長期実行のため、行ごとに過去だけを使うh日累積リターンと分数階差分を一度だけ前計算し、各as-of日に過去範囲を切り出した。先頭・中間・終端の3日×各horizonで標準関数との数値一致を確認した ({input_manifest['causal_transform_equivalence_checks']} checks)。当日closeラベルは切出し後もPITマスクを復元。",
        "- コスト・RuleD・macro等、ML overlay以外の有効設定は本番設定のまま。ML overlayは比較全体で無効化し、理由は履歴期間に対するartifactの学習時点である。評価の終了日はデータ確認済みの2026-09-18で固定。",
        "",
        "## 全期間結果",
        "",
        "| 指標 | 現行w6 | 改訂候補 | 差 |",
        "|---|---:|---:|---:|",
        f"| Net Sharpe (年率、全日) | {_fmt(base['net_sharpe'])} | {_fmt(primary['net_sharpe'])} | {_fmt(primary_sharpe_delta, 4)} |",
        f"| Gross Sharpe (年率) | {_fmt(base['gross_sharpe'])} | {_fmt(primary['gross_sharpe'])} | {_fmt(primary['gross_sharpe'] - base['gross_sharpe'], 4)} |",
        f"| Net累積リターン | {_fmt(base['net_total_return'] * 100, 2)}% | {_fmt(primary['net_total_return'] * 100, 2)}% | {_fmt((primary['net_total_return'] - base['net_total_return']) * 100, 2)}pp |",
        f"| 最大DD | {_fmt(base['max_drawdown'] * 100, 2)}% | {_fmt(primary['max_drawdown'] * 100, 2)}% | {_fmt((primary['max_drawdown'] - base['max_drawdown']) * 100, 2)}pp |",
        f"| Net日次差の平均 | — | — | {primary_return_delta_bps_day:+.4f} bp/day |",
        f"| Net日次差 95% block CI | — | — | [{paired_ci[PRIMARY_NAME]['mean_daily_net_return_delta'][0] * 10000:+.4f}, {paired_ci[PRIMARY_NAME]['mean_daily_net_return_delta'][1] * 10000:+.4f}] bp/day |",
        f"| 年率Net Sharpe差 95% block CI | — | — | [{_fmt(paired_ci[PRIMARY_NAME]['annualized_net_sharpe_delta'][0], 4)}, {_fmt(paired_ci[PRIMARY_NAME]['annualized_net_sharpe_delta'][1], 4)}] |",
        f"| 一方向turnover平均/日 | {_fmt(base['mean_daily_turnover_one_way'], 4)} | {_fmt(primary['mean_daily_turnover_one_way'], 4)} | {_fmt(primary['mean_daily_turnover_one_way'] - base['mean_daily_turnover_one_way'], 4)} |",
        f"| fallback日率 | {_fmt(base['fallback_rate'] * 100, 2)}% | {_fmt(primary['fallback_rate'] * 100, 2)}% | {_fmt((primary['fallback_rate'] - base['fallback_rate']) * 100, 2)}pp |",
        "",
        "### ±感度診断",
        "",
        "| 候補 | Net Sharpe | Δ現行 | 最大DD | turnover増減 | fallback率 | Net日次差 95% CI (bp/day) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in VARIANT_NAMES:
        metric = metrics_by_variant[name]
        ci = paired_ci[name]["mean_daily_net_return_delta"]
        turnover_change = (
            metric["total_turnover_one_way"] / base["total_turnover_one_way"] - 1.0
            if base["total_turnover_one_way"] > 0 else float("nan")
        )
        rows.append(
            f"| {name} | {_fmt(metric['net_sharpe'])} | {_fmt(metric['net_sharpe'] - base['net_sharpe'], 4)} | {_fmt(metric['max_drawdown'] * 100, 2)}% | {_fmt(turnover_change * 100, 2)}% | {_fmt(metric['fallback_rate'] * 100, 2)}% | [{ci[0] * 10000:+.4f}, {ci[1] * 10000:+.4f}] |"
        )
    rows.extend(
        [
            "",
            "診断候補は各w6の非ゼロ値をグリッド上で一段弱める／強める定義。候補採択のための探索ではなく、候補の頑健性を確認するために事前固定した。",
            "",
            "## コスト・エクスポージャー・監査",
            "",
            "| 指標 | 現行w6 | 改訂候補 |",
            "|---|---:|---:|",
            f"| 総コスト | {_fmt(base['costs_total_bps'], 2)} bp | {_fmt(primary['costs_total_bps'], 2)} bp |",
            f"| slippage | {_fmt(base['slippage_total_bps'], 2)} bp | {_fmt(primary['slippage_total_bps'], 2)} bp |",
            f"| financing | {_fmt(base['financing_total_bps'], 2)} bp | {_fmt(primary['financing_total_bps'], 2)} bp |",
            f"| borrow | {_fmt(base['borrow_total_bps'], 2)} bp | {_fmt(primary['borrow_total_bps'], 2)} bp |",
            f"| reverse | {_fmt(base['reverse_total_bps'], 2)} bp | {_fmt(primary['reverse_total_bps'], 2)} bp |",
            f"| gross−cost−net / cost内訳最大誤差 | {base['max_gross_net_reconciliation_error']:.2e} / {base['max_cost_component_reconciliation_error']:.2e} | {primary['max_gross_net_reconciliation_error']:.2e} / {primary['max_cost_component_reconciliation_error']:.2e} |",
            f"| モデルgross平均 / 最大 | {_fmt(base['mean_model_gross'], 3)} / {_fmt(base['max_model_gross'], 3)} | {_fmt(primary['mean_model_gross'], 3)} / {_fmt(primary['max_model_gross'], 3)} |",
            f"| モデルabs(net)最大 | {_fmt(base['max_abs_model_net'], 4)} | {_fmt(primary['max_abs_model_net'], 4)} |",
            f"| side_leverage | {base['side_leverage']:.2f} | {primary['side_leverage']:.2f} |",
            f"| 実効gross平均 / 最大 | {_fmt(base['mean_effective_gross'], 3)} / {_fmt(base['max_effective_gross'], 3)} | {_fmt(primary['mean_effective_gross'], 3)} / {_fmt(primary['max_effective_gross'], 3)} |",
            f"| 実効abs(net)最大 | {_fmt(base['max_abs_effective_net'], 4)} | {_fmt(primary['max_abs_effective_net'], 4)} |",
            f"| leakage監査 | {base['leakage_status_counts']} | {primary['leakage_status_counts']} |",
            f"| 数値監査 | {base['numerical_status_counts']} | {primary['numerical_status_counts']} |",
            f"| on-demand結果 | {base['on_demand_by_horizon_status']} | {primary['on_demand_by_horizon_status']} |",
            "",
            "モデルgross/netはRuleD適用後のw_final、実効値はside_leverage適用後。Risk設定のmax_gross=3.0/max_net=0.05とも照合し、AGENTS.mdのRuleD gross≤2.0/net±0.05はモデルウェイトに適用した。コストはbacktesterの日次return_fraction合計をbp換算。総コストと4内訳、gross−netの最大照合差は数値成果物に保存した。",
            "",
            "## 年別推移",
            "",
            "| 年 | 日数 | 現行Net Sharpe | 改訂Net Sharpe | 現行Net累積 | 改訂Net累積 | 改訂−現行 | 改訂MDD |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    base_annual = annual[annual["variant"] == BASELINE_NAME].set_index("year")
    primary_annual = annual[annual["variant"] == PRIMARY_NAME].set_index("year")
    for year in base_annual.index:
        b = base_annual.loc[year]
        c = primary_annual.loc[year]
        rows.append(
            f"| {int(year)} | {int(b['dates'])} | {_fmt(b['net_sharpe'])} | {_fmt(c['net_sharpe'])} | {_fmt(b['net_compounded_return'] * 100, 2)}% | {_fmt(c['net_compounded_return'] * 100, 2)}% | {_fmt((c['net_compounded_return'] - b['net_compounded_return']) * 100, 2)}pp | {_fmt(c['max_drawdown'] * 100, 2)}% |"
        )

    rows.extend(
        [
            "",
            "## 過学習評価と限界",
            "",
            f"- 現行感応度の先行する有効比較2件と、今回の候補・一段弱め・一段強めの3構成を含む試行数 {trial_count} としてDeflated Sharpeを計算: DSR={_fmt(dsr, 4)}。Sharpeの試行間分散は今回の3候補で推定し、先行2件は異なる予測精度指標の比較なので数値混合していない。従って保守的な試行数補正だが、DSRは近似値。",
            "- 候補ベクトルは経済解釈で事前固定したが、同じ長期期間の感応度情報量比較を既に見ている。今回のバックテストも2015–2026年を使った回顧比較で、独立した未来OOSではない。採用には今後のforward/shadow確認が必要。",
            "- 実測9:10データは少数日に限られ、長期の損益は始値→大引けfallbackである。9:10→大引け戦略の長期実損益と同一視しない。",
            "- PIT multiplierの適用率は今回の実行カウンタでは取得していない。on-demand成功率・fallback率とは別指標のため、ゼロとはみなさない。overnight financing / borrow / reverse費用はバックテストの有効cost設定で計上され、保有費用の日数定義は既存cost modelに従う。",
            "- 既存本番gap cacheは現行priorで生成されているため、そのまま使うと感応度差を検証できない。全日on-demandに統一した比較値であり、当日cache経路を含む本番運用成績とは異なる可能性がある。",
            "- `MACRO_SENS_MATRIX`によるmacro kappa/directionは固定されたまま。今回のw6改訂の対象はBLPX静的priorだけ。",
            "",
            "## 再現情報",
            "",
            f"- 作成時刻: {started_at.isoformat()}",
            f"- 設定: `{CONFIG_PATH.relative_to(ROOT)}`; research effective config fingerprint `{model_config_fingerprint(app_config)}`; production effective config fingerprint `{input_manifest['production_effective_config_fingerprint']}`",
            f"- 対象設定fingerprint: `{hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()}`",
            f"- 実行スクリプト: `{Path(__file__).relative_to(ROOT)}`",
            "- コマンド: `.venv/bin/python -u src/research/scripts/experiments/experiment_sensitivity_rate_prior_v2_20260924.py` (外側の停止期限 4時間)",
            f"- 開始時点のregistry records: {config['comparison'].get('registry_records_before', '記録は実行開始時に取得')}; 成果物: `{RESULTS_DIR.relative_to(ROOT)}/`",
            f"- elapsed by run seconds: {json.dumps(elapsed_by_variant, sort_keys=True)}",
            "- 監査・実行カウンタを含む全メトリクス: `var/results/20260924_sensitivity_rate_prior_v2/summary.json`。日次series、各variantのweights、macro/09:10入力を同フォルダに保存。",
            "- 本番設定・本番コード・gap storeは変更していない。",
            "",
        ]
    )
    (REPORT_DIR / "report.md").write_text("\n".join(rows), encoding="utf-8")

    summary_doc = {
        "experiment_id": config["experiment_id"],
        "decision": decision.value,
        "decision_reasons": decision_reasons,
        "created_at": started_at.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "effective_config_fingerprint": model_config_fingerprint(app_config),
        "production_effective_config_fingerprint": input_manifest["production_effective_config_fingerprint"],
        "research_config_fingerprint": hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest(),
        "input_manifest": input_manifest,
        "evaluation": evaluation,
        "candidate_w6": variant_maps,
        "metrics": metrics_by_variant,
        "paired_ci": paired_ci,
        "annual_metrics_csv": "annual_metrics.csv",
        "dsr": {**dsr_metrics, "value": dsr},
        "elapsed_seconds": elapsed_by_variant,
    }
    (RESULTS_DIR / "summary.json").write_text(
        json.dumps(summary_doc, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )

    # Record every tested candidate vector; the current prior is the shared control.
    registry = ExperimentRegistry(REGISTRY_PATH)
    trial_names = [
        "20260924_jpx33_sensitivity_prior",
        "20260924_jpx33_sensitivity_open_long_correction_1",
    ]
    for name in VARIANT_NAMES:
        metric = metrics_by_variant[name]
        trial_sharpe_dsr = {
            **dsr_metrics,
            "net_sharpe": metric["net_sharpe"],
            "returns": daily_by_variant[name]["net_return"].tolist(),
        }
        variant_decision = decision if name == PRIMARY_NAME else Decision.PENDING
        if name != PRIMARY_NAME:
            variant_decision = Decision.PENDING
        registry.record(
            ExperimentRecord(
                name=f"20260924_sensitivity_rate_prior_v2_{name}",
                hypothesis=str(config["hypothesis"]),
                start_time=started_at,
                end_time=datetime.now(UTC),
                parameters={
                    "research_config": str(CONFIG_PATH.relative_to(ROOT)),
                    "variant": name,
                    "w6": variant_maps[name],
                    "w3_w4_w5": "unchanged from production tickers.py",
                    "backtest_start": evaluation["start_date"],
                    "backtest_end": evaluation["end_date"],
                    "effective_config_fingerprint": model_config_fingerprint(app_config),
                    "production_effective_config_fingerprint": input_manifest["production_effective_config_fingerprint"],
                    "df_exec_fingerprint": input_manifest["df_exec_fingerprint"],
                    "macro_fingerprint": input_manifest["macro_fingerprint"],
                    "gap_cache_policy": "all μ/Ω reads bypassed; same on-demand BLPX path",
                    "trial_count_basis": "2 prior valid sensitivity comparisons plus 3 current candidate vectors",
                },
                metrics={
                    **metric,
                    "paired_ci_vs_current": paired_ci[name],
                    "trials": trial_count,
                    "returns": daily_by_variant[name]["net_return"].tolist(),
                    "trial_sharpes": current_candidate_sharpes,
                    "net_sharpe_frequency": "annual",
                    "trading_days_per_year": TRADING_DAYS,
                    "dsr": compute_deflated_sharpe(trial_sharpe_dsr),
                    "dsr_caveat": "Trial count includes prior sensitivity studies with non-Sharpe outcomes; variance estimated on the three current candidate vectors.",
                },
                decision=variant_decision,
                report_path=str((REPORT_DIR / "report.md").relative_to(ROOT)),
                related_records=trial_names,
            )
        )


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Counter):
        return dict(value)
    raise TypeError(f"Cannot JSON encode {type(value).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Run only the first evaluation date to verify the exact-cache path; do not write study results.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(name)s %(message)s")
    LOG.setLevel(logging.WARNING)
    started_at = datetime.now(UTC)
    config = _load_config()
    evaluation = config["evaluation"]
    production_config_path = ROOT / config["base_config"]
    production_config = load_config_from_yaml(production_config_path, strict=True)
    production_overlay_enabled, production_overlay_path = resolve_overlay_settings(production_config)
    research_overlay_enabled = bool(config["comparison"]["historical_ml_overlay_enabled"])
    if research_overlay_enabled and not production_overlay_enabled:
        raise ValueError("Research config requests an ML overlay that production does not enable")
    app_config = safe_config_copy(production_config)
    if production_overlay_enabled and not research_overlay_enabled:
        # The configured artifact was trained through 2024-12-20. Historical
        # decisions before that cutoff need fold-specific artifacts, which are
        # unavailable, so every paired candidate uses the same ML-disabled path.
        app_config = app_config.model_copy(
            update={
                "v2": app_config.v2.model_copy(update={"ml_overlay_enabled": False}),
                "ml_order_overlay": app_config.ml_order_overlay.model_copy(update={"enabled": False}),
            }
        )
    baseline_labels = copy.deepcopy(SENSITIVITY_LABELS)
    variant_maps = _variant_w6_maps(config)
    frame, eval_dates, historical, input_manifest = _load_owned_inputs(
        app_config,
        str(evaluation["start_date"]),
        str(evaluation["end_date"]),
    )
    input_manifest["production_effective_config_fingerprint"] = model_config_fingerprint(production_config)
    input_manifest["production_overlay_enabled"] = bool(production_overlay_enabled)
    input_manifest["production_overlay_path"] = (
        None if production_overlay_path is None else str(production_overlay_path)
    )
    input_manifest["research_overlay_disabled_reason"] = (
        str(config["comparison"]["historical_ml_overlay_reason"])
        if production_overlay_enabled and not research_overlay_enabled else None
    )
    transform_cache = _build_transform_cache(frame, app_config, eval_dates)
    input_manifest["causal_transform_equivalence_checks"] = int(transform_cache["equivalence_checks"])
    input_manifest["causal_transform_horizons"] = list(transform_cache["horizons"])
    input_manifest["fractional_difference_parameters"] = transform_cache["frac_parameters"]
    before_count = sum(1 for _ in ExperimentRegistry(REGISTRY_PATH))
    config["comparison"]["registry_records_before"] = before_count
    LOG.warning(
        "Starting paired V2 sensitivity backtest for %d dates (%s to %s); open910 complete dates=%d",
        len(eval_dates), eval_dates.min().date(), eval_dates.max().date(),
        input_manifest["open_910_dates_complete_for_17_etfs"],
    )

    prior_diagnostic = _baseline_prior_diagnostic(app_config, historical, eval_dates[0])
    preflight, _preflight_counters = _run_one_day_preflight(
        app_config=app_config,
        frame=frame,
        eval_date=eval_dates[0],
        historical=historical,
        input_manifest=input_manifest,
        transform_cache=transform_cache,
    )
    preflight["baseline_prior_diagnostic"] = prior_diagnostic
    print(json.dumps(preflight, ensure_ascii=False, default=_json_default), flush=True)
    if args.preflight:
        return

    configured_horizons = sorted({1, *(int(h) for h in app_config.v2.mh_horizons)})
    distribution_checks = preflight["distribution_finite"]
    all_omega_finite = all(
        f"distribution_finite_h{h}" in distribution_checks
        and distribution_checks[f"distribution_finite_h{h}"]["omega_finite"]
        == distribution_checks[f"distribution_finite_h{h}"]["omega_total"]
        and distribution_checks[f"distribution_finite_h{h}"]["omega_total"] > 0
        for h in configured_horizons
    )
    preflight_passed = (
        not preflight["fallback"]
        and preflight["decide_exception"] is None
        and preflight["decisions"] == 1
        and all_omega_finite
    )
    if not preflight_passed:
        _write_blocked_preflight(
            config=config,
            app_config=app_config,
            baseline_labels=baseline_labels,
            variant_maps=variant_maps,
            input_manifest=input_manifest,
            prior_diagnostic=prior_diagnostic,
            preflight=preflight,
            started_at=started_at,
        )
        print(f"blocked_report={REPORT_DIR / 'report.md'}", flush=True)
        raise SystemExit(2)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    preflight_artifact = {
        "experiment_id": config["experiment_id"],
        "status": "passed",
        "created_at": started_at.isoformat(),
        "effective_config_fingerprint": model_config_fingerprint(app_config),
        "input_manifest": input_manifest,
        "baseline_prior_diagnostic": prior_diagnostic,
        "preflight": preflight,
        "performance_metrics": None,
        "performance_trial_count_increment": 0,
    }
    (RESULTS_DIR / "preflight.json").write_text(
        json.dumps(preflight_artifact, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )

    results: dict[str, dict[str, Any]] = {}
    counters_by_variant: dict[str, dict[str, Any]] = {}
    elapsed_by_variant: dict[str, float] = {}
    run_maps = {BASELINE_NAME: {tk: float(row["w6"]) for tk, row in baseline_labels.items()}, **variant_maps}
    try:
        for variant_name, values in run_maps.items():
            previous = _install_w6(values)
            counters, originals = _install_measurement_hooks(transform_cache, variant_name)
            started = time.monotonic()
            LOG.warning("Run started: %s", variant_name)
            try:
                results[variant_name] = BacktestEngine.run_v2_backtest(
                    cfg=app_config,
                    gap_input_dir=input_manifest["gap_store_path"],
                    df_exec=frame,
                    start_date=str(evaluation["start_date"]),
                    end_date=str(evaluation["end_date"]),
                    n_jobs=int(config["comparison"]["run_jobs"]),
                    historical_inputs=historical,
                )
            finally:
                elapsed_by_variant[variant_name] = time.monotonic() - started
                counters_by_variant[variant_name] = counters
                _restore_measurement_hooks(originals)
                _restore_w6(previous)
            if len(results[variant_name]["daily_returns"]) != len(eval_dates):
                raise ValueError(
                    f"Backtest returned {len(results[variant_name]['daily_returns'])} dates, "
                    f"expected {len(eval_dates)} for {variant_name}"
                )
            LOG.warning(
                "Run complete: %s elapsed=%.1fs decisions=%s fallback=%s on_demand=%s",
                variant_name,
                elapsed_by_variant[variant_name],
                counters["decisions"],
                int(results[variant_name]["daily_fallback"].sum()),
                {f"h{h}:{status}": count for (h, status), count in counters["on_demand_by_horizon_status"].items()},
            )
    finally:
        _restore_w6(baseline_labels)

    _write_outputs(
        config=config,
        app_config=app_config,
        frame=frame,
        eval_dates=eval_dates,
        historical=historical,
        input_manifest=input_manifest,
        variant_maps=variant_maps,
        baseline_labels=baseline_labels,
        results=results,
        counters_by_variant=counters_by_variant,
        elapsed_by_variant=elapsed_by_variant,
        started_at=started_at,
    )
    print(f"report={REPORT_DIR / 'report.md'}", flush=True)
    print(f"results={RESULTS_DIR}", flush=True)


if __name__ == "__main__":
    main()
