#!/usr/bin/env python3
"""Isolate output aggregation weights in a fixed 33D-to-17D forecast."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/research/scripts/experiments"))

import experiment_jpx33_direct_prediction_20260924 as prior  # noqa: E402
import experiment_jpx33_index_ohlc_20260926 as source_exp  # noqa: E402
from leadlag.core.correlation import compute_baseline_correlation  # noqa: E402
from leadlag.core.gap_adjustment import build_raw_distribution  # noqa: E402
from leadlag.core.residualize import compute_rolling_ols_betas  # noqa: E402
from leadlag.data.market_data_cache import load_df_exec_from_local_cache  # noqa: E402
from leadlag.data.tickers import JP_TICKERS, SENSITIVITY_LABELS, US_TICKERS  # noqa: E402
from leadlag.execution.config import load_config_from_yaml  # noqa: E402
from leadlag.experiment_registry import Decision, ExperimentRecord, ExperimentRegistry  # noqa: E402
from leadlag.utils.dataframe_fingerprint import dataframe_fingerprint  # noqa: E402

EXPERIMENT_ID = "20260926_jpx33_mapping_ablation"
CONFIG_PATH = ROOT / "configs/research/jpx33_mapping_ablation_2010_2026.yaml"
LABEL_PATH = ROOT / "configs/research/sensitivity_audit_jpx33_candidate_20260924.yaml"
PARENT_PATH = ROOT / "configs/research/jpx33_sensitivity_prior_2017.yaml"
CAP_PATH = ROOT / "configs/research/jpx33_yearend_market_cap_2014_2025.csv"
PREVIOUS_RESULT_DIR = ROOT / "var/results/20260926_jpx33_index_ohlc"
REPORT_DIR = ROOT / "reports/20260926_jpx33_mapping_ablation"
RESULTS_DIR = ROOT / "var/results/20260926_jpx33_mapping_ablation"
REGISTRY_PATH = ROOT / "var/experiments/registry.jsonl"
START_DATE = pd.Timestamp("2015-01-05")
END_DATE = pd.Timestamp("2026-09-18")
BASELINE_START = pd.Timestamp("2010-01-01")
BASELINE_END = pd.Timestamp("2014-12-31")
BOOTSTRAP_BLOCK = 20
BOOTSTRAP_RESAMPLES = 5000
BOOTSTRAP_SEED = 20260926
LOG = logging.getLogger(EXPERIMENT_ID)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a YAML mapping in {path}")
    return value


def _equal_child_map(
    industry_rows: list[list[Any]],
    parents: list[str],
    industry_order: list[str],
) -> np.ndarray:
    matrix = np.zeros((len(parents), len(industry_order)), dtype=float)
    for parent_i, parent in enumerate(parents):
        members = [i for i, row in enumerate(industry_rows) if str(row[1]) == parent]
        if not members:
            raise ValueError(f"No 33-industry children found for parent {parent}")
        matrix[parent_i, members] = 1.0 / len(members)
    return matrix


def _fit_simplex_map(
    industry_rows: list[list[Any]],
    parents: list[str],
    industry_order: list[str],
    parent_tickers: dict[str, str],
    industry_residuals: np.ndarray,
    target_residuals: np.ndarray,
    fit_mask: np.ndarray,
    initial_map: np.ndarray,
) -> tuple[np.ndarray, dict[str, list[float]]]:
    matrix = np.zeros_like(initial_map)
    weights_by_parent: dict[str, list[float]] = {}
    ticker_idx = {ticker: i for i, ticker in enumerate(JP_TICKERS)}
    industry_idx = {industry: i for i, industry in enumerate(industry_order)}

    for parent_i, parent in enumerate(parents):
        member_indices = [i for i, row in enumerate(industry_rows) if str(row[1]) == parent]
        if not member_indices:
            raise ValueError(f"No 33-industry children found for parent {parent}")
        if len(member_indices) == 1:
            weights = np.ones(1, dtype=float)
        else:
            X = industry_residuals[fit_mask][:, member_indices]
            y = target_residuals[fit_mask, ticker_idx[parent_tickers[parent]]]
            if not np.isfinite(X).all() or not np.isfinite(y).all():
                raise ValueError(f"Non-finite data in baseline fit for {parent}")
            x0 = initial_map[parent_i, member_indices].copy()

            def objective(value: np.ndarray) -> float:
                error = X @ value - y
                return float(error @ error / len(y))

            def gradient(value: np.ndarray) -> np.ndarray:
                error = X @ value - y
                return 2.0 * (X.T @ error) / len(y)

            result = minimize(
                objective,
                x0,
                jac=gradient,
                method="SLSQP",
                bounds=[(0.0, 1.0)] * len(member_indices),
                constraints=[{
                    "type": "eq",
                    "fun": lambda value: float(np.sum(value) - 1.0),
                    "jac": lambda value: np.ones_like(value),
                }],
                options={"maxiter": 1000, "ftol": 1e-12},
            )
            if not result.success:
                raise RuntimeError(f"Simplex mapping fit failed for {parent}: {result.message}")
            weights = np.clip(np.asarray(result.x, dtype=float), 0.0, 1.0)
            weights /= weights.sum()
        matrix[parent_i, member_indices] = weights
        weights_by_parent[parent] = [float(value) for value in weights]

    if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-10, rtol=0.0):
        raise ValueError("Fitted simplex weights do not sum to one")
    return matrix, weights_by_parent


def _mapping_metrics(
    mapped_realized: np.ndarray,
    target: np.ndarray,
    dates: pd.DatetimeIndex,
    ticker_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_rows = []
    ticker_rows = []
    for name, values in mapped_realized.items():
        error = values - target
        for day_i, date in enumerate(dates):
            daily_rows.append({
                "trade_date": date.strftime("%Y-%m-%d"),
                "mapping": name,
                "mae_bps": float(np.mean(np.abs(error[day_i])) * 10000.0),
                "rmse_bps": float(np.sqrt(np.mean(error[day_i] ** 2)) * 10000.0),
            })
        for ticker_i, ticker in enumerate(ticker_names):
            mapped = values[:, ticker_i]
            actual = target[:, ticker_i]
            ticker_rows.append({
                "mapping": name,
                "ticker": ticker,
                "correlation": float(np.corrcoef(mapped, actual)[0, 1]),
                "mae_bps": float(np.mean(np.abs(mapped - actual)) * 10000.0),
                "rmse_bps": float(np.sqrt(np.mean((mapped - actual) ** 2)) * 10000.0),
                "bias_bps": float(np.mean(mapped - actual) * 10000.0),
            })
    return pd.DataFrame(daily_rows), pd.DataFrame(ticker_rows)


def _block_ci(values: np.ndarray, seed: int) -> list[float]:
    old_seed = prior.BOOTSTRAP_SEED
    try:
        prior.BOOTSTRAP_SEED = seed
        return prior._block_ci(np.asarray(values, dtype=float), seed)
    finally:
        prior.BOOTSTRAP_SEED = old_seed


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true", help="Validate the fixed model and all maps on one date")
    args = parser.parse_args()
    started_at = datetime.now(UTC)
    started_clock = time.monotonic()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    config = _read_yaml(CONFIG_PATH)
    label_cfg = _read_yaml(LABEL_PATH)
    parent_cfg = _read_yaml(PARENT_PATH)
    industry_rows = label_cfg["industries"]["rows"]
    industry_order = [str(row[0]) for row in industry_rows]
    parent_rows = parent_cfg["industries"]["rows"]
    industry_to_parent = {str(row[0]): str(row[1]) for row in parent_rows}
    parent_tickers = {str(k): str(v) for k, v in parent_cfg["sensitivity_grid"]["jp_tickers"].items()}
    parents = list(parent_tickers)
    if set(industry_order) != set(industry_to_parent):
        raise ValueError("Frozen labels and parent mapping do not cover the same 33 industries")
    if [parent_tickers[parent] for parent in parents] != list(JP_TICKERS):
        raise ValueError("Parent map order differs from canonical JP_TICKERS order")

    app = load_config_from_yaml(ROOT / config["inputs"]["production_config"], strict=True)
    market = load_df_exec_from_local_cache(max_stale_bdays=None).copy()
    market.index = pd.DatetimeIndex(pd.to_datetime(market.index)).tz_localize(None).normalize()
    market = market.loc[(market.index >= BASELINE_START) & (market.index <= END_DATE)].sort_index()
    if market.index.has_duplicates:
        raise ValueError("df_exec contains duplicate trade dates")

    source_panel, data_quality, date_alignment = source_exp._load_index_ohlc(
        source_exp.DATA_DIR, industry_order, market.index
    )
    industry_panel = source_panel.reindex(market.index)[industry_order]
    data_quality.to_csv(RESULTS_DIR / "input_data_quality.csv", index=False)
    (RESULTS_DIR / "input_data_alignment.json").write_text(
        json.dumps(date_alignment, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    us_raw = market[[f"us_cc_{ticker}" for ticker in US_TICKERS]].to_numpy(dtype=float)
    y17_raw = market[[f"jp_oc_{ticker}" for ticker in JP_TICKERS]].to_numpy(dtype=float)
    y33_raw = industry_panel.to_numpy(dtype=float)
    topix = market["topix_oc_return"].to_numpy(dtype=float)
    beta_window = int(app.v2.blpx.beta_window)
    beta17 = compute_rolling_ols_betas(y17_raw, topix.reshape(-1, 1), beta_window)[:, :, 0]
    beta33 = compute_rolling_ols_betas(y33_raw, topix.reshape(-1, 1), beta_window)[:, :, 0]
    y17_res = y17_raw - beta17 * topix.reshape(-1, 1)
    y33_res = y33_raw - beta33 * topix.reshape(-1, 1)
    all17 = np.column_stack([us_raw, y17_res])
    all33 = np.column_stack([us_raw, y33_res])

    baseline_mask = (
        (market.index >= BASELINE_START)
        & (market.index <= BASELINE_END)
        & np.isfinite(all17).all(axis=1)
        & np.isfinite(all33).all(axis=1)
    )
    if int(baseline_mask.sum()) < 1000:
        raise ValueError(f"Only {int(baseline_mask.sum())} complete baseline rows for mapping fit")
    c_full_33 = compute_baseline_correlation(
        all33, market.index.values, ewma_half_life=app.v2.blpx.ewma_halflife
    )

    cap_table = pd.read_csv(CAP_PATH)
    expected_years = set(range(2014, 2026))
    if set(cap_table["snapshot_year"].astype(int)) != expected_years:
        raise ValueError("Year-end industry cap snapshots must cover 2014-2025")
    cap_maps, first_snapshot_schedule, map_industries, built_parents = prior._build_weight_maps(
        parent_cfg["industries"]["rows"], parent_tickers, cap_table
    )
    if map_industries != industry_order or built_parents != parents:
        raise ValueError("Cap mapping order differs from the frozen industry/parent order")

    equal_map = _equal_child_map(parent_rows, parents, industry_order)
    fitted_map, fitted_weights = _fit_simplex_map(
        parent_rows,
        parents,
        industry_order,
        parent_tickers,
        y33_res,
        y17_res,
        baseline_mask,
        cap_maps[2014],
    )
    candidate_maps = {
        "annual_cap": cap_maps,
        "fixed_2014_cap": {year: cap_maps[2014] for year in cap_maps},
        "equal_child": {year: equal_map for year in cap_maps},
        "fit_2010_14_simplex": {year: fitted_map for year in cap_maps},
    }

    eval_mask = (market.index >= START_DATE) & (market.index <= END_DATE)
    eval_dates_all = pd.DatetimeIndex(market.index[eval_mask])
    first_sessions = {
        int(year): pd.Timestamp(year_dates[0])
        for year in sorted(set(eval_dates_all.year))
        if len(year_dates := eval_dates_all[eval_dates_all.year == year]) > 0
    }
    snapshot_by_date: dict[pd.Timestamp, int] = {}
    included_dates: list[pd.Timestamp] = []
    excluded_dates: list[str] = []
    for date in eval_dates_all:
        snapshot_year = int(date.year) - (2 if date == first_sessions[int(date.year)] else 1)
        position = market.index.get_loc(date)
        if snapshot_year not in cap_maps or not np.isfinite(all17[position]).all() or not np.isfinite(all33[position]).all():
            excluded_dates.append(date.strftime("%Y-%m-%d"))
            continue
        snapshot_by_date[date] = snapshot_year
        included_dates.append(date)
    paired_dates = pd.DatetimeIndex(included_dates)
    eval_positions = market.index.get_indexer(paired_dates)
    if len(paired_dates) != 2759:
        raise ValueError(f"Expected same 2,759 paired dates as the source experiment; got {len(paired_dates)}")

    previous_forecasts = pd.read_csv(PREVIOUS_RESULT_DIR / "forecasts.csv", parse_dates=["trade_date"])
    previous_17 = (
        previous_forecasts.loc[previous_forecasts["variant"] == "direct17"]
        .pivot(index="trade_date", columns="ticker", values="forecast_residual")
        .reindex(index=paired_dates, columns=JP_TICKERS)
        .to_numpy(dtype=float)
    )
    previous_target = (
        previous_forecasts.loc[previous_forecasts["variant"] == "direct17"]
        .pivot(index="trade_date", columns="ticker", values="target_residual")
        .reindex(index=paired_dates, columns=JP_TICKERS)
        .to_numpy(dtype=float)
    )
    target = y17_res[eval_positions]
    if not np.isfinite(previous_17).all() or not np.isfinite(previous_target).all():
        raise ValueError("Saved direct17 forecasts have missing paired rows")
    if not np.allclose(previous_target, target, atol=1e-10, rtol=0.0):
        raise ValueError("Current ETF targets do not match the previous experiment's saved targets")

    current_labels = {
        industry: {
            factor: float(SENSITIVITY_LABELS[parent_tickers[industry_to_parent[industry]]][factor])
            for factor in ("w3", "w4", "w5", "w6")
        }
        for industry in industry_order
    }
    v0_33 = prior._build_v0_33(current_labels, industry_order)
    model33 = prior._make_model(
        app.v2.blpx,
        len(industry_order),
        "Direct33FixedParentLabelsMappingAblation",
        candidate_mapping=cap_maps[2014],
        industry_order=industry_order,
        industry_to_parent=industry_to_parent,
        parent_tickers=parent_tickers,
    )

    prior.BOOTSTRAP_BLOCK = BOOTSTRAP_BLOCK
    prior.BOOTSTRAP_RESAMPLES = BOOTSTRAP_RESAMPLES
    prior.BOOTSTRAP_SEED = BOOTSTRAP_SEED
    predictions: dict[str, np.ndarray] = {
        "direct17": previous_17,
    }
    mapped_realized: dict[str, np.ndarray] = {
        name: np.full((len(paired_dates), len(JP_TICKERS)), np.nan, dtype=float)
        for name in candidate_maps
    }
    covariance_stats = {
        name: {"psd_failures": 0, "minimum_eigenvalue": float("inf"), "mean_trace_sum": 0.0}
        for name in candidate_maps
    }
    map_predictions = {
        name: np.full((len(paired_dates), len(JP_TICKERS)), np.nan, dtype=float)
        for name in candidate_maps
    }
    raw_prediction_rows: list[dict[str, Any]] = []

    if args.preflight:
        check_position = int(eval_positions[0])
        result = model33.compute_blp_signal(
            all_returns=all33,
            current_index=check_position,
            gap_override=None,
            betas_t=None,
            topix_night_t=None,
            v0_static=v0_33,
            c_full=c_full_33,
            is_residual=True,
            return_matrices=True,
        )
        mu33, omega33 = build_raw_distribution(result, vol_adjusted_target=bool(model33.vol_adjusted_target))
        if np.asarray(mu33).shape != (33,) or np.asarray(omega33).shape != (33, 33):
            raise ValueError("Raw 33D preflight returned unexpected mean/covariance shapes")
        for name, by_year in candidate_maps.items():
            weight = by_year[snapshot_by_date[paired_dates[0]]]
            if not np.allclose(weight.sum(axis=1), 1.0, atol=1e-10, rtol=0.0):
                raise ValueError(f"{name} rows do not sum to one")
            projected = weight @ np.asarray(omega33) @ weight.T
            if not np.isfinite(projected).all():
                raise ValueError(f"{name} mapped covariance is not finite")
        LOG.info("Preflight passed for %d mapping variants", len(candidate_maps))
        return

    for row_i, current_index in enumerate(eval_positions):
        result = model33.compute_blp_signal(
            all_returns=all33,
            current_index=int(current_index),
            gap_override=None,
            betas_t=None,
            topix_night_t=None,
            v0_static=v0_33,
            c_full=c_full_33,
            is_residual=True,
            return_matrices=True,
        )
        mu33, omega33 = build_raw_distribution(result, vol_adjusted_target=bool(model33.vol_adjusted_target))
        mu33 = np.asarray(mu33, dtype=float)
        omega33 = np.asarray(omega33, dtype=float)
        if mu33.shape != (33,) or omega33.shape != (33, 33):
            raise ValueError(f"Unexpected raw 33D distribution on {paired_dates[row_i].date()}")
        if not np.isfinite(mu33).all() or not np.isfinite(omega33).all():
            raise ValueError(f"Non-finite raw 33D distribution on {paired_dates[row_i].date()}")
        date = paired_dates[row_i]
        snapshot_year = snapshot_by_date[date]
        raw_prediction_rows.append({"trade_date": date, **dict(zip(industry_order, mu33, strict=True))})
        for name, by_year in candidate_maps.items():
            weight = by_year[snapshot_year]
            map_predictions[name][row_i] = weight @ mu33
            mapped_realized[name][row_i] = weight @ y33_res[int(current_index)]
            mapped_covariance = weight @ omega33 @ weight.T
            mapped_covariance = 0.5 * (mapped_covariance + mapped_covariance.T)
            eigenvalues = np.linalg.eigvalsh(mapped_covariance)
            stats = covariance_stats[name]
            stats["psd_failures"] += int(float(eigenvalues.min()) < -1e-9)
            stats["minimum_eigenvalue"] = min(float(stats["minimum_eigenvalue"]), float(eigenvalues.min()))
            stats["mean_trace_sum"] += float(np.trace(mapped_covariance))
        if (row_i + 1) % 250 == 0 or row_i + 1 == len(eval_positions):
            LOG.info("Raw 33D forecasts %d/%d", row_i + 1, len(eval_positions))

    prior_33_parent = (
        previous_forecasts.loc[previous_forecasts["variant"] == "direct33_parent_labels"]
        .pivot(index="trade_date", columns="ticker", values="forecast_residual")
        .reindex(index=paired_dates, columns=JP_TICKERS)
        .to_numpy(dtype=float)
    )
    if not np.allclose(map_predictions["annual_cap"], prior_33_parent, atol=1e-9, rtol=0.0):
        max_diff = float(np.max(np.abs(map_predictions["annual_cap"] - prior_33_parent)))
        raise ValueError(f"Annual-cap remapping did not reproduce previous 33D forecasts; max delta={max_diff}")
    predictions.update({f"direct33_{name}": values for name, values in map_predictions.items()})

    daily_forecast = prior._daily_metrics(predictions, target, paired_dates)
    daily_forecast.to_csv(RESULTS_DIR / "daily_forecast_metrics.csv", index=False)
    daily_forecast.to_csv(RESULTS_DIR / "daily_metrics.csv", index=False)
    pd.DataFrame(raw_prediction_rows).to_csv(RESULTS_DIR / "raw_33d_forecasts.csv", index=False)

    forecast_rows = []
    for variant, values in predictions.items():
        for day_i, date in enumerate(paired_dates):
            for ticker_i, ticker in enumerate(JP_TICKERS):
                forecast_rows.append({
                    "trade_date": date.strftime("%Y-%m-%d"),
                    "variant": variant,
                    "ticker": ticker,
                    "forecast_residual": float(values[day_i, ticker_i]),
                    "target_residual": float(target[day_i, ticker_i]),
                    "snapshot_year": snapshot_by_date[date],
                })
    pd.DataFrame(forecast_rows).to_csv(RESULTS_DIR / "mapped_forecasts.csv", index=False)

    map_daily, map_ticker = _mapping_metrics(mapped_realized, target, paired_dates, list(JP_TICKERS))
    map_daily.to_csv(RESULTS_DIR / "daily_mapping_metrics.csv", index=False)
    map_ticker.to_csv(RESULTS_DIR / "mapping_by_etf.csv", index=False)
    map_summary = map_daily.groupby("mapping", sort=False).agg(
        mean_mae_bps=("mae_bps", "mean"), mean_rmse_bps=("rmse_bps", "mean")
    )
    ticker_summary = map_ticker.groupby("mapping", sort=False).agg(
        median_correlation=("correlation", "median"), min_correlation=("correlation", "min"),
        max_correlation=("correlation", "max"),
    )
    mapping_summary = map_summary.join(ticker_summary).reset_index()
    mapping_summary.to_csv(RESULTS_DIR / "mapping_summary.csv", index=False)

    weight_rows = []
    for name, by_year in candidate_maps.items():
        for snapshot_year, matrix in sorted(by_year.items()):
            for parent_i, parent in enumerate(parents):
                for industry_i, industry in enumerate(industry_order):
                    weight = float(matrix[parent_i, industry_i])
                    if weight > 0.0:
                        weight_rows.append({
                            "mapping": name, "snapshot_year": snapshot_year,
                            "parent": parent, "ticker": parent_tickers[parent],
                            "industry33": industry, "weight": weight,
                        })
    weights_frame = pd.DataFrame(weight_rows)
    weights_frame.to_csv(RESULTS_DIR / "mapping_weights.csv", index=False)
    (RESULTS_DIR / "fitted_weights_by_parent.json").write_text(
        json.dumps(fitted_weights, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    forecast_vs_direct17 = {
        name: prior._summarize(daily_forecast, f"direct33_{name}", baseline="direct17")
        for name in candidate_maps
    }
    forecast_vs_cap = {
        name: prior._summarize(daily_forecast, f"direct33_{name}", baseline="direct33_annual_cap")
        for name in candidate_maps if name != "annual_cap"
    }
    mapping_ci_vs_cap = {}
    for index, name in enumerate(candidate_maps):
        if name == "annual_cap":
            continue
        candidate_daily = map_daily.loc[map_daily["mapping"] == name].sort_values("trade_date")
        cap_daily = map_daily.loc[map_daily["mapping"] == "annual_cap"].sort_values("trade_date")
        diff = candidate_daily["mae_bps"].to_numpy() - cap_daily["mae_bps"].to_numpy()
        mapping_ci_vs_cap[name] = {
            "mean_mae_difference_bps": float(diff.mean()),
            "mae_difference_ci95_bps": _block_ci(diff, BOOTSTRAP_SEED + 10 + index),
        }

    covariance_summary = {}
    for name, values in covariance_stats.items():
        covariance_summary[name] = {
            "psd_failures": int(values["psd_failures"]),
            "minimum_eigenvalue": float(values["minimum_eigenvalue"]),
            "mean_trace": float(values["mean_trace_sum"] / len(paired_dates)),
        }

    previous_registry_records = list(ExperimentRegistry(REGISTRY_PATH))
    related_records = [
        item.name for item in previous_registry_records
        if any(word in item.name.lower() for word in ("jpx33", "subsector"))
    ]
    config_hash = hashlib.sha256(
        CONFIG_PATH.read_bytes() + LABEL_PATH.read_bytes() + PARENT_PATH.read_bytes() + CAP_PATH.read_bytes()
    ).hexdigest()
    market_fingerprint = dataframe_fingerprint(market)
    industry_fingerprint = dataframe_fingerprint(industry_panel)
    gate_results = {}
    for name in forecast_vs_cap:
        map_ci = mapping_ci_vs_cap[name]["mae_difference_ci95_bps"]
        forecast = forecast_vs_cap[name]
        gate_results[name] = bool(
            map_ci[1] < 0.0
            and forecast["rank_ic_difference_ci95"][0] > 0.0
            and forecast["mean_mae_difference_bps"] <= 0.0
        )

    summary = {
        "experiment_id": EXPERIMENT_ID,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": time.monotonic() - started_clock,
        "decision": "diagnostic only; no mapping promoted; historical evaluation period already inspected",
        "evaluation": {
            "start_date": str(paired_dates.min().date()),
            "end_date": str(paired_dates.max().date()),
            "paired_dates": int(len(paired_dates)),
            "fixed_baseline_period": [str(BASELINE_START.date()), str(BASELINE_END.date())],
            "baseline_fit_complete_rows": int(baseline_mask.sum()),
            "beta_window": beta_window,
            "bootstrap": {"method": "paired non-circular moving-block", "block_length": BOOTSTRAP_BLOCK,
                          "resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
            "excluded_evaluation_dates": excluded_dates,
            "target": "TOPIX-17 ETF open-to-close returns residualized against TOPIX open-to-close",
            "mapping_fit_target": "2010-2014 ETF residual returns, constrained simplex least squares per parent",
        },
        "mapping_variants": {
            "annual_cap": "prior-year JPX total industry market-cap share within parent; reference",
            "fixed_2014_cap": "2014 total market-cap shares frozen for all evaluation dates",
            "equal_child": "equal weights within children of each parent",
            "fit_2010_14_simplex": "nonnegative sum-to-one least-squares fit on 2010-2014 residual returns, frozen after 2014",
        },
        "mapping_realized_return_metrics": mapping_summary.to_dict(orient="records"),
        "mapping_error_delta_vs_annual_cap": mapping_ci_vs_cap,
        "forecast_vs_direct17": forecast_vs_direct17,
        "forecast_delta_vs_annual_cap": forecast_vs_cap,
        "diagnostic_gate_results_vs_annual_cap": gate_results,
        "covariance_projection_diagnostics": covariance_summary,
        "input_data": {
            "source_provenance": "user-supplied 33 OHLC CSVs; provider not independently authenticated",
            "industry_csv_sha256": date_alignment["source_data_sha256"],
            "industry_rows_min": int(data_quality["rows"].min()),
            "industry_rows_max": int(data_quality["rows"].max()),
            "complete_eval_dates": int(date_alignment["complete_33_series_evaluation_sessions"]),
            "date_alignment": date_alignment,
        },
        "data_fingerprints": {
            "df_exec": market_fingerprint,
            "industry_returns": industry_fingerprint,
            "config_and_mapping_inputs_sha256": config_hash,
        },
        "effective_blpx_config": app.v2.blpx.model_dump(mode="json"),
        "related_trial_count_before_this_run": len(related_records),
        "related_trial_names_before_this_run": related_records,
        "candidate_mapping_trials": 3,
        "all_results_retrospective_pseudo_oos": True,
        "portfolio_costs_and_pnl": "not evaluated",
        "production_modified": False,
    }
    (RESULTS_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    report_lines = [
        "# 33→17縮約だけを差し替えた長期比較", "", "## 判定", "",
        "33次元モデルの予測を固定して写像を比較した。年次総時価総額比を基準に、実績写像誤差と予測精度を分けて評価した。過去期間既知の診断なので、写像候補は本番採用しない。",
        "", "## 実験条件", "",
        f"- paired期間: {paired_dates.min().date()}〜{paired_dates.max().date()}、{len(paired_dates)}日。固定基準・写像学習は2010-01-01〜2014-12-31の{int(baseline_mask.sum())}完全行。評価開始後にウェイトを再学習しない。",
        "- 33D予測器は前回と同じ現行17業種感応度を子業種へ複製したBLPX。予測μ33・入力・c_full・対象日を固定し、μ17=Wμ33だけを候補間で変更。実績写像はy17_resとW y33_resを比較。Ω17=WΩ33W'のPSDとトレースも記録した。",
        "- 候補: 年次総時価総額比（基準）、2014年末比率固定、親業種内等ウェイト、2010〜2014年のETF残差リターンで学習した非負総和1ウェイト。fit候補は既知期間を使う擬似OOS診断のみ。",
        "- 差の95%区間は同日対応を保つ非循環20営業日moving-block bootstrap 5,000回。データ提供元は未認証。",
        "", "## 実績33→17写像の精度", "",
        "| 写像 | 実績集約MAE (bp) | 差 vs 年次cap (95% CI, bp) | 実績集約RMSE (bp) | ETF別相関中央値 | 最小〜最大 |", "|---|---:|---:|---:|---:|---:|",
    ]
    map_lookup = {str(row["mapping"]): row for row in mapping_summary.to_dict(orient="records")}
    for name, label in (("annual_cap", "前年総時価総額比"), ("fixed_2014_cap", "2014年cap固定"),
                        ("equal_child", "親内等ウェイト"), ("fit_2010_14_simplex", "2010-14残差fit")):
        row = map_lookup[name]
        delta = "—" if name == "annual_cap" else mapping_ci_vs_cap[name]
        delta_text = "—" if name == "annual_cap" else (
            f"{delta['mean_mae_difference_bps']:+.2f} "
            f"[{delta['mae_difference_ci95_bps'][0]:+.2f}, {delta['mae_difference_ci95_bps'][1]:+.2f}]"
        )
        report_lines.append(
            f"| {label} | {row['mean_mae_bps']:.2f} | {delta_text} | {row['mean_rmse_bps']:.2f} "
            f"| {row['median_correlation']:.3f} | {row['min_correlation']:.3f}〜{row['max_correlation']:.3f} |"
        )
    report_lines.extend([
        "", "## 予測精度への影響", "",
        "| 写像 | Rank IC | Δ vs direct17 (95% CI) | Δ vs 年次cap (95% CI) | MAE (bp) | ΔMAE vs direct17 (bp) | 方向正答率 |", "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for name, label in (("annual_cap", "前年総時価総額比"), ("fixed_2014_cap", "2014年cap固定"),
                        ("equal_child", "親内等ウェイト"), ("fit_2010_14_simplex", "2010-14残差fit")):
        direct = forecast_vs_direct17[name]
        rank_ci = direct["rank_ic_difference_ci95"]
        cap_delta = None if name == "annual_cap" else forecast_vs_cap[name]
        cap_text = "—" if cap_delta is None else (
            f"{cap_delta['mean_rank_ic_difference']:+.4f} "
            f"[{cap_delta['rank_ic_difference_ci95'][0]:+.4f}, {cap_delta['rank_ic_difference_ci95'][1]:+.4f}]"
        )
        report_lines.append(
            f"| {label} | {direct['candidate_mean_rank_ic']:+.4f} "
            f"| {direct['mean_rank_ic_difference']:+.4f} [{rank_ci[0]:+.4f}, {rank_ci[1]:+.4f}] "
            f"| {cap_text} | {direct['candidate_mean_mae_bps']:.2f} "
            f"| {direct['mean_mae_difference_bps']:+.2f} | {direct['candidate_sign_accuracy']:.3f} |"
        )
    report_lines.extend([
        "", "## 解釈・制約", "",
        "- 実績集約誤差と予測精度は別の層として集計した。写像の実績誤差が改善しても、固定μ33のRank ICが改善しなければ、縮約だけで前回の予測劣化を説明したとは言えない。",
        "- 完全な採否ゲートは、代替写像の実績MAE改善CI上限<0、年次cap比Rank IC改善CI下限>0、予測MAE非悪化の同時成立。全候補の結果とゲートはsummary.jsonに記録。",
        "- ETFの始値→終値ターゲットと33入力CSVの提供元・指数仕様は独立認証されていない。公式TOPIX-17親指数へのトラッキングは未測定。",
        "- 予測精度診断であり、実ポジション、PnL、売買費用、Sharpe、未使用OOSは評価していない。Ω写像はPSD計算を確認したが、リスク配分価値は未評価。production設定は変更していない。",
        "", "## 再現情報", "",
        f"- 設定: `{CONFIG_PATH.relative_to(ROOT)}`。スクリプト: `{Path(__file__).relative_to(ROOT)}`。結果: `{RESULTS_DIR.relative_to(ROOT)}/`。",
        f"- 33 CSV SHA-256: `{date_alignment['source_data_sha256']}`。df_exec fingerprint: `{market_fingerprint}`。設定・写像入力SHA-256: `{config_hash}`。",
        f"- 既登録JPX33/サブセクターrecord: {len(related_records)}件、今回の代替写像候補: 3件。費用後PnLを選択する試行ではないためDSR対象外。",
        f"- 実行: `timeout -k 20 14400 .venv/bin/python -u {Path(__file__).relative_to(ROOT)}`。",
    ])
    report_path = REPORT_DIR / "report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    registry = ExperimentRegistry(REGISTRY_PATH)
    for name, description in (
        ("fixed_2014_cap", "2014 year-end total market-cap shares frozen for all eval dates"),
        ("equal_child", "equal child-industry weights within each TOPIX-17 parent"),
        ("fit_2010_14_simplex", "nonnegative simplex weights fit on 2010-2014 residual returns"),
    ):
        forecast_metric = forecast_vs_cap[name]
        mapping_metric = mapping_ci_vs_cap[name]
        passed = gate_results[name]
        registry.record(ExperimentRecord(
            name=f"{EXPERIMENT_ID}_{name}",
            hypothesis="Changing only the output aggregation improves 33D forecast accuracy and realized mapping fidelity versus prior-year total-cap weights.",
            start_time=started_at,
            end_time=datetime.now(UTC),
            parameters={
                "mapping_variant": name,
                "description": description,
                "target_period": [str(paired_dates.min().date()), str(paired_dates.max().date())],
                "fit_period": [str(BASELINE_START.date()), str(BASELINE_END.date())],
                "mapping_mae_difference_vs_annual_cap_ci95_bps": mapping_metric["mae_difference_ci95_bps"],
                "forecast_rank_ic_difference_vs_annual_cap_ci95": forecast_metric["rank_ic_difference_ci95"],
                "forecast_mae_difference_vs_annual_cap_bps": forecast_metric["mean_mae_difference_bps"],
                "source_data_sha256": date_alignment["source_data_sha256"],
                "config_sha256": config_hash,
                "related_trial_count_before_this_run": len(related_records),
                "related_prior_trial_names": related_records,
                "retrospective_pseudo_oos_only": True,
                "production_modified": False,
            },
            metrics={
                "forecast_vs_direct17": forecast_vs_direct17[name],
                "forecast_vs_annual_cap": forecast_metric,
                "mapping_error_delta_vs_annual_cap": mapping_metric,
                "diagnostic_gate_passed": passed,
                "cost_adjusted_pnl": "not evaluated",
            },
            decision=Decision.PENDING if passed else Decision.REJECTED,
            report_path=str(report_path.relative_to(ROOT)),
            related_records=related_records,
        ))
    LOG.info("Wrote report=%s", report_path)
    LOG.info("Wrote results=%s", RESULTS_DIR)


if __name__ == "__main__":
    main()
