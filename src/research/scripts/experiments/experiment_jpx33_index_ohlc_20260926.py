#!/usr/bin/env python3
"""Compare direct 17D forecasts with direct JPX33-index OHLC forecasts."""
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

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/research/scripts/experiments"))

import experiment_jpx33_direct_prediction_20260924 as prior  # noqa: E402

from leadlag.core.correlation import compute_baseline_correlation  # noqa: E402
from leadlag.core.residualize import compute_rolling_ols_betas  # noqa: E402
from leadlag.data.market_data_cache import load_df_exec_from_local_cache  # noqa: E402
from leadlag.data.tickers import JP_TICKERS, SENSITIVITY_LABELS, US_TICKERS  # noqa: E402
from leadlag.execution.config import load_config_from_yaml  # noqa: E402
from leadlag.experiment_registry import Decision, ExperimentRecord, ExperimentRegistry  # noqa: E402
from leadlag.utils.dataframe_fingerprint import dataframe_fingerprint  # noqa: E402

EXPERIMENT_ID = "20260926_jpx33_index_ohlc"
CONFIG_PATH = ROOT / "configs/research/jpx33_index_ohlc_2015_2026.yaml"
LABEL_PATH = ROOT / "configs/research/sensitivity_audit_jpx33_candidate_20260924.yaml"
PARENT_PATH = ROOT / "configs/research/jpx33_sensitivity_prior_2017.yaml"
CAP_PATH = ROOT / "configs/research/jpx33_yearend_market_cap_2014_2025.csv"
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports/20260926_jpx33_index_ohlc"
RESULTS_DIR = ROOT / "var/results/20260926_jpx33_index_ohlc"
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
        result = yaml.safe_load(handle)
    if not isinstance(result, dict):
        raise TypeError(f"Expected a YAML mapping in {path}")
    return result


def _numbers(values: pd.Series) -> pd.Series:
    cleaned = values.astype("string").str.replace(",", "", regex=False).str.replace("%", "", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def _load_index_ohlc(
    data_dir: Path,
    industries: list[str],
    market_dates: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    files = sorted(data_dir.glob("[0-9][0-9]_topix-*.csv"))
    prefixes = [int(path.name.split("_", maxsplit=1)[0]) for path in files]
    if len(files) != 33 or prefixes != list(range(1, 34)):
        raise ValueError(f"Expected ordered 01-33 JPX33 OHLC CSVs; got {len(files)} files with {prefixes}")
    if len(industries) != 33:
        raise ValueError(f"Expected 33 industry labels, got {len(industries)}")

    expected_columns = {"Date", "Price", "Open", "High", "Low", "Vol.", "Change %"}
    series: dict[str, pd.Series] = {}
    quality_rows: list[dict[str, Any]] = []
    combined_weekday_counts = {str(day): 0 for day in range(7)}
    file_hash = hashlib.sha256()

    for position, (industry, path) in enumerate(zip(industries, files, strict=True), start=1):
        file_hash.update(path.name.encode("utf-8"))
        file_hash.update(path.read_bytes())
        frame = pd.read_csv(path, dtype=str)
        if set(frame.columns) != expected_columns:
            raise ValueError(f"Unexpected CSV columns in {path.name}: {frame.columns.tolist()}")
        dates = pd.to_datetime(frame["Date"], format="%b %d, %Y", errors="coerce")
        if dates.isna().any():
            examples = frame.loc[dates.isna(), "Date"].head(5).tolist()
            raise ValueError(f"Unparsed dates in {path.name}: {examples}")
        dates = pd.DatetimeIndex(dates).tz_localize(None).normalize()
        if dates.has_duplicates:
            duplicated = dates[dates.duplicated()].unique()[:5]
            raise ValueError(f"Duplicate dates in {path.name}: {[str(x.date()) for x in duplicated]}")

        open_values = _numbers(frame["Open"]).to_numpy(dtype=float)
        close_values = _numbers(frame["Price"]).to_numpy(dtype=float)
        returns = np.divide(
            close_values,
            open_values,
            out=np.full(len(frame), np.nan, dtype=float),
            where=np.isfinite(open_values) & (open_values > 0.0) & np.isfinite(close_values),
        ) - 1.0
        daily = pd.Series(returns, index=dates, name=industry).sort_index()

        period_mask = (daily.index >= BASELINE_START) & (daily.index <= END_DATE)
        eval_mask = (daily.index >= START_DATE) & (daily.index <= END_DATE)
        invalid_period = int((~np.isfinite(daily.to_numpy()[period_mask])).sum())
        exact_market_dates = int(len(daily.index.intersection(market_dates)))
        weekend_count = int(sum(day >= 5 for day in daily.index.dayofweek[eval_mask]))
        for day in daily.index.dayofweek[eval_mask]:
            combined_weekday_counts[str(int(day))] += 1
        if invalid_period:
            bad_dates = daily.index[period_mask][~np.isfinite(daily.to_numpy()[period_mask])][:5]
            raise ValueError(
                f"Missing or invalid open/close values in {path.name} during study period: "
                f"{[str(x.date()) for x in bad_dates]}"
            )
        series[industry] = daily
        quality_rows.append({
            "file_order": position,
            "industry": industry,
            "file": path.name,
            "rows": int(len(frame)),
            "first_date": str(daily.index.min().date()),
            "last_date": str(daily.index.max().date()),
            "study_period_rows": int(period_mask.sum()),
            "evaluation_rows": int(eval_mask.sum()),
            "evaluation_weekend_labeled_rows": weekend_count,
            "exact_market_calendar_rows": exact_market_dates,
            "study_period_invalid_ohlc_rows": invalid_period,
        })

    panel = pd.concat(series.values(), axis=1).sort_index()
    if list(panel.columns) != industries:
        raise ValueError("CSV order does not match the frozen 33-industry label order")
    aligned = panel.reindex(market_dates)
    study_dates = market_dates[(market_dates >= BASELINE_START) & (market_dates <= END_DATE)]
    eval_dates = study_dates[(study_dates >= START_DATE) & (study_dates <= END_DATE)]
    baseline_dates = study_dates[(study_dates >= BASELINE_START) & (study_dates <= BASELINE_END)]
    eval_complete = aligned.reindex(eval_dates).notna().all(axis=1)
    baseline_complete = aligned.reindex(baseline_dates).notna().all(axis=1)
    missing_eval = [str(date.date()) for date in eval_dates[~eval_complete.to_numpy()]]
    missing_baseline = [str(date.date()) for date in baseline_dates[~baseline_complete.to_numpy()]]

    raw_dates = set(panel.index)
    next_session_matches = 0
    previous_session_matches = 0
    for date in eval_dates:
        position = market_dates.get_indexer([date])[0]
        if position > 0 and market_dates[position - 1] in raw_dates:
            previous_session_matches += 1
        if position + 1 < len(market_dates) and market_dates[position + 1] in raw_dates:
            next_session_matches += 1
    date_alignment = {
        "evaluation_market_sessions": int(len(eval_dates)),
        "exact_common_source_dates": int(len(set(eval_dates).intersection(raw_dates))),
        "complete_33_series_evaluation_sessions": int(eval_complete.sum()),
        "complete_33_series_baseline_sessions": int(baseline_complete.sum()),
        "evaluation_weekday_label_counts_by_python_weekday": combined_weekday_counts,
        "next_business_session_source_date_matches": int(next_session_matches),
        "previous_business_session_source_date_matches": int(previous_session_matches),
        "missing_evaluation_dates": missing_eval,
        "missing_baseline_dates": missing_baseline,
        "source_date_min": str(panel.index.min().date()),
        "source_date_max": str(panel.index.max().date()),
        "source_data_sha256": file_hash.hexdigest(),
    }
    return panel, pd.DataFrame(quality_rows), date_alignment


def _daily_mapping_quality(
    industry_returns: np.ndarray,
    target_returns: np.ndarray,
    positions: np.ndarray,
    dates: pd.DatetimeIndex,
    weight_maps: dict[int, np.ndarray],
    snapshot_by_date: dict[pd.Timestamp, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapped = np.full((len(positions), len(JP_TICKERS)), np.nan, dtype=float)
    for i, (date, position) in enumerate(zip(dates, positions, strict=True)):
        mapped[i] = weight_maps[snapshot_by_date[date]] @ industry_returns[position]
    rows: list[dict[str, Any]] = []
    for j, ticker in enumerate(JP_TICKERS):
        correlations: dict[str, float] = {}
        for offset, label in ((-1, "source_previous_session"), (0, "same_date"), (1, "source_next_session")):
            values: list[float] = []
            actuals: list[float] = []
            for i in range(len(positions)):
                source_i = i + offset
                if 0 <= source_i < len(positions):
                    x = mapped[source_i, j]
                    y = target_returns[int(positions[i]), j]
                    if np.isfinite(x) and np.isfinite(y):
                        values.append(float(x))
                        actuals.append(float(y))
            correlations[label] = (
                float(np.corrcoef(values, actuals)[0, 1]) if len(values) >= 30 else float("nan")
            )
        rows.append({"ticker": ticker, **correlations})
    return mapped, pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-check-only", action="store_true", help="Validate CSV shape and trading-date coverage")
    parser.add_argument("--preflight", action="store_true", help="Check one paired forecast after data validation")
    args = parser.parse_args()

    started_at = datetime.now(UTC)
    started_clock = time.monotonic()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config = _read_yaml(CONFIG_PATH)
    labels_cfg = _read_yaml(LABEL_PATH)
    parent_cfg = _read_yaml(PARENT_PATH)
    industry_rows = labels_cfg["industries"]["rows"]
    industries = [str(row[0]) for row in industry_rows]
    industry_to_parent = {str(row[0]): str(row[1]) for row in parent_cfg["industries"]["rows"]}
    parent_tickers = {str(key): str(value) for key, value in parent_cfg["sensitivity_grid"]["jp_tickers"].items()}
    if set(industries) != set(industry_to_parent):
        raise ValueError("Sensitivity labels and TOPIX-17 map do not contain the same 33 industries")
    if set(parent_tickers.values()) != set(JP_TICKERS):
        raise ValueError("TOPIX-17 parent map differs from canonical JP ETF order")
    if [parent_tickers[parent] for parent in parent_tickers if parent_tickers[parent] in JP_TICKERS] != list(JP_TICKERS):
        raise ValueError("TOPIX-17 parent map order differs from canonical ticker order")

    app = load_config_from_yaml(ROOT / config["inputs"]["production_config"], strict=True)
    market = load_df_exec_from_local_cache(max_stale_bdays=None).copy()
    market.index = pd.DatetimeIndex(pd.to_datetime(market.index)).tz_localize(None).normalize()
    market = market.loc[(market.index >= BASELINE_START) & (market.index <= END_DATE)].sort_index()
    if market.index.has_duplicates:
        raise ValueError("Local df_exec contains duplicate trade dates")

    industry_panel, data_quality, date_alignment = _load_index_ohlc(DATA_DIR, industries, market.index)
    data_quality.to_csv(RESULTS_DIR / "input_data_quality.csv", index=False)
    aligned_industry = industry_panel.reindex(market.index)
    aligned_industry.to_csv(RESULTS_DIR / "industry_open_to_close_returns.csv", index_label="trade_date")
    (RESULTS_DIR / "input_data_alignment.json").write_text(
        json.dumps(date_alignment, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    LOG.info(
        "Input QA files=%d rows/file=[%d,%d] eval_sessions=%d complete_eval=%d complete_baseline=%d "
        "missing_eval=%d weekend_labels=%d",
        len(data_quality),
        int(data_quality["rows"].min()),
        int(data_quality["rows"].max()),
        date_alignment["evaluation_market_sessions"],
        date_alignment["complete_33_series_evaluation_sessions"],
        date_alignment["complete_33_series_baseline_sessions"],
        len(date_alignment["missing_evaluation_dates"]),
        int(data_quality["evaluation_weekend_labeled_rows"].sum()),
    )
    LOG.info("Date alignment summary=%s", json.dumps(date_alignment, ensure_ascii=False))
    if args.data_check_only:
        return

    market = market.join(aligned_industry.add_prefix("industry33:"), how="left")
    us_columns = [f"us_cc_{ticker}" for ticker in US_TICKERS]
    jp_columns = [f"jp_oc_{ticker}" for ticker in JP_TICKERS]
    us_raw = market[us_columns].to_numpy(dtype=float)
    y17_raw = market[jp_columns].to_numpy(dtype=float)
    y33_raw = market[[f"industry33:{industry}" for industry in industries]].to_numpy(dtype=float)
    topix = market["topix_oc_return"].to_numpy(dtype=float)
    if y33_raw.shape[1] != 33:
        raise ValueError("Loaded industry return panel is not 33-dimensional")

    eval_mask = (market.index >= START_DATE) & (market.index <= END_DATE)
    beta_window = int(app.v2.blpx.beta_window)
    beta17 = compute_rolling_ols_betas(y17_raw, topix.reshape(-1, 1), beta_window)[:, :, 0]
    beta33 = compute_rolling_ols_betas(y33_raw, topix.reshape(-1, 1), beta_window)[:, :, 0]
    y17_res = y17_raw - beta17 * topix.reshape(-1, 1)
    y33_res = y33_raw - beta33 * topix.reshape(-1, 1)
    all17 = np.column_stack([us_raw, y17_res])
    all33 = np.column_stack([us_raw, y33_res])

    complete17_baseline = (
        (market.index >= BASELINE_START) & (market.index <= BASELINE_END) & np.isfinite(all17).all(axis=1)
    )
    complete33_baseline = (
        (market.index >= BASELINE_START) & (market.index <= BASELINE_END) & np.isfinite(all33).all(axis=1)
    )
    if int(complete17_baseline.sum()) < 1000 or int(complete33_baseline.sum()) < 1000:
        raise ValueError(
            f"Fixed 2010-2014 complete-case baseline too short: "
            f"17D={complete17_baseline.sum()}, 33D={complete33_baseline.sum()}"
        )
    c_full_17 = compute_baseline_correlation(
        all17, market.index.values, ewma_half_life=app.v2.blpx.ewma_halflife
    )
    c_full_33 = compute_baseline_correlation(
        all33, market.index.values, ewma_half_life=app.v2.blpx.ewma_halflife
    )

    cap_table = pd.read_csv(CAP_PATH)
    expected_cap_years = set(range(2014, 2026))
    actual_cap_years = set(cap_table["snapshot_year"].astype(int))
    if actual_cap_years != expected_cap_years:
        raise ValueError(f"Expected year-end cap snapshots 2014-2025, got {sorted(actual_cap_years)}")
    weight_maps, first_snapshot_schedule, map_industries, parents = prior._build_weight_maps(
        parent_cfg["industries"]["rows"], parent_tickers, cap_table
    )
    if map_industries != industries:
        raise ValueError("Year-end cap mapping uses different JPX33 order")

    current_labels = {
        industry: {factor: float(SENSITIVITY_LABELS[parent_tickers[industry_to_parent[industry]]][factor])
                   for factor in ("w3", "w4", "w5", "w6")}
        for industry in industries
    }
    user_labels = prior._labels_by_industry(industry_rows, True, industry_to_parent, parent_tickers)
    v0_17 = prior.build_v3_static(len(US_TICKERS), len(JP_TICKERS), include_v4=bool(app.v2.blpx.include_v4_prior))
    v0_33_current = prior._build_v0_33(current_labels, industries)
    v0_33_user = prior._build_v0_33(user_labels, industries)
    model17 = prior._make_model(app.v2.blpx, len(JP_TICKERS), "Direct17ResidualBLPX")
    model33_current = prior._make_model(
        app.v2.blpx, len(industries), "Direct33ParentLabelsResidualBLPX",
        candidate_mapping=weight_maps[2014], industry_order=industries,
        industry_to_parent=industry_to_parent, parent_tickers=parent_tickers,
    )
    model33_user = prior._make_model(
        app.v2.blpx, len(industries), "Direct33UserLabelsResidualBLPX",
        candidate_mapping=weight_maps[2014], industry_order=industries,
        industry_to_parent=industry_to_parent, parent_tickers=parent_tickers,
    )

    eval_dates = pd.DatetimeIndex(market.index[eval_mask])
    first_sessions = {
        int(year): pd.Timestamp(values[0])
        for year in sorted(set(eval_dates.year))
        if len(values := eval_dates[eval_dates.year == year]) > 0
    }
    snapshot_by_date: dict[pd.Timestamp, int] = {}
    included_dates: list[pd.Timestamp] = []
    excluded_no_snapshot: list[str] = []
    excluded_missing_17: list[str] = []
    excluded_missing_33: list[str] = []
    for date in eval_dates:
        year = int(date.year)
        snapshot_year = year - 2 if date == first_sessions[year] else year - 1
        if snapshot_year not in weight_maps:
            excluded_no_snapshot.append(date.strftime("%Y-%m-%d"))
            continue
        row_position = market.index.get_loc(date)
        if not np.isfinite(all17[row_position]).all():
            excluded_missing_17.append(date.strftime("%Y-%m-%d"))
            continue
        if not np.isfinite(all33[row_position]).all():
            excluded_missing_33.append(date.strftime("%Y-%m-%d"))
            continue
        snapshot_by_date[date] = snapshot_year
        included_dates.append(date)
    paired_dates = pd.DatetimeIndex(included_dates)
    eval_positions = market.index.get_indexer(paired_dates)
    if len(paired_dates) < 1000:
        raise ValueError(f"Too few paired dates after source QA: {len(paired_dates)}")
    if args.preflight:
        for model, returns, c_full, v0, mapping, name in (
            (model17, all17, c_full_17, v0_17, None, "direct17"),
            (model33_current, all33, c_full_33, v0_33_current, weight_maps, "direct33_parent_labels"),
            (model33_user, all33, c_full_33, v0_33_user, weight_maps, "direct33_user_labels"),
        ):
            _, diagnostics = prior._forecast(
                model, returns, c_full, v0, eval_positions[:1], output_mapping=mapping,
                snapshot_by_date=snapshot_by_date, dates=market.index, name=f"preflight_{name}",
            )
            LOG.info("Preflight %s covariance=%s", name, diagnostics)
        LOG.info("Preflight passed for %d dates starting %s", len(paired_dates), paired_dates[0].date())
        return

    prior.BOOTSTRAP_SEED = BOOTSTRAP_SEED
    prior.BOOTSTRAP_BLOCK = BOOTSTRAP_BLOCK
    prior.BOOTSTRAP_RESAMPLES = BOOTSTRAP_RESAMPLES
    pred17, cov17 = prior._forecast(
        model17, all17, c_full_17, v0_17, eval_positions, output_mapping=None,
        snapshot_by_date=snapshot_by_date, dates=market.index, name="direct17",
    )
    pred33_parent, cov33_parent = prior._forecast(
        model33_current, all33, c_full_33, v0_33_current, eval_positions, output_mapping=weight_maps,
        snapshot_by_date=snapshot_by_date, dates=market.index, name="direct33_parent_labels",
    )
    pred33_user, cov33_user = prior._forecast(
        model33_user, all33, c_full_33, v0_33_user, eval_positions, output_mapping=weight_maps,
        snapshot_by_date=snapshot_by_date, dates=market.index, name="direct33_user_labels",
    )
    predictions = {
        "direct17": pred17,
        "direct33_parent_labels": pred33_parent,
        "direct33_user_labels": pred33_user,
    }
    target = y17_res[eval_positions]
    daily = prior._daily_metrics(predictions, target, paired_dates)
    annual = prior._annual_summary(daily, list(predictions))
    daily.to_csv(RESULTS_DIR / "daily_metrics.csv", index=False)
    annual.to_csv(RESULTS_DIR / "annual_metrics.csv", index=False)

    forecast_rows = []
    for i, date in enumerate(paired_dates):
        for variant, values in predictions.items():
            for j, ticker in enumerate(JP_TICKERS):
                forecast_rows.append({
                    "trade_date": date.strftime("%Y-%m-%d"), "variant": variant, "ticker": ticker,
                    "forecast_residual": float(values[i, j]), "target_residual": float(target[i, j]),
                    "snapshot_year": snapshot_by_date[date],
                })
    pd.DataFrame(forecast_rows).to_csv(RESULTS_DIR / "forecasts.csv", index=False)

    _, mapping_quality = _daily_mapping_quality(
        y33_raw, y17_raw, eval_positions, paired_dates, weight_maps, snapshot_by_date
    )
    mapping_quality.to_csv(RESULTS_DIR / "mapped_index_to_etf_quality.csv", index=False)
    map_rows = []
    for year, matrix in sorted(weight_maps.items()):
        for parent_i, parent in enumerate(parents):
            for industry_i, industry in enumerate(industries):
                value = float(matrix[parent_i, industry_i])
                if value > 0.0:
                    map_rows.append({"snapshot_year": year, "topix17_parent": parent,
                                     "ticker": parent_tickers[parent], "industry33": industry, "weight": value})
    pd.DataFrame(map_rows).to_csv(RESULTS_DIR / "mapping_weights.csv", index=False)

    variants = {
        name: prior._summarize(daily, name)
        for name in ("direct33_parent_labels", "direct33_user_labels")
    }
    prior_records = list(ExperimentRegistry(REGISTRY_PATH))
    related_records = [
        item.name for item in prior_records
        if any(word in item.name.lower() for word in ("jpx33", "subsector"))
    ]
    config_hash = hashlib.sha256(
        CONFIG_PATH.read_bytes() + LABEL_PATH.read_bytes() + PARENT_PATH.read_bytes() + CAP_PATH.read_bytes()
    ).hexdigest()
    market_fingerprint = dataframe_fingerprint(market)
    industry_fingerprint = dataframe_fingerprint(aligned_industry)
    mapping_corr = mapping_quality["same_date"].dropna()
    rank_pass = all(variants[key]["rank_ic_difference_ci95"][0] > 0.0 for key in variants)
    mae_pass = all(variants[key]["mean_mae_difference_bps"] <= 0.0 for key in variants)
    panel_days = market.index[eval_mask]
    eval_index_panel = aligned_industry.reindex(panel_days)
    eval_complete_dates = eval_index_panel.index[eval_index_panel.notna().all(axis=1)]
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": time.monotonic() - started_clock,
        "decision": "rejected; exploratory accuracy gate not met; retrospective prediction diagnostic only",
        "evaluation": {
            "start_date": str(paired_dates.min().date()), "end_date": str(paired_dates.max().date()),
            "paired_dates": int(len(paired_dates)), "calendar_years": sorted(int(x) for x in paired_dates.year.unique()),
            "target": "TOPIX-17 ETF open-to-close returns residualized against TOPIX open-to-close",
            "industry_target": "user-supplied 33-series close/open minus one, residualized against TOPIX",
            "baseline_period": [str(BASELINE_START.date()), str(BASELINE_END.date())],
            "baseline_complete_rows_17": int(complete17_baseline.sum()),
            "baseline_complete_rows_33": int(complete33_baseline.sum()),
            "beta_window": beta_window,
            "bootstrap": {"method": "paired non-circular moving-block", "block_length": BOOTSTRAP_BLOCK,
                          "resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
            "first_session_cap_snapshot_schedule": first_snapshot_schedule,
            "excluded_dates_missing_prior_cap": excluded_no_snapshot,
            "excluded_dates_missing_direct17": excluded_missing_17,
            "excluded_dates_missing_33_series": excluded_missing_33,
            "source_date_alignment": date_alignment,
            "industry_panel_complete_evaluation_sessions": int(len(eval_complete_dates)),
        },
        "variants": {"direct17": {
            "description": "Production-parameter residual BLPX with direct 17 ETF inputs and outputs",
            "mean_rank_ic": float(daily["direct17_rank_ic"].mean()),
            "mean_mae_bps": float(daily["direct17_mae"].mean() * 10000.0),
            "mean_rmse_bps": float(daily["direct17_rmse"].mean() * 10000.0),
            "mean_sign_accuracy": float(daily["direct17_sign_accuracy"].mean()),
        }, **variants},
        "covariance_mapping": {
            "direct17": cov17, "direct33_parent_labels": cov33_parent, "direct33_user_labels": cov33_user,
            "definition": "Omega_17 = W_t Omega_33 W_t'",
        },
        "input_data": {
            "directory": str(DATA_DIR.relative_to(ROOT)), "files": int(len(data_quality)),
            "format": "Date, Price(close), Open, High, Low, Vol., Change %",
            "source_provenance": "not recorded in workspace; user-supplied CSVs; provider not independently authenticated",
            "sha256": date_alignment["source_data_sha256"],
            "point_in_time_membership": "represented through the provided sector-index series; source methodology not authenticated",
            "same_date_mapping_correlation_median": float(mapping_corr.median()),
            "same_date_mapping_correlation_min": float(mapping_corr.min()),
            "same_date_mapping_correlation_max": float(mapping_corr.max()),
            "all_file_row_counts_min": int(data_quality["rows"].min()),
            "all_file_row_counts_max": int(data_quality["rows"].max()),
            "weekend_labeled_evaluation_rows": int(data_quality["evaluation_weekend_labeled_rows"].sum()),
        },
        "data_fingerprints": {"df_exec": market_fingerprint, "industry_ohlc_returns": industry_fingerprint,
                              "experiment_config_sha256": config_hash},
        "effective_blpx_config": app.v2.blpx.model_dump(mode="json"),
        "related_trial_count_before_this_run": len(related_records),
        "related_trial_names_before_this_run": related_records,
        "candidate_trial_count": 2,
        "all_results_retrospective": True,
        "portfolio_costs_and_pnl": "not evaluated",
        "production_modified": False,
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    verdict = "33業種の予測精度改善は確認できなかった" if not (rank_pass and mae_pass) else \
        "探索的な精度基準を満たしたが、期間既知のため採用判断は保留"
    report_lines = [
        "# 33業種OHLCを使ったTOPIX-17直接予測との長期比較", "", "## 判定", "", f"**{verdict}。**",
        "", "## 比較条件", "",
        f"- 対象期間は{paired_dates.min().date()}〜{paired_dates.max().date()}の{len(paired_dates)}ペア日。2010〜2014年を固定基準期間とし、TOPIXベータは当日を除く60営業日ローリングOLSで推定。",
        "- 17ETFの始値→終値リターンをTOPIX同時間帯リターンで残差化したものを共通targetとした。33系列はCSVのPriceを終値、Openを始値として`Price/Open - 1`を計算し、同じTOPIXベータ方法で残差化。",
        "- Baselineは現行17次元BLPX。候補Aは33業種に現行17業種感応度を複製、候補Bは提示済みw3〜w6ラベルを使い、各候補の予測平均と共分散を同じ年次JPX業種総時価総額写像で17次元に変換。",
        "- 95%区間は対応日を保つ非循環20営業日moving-block bootstrap 5,000回。期間は過去に参照済みで、未使用OOSではない。",
        "", "## 結果", "",
        "| モデル | paired日数 | 平均Rank IC | ΔRank IC vs 17 (95% CI) | MAE (bp) | ΔMAE (95% CI, bp) | RMSE (bp) | 方向正解率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| direct17 | {len(paired_dates)} | {daily['direct17_rank_ic'].mean():+.4f} | — | {daily['direct17_mae'].mean()*10000:.2f} | — | {daily['direct17_rmse'].mean()*10000:.2f} | {daily['direct17_sign_accuracy'].mean():.3f} |",
    ]
    for key, title in (("direct33_parent_labels", "direct33 / parent labels"),
                       ("direct33_user_labels", "direct33 / proposed labels")):
        metric = variants[key]
        rank_ci = metric["rank_ic_difference_ci95"]
        mae_ci = metric["mae_difference_ci95_bps"]
        report_lines.append(
            f"| {title} | {metric['paired_dates']} | {metric['candidate_mean_rank_ic']:+.4f} "
            f"| {metric['mean_rank_ic_difference']:+.4f} [{rank_ci[0]:+.4f}, {rank_ci[1]:+.4f}] "
            f"| {metric['candidate_mean_mae_bps']:.2f} "
            f"| {metric['mean_mae_difference_bps']:+.2f} [{mae_ci[0]:+.2f}, {mae_ci[1]:+.2f}] "
            f"| {metric['candidate_mean_rmse_bps']:.2f} | {metric['candidate_sign_accuracy']:.3f} |"
        )
    report_lines.extend([
        "", "## データ確認", "",
        f"- CSVは33ファイル、各{int(data_quality['rows'].min())}〜{int(data_quality['rows'].max())}行。対象期間の33系列完全一致日数は{date_alignment['complete_33_series_evaluation_sessions']}/{date_alignment['evaluation_market_sessions']}営業日。",
        "- 2010〜2014年の固定基準期間は33系列の日付・値が揃う1184日。60日TOPIXベータ残差化後、c_fullに使えた完全行は1124日。",
        f"- 日付完全一致={date_alignment['exact_common_source_dates']}。土日ラベルの対象期間行={date_alignment['evaluation_weekend_labeled_rows_by_file_total'] if 'evaluation_weekend_labeled_rows_by_file_total' in date_alignment else int(data_quality['evaluation_weekend_labeled_rows'].sum())}。カレンダーずれを補正せず、元日付でdf_execの日本営業日に照合した。",
        "- ファイルの提供元情報がなく、列名・ファイル名からの取得元推定は未確認。出所・利用権とJPX公式系列との一致は認証できていないため、結果は提供CSVベースの診断とする。",
        f"- 年次総時価総額の写像と実ETF始値→終値の同日相関中央値={mapping_corr.median():.3f}、最小={mapping_corr.min():.3f}。値は33→17写像品質の補助診断で、予測精度指標とは別。",
        "", "## 限界と採否", "",
        "- 33→17のW_tは親業種内の前年JPX総時価総額比であり、浮動株調整済みETF保有ウェイトではない。構成・分類の過去改定とデータ提供元の改訂方法も確認できていない。",
        "- これは日足始値→終値の予測精度比較で、運用対象の9:10→大引け、約定、手数料・スリッページ、実ポジションのPnLを検証しない。Sharpe/DSRは該当しない。",
        "- 予測スコアは現在ラベルと提案ラベルの双方を比較し、ラベル案を選び直す追加試行として記録。既知期間での成績をOOSや本番採用の根拠にしない。",
        "- production設定・コードは変更していない。",
        "", "## 再現情報", "",
        f"- 設定: `{CONFIG_PATH.relative_to(ROOT)}`。スクリプト: `{Path(__file__).relative_to(ROOT)}`。結果: `{RESULTS_DIR.relative_to(ROOT)}/`。",
        f"- CSVセットSHA-256: `{date_alignment['source_data_sha256']}`。df_exec fingerprint: `{market_fingerprint}`。33系列fingerprint: `{industry_fingerprint}`。設定SHA-256: `{config_hash}`。",
        f"- 既登録JPX33/サブセクター関連record: {len(related_records)}件。候補2件。DSRはポートフォリオSharpeを選択する実験ではないため算出対象外。",
        f"- 実行: `timeout -k 20 14400 .venv/bin/python -u {Path(__file__).relative_to(ROOT)}`。",
        "",
    ])
    report_path = REPORT_DIR / "report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    registry = ExperimentRegistry(REGISTRY_PATH)
    decision = Decision.PENDING if rank_pass and mae_pass else Decision.REJECTED
    for name, detail in (
        ("direct33_parent_labels", "current TOPIX-17 sensitivity labels copied to 33 industries"),
        ("direct33_user_labels", "user-proposed 33-industry sensitivity labels"),
    ):
        registry.record(ExperimentRecord(
            name=f"{EXPERIMENT_ID}_{name}",
            hypothesis="Using 33 industry index open-to-close histories improves direct 33D forecast accuracy mapped to TOPIX-17.",
            start_time=started_at,
            end_time=datetime.now(UTC),
            parameters={
                "variant": name, "description": detail,
                "target_period": [str(paired_dates.min().date()), str(paired_dates.max().date())],
                "target": summary["evaluation"]["target"],
                "industry_returns": "Price / Open - 1 from user-supplied 33 files",
                "data_source_sha256": date_alignment["source_data_sha256"],
                "data_source_provenance": summary["input_data"]["source_provenance"],
                "mapping": "prior-calendar-year JPX total market-cap share within TOPIX-17 parent",
                "fixed_c_full": [str(BASELINE_START.date()), str(BASELINE_END.date())],
                "bootstrap": summary["evaluation"]["bootstrap"],
                "config_sha256": config_hash,
                "related_trial_count_before_this_run": len(related_records),
                "related_prior_trial_names": related_records,
                "retrospective_only": True,
                "production_modified": False,
            },
            metrics=variants[name], decision=decision,
            report_path=str(report_path.relative_to(ROOT)), related_records=related_records,
        ))
    LOG.info("Wrote report=%s", report_path)
    LOG.info("Wrote results=%s", RESULTS_DIR)


if __name__ == "__main__":
    main()
