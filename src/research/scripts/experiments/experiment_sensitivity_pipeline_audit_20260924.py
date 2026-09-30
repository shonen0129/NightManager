#!/usr/bin/env python3
"""Sensitivity-path audit and short 09:10-proxy V2 comparison.

The historical sample is used only for diagnostics. It is not a fresh OOS
period, and the locally reconstructed 09:10 prices are not executable quotes.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/research/scripts/experiments"))

import experiment_sensitivity_rate_prior_v2_20260924 as v2_helpers  # noqa: E402

from leadlag.core.correlation import build_v3_static  # noqa: E402
from leadlag.data.intraday_inputs import compute_jp_target_returns  # noqa: E402
from leadlag.data.market_data_cache import load_df_exec_from_local_cache  # noqa: E402
from leadlag.data.tickers import (  # noqa: E402
    JP_TICKERS,
    N_JP,
    N_US,
    SENSITIVITY_LABELS,
    US_TICKERS,
)
from leadlag.execution.backtester import BacktestEngine  # noqa: E402
from leadlag.execution.config import load_config_from_yaml  # noqa: E402
from leadlag.experiment_registry import (  # noqa: E402
    Decision,
    ExperimentRecord,
    ExperimentRegistry,
    compute_deflated_sharpe,
)
from leadlag.models.v2.distribution_source import OnDemandDistributionSource  # noqa: E402
from leadlag.runner.model_factory import (  # noqa: E402
    model_config_fingerprint,
    resolve_overlay_settings,
)
from leadlag.utils.dataframe_fingerprint import dataframe_fingerprint  # noqa: E402

EXPERIMENT_ID = "20260924_sensitivity_pipeline_audit"
CONFIG_PATH = ROOT / "configs/research/sensitivity_audit_jpx33_candidate_20260924.yaml"
PRIOR_CONFIG_PATH = ROOT / "configs/research/jpx33_sensitivity_prior_2017.yaml"
RATE_CONFIG_PATH = ROOT / "configs/research/sensitivity_rate_prior_v2_20260924.yaml"
CAP_PATH = ROOT / "configs/research/jpx33_yearend_market_cap_2014_2025.csv"
CAP_SOURCE_PATH = ROOT / "configs/research/jpx33_yearend_market_cap_2014_2025.sources.json"
REPORT_DIR = ROOT / "reports/20260924_sensitivity_pipeline_audit"
RESULTS_DIR = ROOT / "var/results/20260924_sensitivity_pipeline_audit"
REGISTRY_PATH = ROOT / "var/experiments/registry.jsonl"
START_DATE = "2026-03-03"
END_DATE = "2026-08-06"
GRID = (-1.0, -0.6, -0.3, 0.0, 0.3, 0.6, 1.0)
BOOTSTRAP_BLOCK = 20
BOOTSTRAP_RESAMPLES = 5000
BOOTSTRAP_SEED = 20260924
LOG = logging.getLogger(EXPERIMENT_ID)


def _read_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"Expected a YAML mapping: {path}")
    return raw


def _industry_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    section = config["industries"]
    columns = list(section["columns"])
    rows = [dict(zip(columns, row, strict=True)) for row in section["rows"]]
    if len(rows) != 33 or len({row["name"] for row in rows}) != 33:
        raise ValueError("Expected 33 unique JPX industry rows")
    return rows


def _labels_from_industry_table(
    scores_config: dict[str, Any], prior_config: dict[str, Any], caps: pd.DataFrame,
    snapshot_year: int,
) -> dict[str, dict[str, float]]:
    score_rows = _industry_rows(scores_config)
    prior_rows = _industry_rows(prior_config)
    mapping = {str(row["name"]): str(row["topix17"]) for row in prior_rows}
    snapshot = caps.loc[caps["snapshot_year"].astype(int) == int(snapshot_year)]
    cap_by_industry = dict(zip(snapshot["industry"].astype(str), snapshot["market_cap_million_yen"].astype(float), strict=True))
    ticker_by_parent = {str(k): str(v) for k, v in prior_config["sensitivity_grid"]["jp_tickers"].items()}
    sums: dict[str, dict[str, float]] = {parent: {f: 0.0 for f in ("w3", "w4", "w5", "w6")} for parent in ticker_by_parent}
    denominators: dict[str, float] = Counter()
    grid_section = scores_config.get("sensitivity_grid", prior_config["sensitivity_grid"])
    if isinstance(grid_section, dict):
        allowed = set(float(x) for x in grid_section.get("values", GRID))
    else:
        allowed = set(float(x) for x in grid_section)
    for row in score_rows:
        name = str(row["name"])
        if name not in mapping or name not in cap_by_industry:
            raise ValueError(f"Industry mapping/cap missing for {name}")
        parent = mapping[name]
        cap = float(cap_by_industry[name])
        if cap <= 0:
            raise ValueError(f"Non-positive PIT industry market cap for {name}: {cap}")
        denominators[parent] += cap
        for factor in ("w3", "w4", "w5", "w6"):
            score = float(row[factor])
            if score not in allowed:
                raise ValueError(f"Out-of-grid score {name}.{factor}={score}")
            sums[parent][factor] += score * cap
    if set(denominators) != set(ticker_by_parent):
        raise ValueError("The 33→17 parent map does not cover all TOPIX-17 groups")
    result: dict[str, dict[str, float]] = {}
    for parent, ticker in ticker_by_parent.items():
        if ticker not in JP_TICKERS:
            raise ValueError(f"Unexpected TOPIX-17 ticker {ticker}")
        result[ticker] = {factor: sums[parent][factor] / denominators[parent] for factor in sums[parent]}
    return result


def _install_labels(labels: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    previous = copy.deepcopy(SENSITIVITY_LABELS)
    if set(labels) != set(US_TICKERS) | set(JP_TICKERS):
        raise ValueError("Sensitivity labels do not cover the 15 US and 17 JP tickers")
    for ticker in previous:
        SENSITIVITY_LABELS[ticker].clear()
        SENSITIVITY_LABELS[ticker].update({k: float(v) for k, v in labels[ticker].items()})
    return previous


def _current_copy() -> dict[str, dict[str, float]]:
    return {ticker: {k: float(v) for k, v in row.items()} for ticker, row in SENSITIVITY_LABELS.items()}


def _v0_for(labels: dict[str, dict[str, float]]) -> np.ndarray:
    old = _install_labels(labels)
    try:
        return build_v3_static(N_US, N_JP, include_v4=True)
    finally:
        _install_labels(old)


def _v0_sensitivity_diagnostics(
    baseline: dict[str, dict[str, float]], seed: int = BOOTSTRAP_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    base_v0 = _v0_for(baseline)
    perturbed: list[dict[str, Any]] = []
    for ticker in (*US_TICKERS, *JP_TICKERS):
        for factor in ("w3", "w4", "w5", "w6"):
            value = float(baseline[ticker][factor])
            index = GRID.index(value)
            for step, direction in ((-1, "toward_lower"), (1, "toward_higher")):
                shifted_index = index + step
                if not 0 <= shifted_index < len(GRID):
                    continue
                labels = copy.deepcopy(baseline)
                labels[ticker][factor] = float(GRID[shifted_index])
                delta = float(np.linalg.norm(_v0_for(labels) - base_v0, ord="fro"))
                perturbed.append({
                    "ticker": ticker, "factor": factor, "from": value,
                    "to": float(GRID[shifted_index]), "direction": direction,
                    "v0_frobenius_delta": delta,
                })
    one_cell = pd.DataFrame(perturbed).sort_values("v0_frobenius_delta", ascending=False)

    rng = np.random.default_rng(seed)
    random_rows: list[dict[str, Any]] = []
    jp_labels = list(JP_TICKERS)
    all_labels = [*US_TICKERS, *JP_TICKERS]
    for perm_type, tickers in (("JP_within_JP", jp_labels), ("all_assets", all_labels)):
        for perm_i in range(500):
            labels = copy.deepcopy(baseline)
            for factor in ("w3", "w4", "w5", "w6"):
                scores = np.array([baseline[ticker][factor] for ticker in tickers], dtype=float)
                shuffled = rng.permutation(scores)
                for ticker, score in zip(tickers, shuffled, strict=True):
                    labels[ticker][factor] = float(score)
            delta = float(np.linalg.norm(_v0_for(labels) - base_v0, ord="fro"))
            random_rows.append({"permutation_type": perm_type, "permutation": perm_i, "v0_frobenius_delta": delta})

    factors = [np.array([baseline[t][f] for t in (*US_TICKERS, *JP_TICKERS)], dtype=float) for f in ("w3", "w4", "w5", "w6")]
    base = np.ones(N_US + N_JP) / np.sqrt(N_US + N_JP)
    group = np.zeros(N_US + N_JP)
    group[:N_US] = N_JP / np.sqrt(N_US * N_JP * (N_US + N_JP))
    group[N_US:] = -N_US / np.sqrt(N_US * N_JP * (N_US + N_JP))
    residual_norms = []
    orthogonal_basis = [base, group]
    residuals = []
    for factor_name, vector in zip(("w3", "w4", "w5", "w6"), factors, strict=True):
        residual = vector.copy()
        for axis in orthogonal_basis:
            residual -= float(residual @ axis) * axis
        residual_norms.append({"factor": factor_name, "raw_norm": float(np.linalg.norm(vector)), "post_projection_norm": float(np.linalg.norm(residual))})
        residuals.append(residual)
        norm = np.linalg.norm(residual)
        if norm > 1e-10:
            orthogonal_basis.append(residual / norm)
    residual_matrix = np.column_stack(residuals)
    corr = np.corrcoef(residual_matrix.T)
    raw_matrix = np.column_stack(factors)
    for axis in (base, group):
        raw_matrix -= np.outer(axis, axis @ raw_matrix)
    raw_matrix -= raw_matrix.mean(axis=0, keepdims=True)
    raw_corr = np.corrcoef(raw_matrix.T)
    correlation_rows = [
        {"factor_a": f1, "factor_b": f2, "residual_correlation": float(corr[i, j])}
        for i, f1 in enumerate(("w3", "w4", "w5", "w6"))
        for j, f2 in enumerate(("w3", "w4", "w5", "w6")) if j > i
    ]
    summary = {
        "baseline_v0_shape": list(base_v0.shape),
        "baseline_v0_rank": int(np.linalg.matrix_rank(base_v0)),
        "baseline_v0_orthogonality_max_error": float(np.max(np.abs(base_v0.T @ base_v0 - np.eye(base_v0.shape[1])))),
        "w3_w6_post_projection_norms": residual_norms,
        "factor_residual_correlations": correlation_rows,
        "raw_factor_correlations_after_removing_v1_v2": [
            {"factor_a": f1, "factor_b": f2, "correlation": float(raw_corr[i, j])}
            for i, f1 in enumerate(("w3", "w4", "w5", "w6"))
            for j, f2 in enumerate(("w3", "w4", "w5", "w6")) if j > i
        ],
        "raw_factor_design_rank_after_removing_v1_v2": int(np.linalg.matrix_rank(raw_matrix)),
        "raw_factor_design_condition_number_after_removing_v1_v2": float(np.linalg.cond(raw_matrix)),
        "one_cell_shift_count": len(perturbed),
        "one_cell_delta_min": float(one_cell["v0_frobenius_delta"].min()),
        "one_cell_delta_median": float(one_cell["v0_frobenius_delta"].median()),
        "one_cell_delta_max": float(one_cell["v0_frobenius_delta"].max()),
        "random_permutation_count": len(random_rows),
        "random_permutation_v0_delta_median_by_type": {kind: float(np.median([x["v0_frobenius_delta"] for x in random_rows if x["permutation_type"] == kind])) for kind in ("JP_within_JP", "all_assets")},
        "random_permutation_v0_delta_p95_by_type": {kind: float(np.quantile([x["v0_frobenius_delta"] for x in random_rows if x["permutation_type"] == kind], 0.95)) for kind in ("JP_within_JP", "all_assets")},
        "positive_scalar_invariance": {},
    }
    for factor in ("w3", "w4", "w5", "w6"):
        labels = copy.deepcopy(baseline)
        for ticker in labels:
            labels[ticker][factor] *= 7.0
        summary["positive_scalar_invariance"][factor] = float(np.linalg.norm(_v0_for(labels) - base_v0, ord="fro"))
    return one_cell, pd.DataFrame(random_rows), summary


def _factor_design_diagnostics(labels: dict[str, dict[str, float]]) -> dict[str, Any]:
    tickers = (*US_TICKERS, *JP_TICKERS)
    matrix = np.column_stack([
        np.array([labels[ticker][factor] for ticker in tickers], dtype=float)
        for factor in ("w3", "w4", "w5", "w6")
    ])
    n = N_US + N_JP
    v1 = np.ones(n) / np.sqrt(n)
    v2 = np.zeros(n)
    v2[:N_US] = N_JP / np.sqrt(N_US * N_JP * n)
    v2[N_US:] = -N_US / np.sqrt(N_US * N_JP * n)
    residual = matrix.copy()
    for axis in (v1, v2):
        residual -= np.outer(axis, axis @ residual)
    residual -= residual.mean(axis=0, keepdims=True)
    corr = np.corrcoef(residual.T)
    factors = ("w3", "w4", "w5", "w6")
    return {
        "rank": int(np.linalg.matrix_rank(residual)),
        "condition_number": float(np.linalg.cond(residual)),
        "pairwise_correlations": [
            {"factor_a": factors[i], "factor_b": factors[j], "correlation": float(corr[i, j])}
            for i in range(4) for j in range(i + 1, 4)
        ],
    }


def _zero_channel(labels: dict[str, dict[str, float]], channel: str | None) -> dict[str, dict[str, float]]:
    result = copy.deepcopy(labels)
    for ticker in result:
        for factor in ("w3", "w4", "w5", "w6"):
            if channel is None or channel == factor:
                result[ticker][factor] = 0.0
    return result


def _best_one_cell_candidate(
    baseline: dict[str, dict[str, float]], one_cell: pd.DataFrame,
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    row = one_cell.iloc[0]
    result = copy.deepcopy(baseline)
    result[str(row["ticker"])][str(row["factor"])] = float(row["to"])
    return result, row.to_dict()


def _variant_labels(
    baseline: dict[str, dict[str, float]], user33: dict[str, dict[str, float]],
    legacy33: dict[str, dict[str, float]], rate_w6: dict[str, float],
    one_cell: dict[str, dict[str, float]],
) -> dict[str, dict[str, dict[str, float]]]:
    variants: dict[str, dict[str, dict[str, float]]] = {"current": copy.deepcopy(baseline)}
    variants["all_w3_w6_zero"] = _zero_channel(baseline, None)
    for factor in ("w3", "w4", "w5", "w6"):
        variants[f"drop_{factor}"] = _zero_channel(baseline, factor)
    for name, jp_prior in (("user_jpx33_pitcap_2025", user33), ("legacy_jpx33_pitcap_2025", legacy33)):
        labels = copy.deepcopy(baseline)
        for ticker in JP_TICKERS:
            labels[ticker].update(jp_prior[ticker])
        variants[name] = labels
    tickers = (*US_TICKERS, *JP_TICKERS)
    rate_current = {ticker: float(rate_w6[ticker]) for ticker in tickers}
    for label, update_us, update_jp in (("rate_w6_us_only", True, False), ("rate_w6_jp_only", False, True), ("rate_w6_both", True, True)):
        labels = copy.deepcopy(baseline)
        for ticker in tickers:
            if (ticker in US_TICKERS and update_us) or (ticker in JP_TICKERS and update_jp):
                labels[ticker]["w6"] = rate_current[ticker]
        variants[label] = labels
    variants["max_structural_one_cell_step"] = copy.deepcopy(one_cell)
    for seed in (20260925, 20260926, 20260927, 20260928, 20260929):
        rng = np.random.default_rng(seed)
        labels = copy.deepcopy(baseline)
        for factor in ("w3", "w4", "w5", "w6"):
            values = rng.permutation([baseline[ticker][factor] for ticker in JP_TICKERS])
            for ticker, score in zip(JP_TICKERS, values, strict=True):
                labels[ticker][factor] = float(score)
        variants[f"jp_random_permutation_{seed}"] = labels
    return variants


def _moving_block_ci(values: np.ndarray, *, seed: int = BOOTSTRAP_SEED) -> list[float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < BOOTSTRAP_BLOCK:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(len(values) / BOOTSTRAP_BLOCK))
    max_start = len(values) - BOOTSTRAP_BLOCK
    estimates = np.empty(BOOTSTRAP_RESAMPLES, dtype=float)
    for i in range(BOOTSTRAP_RESAMPLES):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        sample = np.concatenate([values[s : s + BOOTSTRAP_BLOCK] for s in starts])[: len(values)]
        estimates[i] = float(np.mean(sample))
    return [float(x) for x in np.quantile(estimates, [0.025, 0.975])]


def _rank_metrics(mu_by_day: dict[str, np.ndarray], target: pd.DataFrame) -> dict[str, Any]:
    rows = []
    for date_str, mu in mu_by_day.items():
        if len(mu.shape) != 1 or mu.shape[0] != N_JP:
            continue
        date = pd.Timestamp(date_str)
        if date not in target.index:
            continue
        actual = target.loc[date].to_numpy(dtype=float)
        mask = np.isfinite(mu) & np.isfinite(actual)
        if int(mask.sum()) < 10:
            continue
        if np.ptp(mu[mask]) <= 1e-14 or np.ptp(actual[mask]) <= 1e-14:
            ic = 0.0
        else:
            ic = float(spearmanr(mu[mask], actual[mask]).statistic)
        rows.append({"trade_date": date_str, "observed_etfs": int(mask.sum()), "rank_ic": ic})
    if not rows:
        return {"paired_dates": 0, "mean_rank_ic": None, "rank_ic_ci95": [None, None], "daily": []}
    daily = pd.DataFrame(rows)
    return {
        "paired_dates": int(len(daily)),
        "mean_rank_ic": float(daily["rank_ic"].mean()),
        "rank_ic_ci95": _moving_block_ci(daily["rank_ic"].to_numpy()),
        "mean_observed_etfs": float(daily["observed_etfs"].mean()),
        "dates_all_17": int((daily["observed_etfs"] == N_JP).sum()),
        "daily": daily.to_dict(orient="records"),
    }


def _mu_effect(mu_base: dict[tuple[str, int], np.ndarray], mu_variant: dict[tuple[str, int], np.ndarray]) -> dict[str, Any]:
    diffs = []
    pairs = []
    for key in set(mu_base) & set(mu_variant):
        a, b = mu_base[key], mu_variant[key]
        mask = np.isfinite(a) & np.isfinite(b)
        if not mask.any():
            continue
        diffs.extend((b[mask] - a[mask]).tolist())
        pairs.extend(zip(a[mask].tolist(), b[mask].tolist(), strict=True))
    if not diffs:
        return {"matched_mu_cells": 0, "mean_abs_delta_bps": None, "p95_abs_delta_bps": None, "corr_with_current_mu": None}
    aa, bb = np.asarray(pairs, dtype=float).T
    corr = float(np.corrcoef(aa, bb)[0, 1]) if np.std(aa) > 0 and np.std(bb) > 0 else None
    abs_bps = np.abs(np.asarray(diffs)) * 10000.0
    return {
        "matched_mu_cells": int(len(diffs)),
        "mean_abs_delta_bps": float(abs_bps.mean()),
        "p95_abs_delta_bps": float(np.quantile(abs_bps, 0.95)),
        "corr_with_current_mu": corr,
    }


def _direct33_data_quality() -> dict[str, Any]:
    """Inspect a current-membership 33-industry basket proxy; never call it PIT."""
    try:
        cap_panel = pd.read_parquet(ROOT / "var/research/subsector/cap_panel.parquet")
        master_path = ROOT / "var/research/subsector/jpx_master/jpx_master.csv"
        master = pd.read_csv(master_path, dtype=str)
        master["industry33_name"] = master["33業種区分"].astype(str).str.strip()
        master["code_norm"] = master["コード"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
        master = master.loc[master["industry33_name"].isin([str(row["name"]) for row in _industry_rows(_read_yaml(PRIOR_CONFIG_PATH))])]
        code_to_industry = dict(zip(master["code_norm"], master["industry33_name"], strict=True))
        industries = [str(row["name"]) for row in _industry_rows(_read_yaml(PRIOR_CONFIG_PATH))]
        matched = []
        raw_dir = ROOT / "var/research/subsector/raw_ohlc"
        for ticker in cap_panel.columns:
            code = str(ticker).removesuffix(".T")
            industry = code_to_industry.get(code)
            if industry is None:
                continue
            path = raw_dir / f"{code}.parquet"
            if not path.exists():
                continue
            matched.append((ticker, code, industry))
        # Recompute sector numerator in vectorized industry groups. The
        # cap-panel constituent list is fixed to a 2026 research basket.
        industry_returns: dict[str, pd.Series] = {}
        coverage_counts: dict[str, int] = {}
        for industry in industries:
            tickers = [ticker for ticker in cap_panel.columns if code_to_industry.get(str(ticker).removesuffix(".T")) == industry]
            numer = pd.Series(0.0, index=cap_panel.index)
            denom = pd.Series(0.0, index=cap_panel.index)
            for ticker in tickers:
                code = str(ticker).removesuffix(".T")
                path = raw_dir / f"{code}.parquet"
                if not path.exists():
                    continue
                raw = pd.read_parquet(path, columns=["open", "close"]).reindex(cap_panel.index)
                ret = (raw["close"] / raw["open"] - 1.0).replace([np.inf, -np.inf], np.nan)
                cap = cap_panel[ticker].astype(float).shift(1)
                valid = ret.notna() & cap.notna() & (cap > 0)
                numer = numer.add(ret.where(valid, 0.0).mul(cap.where(valid, 0.0)), fill_value=0.0)
                denom = denom.add(cap.where(valid, 0.0), fill_value=0.0)
            industry_returns[industry] = numer.div(denom.where(denom > 0.0))
            coverage_counts[industry] = len(tickers)
        prior_cfg = _read_yaml(PRIOR_CONFIG_PATH)
        parent_map = {str(row["name"]): str(row["topix17"]) for row in _industry_rows(prior_cfg)}
        ticker_by_parent = {str(k): str(v) for k, v in prior_cfg["sensitivity_grid"]["jp_tickers"].items()}
        official_caps = pd.read_csv(CAP_PATH)
        official_by_year = {
            int(year): dict(zip(group["industry"].astype(str), group["market_cap_million_yen"].astype(float), strict=True))
            for year, group in official_caps.groupby("snapshot_year", sort=True)
        }
        ret33 = pd.DataFrame(industry_returns, index=cap_panel.index).sort_index()
        ret17 = pd.DataFrame(np.nan, index=ret33.index, columns=JP_TICKERS)
        market_df = load_df_exec_from_local_cache(max_stale_bdays=None)
        first_sessions = set()
        for year in range(2015, 2027):
            year_dates = market_df.index[market_df.index.year == year]
            if len(year_dates) > 0:
                first_sessions.add(pd.Timestamp(year_dates[0]))
        for date in ret33.index:
            year = date.year
            snapshot_year = year - 1
            if snapshot_year < 2014 or snapshot_year > 2025 or date in first_sessions:
                continue
            official_by_industry = official_by_year[snapshot_year]
            cap_weights = {industry: official_by_industry[industry] for industry in industries}
            parent_totals: dict[str, float] = Counter()
            for industry in industries:
                parent_totals[parent_map[industry]] += cap_weights[industry]
            industry_ret = ret33.loc[date]
            for parent, ticker in ticker_by_parent.items():
                value = 0.0
                valid_weight = 0.0
                total = parent_totals[parent]
                for industry in industries:
                    if parent_map[industry] != parent or not np.isfinite(industry_ret[industry]):
                        continue
                    weight = cap_weights[industry] / total
                    value += weight * float(industry_ret[industry])
                    valid_weight += weight
                if valid_weight >= 0.8:
                    ret17.loc[date, ticker] = value / valid_weight
        benchmark_columns = [f"jp_oc_{ticker}" for ticker in JP_TICKERS]
        benchmark = market_df[benchmark_columns].copy()
        benchmark.columns = list(JP_TICKERS)
        joined = ret17.join(benchmark, how="inner", lsuffix="_33agg", rsuffix="_etf").loc["2015-01-05":"2026-09-18"]
        corr_rows = []
        for ticker in JP_TICKERS:
            lhs, rhs = joined[f"{ticker}_33agg"], joined[f"{ticker}_etf"]
            mask = lhs.notna() & rhs.notna()
            corr_rows.append({"ticker": ticker, "n": int(mask.sum()), "pearson_corr": float(lhs[mask].corr(rhs[mask])) if mask.sum() >= 30 else None})
        correlations = [x["pearson_corr"] for x in corr_rows if x["pearson_corr"] is not None]
        corr_median = float(np.median(correlations)) if correlations else None
        corr_min = float(min(correlations)) if correlations else None
        corr_below = int(sum(value < 0.70 for value in correlations))
        latest_cap_date = cap_panel.index[cap_panel.index <= pd.Timestamp("2025-12-31")].max()
        cap_coverage = []
        for industry in industries:
            current_code_count = coverage_counts[industry]
            tickers = [ticker for ticker in cap_panel.columns if code_to_industry.get(str(ticker).removesuffix(".T")) == industry]
            sample_cap_m = float(cap_panel.loc[latest_cap_date, tickers].sum(skipna=True) / 1e6) if tickers else 0.0
            cap_coverage.append({
                "industry": industry,
                "selected_current_members_with_prices": current_code_count,
                "selected_current_market_cap_coverage_vs_jpx_2025": sample_cap_m / official_by_industry[industry] if official_by_industry[industry] > 0 else None,
            })
        median_cap_coverage = float(np.median([r["selected_current_market_cap_coverage_vs_jpx_2025"] for r in cap_coverage]))
        all33_coverage = ret33.loc["2015-01-05":"2026-09-18"].notna().all(axis=1)
        return {
            "status": "diagnostic_only_current_membership",
            "master_snapshot_date": str(master["日付"].max()),
            "historical_membership_point_in_time_available": False,
            "cap_panel_rows": int(len(cap_panel)),
            "cap_panel_assets": int(len(cap_panel.columns)),
            "matched_assets_with_current_industry": int(len(matched)),
            "industry_coverage_counts": coverage_counts,
            "industry_dates_all_33_valid": int(all33_coverage.sum()),
            "industry_dates_total": int(len(all33_coverage)),
            "first_jp_session_per_year_excluded_due_to_cap_publication_time": int(len(first_sessions)),
            "selected_basket_market_cap_coverage_median": median_cap_coverage,
            "selected_basket_market_cap_coverage_min": float(min(r["selected_current_market_cap_coverage_vs_jpx_2025"] for r in cap_coverage)),
            "map_to_etf_corr_median": corr_median,
            "map_to_etf_corr_min": corr_min,
            "map_to_etf_corr_below_0_70": corr_below,
            "map_to_etf_corr_by_ticker": corr_rows,
            "selected_basket_cap_coverage_by_industry": cap_coverage,
            "limitations": [
                "JPX master is a 2026-07-31 snapshot; historical industry membership/changes are unavailable locally.",
                "The cap panel and price panel cover a selected 494-company current basket, not all point-in-time constituents.",
                "JPX yearly industry total market cap is not float-adjusted TOPIX-17 ETF holdings; the market universe switches from First Section to Prime in 2022.",
                "Price inputs are daily OHLC open-to-close returns; there are no matching historical 09:10 executable quotes for these 33 industry baskets.",
            ],
            "quality_gate_checks": {
                "all_17_parent_correlations_at_least_0_70": bool(len(correlations) == N_JP and corr_min is not None and corr_min >= 0.70),
                "median_parent_correlation_at_least_0_70": bool(corr_median is not None and corr_median >= 0.70),
                "historical_membership_point_in_time_available": False,
            },
            "quality_gate_reasons": [
                "point-in-time member/classification history is unavailable",
                f"{corr_below} TOPIX-17 basket correlations fall below 0.70",
                "the minimum basket correlation is below 0.70" if corr_min is None or corr_min < 0.70 else "",
            ],
            "quality_gate_passed": False,
        }
    except Exception as exc:
        return {"status": "quality_gate_error", "error": f"{type(exc).__name__}: {exc}", "quality_gate_passed": False}


def _write_report(summary: dict[str, Any], variants_table: pd.DataFrame) -> None:
    user = summary["variants"].get("user_jpx33_pitcap_2025", {})
    current = summary["variants"].get("current", {})
    candidate_dsrs = [
        float(metric["dsr"])
        for name, metric in summary["variants"].items()
        if name != "current" and metric.get("dsr") is not None
    ]
    dsr_range = (min(candidate_dsrs), max(candidate_dsrs)) if candidate_dsrs else (float("nan"), float("nan"))
    rows = [
        "# 感応度ラベルの実効経路監査と09:10 proxy比較",
        "",
        "## 判定",
        "",
        "**本番感応度は現行のまま維持し、承認済みとは扱わない。** 提示されたJPX33案、過去の名目金利w6案、ランダム置換、チャンネル除外を同一期間・同一V2実行経路で比較した。いずれも短い既知期間の価格proxyによる診断で、将来OOSでも実約定検証でもない。",
        "",
        "## 事前基準と対象",
        "",
        f"- 対象: {summary['evaluation']['start_date']}〜{summary['evaluation']['end_date']}、V2 {summary['evaluation']['simulation_dates']}営業日。実験前に固定した判定条件は年率Net Sharpe差+0.05以上、20営業日block bootstrapの日次Net差95%CI下限>0、DD悪化≤1pp、turnover増加≤10%、fallback非増加、市場中立・gross制約内。",
        f"- データ: 局所09:10価格proxyは{summary['evaluation']['open_910_dates_any']}日、17ETF全銘柄揃い{summary['evaluation']['open_910_dates_all17']}日。proxyは5分足中値/終値から再構成され、実際に約定可能な9:10 bid/askではない。全評価日損益にはproxyがない日の始値→大引けfallbackも含む。",
        "- 期間は2026-09-24時点で既に確認済みであり、未使用OOSではない。9:10 proxy標本は採択に必要な252日にも達しないため、数値が閾値を超えても採用できない。",
        "- V2分布は毎変種でcacheをbypassしてon-demandで再計算。その他設定、ギャップ履歴、コスト仮定、リスク制約、取引ETFは共通。",
        "- w3〜w6の寄与を分離するため、production.yamlで有効なML注文オーバーレイは全変種で無効化した。したがってこの損益は感応度prior→BLPX→V2ウェイト/費用経路の比較であり、現行production overlayとの相互作用を含む完全な本番再現ではない。forward shadowでは開始時に凍結した同一overlay artifactをbaseline/candidate双方へ適用する。",
        "- 指標のslippage/financing/borrow/reverse費用は既存バックテストcost model上の推定値で、約定ログから観測した費用ではない。",
        "- 業種別時価総額の一次資料は[JPXの月次統計](https://www.jpx.co.jp/markets/statistics-equities/monthly/)で、掲載時刻は第1営業日13時以降。2022年4月以降はPrimeの値を含み、それ以前のFirst Section値との市場ユニバース差を含む。分類所属変更履歴は[証券コード協議会の公表資料](https://www.jpx.co.jp/sicc/sectors/)を参照。",
        "",
        "## 感応度がモデルへ伝わるか",
        "",
        f"- 現行 `build_v3_static` のV0形状は{summary['matrix_audit']['baseline_v0_shape']}、rank {summary['matrix_audit']['baseline_v0_rank']}、直交誤差最大 {summary['matrix_audit']['baseline_v0_orthogonality_max_error']:.2e}。raw各因子は逐次Gram–Schmidtと単位長正規化を通る。各w列に正の定数7を掛けた場合のV0差は因子別に記録し、定性的な強度ラベルを倍率として読み替えられないことを確認した。",
        f"- 全{summary['matrix_audit']['one_cell_shift_count']}の1段単セル摂動でV0 Frobenius差はmin={summary['matrix_audit']['one_cell_delta_min']:.4f}、median={summary['matrix_audit']['one_cell_delta_median']:.4f}、max={summary['matrix_audit']['one_cell_delta_max']:.4f}。JP内500置換の差はmedian={summary['matrix_audit']['random_permutation_v0_delta_median_by_type']['JP_within_JP']:.4f}、95th percentile={summary['matrix_audit']['random_permutation_v0_delta_p95_by_type']['JP_within_JP']:.4f}。全32銘柄間500置換はmedian={summary['matrix_audit']['random_permutation_v0_delta_median_by_type']['all_assets']:.4f}、95th percentile={summary['matrix_audit']['random_permutation_v0_delta_p95_by_type']['all_assets']:.4f}。",
        "- 下表の μ 差、最終ウェイト差、turnover、cost後PnLが、w3〜w6ブロックの実効寄与を示す。感応度ラベルはV0→BLPX μ→V2最終ウェイト→コスト控除損益まで追跡した。",
        "",
        "## 同条件の短期V2比較",
        "",
        "| 変種 | 平均|Δμ| (bp) | avg weight L1差 | Net Sharpe | ΔNet Sharpe | Gross Sharpe | ΔGross Sharpe | Max DD | avg turnover/day | turnover差 | Net差 CI 95% (bp/日) | fallback率 | Rank IC日数 | 17ETF全揃い日 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in variants_table.iterrows():
        rows.append(
            f"| {row['variant']} | {row['mean_abs_mu_delta_bps']} | {row['mean_l1_weight_delta']} | {row['net_sharpe']} | {row['net_sharpe_delta']} | {row['gross_sharpe']} | {row['gross_sharpe_delta']} | {row['max_drawdown']} | {row['mean_daily_turnover_one_way']} | {row['turnover_change']} | {row['net_daily_delta_ci95_bps']} | {row['fallback_rate']} | {row['rank_ic_dates']} | {row['dates_all17']} |"
        )
    rows.extend([
        "",
        "Net/Gross Sharpeは245営業日年率化、最大DD・return系列はフラット日を含む全評価日に同じ計算を適用。turnoverは片道。CIは同じ日付のpaired non-circular 20営業日block bootstrap、5,000回。Rank IC評価日数は10銘柄以上、17ETF全揃い日は全17銘柄が観測できた日数。",
        "",
        "### コスト・エクスポージャー",
        "",
        "| 変種 | slippage (bp期間計) | financing (bp) | borrow (bp) | reverse (bp) | 総cost (bp) | 平均日cost (bp) | model gross max | model net max abs | effective gross max | effective net max abs | side leverage | fallback率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for name, metric in summary["variants"].items():
        rows.append(
            f"| {name} | {metric.get('slippage_total_bps', 0.0):.2f} | {metric.get('financing_total_bps', 0.0):.2f} | {metric.get('borrow_total_bps', 0.0):.2f} | {metric.get('reverse_total_bps', 0.0):.2f} | {metric.get('costs_total_bps', 0.0):.2f} | {metric.get('mean_daily_cost_bps', 0.0):.3f} | {metric.get('max_model_gross', float('nan')):.4f} | {metric.get('max_abs_model_net', float('nan')):.4f} | {metric.get('max_effective_gross', float('nan')):.4f} | {metric.get('max_abs_effective_net', float('nan')):.4f} | {metric.get('side_leverage', float('nan')):.2f} | {metric.get('fallback_rate', float('nan')):.2%} |"
        )
    rows.extend([
        "",
        "コストはバックテストのモデル推定で、実約定差は未測定。設定は片道slippage 5bp、年率financing 2.50%、borrow 1.15%、reverse 2.0bp/day、side leverage 1.5。model exposure制約 (gross ≤2.0, |net|≤0.05) とside leverage後effective exposureは分けた。PIT multiplier適用率は未取得。",
        "",
        "### V2監査",
        "",
        "| 変種 | decisions | leakage status counts | numerical status counts | on-demand status counts |",
        "|---|---:|---|---|---|",
    ])
    for name, metric in summary["variants"].items():
        rows.append(
            f"| {name} | {metric.get('decisions_counted')} | `{json.dumps(metric.get('leakage_status_counts', {}), ensure_ascii=False)}` | `{json.dumps(metric.get('numerical_status_counts', {}), ensure_ascii=False)}` | `{json.dumps(metric.get('on_demand_by_horizon_status', {}), ensure_ascii=False)}` |"
        )
    rows.extend([
        "",
        "V2各decisionで取得したleakage/numerical statusを上表に保存。汎用 `ComplianceAuditor` はこの実験ランで未実行であり、V2側のstatusで置き換えていない。",
        "",
        "## JPX33候補とw6分離",
        "",
        f"- 提示された33業種w3〜w6候補を、2025年末業種時価総額で17ETF priorへ写像し、2026年3〜8月に適用した。予測日より前に公表された年末表を使うため、この比較のウェイト参照時点は過去の2017固定案より適切。ただしJPX総時価総額であり浮動株調整済みETF保有比率ではない。候補の平均Rank ICは{user.get('rank_ic', {}).get('mean_rank_ic')}、現行は{current.get('rank_ic', {}).get('mean_rank_ic')}。",
        "- 33→17の長期予測仮説は本V2表とは別。公式年末業種時価総額から得た年次W_tは浮動株調整ETFウェイトではない。ローカルの銘柄業種表が2026-07-31の一時点しかなく、2015〜2026のPIT業種変更・上場廃止・指数採用履歴を復元できない。よって直接33次元BLPXの本採用比較は品質ゲート未通過で保留し、現行時点の494銘柄proxyで計算した補助品質値をsummaryに残す。",
        f"- current-membership 494銘柄proxyの補助品質値: 33→17のETF対照相関 median={summary['direct33_quality'].get('map_to_etf_corr_median')} / min={summary['direct33_quality'].get('map_to_etf_corr_min')}、0.70未満={summary['direct33_quality'].get('map_to_etf_corr_below_0_70')}銘柄、選定バスケット市場時価総額カバー率 median={summary['direct33_quality'].get('selected_basket_market_cap_coverage_median')}。この結果も現行構成銘柄を過去へ遡及した代理で、採択根拠にはしない。",
        "- 2015〜2017年に固定2017年末ウェイトを使った部分は、前年末の公表済み時価総額を使う年次PITへ訂正し影響を測定した。714日の日次open-to-close residual予測でPIT−固定2017のΔRank ICは−0.000044（95% CI [−0.000274,+0.000124]）、ΔMAEは−0.00028bp（95% CI [−0.00060,+0.00010]）。差はほぼゼロで、9:10ターゲットでもない。詳細は `reports/20260924_jpx33_pitcap_correction_2015_2017/report.md`。",
        "- w6はUS-only / JP-only / 両側変更を分けて測定した。既存候補名は「名目金利寄り」だが、米国/日本の金利市場、通貨、年限、変化量か水準か、予想外変化か、観測時刻を定義していないため、経済的な金利ベータとは解釈しない。",
        "",
        "## 構造面の確認",
        "",
        f"`build_v3_static` はw3〜w6を順次直交化して各列を単位長へ正規化する。正の一括倍率は消えるが、単セル変更・符号変更・置換は投影方向を変える。現行w5/w6の入力相関は{next(x['correlation'] for x in summary['matrix_audit']['raw_factor_correlations_after_removing_v1_v2'] if x['factor_a']=='w5' and x['factor_b']=='w6'):.3f}、4因子のrank={summary['matrix_audit']['raw_factor_design_rank_after_removing_v1_v2']}・条件数={summary['matrix_audit']['raw_factor_design_condition_number_after_removing_v1_v2']:.3f}。これは値を「感応度の大きさ」と読むのではなく、事前部分空間の方向指定として読む必要があるという実装上の意味である。直交誤差は構築手順の性質なので重複の診断には使わない。",
        "",
        "## 意思決定と次段階",
        "",
        "- 研究候補を採択できる条件（上記Net Sharpe・CI・DD・turnover・fallback制約）を満たす十分なデータはない。既知の過去比較だけを根拠に現行感応度を変更しない。",
        "- 9:10 targetの実測データと実約定スプレッド/slippage/空売り費用、取引不能ログを日々保存し、仕様を凍結した翌営業日以降のforward期間で比較する。まず252営業日のshadow測定を計画し、全17ETFデータが揃う営業日数と実約定率を併記する。採否はコスト後PnLと事前基準で決める。",
        "- 33業種直接予測は、PIT構成銘柄/分類と float-adjusted W_t、33次元Ωのrolling推定、W_t Ω33 W_t'写像を再現できる時点付きデータを取得してから同じV2制約・執行対象で比較する。",
        "- w6を再試験する場合は因子の対象市場・通貨・年限・金利変化定義・予想外成分・観測時刻を先に凍結し、US-only/JP-only/bothの3比較を維持する。",
        "",
        "## 再現情報",
        "",
        f"- 設定: `{CONFIG_PATH.relative_to(ROOT)}`, prior map `{PRIOR_CONFIG_PATH.relative_to(ROOT)}`, official cap PIT table `{CAP_PATH.relative_to(ROOT)}`。",
        f"- 実験スクリプト: `{Path(__file__).relative_to(ROOT)}`。期間、変種、μ、weights、日次PnL、cost内訳、監査カウンタは `{RESULTS_DIR.relative_to(ROOT)}/`。",
        f"- df_exec SHA-256: `{summary['df_exec_fingerprint']}`。実効V2 config fingerprint: `{summary['effective_config_fingerprint']}`。",
        f"- Registry開始件数 {summary['registry_records_before']}、関連する先行感応度record {summary['relevant_prior_trials']}、今回比較変種 {summary['candidate_trial_count']}、補正trial数 {summary['dsr_trial_count']}。短期103日での候補DSR範囲は{dsr_range[0]:.3f}〜{dsr_range[1]:.3f}。日次Netリターンと245日年率Sharpeによる試行回数補正の診断値で、採否の証拠には使わない。",
        "- 再実行: `timeout -k 20 14400 .venv/bin/python -u src/research/scripts/experiments/experiment_sensitivity_pipeline_audit_20260924.py`。PIT訂正診断は `timeout -k 20 3600 .venv/bin/python -u src/research/scripts/experiments/experiment_jpx33_pitcap_correction_2015_2017_20260924.py`。",
        "- production config、ticker definitions、live cache/storeは変更していない。",
        "",
    ])
    (REPORT_DIR / "report.md").write_text("\n".join(rows), encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    started_at = datetime.now(UTC)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    config = _read_yaml(CONFIG_PATH)
    prior_config = _read_yaml(PRIOR_CONFIG_PATH)
    rate_config = _read_yaml(RATE_CONFIG_PATH)
    caps = pd.read_csv(CAP_PATH)
    cap_manifest = json.loads(CAP_SOURCE_PATH.read_text(encoding="utf-8"))
    if len(_industry_rows(config)) != 33:
        raise ValueError("Candidate table was not parsed as 33 rows")
    for year in range(2014, 2026):
        if len(caps.loc[caps["snapshot_year"].astype(int) == year]) != 33:
            raise ValueError(f"Point-in-time cap table missing complete snapshot for {year}")

    baseline_labels = _current_copy()
    user33 = _labels_from_industry_table(config, prior_config, caps, 2025)
    legacy33_cfg = copy.deepcopy(prior_config)
    legacy33 = _labels_from_industry_table(legacy33_cfg, prior_config, caps, 2025)
    rate_w6 = {str(k): float(v) for k, v in rate_config["w6_candidate"].items()}
    onecell_frame, random_matrix_frame, matrix_summary = _v0_sensitivity_diagnostics(baseline_labels)
    onecell_labels, onecell_meta = _best_one_cell_candidate(baseline_labels, onecell_frame)
    variants = _variant_labels(baseline_labels, user33, legacy33, rate_w6, onecell_labels)
    if len(variants) < 12:
        raise ValueError(f"Expected a broad sensitivity audit, got {len(variants)} variants")
    onecell_frame.to_csv(RESULTS_DIR / "one_cell_v0_sensitivity.csv", index=False)
    random_matrix_frame.to_csv(RESULTS_DIR / "random_permutation_v0_sensitivity.csv", index=False)
    pd.DataFrame([{"ticker": tk, **user33[tk]} for tk in JP_TICKERS]).to_csv(RESULTS_DIR / "user_jpx33_2025cap_aggregated_labels.csv", index=False)

    production_config = load_config_from_yaml(ROOT / "configs/production/production.yaml", strict=True)
    overlay_enabled, overlay_path = resolve_overlay_settings(production_config)
    app_config = v2_helpers.safe_config_copy(production_config)
    if overlay_enabled:
        app_config = app_config.model_copy(update={
            "v2": app_config.v2.model_copy(update={"ml_overlay_enabled": False}),
            "ml_order_overlay": app_config.ml_order_overlay.model_copy(update={"enabled": False}),
        })
    frame, eval_dates, historical, input_manifest = v2_helpers._load_owned_inputs(app_config, START_DATE, END_DATE)
    input_manifest["production_effective_config_fingerprint"] = model_config_fingerprint(production_config)
    input_manifest["production_overlay_enabled"] = bool(overlay_enabled)
    input_manifest["production_overlay_path"] = None if overlay_path is None else str(overlay_path)
    transforms = v2_helpers._build_transform_cache(frame, app_config, eval_dates)
    if historical.open_910_returns is None:
        raise ValueError("HistoricalInputs has no local 09:10 proxy panel")
    open_to_910 = historical.open_910_returns.loc[
        (historical.open_910_returns.index >= pd.Timestamp(START_DATE))
        & (historical.open_910_returns.index <= pd.Timestamp(END_DATE))
    ].copy()
    open_to_910.columns = list(JP_TICKERS)
    target_values = compute_jp_target_returns(
        frame, JP_TICKERS, open_910_returns=historical.open_910_returns
    )
    target_panel = pd.DataFrame(target_values, index=frame.index, columns=JP_TICKERS).loc[eval_dates].copy()
    baseline_returns: np.ndarray | None = None
    results: dict[str, Any] = {}
    daily_by_variant: dict[str, pd.DataFrame] = {}
    mu_by_variant: dict[str, dict[tuple[str, int], np.ndarray]] = {}
    weights_by_variant: dict[str, np.ndarray] = {}
    elapsed_by_variant: dict[str, float] = {}
    for variant_name, labels in variants.items():
        LOG.info("Starting %s", variant_name)
        previous_labels = _install_labels(labels)
        counters, originals = v2_helpers._install_measurement_hooks(transforms, variant_name)
        captured_mu: dict[tuple[str, int], np.ndarray] = {}
        hooked_resolve = OnDemandDistributionSource.resolve

        def capture_resolve(self: Any, trade_date: str, df_exec: Any, current_prices: Any, *, horizon: int = 1, **kwargs: Any) -> Any:
            result = hooked_resolve(self, trade_date, df_exec, current_prices, horizon=horizon, **kwargs)
            if result.mu_gap is not None:
                captured_mu[(str(trade_date), int(horizon))] = np.asarray(result.mu_gap, dtype=float).copy()
            return result

        OnDemandDistributionSource.resolve = capture_resolve  # type: ignore[method-assign]
        started = time.monotonic()
        try:
            result = BacktestEngine.run_v2_backtest(
                cfg=app_config,
                gap_input_dir=input_manifest["gap_store_path"],
                df_exec=frame,
                start_date=START_DATE,
                end_date=END_DATE,
                n_jobs=1,
                historical_inputs=historical,
            )
        finally:
            elapsed_by_variant[variant_name] = time.monotonic() - started
            v2_helpers._restore_measurement_hooks(originals)
            _install_labels(previous_labels)
        if len(result["daily_returns"]) != len(eval_dates):
            raise ValueError(f"{variant_name} returned {len(result['daily_returns'])} days; expected {len(eval_dates)}")
        metric, daily = v2_helpers._metrics(result)
        metric["leakage_status_counts"] = dict(counters["leakage_status"])
        metric["numerical_status_counts"] = dict(counters["numerical_status"])
        metric["decisions_counted"] = int(counters["decisions"])
        metric["on_demand_by_horizon_status"] = {f"h{h}:{status}": int(count) for (h, status), count in counters["on_demand_by_horizon_status"].items()}
        metric["elapsed_seconds"] = elapsed_by_variant[variant_name]
        results[variant_name] = {"metric": metric, "error": None}
        daily_by_variant[variant_name] = daily
        mu_by_variant[variant_name] = captured_mu
        weights = np.asarray(result["weights"], dtype=float)
        weights_by_variant[variant_name] = weights
        if variant_name == "current":
            baseline_returns = np.asarray(result["daily_returns"], dtype=float)
        # Keep only compact daily outputs plus positions; do not pickle Python objects.
        pd.DataFrame(weights, index=result["daily_returns"].index, columns=JP_TICKERS).to_csv(
            RESULTS_DIR / f"weights_{variant_name}.csv", index_label="trade_date"
        )
        daily.to_csv(RESULTS_DIR / f"daily_{variant_name}.csv", index_label="trade_date")
        pd.DataFrame([
            {"trade_date": date, "horizon": horizon, **{f"mu_{ticker}": float(value) for ticker, value in zip(JP_TICKERS, mu, strict=True)}}
            for (date, horizon), mu in sorted(captured_mu.items()) if mu.shape == (N_JP,)
        ]).to_csv(RESULTS_DIR / f"mu_{variant_name}.csv", index=False)
        LOG.info("Completed %s: days=%s Sharpe=%.4f fallback=%s", variant_name, metric["n_observations"], metric["net_sharpe"], metric["fallback_days"])
    if baseline_returns is None:
        raise RuntimeError("Current baseline was not executed")

    baseline_mu = mu_by_variant["current"]
    summary_variants: dict[str, Any] = {}
    table_rows: list[dict[str, Any]] = []
    candidate_names = [name for name in variants if name != "current"]
    prior_records = list(ExperimentRegistry(REGISTRY_PATH))
    relevant_prior = [r for r in prior_records if any(tag in r.name.lower() for tag in ("sensitiv", "jpx33"))]
    trial_count = len(relevant_prior) + len(candidate_names)
    sharpe_trials = [float(results[name]["metric"]["net_sharpe"]) for name in candidate_names if np.isfinite(results[name]["metric"]["net_sharpe"])]
    base_mu_h1 = {date: mu for (date, horizon), mu in baseline_mu.items() if horizon == 1}
    base_rank = _rank_metrics(base_mu_h1, target_panel)
    base_rank_open_to_0910 = _rank_metrics(base_mu_h1, open_to_910)
    summary_variants["current"] = {
        **results["current"]["metric"],
        "rank_ic": base_rank,
        "rank_ic_open_to_0910_diagnostic": base_rank_open_to_0910,
        "mu_effect_vs_current": {"matched_mu_cells": len(baseline_mu), "mean_abs_delta_bps": 0.0, "p95_abs_delta_bps": 0.0, "corr_with_current_mu": 1.0},
    }
    table_rows.append({
        "variant": "current baseline",
        "mean_abs_mu_delta_bps": 0.0,
        "mean_l1_weight_delta": 0.0,
        "net_sharpe": round(float(results["current"]["metric"]["net_sharpe"]), 4),
        "net_sharpe_delta": 0.0,
        "gross_sharpe": round(float(results["current"]["metric"]["gross_sharpe"]), 4),
        "gross_sharpe_delta": 0.0,
        "max_drawdown": f"{float(results['current']['metric']['max_drawdown'])*100:.2f}%",
        "mean_daily_turnover_one_way": round(float(results["current"]["metric"]["mean_daily_turnover_one_way"]), 5),
        "turnover_change": "—",
        "net_daily_delta_ci95_bps": "—",
        "fallback_days": int(results["current"]["metric"]["fallback_days"]),
        "fallback_rate": f"{float(results['current']['metric']['fallback_rate'])*100:.2f}%",
        "rank_ic_dates": int(base_rank.get("paired_dates", 0)),
        "dates_all17": int(input_manifest["open_910_dates_complete_for_17_etfs"]),
    })

    for name in candidate_names:
        metric = results[name]["metric"]
        daily = daily_by_variant[name]
        base_daily = daily_by_variant["current"]
        aligned = base_daily.join(daily, lsuffix="_base", rsuffix="_candidate", how="inner")
        if len(aligned) != len(base_daily):
            raise ValueError(f"Daily result alignment failure for {name}")
        net_delta = daily["net_return"].to_numpy(dtype=float) - base_daily["net_return"].to_numpy(dtype=float)
        ci = _moving_block_ci(net_delta)
        weight_a = weights_by_variant["current"]
        weight_b = weights_by_variant[name]
        same_shape = weight_a.shape == weight_b.shape
        l1_delta = np.abs(weight_b - weight_a).sum(axis=1) if same_shape else np.full(1, np.nan)
        mu_h1 = {date: mu for (date, horizon), mu in mu_by_variant[name].items() if horizon == 1}
        rank = _rank_metrics(mu_h1, target_panel)
        rank_open_to_0910 = _rank_metrics(mu_h1, open_to_910)
        mu_effect = _mu_effect(baseline_mu, mu_by_variant[name])
        baseline_sharpe = results["current"]["metric"]["net_sharpe"]
        turn_change = metric["total_turnover_one_way"] / max(results["current"]["metric"]["total_turnover_one_way"], 1e-12) - 1.0
        dd_worsening_pp = (results["current"]["metric"]["max_drawdown"] - metric["max_drawdown"]) * 100.0
        model_gross_ok = metric["max_model_gross"] <= 2.0 + 1e-8
        model_net_ok = metric["max_abs_model_net"] <= 0.05 + 1e-8
        metric["paired_net_return_delta_ci95"] = ci
        metric["mean_daily_net_delta_bps"] = float(net_delta.mean() * 10000.0)
        metric["mean_l1_weight_delta"] = float(np.nanmean(l1_delta))
        metric["dates_with_changed_final_weights"] = int((l1_delta > 1e-10).sum())
        metric["mu_effect_vs_current"] = mu_effect
        metric["rank_ic"] = rank
        metric["rank_ic_open_to_0910_diagnostic"] = rank_open_to_0910
        metric["drawdown_worsening_pp_vs_current"] = dd_worsening_pp
        metric["turnover_change_vs_current_fraction"] = float(turn_change)
        metric["model_constraints_pass"] = bool(model_gross_ok and model_net_ok)
        dsr_metrics = {
            "net_sharpe": metric["net_sharpe"], "net_sharpe_frequency": "annual",
            "trading_days_per_year": v2_helpers.TRADING_DAYS, "trials": max(trial_count, 1),
            "n_observations": metric["n_observations"], "returns": daily["net_return"].tolist(),
            "trial_sharpes": sharpe_trials,
        }
        metric["dsr"] = compute_deflated_sharpe(dsr_metrics)
        metric["dsr_trial_count"] = trial_count
        summary_variants[name] = metric
        table_rows.append({
            "variant": name,
            "mean_abs_mu_delta_bps": round(mu_effect["mean_abs_delta_bps"], 4) if mu_effect["mean_abs_delta_bps"] is not None else "n/a",
            "mean_l1_weight_delta": round(float(np.nanmean(l1_delta)), 5),
            "net_sharpe": round(float(metric["net_sharpe"]), 4),
            "net_sharpe_delta": round(float(metric["net_sharpe"] - baseline_sharpe), 4),
            "gross_sharpe": round(float(metric["gross_sharpe"]), 4),
            "gross_sharpe_delta": round(float(metric["gross_sharpe"] - results["current"]["metric"]["gross_sharpe"]), 4),
            "max_drawdown": f"{float(metric['max_drawdown'])*100:.2f}%",
            "mean_daily_turnover_one_way": round(float(metric["mean_daily_turnover_one_way"]), 5),
            "turnover_change": f"{turn_change*100:+.2f}%",
            "net_daily_delta_ci95_bps": f"[{ci[0]*10000:+.3f}, {ci[1]*10000:+.3f}]",
            "fallback_days": int(metric["fallback_days"]),
            "fallback_rate": f"{float(metric['fallback_rate'])*100:.2f}%",
            "rank_ic_dates": int(rank.get("paired_dates", 0)),
            "dates_all17": int(input_manifest["open_910_dates_complete_for_17_etfs"]),
        })
    table = pd.DataFrame(table_rows)
    table.to_csv(RESULTS_DIR / "variant_comparison.csv", index=False)
    matrix_summary["one_cell_top"] = onecell_meta
    matrix_summary["positive_scalar_invariance"] = matrix_summary["positive_scalar_invariance"]
    matrix_audit = matrix_summary
    matrix_report = {
        key: value for key, value in matrix_audit.items()
        if key != "one_cell_top"
    }
    matrix_report["one_cell_top"] = onecell_meta
    candidate_design_labels = copy.deepcopy(baseline_labels)
    for ticker in JP_TICKERS:
        candidate_design_labels[ticker].update(user33[ticker])
    matrix_report["user_jpx33_pitcap_2025_design"] = _factor_design_diagnostics(candidate_design_labels)

    open910_any = int(input_manifest["open_910_dates_with_any_finite_price"])
    open910_all = int(input_manifest["open_910_dates_complete_for_17_etfs"])
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "created_at": started_at.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "evaluation": {
            "start_date": START_DATE, "end_date": END_DATE,
            "simulation_dates": int(len(eval_dates)),
            "open_910_dates_any": open910_any,
            "open_910_dates_all17": open910_all,
            "min_paired_dates_for_adoption": 252,
            "target_is_actual_executable_quote": False,
            "target_fallback_present": True,
            "acceptance_thresholds": rate_config["evaluation"]["adoption_thresholds"],
        },
        "input_manifest": input_manifest,
        "df_exec_fingerprint": dataframe_fingerprint(frame),
        "effective_config_fingerprint": model_config_fingerprint(app_config),
        "research_config_sha256": hashlib.sha256(CONFIG_PATH.read_bytes() + PRIOR_CONFIG_PATH.read_bytes() + CAP_PATH.read_bytes() + RATE_CONFIG_PATH.read_bytes()).hexdigest(),
        "cap_source_manifest": cap_manifest,
        "matrix_audit": matrix_report,
        "variants": summary_variants,
        "direct33_quality": _direct33_data_quality(),
        "registry_records_before": len(prior_records),
        "relevant_prior_trials": len(relevant_prior),
        "candidate_trial_count": len(candidate_names),
        "dsr_trial_count": trial_count,
        "elapsed_seconds": elapsed_by_variant,
        "production_modified": False,
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=v2_helpers._json_default), encoding="utf-8")
    (RESULTS_DIR / "open_910_target_returns.csv").write_text(open_to_910.to_csv(index_label="trade_date"), encoding="utf-8")
    (RESULTS_DIR / "backtest_target_returns.csv").write_text(target_panel.to_csv(index_label="trade_date"), encoding="utf-8")
    _write_report(summary, table)

    registry = ExperimentRegistry(REGISTRY_PATH)
    now = datetime.now(UTC)
    related = [r.name for r in relevant_prior]
    for name in candidate_names:
        metric = summary_variants[name]
        registry.record(ExperimentRecord(
            name=f"{EXPERIMENT_ID}_{name}",
            hypothesis="Sensitivity labels w3-w6 materially change 09:10 μ, final weights, and cost-adjusted portfolio returns; the user-proposed 33-industry prior or separated nominal-rate w6 may improve the current prior.",
            start_time=started_at,
            end_time=now,
            parameters={
                "variant": name,
                "candidate_labels": variants[name],
                "target_window": [START_DATE, END_DATE],
                "target": "09:10-to-close when local 5-minute price proxy is available; otherwise engine fallback",
                "cache_policy": "bypass μ/Ω cache and recompute on-demand BLPX for each decision",
                "cap_snapshot": "2025 year-end total JPX industry market capitalization; recent comparison dates begin 2026-03-03",
                "prior_cap_fingerprint": hashlib.sha256(CAP_PATH.read_bytes()).hexdigest(),
                "production_effective_config_fingerprint": model_config_fingerprint(production_config),
                "research_effective_config_fingerprint": model_config_fingerprint(app_config),
                "df_exec_fingerprint": summary["df_exec_fingerprint"],
                "related_prior_trial_count": len(relevant_prior),
                "all_prior_trial_ids": related,
                "adoption": "diagnostic only; no future OOS or executable quotes",
            },
            metrics=metric,
            decision=Decision.PENDING,
            report_path=str((REPORT_DIR / "report.md").relative_to(ROOT)),
            related_records=related,
        ))
    print(f"report={REPORT_DIR / 'report.md'}", flush=True)
    print(f"results={RESULTS_DIR}", flush=True)


if __name__ == "__main__":
    main()
