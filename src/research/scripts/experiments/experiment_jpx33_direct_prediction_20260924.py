#!/usr/bin/env python3
"""Retrospective proxy comparison: direct 17-ETF vs JPX33 residual BLPX.

JPX33 target baskets use the local selected 494-stock, current-classification
research universe. Results are therefore diagnostic and cannot validate a
point-in-time, executable 09:10 strategy.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.core.correlation import (  # noqa: E402
    _orthogonalize_and_normalize,
    build_base_vectors,
    build_v3_static,
    compute_baseline_correlation,
)
from leadlag.core.gap_adjustment import build_raw_distribution  # noqa: E402
from leadlag.core.residualize import compute_rolling_ols_betas  # noqa: E402
from leadlag.data.market_data_cache import load_df_exec_from_local_cache  # noqa: E402
from leadlag.data.tickers import JP_TICKERS, SENSITIVITY_LABELS, US_TICKERS  # noqa: E402
from leadlag.execution.config import load_config_from_yaml  # noqa: E402
from leadlag.experiment_registry import Decision, ExperimentRecord, ExperimentRegistry  # noqa: E402
from leadlag.models.blpx.model import ProductionBLPXModel  # noqa: E402
from leadlag.utils.dataframe_fingerprint import dataframe_fingerprint  # noqa: E402

EXPERIMENT_ID = "20260924_jpx33_direct_prediction"
CONFIG_PATH = ROOT / "configs/research/jpx33_direct_prediction_2015_2026.yaml"
INDUSTRY_LABEL_PATH = ROOT / "configs/research/sensitivity_audit_jpx33_candidate_20260924.yaml"
PARENT_MAP_PATH = ROOT / "configs/research/jpx33_sensitivity_prior_2017.yaml"
CAP_PATH = ROOT / "configs/research/jpx33_yearend_market_cap_2014_2025.csv"
MASTER_PATH = ROOT / "var/research/subsector/jpx_master/jpx_master.csv"
CAP_PANEL_PATH = ROOT / "var/research/subsector/cap_panel.parquet"
RAW_PRICE_DIR = ROOT / "var/research/subsector/raw_ohlc"
REPORT_DIR = ROOT / "reports/20260924_jpx33_direct_prediction"
RESULTS_DIR = ROOT / "var/results/20260924_jpx33_direct_prediction"
REGISTRY_PATH = ROOT / "var/experiments/registry.jsonl"
START_DATE = "2015-01-05"
END_DATE = "2026-09-18"
BASELINE_START = "2010-01-01"
BASELINE_END = "2014-12-31"
BOOTSTRAP_BLOCK = 20
BOOTSTRAP_RESAMPLES = 5000
BOOTSTRAP_SEED = 20260924
N_US = len(US_TICKERS)
LOG = logging.getLogger(EXPERIMENT_ID)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected YAML mapping in {path}")
    return value


def _normalize_code(value: Any) -> str:
    return str(value).strip().replace(".0", "")


def _load_industry_proxy(
    cap_panel: pd.DataFrame,
    industries: list[str],
) -> tuple[pd.DataFrame, dict[str, int], dict[str, list[str]]]:
    """Build daily 33-industry open-to-close returns using lagged cap weights."""
    master = pd.read_csv(MASTER_PATH, dtype=str)
    master["code_norm"] = master["コード"].map(_normalize_code)
    master["industry33_name"] = master["33業種区分"].astype(str).str.strip()
    master = master.loc[master["industry33_name"].isin(industries)].copy()
    if master["code_norm"].duplicated().any():
        duplicates = master.loc[master["code_norm"].duplicated(), "code_norm"].tolist()
        raise ValueError(f"JPX current master contains duplicate 33-industry mappings: {duplicates[:5]}")
    code_to_industry = dict(zip(master["code_norm"], master["industry33_name"], strict=True))
    tickers_by_industry: dict[str, list[str]] = defaultdict(list)
    for ticker in cap_panel.columns:
        industry = code_to_industry.get(str(ticker).removesuffix(".T"))
        if industry is not None:
            tickers_by_industry[industry].append(str(ticker))

    dates = pd.DatetimeIndex(pd.to_datetime(cap_panel.index)).tz_localize(None).normalize()
    cap_panel = cap_panel.copy()
    cap_panel.index = dates
    cap_panel = cap_panel.sort_index()
    returns = pd.DataFrame(index=cap_panel.index, columns=industries, dtype=float)
    counts: dict[str, int] = {}

    for industry in industries:
        tickers = sorted(tickers_by_industry.get(industry, []))
        counts[industry] = len(tickers)
        if not tickers:
            raise ValueError(f"No selected current basket constituents for JPX industry {industry}")
        ret_cols: list[np.ndarray] = []
        cap_cols: list[np.ndarray] = []
        for ticker in tickers:
            code = ticker.removesuffix(".T")
            path = RAW_PRICE_DIR / f"{code}.parquet"
            if not path.exists():
                continue
            raw = pd.read_parquet(path, columns=["open", "close"])
            raw.index = pd.to_datetime(raw.index).tz_localize(None).normalize()
            raw = raw.reindex(cap_panel.index)
            open_px = raw["open"].to_numpy(dtype=float)
            close_px = raw["close"].to_numpy(dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                daily_return = close_px / open_px - 1.0
            prior_cap = cap_panel[ticker].astype(float).shift(1).to_numpy(dtype=float)
            valid = np.isfinite(daily_return) & np.isfinite(prior_cap) & (prior_cap > 0.0)
            ret_cols.append(np.where(valid, daily_return, 0.0))
            cap_cols.append(np.where(valid, prior_cap, 0.0))
        if not ret_cols:
            raise ValueError(f"No raw price files found for JPX industry {industry}")
        ret_matrix = np.column_stack(ret_cols)
        cap_matrix = np.column_stack(cap_cols)
        denominator = cap_matrix.sum(axis=1)
        numerator = (ret_matrix * cap_matrix).sum(axis=1)
        returns[industry] = np.divide(
            numerator,
            denominator,
            out=np.full(len(denominator), np.nan, dtype=float),
            where=denominator > 0.0,
        )
    return returns, counts, dict(tickers_by_industry)


def _build_weight_maps(
    industry_rows: list[list[Any]],
    parent_tickers: dict[str, str],
    cap_table: pd.DataFrame,
) -> tuple[dict[int, np.ndarray], dict[int, str], list[str], list[str]]:
    industries = [str(row[0]) for row in industry_rows]
    parents = list(parent_tickers)
    caps_by_year = {
        int(year): dict(zip(group["industry"].astype(str), group["market_cap_million_yen"].astype(float), strict=True))
        for year, group in cap_table.groupby("snapshot_year", sort=True)
    }
    weight_maps: dict[int, np.ndarray] = {}
    for snapshot_year, caps in caps_by_year.items():
        matrix = np.zeros((len(parents), len(industries)), dtype=float)
        for parent_idx, parent in enumerate(parents):
            member_indices = [i for i, row in enumerate(industry_rows) if str(row[1]) == parent]
            if not member_indices:
                raise ValueError(f"No JPX33 children map to TOPIX-17 parent {parent}")
            total = sum(caps[industries[i]] for i in member_indices)
            if total <= 0.0:
                raise ValueError(f"Non-positive cap for TOPIX-17 parent {parent} in {snapshot_year}")
            for industry_idx in member_indices:
                matrix[parent_idx, industry_idx] = caps[industries[industry_idx]] / total
        if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-12, rtol=0.0):
            raise ValueError(f"JPX33-to-17 weight rows do not sum to one for {snapshot_year}")
        weight_maps[snapshot_year] = matrix
    first_snapshot_by_year: dict[int, int] = {}
    required = set(range(2015, 2027))
    for trade_year in required:
        snapshot_year = trade_year - 2 if trade_year == 2015 else trade_year - 1
        if snapshot_year in weight_maps:
            first_snapshot_by_year[trade_year] = snapshot_year
        else:
            # 2015's first session has no 2013 cap snapshot and will be excluded.
            first_snapshot_by_year[trade_year] = -1
    return weight_maps, {k: str(v) for k, v in first_snapshot_by_year.items()}, industries, parents


def _build_v0_33(
    labels_by_industry: dict[str, dict[str, float]],
    industry_order: list[str],
) -> np.ndarray:
    base = build_base_vectors(N_US, len(industry_order))
    basis = [base["v1"], base["v2"]]
    columns = [base["v1"], base["v2"]]
    for factor in ("w3", "w4", "w5", "w6"):
        values = np.array(
            [SENSITIVITY_LABELS[ticker][factor] for ticker in US_TICKERS]
            + [labels_by_industry[industry][factor] for industry in industry_order],
            dtype=float,
        )
        vector = _orthogonalize_and_normalize(values, basis)
        if np.linalg.norm(vector) <= 1e-10:
            raise ValueError(f"Custom 33D static prior has a degenerate {factor} vector")
        columns.append(vector)
        basis.append(vector)
    return np.column_stack(columns)


def _labels_by_industry(
    rows: list[list[Any]],
    use_user_labels: bool,
    industry_to_parent: dict[str, str],
    parent_tickers: dict[str, str],
) -> dict[str, dict[str, float]]:
    if use_user_labels:
        return {
            str(row[0]): {factor: float(row[idx]) for factor, idx in (("w3", 1), ("w4", 2), ("w5", 3), ("w6", 4))}
            for row in rows
        }
    return {
        industry: copy.deepcopy(SENSITIVITY_LABELS[parent_tickers[industry_to_parent[industry]]])
        for industry in industry_to_parent
    }


def _block_ci(values: np.ndarray, seed: int) -> list[float]:
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or len(values) < BOOTSTRAP_BLOCK:
        raise ValueError("Bootstrap input must be finite and at least one block long")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(len(values) / BOOTSTRAP_BLOCK))
    max_start = len(values) - BOOTSTRAP_BLOCK
    samples = np.empty(BOOTSTRAP_RESAMPLES, dtype=float)
    for i in range(BOOTSTRAP_RESAMPLES):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        sample = np.concatenate([values[start : start + BOOTSTRAP_BLOCK] for start in starts])[: len(values)]
        samples[i] = float(np.mean(sample))
    return [float(x) for x in np.quantile(samples, [0.025, 0.975])]


def _safe_rank_ic(prediction: np.ndarray, target: np.ndarray) -> float:
    if np.ptp(prediction) <= 1e-14 or np.ptp(target) <= 1e-14:
        return 0.0
    value = spearmanr(prediction, target).statistic
    return float(value) if np.isfinite(value) else 0.0


def _daily_metrics(
    predictions: dict[str, np.ndarray],
    target: np.ndarray,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for i, date in enumerate(dates):
        valid_target = np.isfinite(target[i])
        if int(valid_target.sum()) < 10:
            continue
        row: dict[str, Any] = {"trade_date": date.strftime("%Y-%m-%d"), "observed_etfs": int(valid_target.sum())}
        for name, forecast in predictions.items():
            valid = valid_target & np.isfinite(forecast[i])
            if int(valid.sum()) < 10:
                raise ValueError(f"Insufficient comparable ETFs for {name} on {date.date()}")
            x = forecast[i, valid]
            actual = target[i, valid]
            error = x - actual
            row[f"{name}_rank_ic"] = _safe_rank_ic(x, actual)
            row[f"{name}_mae"] = float(np.mean(np.abs(error)))
            row[f"{name}_rmse"] = float(np.sqrt(np.mean(error**2)))
            row[f"{name}_sign_accuracy"] = float(np.mean(np.sign(x) == np.sign(actual)))
        rows.append(row)
    daily = pd.DataFrame(rows)
    if daily.empty:
        raise ValueError("No paired 17-ETF forecast dates passed evaluation checks")
    return daily


def _summarize(daily: pd.DataFrame, variant: str, baseline: str = "direct17") -> dict[str, Any]:
    rank_delta = daily[f"{variant}_rank_ic"].to_numpy() - daily[f"{baseline}_rank_ic"].to_numpy()
    mae_delta = daily[f"{variant}_mae"].to_numpy() - daily[f"{baseline}_mae"].to_numpy()
    rmse_delta = daily[f"{variant}_rmse"].to_numpy() - daily[f"{baseline}_rmse"].to_numpy()
    return {
        "paired_dates": int(len(daily)),
        "baseline_mean_rank_ic": float(daily[f"{baseline}_rank_ic"].mean()),
        "candidate_mean_rank_ic": float(daily[f"{variant}_rank_ic"].mean()),
        "mean_rank_ic_difference": float(rank_delta.mean()),
        "rank_ic_difference_ci95": _block_ci(rank_delta, BOOTSTRAP_SEED),
        "baseline_mean_mae_bps": float(daily[f"{baseline}_mae"].mean() * 10000.0),
        "candidate_mean_mae_bps": float(daily[f"{variant}_mae"].mean() * 10000.0),
        "mean_mae_difference_bps": float(mae_delta.mean() * 10000.0),
        "mae_difference_ci95_bps": [x * 10000.0 for x in _block_ci(mae_delta, BOOTSTRAP_SEED + 1)],
        "baseline_mean_rmse_bps": float(daily[f"{baseline}_rmse"].mean() * 10000.0),
        "candidate_mean_rmse_bps": float(daily[f"{variant}_rmse"].mean() * 10000.0),
        "mean_rmse_difference_bps": float(rmse_delta.mean() * 10000.0),
        "rmse_difference_ci95_bps": [x * 10000.0 for x in _block_ci(rmse_delta, BOOTSTRAP_SEED + 2)],
        "baseline_sign_accuracy": float(daily[f"{baseline}_sign_accuracy"].mean()),
        "candidate_sign_accuracy": float(daily[f"{variant}_sign_accuracy"].mean()),
        "exploratory_accuracy_gate_passed": bool(
            _block_ci(rank_delta, BOOTSTRAP_SEED)[0] > 0.0 and mae_delta.mean() <= 0.0
        ),
    }


def _make_model(
    blpx_cfg: Any,
    n_j: int,
    model_name: str,
    *,
    candidate_mapping: np.ndarray | None = None,
    industry_order: list[str] | None = None,
    industry_to_parent: dict[str, str] | None = None,
    parent_tickers: dict[str, str] | None = None,
) -> ProductionBLPXModel:
    cfg = blpx_cfg.model_copy(update={"n_j": n_j, "model_name": model_name})
    model = ProductionBLPXModel(cfg)
    model.clear_caches()
    if candidate_mapping is not None:
        assert industry_order is not None and industry_to_parent is not None and parent_tickers is not None
        parent_model = ProductionBLPXModel(blpx_cfg)
        fixed_map = np.asarray(candidate_mapping, dtype=float).T @ parent_model.M_sector
        if fixed_map.shape != (n_j, N_US):
            raise ValueError(f"Expanded static sector prior has the wrong shape: {fixed_map.shape}")
        model.M_sector = fixed_map
        model._M_sector_fixed = fixed_map.copy()
        ticker_to_parent = {ticker: parent for parent, ticker in parent_tickers.items()}
        expanded_indices: dict[int, list[int]] = {}
        for us_ticker, parent_list in model._SECTOR_MAPPING_STRUCTURE.items():
            if us_ticker not in US_TICKERS:
                continue
            parent_names = {ticker_to_parent[ticker] for ticker in parent_list if ticker in ticker_to_parent}
            expanded_indices[US_TICKERS.index(us_ticker)] = [
                i for i, industry in enumerate(industry_order) if industry_to_parent[industry] in parent_names
            ]
        model._sector_mapping_indices = expanded_indices
    return model


def _forecast(
    model: ProductionBLPXModel,
    all_returns: np.ndarray,
    c_full: np.ndarray,
    v0: np.ndarray,
    eval_positions: np.ndarray,
    *,
    output_mapping: dict[int, np.ndarray] | None,
    snapshot_by_date: dict[pd.Timestamp, int],
    dates: pd.DatetimeIndex,
    name: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    output_dimension = model.n_j if output_mapping is None else len(JP_TICKERS)
    output = np.full((len(eval_positions), output_dimension), np.nan)
    covariance_diag = {"psd_failures": 0, "minimum_eigenvalue": None, "mean_trace": 0.0, "count": 0}
    trace_total = 0.0
    for row_i, current_index in enumerate(eval_positions):
        result = model.compute_blp_signal(
            all_returns=all_returns,
            current_index=int(current_index),
            gap_override=None,
            betas_t=None,
            topix_night_t=None,
            v0_static=v0,
            c_full=c_full,
            is_residual=True,
            return_matrices=True,
        )
        mu, omega = build_raw_distribution(result, vol_adjusted_target=bool(model.vol_adjusted_target))
        mu = np.asarray(mu, dtype=float)
        omega = np.asarray(omega, dtype=float)
        if output_mapping is None:
            if mu.shape != (len(JP_TICKERS),):
                raise ValueError(f"Unexpected direct17 forecast shape {mu.shape}")
            mapped_mu, mapped_omega = mu, omega
        else:
            trade_date = pd.Timestamp(dates[current_index])
            snapshot_year = snapshot_by_date[trade_date]
            mapping = output_mapping[snapshot_year]
            mapped_mu = mapping @ mu
            mapped_omega = mapping @ omega @ mapping.T
        if not np.isfinite(mapped_mu).all() or not np.isfinite(mapped_omega).all():
            raise ValueError(f"Non-finite {name} distribution on {dates[current_index].date()}")
        eigenvalues = np.linalg.eigvalsh(0.5 * (mapped_omega + mapped_omega.T))
        if float(eigenvalues.min()) < -1e-9:
            covariance_diag["psd_failures"] += 1
        current_min = float(eigenvalues.min())
        if covariance_diag["minimum_eigenvalue"] is None or current_min < covariance_diag["minimum_eigenvalue"]:
            covariance_diag["minimum_eigenvalue"] = current_min
        trace_total += float(np.trace(mapped_omega))
        covariance_diag["count"] += 1
        output[row_i] = mapped_mu
        if (row_i + 1) % 250 == 0 or row_i + 1 == len(eval_positions):
            LOG.info("%s forecasts %d/%d", name, row_i + 1, len(eval_positions))
    covariance_diag["mean_trace"] = trace_total / max(1, covariance_diag["count"])
    return output, covariance_diag


def _annual_summary(daily: pd.DataFrame, variants: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    work = daily.copy()
    work["year"] = pd.to_datetime(work["trade_date"]).dt.year
    for year, group in work.groupby("year", sort=True):
        base_rank = float(group["direct17_rank_ic"].mean())
        base_mae = float(group["direct17_mae"].mean() * 10000.0)
        row: dict[str, Any] = {
            "year": int(year),
            "paired_dates": int(len(group)),
            "direct17_rank_ic": base_rank,
            "direct33_parent_labels_rank_ic": float(group["direct33_parent_labels_rank_ic"].mean()),
            "direct33_parent_labels_rank_ic_delta": float(group["direct33_parent_labels_rank_ic"].mean()) - base_rank,
            "direct17_mae_bps": base_mae,
            "direct33_parent_labels_mae_bps": float(group["direct33_parent_labels_mae"].mean() * 10000.0),
            "direct33_parent_labels_mae_delta_bps": float(group["direct33_parent_labels_mae"].mean() * 10000.0) - base_mae,
            "direct33_user_labels_rank_ic": float(group["direct33_user_labels_rank_ic"].mean()),
            "direct33_user_labels_rank_ic_delta": float(group["direct33_user_labels_rank_ic"].mean()) - base_rank,
            "direct33_user_labels_mae_bps": float(group["direct33_user_labels_mae"].mean() * 10000.0),
            "direct33_user_labels_mae_delta_bps": float(group["direct33_user_labels_mae"].mean() * 10000.0) - base_mae,
        }
        rows.append(row)
    columns = [
        "year", "paired_dates", "direct17_rank_ic",
        "direct33_parent_labels_rank_ic", "direct33_parent_labels_rank_ic_delta",
        "direct17_mae_bps", "direct33_parent_labels_mae_bps", "direct33_parent_labels_mae_delta_bps",
        "direct33_user_labels_rank_ic", "direct33_user_labels_rank_ic_delta",
        "direct33_user_labels_mae_bps", "direct33_user_labels_mae_delta_bps",
    ]
    return pd.DataFrame(rows, columns=columns)


def _refresh_saved_report() -> None:
    """Fix/report annual aggregates from the saved paired daily evidence only."""
    summary_path = RESULTS_DIR / "summary.json"
    daily_path = RESULTS_DIR / "daily_metrics.csv"
    report_path = REPORT_DIR / "report.md"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    daily = pd.read_csv(daily_path)
    annual = _annual_summary(daily, ["direct17", "direct33_parent_labels", "direct33_user_labels"])
    annual.to_csv(RESULTS_DIR / "annual_metrics.csv", index=False)

    reference_daily_path = ROOT / "var/results/20260924_sensitivity_pipeline_audit_long/daily_current.csv"
    if reference_daily_path.exists():
        reference_dates = set(pd.read_csv(reference_daily_path, usecols=["trade_date"])["trade_date"].astype(str))
        candidate_dates = set(daily["trade_date"].astype(str))
        cap_excluded = set(summary["evaluation"].get("excluded_dates_missing_prior_year_minus_two_cap", []))
        proxy_excluded = sorted(reference_dates - candidate_dates - cap_excluded)
        summary["evaluation"]["reference_direct17_dates"] = len(reference_dates)
        summary["evaluation"]["excluded_dates_missing_industry_proxy"] = proxy_excluded
        summary["evaluation"]["industry_proxy_coverage_end_date"] = str(pd.Timestamp(max(candidate_dates)).date())
        summary["evaluation"]["excluded_after_2026_08_21_count"] = sum(date > "2026-08-21" for date in proxy_excluded)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    if not report_path.exists():
        raise FileNotFoundError(f"Cannot refresh missing report: {report_path}")
    report = report_path.read_text(encoding="utf-8")
    table_start = report.index("| 年 | 日数 |")
    table_end = report.index("\n## データ品質", table_start)
    table = [
        "| 年 | 日数 | 17 Rank IC | 33親ラベル Rank IC | 差 | 17 MAE (bp) | 33親ラベル MAE (bp) | 差 (bp) | 33提案ラベル Rank IC | 差 | 33提案ラベル MAE (bp) | 差 (bp) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in annual.iterrows():
        table.append(
            f"| {int(row['year'])} | {int(row['paired_dates'])} | {row['direct17_rank_ic']:+.4f} "
            f"| {row['direct33_parent_labels_rank_ic']:+.4f} | {row['direct33_parent_labels_rank_ic_delta']:+.4f} "
            f"| {row['direct17_mae_bps']:.2f} | {row['direct33_parent_labels_mae_bps']:.2f} "
            f"| {row['direct33_parent_labels_mae_delta_bps']:+.2f} | {row['direct33_user_labels_rank_ic']:+.4f} "
            f"| {row['direct33_user_labels_rank_ic_delta']:+.4f} | {row['direct33_user_labels_mae_bps']:.2f} "
            f"| {row['direct33_user_labels_mae_delta_bps']:+.2f} |"
        )
    report = report[:table_start] + "\n".join(table) + report[table_end:]
    report = report.replace("494銘柄（494銘柄）", "494銘柄")
    if "excluded_dates_missing_industry_proxy" in summary["evaluation"]:
        missing = summary["evaluation"]["excluded_dates_missing_industry_proxy"]
        end_date = summary["evaluation"]["industry_proxy_coverage_end_date"]
        marker = "- 指標は営業日ごとの17ETF横断Rank IC、MAE、RMSE、方向正解率。"
        detail = (
            f"- 対象期間の17ETF基準日{summary['evaluation']['reference_direct17_dates']}日に対し、33業種proxyの有効な最後の日は{end_date}。"
            f"業種proxy欠損で{len(missing)}日（うち2026-08-21以降{summary['evaluation']['excluded_after_2026_08_21_count']}日）を除外し、"
            "2015-01-05は前年ウェイト取得不能で除外した。"
        )
        if detail not in report:
            report = report.replace(marker, detail + "\n" + marker)
    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true", help="Validate one paired forecast per variant")
    parser.add_argument("--report-only", action="store_true", help="Rebuild annual tables from saved daily results")
    args = parser.parse_args()
    if args.report_only:
        _refresh_saved_report()
        LOG.info("Refreshed report=%s", REPORT_DIR / "report.md")
        return
    started_at = datetime.now(UTC)
    started_clock = time.monotonic()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    config = _read_yaml(CONFIG_PATH)
    candidate_cfg = _read_yaml(INDUSTRY_LABEL_PATH)
    parent_cfg = _read_yaml(PARENT_MAP_PATH)
    industries_rows = candidate_cfg["industries"]["rows"]
    industry_order = [str(row[0]) for row in industries_rows]
    industry_to_parent = {str(row[0]): str(row[1]) for row in parent_cfg["industries"]["rows"]}
    parent_tickers = {str(k): str(v) for k, v in parent_cfg["sensitivity_grid"]["jp_tickers"].items()}
    if set(industry_order) != set(industry_to_parent):
        raise ValueError("Frozen sensitivity labels and parent-map do not cover the same 33 industries")
    if set(parent_tickers.values()) != set(JP_TICKERS):
        raise ValueError("JPX33-to-TOPIX17 parents differ from the canonical 17 ETF universe")
    ordered_parents = [parent for parent, ticker in parent_tickers.items() if ticker in JP_TICKERS]
    if [parent_tickers[parent] for parent in ordered_parents] != list(JP_TICKERS):
        raise ValueError("JPX33 parent-map row order differs from canonical JP_TICKERS order")

    cap_table = pd.read_csv(CAP_PATH)
    cap_years = set(cap_table["snapshot_year"].astype(int))
    if cap_years != set(range(2014, 2026)):
        raise ValueError(f"Expected complete official JPX cap snapshots 2014-2025, got {sorted(cap_years)}")
    weight_maps, first_snapshot_by_year, industries, parents = _build_weight_maps(
        parent_cfg["industries"]["rows"], parent_tickers, cap_table
    )
    if industries != industry_order:
        raise ValueError("Frozen industry sensitivity labels and parent map use a different industry order")
    cap_panel = pd.read_parquet(CAP_PANEL_PATH)
    industry_proxy, industry_counts, tickers_by_industry = _load_industry_proxy(cap_panel, industries)

    app = load_config_from_yaml(ROOT / config["inputs"]["production_config"], strict=True)
    market = load_df_exec_from_local_cache(max_stale_bdays=None).copy()
    market.index = pd.to_datetime(market.index).tz_localize(None).normalize()
    market = market.loc[(market.index >= BASELINE_START) & (market.index <= END_DATE)].sort_index()
    if market.index.has_duplicates:
        raise ValueError("Local df_exec has duplicate dates")
    industry_proxy = industry_proxy.reindex(market.index)
    us_raw = market[[f"us_cc_{ticker}" for ticker in US_TICKERS]].to_numpy(dtype=float)
    y17_raw = market[[f"jp_oc_{ticker}" for ticker in JP_TICKERS]].to_numpy(dtype=float)
    y33_raw = industry_proxy.to_numpy(dtype=float)
    topix = market["topix_oc_return"].to_numpy(dtype=float)
    if y33_raw.shape[1] != 33:
        raise ValueError("JPX33 proxy dimensionality is invalid")
    eval_data_mask = (market.index >= START_DATE) & (market.index <= END_DATE)
    if not np.isfinite(us_raw[eval_data_mask]).all() or not np.isfinite(topix[eval_data_mask]).all():
        raise ValueError("Evaluation-period US input or TOPIX open-to-close data contains missing values")

    beta_window = int(app.v2.blpx.beta_window)
    beta17 = compute_rolling_ols_betas(y17_raw, topix.reshape(-1, 1), beta_window)[:, :, 0]
    beta33 = compute_rolling_ols_betas(y33_raw, topix.reshape(-1, 1), beta_window)[:, :, 0]
    y17_res = y17_raw - beta17 * topix.reshape(-1, 1)
    y33_res = y33_raw - beta33 * topix.reshape(-1, 1)
    all17 = np.column_stack([us_raw, y17_res])
    all33 = np.column_stack([us_raw, y33_res])

    base17_rows = (
        (market.index >= BASELINE_START)
        & (market.index <= BASELINE_END)
        & np.isfinite(all17).all(axis=1)
    )
    base33_rows = (
        (market.index >= BASELINE_START)
        & (market.index <= BASELINE_END)
        & np.isfinite(all33).all(axis=1)
    )
    if int(base17_rows.sum()) < 1000 or int(base33_rows.sum()) < 1000:
        raise ValueError(
            f"Fixed 2010-2014 complete-case prior has too few rows: 17D={base17_rows.sum()}, 33D={base33_rows.sum()}"
        )
    c_full_17 = compute_baseline_correlation(
        all17, market.index.values, ewma_half_life=app.v2.blpx.ewma_halflife
    )
    c_full_33 = compute_baseline_correlation(
        all33, market.index.values, ewma_half_life=app.v2.blpx.ewma_halflife
    )

    current_labels = {
        industry: {factor: float(SENSITIVITY_LABELS[parent_tickers[industry_to_parent[industry]]][factor])
                  for factor in ("w3", "w4", "w5", "w6")}
        for industry in industries
    }
    user_labels = _labels_by_industry(industries_rows, True, industry_to_parent, parent_tickers)
    v0_17 = build_v3_static(len(US_TICKERS), len(JP_TICKERS), include_v4=bool(app.v2.blpx.include_v4_prior))
    v0_33_current = _build_v0_33(current_labels, industries)
    v0_33_user = _build_v0_33(user_labels, industries)
    model17 = _make_model(app.v2.blpx, len(JP_TICKERS), "Direct17ResidualBLPX")
    model33_current = _make_model(
        app.v2.blpx,
        len(industries),
        "Direct33ParentLabelsResidualBLPX",
        candidate_mapping=weight_maps[2014],
        industry_order=industries,
        industry_to_parent=industry_to_parent,
        parent_tickers=parent_tickers,
    )
    model33_user = _make_model(
        app.v2.blpx,
        len(industries),
        "Direct33UserLabelsResidualBLPX",
        candidate_mapping=weight_maps[2014],
        industry_order=industries,
        industry_to_parent=industry_to_parent,
        parent_tickers=parent_tickers,
    )

    eval_mask = (market.index >= START_DATE) & (market.index <= END_DATE)
    eval_dates = pd.DatetimeIndex(market.index[eval_mask])
    first_sessions = {int(year): pd.Timestamp(ds[0]) for year in sorted(set(eval_dates.year))
                      if len(ds := eval_dates[eval_dates.year == year]) > 0}
    snapshot_by_date: dict[pd.Timestamp, int] = {}
    included: list[pd.Timestamp] = []
    excluded_first_session_no_snapshot: list[str] = []
    excluded_missing_direct17: list[str] = []
    excluded_missing_industry_proxy: list[str] = []
    for date in eval_dates:
        trade_year = int(date.year)
        snapshot_year = trade_year - 2 if date == first_sessions[trade_year] else trade_year - 1
        if snapshot_year not in weight_maps:
            excluded_first_session_no_snapshot.append(date.strftime("%Y-%m-%d"))
            continue
        row = market.index.get_loc(date)
        if not np.isfinite(all17[row]).all():
            excluded_missing_direct17.append(date.strftime("%Y-%m-%d"))
            continue
        if not np.isfinite(all33[row]).all():
            excluded_missing_industry_proxy.append(date.strftime("%Y-%m-%d"))
            continue
        snapshot_by_date[date] = snapshot_year
        included.append(date)
    paired_dates = pd.DatetimeIndex(included)
    eval_positions = market.index.get_indexer(paired_dates)
    if len(paired_dates) < 1000:
        raise ValueError(f"Too few valid paired forecast dates: {len(paired_dates)}")

    if args.preflight:
        for model, returns, c_full, v0, mapping, name in (
            (model17, all17, c_full_17, v0_17, None, "direct17"),
            (model33_current, all33, c_full_33, v0_33_current, weight_maps, "direct33_parent_labels"),
            (model33_user, all33, c_full_33, v0_33_user, weight_maps, "direct33_user_labels"),
        ):
            _, diag = _forecast(
                model, returns, c_full, v0, eval_positions[:1], output_mapping=mapping,
                snapshot_by_date=snapshot_by_date, dates=market.index, name=f"preflight_{name}",
            )
            LOG.info("Preflight %s covariance=%s", name, diag)
        LOG.info("Preflight passed; dates=%d, first=%s", len(paired_dates), paired_dates[0].date())
        return

    prediction17, cov17 = _forecast(
        model17, all17, c_full_17, v0_17, eval_positions,
        output_mapping=None, snapshot_by_date=snapshot_by_date, dates=market.index, name="direct17",
    )
    prediction33_current, cov33_current = _forecast(
        model33_current, all33, c_full_33, v0_33_current, eval_positions,
        output_mapping=weight_maps, snapshot_by_date=snapshot_by_date, dates=market.index,
        name="direct33_parent_labels",
    )
    prediction33_user, cov33_user = _forecast(
        model33_user, all33, c_full_33, v0_33_user, eval_positions,
        output_mapping=weight_maps, snapshot_by_date=snapshot_by_date, dates=market.index,
        name="direct33_user_labels",
    )

    prediction_arrays = {
        "direct17": prediction17,
        "direct33_parent_labels": prediction33_current,
        "direct33_user_labels": prediction33_user,
    }
    target = y17_res[eval_positions]
    daily = _daily_metrics(prediction_arrays, target, paired_dates)
    daily_path = RESULTS_DIR / "daily_metrics.csv"
    daily.to_csv(daily_path, index=False)
    annual = _annual_summary(daily, list(prediction_arrays))
    annual.to_csv(RESULTS_DIR / "annual_metrics.csv", index=False)

    forecast_rows = []
    for i, date in enumerate(paired_dates):
        for variant, values in prediction_arrays.items():
            for j, ticker in enumerate(JP_TICKERS):
                forecast_rows.append({"trade_date": date.strftime("%Y-%m-%d"), "variant": variant,
                                      "ticker": ticker, "forecast_residual": float(values[i, j]),
                                      "target_residual": float(target[i, j]),
                                      "snapshot_year": snapshot_by_date[date]})
    pd.DataFrame(forecast_rows).to_csv(RESULTS_DIR / "forecasts.csv", index=False)

    map_rows = []
    for snapshot_year, matrix in sorted(weight_maps.items()):
        for parent_idx, parent in enumerate(parents):
            for industry_idx, industry in enumerate(industries):
                weight = float(matrix[parent_idx, industry_idx])
                if weight > 0.0:
                    map_rows.append({"snapshot_year": snapshot_year, "topix17_parent": parent,
                                     "ticker": parent_tickers[parent], "industry33": industry,
                                     "weight": weight})
    pd.DataFrame(map_rows).to_csv(RESULTS_DIR / "mapping_weights.csv", index=False)

    # Structural quality check: how well the official-cap mapping of the
    # current-member stock baskets reproduces each tradable TOPIX-17 ETF.
    raw_proxy17 = np.full((len(paired_dates), len(JP_TICKERS)), np.nan)
    for i, (date, position) in enumerate(zip(paired_dates, eval_positions, strict=True)):
        raw_proxy17[i] = weight_maps[snapshot_by_date[date]] @ y33_raw[position]
    proxy_quality_rows = []
    y17_eval_raw = y17_raw[eval_positions]
    for j, ticker in enumerate(JP_TICKERS):
        mask = np.isfinite(raw_proxy17[:, j]) & np.isfinite(y17_eval_raw[:, j])
        corr = float(np.corrcoef(raw_proxy17[mask, j], y17_eval_raw[mask, j])[0, 1]) if mask.sum() >= 30 else float("nan")
        proxy_quality_rows.append({"ticker": ticker, "paired_dates": int(mask.sum()), "pearson_corr": corr})
    pd.DataFrame(proxy_quality_rows).to_csv(RESULTS_DIR / "mapped_proxy_quality.csv", index=False)

    # Selected current-basket market-cap coverage by industry on the latest
    # available 2025 year-end cap-panel date.
    cap_index = pd.DatetimeIndex(pd.to_datetime(cap_panel.index)).tz_localize(None).normalize()
    cap_panel.index = cap_index
    latest_cap_date = cap_panel.index[cap_panel.index <= pd.Timestamp("2025-12-31")].max()
    official_2025 = cap_table.loc[cap_table["snapshot_year"] == 2025].set_index("industry")["market_cap_million_yen"]
    coverage_rows = []
    for industry in industries:
        tickers = tickers_by_industry.get(industry, [])
        selected_cap_m = float(cap_panel.loc[latest_cap_date, tickers].sum(skipna=True) / 1e6) if tickers else 0.0
        official_cap_m = float(official_2025[industry])
        coverage_rows.append({"industry33": industry, "selected_tickers": industry_counts[industry],
                              "selected_cap_million_yen": selected_cap_m,
                              "official_cap_million_yen": official_cap_m,
                              "selected_cap_coverage": selected_cap_m / official_cap_m if official_cap_m else None})
    pd.DataFrame(coverage_rows).to_csv(RESULTS_DIR / "industry_coverage.csv", index=False)

    metric_summaries = {
        "direct33_parent_labels": _summarize(daily, "direct33_parent_labels"),
        "direct33_user_labels": _summarize(daily, "direct33_user_labels"),
    }
    prior_records = list(ExperimentRegistry(REGISTRY_PATH))
    related_records = [
        record.name for record in prior_records
        if any(token in record.name.lower() for token in ("jpx33", "subsector"))
    ]
    config_hash = hashlib.sha256(
        CONFIG_PATH.read_bytes() + INDUSTRY_LABEL_PATH.read_bytes() + PARENT_MAP_PATH.read_bytes()
        + CAP_PATH.read_bytes()
    ).hexdigest()
    user_panel_fingerprint = dataframe_fingerprint(industry_proxy)
    market_fingerprint = dataframe_fingerprint(market)
    proxy_corr_values = [r["pearson_corr"] for r in proxy_quality_rows if np.isfinite(r["pearson_corr"])]
    cap_cover_values = [float(r["selected_cap_coverage"]) for r in coverage_rows]
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": time.monotonic() - started_clock,
        "decision": "pending; retrospective proxy results cannot establish PIT 33-industry direct-prediction performance",
        "evaluation": {
            "start_date": str(paired_dates.min().date()),
            "end_date": str(paired_dates.max().date()),
            "paired_dates": int(len(paired_dates)),
            "calendar_years": sorted(int(x) for x in paired_dates.year.unique()),
            "target": "TOPIX-17 open-to-close returns residualized against TOPIX open-to-close; neither realized 09:10 quotes nor PnL",
            "baseline_period": [BASELINE_START, BASELINE_END],
            "baseline_complete_rows_17": int(base17_rows.sum()),
            "baseline_complete_rows_33": int(base33_rows.sum()),
            "beta_window": beta_window,
            "annualization": None,
            "bootstrap": {"method": "paired non-circular moving-block", "block_length": BOOTSTRAP_BLOCK,
                          "resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
            "first_session_cap_schedule": first_snapshot_by_year,
            "excluded_dates_missing_prior_year_minus_two_cap": excluded_first_session_no_snapshot,
            "reference_direct17_dates": int(len(eval_dates)),
            "excluded_dates_missing_direct17": excluded_missing_direct17,
            "excluded_dates_missing_industry_proxy": excluded_missing_industry_proxy,
            "industry_proxy_coverage_end_date": str(
                market.index[eval_mask & np.isfinite(all33).all(axis=1)].max().date()
            ),
            "excluded_after_2026_08_21_count": int(
                sum(date > "2026-08-21" for date in excluded_missing_industry_proxy)
            ),
        },
        "variants": {"direct17": {
            "description": "Production-parameter residual BLPX, current 17-ETF targets and labels",
            "mean_rank_ic": float(daily["direct17_rank_ic"].mean()),
            "mean_mae_bps": float(daily["direct17_mae"].mean() * 10000.0),
            "mean_rmse_bps": float(daily["direct17_rmse"].mean() * 10000.0),
        }, **metric_summaries},
        "covariance_mapping": {
            "direct17": cov17,
            "direct33_parent_labels": cov33_current,
            "direct33_user_labels": cov33_user,
            "definition": "Omega_17 = W_t Omega_33 W_t'",
        },
        "industry_proxy": {
            "universe": "494 current-member tickers with valid latest shares and 2026-07-31 JPX 33-industry snapshot",
            "classification_snapshot_date": str(pd.to_datetime(pd.read_csv(MASTER_PATH, dtype=str)["日付"], format="%Y%m%d").max().date()),
            "industry_counts": industry_counts,
            "cap_coverage_median_2025": float(np.median(cap_cover_values)),
            "cap_coverage_min_2025": float(np.min(cap_cover_values)),
            "mapped_proxy_etf_correlation_median": float(np.median(proxy_corr_values)),
            "mapped_proxy_etf_correlation_min": float(np.min(proxy_corr_values)),
            "mapped_proxy_etf_correlation_below_0_70": int(sum(x < 0.70 for x in proxy_corr_values)),
            "point_in_time_membership_available": False,
            "point_in_time_classification_available": False,
            "historical_09_10_industry_prices_available": False,
        },
        "data_fingerprints": {"df_exec": market_fingerprint, "industry_proxy": user_panel_fingerprint,
                              "experiment_config_sha256": config_hash},
        "effective_blpx_config": app.v2.blpx.model_dump(mode="json"),
        "related_trial_count": len(related_records),
        "related_trial_names": related_records,
        "candidate_trial_count": 2,
        "all_results_retrospective": True,
        "v2_portfolio_backtest": "not run; forecast accuracy only",
        "production_modified": False,
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    annual_display = annual.copy()
    for col in annual_display.columns:
        if col.endswith("rank_ic") or col.endswith("rank_ic_delta"):
            annual_display[col] = annual_display[col].map(lambda x: f"{x:+.4f}")
        elif "mae" in col and col != "paired_dates":
            annual_display[col] = annual_display[col].map(lambda x: f"{x:+.2f}")
    rows = [
        "# JPX33を個別予測してTOPIX-17へ集約する長期診断",
        "",
        "## 判定",
        "",
        "**33業種案は予測指標が分かれ、総合的な精度向上は確認できなかった。** 主比較ではRank ICが−0.0147（95% CI [−0.0217, −0.0081]）、方向正解率も1.5pt低下した一方、MAEは1.39bp、RMSEは1.47bp改善した。33業種・構成銘柄のPIT履歴と9:10業種価格がないため、これらは代理データによる遡及診断であり採用判断には使わない。",
        "",
        "## 事前固定した比較",
        "",
        f"- 期間: {paired_dates.min().date()}〜{paired_dates.max().date()}、{len(paired_dates)}日。学習・相関窓は当日を含めず、c_fullは2010〜2014年の完全行のみ。TOPIXベータは60営業日ローリングOLSで当日を除外した。",
        "- Baseline: 現行17ETF系列・現行17銘柄ラベルで直接予測。比較対象A: 33次元で予測し現行の17銘柄感応度を各子業種へ複製。比較対象B: 上位モデル提示の33業種w3〜w6を適用。Aが次元拡張の効果を切り分ける主比較、Bは提示ラベルを使った別試行。",
        "- 33→17写像は予測平均 μ17=W_t μ33、共分散 Ω17=W_t Ω33 W_t'。W_tは親TOPIX-17業種内で正規化した前年JPX年末業種総時価総額。各年の第1取引日は前々年末値を使い、2015年の第1取引日は2013年値がなく除外した。浮動株調整ETF保有比率ではない。",
        "- 評価targetは17ETFの始値→大引けリターンからTOPIX始値→大引け成分をローリング除去した残差。候補33系列も同じ時間帯の個別株バスケットproxyから作成。実9:10 target・実約定・ポートフォリオ損益は未評価。",
        "- 指標は営業日ごとの17ETF横断Rank IC、MAE、RMSE、方向正解率。差の95%区間は対応日を保つ非循環20営業日moving-block bootstrap 5,000回。",
        "",
        "## 予測精度",
        "",
        "| モデル | paired日数 | 平均Rank IC | ΔRank IC vs 17 (95% CI) | MAE (bp) | ΔMAE vs 17 (95% CI, bp) | RMSE (bp) | 方向正解率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    direct = summary["variants"]["direct17"]
    rows.append(f"| direct17 | {len(paired_dates)} | {direct['mean_rank_ic']:+.4f} | — | {direct['mean_mae_bps']:.2f} | — | {direct['mean_rmse_bps']:.2f} | {daily['direct17_sign_accuracy'].mean():.3f} |")
    for variant, label in (("direct33_parent_labels", "direct33 / parent labels"), ("direct33_user_labels", "direct33 / proposed labels")):
        metric = metric_summaries[variant]
        rank_ci = metric["rank_ic_difference_ci95"]
        mae_ci = metric["mae_difference_ci95_bps"]
        rows.append(f"| {label} | {len(paired_dates)} | {metric['candidate_mean_rank_ic']:+.4f} | {metric['mean_rank_ic_difference']:+.4f} [{rank_ci[0]:+.4f}, {rank_ci[1]:+.4f}] | {metric['candidate_mean_mae_bps']:.2f} | {metric['mean_mae_difference_bps']:+.2f} [{mae_ci[0]:+.2f}, {mae_ci[1]:+.2f}] | {metric['candidate_mean_rmse_bps']:.2f} | {metric['candidate_sign_accuracy']:.3f} |")
    rows.extend([
        "",
        "| 年 | 日数 | 17 Rank IC | 33親ラベル Rank IC | 差 | 17 MAE (bp) | 33親ラベル MAE (bp) | 差 (bp) | 33提案ラベル Rank IC | 差 | 33提案ラベル MAE (bp) | 差 (bp) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for _, row in annual_display.iterrows():
        rows.append("| " + " | ".join(str(row[col]) for col in annual_display.columns) + " |")
    rows.extend([
        "",
        "## データ品質と写像確認",
        "",
        f"- 33業種basketは現在分類の{sum(industry_counts.values())}銘柄（494銘柄）を使用。業種ごとの選定銘柄数は{min(industry_counts.values())}〜{max(industry_counts.values())}。2025年末のJPX業種総時価総額に対する選定銘柄capカバー率は中央値{summary['industry_proxy']['cap_coverage_median_2025']:.1%}、最低{summary['industry_proxy']['cap_coverage_min_2025']:.1%}。",
        f"- 年次capウェイトで集約した33業種proxyと実ETF open-to-closeの相関は中央値{summary['industry_proxy']['mapped_proxy_etf_correlation_median']:.3f}、最低{summary['industry_proxy']['mapped_proxy_etf_correlation_min']:.3f}。17 ETF中{summary['industry_proxy']['mapped_proxy_etf_correlation_below_0_70']}本が0.70未満。対象別は `mapped_proxy_quality.csv`。",
        "- 分類表は2026-07-31の1時点のみ。上場廃止銘柄・過去のTOPIX採用入替・過去の業種変更を含むPITメンバー履歴ではない。選定494銘柄の過去リターンはサバイバーシップを含む。株数は最新時点から逆算した分割調整近似で、増資・自社株買いを反映しない。",
        f"- Ωの17次元写像は両33候補で全{len(paired_dates)}日を計算。PSD違反数は親ラベル={cov33_current['psd_failures']}, 提示ラベル={cov33_user['psd_failures']}。最小固有値はそれぞれ{cov33_current['minimum_eigenvalue']:.3e}, {cov33_user['minimum_eigenvalue']:.3e}。",
        "",
        "## 解釈・採否",
        "",
        "探索的精度ゲートはRank IC差の95%区間下限>0かつ平均MAE差≤0と定義した。過去期間はこの作業以前にも閲覧済みで、ラベル案も事前に提示されたため、基準を満たしても未使用OOSや本番採用の証拠にはならない。加えて、入力バスケットのPIT分類・完全な構成銘柄・float-adjusted W_t・9:10価格は揃っていない。今回の結果だけでproductionモデルは変更しない。",
        "33業種提案ラベル版と17親ラベル複製版の差は、感応度値と33次元 prior subspace の両方を変える。主比較のparent-label版が粒度変更の中心的診断だが、33業種株バスケット自体がPIT再現でないため、正式な粒度仮説の結論にはならない。",
        "コスト、turnover、最大DD、V2ポジション制約の損益効果、Deflated Sharpeは未評価。ここで報告するのは予測精度診断であり、戦略バックテストではない。",
        "",
        "## 再現情報",
        "",
        f"- 設定: `{CONFIG_PATH.relative_to(ROOT)}`。スクリプト: `{Path(__file__).relative_to(ROOT)}`。成果物: `{RESULTS_DIR.relative_to(ROOT)}/`。",
        f"- df_exec fingerprint: `{market_fingerprint}`。33業種proxy fingerprint: `{user_panel_fingerprint}`。実験設定SHA-256: `{config_hash}`。",
        f"- production BLPX parameters: `{json.dumps(summary['effective_blpx_config'], ensure_ascii=False, sort_keys=True)}`。本番コード・設定は変更していない。",
        f"- 関連する既登録JPX33/サブセクター試行: {len(related_records)}件。今回の候補比較は2件。主な既存参考は `reports/20260924_sensitivity_pipeline_audit/report.md` および `reports/subsector_refinement/phase2_blpx/subsector_ic_report_expanded.md`。",
        f"- 再実行: `timeout -k 20 14400 .venv/bin/python -u {Path(__file__).relative_to(ROOT)}`。",
        "",
    ])
    report_path = REPORT_DIR / "report.md"
    report_path.write_text("\n".join(rows), encoding="utf-8")

    registry = ExperimentRegistry(REGISTRY_PATH)
    ended_at = datetime.now(UTC)
    for variant, description in (
        ("direct33_parent_labels", "33 industry labels inherited from current TOPIX-17 labels"),
        ("direct33_user_labels", "User-proposed 33 industry sensitivity labels"),
    ):
        registry.record(ExperimentRecord(
            name=f"{EXPERIMENT_ID}_{variant}",
            hypothesis="Direct JPX33 forecasts projected to TOPIX-17 improve forecast accuracy versus direct 17-ETF forecasts.",
            start_time=started_at,
            end_time=ended_at,
            parameters={
                "variant": variant,
                "description": description,
                "target_window": [str(paired_dates.min().date()), str(paired_dates.max().date())],
                "target": summary["evaluation"]["target"],
                "mapping": "PIT prior-year JPX33 total market cap within TOPIX17, not float-adjusted ETF holdings",
                "industry_proxy_universe": "selected 494 current-member stocks, current 2026-07-31 industry snapshot",
                "baseline_c_full": [BASELINE_START, BASELINE_END],
                "bootstrap": summary["evaluation"]["bootstrap"],
                "data_fingerprints": summary["data_fingerprints"],
                "experiment_config_sha256": config_hash,
                "related_prior_trial_count": len(related_records),
                "related_prior_trial_names": related_records,
                "retrospective_pseudo_oos_only": True,
                "production_modified": False,
            },
            metrics=metric_summaries[variant],
            decision=Decision.PENDING,
            report_path=str(report_path.relative_to(ROOT)),
            related_records=related_records,
        ))
    LOG.info("Wrote report=%s", report_path)
    LOG.info("Wrote results=%s", RESULTS_DIR)


if __name__ == "__main__":
    main()
