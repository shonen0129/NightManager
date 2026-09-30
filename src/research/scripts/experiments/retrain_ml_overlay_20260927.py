"""Rebuild and retrain the production ML overlay from audited historical inputs.

The script keeps all candidates under var/results and never edits production
configuration or the production artifact pointer. Stages are deliberately
separate so each expensive step has a durable, reviewable checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import multiprocessing
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.config.schemas import ProductionV2RunConfig
from leadlag.data import macro as macro_data
from leadlag.data.adr_features import load_adr_features, normalize_adr_features
from leadlag.data.intraday_inputs import build_open_910_returns, compute_jp_target_returns
from leadlag.data.market_data_cache import load_raw_cache
from leadlag.data.pit_lake import PITDataLake
from leadlag.data.preprocessor import preprocess_data
from leadlag.data.rank_reversal import load_rank_reversal_frame
from leadlag.data.tickers import JP_TICKERS
from leadlag.data.validation import validate_exec_record
from leadlag.domain.inputs import DecisionInputs, HistoricalInputs
from leadlag.execution.config import load_config_from_yaml
from leadlag.experiment_registry import Decision
from leadlag.models.ml_order_overlay import DEFAULT_LGBM_KWARGS
from leadlag.models.ml_overlay_artifact import load_overlay_model, save_overlay_model
from leadlag.models.ml_overlay_features import _precompute_market_vol
from leadlag.runner.production import ProductionRunner
from leadlag.utils.dataframe_fingerprint import dataframe_fingerprint
from research.experiment_utils import record_simple_experiment
from research.experiments.ml_overlay_training import (
    _build_training_decision_model,
    _collect_training_data,
    _train_overlay_lgbm,
)

REPORT = ROOT / "reports/20260927_ml_overlay_retrain"
WORK = ROOT / "var/results/20260927_ml_overlay_retrain"
INPUT = WORK / "inputs"
GAP = ROOT / "var/live/pipeline_data/gap_adjusted_distribution"
RECONCILIATION = REPORT / "reconciliation.json"
ARTIFACT = WORK / "artifact"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def _load(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _rebuild_df_exec() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply only independently sourced open corrections and explicit exclusions."""
    if not RECONCILIATION.is_file():
        raise FileNotFoundError(RECONCILIATION)
    reconciliation = json.loads(RECONCILIATION.read_text(encoding="utf-8"))
    source_db = ROOT / "var/market_data/etf_prices.sqlite"
    source_hash = _sha256(source_db)
    if source_hash != reconciliation["raw_sqlite_sha256"]:
        raise ValueError("Raw OHLC SQLite changed since official-price reconciliation")
    raw = load_raw_cache()
    corrected = {
        key: value.copy(deep=True) if isinstance(value, pd.DataFrame) else value
        for key, value in raw.items()
    }
    for key in ("jp_open", "jp_close"):
        corrected[key].index = pd.to_datetime(corrected[key].index).tz_localize(None).normalize()
    pdf_dir = REPORT / "inputs/jpx_daily"
    cases = reconciliation["cases"]
    for case in cases:
        ymd = case["date"].replace("-", "")
        pdf_path = pdf_dir / f"{ymd}.pdf"
        if not pdf_path.is_file() or _sha256(pdf_path) != case["pdf_sha256"]:
            raise ValueError(f"JPX daily source PDF hash mismatch: {pdf_path}")
        date, ticker = pd.Timestamp(case["date"]), case["ticker"]
        if float(corrected["jp_open"].at[date, ticker]) != 0.0:
            raise ValueError(f"Expected source zero at {date.date()} {ticker}")
        if case["disposition"] == "restore_from_jpx_am_open":
            if case["am_open"] is None or case["pm_close"] is None:
                raise ValueError(f"Incomplete official price proof at {date.date()} {ticker}")
            if abs(float(case["raw_close"]) - float(case["pm_close"])) > 0.01:
                raise ValueError(f"Close basis mismatch was not excluded at {date.date()} {ticker}")
            corrected["jp_open"].at[date, ticker] = float(case["am_open"])
        elif case["disposition"] in (
            "exclude_no_am_session_quote",
            "exclude_close_price_basis_unresolved",
        ):
            corrected["jp_open"].at[date, ticker] = np.nan
        else:
            raise ValueError(f"Unknown reconciliation disposition: {case['disposition']}")

    special_repairs = reconciliation.get("additional_official_repairs", {})
    if special_repairs:
        source_pdf = pdf_dir / "20251024.pdf"
        if (
            not source_pdf.is_file()
            or _sha256(source_pdf) != special_repairs.get("source_pdf_sha256")
        ):
            raise ValueError("JPX 2025-10-24 source PDF is absent or has changed")
        prior_date = pd.Timestamp("2025-10-23")
        source_date = pd.Timestamp("2025-10-24")
        repaired_tickers: set[str] = set()
        for repair in special_repairs.get("restored_rows", []):
            if (
                repair["date"] != "2025-10-24"
                or repair["disposition"] != "restore_open_and_close_from_jpx_daily_quote"
            ):
                raise ValueError(f"Unknown official repair row: {repair}")
            ticker = repair["ticker"]
            if ticker in repaired_tickers or ticker not in JP_TICKERS:
                raise ValueError(f"Duplicate or unexpected official repair ticker: {ticker}")
            repaired_tickers.add(ticker)
            if pd.notna(corrected["jp_open"].at[source_date, ticker]) or pd.notna(
                corrected["jp_close"].at[source_date, ticker]
            ):
                raise ValueError(f"Expected missing raw prices at {source_date.date()} {ticker}")
            if abs(
                float(repair["official_previous_close"])
                - float(repair["raw_previous_close"])
            ) > 0.01 or abs(
                float(corrected["jp_close"].at[prior_date, ticker])
                - float(repair["raw_previous_close"])
            ) > 0.01:
                raise ValueError(f"Official prior-close basis mismatch for {ticker}")
            corrected["jp_open"].at[source_date, ticker] = float(repair["am_open"])
            corrected["jp_close"].at[source_date, ticker] = float(repair["pm_close"])
        if len(repaired_tickers) != 16 or "1629.T" in repaired_tickers:
            raise ValueError(f"Unexpected repaired ETF set: {sorted(repaired_tickers)}")
        scale = special_repairs.get("local_scale_exception", {})
        if scale.get("ticker") != "1629.T":
            raise ValueError("1629.T local price-scale exception is not documented")
        if abs(
            float(corrected["jp_close"].at[source_date, "1629.T"])
            - float(scale["raw_local_close"])
        ) > 0.01 or abs(
            float(corrected["jp_close"].at[prior_date, "1629.T"])
            - float(scale["raw_local_previous_close"])
        ) > 0.01:
            raise ValueError("1629.T documented local prices changed")

    base = preprocess_data(raw, strict_validation=False)
    rebuilt = preprocess_data(corrected, strict_validation=False)
    provisional_mask = rebuilt["is_provisional"].fillna(True).astype(bool)
    provisional_dates = [str(date.date()) for date in rebuilt.index[provisional_mask]]
    rebuilt = rebuilt.loc[~provisional_mask].copy()
    row_errors = []
    for date, row in rebuilt.iterrows():
        alerts = validate_exec_record(row.to_dict())
        if alerts:
            row_errors.append((str(date.date()), alerts))
    if row_errors:
        raise ValueError(f"Retained execution rows failed validation: {row_errors[:5]}")
    if rebuilt["is_provisional"].fillna(True).astype(bool).any():
        raise ValueError("Provisional rows remain after filtering")

    base_dates = set(base.index.strftime("%Y-%m-%d"))
    rebuilt_dates = set(rebuilt.index.strftime("%Y-%m-%d"))
    restored = sorted(rebuilt_dates - base_dates)
    removed = sorted(base_dates - rebuilt_dates)
    excluded_dates = sorted({case["date"] for case in cases if case["disposition"].startswith("exclude_")})
    expected_restored = sorted(
        {case["date"] for case in cases if case["disposition"] == "restore_from_jpx_am_open"}
        - set(excluded_dates)
    )
    if special_repairs.get("restored_rows"):
        expected_restored = sorted(
            set(expected_restored) | {"2025-10-24", "2025-10-27", "2025-10-28"}
        )
    if set(restored) != set(expected_restored):
        raise ValueError(f"Unexpected restored date set: {restored}")
    if set(removed) != set(provisional_dates):
        raise ValueError(f"Unexpected rows removed beyond provisional dates: {removed}")

    common_after_train_start = rebuilt.index.intersection(base.index)
    common_after_train_start = common_after_train_start[
        common_after_train_start >= pd.Timestamp("2015-01-05")
    ]
    numeric_cols = [column for column in rebuilt.select_dtypes(include=[np.number]).columns if column in base]
    actual = rebuilt.loc[common_after_train_start, numeric_cols].to_numpy(dtype=float)
    previous = base.loc[common_after_train_start, numeric_cols].to_numpy(dtype=float)
    different = ~np.isclose(actual, previous, rtol=0, atol=1e-12, equal_nan=True)
    differences = {
        (pd.Timestamp(date), numeric_cols[column_index])
        for row_index, date in enumerate(common_after_train_start)
        for column_index in np.flatnonzero(different[row_index])
    }
    allowed_beta_differences = {
        (pd.Timestamp(date), f"jp_beta_{ticker}")
        for date in common_after_train_start
        for ticker in JP_TICKERS
        if f"jp_beta_{ticker}" in numeric_cols
        and pd.isna(base.at[date, f"jp_beta_{ticker}"])
        and np.isfinite(rebuilt.at[date, f"jp_beta_{ticker}"])
    }
    unexpected_differences = differences - allowed_beta_differences
    if unexpected_differences or differences != allowed_beta_differences:
        raise ValueError(
            "Post-2015 corrections changed cells outside expected beta recovery: "
            f"unexpected={sorted(unexpected_differences)[:5]}, "
            f"expected_beta={len(allowed_beta_differences)}, actual={len(differences)}"
        )
    post_2015_differences = len(differences)

    audit = {
        "raw_sqlite_sha256": source_hash,
        "df_exec_fingerprint": dataframe_fingerprint(rebuilt),
        "df_exec_rows": len(rebuilt),
        "df_exec_first": str(rebuilt.index.min().date()),
        "df_exec_last": str(rebuilt.index.max().date()),
        "official_open_restored_cells": sum(c["disposition"] == "restore_from_jpx_am_open" for c in cases),
        "additional_official_open_restored_cells": len(special_repairs.get("restored_rows", [])),
        "additional_official_close_restored_cells": len(special_repairs.get("restored_rows", [])),
        "additional_official_source_pdf_sha256": special_repairs.get("source_pdf_sha256"),
        "explicitly_excluded_cells": sum(c["disposition"].startswith("exclude_") for c in cases),
        "explicitly_excluded_trade_dates": excluded_dates,
        "restored_dates_vs_uncorrected_nonstrict": restored,
        "excluded_provisional_dates": provisional_dates,
        "retained_row_audit_failures": len(row_errors),
        "post_2015_numeric_differences_vs_uncorrected_nonstrict": post_2015_differences,
        "post_2015_differences_are_only_recovered_beta_cells": True,
        "post_2015_restored_beta_cells": post_2015_differences,
        "baseline_2010_2014_beta_nan_cells": int(
            rebuilt.loc[
                (rebuilt.index >= "2010-01-01") & (rebuilt.index < "2015-01-01"),
                [f"jp_beta_{ticker}" for ticker in JP_TICKERS],
            ].isna().sum().sum()
        ),
    }
    return rebuilt, audit


def prepare() -> None:
    frame, audit = _rebuild_df_exec()
    config = load_config_from_yaml(ROOT / "configs/production/production.yaml", strict=True)
    train_start = pd.Timestamp("2015-01-05")
    provisional_mask = frame["is_provisional"].fillna(True).astype(bool)
    open_910 = build_open_910_returns(frame, JP_TICKERS)
    targets = compute_jp_target_returns(
        frame, JP_TICKERS, open_910_returns=open_910, allow_implicit_io=True
    )
    beta_masked_target_cells: list[dict[str, str]] = []
    for idx, date in enumerate(frame.index):
        if date < train_start or bool(provisional_mask.iloc[idx]):
            continue
        for ticker_index, ticker in enumerate(JP_TICKERS):
            if np.isfinite(targets[idx, ticker_index]) and not np.isfinite(
                frame.iloc[idx][f"jp_beta_{ticker}"]
            ):
                targets[idx, ticker_index] = np.nan
                beta_masked_target_cells.append(
                    {"date": str(date.date()), "ticker": ticker, "reason": "beta_missing"}
                )
    eligible = [
        date
        for idx, date in enumerate(frame.index)
        if date >= train_start
        and not bool(provisional_mask.iloc[idx])
        and np.isfinite(targets[idx]).any()
    ]
    if not eligible:
        raise ValueError("No usable post-2015 target cells")
    complete_label_dates = [
        date
        for idx, date in enumerate(frame.index)
        if date >= train_start
        and not bool(provisional_mask.iloc[idx])
        and np.isfinite(targets[idx]).all()
    ]
    if not complete_label_dates:
        raise ValueError("No complete post-2015 label dates")
    train_end = pd.Timestamp(complete_label_dates[-1]).normalize()
    if str(train_end.date()) != "2026-09-25":
        raise ValueError(f"Latest complete label date changed: {train_end.date()}")
    train_dates = pd.DatetimeIndex([date for date in eligible if date <= train_end])
    label_missing_cells = int(
        sum(
            not np.isfinite(targets[idx, ticker_index])
            for idx, date in enumerate(frame.index)
            if train_start <= date <= train_end
            for ticker_index in range(len(JP_TICKERS))
        )
    )
    end_exclusive = (train_end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    macro_requested_at = datetime.now(UTC).isoformat()
    macro_prices = macro_data.load_macro_prices(
        start=frame.index.min().strftime("%Y-%m-%d"),
        end=end_exclusive,
        period="max",
        timeout=25.0,
        cache={},
    )
    macro_received_at = datetime.now(UTC).isoformat()
    if macro_prices.empty or macro_prices.index.max() < pd.Timestamp("2026-09-24"):
        raise ValueError("Macro history does not cover the final known decision inputs")

    adr = load_adr_features()
    if adr is not None:
        adr = normalize_adr_features(adr)
        adr = adr.loc[adr.index <= train_end].copy()
    gap_dir = Path(config.gap_distribution_dir or config.v2.gap_input_dir)
    if not gap_dir.is_absolute():
        gap_dir = ROOT / gap_dir
    rank = None
    if config.v2.cs_overlay_enabled:
        rank = load_rank_reversal_frame(
            gap_dir, train_dates, file_pattern=config.v2.cs_rank_reversal_file_pattern
        )
    pit_values: dict[str, np.ndarray] = {}
    pit_dates: dict[str, np.ndarray] = {}
    pit_warnings: set[str] = set()
    from leadlag.models.v2.pit import load_pit_ir_history

    for date in train_dates:
        key = str(date.date())
        values, alerts, history_dates = load_pit_ir_history(gap_dir, key)
        pit_values[key] = values
        pit_dates[key] = history_dates
        pit_warnings.update(alerts)
    history_kwargs = {
        "open_910_returns": open_910,
        "macro_prices": macro_prices,
        "adr_features_frame": adr,
        "rank_reversal_signals": rank,
        "pit_ir_history": pit_values,
        "pit_history_trade_dates": pit_dates,
        "source": "ml_overlay_training_20260927",
        "observed_at_by_date": {
            str(date.date()): {
                "open_910_returns": f"{date.date()} 09:10",
                "macro_prices": f"{date.date()} 09:00",
                "adr_features": f"{date.date()} 09:00",
                "rank_reversal_signals": f"{date.date()} 09:00",
                "pit_ir_history": f"{date.date()} 09:10",
            }
            for date in train_dates
        },
    }
    INPUT.mkdir(parents=True, exist_ok=True)
    _dump(INPUT / "df_exec_corrected.pkl", frame)
    _dump(INPUT / "open_910_returns.pkl", open_910)
    _dump(INPUT / "targets.npy.pkl", targets)
    _dump(INPUT / "market_vol.pkl", _precompute_market_vol(frame))
    _dump(INPUT / "history_kwargs.pkl", history_kwargs)
    _dump(INPUT / "train_dates.pkl", train_dates)
    _dump(INPUT / "macro_prices.pkl", macro_prices)
    if adr is not None:
        _dump(INPUT / "adr.pkl", adr)
    if rank is not None:
        _dump(INPUT / "rank_reversal.pkl", rank)
    (INPUT / "run_config.json").write_text(
        json.dumps(config.v2.model_dump(mode="json"), indent=2, default=str) + "\n"
    )
    file_hashes = {
        str(path.relative_to(ROOT)): _sha256(path)
        for path in sorted(INPUT.iterdir())
        if path.is_file() and path.name != "input_provenance.json"
    }
    reconciliation_hash = _sha256(RECONCILIATION)
    provenance = {
        "raw_sqlite_sha256": audit["raw_sqlite_sha256"],
        "reconciliation_sha256": reconciliation_hash,
        "df_exec_fingerprint": audit["df_exec_fingerprint"],
        "input_file_sha256": file_hashes,
        "macro_provider": "Yahoo Finance via yfinance",
        "macro_requested_at": macro_requested_at,
        "macro_received_at": macro_received_at,
        "macro_last_price_date": str(macro_prices.index.max().date()),
        "historical_provider_available_at_proven": False,
        "availability_evidence": "session convention only; actual historical source retrieval timestamps unavailable",
        "official_open_restored_cells": audit["official_open_restored_cells"],
        "additional_official_open_restored_cells": audit["additional_official_open_restored_cells"],
        "additional_official_close_restored_cells": audit["additional_official_close_restored_cells"],
        "explicitly_excluded_cells": audit["explicitly_excluded_cells"],
        "explicitly_excluded_trade_dates": audit["explicitly_excluded_trade_dates"],
        "excluded_provisional_dates": audit["excluded_provisional_dates"],
        "post_2015_restored_beta_cells": audit["post_2015_restored_beta_cells"],
        "beta_masked_target_cells": beta_masked_target_cells,
        "beta_masked_target_cell_count": len(beta_masked_target_cells),
        "missing_target_cell_count_within_training_window": label_missing_cells,
        "training_start": str(train_start.date()),
        "training_end": str(train_end.date()),
        "training_date_count": len(train_dates),
        "gap_input_dir": str(gap_dir),
        "rank_reversal_rows": 0 if rank is None else len(rank),
        "adr_rows": 0 if adr is None else len(adr),
        "pit_warning_count": len(pit_warnings),
        "pit_warnings": sorted(pit_warnings),
    }
    _write_json(INPUT / "input_provenance.json", provenance)
    _write_json(REPORT / "input_provenance.json", {**provenance, **audit})
    (REPORT / "evaluation_plan.md").write_text(
        "# 2026-09-27 ML overlay retraining evaluation plan\n\n"
        "One fixed candidate; retain production LightGBM features, raw target, "
        "10 bps round-trip cost, seed 42 and existing parameters. Train from "
        "2015-01-05 through the latest complete label date. Use 2020-2024 expanding "
        "historical folds, removing the final five trading days before each fold; "
        "these periods are reused diagnostics, not new independent OOS. Require "
        "at least 95% date coverage per ticker, no more than 5 percentage points of "
        "cross-ticker coverage imbalance, and explicit accounting for every source/label "
        "exclusion. "
        "For historical fold P&L, keep flat/fallback days, production costs, weights, "
        "and all four cost components. Use 20-day paired block bootstrap, 1,000 "
        "samples, seed 42. Do not promote without at least 250 new forward labels, "
        "no net Sharpe or max drawdown deterioration, acceptable costs/turnover, "
        "and resolved data lineage. Historical provider available_at remains unproven.\n"
    )
    _write_json(REPORT / "rebuild_audit.json", audit)
    print(
        json.dumps(
            {
                "stage": "prepare",
                "df_exec_rows": len(frame),
                "train_dates": len(train_dates),
                "train_end": str(train_end.date()),
                "target_rows": len(train_dates) * len(JP_TICKERS),
                "beta_masked_target_cells": len(beta_masked_target_cells),
                "missing_target_cells_within_training_window": label_missing_cells,
                "macro_rows": len(macro_prices),
                "macro_last_date": str(macro_prices.index.max().date()),
                "rank_rows": 0 if rank is None else len(rank),
                "pit_warnings": len(pit_warnings),
                "rebuild_audit": audit,
            },
            ensure_ascii=False,
            default=str,
        ),
        flush=True,
    )


class _DecisionCapture:
    def __init__(self, model: Any, chunk_id: str, total: int, progress_dir: Path) -> None:
        self.model = model
        self.chunk_id = chunk_id
        self.total = total
        self.progress_dir = progress_dir
        self.decisions: dict[pd.Timestamp, Any] = {}
        self.started = time.monotonic()

    def decide(self, **kwargs: Any) -> Any:
        result = self.model.decide(**kwargs)
        date = pd.Timestamp(kwargs["inputs"].trade_date).normalize()
        self.decisions[date] = result
        if len(self.decisions) % 50 == 0 or len(self.decisions) == self.total:
            progress = {
                "chunk": self.chunk_id,
                "completed": len(self.decisions),
                "total": self.total,
                "date": str(date.date()),
                "elapsed_seconds": round(time.monotonic() - self.started, 1),
            }
            _write_json(self.progress_dir / f"chunk_{self.chunk_id}.json", progress)
            print(json.dumps(progress), flush=True)
        return result


def _collect_chunk(args: tuple[str, list[str], str, str]) -> dict[str, Any]:
    chunk_id, date_values, work_dir, gap_dir = args
    logging.getLogger().setLevel(logging.ERROR)
    work = Path(work_dir)
    inputs = work / "inputs"
    frame = pd.read_pickle(inputs / "df_exec_corrected.pkl")
    targets = _load(inputs / "targets.npy.pkl")
    market_vol = _load(inputs / "market_vol.pkl")
    history_kwargs = _load(inputs / "history_kwargs.pkl")
    history = HistoricalInputs(frame, **history_kwargs)
    run_cfg = ProductionV2RunConfig.model_validate_json((inputs / "run_config.json").read_text())
    dates = pd.DatetimeIndex(pd.to_datetime(date_values))
    model = _build_training_decision_model(run_cfg)
    capture = _DecisionCapture(model, chunk_id, len(dates), work / "progress")
    training = _collect_training_data(
        dates,
        frame,
        targets,
        Path(gap_dir),
        run_cfg,
        market_vol,
        per_ticker_interactions=True,
        adr_df=history_kwargs["adr_features_frame"],
        target_type="raw",
        open_910_returns=history_kwargs["open_910_returns"],
        historical_inputs=history,
        decision_model=capture,
    )
    _dump(work / "collected" / f"chunk_{chunk_id}.pkl", {"training": training, "decisions": capture.decisions})
    return {
        "chunk": chunk_id,
        "dates": len(dates),
        "decisions": len(capture.decisions),
        "training_rows": len(training),
        "training_dates": int(training["trade_date"].nunique()) if len(training) else 0,
        "elapsed_seconds": round(time.monotonic() - capture.started, 1),
    }


def _load_collection() -> tuple[pd.DataFrame, dict[pd.Timestamp, Any]]:
    parts = [_load(WORK / "collected" / f"chunk_{idx}.pkl") for idx in range(4)]
    training = pd.concat([part["training"] for part in parts], ignore_index=True)
    training = training.sort_values(["trade_date", "ticker"]).reset_index(drop=True)
    decisions = {date: value for part in parts for date, value in part["decisions"].items()}
    return training, decisions


def benchmark() -> None:
    dates = pd.read_pickle(INPUT / "train_dates.pkl")
    selected = pd.DatetimeIndex([dates[0], dates[100], dates[len(dates) // 2], dates[-2], dates[-1]])
    result = _collect_chunk(("benchmark", [str(d) for d in selected], str(WORK), str(GAP)))
    chunk = _load(WORK / "collected" / "chunk_benchmark.pkl")
    summary = []
    for date, decision in sorted(chunk["decisions"].items()):
        summary.append(
            {
                "date": str(date.date()),
                "gap_data_missing": bool(decision.fallback.get("gap_data_missing", False)),
                "audit_failure": bool(decision.fallback.get("audit_failure", False)),
                "numerical": decision.numerical.get("status"),
                "leakage": decision.leakage.get("status"),
            }
        )
    output = {"benchmark": result, "sample_decisions": summary}
    _write_json(REPORT / "on_demand_benchmark.json", output)
    print(json.dumps(output, ensure_ascii=False), flush=True)
    if result["training_rows"] < 4 * len(JP_TICKERS):
        raise ValueError("On-demand benchmark collected fewer than four complete dates")


def collect() -> None:
    dates = pd.read_pickle(INPUT / "train_dates.pkl")
    chunks = [(str(i), [str(d) for d in chunk], str(WORK), str(GAP))
              for i, chunk in enumerate(np.array_split(dates, 4))]
    (WORK / "collected").mkdir(parents=True, exist_ok=True)
    (WORK / "progress").mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn")) as executor:
        futures = [executor.submit(_collect_chunk, chunk) for chunk in chunks]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result), flush=True)
    results.sort(key=lambda value: value["chunk"])
    training, decisions = _load_collection()
    expected_dates = set(pd.read_pickle(INPUT / "train_dates.pkl"))
    collected_dates = set(pd.to_datetime(training["trade_date"]).dt.normalize()) if len(training) else set()
    missing_dates = sorted(expected_dates - collected_dates)
    ticker_counts = training.groupby("ticker").size().to_dict()
    collection = {
        "chunks": results,
        "requested_dates": len(expected_dates),
        "decision_dates": len(decisions),
        "training_dates": len(collected_dates),
        "training_rows": len(training),
        "excluded_dates_no_training_rows": [str(date.date()) for date in missing_dates],
        "rows_per_ticker": ticker_counts,
        "fallback_or_audit_failures": {
            "dates": [
                str(date.date())
                for date, result in sorted(decisions.items())
                if result.fallback.get("gap_data_missing", False)
                or result.fallback.get("audit_failure", False)
            ]
        },
    }
    if len(training):
        rates = {ticker: count / len(expected_dates) for ticker, count in ticker_counts.items()}
        collection["ticker_date_coverage"] = rates
        collection["minimum_ticker_date_coverage"] = min(rates.values()) if rates else 0.0
        collection["maximum_ticker_date_coverage"] = max(rates.values()) if rates else 0.0
        collection["ticker_coverage_imbalance"] = (
            max(rates.values()) - min(rates.values()) if rates else 1.0
        )
        if len(rates) != len(JP_TICKERS) or min(rates.values()) < 0.95:
            raise ValueError(f"Insufficient or skewed training sample by ticker: {rates}")
        if collection["ticker_coverage_imbalance"] > 0.05:
            raise ValueError(f"Ticker coverage differs by more than five percentage points: {rates}")
        if collection["training_dates"] / len(expected_dates) < 0.95:
            raise ValueError("Insufficient date coverage after on-demand/audit exclusions")
    collection["beta_masked_target_cell_count"] = len(
        json.loads((INPUT / "input_provenance.json").read_text()).get(
            "beta_masked_target_cells", []
        )
    )
    _dump(WORK / "collected" / "training_frame.pkl", training)
    _write_json(REPORT / "collection.json", collection)
    print(json.dumps(collection, ensure_ascii=False, default=str), flush=True)


def evaluate_walkforward() -> None:
    import research.scripts.experiments.structural_artifact_evaluate as evaluator

    evaluator.ROOT = ROOT
    evaluator.REPORT = REPORT
    evaluator.WORK = WORK / "walkforward"
    evaluator.GAP = GAP
    (evaluator.WORK / "artifacts").mkdir(parents=True, exist_ok=True)
    frame = pd.read_pickle(INPUT / "df_exec_corrected.pkl")
    history_kwargs = _load(INPUT / "history_kwargs.pkl")
    history = HistoricalInputs(frame, **history_kwargs)
    app = load_config_from_yaml(ROOT / "configs/production/production.yaml", strict=True)
    run_cfg = app.v2
    training, base = _load_collection()
    script_hash = _sha256(Path(__file__))
    output: dict[str, Any] = {
        "candidate_parameterizations": 1,
        "fold_count": 5,
        "years": list(range(2020, 2025)),
        "historically_used_periods": True,
        "independent_oos": False,
        "purge_trading_days": 5,
        "trial_count_before_this_run": _ml_training_trial_count(),
        "dsr": None,
        "dsr_reason": "No new parameters; historic trial Sharpe variance is not an independent estimate.",
        "folds": [],
    }
    for year in range(2020, 2025):
        year_dates = frame.index[(frame.index >= f"{year}-01-01") & (frame.index <= f"{year}-12-31")]
        model, directory, provenance = evaluator.train_candidate(training, frame, run_cfg, year, script_hash)
        candidate = evaluator.apply_candidate(base, year_dates, frame, history, model)
        base_result, base_summary, costs = evaluator.simulate(frame, history, app, base, year_dates)
        candidate_result, candidate_summary, _ = evaluator.simulate(frame, history, app, candidate, year_dates)
        fold = {
            "year": year,
            "train_cutoff": provenance["train_end"],
            "train_rows": provenance["training_rows"],
            "train_dates": provenance["training_dates"],
            "artifact": str(directory.relative_to(ROOT)),
            "artifact_version": model.metadata["artifact_version"],
            "baseline": base_summary,
            "candidate": candidate_summary,
            "paired_20_day_bootstrap": evaluator.block_interval(
                (candidate_result["daily_returns"] - base_result["daily_returns"]).to_numpy()
            ),
        }
        output["folds"].append(fold)
        output["costs"] = costs
        _dump(evaluator.WORK / "evaluation" / f"{year}.pkl", {"baseline": base_result, "candidate": candidate_result})
        _write_json(REPORT / "walkforward_partial.json", output)
        print(json.dumps({"fold": year, "cutoff": provenance["train_end"],
                          "baseline_net_sharpe": base_summary["net_sharpe"],
                          "candidate_net_sharpe": candidate_summary["net_sharpe"],
                          "candidate_max_drawdown": candidate_summary["max_drawdown"]}), flush=True)
    _write_json(REPORT / "walkforward.json", output)


def _ml_training_trial_count() -> int:
    registry_path = ROOT / "var/experiments/registry.jsonl"
    if not registry_path.is_file():
        return 0
    from leadlag.experiment_registry import ExperimentRegistry

    return sum(1 for _ in ExperimentRegistry(registry_path).iter_records(name="ml_overlay_training"))


def train_final() -> None:
    from leadlag.utils.timestamps import normalize_jst_date

    frame = pd.read_pickle(INPUT / "df_exec_corrected.pkl")
    training, _ = _load_collection()
    train_start = pd.Timestamp("2015-01-05")
    train_end = pd.Timestamp("2026-09-25")
    training = training.loc[
        (pd.to_datetime(training["trade_date"]) >= train_start)
        & (pd.to_datetime(training["trade_date"]) <= train_end)
    ].copy()
    expected_dates = pd.read_pickle(INPUT / "train_dates.pkl")
    counts = training.groupby("ticker").size()
    if len(counts) != len(JP_TICKERS) or (counts < 0.95 * len(expected_dates)).any():
        raise ValueError(f"Candidate sample coverage gate failed: {counts.to_dict()}")
    rates = counts / len(expected_dates)
    if float(rates.max() - rates.min()) > 0.05:
        raise ValueError(f"Candidate ticker coverage imbalance exceeds five points: {rates.to_dict()}")
    if training.duplicated(["trade_date", "ticker"]).any():
        raise ValueError("Duplicate trade_date/ticker training samples")
    if not np.isfinite(training.select_dtypes(include=[np.number]).to_numpy(dtype=float)).all():
        raise ValueError("Training feature/target frame contains non-finite values")
    model = _train_overlay_lgbm(
        training,
        lgbm_kwargs=DEFAULT_LGBM_KWARGS,
        use_ticker=True,
        use_classification=False,
        per_ticker_interactions=True,
        p_trade_scale=1.0,
    )
    run_cfg = ProductionV2RunConfig.model_validate_json((INPUT / "run_config.json").read_text())
    frame_hash = dataframe_fingerprint(frame.loc[frame.index <= train_end])
    config_hash = hashlib.sha256(
        json.dumps(run_cfg.model_dump(), sort_keys=True, default=str).encode()
    ).hexdigest()
    history_kwargs = _load(INPUT / "history_kwargs.pkl")
    open_910 = history_kwargs["open_910_returns"]
    reconciliation = json.loads(RECONCILIATION.read_text(encoding="utf-8"))
    special_repairs = reconciliation.get("additional_official_repairs", {})
    provenance = json.loads((REPORT / "input_provenance.json").read_text(encoding="utf-8"))
    metadata = {
        "metadata_status": "verified",
        "train_start": normalize_jst_date(train_start).strftime("%Y-%m-%d"),
        "train_end": normalize_jst_date(train_end).strftime("%Y-%m-%d"),
        "label_asof_end": normalize_jst_date(train_end).strftime("%Y-%m-%d"),
        "data_hash": frame_hash,
        "config_hash": config_hash,
        "source_code_hash": _sha256(Path(__file__)),
        "input_provenance_sha256": _sha256(REPORT / "input_provenance.json"),
        "evaluation_plan_sha256": _sha256(REPORT / "evaluation_plan.md"),
        "reconciliation_sha256": _sha256(RECONCILIATION),
        "df_exec_hash": _sha256(INPUT / "df_exec_corrected.pkl"),
        "open_910_hash": dataframe_fingerprint(open_910),
        "macro_prices_hash": dataframe_fingerprint(history_kwargs["macro_prices"]),
        "adr_features_hash": None if history_kwargs["adr_features_frame"] is None else dataframe_fingerprint(history_kwargs["adr_features_frame"]),
        "rank_reversal_hash": None if history_kwargs["rank_reversal_signals"] is None else dataframe_fingerprint(history_kwargs["rank_reversal_signals"]),
        "historical_provider_available_at_proven": False,
        "availability_evidence": "session convention only; actual historical provider retrieval times unavailable",
        "macro_requested_at": provenance["macro_requested_at"],
        "macro_received_at": provenance["macro_received_at"],
        "training_rows": len(training),
        "training_dates": int(training["trade_date"].nunique()),
        "candidate_parameterizations": 1,
        "walkforward_folds": 5,
        "walkforward_purge_trading_days": 5,
        "final_fit_purged_trading_days": 0,
        "excluded_source_cells": int(reconciliation["excluded_open_count"]),
        "excluded_source_trade_dates": reconciliation["excluded_trade_dates"],
        "additional_official_source_cells_restored": (
            int(special_repairs.get("restored_open_cells", 0))
            + int(special_repairs.get("restored_close_cells", 0))
        ),
        "additional_official_source_pdf_sha256": special_repairs.get("source_pdf_sha256"),
        "beta_masked_target_cell_count": provenance["beta_masked_target_cell_count"],
        "missing_target_cell_count_within_training_window": provenance[
            "missing_target_cell_count_within_training_window"
        ],
        "ticker_date_coverage": rates.to_dict(),
        "ticker_coverage_imbalance": float(rates.max() - rates.min()),
        "seed": 42,
        "target_type": "raw",
        "round_trip_cost_bps": 10.0,
        "lgbm_kwargs": DEFAULT_LGBM_KWARGS,
        "promotion_eligible": False,
        "promotion_blockers": [
            "250 new forward labeled trading days have not been collected",
            "historical provider available_at remains unverified",
        ],
    }
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    save_overlay_model(model, ARTIFACT, training_metadata=metadata)
    loaded = load_overlay_model(ARTIFACT)
    if loaded.metadata["train_end"] != "2026-09-25" or loaded.metadata["training_rows"] != len(training):
        raise ValueError("Published candidate metadata failed round-trip verification")
    result = {
        "status": "candidate_created_shadow_only",
        "artifact_root": str(ARTIFACT.relative_to(ROOT)),
        "artifact_version": loaded.metadata["artifact_version"],
        "model_sha256": loaded.metadata["model_sha256"],
        "metadata_sha256": _sha256(ARTIFACT / "versions" / loaded.metadata["artifact_version"] / "metadata.json"),
        "train_start": loaded.metadata["train_start"],
        "train_end": loaded.metadata["train_end"],
        "training_rows": len(training),
        "training_dates": int(training["trade_date"].nunique()),
        "rows_per_ticker": counts.to_dict(),
        "lgbm_kwargs": DEFAULT_LGBM_KWARGS,
        "historical_provider_available_at_proven": False,
        "production_current_changed": False,
    }
    _write_json(REPORT / "candidate.json", result)
    record_simple_experiment(
        name="ml_overlay_training",
        hypothesis="Retrain the fixed production overlay with all complete labels through 2026-09-25.",
        parameters={
            "train_start": "2015-01-05",
            "train_end": "2026-09-25",
            "target_type": "raw",
            "round_trip_cost_bps": 10.0,
            "lgbm_kwargs": DEFAULT_LGBM_KWARGS,
            "artifact_version": loaded.metadata["artifact_version"],
            "data_hash": frame_hash,
            "config_hash": config_hash,
        },
        metrics={
            "status": "candidate_created_shadow_only",
            "training_rows": len(training),
            "training_dates": int(training["trade_date"].nunique()),
            "candidate_parameterizations": 1,
            "forward_oos_days": 0,
            "forward_oos_required": 250,
            "historical_provider_available_at_proven": False,
        },
        decision=Decision.PENDING,
        report_path="reports/20260927_ml_overlay_retrain/report.md",
        registry_path=ROOT / "var/experiments/registry.jsonl",
    )
    print(json.dumps(result, ensure_ascii=False, default=str), flush=True)


def dry_run() -> None:
    """Load final candidate in production wiring and run a fold OOS date without API/order adapters."""
    if not (ARTIFACT / "CURRENT").is_file():
        raise FileNotFoundError(ARTIFACT / "CURRENT")
    final_model = load_overlay_model(ARTIFACT)
    app = load_config_from_yaml(ROOT / "configs/production/production.yaml", strict=True)
    final_app = app.model_copy(
        deep=True,
        update={"v2": app.v2.model_copy(deep=True, update={
            "ml_overlay_enabled": True,
            "ml_overlay_model_dir": str(ARTIFACT),
        })},
    )
    final_runner = ProductionRunner(final_app)
    if not final_runner._overlay_enabled:
        raise ValueError("Final candidate did not load as an enabled production overlay")

    fold_root = WORK / "walkforward/artifacts/evaluation_2020"
    fold_model = load_overlay_model(fold_root)
    frame = pd.read_pickle(INPUT / "df_exec_corrected.pkl")
    history_kwargs = _load(INPUT / "history_kwargs.pkl")
    history = HistoricalInputs(frame, **history_kwargs)
    dates = frame.index[(frame.index >= "2020-01-01") & (frame.index <= "2020-12-31")]
    lake = PITDataLake(frame)
    result = None
    used_date = None
    fold_app = app.model_copy(
        deep=True,
        update={"v2": app.v2.model_copy(deep=True, update={
            "ml_overlay_enabled": True,
            "ml_overlay_model_dir": str(fold_root),
        })},
    )
    runner = ProductionRunner(fold_app)
    for date in dates:
        snapshot = lake.get_execution_snapshot(date + pd.Timedelta(hours=9, minutes=10), history.open_910_returns)
        known = snapshot.to_known_inputs(
            sig_date=frame.loc[date].get("sig_date"),
            observed_at={
                "us_returns": f"{date.date()} 09:00",
                "jp_gap_returns": f"{date.date()} 09:10",
                "jp_betas": f"{date.date()} 09:10",
                "topix_night_return": f"{date.date()} 09:10",
                "current_prices": f"{date.date()} 09:10",
                "prev_closes": f"{date.date()} 09:10",
            },
            source="ml_overlay_retrain_api_disabled_dry_run",
        )
        inputs = DecisionInputs(
            known=known,
            historical=history,
            gap_input_dir=GAP,
            use_file_cache=True,
        )
        try:
            candidate_decision = runner.run(inputs)
        except Exception:
            continue
        if candidate_decision.summary.get("overlay_applied"):
            result, used_date = candidate_decision, date
            break
    if result is None:
        raise ValueError("No valid expanding-fold production dry-run date applied the overlay")
    dry_run_result = {
        "final_candidate_loaded_by_production_runner": bool(final_runner._overlay_enabled),
        "final_candidate_version": final_model.metadata["artifact_version"],
        "fold_candidate_version_used_for_valid_oos_dry_run": fold_model.metadata["artifact_version"],
        "fold_train_end": fold_model.metadata["train_end"],
        "dry_run_trade_date": str(used_date.date()),
        "overlay_applied": result.summary.get("overlay_applied"),
        "numerical_audit": result.numerical.get("status"),
        "leakage_audit": result.leakage.get("status"),
        "fallback": result.fallback,
        "api_enabled": False,
        "broker_adapter_created": False,
        "orders_sent": 0,
        "production_current_changed": False,
    }
    _write_json(REPORT / "dry_run.json", dry_run_result)
    print(json.dumps(dry_run_result, ensure_ascii=False, default=str), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "benchmark", "collect", "walkforward", "train", "dry-run"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    for logger_name in (
        "leadlag.core.market_calendar",
        "leadlag.data.preprocessor",
        "leadlag.data.validation",
    ):
        logging.getLogger(logger_name).setLevel(logging.ERROR)
    if args.stage == "prepare":
        prepare()
    elif args.stage == "benchmark":
        benchmark()
    elif args.stage == "collect":
        collect()
    elif args.stage == "walkforward":
        evaluate_walkforward()
    elif args.stage == "train":
        train_final()
    elif args.stage == "dry-run":
        dry_run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
