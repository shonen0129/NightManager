"""Fixed-input, bounded externally, structural artifact acceptance execution.

Commands prepare/benchmark/collect/evaluate preserve separate evidence. This
script never opens a broker or updates production config/artifact pointers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import multiprocessing
import pickle
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from leadlag.config.schemas import AppConfig, ProductionV2RunConfig
from leadlag.data import macro
from leadlag.data.intraday_inputs import compute_jp_target_returns
from leadlag.data.rank_reversal import load_rank_reversal_frame
from leadlag.data.tickers import JP_TICKERS
from leadlag.domain.inputs import HistoricalInputs
from leadlag.execution.config import load_config_from_yaml
from leadlag.models.ml_overlay_features import _precompute_market_vol
from leadlag.models.v2.pit import load_pit_ir_history
from leadlag.runner.model_factory import build_v2_model_bundle
from research.experiments.ml_overlay_training import _collect_training_data

ROOT = Path(__file__).resolve().parents[4]
REPORT = ROOT / "reports/20260922_production_acceptance"
WORK = ROOT / "var/results/20260922_production_acceptance"
INPUT = WORK / "inputs"
GAP = INPUT / "gap"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle)
    temporary.replace(path)


def load(path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def prepare():
    frame = pd.read_pickle(INPUT / "df_exec.pkl")
    app = load_config_from_yaml(ROOT / "configs/production/production.yaml", strict=True)
    dates = pd.DatetimeIndex(frame.index[frame.index >= "2015-01-05"])
    requested_at = datetime.now(UTC).isoformat()
    prices = macro.load_macro_prices(start=str(frame.index.min().date()),
                                    end=str((frame.index.max() + pd.Timedelta(days=1)).date()),
                                    period="max")
    received_at = datetime.now(UTC).isoformat()
    prices.to_pickle(INPUT / "macro.pkl")
    GAP.mkdir(parents=True, exist_ok=True)
    source = ROOT / str(app.gap_distribution_dir)
    for name in ("full_history_diagnostics.csv", "portfolio_gap_distribution_diagnostics.csv"):
        selected = next((p for p in (source / name, source.parent / name) if p.is_file()), None)
        if selected:
            shutil.copy2(selected, GAP / name)
    pit, pit_dates = {}, {}
    warnings = set()
    for date in dates:
        values, alerts, history_dates = load_pit_ir_history(GAP, str(date.date()))
        pit[str(date.date())], pit_dates[str(date.date())] = values, history_dates
        warnings.update(alerts)
    rank = load_rank_reversal_frame(source, dates, file_pattern=app.v2.cs_rank_reversal_file_pattern)
    history_kwargs = {
        "frame": frame,
        "open_910_returns": pd.read_pickle(INPUT / "open_910_returns.pkl"),
        "macro_prices": prices,
        "adr_features_frame": pd.read_pickle(INPUT / "adr.pkl"),
        "rank_reversal_signals": rank,
        "pit_ir_history": pit, "pit_history_trade_dates": pit_dates,
        "source": "fixed_structural_acceptance",
    }
    dump(INPUT / "history_kwargs.pkl", history_kwargs)
    write_json(INPUT / "config.json", app.v2.model_dump(mode="json"))
    write_json(REPORT / "input_provenance.json", {
        "macro_provider": "Yahoo Finance via yfinance", "macro_requested_at": requested_at,
        "macro_received_at": received_at, "historical_provider_available_at": None,
        "historical_available_at_proven": False,
        "session_cutoff": "09:10 JST; historical provider timestamp evidence unavailable",
        "snapshot_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in INPUT.rglob("*") if p.is_file()},
        "pit_warnings": sorted(warnings), "rank_rows": 0 if rank is None else len(rank),
        "evaluation_end": str(dates[-1].date()), "trade_dates": len(dates),
    })
    print(json.dumps({"prepared_dates": len(dates), "macro_rows": len(prices),
                      "rank_rows": 0 if rank is None else len(rank), "pit_warnings": sorted(warnings)}), flush=True)


def resources():
    kwargs = load(INPUT / "history_kwargs.pkl")
    frame = kwargs.pop("frame")
    history = HistoricalInputs(frame, **kwargs)
    config = ProductionV2RunConfig.model_validate_json((INPUT / "config.json").read_text())
    return frame, history, config


def collect_chunk(args):
    chunk_id, dates = args
    logging.getLogger().setLevel(logging.ERROR)
    start = time.monotonic()
    frame, history, config = resources()
    app = AppConfig(v2=config.model_copy(update={"ml_overlay_enabled": False}))
    model = build_v2_model_bundle(app).decision_model
    decisions = {}

    class Capture:
        def decide(self, **kwargs):
            result = model.decide(**kwargs)
            date = kwargs["inputs"].trade_date
            decisions[date] = result
            if len(decisions) % 25 == 0:
                write_json(WORK / "progress" / f"chunk_{chunk_id}.json", {
                    "completed": len(decisions), "total": len(dates),
                    "elapsed_seconds": time.monotonic() - start, "date": str(date.date()),
                })
            return result

    labels = compute_jp_target_returns(frame, JP_TICKERS,
                                      open_910_returns=history.open_910_returns)
    training = _collect_training_data(
        pd.DatetimeIndex(dates), frame, labels, GAP, app.v2, _precompute_market_vol(frame),
        per_ticker_interactions=True, adr_df=history.adr_features,
        open_910_returns=history.open_910_returns, historical_inputs=history,
        decision_model=Capture(),
    )
    output = WORK / "collected" / f"chunk_{chunk_id}.pkl"
    dump(output, {"training": training, "decisions": decisions})
    return {"chunk": chunk_id, "dates": len(dates), "decisions": len(decisions),
            "training_rows": len(training), "elapsed_seconds": time.monotonic() - start}


def collect(benchmark=False):
    frame = pd.read_pickle(INPUT / "df_exec.pkl")
    dates = frame.index[frame.index >= "2015-01-05"]
    if benchmark:
        dates = pd.DatetimeIndex([dates[0], dates[100], dates[len(dates)//2], dates[-2], dates[-1]])
        results = [collect_chunk(("benchmark", dates))]
    else:
        chunks = [(str(i), chunk) for i, chunk in enumerate(np.array_split(dates, 4))]
        results = []
        with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn")) as executor:
            pending = [executor.submit(collect_chunk, chunk) for chunk in chunks]
            for future in as_completed(pending):
                result = future.result()
                results.append(result)
                print(json.dumps(result), flush=True)
    write_json(REPORT / ("collection_benchmark.json" if benchmark else "collection.json"), results)
    print(json.dumps(results), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "benchmark", "collect"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    if args.stage == "prepare":
        prepare()
    else:
        collect(args.stage == "benchmark")


if __name__ == "__main__":
    main()
