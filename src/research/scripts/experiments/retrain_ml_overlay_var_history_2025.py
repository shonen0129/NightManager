"""Train a fixed-parameter overlay through the last 2025 JP trading session.

Inputs are the independently reconciled training artifacts from the 2026-09-27
rebuild. The script creates an immutable research artifact and never changes the
production pointer. Historical return replay/promotion is handled separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.config.schemas import ProductionV2RunConfig
from leadlag.data.tickers import JP_TICKERS
from leadlag.experiment_registry import Decision
from research.experiments.ml_overlay_training import DEFAULT_LGBM_KWARGS
from leadlag.models.ml_overlay_artifact import load_overlay_model, save_overlay_model
from leadlag.utils.dataframe_fingerprint import dataframe_fingerprint
from research.experiment_utils import record_simple_experiment
from research.experiments.ml_overlay_training import _train_overlay_lgbm

SOURCE_WORK = ROOT / "var/results/20260927_ml_overlay_retrain"
SOURCE_REPORT = ROOT / "reports/20260927_ml_overlay_retrain"
WORK = ROOT / "var/results/20260927_ml_overlay_var_history_2025"
REPORT = ROOT / "reports/20260927_ml_overlay_var_history"
ARTIFACT = WORK / "artifact"
TRAIN_START = pd.Timestamp("2015-01-05")
REQUESTED_CUTOFF = pd.Timestamp("2025-12-31")


def _read_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")


def train() -> None:
    frame = _read_pickle(SOURCE_WORK / "inputs/df_exec_corrected.pkl")
    collected = _read_pickle(SOURCE_WORK / "collected/chunk_0.pkl")
    chunks = [collected]
    for index in range(1, 4):
        chunks.append(_read_pickle(SOURCE_WORK / f"collected/chunk_{index}.pkl"))
    training = pd.concat([item["training"] for item in chunks], ignore_index=True)
    training = training.sort_values(["trade_date", "ticker"]).reset_index(drop=True)
    training_dates = pd.to_datetime(training["trade_date"]).dt.normalize()
    training = training.loc[training_dates <= REQUESTED_CUTOFF].copy()
    training_dates = pd.to_datetime(training["trade_date"]).dt.normalize()

    eligible_dates = pd.DatetimeIndex(_read_pickle(SOURCE_WORK / "inputs/train_dates.pkl"))
    expected_dates = eligible_dates[
        (eligible_dates >= TRAIN_START) & (eligible_dates <= REQUESTED_CUTOFF)
    ]
    if expected_dates.empty or expected_dates.max() != pd.Timestamp("2025-12-30"):
        raise ValueError(
            "Expected the last complete JP label date through 2025-12-31 to be 2025-12-30; "
            f"got {expected_dates.max() if not expected_dates.empty else None}"
        )
    train_end = pd.Timestamp(expected_dates.max()).normalize()
    if training.empty or training_dates.max() != train_end:
        raise ValueError(f"Collected label cutoff mismatch: expected {train_end.date()}")
    if training.duplicated(["trade_date", "ticker"]).any():
        raise ValueError("Duplicate trade_date/ticker rows in cutoff training sample")
    if set(training["ticker"].unique()) != set(JP_TICKERS):
        raise ValueError("Cutoff sample does not contain the full production JP ticker universe")
    numeric = training.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Cutoff training feature/target frame contains non-finite values")

    counts = training.groupby("ticker").size()
    if len(counts) != len(JP_TICKERS) or (counts < 0.95 * len(expected_dates)).any():
        raise ValueError(f"Cutoff candidate sample coverage gate failed: {counts.to_dict()}")
    rates = counts / len(expected_dates)
    if float(rates.max() - rates.min()) > 0.05:
        raise ValueError(f"Cutoff candidate coverage imbalance exceeds five points: {rates.to_dict()}")

    run_cfg = ProductionV2RunConfig.model_validate_json(
        (SOURCE_WORK / "inputs/run_config.json").read_text(encoding="utf-8")
    )
    frame_through_cutoff = frame.loc[frame.index <= train_end].copy()
    history = _read_pickle(SOURCE_WORK / "inputs/history_kwargs.pkl")
    source_provenance_path = SOURCE_WORK / "inputs/input_provenance.json"
    source_provenance = json.loads(source_provenance_path.read_text(encoding="utf-8"))
    config_hash = hashlib.sha256(
        json.dumps(run_cfg.model_dump(), sort_keys=True, default=str).encode()
    ).hexdigest()
    model = _train_overlay_lgbm(
        training,
        lgbm_kwargs=DEFAULT_LGBM_KWARGS,
        use_ticker=True,
        use_classification=False,
        per_ticker_interactions=True,
        p_trade_scale=1.0,
    )

    data_hash = dataframe_fingerprint(frame_through_cutoff)
    training_hash = dataframe_fingerprint(training)
    metadata = {
        "metadata_status": "verified",
        "train_start": str(TRAIN_START.date()),
        "train_end": str(train_end.date()),
        "label_asof_end": str(train_end.date()),
        "data_hash": data_hash,
        "config_hash": config_hash,
        "training_frame_hash": training_hash,
        "source_artifact_version": "20260926T182011981905Z-f8b4ec178e88",
        "source_input_provenance_sha256": _sha256(source_provenance_path),
        "source_reconciliation_sha256": _sha256(SOURCE_REPORT / "reconciliation.json"),
        "cutoff_input_df_exec_hash": dataframe_fingerprint(frame_through_cutoff),
        "open_910_hash": dataframe_fingerprint(history["open_910_returns"].loc[:train_end]),
        "macro_prices_hash": dataframe_fingerprint(history["macro_prices"].loc[:train_end]),
        "adr_features_hash": (
            None if history["adr_features_frame"] is None else
            dataframe_fingerprint(history["adr_features_frame"].loc[:train_end])
        ),
        "rank_reversal_hash": (
            None if history["rank_reversal_signals"] is None else
            dataframe_fingerprint(history["rank_reversal_signals"].loc[:train_end])
        ),
        "historical_provider_available_at_proven": False,
        "availability_evidence": "session convention only; historical provider timestamps are unavailable",
        "requested_training_cutoff": str(REQUESTED_CUTOFF.date()),
        "training_rows": int(len(training)),
        "training_dates": int(training_dates.nunique()),
        "rows_per_ticker": counts.to_dict(),
        "ticker_date_coverage": rates.to_dict(),
        "ticker_coverage_imbalance": float(rates.max() - rates.min()),
        "candidate_parameterizations": 1,
        "seed": 42,
        "target_type": "raw",
        "round_trip_cost_bps": 10.0,
        "lgbm_kwargs": DEFAULT_LGBM_KWARGS,
        "training_frame_first": str(frame_through_cutoff.index.min().date()),
        "training_frame_last": str(frame_through_cutoff.index.max().date()),
        "promotion_eligible": False,
        "promotion_blockers": [
            "historical provider available_at remains unverified",
            "2026 post-cutoff history is retrospective and is not a new prospective 250-day gate",
        ],
    }
    save_overlay_model(model, ARTIFACT, training_metadata=metadata)
    loaded = load_overlay_model(ARTIFACT)
    if loaded.metadata["train_end"] != "2025-12-30":
        raise ValueError("Published artifact does not have the expected label cutoff")
    if loaded.metadata["training_frame_hash"] != training_hash:
        raise ValueError("Published artifact cutoff training hash failed round-trip check")

    result = {
        "status": "candidate_created",
        "artifact_root": str(ARTIFACT.relative_to(ROOT)),
        "artifact_version": loaded.metadata["artifact_version"],
        "model_sha256": loaded.metadata["model_sha256"],
        "train_start": loaded.metadata["train_start"],
        "requested_cutoff": str(REQUESTED_CUTOFF.date()),
        "actual_last_label_date": loaded.metadata["train_end"],
        "training_dates": loaded.metadata["training_dates"],
        "training_rows": loaded.metadata["training_rows"],
        "rows_per_ticker": counts.to_dict(),
        "lgbm_kwargs": DEFAULT_LGBM_KWARGS,
        "historical_provider_available_at_proven": False,
        "promotion_eligible": False,
        "production_pointer_changed": False,
    }
    _write_json(REPORT / "2025_cutoff_candidate.json", result)
    _write_json(WORK / "training_provenance.json", metadata)
    record_simple_experiment(
        name="ml_overlay_var_history_cutoff_2025",
        hypothesis="A fixed-parameter overlay trained through the last 2025 JP session can provide valid post-cutoff historical VaR returns, avoiding in-sample overlay application.",
        parameters={
            "train_start": str(TRAIN_START.date()),
            "requested_cutoff": str(REQUESTED_CUTOFF.date()),
            "actual_train_end": str(train_end.date()),
            "lgbm_kwargs": DEFAULT_LGBM_KWARGS,
            "artifact_version": loaded.metadata["artifact_version"],
            "data_hash": data_hash,
            "training_frame_hash": training_hash,
        },
        metrics={
            "training_dates": loaded.metadata["training_dates"],
            "training_rows": loaded.metadata["training_rows"],
            "prospective_oos_days": 0,
            "provider_available_at_proven": False,
        },
        decision=Decision.PENDING,
        report_path="reports/20260927_ml_overlay_var_history/report.md",
        registry_path=ROOT / "var/experiments/registry.jsonl",
    )
    print(json.dumps(result, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["train"])
    args = parser.parse_args()
    train()
