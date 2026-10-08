"""Create an audited VaR history with train-end-safe ML overlay versions."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import pickle
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.core.risk import compute_var_es
from leadlag.data.market_data_cache import load_df_exec_from_local_cache
from leadlag.execution import var_inputs
from leadlag.execution.backtester import BacktestEngine
from leadlag.execution.config import load_config_from_yaml
from leadlag.execution.var_history import _build_var_historical_inputs
from leadlag.models.ml_overlay_artifact import load_overlay_model
from leadlag.utils.dataframe_fingerprint import dataframe_fingerprint

SOURCE_WORK = ROOT / "var/results/20260927_ml_overlay_retrain"
WORK = ROOT / "var/results/20260927_ml_overlay_var_history_2025"
PRODUCTION_ARTIFACT_ROOT = ROOT / "models/ml_order_overlay/production_20260923"
CANDIDATE_ROOT = ROOT / "var/results/20260927_ml_overlay_var_history_2025/artifact"
OLD_VERSION = "20260922T011723828589Z-7e1a81ab2617"
REPORT_DIR = ROOT / "reports/20260927_ml_overlay_var_history"
PERSISTED_MACRO_PATH = ROOT / "var/market_data/macro_prices_verified.pkl"


def _json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _persist_verified_macro_snapshot() -> dict[str, Any]:
    source = SOURCE_WORK / "inputs/macro_prices.pkl"
    if not source.is_file():
        raise FileNotFoundError(source)
    frame = _load_pickle(source)
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Audited macro snapshot is empty or invalid")
    PERSISTED_MACRO_PATH.parent.mkdir(parents=True, exist_ok=True)
    if PERSISTED_MACRO_PATH.exists():
        current = _load_pickle(PERSISTED_MACRO_PATH)
        if not current.equals(frame):
            raise ValueError(
                f"Refusing to replace existing verified macro snapshot: {PERSISTED_MACRO_PATH}"
            )
    else:
        temporary = PERSISTED_MACRO_PATH.with_suffix(".pkl.tmp")
        with temporary.open("wb") as handle:
            pickle.dump(frame, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, PERSISTED_MACRO_PATH)
    source_provenance = json.loads(
        (SOURCE_WORK / "inputs/input_provenance.json").read_text(encoding="utf-8")
    )
    snapshot_provenance = {
        "source": str(source.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "macro_dataframe_fingerprint": dataframe_fingerprint(frame),
        "macro_provider": source_provenance.get("macro_provider"),
        "macro_requested_at": source_provenance.get("macro_requested_at"),
        "macro_received_at": source_provenance.get("macro_received_at"),
        "macro_last_price_date": str(frame.index.max().date()),
        "historical_provider_available_at_proven": source_provenance.get(
            "historical_provider_available_at_proven", False
        ),
    }
    provenance_path = PERSISTED_MACRO_PATH.with_suffix(".provenance.json")
    provenance_path.write_text(
        json.dumps(snapshot_provenance, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    return snapshot_provenance


def _load_version(root: Path, version: str) -> Any:
    """Use the production loader to validate one immutable inactive version."""
    with tempfile.TemporaryDirectory(prefix="leadlag-overlay-version-") as temporary:
        temp_root = Path(temporary)
        (temp_root / "versions").symlink_to(root / "versions", target_is_directory=True)
        (temp_root / "CURRENT").write_text(version + "\n", encoding="utf-8")
        return load_overlay_model(temp_root)


def create() -> None:
    logging.basicConfig(level=logging.WARNING)
    for logger_name in ("leadlag.core.pipeline", "leadlag.data.preprocessor"):
        logging.getLogger(logger_name).setLevel(logging.ERROR)
    trade_date = pd.Timestamp("2026-09-28")
    required_last = pd.Timestamp("2026-09-25")
    frame = load_df_exec_from_local_cache(max_stale_bdays=None)
    frame = frame.loc[frame.index <= required_last].copy()
    macro_provenance = _persist_verified_macro_snapshot()
    available = pd.DatetimeIndex(frame.index[frame.index < trade_date])
    window = 250
    desired_history_days = window + 19
    if len(available) < window or available.max() != required_last:
        raise ValueError(
            f"Expected >= {window} completed rows ending 2026-09-25; "
            f"got {len(available)}, last={available.max() if len(available) else None}"
        )
    history_days = min(desired_history_days, len(available))
    sim_dates = available[-history_days:]
    if sim_dates[0] < pd.Timestamp("2025-01-05"):
        raise ValueError("Selected VaR history reaches before the prior valid artifact cutoff")
    new_cutoff = pd.Timestamp("2025-12-30")
    old_dates = sim_dates[sim_dates <= new_cutoff]
    new_dates = sim_dates[sim_dates > new_cutoff]
    if len(old_dates) == 0 or len(old_dates) + len(new_dates) < window:
        raise ValueError(
            f"Versioned history insufficient: old={len(old_dates)}, new={len(new_dates)}"
        )

    app = load_config_from_yaml(ROOT / "configs/production/production.yaml", strict=True)
    gap_source = app.gap_distribution_dir
    if gap_source and not Path(gap_source).is_absolute():
        gap_source = ROOT / gap_source
    gap_snapshot, gap_hash, gap_owner = var_inputs._snapshot_gap_input(
        Path(gap_source) if gap_source else None,
        max_trade_date=required_last,
    )
    if gap_owner is None or gap_snapshot is None:
        raise RuntimeError("Could not obtain a stable immutable gap-input snapshot")
    try:
        # Use the same adapter boundary and run-owned snapshot as live VaR.
        # The adapter falls back only to the persisted, provenance-recorded
        # macro snapshot created just above when network retrieval is offline.
        history = _build_var_historical_inputs(
            frame,
            app,
            gap_snapshot,
            sim_dates,
            overlay_enabled=True,
        )
        old_model = _load_version(PRODUCTION_ARTIFACT_ROOT, OLD_VERSION)
        new_model = load_overlay_model(CANDIDATE_ROOT)
        if old_model.metadata["train_end"] != "2024-12-20":
            raise ValueError(f"Unexpected prior artifact train_end: {old_model.metadata['train_end']}")
        if new_model.metadata["train_end"] != "2025-12-30":
            raise ValueError(f"Unexpected candidate train_end: {new_model.metadata['train_end']}")

        audit_rows: list[dict[str, Any]] = []
        chunks: list[dict[str, Any]] = []
        for label, dates, model in (
            ("prior_artifact", old_dates, old_model),
            ("cutoff_2025_artifact", new_dates, new_model),
        ):
            started = time.monotonic()
            chunk = BacktestEngine.run_v2_backtest(
                cfg=app,
                gap_input_dir=gap_snapshot,
                df_exec=frame,
                start_date=str(dates.min().date()),
                end_date=str(dates.max().date()),
                n_jobs=4,
                overlay_model=model,
                historical_inputs=history,
            )
            if not chunk["daily_returns"].index.equals(dates):
                raise ValueError(f"{label} returned a different date set than requested")
            if len(chunk["v2_summaries"]) != len(dates):
                raise ValueError(f"{label} did not retain one summary per decision date")
            for date, summary, is_fallback, weights_row in zip(
                dates,
                chunk["v2_summaries"],
                chunk["daily_fallback"].to_numpy(dtype=bool),
                chunk["weights"].to_numpy(dtype=float),
            ):
                audit = summary.get("audit_status", {})
                audit_rows.append(
                    {
                        "trade_date": str(date.date()),
                        "numerical": audit.get("numerical"),
                        "leakage": audit.get("leakage"),
                        "fallback": bool(is_fallback or audit.get("fallback", False)),
                        "overlay_applied": bool(summary.get("overlay_applied")),
                        "weight_gross": float(np.sum(np.abs(weights_row))),
                        "weight_net": float(np.sum(weights_row)),
                    }
                )
            if chunk["daily_returns"].isna().any() or not np.isfinite(
                chunk["daily_returns"].to_numpy(dtype=float)
            ).all():
                raise ValueError(f"{label} returned non-finite daily returns")
            chunks.append(chunk)
            logging.warning(
                "%s completed %d decisions in %.1fs; fallback=%d",
                label,
                len(dates),
                time.monotonic() - started,
                int(chunk["daily_fallback"].sum()),
            )

        returns = pd.concat([chunk["daily_returns"] for chunk in chunks]).sort_index()
        fallbacks = pd.concat([chunk["daily_fallback"] for chunk in chunks]).sort_index()
        weights = pd.concat([chunk["weights"] for chunk in chunks]).sort_index()
        costs = {
            key: float(sum(chunk[key].sum() for chunk in chunks))
            for key in (
                "daily_slip_costs",
                "daily_financing_costs",
                "daily_borrow_costs",
                "daily_reverse_costs",
                "daily_costs",
            )
        }
        if not returns.index.equals(sim_dates) or len(returns) != history_days:
            raise ValueError("The joined history does not cover the expected trading rows")
        if int(fallbacks.sum()) != 0:
            raise ValueError(f"Historical replay contained {int(fallbacks.sum())} fallback days")
        if len(audit_rows) != len(returns):
            raise ValueError(
                f"Expected an audited decision on every date, got {len(audit_rows)}/{len(returns)}"
            )
        if any(row["numerical"] != "PASSED" or row["leakage"] != "PASSED" for row in audit_rows):
            bad = [row for row in audit_rows if row["numerical"] != "PASSED" or row["leakage"] != "PASSED"]
            raise ValueError(f"Audit failure in versioned history: {bad[:5]}")
        max_gross = float(weights.abs().sum(axis=1).max())
        max_abs_net = float(weights.sum(axis=1).abs().max())
        if max_gross > 2.0 + 1e-6 or max_abs_net > 0.05 + 1e-6:
            raise ValueError(f"Model exposure contract failed: gross={max_gross}, net={max_abs_net}")
        risk = app.risk
        var_es = compute_var_es(
            returns,
            confidence=risk.var_confidence,
            window=risk.var_window,
            var_method=risk.var_method,
        )
        if not var_es.available or var_es.samples != risk.var_window:
            raise ValueError(f"History is insufficient for configured VaR window: {var_es}")

        output = {
            "daily_returns": returns,
            "daily_fallback": fallbacks,
            "weights": weights,
            "audit_rows": audit_rows,
            "window_start": str(returns.index.min().date()),
            "window_end": str(returns.index.max().date()),
            "date_count": len(returns),
            "old_artifact_dates": len(old_dates),
            "old_artifact_version": OLD_VERSION,
            "new_artifact_dates": len(new_dates),
            "new_artifact_version": new_model.metadata["artifact_version"],
            "gross_cost_sums": costs,
            "gap_input_fingerprint": gap_hash,
            "macro_snapshot_provenance": macro_provenance,
        }
        result_path = WORK / "history" / "versioned_var_returns.pkl"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with result_path.open("wb") as handle:
            import pickle

            pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
        report = {
            "status": "created_and_audited",
            "trade_date": str(trade_date.date()),
            "required_last_completed_date": str(required_last.date()),
            "var_window": risk.var_window,
            "history_days": len(returns),
            "history_start": str(returns.index.min().date()),
            "history_end": str(returns.index.max().date()),
            "df_exec_rows_in_run_owned_snapshot": len(frame),
            "df_exec_hash_in_run_owned_snapshot": dataframe_fingerprint(frame),
            "cutoff_boundary": "2025-12-30 (last JP trading label date through requested 2025-12-31)",
            "prior_artifact": {
                "version": OLD_VERSION,
                "train_end": old_model.metadata["train_end"],
                "applied_days": len(old_dates),
            },
            "cutoff_2025_artifact": {
                "version": new_model.metadata["artifact_version"],
                "train_end": new_model.metadata["train_end"],
                "applied_days": len(new_dates),
            },
            "overlay_applied_days": sum(row["overlay_applied"] for row in audit_rows),
            "overlay_skipped_days": sum(not row["overlay_applied"] for row in audit_rows),
            "overlay_skipped_dates": [
                row["trade_date"] for row in audit_rows if not row["overlay_applied"]
            ],
            "fallback_days": int(fallbacks.sum()),
            "numerical_audit": {
                "passed": sum(row["numerical"] == "PASSED" for row in audit_rows),
                "failed": sum(row["numerical"] == "FAILED" for row in audit_rows),
            },
            "leakage_audit": {
                "passed": sum(row["leakage"] == "PASSED" for row in audit_rows),
                "flat": sum(row["leakage"] == "FLAT" for row in audit_rows),
                "failed": sum(row["leakage"] == "FAILED" for row in audit_rows),
            },
            "model_weight_constraints": {
                "max_gross": max_gross,
                "max_abs_net": max_abs_net,
            },
            "effective_side_leverage": chunks[-1]["side_leverage"],
            "var_es": var_es.__dict__,
            "daily_return_mean": float(returns.mean()),
            "daily_return_std": float(returns.std(ddof=1)),
            "daily_return_min": float(returns.min()),
            "daily_return_max": float(returns.max()),
            "gross_cost_sums": costs,
            "gap_input_fingerprint": gap_hash,
            "artifact_train_end_beyond_live_last_data": False,
            "historical_provider_available_at_proven": False,
            "result_path": str(result_path.relative_to(ROOT)),
            "production_pointer_changed": False,
        }
        _json_write(REPORT_DIR / "history.json", report)
        _json_write(REPORT_DIR / "history_audits.json", {"decisions": audit_rows})
        print(json.dumps(report, ensure_ascii=False, default=str), flush=True)
    finally:
        gap_owner.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["create"])
    parser.parse_args()
    create()
