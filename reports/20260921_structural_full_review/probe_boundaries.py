"""Offline review reproductions; no implementation or production-data writes."""
from __future__ import annotations

import json
import logging
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from leadlag.data.pit_lake import PITDataLake
from leadlag.data.intraday_inputs import compute_jp_target_returns
from leadlag.data.tickers import JP_TICKERS
from leadlag.domain.inputs import HistoricalInputs
from leadlag.execution import backtester, var_history
from leadlag.execution.config import load_config_from_yaml
from leadlag.runner.model_factory import build_v2_model_bundle
from research.experiments.ml_overlay_training import _collect_training_data

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def prices_probe() -> dict:
    frame = pd.read_csv(
        ROOT / "tests/regression/baselines/df_exec_20260814.csv.gz",
        index_col=0, parse_dates=True,
    )
    date = pd.Timestamp("2026-08-14")
    frame = frame.loc[:date]
    app = load_config_from_yaml(ROOT / "configs/production/production.yaml", strict=True)
    try:
        build_v2_model_bundle(app)
    except (ValueError, FileNotFoundError) as exc:
        production_bootstrap = {"status": "blocked", "error": str(exc)}
    else:
        production_bootstrap = {"status": "success"}
    app = app.model_copy(update={"v2": app.v2.model_copy(update={"ml_overlay_enabled": False})})
    results = {}
    for label, last_returns in (
        ("zero_control", np.zeros(len(JP_TICKERS))),
        ("nonzero_0910", np.linspace(-0.01, 0.01, len(JP_TICKERS))),
    ):
        intraday = pd.DataFrame(0.0, index=frame.index, columns=JP_TICKERS)
        intraday.loc[date] = last_returns
        past_dates = frame.index[frame.index < date][-300:]
        history = HistoricalInputs(
            frame, source="review_fixed_inputs",
            open_910_returns=intraday,
            pit_ir_history=np.linspace(-1.0, 1.0, len(past_dates)),
            pit_history_trade_dates=past_dates.to_numpy(),
        )
        real_bundle = build_v2_model_bundle(app)
        captured = {}

        class CaptureModel:
            def decide(self, **kwargs):
                captured["inputs"] = kwargs["inputs"]
                captured["decision"] = real_bundle.decision_model.decide(**kwargs)
                return captured["decision"]

        captured_bundle = SimpleNamespace(
            decision_model=CaptureModel(), run_config=real_bundle.run_config,
            overlay_enabled=real_bundle.overlay_enabled,
        )
        with patch.object(backtester, "build_v2_model_bundle", return_value=captured_bundle):
            weights, flags, summaries = backtester.BacktestEngine._generate_v2_weights(
                frame, app, None, pd.DatetimeIndex([date]), len(JP_TICKERS),
                None, None, 1, historical_inputs=history,
            )
        captured_inputs = captured["inputs"]
        raw_prices = dict(captured_inputs.known.current_prices)
        expected_prices = {
            ticker: float(frame.loc[date, f"jp_open_trade_{ticker}"]) * (1.0 + last_returns[j])
            for j, ticker in enumerate(JP_TICKERS)
        }
        corrected_inputs = PITDataLake(frame).build_decision_inputs(
            date + pd.Timedelta(hours=9, minutes=10),
            current_prices=expected_prices,
            historical=history, source="review_price_adapter", use_file_cache=True,
        )
        # Keep provenance/cutoff identical so only prices and gaps differ.
        corrected_inputs = replace(
            captured_inputs,
            known=replace(
                captured_inputs.known,
                current_prices=corrected_inputs.known.current_prices,
                jp_gap_returns=corrected_inputs.known.jp_gap_returns,
            ),
        )
        corrected = build_v2_model_bundle(app).decision_model.decide(
            inputs=corrected_inputs, overlay_enabled=real_bundle.overlay_enabled,
            use_file_cache=True,
        )
        actual = captured["decision"]
        results[label] = {
            "actual_prices": raw_prices, "expected_0910_prices": expected_prices,
            "max_gap_abs_diff": float(np.max(np.abs(captured_inputs.known.jp_gap_returns - corrected_inputs.known.jp_gap_returns))),
            "max_mu_abs_diff": float(np.max(np.abs(actual.mu_gap - corrected.mu_gap))),
            "max_score_abs_diff": float(np.max(np.abs(actual.scores - corrected.scores))),
            "max_weight_abs_diff": float(np.max(np.abs(actual.w_final - corrected.w_final))),
            "actual_weights": actual.w_final.tolist(), "corrected_weights": corrected.w_final.tolist(),
            "backtest_flat_flag": bool(flags[0]),
            "actual_fallback": actual.fallback, "corrected_fallback": corrected.fallback,
            "actual_numerical_audit": actual.numerical, "corrected_numerical_audit": corrected.numerical,
        }
        if label == "nonzero_0910":
            with tempfile.TemporaryDirectory(prefix="leadlag-review-training-") as directory:
                training = _collect_training_data(
                    pd.DatetimeIndex([date]), frame,
                    compute_jp_target_returns(frame, JP_TICKERS, open_910_returns=intraday),
                    Path(directory), app.v2,
                    pd.DataFrame(0.02, index=frame.index, columns=JP_TICKERS),
                    open_910_returns=intraday, historical_inputs=history,
                    decision_model=build_v2_model_bundle(app).decision_model,
                )
            assert len(training) == len(JP_TICKERS)
            trained = training.set_index("ticker").loc[JP_TICKERS]
            results[label]["training"] = {
                "rows": len(trained),
                "max_gap_diff_from_0910": float(np.max(np.abs(trained["gap"].to_numpy() - corrected_inputs.known.jp_gap_returns))),
                "max_score_diff_from_bt": float(np.max(np.abs(trained["score"].to_numpy() - actual.scores))),
                "max_score_diff_from_0910": float(np.max(np.abs(trained["score"].to_numpy() - corrected.scores))),
            }
    results["method"] = {
        "config": "Inherited production config, ONLY ml_overlay_enabled=False in isolated probe; actual BLPX/MH",
        "production_bootstrap": production_bootstrap,
        "frame": "Fixed 2026-08-14 regression frame; fixed 300-row PIT history",
        "intraday": "Synthetic controlled perturbation, NOT live evidence or performance evaluation",
        "macro_adr_rank": "Explicitly absent in both paths, identical safe-skip behavior",
    }
    assert results["nonzero_0910"]["max_gap_abs_diff"] > 0.005
    assert results["nonzero_0910"]["max_mu_abs_diff"] > 1e-6
    assert results["nonzero_0910"]["training"]["max_gap_diff_from_0910"] > 0.005
    return results


def deadline_probe() -> dict:
    app = load_config_from_yaml(ROOT / "configs/production/production.yaml", strict=True)
    app = app.model_copy(update={"v2": app.v2.model_copy(update={"ml_overlay_enabled": False})})
    frame = pd.DataFrame({"x": [1.0, 2.0]}, index=pd.to_datetime(["2026-08-13", "2026-08-14"]))
    history = HistoricalInputs(frame)
    actual_property = HistoricalInputs.fingerprint
    count = 0

    def slow_fingerprint(self):
        nonlocal count
        count += 1
        time.sleep(0.5)
        return actual_property.fget(self)

    with tempfile.TemporaryDirectory(prefix="leadlag-review-var-") as directory:
        with (
            patch.object(var_history, "load_df_exec_from_local_cache", return_value=frame),
            patch.object(var_history, "load_config_from_yaml", return_value=app),
            patch.object(var_history.var_inputs, "_snapshot_gap_input", return_value=(None, "none", None)),
            patch.object(var_history, "_build_var_historical_inputs", return_value=history),
            patch.object(HistoricalInputs, "fingerprint", property(slow_fingerprint)),
            patch.object(var_history.var_inputs, "_file_manifest_fingerprint", return_value="code") as code_hash,
        ):
            start = time.monotonic()
            output = var_history.get_hist_returns_for_risk(
                None, {"var_history_timeout": 0.1}, directory, pd.Timestamp("2026-08-17")
            )
            elapsed = time.monotonic() - start
    assert count == 1 and elapsed >= 0.5 and output.empty and not code_hash.called
    return {
        "configured_timeout_seconds": 0.1, "injected_fingerprint_delay_seconds": 0.5,
        "elapsed_seconds": elapsed, "empty_risk_history": bool(output.empty),
        "fingerprint_calls": count, "subsequent_code_hash_called": code_hash.called,
        "method": "Controlled finite delay at real HistoricalInputs.fingerprint; local cache, no worker or broker",
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    results = {"prices": prices_probe(), "var_deadline": deadline_probe()}
    (OUT / "probe_results.json").write_text(json.dumps(results, indent=2, default=str) + "\n")
    print(json.dumps(results, indent=2, default=str))
