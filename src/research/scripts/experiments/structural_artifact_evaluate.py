"""Evaluate the predeclared, fixed structural acceptance candidate.

Run after structural_artifact_acceptance.py collect, with an external watchdog.
No broker access, production config mutation, or model parameter search.
"""
from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from leadlag.data.pit_lake import PITDataLake
from leadlag.data.tickers import JP_TICKERS
from leadlag.domain.inputs import DecisionInputs
from leadlag.execution.backtester import BacktestEngine
from leadlag.execution.config import load_config_from_yaml
from leadlag.experiment_registry import Decision, ExperimentRecord, ExperimentRegistry
from leadlag.models.ml_order_overlay import DEFAULT_LGBM_KWARGS
from leadlag.models.ml_overlay_artifact import load_overlay_model, save_overlay_model
from leadlag.models.ml_overlay_inference import apply_overlay
from leadlag.runner.production import ProductionRunner
from leadlag.utils.dataframe_fingerprint import dataframe_fingerprint
from research.experiments.ml_overlay_training import _train_overlay_lgbm
from research.scripts.experiments.structural_artifact_acceptance import (
    GAP,
    REPORT,
    ROOT,
    WORK,
    dump,
    load,
    resources,
    write_json,
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def summarize(result, decisions, dates, leverage):
    net = result["daily_returns"].to_numpy()
    gross = result["daily_returns_gross"].to_numpy()
    costs = result["daily_costs"].to_numpy()
    parts = ["slip", "financing", "borrow", "reverse"]
    component_sum = sum(result[f"daily_{part}_costs"].to_numpy() for part in parts)
    if not np.isfinite(np.r_[net, gross, costs]).all():
        raise ValueError("Non-finite P&L: no dates may be dropped from the evaluation")
    np.testing.assert_allclose(gross - costs, net, atol=1e-14, rtol=0)
    np.testing.assert_allclose(component_sum, costs, atol=1e-14, rtol=0)
    weights = result["weights"].to_numpy()
    net_exposure = np.abs(weights.sum(axis=1))
    gross_exposure = np.abs(weights).sum(axis=1)
    assert net_exposure.max() <= 0.05 + 1e-10
    assert gross_exposure.max() <= 2.0 + 1e-10
    def sharpe(values):
        sigma = np.std(values, ddof=1)
        return float(np.mean(values) / sigma * np.sqrt(245)) if sigma > 1e-16 else None
    available = [decisions[date] for date in dates if date in decisions]
    return {
        "start": str(dates[0].date()), "end": str(dates[-1].date()), "days": len(dates),
        "net_sharpe": sharpe(net), "gross_sharpe": sharpe(gross),
        "max_drawdown": float(result["drawdown"].min()),
        "net_cumulative_return": float(np.prod(1 + net) - 1),
        "daily_net_mean": float(net.mean()),
        "daily_turnover_mean_raw_weight_units": float(result["daily_turnover"].mean()),
        "terminal_fallback_days": int(result["daily_fallback"].sum()),
        "all_zero_weight_days": int((gross_exposure < 1e-12).sum()),
        "overlay_applied_days": sum(bool(x.summary.get("overlay_applied")) for x in available),
        "pit_multiplier_counts": dict(Counter(str(x.pit_binning.get("multiplier")) for x in available)),
        "pit_alert_counts": dict(Counter(a for x in available for a in x.alerts if "pit" in a.lower())),
        "numerical_audit_counts": dict(Counter(x.numerical.get("status", "missing") for x in available)),
        "leakage_audit_counts": dict(Counter(x.leakage.get("status", "missing") for x in available)),
        "missing_decision_dates": [str(d.date()) for d in dates if d not in decisions],
        "distribution_route_counts": "not instrumented; owned gap directory contains no distribution bundles",
        "max_abs_model_net": float(net_exposure.max()), "max_model_gross": float(gross_exposure.max()),
        "max_abs_effective_net": float(net_exposure.max() * leverage),
        "max_effective_gross": float(gross_exposure.max() * leverage),
        "cost_period_sum_return_fraction": {part: float(result[f"daily_{part}_costs"].sum()) for part in parts},
        "total_cost_period_sum_return_fraction": float(costs.sum()),
        "cost_identity_max_error": float(np.max(np.abs(gross - costs - net))),
    }


def simulate(frame, history, app, decisions, dates):
    costs = BacktestEngine._resolve_v2_backtest_cost_params(app, *([None] * 7))
    weights = np.asarray([decisions[d].w_final if d in decisions else np.zeros(17) for d in dates])
    flags = np.asarray([d not in decisions or bool(decisions[d].fallback.get("gap_data_missing")
                         or decisions[d].fallback.get("audit_failure")) for d in dates])
    summaries = [decisions[d].summary if d in decisions else {"trade_date": str(d.date()),
                 "error": "collector rejected decision; retained as flat"} for d in dates]
    target, gap = BacktestEngine._compute_target_and_gap_returns(
        frame, frame.index, dates, open_910_returns=history.open_910_returns)
    pnl = BacktestEngine._simulate_daily_pnl(
        weights, target, gap, dates, costs["slip_bps"] / 10000,
        costs["fin_annual"] / 365, costs["borrow_annual"] / 365, costs["rev_bps"] / 10000,
        costs["alpha_long"], costs["alpha_short"], costs["side_leverage"],
    )
    result = BacktestEngine._assemble_v2_results(
        pnl, pd.DataFrame(weights, index=dates, columns=JP_TICKERS), flags, summaries, dates,
        costs["alpha_long"], costs["alpha_short"], costs["side_leverage"],
    )
    return result, summarize(result, decisions, dates, costs["side_leverage"]), costs


def train_candidate(training, frame, config, year, code_hash):
    prior_dates = frame.index[(frame.index >= "2015-01-05") & (frame.index < f"{year}-01-01")]
    cutoff = prior_dates[-6]
    rows = training.loc[training.trade_date <= cutoff].copy()
    if rows.empty:
        raise ValueError(f"No training observations before {cutoff}")
    directory = WORK / "artifacts" / f"evaluation_{year}"
    model = _train_overlay_lgbm(rows, lgbm_kwargs=DEFAULT_LGBM_KWARGS,
                                per_ticker_interactions=True, p_trade_scale=1.0)
    provenance = {
        "metadata_status": "verified", "train_start": "2015-01-05", "train_end": str(cutoff.date()),
        "label_asof_end": str(rows.trade_date.max().date()),
        "data_hash": dataframe_fingerprint(rows.reset_index(drop=True)),
        "config_hash": digest(config.model_dump(mode="json")), "source_code_hash": code_hash,
        "input_provenance_sha256": hashlib.sha256((REPORT / "input_provenance.json").read_bytes()).hexdigest(),
        "evaluation_plan_sha256": hashlib.sha256((REPORT / "evaluation_plan.md").read_bytes()).hexdigest(),
        "training_rows": len(rows), "training_dates": int(rows.trade_date.nunique()),
        "purged_trading_days": 5, "historical_provider_available_at_proven": False,
        "availability_evidence": "session convention; actual historical retrieval times unavailable",
        "seed": 42, "target_type": "raw", "lgbm_kwargs": DEFAULT_LGBM_KWARGS,
    }
    save_overlay_model(model, directory, provenance)
    loaded = load_overlay_model(directory)
    return loaded, directory, provenance


def apply_candidate(base, dates, frame, history, model):
    decisions = {}
    lake = PITDataLake(frame)
    intraday = history.open_910_returns
    for date in dates:
        if date not in base:
            continue
        as_of = date + pd.Timedelta(hours=9, minutes=10)
        snapshot = lake.get_execution_snapshot(as_of, intraday)
        decisions[date] = apply_overlay(
            base[date], history.calculation_frame(as_of), model, str(date.date()),
            snapshot=snapshot, adr_features=history.adr_features, allow_implicit_io=False,
        )
    return decisions


def block_interval(delta):
    rng = np.random.default_rng(42)
    count, block = len(delta), 20
    samples = []
    for _ in range(1000):
        starts = rng.integers(0, count, int(np.ceil(count / block)))
        indices = np.concatenate([(start + np.arange(block)) % count for start in starts])[:count]
        samples.append(float(np.mean(delta[indices])))
    return {"mean_daily_difference": float(np.mean(delta)), "block_days": block, "samples": 1000,
            "seed": 42, "mean_daily_difference_ci95": np.quantile(samples, [.025, .975]).tolist()}


def parity(frame, history, app, model_dir, decisions):
    candidate_app = app.model_copy(deep=True, update={"v2": app.v2.model_copy(
        deep=True, update={"ml_overlay_enabled": True, "ml_overlay_model_dir": str(model_dir)})})
    runner = ProductionRunner(candidate_app)
    lake = PITDataLake(frame)
    observed = history.open_910_returns.notna().sum(axis=1)
    after = observed.loc[observed.index >= "2025-01-01"]
    dates = []
    for mask in (after == 17, (after > 0) & (after < 17), after == 0):
        if mask.any():
            dates.append(after[mask].index[-1])
    rank = history.rank_reversal_signals
    if rank is not None and len(rank):
        dates.append(rank.index[-1])
    output = []
    for date in sorted(set(dates)):
        as_of = date + pd.Timedelta(hours=9, minutes=10)
        snapshot = lake.get_execution_snapshot(as_of, history.open_910_returns)
        known = snapshot.to_known_inputs(sig_date=frame.loc[date, "sig_date"],
                                        source="fixed_acceptance_runner")
        actual = runner.run(DecisionInputs(known=known, historical=history, gap_input_dir=GAP,
                                          use_file_cache=True))
        bt = BacktestEngine.run_v2_backtest(candidate_app, GAP, frame, str(date.date()),
                                           str(date.date()), historical_inputs=history)
        fast = decisions[date]
        comparison = {"date": str(date.date()), "price_sources": dict(snapshot.price_sources),
                      "live_bt_max_weight_error": float(np.max(np.abs(actual.w_final - bt["weights"].iloc[0].values))),
                      "live_collected_max_weight_error": float(np.max(np.abs(actual.w_final - fast.w_final))),
                      "gross": float(np.abs(actual.w_final).sum()), "fallback": actual.fallback,
                      "overlay_applied": actual.summary.get("overlay_applied", 0),
                      "numerical": actual.numerical["status"], "leakage": actual.leakage["status"]}
        assert comparison["live_bt_max_weight_error"] <= 1e-10, comparison
        assert comparison["live_collected_max_weight_error"] <= 1e-10, comparison
        assert comparison["gross"] > 0 and comparison["overlay_applied"] == 1, comparison
        assert comparison["numerical"] == "PASSED" and comparison["leakage"] == "PASSED", comparison
        output.append(comparison)
        dump(WORK / "parity" / f"{date.date()}.pkl", {"live": actual, "backtest": bt})
    write_json(REPORT / "artifact_parity.json", output)
    return output


def main():
    logging.basicConfig(level=logging.ERROR)
    frame, history, config = resources()
    app = load_config_from_yaml(ROOT / "configs/production/production.yaml", strict=True)
    assert config == app.v2, "Effective config changed after the input snapshot was fixed"
    paths = [WORK / "collected" / f"chunk_{i}.pkl" for i in range(4)]
    chunks = [load(path) for path in paths]
    training = pd.concat([c["training"] for c in chunks], ignore_index=True).sort_values(["trade_date", "ticker"])
    base = {key: value for c in chunks for key, value in c["decisions"].items()}
    code_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted((ROOT / "src").rglob("*.py"))}
    write_json(REPORT / "evaluation_source_manifest.json", code_hashes)
    output = {"folds": [], "candidate_count": 1, "annual_factor": 245,
              "historically_used_periods": True, "dsr": None,
              "dsr_reason": "No new parameters; independent historical trial variance unavailable"}
    combined_base, combined_candidate = {}, {}
    final_dir = None
    for year in range(2020, 2026):
        end = f"{year}-12-31" if year < 2025 else "2026-08-17"
        dates = frame.index[(frame.index >= f"{year}-01-01") & (frame.index <= end)]
        model, directory, provenance = train_candidate(training, frame, config, year, digest(code_hashes))
        candidate = apply_candidate(base, dates, frame, history, model)
        baseline_results, baseline_summary, costs = simulate(frame, history, app, base, dates)
        candidate_results, candidate_summary, _ = simulate(frame, history, app, candidate, dates)
        fold = {"evaluation_year": year, "artifact": str(directory.relative_to(ROOT)),
                "artifact_version": model.metadata["artifact_version"], "provenance": provenance,
                "baseline": baseline_summary, "candidate": candidate_summary,
                "paired_bootstrap": block_interval((candidate_results["daily_returns"] - baseline_results["daily_returns"]).to_numpy())}
        output["folds"].append(fold)
        output["costs"] = costs
        dump(WORK / "evaluation" / f"{year}.pkl", {"baseline": baseline_results, "candidate": candidate_results})
        combined_base.update({d: base[d] for d in dates if d in base})
        combined_candidate.update(candidate)
        print(json.dumps({"year": year, "baseline_net_sharpe": baseline_summary["net_sharpe"],
                          "candidate_net_sharpe": candidate_summary["net_sharpe"]}), flush=True)
        write_json(REPORT / "artifact_evaluation_partial.json", output)
        if year == 2025:
            final_dir = directory
    dates = frame.index[(frame.index >= "2020-01-01") & (frame.index <= "2026-08-17")]
    baseline_results, baseline_summary, _ = simulate(frame, history, app, combined_base, dates)
    candidate_results, candidate_summary, _ = simulate(frame, history, app, combined_candidate, dates)
    output["aggregate"] = {"baseline": baseline_summary, "candidate": candidate_summary,
                           "paired_bootstrap": block_interval((candidate_results["daily_returns"] - baseline_results["daily_returns"]).to_numpy())}
    output["numerical_promotion_criteria_pass"] = bool(
        candidate_summary["net_sharpe"] >= baseline_summary["net_sharpe"]
        and candidate_summary["max_drawdown"] >= baseline_summary["max_drawdown"])
    output["production_promoted"] = False
    output["candidate_artifact"] = str(final_dir.relative_to(ROOT))
    output["parity"] = parity(frame, history, app, final_dir, combined_candidate)
    dump(WORK / "evaluation" / "aggregate.pkl", {"baseline": baseline_results, "candidate": candidate_results})
    write_json(REPORT / "artifact_evaluation.json", output)
    record = ExperimentRecord(
        "20260922_structural_artifact_fixed_candidate", "Versioned retraining and common 09:10 inputs preserve usable chronological OOS behavior",
        end_time=datetime.now(UTC), parameters={"plan": str(REPORT / "evaluation_plan.md"),
            "source_code_hash": digest(code_hashes), "folds": [2020, 2021, 2022, 2023, 2024, 2025], "candidate_count": 1},
        metrics={**candidate_summary, "dsr": None, "numerical_promotion_criteria_pass": output["numerical_promotion_criteria_pass"]},
        decision=Decision.PENDING if output["numerical_promotion_criteria_pass"] else Decision.REJECTED,
        report_path="reports/20260922_production_acceptance/report.md",
    )
    registry = ExperimentRegistry(ROOT / "var/experiments/registry.jsonl")
    if not any(previous.name == record.name for previous in registry):
        registry.record(record)
    print(json.dumps({"artifact": output["candidate_artifact"], "promotion_criteria": output["numerical_promotion_criteria_pass"],
                      "parity_dates": len(output["parity"])}), flush=True)


if __name__ == "__main__":
    main()
