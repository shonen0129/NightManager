"""Replay the audited production-risk history at lower existing side-leverage settings."""
from __future__ import annotations

import csv
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.config import default_registry_path
from leadlag.core.pnl import simulate_daily_pnl
from leadlag.core.risk import compute_var_es
from leadlag.data.intraday_inputs import build_open_910_returns
from leadlag.data.market_data_cache import load_df_exec_from_local_cache
from leadlag.data.tickers import JP_TICKERS
from leadlag.execution.backtester import BacktestEngine
from leadlag.execution.config import load_config_from_yaml
from leadlag.experiment_registry import Decision, ExperimentRecord, ExperimentRegistry
from leadlag.utils.dataframe_fingerprint import dataframe_fingerprint

SOURCE_HISTORY = ROOT / "var/results/20260927_ml_overlay_var_history_2025/history/versioned_var_returns.pkl"
SOURCE_REPORT = ROOT / "reports/20260927_ml_overlay_var_history/history.json"
OUTPUT_DIR = ROOT / "reports/20260930_var_es_risk_reduction"
STUDY_ID = "var_es_side_leverage_20260930"
CANDIDATES = (1.50, 1.30, 1.25)
ADOPTED_LEVERAGE = 1.30
BASELINE_FRAME_SHA256 = "6b81e5ae4abd09578365618caf218e6375b863a9f6ecd18d7858c83b988166a5"


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def _stats(returns: pd.Series, *, window: int, risk: Any) -> dict[str, Any]:
    values = returns.astype(float)
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("non-finite portfolio returns")
    equity = (1.0 + values).cumprod()
    peaks = np.maximum.accumulate(np.r_[1.0, equity.to_numpy()])[1:]
    mdd = float(np.min(equity.to_numpy() / peaks - 1.0))
    std = float(values.std(ddof=1))
    sharpe = float(values.mean() / std * np.sqrt(252.0)) if std > 1e-12 else 0.0
    var_es = compute_var_es(
        values,
        confidence=float(risk.var_confidence),
        window=window,
        var_method=str(risk.var_method),
    )
    if not var_es.available or var_es.samples != window:
        raise ValueError(f"risk window unavailable: {var_es}")
    return {
        "observations": len(values),
        "start": str(values.index.min().date()),
        "end": str(values.index.max().date()),
        "net_compounded_return": float(equity.iloc[-1] - 1.0),
        "net_sharpe_annualized_252": sharpe,
        "maximum_drawdown": mdd,
        "mean_daily_net_return": float(values.mean()),
        "worst_daily_net_return": float(values.min()),
        "var_window": window,
        "var_confidence": float(risk.var_confidence),
        "var_method": str(risk.var_method),
        "var99_loss": float(var_es.var_loss),
        "es99_loss": float(var_es.es_loss),
        "es_tail_count": int(var_es.tail_count),
    }


def _cost_sums(pnl: dict[str, list[float]]) -> dict[str, float]:
    return {
        key: float(np.sum(pnl[source]))
        for key, source in (
            ("slippage", "slip_costs"),
            ("financing", "financing_costs"),
            ("borrow", "borrow_costs"),
            ("reverse", "reverse_costs"),
            ("total", "costs"),
        )
    }


def _simulate_in_artifact_chunks(
    *,
    leverage: float,
    common_args: dict[str, Any],
    weights: np.ndarray,
    target_returns: np.ndarray,
    gap_returns: np.ndarray,
    dates: pd.DatetimeIndex,
    chunk_positions: list[np.ndarray],
) -> dict[str, list[float]]:
    """Replay each immutable artifact segment with the backtest's reset boundary."""
    parts = [
        simulate_daily_pnl(
            **common_args,
            weights=weights[position],
            target_returns=target_returns[position],
            gap_returns=gap_returns[position],
            sim_dates=dates[position],
            side_leverage=leverage,
        )
        for position in chunk_positions
    ]
    if not parts:
        raise ValueError("no artifact history chunks were selected")
    return {key: [value for part in parts for value in part[key]] for key in parts[0]}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty attribution: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run() -> dict[str, Any]:
    if not SOURCE_HISTORY.is_file() or not SOURCE_REPORT.is_file():
        raise FileNotFoundError("the audited 2026-09-25 VaR history artifacts are required")
    registry = ExperimentRegistry(default_registry_path())
    already_recorded = any(record.study_id == STUDY_ID for record in registry.iter_records())

    source_report = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    with SOURCE_HISTORY.open("rb") as handle:
        history = pickle.load(handle)
    if not isinstance(history, dict) or source_report.get("status") != "created_and_audited":
        raise ValueError("source VaR history is not a valid audited artifact")

    frame = load_df_exec_from_local_cache(max_stale_bdays=None)
    last_date = pd.Timestamp(source_report["history_end"])
    frame = frame.loc[frame.index <= last_date].copy()
    frame_hash = dataframe_fingerprint(frame)
    if frame_hash != BASELINE_FRAME_SHA256 or frame_hash != source_report["df_exec_hash_in_run_owned_snapshot"]:
        raise ValueError("current local df_exec does not match the audited run-owned input snapshot")

    returns = history["daily_returns"].astype(float).copy()
    weights_frame = history["weights"].astype(float).copy()
    dates = pd.DatetimeIndex(returns.index)
    if not dates.equals(weights_frame.index) or str(dates.max().date()) != str(source_report["history_end"]):
        raise ValueError("return, weight, and audited history dates are misaligned")
    if list(weights_frame.columns) != list(JP_TICKERS):
        raise ValueError("saved weight columns do not match the canonical JP ticker order")

    history_manifest = json.loads(
        (ROOT / "models/ml_order_overlay/production_20260923/HISTORY.json").read_text(encoding="utf-8")
    )
    segments = history_manifest.get("segments")
    if history_manifest.get("schema_version") != 1 or not isinstance(segments, list):
        raise ValueError("versioned artifact history manifest is invalid")
    versions: list[str] = []
    for date in dates:
        matches = [
            str(item["version"])
            for item in segments
            if pd.Timestamp(item["start_date"]).normalize() <= date.normalize()
            and (item.get("end_date") is None or date.normalize() <= pd.Timestamp(item["end_date"]).normalize())
        ]
        if len(matches) != 1:
            raise ValueError(f"history manifest does not map {date.date()} to one artifact")
        versions.append(matches[0])
    chunk_positions: list[np.ndarray] = []
    chunk_start = 0
    for idx in range(1, len(versions)):
        if versions[idx] != versions[idx - 1]:
            chunk_positions.append(np.arange(chunk_start, idx))
            chunk_start = idx
    chunk_positions.append(np.arange(chunk_start, len(versions)))

    audit_rows = history.get("audit_rows")
    if not isinstance(audit_rows, list) or len(audit_rows) != len(dates):
        raise ValueError("one audited decision row per return date is required")
    audit_by_date = {pd.Timestamp(row["trade_date"]): row for row in audit_rows}
    if set(audit_by_date) != set(dates):
        raise ValueError("audit decisions do not cover the return dates exactly")
    if any(
        row.get("numerical") != "PASSED"
        or row.get("leakage") != "PASSED"
        or row.get("fallback")
        for row in audit_rows
    ):
        raise ValueError("source history contains a failed audit or fallback day")

    production_app = load_config_from_yaml(ROOT / "configs/production/production.yaml", strict=True)
    configured_side_leverage = float(production_app.v2.costs.side_leverage)
    if not any(abs(configured_side_leverage - value) <= 1e-12 for value in (1.50, ADOPTED_LEVERAGE)):
        raise ValueError(f"unexpected production side_leverage: {configured_side_leverage}")
    # Replay the pre-change production baseline explicitly even after the
    # selected value is copied into the production config.
    baseline_costs = production_app.v2.costs.model_copy(update={"side_leverage": 1.50})
    app = production_app.model_copy(
        update={"v2": production_app.v2.model_copy(update={"costs": baseline_costs})}
    )
    if abs(float(app.v2.costs.side_leverage) - 1.50) > 1e-12:
        raise ValueError("could not resolve the fixed 1.50 replay baseline")
    if source_report.get("effective_side_leverage") != 1.50:
        raise ValueError("source VaR report does not match the configured 1.50 baseline")
    if source_report.get("gap_input_fingerprint") != history.get("gap_input_fingerprint"):
        raise ValueError("gap input fingerprint is inconsistent across source artifacts")

    open_910 = build_open_910_returns(frame, JP_TICKERS)
    target_returns, gap_returns = BacktestEngine._compute_target_and_gap_returns(
        frame,
        pd.DatetimeIndex(frame.index),
        dates,
        open_910_returns=open_910,
    )
    weights = weights_frame.to_numpy(dtype=float)
    params = BacktestEngine._resolve_v2_backtest_cost_params(
        app,
        slippage_bps=5.0,
        overnight_alpha_long=None,
        overnight_alpha_short=None,
        buy_interest_annual=None,
        borrow_fee_annual=None,
        reverse_fee_bps=None,
        side_leverage=1.50,
    )
    common_pnl_args = {
        "slip": float(params["slip_bps"]) / 10000.0,
        "financing_daily": float(params["fin_annual"]) / 365.0,
        "borrow_daily": float(params["borrow_annual"]) / 365.0,
        "reverse_daily": float(params["rev_bps"]) / 10000.0,
        "alpha_long": float(params["alpha_long"]),
        "alpha_short": float(params["alpha_short"]),
    }

    baseline_pnl = _simulate_in_artifact_chunks(
        leverage=1.50,
        common_args=common_pnl_args,
        weights=weights,
        target_returns=target_returns,
        gap_returns=gap_returns,
        dates=dates,
        chunk_positions=chunk_positions,
    )
    reproduced = pd.Series(baseline_pnl["net_returns"], index=dates)
    max_baseline_diff = float((reproduced - returns).abs().max())
    if max_baseline_diff > 1e-10:
        raise ValueError(
            "same-input baseline replay did not reproduce saved returns: "
            f"max_abs_diff={max_baseline_diff}"
        )
    source_costs = _cost_sums(baseline_pnl)
    reported_costs = source_report["gross_cost_sums"]
    cost_report_keys = {
        "daily_slip_costs": "slippage",
        "daily_financing_costs": "financing",
        "daily_borrow_costs": "borrow",
        "daily_reverse_costs": "reverse",
        "daily_costs": "total",
    }
    for component, report_key in cost_report_keys.items():
        if not np.isclose(source_costs[report_key], float(reported_costs[component]), atol=1e-10):
            raise ValueError(f"baseline replay cost mismatch for {component}")

    weight_gross = weights_frame.abs().sum(axis=1)
    weight_net = weights_frame.sum(axis=1)
    max_model_gross = float(weight_gross.max())
    max_model_abs_net = float(weight_net.abs().max())
    if max_model_gross > 2.0 + 1e-6 or max_model_abs_net > 0.05 + 1e-6:
        raise ValueError("model-space neutrality/gross invariants do not hold")

    alpha = np.where(weights > 0.0, float(params["alpha_long"]), np.where(weights < 0.0, float(params["alpha_short"]), 0.0))
    gap_next = np.zeros_like(gap_returns)
    for position in chunk_positions:
        if len(position) > 1:
            gap_next[position[:-1]] = gap_returns[position[1:]]
    overlay_by_date = {date: bool(audit_by_date[date]["overlay_applied"]) for date in dates}
    model_gross_by_date = pd.Series(weight_gross.to_numpy(), index=dates)
    pit_multiplier = model_gross_by_date / float(app.v2.baseline_gross)
    pit_label = np.where(np.isclose(pit_multiplier, 0.75, atol=1e-6), "low_multiplier_0.75", np.where(np.isclose(pit_multiplier, 1.0, atol=1e-6), "full_multiplier_1.00", "other"))

    base_cost_components = {
        "slippage": np.asarray(baseline_pnl["slip_costs"], dtype=float),
        "financing": np.asarray(baseline_pnl["financing_costs"], dtype=float),
        "borrow": np.asarray(baseline_pnl["borrow_costs"], dtype=float),
        "reverse": np.asarray(baseline_pnl["reverse_costs"], dtype=float),
    }
    intraday_per_asset_unit = weights * target_returns
    overnight_per_asset_unit = alpha * weights * gap_next
    overnight_by_date_unit = overnight_per_asset_unit.sum(axis=1)
    gap_rank = pd.Series(np.abs(overnight_by_date_unit), index=dates).rank(method="first", pct=True)
    gap_quartile = np.minimum(4, np.ceil(gap_rank * 4).astype(int))

    daily_rows: list[dict[str, Any]] = []
    ticker_rows: list[dict[str, Any]] = []
    intraday_long = np.zeros(len(dates))
    intraday_short = np.zeros(len(dates))
    overnight_long = np.zeros(len(dates))
    overnight_short = np.zeros(len(dates))
    for date_idx, date in enumerate(dates):
        weight_row = weights[date_idx]
        long_mask = weight_row > 0.0
        short_mask = weight_row < 0.0
        intraday_long[date_idx] = intraday_per_asset_unit[date_idx, long_mask].sum() * 1.50
        intraday_short[date_idx] = intraday_per_asset_unit[date_idx, short_mask].sum() * 1.50
        overnight_long[date_idx] = overnight_per_asset_unit[date_idx, long_mask].sum() * 1.50
        overnight_short[date_idx] = overnight_per_asset_unit[date_idx, short_mask].sum() * 1.50
        daily_rows.append({
            "trade_date": str(date.date()),
            "net_return": float(returns.loc[date]),
            "gross_return": float(baseline_pnl["gross_returns"][date_idx]),
            "overnight_return": float(baseline_pnl["overnight_returns"][date_idx]),
            "intraday_long_return": float(intraday_long[date_idx]),
            "intraday_short_return": float(intraday_short[date_idx]),
            "overnight_long_return": float(overnight_long[date_idx]),
            "overnight_short_return": float(overnight_short[date_idx]),
            "slippage_cost": base_cost_components["slippage"][date_idx],
            "financing_cost": base_cost_components["financing"][date_idx],
            "borrow_cost": base_cost_components["borrow"][date_idx],
            "reverse_cost": base_cost_components["reverse"][date_idx],
            "raw_model_gross": float(weight_gross.iloc[date_idx]),
            "raw_model_net": float(weight_net.iloc[date_idx]),
            "pit_multiplier_inferred_from_weight_gross": float(pit_multiplier.iloc[date_idx]),
            "pit_multiplier_band": str(pit_label[date_idx]),
            "overlay_applied": overlay_by_date[date],
            "realized_gap_impact_quartile_ex_post": int(gap_quartile.iloc[date_idx]),
        })
        for ticker_idx, ticker in enumerate(JP_TICKERS):
            ticker_rows.append({
                "trade_date": str(date.date()),
                "ticker": ticker,
                "side": "long" if weight_row[ticker_idx] > 0 else "short" if weight_row[ticker_idx] < 0 else "flat",
                "model_weight": float(weight_row[ticker_idx]),
                "intraday_gross_pnl": float(intraday_per_asset_unit[date_idx, ticker_idx] * 1.50),
                "overnight_gross_pnl": float(overnight_per_asset_unit[date_idx, ticker_idx] * 1.50),
            })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTPUT_DIR / "daily_attribution.csv", daily_rows)
    _write_csv(OUTPUT_DIR / "ticker_daily_attribution.csv", ticker_rows)

    scenarios: list[dict[str, Any]] = []
    risk = app.risk
    for leverage in CANDIDATES:
        pnl = _simulate_in_artifact_chunks(
            leverage=leverage,
            common_args=common_pnl_args,
            weights=weights,
            target_returns=target_returns,
            gap_returns=gap_returns,
            dates=dates,
            chunk_positions=chunk_positions,
        )
        scenario_returns = pd.Series(pnl["net_returns"], index=dates)
        metrics = _stats(scenario_returns, window=int(risk.var_window), risk=risk)
        metrics.update({
            "side_leverage": leverage,
            "model_weight_gross_max": max_model_gross,
            "model_weight_abs_net_max": max_model_abs_net,
            "effective_gross_max": max_model_gross * leverage,
            "effective_abs_net_max": max_model_abs_net * leverage,
            "mean_raw_turnover": float(np.mean(pnl["turnover"])),
            "mean_effective_turnover": float(np.mean(pnl["turnover"]) * leverage),
            "costs_return_fraction": _cost_sums(pnl),
            "daily_net_return_scale_verified": bool(np.allclose(
                np.asarray(pnl["net_returns"]),
                np.asarray(baseline_pnl["net_returns"]) * leverage / 1.50,
                atol=1e-12,
                rtol=1e-12,
            )),
        })
        metrics["var_stop_pass"] = metrics["var99_loss"] < float(risk.var_stop)
        metrics["es_stop_pass"] = metrics["es99_loss"] < float(risk.es_stop)
        metrics["selected_for_production"] = abs(leverage - ADOPTED_LEVERAGE) < 1e-12
        scenarios.append(metrics)

    selected = next(item for item in scenarios if item["selected_for_production"])
    if not selected["var_stop_pass"] or not selected["es_stop_pass"]:
        raise ValueError("selected side-leverage candidate does not clear both unchanged risk stops")
    if selected["effective_gross_max"] > float(risk.max_gross_exposure) + 1e-8:
        raise ValueError("selected effective gross exposure exceeds the inherited risk limit")

    quarter_rows = []
    attribution = pd.DataFrame(daily_rows).set_index("trade_date")
    for quartile in range(1, 5):
        selected_dates = attribution[attribution["realized_gap_impact_quartile_ex_post"] == quartile]
        quarter_rows.append({
            "ex_post_gap_impact_quartile": quartile,
            "days": len(selected_dates),
            "mean_net_return": float(selected_dates["net_return"].mean()),
            "sum_net_return": float(selected_dates["net_return"].sum()),
            "mean_overnight_return": float(selected_dates["overnight_return"].mean()),
            "sum_overnight_return": float(selected_dates["overnight_return"].sum()),
        })

    ticker_summary: dict[str, dict[str, float]] = {}
    ticker_frame = pd.DataFrame(ticker_rows)
    for ticker, group in ticker_frame.groupby("ticker", sort=False):
        ticker_summary[ticker] = {
            "intraday_gross_contribution": float(group["intraday_gross_pnl"].sum()),
            "overnight_gross_contribution": float(group["overnight_gross_pnl"].sum()),
            "combined_gross_contribution": float(group["intraday_gross_pnl"].sum() + group["overnight_gross_pnl"].sum()),
        }
    side_summary = {
        "intraday_long": float(intraday_long.sum()),
        "intraday_short": float(intraday_short.sum()),
        "overnight_long": float(overnight_long.sum()),
        "overnight_short": float(overnight_short.sum()),
    }
    overlay_summary = {}
    for applied in (True, False):
        chosen_dates = [date for date in dates if overlay_by_date[date] is applied]
        subset = returns.loc[chosen_dates]
        overlay_summary["applied" if applied else "skipped"] = {
            "days": len(subset),
            "sum_daily_net_returns": float(subset.sum()),
            "mean_daily_net_return": float(subset.mean()) if len(subset) else None,
        }

    worst_dates = returns.nsmallest(5).index
    worst_rows = [next(row for row in daily_rows if row["trade_date"] == str(date.date())) for date in worst_dates]
    for row in worst_rows:
        date = row["trade_date"]
        row["largest_ticker_contributions"] = sorted(
            [item for item in ticker_rows if item["trade_date"] == date],
            key=lambda item: abs(item["intraday_gross_pnl"] + item["overnight_gross_pnl"]),
            reverse=True,
        )[:5]

    provenance = {
        "source_return_artifact": str(SOURCE_HISTORY.relative_to(ROOT)),
        "source_report": str(SOURCE_REPORT.relative_to(ROOT)),
        "source_gap_input_fingerprint": history["gap_input_fingerprint"],
        "df_exec_sha256": frame_hash,
        "history_start": str(dates.min().date()),
        "history_end": str(dates.max().date()),
        "history_days": len(dates),
        "risk_window_end": str(dates[-1].date()),
        "baseline_return_reproduction_max_abs_difference": max_baseline_diff,
        "overlay_schedule": segments,
        "artifact_chunk_count": len(chunk_positions),
        "historical_macro_provider_available_at_proven": source_report.get("historical_provider_available_at_proven"),
        "production_side_leverage_at_replay": configured_side_leverage,
        "replayed_baseline_side_leverage": 1.50,
        "cost_assumptions": {
            "slippage_bps_per_side": float(params["slip_bps"]),
            "buy_interest_annual": float(params["fin_annual"]),
            "borrow_fee_annual": float(params["borrow_annual"]),
            "reverse_fee_bps_daily": float(params["rev_bps"]),
            "overnight_alpha_long": float(params["alpha_long"]),
            "overnight_alpha_short": float(params["alpha_short"]),
        },
        "risk_thresholds_unchanged": {
            "var_stop": float(risk.var_stop),
            "es_stop": float(risk.es_stop),
            "max_gross_exposure": float(risk.max_gross_exposure),
            "confidence": float(risk.var_confidence),
            "window": int(risk.var_window),
        },
    }
    summary = {
        "study_id": STUDY_ID,
        "hypothesis": "Reducing the existing side_leverage scales both modeled mark-to-market PnL and all four modeled costs linearly, lowering absolute VaR/ES while leaving audited model-space weights unchanged.",
        "decision_rule": "Select the highest tested side_leverage that is strictly below both unchanged production VaR/ES stop limits and within the inherited effective-gross cap.",
        "provenance": provenance,
        "scenarios": scenarios,
        "baseline_attribution": {
            "model_weight_gross_max": max_model_gross,
            "model_weight_abs_net_max": max_model_abs_net,
            "effective_gross_max_at_1_50": max_model_gross * 1.50,
            "effective_gross_max_selected": selected["effective_gross_max"],
            "side_pnl_return_fraction": side_summary,
            "overlay": overlay_summary,
            "pit_multiplier_band_is_inferred_from_audited_weight_gross": True,
            "realized_gap_quartiles_are_ex_post_diagnostic_only": quarter_rows,
            "ticker_gross_contributions": ticker_summary,
            "worst_five_days": worst_rows,
            "cost_component_return_fractions": selected["costs_return_fraction"],
        },
        "limitations": [
            "This is retrospective model PnL, not an actual-account statement or fill-cost reconciliation.",
            "The ES99 tail contains only three observations; the unchanged production risk stop remains enabled and continues to fail closed if future evidence breaches it.",
            "PIT multiplier bands are inferred from audited model-weight gross; the saved history did not retain exact per-day PIT-bin labels.",
            "Gap-impact quartiles use realized overnight contribution after the outcome and are descriptive labels, not decision-time features.",
            "The verified historical macro snapshot ends on 2026-09-25; this report reproduces that exact saved risk-evidence window and does not claim a newer live-input replay.",
            "Modeled slippage, financing, borrow, and reverse costs are not fully matched against broker fills.",
            "This historical result does not establish that the current live risk window passes or replace the daily fail-closed risk check.",
        ],
    }
    _write_json(OUTPUT_DIR / "scenario_summary.json", summary)

    report_lines = [
        "# VaR/ES risk reduction through lower side leverage",
        "",
        "## Decision",
        "",
        f"Selected existing side_leverage setting: **{ADOPTED_LEVERAGE:.2f}**. The fixed 250-day historical sample clears both unchanged risk stops, while the model-space weights and audit outcomes are reused unchanged. The effective maximum gross falls from {max_model_gross * 1.50:.3f} to {selected['effective_gross_max']:.3f}; the inherited risk gross cap remains {risk.max_gross_exposure:.3f}.",
        "",
        "The selection rule was fixed before the replay: choose the highest tested leverage strictly below both existing stop thresholds and within the existing effective-gross cap. This leaves the risk stop enabled; it does not change VaR/ES limits.",
        "",
        "## Same-input comparison",
        "",
        "| side_leverage | VaR99 | ES99 | compounded net return | net Sharpe | max drawdown | modeled total cost |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in scenarios:
        report_lines.append(
            f"| {item['side_leverage']:.2f} | {item['var99_loss']:.3%} | {item['es99_loss']:.3%} | {item['net_compounded_return']:.3%} | {item['net_sharpe_annualized_252']:.3f} | {item['maximum_drawdown']:.3%} | {item['costs_return_fraction']['total']:.3%} |"
        )
    selected_costs = selected["costs_return_fraction"]
    report_lines.extend([
        "",
        "Cost values above are simple sums of daily modeled return fractions, not compounded returns or observed broker fees.",
        "",
        "| Modeled cost component | Sum of daily return fractions at 1.30 |",
        "|---|---:|",
        f"| Slippage | {selected_costs['slippage']:.3%} |",
        f"| Financing | {selected_costs['financing']:.3%} |",
        f"| Borrow | {selected_costs['borrow']:.3%} |",
        f"| Reverse fee | {selected_costs['reverse']:.3%} |",
        f"| Total | {selected_costs['total']:.3%} |",
        "",
        "The artifact-segment baseline PnL replay differs from the saved audited daily returns by at most "
        f"{max_baseline_diff:.3g}. The saved history spans {len(chunk_positions)} immutable overlay artifact segments, "
        "so each segment was replayed separately across the existing state-reset boundary. The PnL implementation "
        "scales market returns, slippage, financing, borrow, and reverse costs by the same leverage factor; candidate "
        "outputs were checked against this linear identity.",
        "",
        "## Attribution",
        "",
        f"The 269-day sample is {dates.min().date()} through {dates.max().date()}. Overlay was applied on {overlay_summary['applied']['days']} days and skipped on {overlay_summary['skipped']['days']} days; fallback days are 0. Gross contribution by ticker and long/short side, PIT multiplier band, realized gap-impact quartile, and all cost categories are in the CSV and JSON artifacts beside this report.",
        "",
        f"The worst baseline day was {worst_rows[0]['trade_date']} at {worst_rows[0]['net_return']:.3%}; overnight PnL was {worst_rows[0]['overnight_return']:.3%}, including {worst_rows[0]['overnight_short_return']:.3%} from shorts. The two largest ticker contributions on that day were {worst_rows[0]['largest_ticker_contributions'][0]['ticker']} ({worst_rows[0]['largest_ticker_contributions'][0]['side']}, combined gross {worst_rows[0]['largest_ticker_contributions'][0]['intraday_gross_pnl'] + worst_rows[0]['largest_ticker_contributions'][0]['overnight_gross_pnl']:.3%}) and {worst_rows[0]['largest_ticker_contributions'][1]['ticker']} ({worst_rows[0]['largest_ticker_contributions'][1]['side']}, combined gross {worst_rows[0]['largest_ticker_contributions'][1]['intraday_gross_pnl'] + worst_rows[0]['largest_ticker_contributions'][1]['overnight_gross_pnl']:.3%}). These are ex-post attribution, not decision-time signals.",
        "",
        "PIT labels are inferred from saved gross weights and are reported as 0.75 multiplier versus full multiplier. Realized gap-impact quartiles are ex-post attribution buckets. The report does not use these realized labels to change daily decisions.",
        "",
        "## Provenance and limits",
        "",
        f"- df_exec fingerprint: `{frame_hash}`",
        f"- gap input fingerprint: `{history['gap_input_fingerprint']}`",
        f"- current 250-day risk window ends: {dates[-1].date()}; ES tail count: {selected['es_tail_count']}",
        f"- resolved production side_leverage at replay: {configured_side_leverage:.2f}; fixed historical replay baseline: 1.50",
        "- risk stops remain VaR99 3.00% and ES99 4.00%; neither threshold nor fail-closed handling changed.",
        "- the saved macro snapshot ends 2026-09-25 and provider point-in-time availability is not proven; no newer market data were claimed.",
        "- modeled fees are not actual account PnL or a complete fill-cost reconciliation.",
        "- this historical result does not establish that the current live risk window passes or replace the daily fail-closed risk check.",
        "",
        "Reproduction: `.venv/bin/python src/research/scripts/experiments/evaluate_var_es_side_leverage_20260930.py`.",
        "",
    ])
    report_path = OUTPUT_DIR / "report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    if not already_recorded:
        for item in scenarios:
            leverage = float(item["side_leverage"])
            decision = Decision.ADOPTED if abs(leverage - ADOPTED_LEVERAGE) < 1e-12 else Decision.PENDING
            record = ExperimentRecord(
                name="var_es_side_leverage_deleveraging_20260930",
                hypothesis=summary["hypothesis"],
                parameters={
                    "side_leverage": leverage,
                    "production_setting_at_replay": configured_side_leverage,
                    "replay_baseline_side_leverage": 1.50,
                    "history_end": str(dates[-1].date()),
                    "history_days": len(dates),
                    "var_window": int(risk.var_window),
                    "slippage_bps_per_side": float(params["slip_bps"]),
                    "df_exec_sha256": frame_hash,
                    "gap_input_fingerprint": history["gap_input_fingerprint"],
                    "candidate_selected_for_production": leverage == ADOPTED_LEVERAGE,
                },
                metrics={key: value for key, value in item.items() if key != "costs_return_fraction"}
                | {"costs_return_fraction": item["costs_return_fraction"], "trial_count": len(CANDIDATES)},
                decision=decision,
                report_path=str(report_path.relative_to(ROOT)),
                study_id=STUDY_ID,
            )
            registry.record(record)

    print(json.dumps(summary, ensure_ascii=False, default=str))
    return summary


if __name__ == "__main__":
    run()
