#!/usr/bin/env python3
"""Evaluate PIT, ticker-specific overnight inventory sizing against fixed alpha."""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from leadlag.core.pnl import simulate_daily_pnl
from leadlag.data.intraday_inputs import build_open_910_returns
from leadlag.data.market_data_cache import load_df_exec_from_local_cache
from leadlag.data.tickers import JP_TICKERS
from leadlag.execution.backtester import BacktestEngine
from leadlag.execution.config import load_config_from_yaml
from leadlag.experiment_registry import Decision
from leadlag.reporting.metrics import MetricsSpec, calculate_metrics
from research.experiment_utils import record_backtest_experiment
from research.overnight_inventory import (
    AdaptiveCarryConfig,
    adaptive_carry_masks,
    attribute_inventory_transitions,
    realized_inventory_reuse,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = ROOT / "configs" / "research" / "adaptive_overnight_inventory_20261008.yaml"
PRODUCTION_CONFIG_PATH = ROOT / "configs" / "production" / "production.yaml"
OUTPUT_DIR = ROOT / "reports" / "20261008_adaptive_overnight_inventory"
REPORT_PATH = OUTPUT_DIR / "report.md"
ANNUALIZATION = 245
STUDY_ID = "adaptive-overnight-inventory-pit-2026-10-08"


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        result = yaml.safe_load(handle)
    if not isinstance(result, dict):
        raise ValueError(f"Expected a YAML mapping at {path}")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).date().isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return [_safe_json(item) for item in value.tolist()]
    return value


def _calendar_days(dates: pd.DatetimeIndex) -> np.ndarray:
    """Incoming intervals for fees on previous-close inventory."""
    values = np.zeros(len(dates), dtype=float)
    if len(dates) > 1:
        values[1:] = np.diff(dates.values).astype("timedelta64[D]").astype(int)
    return values


def _read_gap_store_coverage(path: Path) -> dict[str, Any]:
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        row = connection.execute(
            "SELECT COUNT(DISTINCT trade_date), MIN(trade_date), MAX(trade_date) "
            "FROM gap_matrices WHERE matrix_type='mu' AND horizon=-1"
        ).fetchone()
    if row is None or not row[0]:
        return {"dates": 0, "start": None, "end": None}
    return {"dates": int(row[0]), "start": str(row[1]), "end": str(row[2])}


def _versioned_model_weights(
    *,
    app_config: Any,
    df_exec: pd.DataFrame,
    gap_store: Path,
    start_date: str,
    end_date: str,
) -> tuple[pd.DataFrame, pd.Series, list[dict[str, Any]]]:
    """Generate each history segment with its train-end-safe overlay version."""
    artifact_root = ROOT / app_config.v2.ml_overlay_model_dir
    history_path = artifact_root / "HISTORY.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    requested_start = pd.Timestamp(start_date).normalize()
    requested_end = (
        pd.Timestamp(df_exec.index.max()).normalize()
        if end_date == "latest"
        else pd.Timestamp(end_date).normalize()
    )
    weight_parts: list[pd.DataFrame] = []
    fallback_parts: list[pd.Series] = []
    schedule: list[dict[str, Any]] = []
    for segment in history.get("segments", []):
        segment_start = max(requested_start, pd.Timestamp(segment["start_date"]).normalize())
        segment_end = min(
            requested_end,
            pd.Timestamp(segment["end_date"]).normalize()
            if segment.get("end_date")
            else requested_end,
        )
        if segment_start > segment_end:
            continue
        version = str(segment["version"])
        overlay_dir = artifact_root / "versions" / version
        metadata_path = overlay_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        train_end = pd.Timestamp(metadata["train_end"]).normalize()
        if train_end >= segment_start:
            raise ValueError(
                f"Overlay {version} trained through {train_end.date()} but segment begins "
                f"{segment_start.date()}"
            )
        segment_end_arg = "latest" if segment_end == requested_end and end_date == "latest" else str(segment_end.date())
        logger.info(
            "Running versioned model segment %s..%s with overlay %s (train_end=%s)",
            segment_start.date(), segment_end_arg, version, train_end.date(),
        )
        # Reuse the canonical immutable-version loader. It validates the
        # artifact pair without changing the production CURRENT pointer.
        from leadlag.execution.var_history import _load_overlay_version

        overlay_model = _load_overlay_version(artifact_root, version)
        segment_results = BacktestEngine.run_v2_backtest(
            cfg=app_config,
            gap_input_dir=gap_store,
            df_exec=df_exec,
            start_date=str(segment_start.date()),
            end_date=segment_end_arg,
            overlay_model=overlay_model,
            n_jobs=1,
        )
        overlay_statuses = Counter(
            str(item.get("overlay_status", "not_applicable"))
            for item in segment_results["v2_summaries"]
            if isinstance(item, dict)
        )
        overlay_reasons = Counter(
            str(item.get("overlay_reason"))
            for item in segment_results["v2_summaries"]
            if isinstance(item, dict) and item.get("overlay_status") == "skipped"
        )
        weight_parts.append(segment_results["weights"].reindex(columns=JP_TICKERS))
        fallback_parts.append(segment_results["daily_fallback"].astype(bool))
        schedule.append(
            {
                "segment_start": str(segment_start.date()),
                "segment_end_requested": segment_end_arg,
                "overlay_version": version,
                "overlay_train_end": str(train_end.date()),
                "overlay_model_sha256": metadata.get("model_sha256"),
                "n_weight_rows": int(len(segment_results["weights"])),
                "fallback_rate": float(segment_results["daily_fallback"].mean()),
                "overlay_status_counts": dict(overlay_statuses),
                "overlay_skip_reason_counts": dict(overlay_reasons),
            }
        )
    if not weight_parts:
        raise ValueError("No train-end-safe overlay versions cover the requested backtest period")
    weights = pd.concat(weight_parts).sort_index()
    fallback = pd.concat(fallback_parts).sort_index()
    if weights.index.has_duplicates or fallback.index.has_duplicates:
        raise ValueError("Overlay history segments overlap in their decision dates")
    if not weights.index.equals(fallback.index):
        raise ValueError("Versioned weight and fallback date indexes differ")
    return weights, fallback, schedule


def _fixed_masks(weights: np.ndarray, alpha_long: float, alpha_short: float) -> np.ndarray:
    masks = np.where(weights > 0.0, alpha_long, np.where(weights < 0.0, alpha_short, 0.0))
    if len(masks):
        masks[-1] = 0.0
    return masks.astype(float)


def _replay(
    *,
    weights: np.ndarray,
    target_returns: np.ndarray,
    gap_returns: np.ndarray,
    dates: pd.DatetimeIndex,
    alpha_masks: np.ndarray,
    calendar_days: np.ndarray,
    slip: float,
    financing_annual: float,
    borrow_annual: float,
    reverse_fee_bps: float,
    side_leverage: float,
    fallback: pd.Series,
    policy_name: str,
) -> dict[str, Any]:
    pnl = simulate_daily_pnl(
        weights=weights,
        target_returns=target_returns,
        gap_returns=gap_returns,
        sim_dates=dates,
        slip=slip,
        financing_daily=financing_annual / 365.0,
        borrow_daily=borrow_annual / 365.0,
        reverse_daily=reverse_fee_bps / 10_000.0,
        alpha_long=0.0,
        alpha_short=0.0,
        alpha_masks=alpha_masks,
        calendar_days=calendar_days,
        side_leverage=side_leverage,
    )
    frame = pd.DataFrame(
        {
            "gross": pnl["gross_returns"],
            "net": pnl["net_returns"],
            "cost": pnl["costs"],
            "slippage": pnl["slip_costs"],
            "financing": pnl["financing_costs"],
            "borrow": pnl["borrow_costs"],
            "reverse": pnl["reverse_costs"],
            "overnight": pnl["overnight_returns"],
            "turnover": pnl["turnover"],
            "execution_volume": pnl["execution_volume"],
            "gross_exposure": pnl["gross_exps"],
            "fallback": fallback.to_numpy(dtype=bool),
        },
        index=dates,
    )
    if not np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all():
        raise ValueError(f"{policy_name} produced a non-finite daily PnL series")
    if not np.allclose(frame["gross"] - frame["cost"], frame["net"], atol=1e-12):
        raise ValueError(f"{policy_name} violates gross - costs = net")
    if not np.allclose(
        frame["cost"], frame[["slippage", "financing", "borrow", "reverse"]].sum(axis=1), atol=1e-12
    ):
        raise ValueError(f"{policy_name} cost components do not reconcile")
    if not np.allclose(frame["execution_volume"], 2.0 * frame["turnover"], atol=1e-12):
        raise ValueError(f"{policy_name} execution volume and turnover do not reconcile")
    frame["policy"] = policy_name
    frame["carry_alpha_mean"] = alpha_masks.mean(axis=1)
    return {"daily": frame, "alpha_masks": alpha_masks}


def _stats(returns: pd.Series) -> dict[str, float | int]:
    values = returns.to_numpy(dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("metrics require finite returns for every included evaluation day")
    summary = calculate_metrics(
        returns,
        spec=MetricsSpec(annualization_periods=ANNUALIZATION, include_flat_days=True),
    )
    return {
        "n": int(len(values)),
        "net_sharpe": float(summary["Sharpe"]),
        "annual_return": float(summary["AR"]),
        "annual_volatility": float(summary["RISK"]),
        "max_drawdown": float(summary["MDD"]),
        "net_pnl_sum": float(values.sum()),
        "net_pnl_compounded": float(np.prod(1.0 + values) - 1.0),
    }


def _gross_stats(returns: pd.Series) -> dict[str, float]:
    summary = calculate_metrics(
        returns,
        spec=MetricsSpec(annualization_periods=ANNUALIZATION, include_flat_days=True),
    )
    values = returns.to_numpy(dtype=float)
    return {
        "gross_sharpe": float(summary["Sharpe"]),
        "gross_pnl_sum": float(values.sum()),
        "gross_pnl_compounded": float(np.prod(1.0 + values) - 1.0),
    }


def _summary_row(
    *,
    name: str,
    replay: dict[str, Any],
    alpha_masks: np.ndarray,
    weights: np.ndarray,
    dates: pd.DatetimeIndex,
    oos_start_index: int,
) -> dict[str, Any]:
    daily = replay["daily"].iloc[oos_start_index:]
    returns = daily["net"]
    row: dict[str, Any] = {"policy": name, **_stats(returns), **_gross_stats(daily["gross"])}
    row.update(
        {
            "slippage_cost": float(daily["slippage"].sum()),
            "financing_cost": float(daily["financing"].sum()),
            "borrow_cost": float(daily["borrow"].sum()),
            "reverse_cost": float(daily["reverse"].sum()),
            "total_cost": float(daily["cost"].sum()),
            "carry_cost": float(daily[["financing", "borrow", "reverse"]].to_numpy().sum()),
            "mean_daily_turnover": float(daily["turnover"].mean()),
            "sum_execution_volume": float(daily["execution_volume"].sum()),
            "mean_gross_exposure": float(daily["gross_exposure"].mean()),
            "fallback_rate": float(daily["fallback"].mean()),
            "mean_carry_alpha_long": _weighted_alpha_mean(
                alpha_masks[oos_start_index:], weights[oos_start_index:], positive=True
            ),
            "mean_carry_alpha_short": _weighted_alpha_mean(
                alpha_masks[oos_start_index:], weights[oos_start_index:], positive=False
            ),
            **realized_inventory_reuse(
                weights=weights[oos_start_index:], alpha_masks=alpha_masks[oos_start_index:]
            ),
        }
    )
    row["gross_minus_cost_error"] = float(
        (daily["gross"] - daily["cost"] - daily["net"]).abs().max()
    )
    return row


def _weighted_alpha_mean(alpha: np.ndarray, weights: np.ndarray, *, positive: bool) -> float:
    selected = weights > 0 if positive else weights < 0
    exposures = np.abs(weights[selected])
    if not len(exposures) or exposures.sum() <= 0:
        return 0.0
    return float(np.average(alpha[selected], weights=exposures))


def _paired_block_bootstrap(
    delta: np.ndarray,
    *,
    block_sessions: int,
    resamples: int,
    seed: int,
) -> dict[str, float | int]:
    delta = np.asarray(delta, dtype=float)
    if delta.ndim != 1 or not len(delta) or not np.isfinite(delta).all():
        raise ValueError("paired bootstrap requires a non-empty finite daily delta")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(len(delta) / block_sessions))
    sampled_means = np.empty(resamples, dtype=float)
    for sample in range(resamples):
        starts = rng.integers(0, len(delta), size=n_blocks)
        indices = np.concatenate(
            [(start + np.arange(block_sessions)) % len(delta) for start in starts]
        )[: len(delta)]
        sampled_means[sample] = float(delta[indices].mean())
    lower, upper = np.quantile(sampled_means, [0.025, 0.975])
    return {
        "block_sessions": block_sessions,
        "resamples": resamples,
        "seed": seed,
        "mean_daily_net_delta": float(delta.mean()),
        "ci95_mean_daily_net_delta": [float(lower), float(upper)],
    }


def _segment_rows(
    *,
    replay: dict[str, Any],
    weights: np.ndarray,
    alpha_masks: np.ndarray,
    target_returns: np.ndarray,
    gap_returns: np.ndarray,
    dates: pd.DatetimeIndex,
    calendar_days: np.ndarray,
    oos_start_index: int,
    slip: float,
    financing_annual: float,
    borrow_annual: float,
    reverse_fee_bps: float,
    side_leverage: float,
) -> list[dict[str, Any]]:
    daily = replay["daily"]
    output: list[dict[str, Any]] = []
    if len(dates) < 2:
        return output
    following = weights[1:]
    group_masks: dict[str, np.ndarray] = {
        "calendar_gap_1d": calendar_days[1:] == 1,
        "weekend_or_short_holiday_2_3d": (calendar_days[1:] >= 2) & (calendar_days[1:] <= 3),
        "extended_holiday_4d_plus": calendar_days[1:] >= 4,
    }
    eligible_positions = np.arange(len(dates) - 1) >= oos_start_index
    attributed = attribute_inventory_transitions(
        weights=weights,
        target_returns=target_returns,
        gap_returns=gap_returns,
        alpha_masks=alpha_masks,
        sim_dates=dates,
        calendar_days=calendar_days,
        slip=slip,
        financing_daily=financing_annual / 365.0,
        borrow_daily=borrow_annual / 365.0,
        reverse_daily=reverse_fee_bps / 10_000.0,
        side_leverage=side_leverage,
    )
    for label, raw_mask in group_masks.items():
        selected_positions = np.flatnonzero(raw_mask & eligible_positions)
        if not len(selected_positions):
            continue
        part = daily.iloc[selected_positions]
        carried = alpha_masks[selected_positions] * weights[selected_positions]
        reused = np.where(
            np.sign(carried) == np.sign(following[selected_positions]),
            np.minimum(np.abs(carried), np.abs(following[selected_positions])),
            0.0,
        )
        carried_sum = float(np.abs(carried).sum())
        selected_dates = dates[selected_positions]
        next_open_slippage_delta = float(
            attributed.loc[
                attributed["date"].isin(selected_dates), "next_open_slippage_delta_vs_flat"
            ].sum()
        )
        output.append(
            {
                "segment": label,
                "unit": "portfolio_sessions",
                "n": int(len(selected_positions)),
                "gross_pnl_sum": float(part["gross"].sum()),
                "net_pnl_sum": float(part["net"].sum()),
                "mean_net": float(part["net"].mean()),
                "overnight_pnl_sum": float(part["overnight"].sum()),
                "slippage_cost": float(part["slippage"].sum()),
                "next_open_slippage_delta_vs_flat": next_open_slippage_delta,
                "carry_cost": float(part[["financing", "borrow", "reverse"]].to_numpy().sum()),
                "reuse_ratio": float(reused.sum() / carried_sum) if carried_sum else 0.0,
            }
        )
    if dates[-1] >= dates[oos_start_index]:
        terminal = daily.iloc[[-1]]
        output.append(
            {
                "segment": "terminal_final_close",
                "unit": "portfolio_sessions",
                "n": 1,
                "gross_pnl_sum": float(terminal["gross"].sum()),
                "net_pnl_sum": float(terminal["net"].sum()),
                "mean_net": float(terminal["net"].mean()),
                "overnight_pnl_sum": float(terminal["overnight"].sum()),
                "slippage_cost": float(terminal["slippage"].sum()),
                "next_open_slippage_delta_vs_flat": 0.0,
                "carry_cost": float(terminal[["financing", "borrow", "reverse"]].to_numpy().sum()),
                "reuse_ratio": 0.0,
            }
        )

    eligible_dates = dates[np.flatnonzero(eligible_positions)]
    active_transitions = attributed.loc[
        attributed["date"].isin(eligible_dates) & (attributed["current_weight"] != 0.0)
    ]
    transition_labels = {
        "next_signal_reversal": "next_signal_reversal",
        "next_signal_same_direction": "next_signal_same_direction",
        "next_signal_flat": "next_signal_flat",
    }
    for label, transition_class in transition_labels.items():
        part = active_transitions.loc[active_transitions["transition_class"] == transition_class]
        if part.empty:
            continue
        carried_sum = float(part["carried_notional"].sum())
        output.append(
            {
                "segment": label,
                "unit": "ticker_session_transitions",
                "n": int(len(part)),
                "gross_pnl_sum": float(part["gross_pnl"].sum()),
                "net_pnl_sum": float(part["net_pnl"].sum()),
                "mean_net": float(part["net_pnl"].mean()),
                "overnight_pnl_sum": float(part["overnight_pnl"].sum()),
                "slippage_cost": float(part["slippage_cost"].sum()),
                "next_open_slippage_delta_vs_flat": float(
                    part["next_open_slippage_delta_vs_flat"].sum()
                ),
                "carry_cost": float(
                    part[["financing_cost", "borrow_cost", "reverse_cost"]].to_numpy().sum()
                ),
                "reuse_ratio": (
                    float(part["reused_notional"].sum() / carried_sum) if carried_sum else 0.0
                ),
            }
        )
    return output


def _policy_config(raw: dict[str, Any], *, lookback: int | None = None, reversal_multiplier: float | None = None) -> AdaptiveCarryConfig:
    model = raw["model"]
    transition_prior = model["transition_dirichlet_prior"]
    overlap_prior = model["overlap_beta_prior"]
    return AdaptiveCarryConfig(
        lookback_sessions=int(lookback if lookback is not None else model["lookback_sessions"]),
        transition_prior_same=float(transition_prior["same_direction"]),
        transition_prior_reverse=float(transition_prior["reversal"]),
        transition_prior_flat=float(transition_prior["flat"]),
        overlap_prior_alpha=float(overlap_prior["alpha"]),
        overlap_prior_beta=float(overlap_prior["beta"]),
        reversal_exit_cost_multiplier=float(
            reversal_multiplier if reversal_multiplier is not None else model["reversal_exit_cost_multiplier"]
        ),
    )


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    research_config = _read_yaml(CONFIG_PATH)
    app_config = load_config_from_yaml(PRODUCTION_CONFIG_PATH)
    cost_config = app_config.v2.costs
    alpha_long = float(cost_config.overnight_alpha_long)
    alpha_short = float(cost_config.overnight_alpha_short)
    if (alpha_long, alpha_short) != (0.75, 0.50):
        raise ValueError(
            "The research baseline must resolve to overnight_alpha_long=0.75 and short=0.50; "
            f"got {(alpha_long, alpha_short)}"
        )

    df_exec = load_df_exec_from_local_cache(max_stale_bdays=None)
    gap_store = ROOT / "var" / "live" / "pipeline_data" / "gap_adjusted_distribution" / "gap_store.sqlite"
    overlay_root = ROOT / app_config.v2.ml_overlay_model_dir
    if not gap_store.exists():
        raise FileNotFoundError(f"Configured research input does not exist: {gap_store}")
    if not (overlay_root / "HISTORY.json").exists():
        raise FileNotFoundError(f"Resolved production overlay history is missing: {overlay_root}")

    bt = research_config["backtest"]
    weights_frame, fallback, overlay_schedule = _versioned_model_weights(
        app_config=app_config,
        df_exec=df_exec,
        gap_store=gap_store,
        start_date=str(bt["start_date"]),
        end_date=str(bt["end_date"]),
    )
    weights_frame = weights_frame.reindex(columns=JP_TICKERS).astype(float)
    dates = pd.DatetimeIndex(weights_frame.index).normalize()
    if not dates.is_monotonic_increasing or dates.has_duplicates:
        raise ValueError("V2 backtest weights have an invalid date index")
    if not np.isfinite(weights_frame.to_numpy()).all():
        raise ValueError("V2 backtest weights contain non-finite entries")
    weights = weights_frame.to_numpy(dtype=float)
    fallback = fallback.reindex(dates).astype(bool)
    if not fallback.index.equals(dates):
        raise ValueError("fallback flags do not align with weights")
    if not np.any(np.abs(weights).sum(axis=1) > 0):
        raise ValueError("All generated V2 weights are flat; carry comparison is invalid")

    sim_dates, start_idx, end_idx = BacktestEngine._resolve_sim_dates(
        df_exec, str(bt["start_date"]), str(bt["end_date"]), 0
    )
    expected_dates = pd.DatetimeIndex(sim_dates[start_idx : end_idx + 1]).normalize()
    if not dates.equals(expected_dates):
        raise ValueError("Backtest weight dates differ from the execution-data date slice")
    open_910_returns = build_open_910_returns(df_exec, JP_TICKERS)
    target_returns, gap_returns, morning_returns = BacktestEngine._compute_price_intervals(
        df_exec, sim_dates, pd.DatetimeIndex(sim_dates[start_idx : end_idx + 1]), open_910_returns
    )
    # Research uses one close-to-entry mark, matched to the intraday target.
    gap_returns = (1.0 + gap_returns) * (1.0 + morning_returns) - 1.0
    calendar_days = _calendar_days(dates)

    slip = float(cost_config.slippage_bps_per_side) / 10_000.0
    financing_annual = float(cost_config.buy_interest_annual)
    borrow_annual = float(cost_config.borrow_fee_annual)
    reverse_fee_bps = float(cost_config.reverse_fee_bps)
    side_leverage = float(cost_config.side_leverage)
    baseline_masks = _fixed_masks(weights, alpha_long, alpha_short)
    main_config = _policy_config(research_config)
    adaptive_masks, diagnostics = adaptive_carry_masks(
        weights=weights,
        gap_returns=gap_returns,
        sim_dates=dates,
        slip=slip,
        financing_annual=financing_annual,
        borrow_annual=borrow_annual,
        reverse_fee_bps=reverse_fee_bps,
        config=main_config,
    )
    warmup = main_config.lookback_sessions + 1
    if len(dates) <= warmup:
        raise ValueError(
            f"Need more than {warmup} sessions for the predeclared OOS warm-up; got {len(dates)}"
        )
    oos_start_index = warmup
    oos_dates = dates[oos_start_index:]
    if len(oos_dates) < 100:
        raise ValueError(f"Only {len(oos_dates)} OOS sessions remain after warm-up")

    policy_masks: dict[str, np.ndarray] = {"fixed_alpha_0.75_0.50": baseline_masks}
    policy_masks["adaptive_252"] = adaptive_masks
    model_config = research_config["model"]
    for multiplier in model_config["sensitivity"]["lookback_multipliers"]:
        lookback = int(round(main_config.lookback_sessions * float(multiplier)))
        mask, _ = adaptive_carry_masks(
            weights=weights,
            gap_returns=gap_returns,
            sim_dates=dates,
            slip=slip,
            financing_annual=financing_annual,
            borrow_annual=borrow_annual,
            reverse_fee_bps=reverse_fee_bps,
            config=_policy_config(research_config, lookback=lookback),
        )
        policy_masks[f"adaptive_lookback_{lookback}"] = mask
    for multiplier in model_config["sensitivity"]["reversal_exit_cost_multipliers"]:
        value = float(multiplier)
        mask, _ = adaptive_carry_masks(
            weights=weights,
            gap_returns=gap_returns,
            sim_dates=dates,
            slip=slip,
            financing_annual=financing_annual,
            borrow_annual=borrow_annual,
            reverse_fee_bps=reverse_fee_bps,
            config=_policy_config(research_config, reversal_multiplier=value),
        )
        policy_masks[f"adaptive_reversal_cost_{value:.1f}x"] = mask

    replay_by_policy: dict[str, dict[str, Any]] = {}
    summary_rows: list[dict[str, Any]] = []
    daily_outputs: list[pd.DataFrame] = []
    for name, masks in policy_masks.items():
        replay = _replay(
            weights=weights,
            target_returns=target_returns,
            gap_returns=gap_returns,
            dates=dates,
            alpha_masks=masks,
            calendar_days=calendar_days,
            slip=slip,
            financing_annual=financing_annual,
            borrow_annual=borrow_annual,
            reverse_fee_bps=reverse_fee_bps,
            side_leverage=side_leverage,
            fallback=fallback,
            policy_name=name,
        )
        replay_by_policy[name] = replay
        daily_outputs.append(replay["daily"])
        summary_rows.append(
            _summary_row(
                name=name,
                replay=replay,
                alpha_masks=masks,
                weights=weights,
                dates=dates,
                oos_start_index=oos_start_index,
            )
        )
    summary_frame = pd.DataFrame(summary_rows)

    stress_rows: list[dict[str, Any]] = []
    for stress_borrow in research_config["cost_stress"]["high_short_borrow_annual"]:
        stress_borrow = float(stress_borrow)
        stress_masks, _ = adaptive_carry_masks(
            weights=weights,
            gap_returns=gap_returns,
            sim_dates=dates,
            slip=slip,
            financing_annual=financing_annual,
            borrow_annual=stress_borrow,
            reverse_fee_bps=reverse_fee_bps,
            config=main_config,
        )
        for name, masks in (
            ("fixed_alpha_0.75_0.50", baseline_masks),
            ("adaptive_252", stress_masks),
        ):
            replay = _replay(
                weights=weights,
                target_returns=target_returns,
                gap_returns=gap_returns,
                dates=dates,
                alpha_masks=masks,
                calendar_days=calendar_days,
                slip=slip,
                financing_annual=financing_annual,
                borrow_annual=stress_borrow,
                reverse_fee_bps=reverse_fee_bps,
                side_leverage=side_leverage,
                fallback=fallback,
                policy_name=f"{name}_borrow_{stress_borrow:.0%}",
            )
            row = _summary_row(
                name=name,
                replay=replay,
                alpha_masks=masks,
                weights=weights,
                dates=dates,
                oos_start_index=oos_start_index,
            )
            row["short_borrow_annual"] = stress_borrow
            stress_rows.append(row)
            daily_outputs.append(replay["daily"])
    stress_frame = pd.DataFrame(stress_rows)

    segment_rows: list[dict[str, Any]] = []
    transition_outputs: list[pd.DataFrame] = []
    for name in ("fixed_alpha_0.75_0.50", "adaptive_252"):
        transitions = attribute_inventory_transitions(
            weights=weights,
            target_returns=target_returns,
            gap_returns=gap_returns,
            alpha_masks=policy_masks[name],
            sim_dates=dates,
            calendar_days=calendar_days,
            slip=slip,
            financing_daily=financing_annual / 365.0,
            borrow_daily=borrow_annual / 365.0,
            reverse_daily=reverse_fee_bps / 10_000.0,
            side_leverage=side_leverage,
        )
        transitions = transitions.loc[transitions["date"] >= oos_dates[0]].copy()
        transitions["ticker"] = transitions["ticker_index"].map(
            {index: ticker for index, ticker in enumerate(JP_TICKERS)}
        )
        transitions["policy"] = name
        transition_outputs.append(transitions)
        rows = _segment_rows(
            replay=replay_by_policy[name],
            weights=weights,
            alpha_masks=policy_masks[name],
            target_returns=target_returns,
            gap_returns=gap_returns,
            dates=dates,
            calendar_days=calendar_days,
            oos_start_index=oos_start_index,
            slip=slip,
            financing_annual=financing_annual,
            borrow_annual=borrow_annual,
            reverse_fee_bps=reverse_fee_bps,
            side_leverage=side_leverage,
        )
        for row in rows:
            segment_rows.append({"policy": name, **row})
    segment_frame = pd.DataFrame(segment_rows)

    baseline_daily = replay_by_policy["fixed_alpha_0.75_0.50"]["daily"].iloc[oos_start_index:]
    adaptive_daily = replay_by_policy["adaptive_252"]["daily"].iloc[oos_start_index:]
    bootstrap_cfg = bt["block_bootstrap"]
    bootstrap = _paired_block_bootstrap(
        adaptive_daily["net"].to_numpy() - baseline_daily["net"].to_numpy(),
        block_sessions=int(bootstrap_cfg["block_sessions"]),
        resamples=int(bootstrap_cfg["resamples"]),
        seed=int(bootstrap_cfg["seed"]),
    )

    diagnostic_frame = diagnostics.copy()
    diagnostic_frame["ticker"] = diagnostic_frame["ticker_index"].map(
        {index: ticker for index, ticker in enumerate(JP_TICKERS)}
    )
    diagnostic_frame = diagnostic_frame.loc[diagnostic_frame["date"] >= oos_dates[0]]
    ticker_rows: list[dict[str, Any]] = []
    for j, ticker in enumerate(JP_TICKERS):
        ticker_rows.append(
            {
                "ticker": ticker,
                "observations": int(np.count_nonzero(weights[oos_start_index:, j])),
                "mean_abs_weight": float(np.abs(weights[oos_start_index:, j]).mean()),
                "adaptive_alpha_long": _weighted_alpha_mean(
                    adaptive_masks[oos_start_index:, [j]], weights[oos_start_index:, [j]], positive=True
                ),
                "adaptive_alpha_short": _weighted_alpha_mean(
                    adaptive_masks[oos_start_index:, [j]], weights[oos_start_index:, [j]], positive=False
                ),
                "mean_estimated_same_probability": float(
                    diagnostic_frame.loc[
                        (diagnostic_frame["ticker"] == ticker)
                        & (diagnostic_frame["same_probability"].notna()),
                        "same_probability",
                    ].mean()
                ),
            }
        )
    ticker_frame = pd.DataFrame(ticker_rows)

    weights_frame.to_csv(OUTPUT_DIR / "daily_weights.csv", float_format="%.12g")
    pd.DataFrame(
        {name: masks.mean(axis=1) for name, masks in policy_masks.items()}, index=dates
    ).to_csv(OUTPUT_DIR / "daily_mean_carry_alpha.csv", float_format="%.12g")
    pd.concat(daily_outputs).to_csv(OUTPUT_DIR / "daily_policy_replay.csv", float_format="%.12g")
    diagnostic_frame.to_csv(OUTPUT_DIR / "ticker_carry_diagnostics.csv", index=False, float_format="%.12g")
    summary_frame.to_csv(OUTPUT_DIR / "policy_summary.csv", index=False, float_format="%.12g")
    stress_frame.to_csv(OUTPUT_DIR / "borrow_stress_summary.csv", index=False, float_format="%.12g")
    segment_frame.to_csv(OUTPUT_DIR / "calendar_and_reversal_segments.csv", index=False, float_format="%.12g")
    ticker_frame.to_csv(OUTPUT_DIR / "ticker_summary.csv", index=False, float_format="%.12g")
    pd.concat(transition_outputs, ignore_index=True).to_csv(
        OUTPUT_DIR / "ticker_transition_attribution.csv", index=False, float_format="%.12g"
    )

    main_row = summary_frame.loc[summary_frame["policy"] == "adaptive_252"].iloc[0].to_dict()
    base_row = summary_frame.loc[summary_frame["policy"] == "fixed_alpha_0.75_0.50"].iloc[0].to_dict()
    trial_sharpes = summary_frame["net_sharpe"].astype(float).to_numpy()
    trial_variance_annual = float(np.var(trial_sharpes, ddof=1)) if len(trial_sharpes) > 1 else 0.0
    historical_known_variants = 3
    current_window_variants = len(policy_masks)
    # Count every prior candidate and every current comparison, including the
    # repeated baseline, as a conservative upper bound on the trial family.
    estimated_trial_count = historical_known_variants + current_window_variants
    current_trial_sharpes = trial_sharpes.tolist()
    gap_coverage = _read_gap_store_coverage(gap_store)
    overlay_metadata = overlay_root / "PROMOTION.json"
    promotion = json.loads(overlay_metadata.read_text(encoding="utf-8")) if overlay_metadata.exists() else {}
    payload = {
        "created_date": "2026-10-08",
        "hypothesis": "Ticker-specific carry sized from PIT continuation, overlap, overnight return and cost estimates can improve OOS net results over fixed alpha.",
        "production_changed": False,
        "production_config": str(PRODUCTION_CONFIG_PATH.relative_to(ROOT)),
        "production_config_sha256": _sha256(PRODUCTION_CONFIG_PATH),
        "research_config": str(CONFIG_PATH.relative_to(ROOT)),
        "research_config_sha256": _sha256(CONFIG_PATH),
        "overlay_model_dir": str(overlay_root.relative_to(ROOT)),
        "overlay_schedule": overlay_schedule,
        "overlay_status_counts": dict(
            sum(
                (Counter(segment["overlay_status_counts"]) for segment in overlay_schedule),
                Counter(),
            )
        ),
        "overlay_skip_reason_counts": dict(
            sum(
                (Counter(segment["overlay_skip_reason_counts"]) for segment in overlay_schedule),
                Counter(),
            )
        ),
        "overlay_applied_rows": int(
            sum(
                count
                for segment in overlay_schedule
                for status, count in segment["overlay_status_counts"].items()
                if status == "applied"
            )
        ),
        "overlay_promotion_sha256": _sha256(overlay_metadata) if overlay_metadata.exists() else None,
        "overlay_promotion_gate_status": promotion.get("gate_status", {}),
        "overlay_operator_override": promotion.get("operator_override"),
        "gap_store": str(gap_store.relative_to(ROOT)),
        "gap_store_matrices": gap_coverage["dates"],
        "gap_store_trade_date_range": [gap_coverage["start"], gap_coverage["end"]],
        "source_data_start": str(pd.Timestamp(df_exec.index.min()).date()),
        "source_data_end": str(pd.Timestamp(df_exec.index.max()).date()),
        "backtest_start": str(dates[0].date()),
        "backtest_end": str(dates[-1].date()),
        "oos_start": str(oos_dates[0].date()),
        "oos_end": str(oos_dates[-1].date()),
        "n_backtest_sessions": int(len(dates)),
        "n_oos_sessions": int(len(oos_dates)),
        "fallback_rate_oos": float(fallback.iloc[oos_start_index:].mean()),
        "nonflat_weight_days_oos": int(np.count_nonzero(np.abs(weights[oos_start_index:]).sum(axis=1))),
        "initial_inventory": "flat at first backtest entry",
        "terminal_policy": "liquidate all remaining inventory at final close; last alpha mask is zero",
        "carry_accounting_contract": "previous close to current 09:10 mark, attributed to outgoing trade date",
        "inventory_state": "one continuous replay across every date; no resets",
        "execution_volume_contract": "side_leverage * (opening inventory flow + closing inventory flow); turnover = volume / 2",
        "side_leverage": side_leverage,
        "slippage_bps_one_way": slip * 10_000,
        "buy_interest_annual": financing_annual,
        "borrow_fee_annual_baseline": borrow_annual,
        "reverse_fee_bps_per_calendar_day": reverse_fee_bps,
        "baseline_alpha_long": alpha_long,
        "baseline_alpha_short": alpha_short,
        "adaptive_config": main_config.__dict__,
        "summary": summary_frame.to_dict(orient="records"),
        "borrow_stress": stress_frame.to_dict(orient="records"),
        "calendar_and_reversal_segments": segment_frame.to_dict(orient="records"),
        "paired_block_bootstrap": bootstrap,
        "trial_count_estimate": estimated_trial_count,
        "trial_count_basis": {
            "previous_fixed_alpha_variants_found_in_existing_result_files": historical_known_variants,
            "current_window_variants_including_baseline": current_window_variants,
            "repeated_baseline_counted_twice_conservatively": True,
            "historical_same_window_sharpe_series_available": False,
        },
        "dsr_trial_sharpes_current_window": current_trial_sharpes,
        "dsr_trial_sharpe_variance_annual_current_window": trial_variance_annual,
        "decision": "PENDING",
        "decision_reason": "Single short OOS interval, model-based slippage, and no actual fill or ticker-level borrow records; the experiment does not meet the predeclared adoption evidence gate.",
    }
    metrics_path = OUTPUT_DIR / "metrics.json"
    metrics_path.write_text(
        json.dumps(_safe_json(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    annual_sharpes = ", ".join(f"{value:.4f}" for value in current_trial_sharpes)
    report = [
        "# 銘柄別オーバーナイト在庫 sizing 研究\n\n",
        "## 仮説と事前条件\n\n",
        "固定alphaの持越し在庫を、銘柄ごとの翌朝継続確率・継続時に再利用できる期待量・再売買回避費用から計算し、予想overnight PnLと保有費用、反転/flat時の手仕舞い費用を差し引いて決める。日次の同じ本番ウェイトを両方式に使い、carry配分だけを比較した。\n\n",
        f"- Production baseline: `overnight_alpha_long={alpha_long:.2f}`, `short={alpha_short:.2f}`。本番設定は変更していない。\n",
        f"- 期間: モデル生成 {dates[0].date()}–{dates[-1].date()}、評価 {oos_dates[0].date()}–{oos_dates[-1].date()}（{len(oos_dates)}日）。評価開始までは252本の過去遷移を使うwarm-up。年率化245日、全評価日を含む。\n",
        f"- 解決済み本番 side leverage={side_leverage:.2f}、片道slippage={slip*10000:.1f}bp、long financing={financing_annual:.2%}/年、short borrow={borrow_annual:.2%}/年、reverse={reverse_fee_bps:.1f}bp/暦日。\n",
        f"- Overlay: `{overlay_root.relative_to(ROOT)}` の `HISTORY.json` に従い、各期間で学習cutoffが未来にならないartifactを使用。gap storeは{payload['gap_store_trade_date_range'][0]}–{payload['gap_store_trade_date_range'][1]}の{gap_coverage['dates']}日分で、それ以外は有効設定のon-demand BLPX経路を使った。評価期間fallback率={payload['fallback_rate_oos']:.2%}。\n",
        f"- Overlay適用状態: {payload['overlay_status_counts']}、適用行={payload['overlay_applied_rows']}。skip理由: {payload['overlay_skip_reason_counts']}。ADR入力不足なら、ウェイトはML overlay適用前のV2値となる。\n",
        "- 実行ログではmacro価格を読み込めず、412/412日でmacro data insufficient (0 rows)。この比較は当該macro調整も反映していない。\n",
        "- 現行overlayの promotion record にはoperator overrideと未通過gateが記録されている。これはcarryの研究replayであり、overlayや本番リスク停止の受入/解除を意味しない。\n",
        "- inventoryは開始時flatから全日連続でreplay。終端は最終引け全清算、terminal alphaを0にして売買量と片道費用を計上した。execution volumeは実効opening/closing inventory flow合計、turnoverはその1/2。\n\n",
        "### 時点制約\n\n",
        "各日のcarry決定は、当日closeまでに分かる現在weightの符号、当日09:10までに確定した過去gap、そして決定日前日までに終わった銘柄別weight遷移だけを使う。次営業日のweight・target return・gapはalpha計算関数へ渡していない。次営業日weightは再利用率と反転分類の事後評価だけに使う。\n\n",
        "同方向時の費用便益は `P(same) × E[overlap] × 2 × one-way slippage`。これへ方向付き過去平均gap returnを加え、暦日financing/borrow/reverse、反転時の持越し在庫exit slip、flat時のexit slipを引いた値をround-trip slipで割り、0–1へclipする。実際のPnLにはovernight markとcore ledgerの売買費用を一度だけ計上する。\n\n",
        "## OOS結果\n\n",
        "すべてのPnL値はポートフォリオreturn fraction。gross/net PnLは日次単純returnの合計と複利値を並記。MDDは初期wealth=1から計算。\n\n",
        "| Policy | n | Net Sharpe | MDD | Gross PnL sum / comp. | Net PnL sum / comp. | Slip cost | Carry cost | Turnover/day | Reused inventory | Fallback |\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for row in (base_row, main_row):
        report.append(
            f"| {row['policy']} | {row['n']} | {row['net_sharpe']:.3f} | {row['max_drawdown']:.2%} | "
            f"{row['gross_pnl_sum']:.4f} / {row['gross_pnl_compounded']:.2%} | "
            f"{row['net_pnl_sum']:.4f} / {row['net_pnl_compounded']:.2%} | "
            f"{row['slippage_cost']:.4f} | {row['carry_cost']:.4f} | {row['mean_daily_turnover']:.4f} | "
            f"{row['reuse_ratio']:.2%} | {row['fallback_rate']:.2%} |\n"
        )
    report.extend(
        [
            "\nコストのreturn fraction内訳:\n\n",
            "| Policy | Slippage | Financing | Borrow | Reverse | Total costs | Overnight PnL |\n",
            "|---|---:|---:|---:|---:|---:|---:|\n",
        ]
    )
    for row in (base_row, main_row):
        daily = replay_by_policy[row["policy"]]["daily"].iloc[oos_start_index:]
        report.append(
            f"| {row['policy']} | {row['slippage_cost']:.5f} | {row['financing_cost']:.5f} | "
            f"{row['borrow_cost']:.5f} | {row['reverse_cost']:.5f} | {row['total_cost']:.5f} | "
            f"{daily['overnight'].sum():.5f} |\n"
        )
    report.extend(
        [
            "\n### 事前パラメータ感度\n\n",
            "| Policy | Net Sharpe | MDD | Net PnL sum | Slip | Carry cost | Reuse ratio | 平均alpha(long/short) |\n",
            "|---|---:|---:|---:|---:|---:|---:|---:|\n",
        ]
    )
    for row in summary_rows:
        report.append(
            f"| {row['policy']} | {row['net_sharpe']:.3f} | {row['max_drawdown']:.2%} | {row['net_pnl_sum']:.4f} | "
            f"{row['slippage_cost']:.4f} | {row['carry_cost']:.4f} | {row['reuse_ratio']:.2%} | "
            f"{row['mean_carry_alpha_long']:.3f}/{row['mean_carry_alpha_short']:.3f} |\n"
        )
    report.extend(
        [
            "\n### Short borrow stress\n\n",
            "Borrow fee is a hypothetical uniform annual rate on short carried exposure; historical ticker-level borrow/reverse charges were unavailable.\n\n",
            "| Borrow annual | Policy | Net Sharpe | MDD | Net PnL sum | Borrow cost | Reverse cost | Avg short alpha | Reuse ratio |\n",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|\n",
        ]
    )
    for row in stress_rows:
        report.append(
            f"| {row['short_borrow_annual']:.0%} | {row['policy']} | {row['net_sharpe']:.3f} | "
            f"{row['max_drawdown']:.2%} | {row['net_pnl_sum']:.4f} | {row['borrow_cost']:.5f} | "
            f"{row['reverse_cost']:.5f} | {row['mean_carry_alpha_short']:.3f} | {row['reuse_ratio']:.2%} |\n"
        )
    report.extend(
        [
            "\n### 連休・翌朝反転の事後区分\n\n",
            "区分は次営業日の実現date gap・実現weightを使った事後診断で、carry decisionには使っていない。calendar区分はportfolio session単位、same/reversal/flat区分は銘柄×session単位の加算可能な台帳。日次PnLはoutgoing trade dateへ帰属し、overnight markを含む。`Next-open slip delta vs flat` は次営業日寄りの実slippageを、carry在庫なしで同じtargetを売買した反実仮想と比べた増減費用（負値は節約）。\n\n",
            "| Policy | Segment | Unit | n | Gross sum | Net sum | Mean net / unit | Overnight | Actual slip | Next-open slip delta vs flat | Carry cost | Reuse ratio |\n",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
        ]
    )
    for row in segment_rows:
        report.append(
            f"| {row['policy']} | {row['segment']} | {row['unit']} | {row['n']} | {row['gross_pnl_sum']:.4f} | "
            f"{row['net_pnl_sum']:.4f} | {row['mean_net']:.5f} | {row['overnight_pnl_sum']:.5f} | "
            f"{row['slippage_cost']:.5f} | {row['next_open_slippage_delta_vs_flat']:.5f} | "
            f"{row['carry_cost']:.5f} | {row['reuse_ratio']:.2%} |\n"
        )
    report.extend(
        [
            "\n### 比較の不確実性と判定\n\n",
            f"Adaptive minus fixed-alpha paired daily net delta: {bootstrap['mean_daily_net_delta']:.6f}; "
            f"20-session circular block bootstrap 95% CI {bootstrap['ci95_mean_daily_net_delta']} "
            f"({bootstrap['resamples']} resamples, seed {bootstrap['seed']}). CI is an uncertainty interval, not a probability of strategy superiority.\n\n",
            f"DSR trial count estimate: {estimated_trial_count} (3 prior fixed-alpha variants found in existing result files plus {current_window_variants} current-window policies; the repeated baseline is counted twice as a conservative upper bound). Same-window annual Sharpe values used for the cross-trial variance proxy: `{annual_sharpes}`. Earlier historical candidates do not have matching-date Sharpe series, so DSR remains approximate; the exact trial family may be larger.\n\n",
            f"判定: **PENDING**。OOSは{len(oos_dates)}日で単一区間、実約定価格・ticker別borrow/逆日歩の実費がなく、carry accountingも数量・cash driftではなくweight-based simulated notionalであるため本番採用判断の証拠ゲートに達していない。\n\n",
            "再現コマンド:\n\n",
            "```sh\n",
            "timeout -k 10s 3600s .venv/bin/python src/research/scripts/experiments/experiment_adaptive_overnight_inventory_20261008.py 2>&1 | tee reports/20261008_adaptive_overnight_inventory/run.log\n",
            "timeout -k 10s 180s .venv/bin/python src/research/scripts/experiments/refresh_adaptive_overnight_inventory_report_20261008.py\n",
            "```\n\n",
            "Artifacts: `metrics.json`, `policy_summary.csv`, `borrow_stress_summary.csv`, `calendar_and_reversal_segments.csv`, `ticker_transition_attribution.csv`, `ticker_carry_diagnostics.csv`, `ticker_summary.csv`, `daily_policy_replay.csv`.\n\n",
            "Registry result will be appended below.\n",
        ]
    )
    REPORT_PATH.write_text("".join(report), encoding="utf-8")

    candidate_results = {
        "daily_returns": replay_by_policy["adaptive_252"]["daily"]["net"].iloc[oos_start_index:],
        "daily_fallback": replay_by_policy["adaptive_252"]["daily"]["fallback"].iloc[oos_start_index:],
        "daily_turnover": replay_by_policy["adaptive_252"]["daily"]["turnover"].iloc[oos_start_index:],
        "daily_gross_exps": replay_by_policy["adaptive_252"]["daily"]["gross_exposure"].iloc[oos_start_index:],
    }
    registry_record = record_backtest_experiment(
        name=Path(__file__).stem,
        hypothesis="PIT ticker-specific carry sizing using same-direction reuse probability and all modeled carry cashflows improves net OOS results over fixed alpha.",
        app_config=app_config,
        results=candidate_results,
        extra_metrics={
            "study_id": STUDY_ID,
            "trials": estimated_trial_count,
            "trial_sharpe_variance": trial_variance_annual,
            "trial_sharpe_variance_frequency": "annual",
            "trial_sharpes": current_trial_sharpes,
            "trading_days_per_year": ANNUALIZATION,
            "include_flat_days": True,
            "oos_start": str(oos_dates[0].date()),
            "oos_end": str(oos_dates[-1].date()),
            "baseline_policy": "fixed alpha long=0.75 short=0.50",
            "adaptive_policy": main_config.__dict__,
            "paired_block_bootstrap": bootstrap,
            "production_changed": False,
            "report_path": str(REPORT_PATH.relative_to(ROOT)),
        },
        decision=Decision.PENDING,
        reason="Short single OOS interval; simulated linear costs and no actual fill or ticker-level borrow evidence.",
        report_path=REPORT_PATH.relative_to(ROOT),
        registry_path=ROOT / "var" / "experiments" / "registry.jsonl",
        metrics_spec=MetricsSpec(annualization_periods=ANNUALIZATION, include_flat_days=True),
        study_id=STUDY_ID,
    )
    report.append(
        f"\nRegistry: record_id=`{registry_record.record_id}`, decision=`{registry_record.decision.value}`, "
        f"DSR={registry_record.deflated_sharpe()!r}, study_id=`{STUDY_ID}`.\n"
    )
    REPORT_PATH.write_text("".join(report), encoding="utf-8")
    payload["registry_record_id"] = registry_record.record_id
    payload["deflated_sharpe"] = registry_record.deflated_sharpe()
    metrics_path.write_text(
        json.dumps(_safe_json(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "Completed adaptive carry replay: OOS %s..%s (%d days), base net Sharpe %.3f, adaptive %.3f, report=%s",
        oos_dates[0].date(), oos_dates[-1].date(), len(oos_dates),
        base_row["net_sharpe"], main_row["net_sharpe"], REPORT_PATH,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
