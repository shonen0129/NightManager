#!/usr/bin/env python3
"""Map saved NightManager sector alpha to liquid same-sector stocks.

Research-only repricing. This experiment does not optimize or score ETF tracking
error and does not change production code or configuration.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import itertools
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/research/scripts/experiments"))

from leadlag.config.paths import default_registry_path  # noqa: E402
from leadlag.data.tickers import JP_TICKERS  # noqa: E402
from leadlag.execution.config import load_config_from_yaml  # noqa: E402
from leadlag.experiment_registry import (  # noqa: E402
    Decision,
    ExperimentRecord,
    ExperimentRegistry,
    compute_deflated_sharpe,
)

_PREVIOUS_PATH = ROOT / "src/research/scripts/experiments/experiment_nightmanager_etf_stock_proxy_20261007.py"
_PREVIOUS_SPEC = importlib.util.spec_from_file_location("nightmanager_etf_stock_proxy_reuse", _PREVIOUS_PATH)
if _PREVIOUS_SPEC is None or _PREVIOUS_SPEC.loader is None:
    raise RuntimeError(f"Could not load prior research helper: {_PREVIOUS_PATH}")
_PREVIOUS = importlib.util.module_from_spec(_PREVIOUS_SPEC)
sys.modules[_PREVIOUS_SPEC.name] = _PREVIOUS
_PREVIOUS_SPEC.loader.exec_module(_PREVIOUS)

LOG = logging.getLogger("nightmanager_direct_stock_alpha")
CONFIG_PATH = ROOT / "configs/research/nightmanager_direct_stock_alpha_20261007.yaml"
REPORT_DIR = ROOT / "reports/20261007_nightmanager_direct_stock_alpha"
RESULT_DIR = ROOT / "var/results/20261007_nightmanager_direct_stock_alpha"
EXPERIMENT_ID = "nightmanager_direct_stock_alpha_20261007"
ETF_COST_BPS = 5.0
MISSING_STOCK_COST_BPS = 50.0
PRIMARY_NAMES = [
    "etf_baseline",
    "single_stock_hybrid",
    "stock3_hybrid",
    "cost_aware_stock3_hybrid",
]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def _read_config() -> dict[str, Any]:
    value = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid research configuration: {CONFIG_PATH}")
    return value


def _load_inputs(config: dict[str, Any]) -> dict[str, Any]:
    source_path = ROOT / config["reuse"]["prior_market_loader_config"]
    source_config = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    inputs = _PREVIOUS._load_market_inputs(source_config)

    active_config_path = ROOT / config["reuse"]["active_production_config"]
    active_config = load_config_from_yaml(active_config_path, strict=False)
    side_leverage = float(active_config.strategy.side_leverage)
    expected_leverage = float(config["reuse"]["production_side_leverage_required"])
    if not np.isclose(side_leverage, expected_leverage, atol=1e-12, rtol=0.0):
        raise ValueError(
            f"Active production side_leverage={side_leverage} does not match "
            f"requested experiment value {expected_leverage}"
        )
    inputs["active_production_config"] = active_config
    inputs["side_leverage"] = side_leverage

    summary = inputs["input_summary"]
    snapshot = summary.get("mapping_snapshot", {})
    summary["mapping_snapshot"] = {
        key: value for key, value in snapshot.items() if key != "valid_tickers"
    }
    weights = inputs["weights"]
    observed = inputs["weight_observed"]
    model_gross = np.nansum(np.abs(weights[observed]), axis=1)
    model_net = np.nansum(weights[observed], axis=1)
    summary["saved_side_leverage"] = float(summary.get("saved_side_leverage", np.nan))
    summary["pnl_side_leverage_active_production"] = side_leverage
    summary["effective_gross_mean_post_side_leverage"] = float(np.mean(model_gross) * side_leverage)
    summary["effective_gross_max_post_side_leverage"] = float(np.max(model_gross) * side_leverage)
    summary["effective_net_abs_max_post_side_leverage"] = float(np.max(np.abs(model_net)) * side_leverage)
    summary["active_production_config_path"] = str(active_config_path.relative_to(ROOT))
    summary["active_production_config_sha256"] = _sha256_file(active_config_path)

    start = pd.Timestamp(config["evaluation"]["start_date"])
    end = pd.Timestamp(config["evaluation"]["end_date"])
    eval_mask = inputs["weight_observed"] & (inputs["dates"] >= start) & (inputs["dates"] <= end)
    eval_indices = np.flatnonzero(eval_mask)
    if len(eval_indices) == 0:
        raise RuntimeError("No observed NightManager target weights in the configured evaluation interval")
    if len(eval_indices) != int(eval_mask.sum()):
        raise RuntimeError("Evaluation date selection failed")

    gap_dir = ROOT / config["reuse"]["gap_distribution_dir"]
    matrices_dir = gap_dir / "matrices"
    mu_all = np.full((len(inputs["dates"]), len(JP_TICKERS)), np.nan, dtype=np.float64)
    mu_files_used: list[str] = []
    missing_mu_dates: list[str] = []
    for t in eval_indices:
        date = inputs["dates"][t]
        path = matrices_dir / f"mu_gap_{date:%Y%m%d}.npy"
        if not path.exists():
            missing_mu_dates.append(str(date.date()))
            continue
        mu = np.asarray(np.load(path), dtype=np.float64)
        if mu.shape != (len(JP_TICKERS),):
            raise ValueError(f"Unexpected mu_gap shape for {date.date()}: {mu.shape}")
        if not np.isfinite(mu).all():
            missing_mu_dates.append(str(date.date()))
            continue
        mu_all[t] = mu
        mu_files_used.append(str(path.relative_to(ROOT)))

    # Audit the trade-date and ticker order against the prior long-form forecast export.
    long_path = gap_dir / "gap_adjusted_distribution_long.csv"
    long = pd.read_csv(long_path, usecols=["signal_date", "trade_date", "ticker", "mu_gap"],
                       parse_dates=["signal_date", "trade_date"])
    long["trade_date"] = pd.DatetimeIndex(long["trade_date"]).normalize()
    long["signal_date"] = pd.DatetimeIndex(long["signal_date"]).normalize()
    long = long[long["trade_date"].isin(inputs["dates"][eval_indices])]
    long["ticker"] = long["ticker"].astype(str)
    long_pivot = long.pivot_table(index="trade_date", columns="ticker", values="mu_gap", aggfunc="first")
    ordered = long_pivot.reindex(index=inputs["dates"][eval_indices], columns=JP_TICKERS)
    matrix_compare_rows = []
    max_abs_diff = 0.0
    for t in eval_indices:
        if not np.isfinite(mu_all[t]).all():
            continue
        check = ordered.loc[inputs["dates"][t]].to_numpy(dtype=np.float64)
        if not np.isfinite(check).all():
            continue
        diff = float(np.max(np.abs(mu_all[t] - check)))
        max_abs_diff = max(max_abs_diff, diff)
        matrix_compare_rows.append(diff)
    if matrix_compare_rows and max_abs_diff > 1e-10:
        raise RuntimeError(f"Per-date mu_gap matrices disagree with long CSV (max abs diff={max_abs_diff})")

    signal_dates = (
        long.drop_duplicates("trade_date").set_index("trade_date")["signal_date"]
        .reindex(inputs["dates"][eval_indices])
    )
    eval_date_values = inputs["dates"][eval_indices]
    signal_date_values = pd.DatetimeIndex(signal_dates.to_numpy())
    signal_present = ~signal_date_values.isna()
    if not (signal_date_values[signal_present] < eval_date_values[signal_present]).all():
        raise RuntimeError("A stored mu_gap signal date is not earlier than its trade date")
    inputs["eval_indices"] = eval_indices
    inputs["eval_dates"] = inputs["dates"][eval_indices]
    inputs["eval_weights"] = inputs["weights"][eval_indices] * side_leverage
    inputs["eval_mu"] = mu_all[eval_indices]
    inputs["eval_mu_available"] = np.isfinite(mu_all[eval_indices]).all(axis=1)
    inputs["eval_signal_dates"] = signal_dates.to_numpy()
    inputs["missing_mu_dates"] = missing_mu_dates
    inputs["mu_audit"] = {
        "matrix_dir": str(matrices_dir.relative_to(ROOT)),
        "matrix_files_used": len(mu_files_used),
        "weight_days": len(eval_indices),
        "mu_available_days": int(np.isfinite(mu_all[eval_indices]).all(axis=1).sum()),
        "missing_or_nonfinite_mu_dates": missing_mu_dates,
        "long_csv": str(long_path.relative_to(ROOT)),
        "matrix_vs_long_csv_max_abs_difference": max_abs_diff if matrix_compare_rows else None,
        "realized_target_return_column_read": False,
    }
    inputs["config"] = config
    return inputs


def _strategy_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    e = config["evaluation"]
    base = {
        "beta_window": int(e["sector_beta_window"]),
        "recent_beta_window": int(e["recent_beta_window"]),
        "stability_diff": float(e["beta_stability_abs_diff"]),
        "beta_min": float(e["beta_min"]),
        "beta_max": float(e["beta_max"]),
        "min_adv": float(e["minimum_adv_jpy"]),
        "top_k": int(e["candidate_pool_top_k"]),
        "cap": float(e["max_individual_stock_weight_of_nav"]),
        "lambda_turn_bps": float(e["cost_turnover_penalty_bps"]),
    }
    specs = [
        {"name": "etf_baseline", "kind": "baseline", "primary": True, **base},
        {"name": "single_stock_hybrid", "kind": "single", "primary": True, **base},
        {"name": "stock3_hybrid", "kind": "stock3", "primary": True, **base},
        {"name": "cost_aware_stock3_hybrid", "kind": "costaware3", "primary": True, **base},
    ]
    sensitivity = e["sensitivity"]
    sensitivity_changes: list[tuple[str, dict[str, Any]]] = []
    for value in sensitivity["beta_window"]:
        w = int(value)
        sensitivity_changes.append((f"costaware_beta{w}", {"beta_window": w, "recent_beta_window": max(60, w // 2)}))
    for value in sensitivity["beta_stability_abs_diff"]:
        v = float(value)
        sensitivity_changes.append((f"costaware_stability{int(v * 100):02d}", {"stability_diff": v}))
    for lower, upper in sensitivity["beta_band"]:
        lo, hi = float(lower), float(upper)
        sensitivity_changes.append((f"costaware_beta_band_{int(lo*100)}_{int(hi*100)}",
                                    {"beta_min": lo, "beta_max": hi}))
    for value in sensitivity["minimum_adv_jpy"]:
        v = float(value)
        sensitivity_changes.append((f"costaware_adv{int(v / 1_000_000)}m", {"min_adv": v}))
    for value in sensitivity["candidate_pool_top_k"]:
        v = int(value)
        sensitivity_changes.append((f"costaware_k{v}", {"top_k": v}))
    for value in sensitivity["max_individual_stock_weight_of_nav"]:
        v = float(value)
        sensitivity_changes.append((f"costaware_cap{int(v * 100)}pct", {"cap": v}))
    for value in sensitivity["cost_turnover_penalty_bps"]:
        v = float(value)
        sensitivity_changes.append((f"costaware_turnlambda{int(v)}", {"lambda_turn_bps": v}))
    for name, changes in sensitivity_changes:
        spec = {"name": name, "kind": "costaware3", "primary": False, **copy.deepcopy(base)}
        spec.update(changes)
        specs.append(spec)
    names = [item["name"] for item in specs]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate strategy names: {names}")
    return specs


def _rolling_beta_for_group(
    stock_returns: np.ndarray,
    sector_returns: np.ndarray,
    group: np.ndarray,
    eval_indices: np.ndarray,
    window: int,
) -> np.ndarray:
    """Rolling OLS stock-on-parent-sector beta; all windows end at t-1."""
    n_days = stock_returns.shape[0]
    result = np.full((len(eval_indices), len(group)), np.nan, dtype=np.float32)
    if window <= 1:
        raise ValueError("Beta lookback must exceed one session")
    eligible_positions = np.flatnonzero(eval_indices >= window)
    if len(eligible_positions) == 0:
        return result
    X = stock_returns[:n_days, group]
    y = sector_returns[:n_days]
    valid = np.isfinite(X) & np.isfinite(y[:, None])
    x0 = np.where(valid, X, 0.0)
    y0 = np.where(valid, y[:, None], 0.0)
    count_prefix = np.vstack([np.zeros((1, len(group))), np.cumsum(valid, axis=0, dtype=np.int32)])
    sx_prefix = np.vstack([np.zeros((1, len(group))), np.cumsum(x0, axis=0)])
    sy_prefix = np.vstack([np.zeros((1, len(group))), np.cumsum(y0, axis=0)])
    sy2_prefix = np.vstack([np.zeros((1, len(group))), np.cumsum(y0 * y0, axis=0)])
    sxy_prefix = np.vstack([np.zeros((1, len(group))), np.cumsum(x0 * y0, axis=0)])
    hi = eval_indices[eligible_positions]
    lo = hi - window
    count = count_prefix[hi] - count_prefix[lo]
    sx = sx_prefix[hi] - sx_prefix[lo]
    sy = sy_prefix[hi] - sy_prefix[lo]
    sy2 = sy2_prefix[hi] - sy2_prefix[lo]
    sxy = sxy_prefix[hi] - sxy_prefix[lo]
    n = np.maximum(count, 1)
    covariance = sxy - (sx * sy / n)
    variance_y = sy2 - (sy * sy / n)
    beta = np.divide(
        covariance,
        variance_y,
        out=np.full_like(covariance, np.nan, dtype=np.float64),
        where=(count == window) & (variance_y > 1e-14),
    )
    result[eligible_positions] = beta.astype(np.float32)
    return result


def _build_beta_cache(inputs: dict[str, Any], specs: list[dict[str, Any]]) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    windows = {
        (int(spec["beta_window"]), int(spec["recent_beta_window"]))
        for spec in specs if spec["kind"] != "baseline"
    }
    ticker_to_etf = inputs["ticker_to_etf"]
    tickers = inputs["tickers"]
    ticker_to_index = inputs["ticker_to_index"]
    groups = {
        etf: np.asarray([ticker_to_index[t] for t in tickers if ticker_to_etf[t] == etf], dtype=np.int32)
        for etf in JP_TICKERS
    }
    beta_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for window, recent in sorted(windows):
        key = (window, recent)
        if key in beta_cache:
            continue
        LOG.info("Computing strictly historical beta windows %d / %d", window, recent)
        main = np.full((len(inputs["eval_indices"]), len(tickers)), np.nan, dtype=np.float32)
        fast = np.full_like(main, np.nan)
        for sector_index, etf in enumerate(JP_TICKERS):
            group = groups[etf]
            main[:, group] = _rolling_beta_for_group(
                inputs["stock_returns"], inputs["etf_returns"][:, sector_index],
                group, inputs["eval_indices"], window,
            )
            fast[:, group] = _rolling_beta_for_group(
                inputs["stock_returns"], inputs["etf_returns"][:, sector_index],
                group, inputs["eval_indices"], recent,
            )
        beta_cache[key] = (main, fast)
    inputs["stock_groups"] = groups
    inputs["stock_sector_indices"] = np.asarray(
        [JP_TICKERS.index(ticker_to_etf[ticker]) for ticker in tickers], dtype=np.int16
    )
    return beta_cache


def _allocate_beta_exposure(
    target_exposure: float,
    betas: np.ndarray,
    cap: float,
) -> tuple[np.ndarray, float]:
    """Allocate long stock weights while preserving beta-equivalent sector exposure."""
    if target_exposure <= 0.0 or len(betas) == 0:
        return np.zeros(len(betas), dtype=np.float64), max(0.0, target_exposure)
    beta = np.asarray(betas, dtype=np.float64)
    if not np.isfinite(beta).all() or np.any(beta <= 0.0):
        raise ValueError("Stock allocations require positive finite rolling betas")
    target_stock_beta = min(float(target_exposure), float(cap * beta.sum()))
    remaining = target_stock_beta
    weights = np.zeros(len(beta), dtype=np.float64)
    free = np.ones(len(beta), dtype=bool)
    while free.any() and remaining > 1e-14:
        denominator = float(np.sum(beta[free] ** 2))
        if denominator <= 0.0:
            break
        multiplier = remaining / denominator
        proposed = multiplier * beta[free]
        free_indices = np.flatnonzero(free)
        capped = proposed > cap + 1e-14
        if not capped.any():
            weights[free_indices] = proposed
            remaining = 0.0
            break
        capped_indices = free_indices[capped]
        weights[capped_indices] = cap
        remaining -= float(np.sum(beta[capped_indices] * cap))
        free[capped_indices] = False
    residual_etf = max(0.0, float(target_exposure - np.dot(beta, weights)))
    return weights, residual_etf


def _stock_cost_rates(inputs: dict[str, Any], raw_index: int) -> np.ndarray:
    values = inputs["unit_cost_bps"][raw_index].copy()
    missing = ~np.isfinite(values) | (values < 0.0)
    values[missing] = MISSING_STOCK_COST_BPS
    return values


def _choose_cost_aware_subset(
    exposure: float,
    pool: np.ndarray,
    betas: np.ndarray,
    cap: float,
    previous_stock: np.ndarray,
    previous_active_sector: np.ndarray,
    previous_etf: float,
    current_stock_costs: np.ndarray,
    lambda_turn_bps: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Enumerate up-to-three names, minimizing estimated cost plus turnover penalty."""
    if not len(pool):
        return np.asarray([], dtype=np.int32), np.asarray([], dtype=np.float64), max(0.0, exposure), float("nan")
    # The subset family is small (at most 12 choose 1..3), so enumerate supports
    # once and score all of them with vectorized capped beta allocations.
    supports = [
        tuple(item)
        for size in range(1, min(3, len(pool)) + 1)
        for item in itertools.combinations(range(len(pool)), size)
    ]
    local_ids = np.full((len(supports), 3), -1, dtype=np.int32)
    for row, support in enumerate(supports):
        local_ids[row, :len(support)] = support
    active = local_ids >= 0
    beta_choices = np.zeros_like(local_ids, dtype=np.float64)
    rows, slots = np.nonzero(active)
    beta_choices[rows, slots] = betas[local_ids[rows, slots]]

    target_stock_beta = np.minimum(float(exposure), cap * beta_choices.sum(axis=1))
    remaining = target_stock_beta.copy()
    target_weights = np.zeros_like(beta_choices)
    free = active.copy()
    for _ in range(3):
        denominator = (beta_choices**2 * free).sum(axis=1)
        proposed = np.divide(
            remaining[:, None] * beta_choices * free,
            denominator[:, None],
            out=np.zeros_like(beta_choices),
            where=denominator[:, None] > 1e-16,
        )
        capped = free & (proposed > cap + 1e-14)
        rows_with_caps = capped.any(axis=1)
        if rows_with_caps.any():
            target_weights[capped] = cap
            remaining -= (beta_choices * capped).sum(axis=1) * cap
            free[capped] = False
        finalized = free.any(axis=1) & ~rows_with_caps
        if finalized.any():
            final_slots = finalized[:, None] & free
            target_weights[final_slots] = proposed[final_slots]
            remaining[finalized] = 0.0
            free[finalized] = False
    new_pool = np.zeros((len(supports), len(pool)), dtype=np.float64)
    for slot in range(3):
        valid_slot = local_ids[:, slot] >= 0
        new_pool[np.flatnonzero(valid_slot), local_ids[valid_slot, slot]] = target_weights[valid_slot, slot]
    residual = np.maximum(0.0, float(exposure) - (target_weights * beta_choices).sum(axis=1))

    old_pool = previous_stock[pool]
    delta_pool = new_pool - old_pool[None, :]
    rate_pool = current_stock_costs[pool]
    pool_cost = np.abs(delta_pool) @ rate_pool
    pool_turn = 0.5 * np.abs(delta_pool).sum(axis=1)
    outside = previous_active_sector[~np.isin(previous_active_sector, pool)]
    outside_cost = float(np.dot(np.abs(previous_stock[outside]), current_stock_costs[outside]))
    outside_turn = 0.5 * float(np.abs(previous_stock[outside]).sum())
    etf_delta = residual - float(previous_etf)
    etf_cost = np.abs(etf_delta) * ETF_COST_BPS
    etf_turn = 0.5 * np.abs(etf_delta)
    scores = (
        pool_cost + outside_cost + etf_cost
        + lambda_turn_bps * (pool_turn + outside_turn + etf_turn)
    )
    winner = int(np.argmin(scores))
    selected = active[winner]
    chosen = pool[local_ids[winner, selected]]
    weights = target_weights[winner, selected]
    return chosen, weights, float(residual[winner]), float(scores[winner])


def _one_strategy(
    inputs: dict[str, Any],
    spec: dict[str, Any],
    beta_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    dates: pd.DatetimeIndex = inputs["eval_dates"]
    raw_indices = inputs["eval_indices"]
    target_weights: np.ndarray = inputs["eval_weights"]
    mu: np.ndarray = inputs["eval_mu"]
    mu_available: np.ndarray = inputs["eval_mu_available"]
    stock_returns = inputs["stock_returns"]
    etf_returns = inputs["etf_returns"]
    adv = inputs["adv"]
    tickers = inputs["tickers"]
    groups = inputs["stock_groups"]
    sector_of_stock = inputs["stock_sector_indices"]
    n_dates = len(dates)
    n_stocks = len(tickers)
    n_sectors = len(JP_TICKERS)
    kind = spec["kind"]

    if kind == "baseline":
        main_beta = recent_beta = None
    else:
        main_beta, recent_beta = beta_cache[(spec["beta_window"], spec["recent_beta_window"])]

    stock_costs_all = np.empty((n_dates, n_stocks), dtype=np.float32)
    for row, raw_t in enumerate(raw_indices):
        stock_costs_all[row] = _stock_cost_rates(inputs, int(raw_t)).astype(np.float32)

    daily: list[dict[str, Any]] = []
    sector_daily: list[dict[str, Any]] = []
    allocations: list[dict[str, Any]] = []
    previous_stock = np.zeros(n_stocks, dtype=np.float64)
    previous_etf = np.zeros(n_sectors, dtype=np.float64)

    for row, raw_t in enumerate(raw_indices):
        raw_t = int(raw_t)
        date = dates[row]
        sector_target = target_weights[row].copy()
        etf_position = sector_target.copy()
        stock_position = np.zeros(n_stocks, dtype=np.float64)
        beta_for_day = None if main_beta is None else main_beta[row].astype(np.float64, copy=False)
        mu_today = mu[row]
        can_map = kind != "baseline" and mu_available[row]

        if can_map:
            for sector_index, etf in enumerate(JP_TICKERS):
                exposure = float(sector_target[sector_index])
                if exposure <= 1e-12:
                    continue
                group = groups[etf]
                beta_sector = beta_for_day[group]
                beta_recent = recent_beta[row, group].astype(np.float64, copy=False)
                adv_sector = adv[raw_t, group]
                costs_sector = stock_costs_all[row, group]
                cost_observed = np.isfinite(inputs["unit_cost_bps"][raw_t, group]) & (
                    inputs["unit_cost_bps"][raw_t, group] >= 0.0
                )
                valid = (
                    np.isfinite(beta_sector) & np.isfinite(beta_recent)
                    & (beta_sector >= spec["beta_min"]) & (beta_sector <= spec["beta_max"])
                    & (np.abs(beta_sector - beta_recent) <= spec["stability_diff"])
                    & np.isfinite(adv_sector) & (adv_sector >= spec["min_adv"])
                    & np.isfinite(costs_sector) & cost_observed
                )
                eligible = group[valid]
                if not len(eligible):
                    continue
                eligible_adv = adv[raw_t, eligible]
                eligible_order = np.lexsort((eligible, -eligible_adv))
                pool = eligible[eligible_order[:int(spec["top_k"])]]
                pool_betas = beta_for_day[pool]
                if kind == "single":
                    chosen = pool[:1]
                    chosen_weights, residual = _allocate_beta_exposure(exposure, beta_for_day[chosen], spec["cap"])
                    objective = np.nan
                elif kind == "stock3":
                    chosen = pool[:3]
                    chosen_weights, residual = _allocate_beta_exposure(exposure, beta_for_day[chosen], spec["cap"])
                    objective = np.nan
                else:
                    previous_active_sector = np.flatnonzero(
                        (sector_of_stock == sector_index) & (previous_stock > 1e-12)
                    )
                    chosen, chosen_weights, residual, objective = _choose_cost_aware_subset(
                        exposure=exposure,
                        pool=pool,
                        betas=pool_betas,
                        cap=float(spec["cap"]),
                        previous_stock=previous_stock,
                        previous_active_sector=previous_active_sector,
                        previous_etf=float(previous_etf[sector_index]),
                        current_stock_costs=stock_costs_all[row],
                        lambda_turn_bps=float(spec["lambda_turn_bps"]),
                    )
                if len(chosen):
                    beta_position = float(np.dot(chosen_weights, beta_for_day[chosen]))
                    if abs(beta_position + residual - exposure) > 1e-8:
                        raise RuntimeError(f"Beta-equivalent exposure was not maintained on {date.date()} / {etf}")
                    if np.any(chosen_weights > float(spec["cap"]) + 1e-10):
                        raise RuntimeError(f"Per-stock NAV cap exceeded on {date.date()} / {etf}")
                    stock_position[chosen] = chosen_weights
                    etf_position[sector_index] = max(0.0, exposure - beta_position)
                    signal_date = inputs["eval_signal_dates"][row]
                    for ticker_index, weight in zip(chosen, chosen_weights, strict=True):
                        allocations.append({
                            "trade_date": str(date.date()),
                            "signal_date": None if pd.isna(signal_date) else str(pd.Timestamp(signal_date).date()),
                            "variant": spec["name"],
                            "topix17_etf": etf,
                            "ticker": tickers[int(ticker_index)],
                            "sector_mu_gap": float(mu_today[sector_index]),
                            "rolling_beta": float(beta_for_day[ticker_index]),
                            "beta_training_start": str(inputs["dates"][raw_t - int(spec["beta_window"])].date()),
                            "beta_training_end": str(inputs["dates"][raw_t - 1].date()),
                            "expected_stock_return_beta_times_sector_signal": float(
                                beta_for_day[ticker_index] * mu_today[sector_index]
                            ),
                            "target_stock_weight_nav": float(weight),
                            "stock_beta_exposure": float(weight * beta_for_day[ticker_index]),
                            "residual_etf_weight": float(etf_position[sector_index]),
                            "trailing_adv_jpy": float(adv[raw_t, ticker_index]),
                            "estimated_stock_cost_bps_per_side": float(stock_costs_all[row, ticker_index]),
                            "max_stock_weight_nav": float(spec["cap"]),
                            "sector_target_effective_weight": exposure,
                            "cost_aware_selection_objective": None if not np.isfinite(objective) else float(objective),
                        })

        # If the stored 17D forecast is absent, keep the complete ETF target that day.
        if not can_map:
            stock_position[:] = 0.0
            etf_position = sector_target.copy()

        current_stock_returns = stock_returns[raw_t]
        missing_return_mask = (~np.isfinite(current_stock_returns)) & (np.abs(stock_position) > 0.0)
        used_stock_returns = np.nan_to_num(current_stock_returns, nan=0.0, posinf=0.0, neginf=0.0)
        current_etf_returns = etf_returns[raw_t]
        if not np.isfinite(current_etf_returns).all():
            raise RuntimeError(f"Missing ETF open/close return on target-weight date {date.date()}")

        stock_gross_by_sector = np.bincount(
            sector_of_stock,
            weights=stock_position * used_stock_returns,
            minlength=n_sectors,
        )
        etf_gross_by_sector = etf_position * current_etf_returns
        gross_by_sector = stock_gross_by_sector + etf_gross_by_sector

        # Each changed long or short leg incurs a one-way per-side charge.
        stock_delta = stock_position - previous_stock
        stock_delta_abs = np.abs(stock_delta)
        current_stock_costs = stock_costs_all[row].astype(np.float64)
        stock_trade_cost_by_sector = np.bincount(
            sector_of_stock,
            weights=stock_delta_abs * current_stock_costs / 10000.0,
            minlength=n_sectors,
        )
        etf_long_now = np.maximum(etf_position, 0.0)
        etf_long_prev = np.maximum(previous_etf, 0.0)
        etf_short_now = np.maximum(-etf_position, 0.0)
        etf_short_prev = np.maximum(-previous_etf, 0.0)
        etf_long_cost = np.abs(etf_long_now - etf_long_prev) * ETF_COST_BPS / 10000.0
        etf_short_cost = np.abs(etf_short_now - etf_short_prev) * ETF_COST_BPS / 10000.0
        sector_cost = stock_trade_cost_by_sector + etf_long_cost + etf_short_cost
        total_cost = float(sector_cost.sum())
        one_way_turnover = 0.5 * (float(stock_delta_abs.sum()) + float(np.abs(etf_position - previous_etf).sum()))

        long_gross = float(np.dot(np.maximum(stock_position, 0.0), used_stock_returns))
        etf_long_gross = float(np.dot(etf_long_now, current_etf_returns))
        etf_short_gross = float(np.dot(-etf_short_now, current_etf_returns))
        long_gross += etf_long_gross
        short_gross = etf_short_gross
        long_cost = float(stock_trade_cost_by_sector.sum() + etf_long_cost.sum())
        short_cost = float(etf_short_cost.sum())
        gross = float(gross_by_sector.sum())
        net = gross - total_cost

        if beta_for_day is None:
            stock_beta_by_sector = np.zeros(n_sectors, dtype=np.float64)
            current_beta_exposure = etf_position.copy()
        else:
            stock_beta_by_sector = np.bincount(
                sector_of_stock,
                weights=stock_position * np.nan_to_num(beta_for_day, nan=0.0),
                minlength=n_sectors,
            )
            current_beta_exposure = etf_position + stock_beta_by_sector
        nominal_sector_exposure = etf_position + np.bincount(
            sector_of_stock, weights=stock_position, minlength=n_sectors
        )
        expected_alpha_by_sector = current_beta_exposure * np.nan_to_num(mu_today, nan=0.0)
        if not mu_available[row]:
            expected_alpha_by_sector[:] = np.nan
        gross_exposure = float(np.abs(stock_position).sum() + np.abs(etf_position).sum())
        beta_error = current_beta_exposure - sector_target
        nominal_error = nominal_sector_exposure - sector_target
        daily.append({
            "trade_date": date,
            "variant": spec["name"],
            "gross_return": gross,
            "net_return": net,
            "estimated_transaction_cost_return": total_cost,
            "estimated_transaction_cost_bps_nav": total_cost * 10000.0,
            "one_way_turnover": one_way_turnover,
            "gross_exposure": gross_exposure,
            "long_gross_contribution": long_gross,
            "short_gross_contribution": short_gross,
            "long_transaction_cost": long_cost,
            "short_transaction_cost": short_cost,
            "long_net_contribution": long_gross - long_cost,
            "short_net_contribution": short_gross - short_cost,
            "missing_selected_stock_return_weight": float(np.abs(stock_position[missing_return_mask]).sum()),
            "missing_selected_stock_names": int(missing_return_mask.sum()),
            "mu_available": bool(mu_available[row]),
            "beta_exposure_abs_error": float(np.abs(beta_error).sum()),
            "nominal_sector_exposure_abs_error": float(np.abs(nominal_error).sum()),
            "expected_alpha_from_stored_signal": float(np.nansum(expected_alpha_by_sector)),
        })

        for sector_index, etf in enumerate(JP_TICKERS):
            sector_daily.append({
                "trade_date": date,
                "variant": spec["name"],
                "topix17_etf": etf,
                "sector_target_weight": float(sector_target[sector_index]),
                "nominal_sector_exposure": float(nominal_sector_exposure[sector_index]),
                "beta_adjusted_sector_exposure": float(current_beta_exposure[sector_index]),
                "stock_beta_exposure": float(stock_beta_by_sector[sector_index]),
                "beta_exposure_error": float(beta_error[sector_index]),
                "nominal_exposure_error": float(nominal_error[sector_index]),
                "gross_pnl_contribution": float(gross_by_sector[sector_index]),
                "net_pnl_contribution": float(gross_by_sector[sector_index] - sector_cost[sector_index]),
                "estimated_cost_return": float(sector_cost[sector_index]),
                "estimated_cost_bps_nav": float(sector_cost[sector_index] * 10000.0),
                "one_way_turnover": 0.5 * (
                    float(np.abs(stock_delta[groups[etf]]).sum())
                    + abs(float(etf_position[sector_index] - previous_etf[sector_index]))
                ),
                "long_gross_contribution": float(
                    stock_gross_by_sector[sector_index]
                    + etf_long_now[sector_index] * current_etf_returns[sector_index]
                ),
                "short_gross_contribution": float(
                    -etf_short_now[sector_index] * current_etf_returns[sector_index]
                ),
                "expected_alpha_from_stored_signal": (
                    float(expected_alpha_by_sector[sector_index]) if mu_available[row] else np.nan
                ),
                "mu_gap_sector": float(mu_today[sector_index]) if mu_available[row] else np.nan,
            })

        previous_stock = stock_position
        previous_etf = etf_position

    daily_frame = pd.DataFrame(daily)
    sector_frame = pd.DataFrame(sector_daily)
    allocation_frame = pd.DataFrame(allocations)
    if not allocation_frame.empty:
        if not (pd.to_datetime(allocation_frame["beta_training_end"]) < pd.to_datetime(allocation_frame["trade_date"])).all():
            raise RuntimeError("Rolling beta training window includes the trade date")
        if (allocation_frame["target_stock_weight_nav"] > allocation_frame["max_stock_weight_nav"] + 1e-10).any():
            raise RuntimeError("Saved stock allocation exceeds its configured concentration cap")
    return {
        "spec": spec,
        "daily": daily_frame,
        "sector_daily": sector_frame,
        "allocations": allocation_frame,
    }


def _max_drawdown(returns: np.ndarray) -> float:
    equity = np.cumprod(1.0 + np.asarray(returns, dtype=np.float64))
    peaks = np.maximum.accumulate(np.concatenate(([1.0], equity))) [1:]
    return float(np.min(equity / peaks - 1.0)) if len(equity) else float("nan")


def _series_metrics(returns: np.ndarray, annualization: int) -> dict[str, Any]:
    values = np.asarray(returns, dtype=np.float64)
    if not np.isfinite(values).all() or len(values) < 2:
        raise ValueError("Metrics require a complete finite daily return series")
    std = float(np.std(values, ddof=1))
    sharpe = float(np.mean(values) / std * np.sqrt(annualization)) if std > 1e-15 else float("nan")
    return {
        "observations": int(len(values)),
        "mean_daily_return": float(np.mean(values)),
        "total_return": float(np.prod(1.0 + values) - 1.0),
        "annualized_sharpe": sharpe,
        "max_drawdown": _max_drawdown(values),
        "daily_volatility": std,
    }


def _block_bootstrap_ci(values: np.ndarray, block: int, samples: int, seed: int) -> list[float]:
    series = np.asarray(values, dtype=np.float64)
    n = len(series)
    if n < block:
        return [float(np.mean(series)), float(np.mean(series))]
    rng = np.random.default_rng(seed)
    full_count, remainder = divmod(n, block)
    block_count = full_count + int(remainder > 0)
    extended = np.concatenate((series, series[:block]))
    prefix = np.concatenate(([0.0], np.cumsum(extended)))
    full_block_sums = prefix[block:block + n] - prefix[:n]
    starts = rng.integers(0, n, size=(samples, block_count))
    sums = full_block_sums[starts[:, :full_count]].sum(axis=1)
    if remainder:
        partial_block_sums = prefix[remainder:remainder + n] - prefix[:n]
        sums += partial_block_sums[starts[:, full_count]]
    means = sums / n
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _summarize(
    all_daily: pd.DataFrame,
    all_sector: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    e = config["evaluation"]
    annualization = int(e["annualization_sessions"])
    baseline = all_daily.loc[all_daily["variant"] == "etf_baseline"].set_index("trade_date")
    rows = []
    paired_rows = []
    strategy_names = list(all_daily["variant"].drop_duplicates())
    bootstrap = int(e["bootstrap_resamples"])
    block = int(e["bootstrap_block_sessions"])
    seed = int(e["bootstrap_seed"])
    for index, name in enumerate(strategy_names):
        frame = all_daily.loc[all_daily["variant"] == name].sort_values("trade_date").copy()
        metrics = _series_metrics(frame["gross_return"].to_numpy(), annualization)
        net_metrics = _series_metrics(frame["net_return"].to_numpy(), annualization)
        base = baseline.reindex(pd.DatetimeIndex(frame["trade_date"]))
        delta_net = frame["net_return"].to_numpy() - base["net_return"].to_numpy()
        delta_gross = frame["gross_return"].to_numpy() - base["gross_return"].to_numpy()
        ci = [0.0, 0.0] if name == "etf_baseline" else _block_bootstrap_ci(delta_net, block, bootstrap, seed + index)
        row = {
            "variant": name,
            "n_observations": len(frame),
            "gross_sharpe": metrics["annualized_sharpe"],
            "net_sharpe": net_metrics["annualized_sharpe"],
            "gross_total_return": metrics["total_return"],
            "net_total_return": net_metrics["total_return"],
            "gross_max_drawdown": metrics["max_drawdown"],
            "net_max_drawdown": net_metrics["max_drawdown"],
            "mean_daily_gross_return": metrics["mean_daily_return"],
            "mean_daily_net_return": net_metrics["mean_daily_return"],
            "average_one_way_turnover": float(frame["one_way_turnover"].mean()),
            "total_one_way_turnover": float(frame["one_way_turnover"].sum()),
            "average_estimated_cost_bps_nav_per_day": float(frame["estimated_transaction_cost_bps_nav"].mean()),
            "total_estimated_cost_return": float(frame["estimated_transaction_cost_return"].sum()),
            "long_gross_contribution_sum": float(frame["long_gross_contribution"].sum()),
            "short_gross_contribution_sum": float(frame["short_gross_contribution"].sum()),
            "long_net_contribution_sum": float(frame["long_net_contribution"].sum()),
            "short_net_contribution_sum": float(frame["short_net_contribution"].sum()),
            "mean_gross_exposure": float(frame["gross_exposure"].mean()),
            "max_gross_exposure": float(frame["gross_exposure"].max()),
            "missing_mu_days": int((~frame["mu_available"]).sum()),
            "missing_selected_stock_return_weight_days": int((frame["missing_selected_stock_return_weight"] > 0).sum()),
            "mean_missing_selected_stock_return_weight": float(frame["missing_selected_stock_return_weight"].mean()),
            "mean_beta_exposure_abs_error": float(frame["beta_exposure_abs_error"].mean()),
            "p95_beta_exposure_abs_error": float(frame["beta_exposure_abs_error"].quantile(0.95)),
            "mean_nominal_sector_exposure_abs_error": float(frame["nominal_sector_exposure_abs_error"].mean()),
            "paired_mean_daily_net_difference_vs_etf": float(delta_net.mean()),
            "paired_mean_daily_net_difference_vs_etf_bps": float(delta_net.mean() * 10000.0),
            "paired_net_difference_block_bootstrap_ci95": ci,
            "paired_mean_daily_gross_difference_vs_etf_bps": float(delta_gross.mean() * 10000.0),
        }
        rows.append(row)
        paired_rows.extend({
            "trade_date": date,
            "variant": name,
            "gross_return": float(gross),
            "net_return": float(net),
            "baseline_gross_return": float(base_gross),
            "baseline_net_return": float(base_net),
            "gross_pnl_difference_vs_etf": float(gross - base_gross),
            "net_pnl_difference_vs_etf": float(net - base_net),
        } for date, gross, net, base_gross, base_net in zip(
            frame["trade_date"], frame["gross_return"], frame["net_return"],
            base["gross_return"], base["net_return"], strict=True
        ))
    summary = pd.DataFrame(rows)

    # DSR uses all predeclared variants in this run as the within-family trial set.
    sharpes = summary["net_sharpe"].to_numpy(dtype=np.float64)
    trial_count = int(len(sharpes))
    dsr_values: list[float | None] = []
    for name in strategy_names:
        returns = all_daily.loc[all_daily["variant"] == name].sort_values("trade_date")["net_return"].to_numpy()
        row = summary.loc[summary["variant"] == name].iloc[0]
        metrics = {
            "net_sharpe": float(row["net_sharpe"]),
            "metric_status": "valid",
            "metric_schema_version": "daily-v1",
            "net_sharpe_frequency": "annual",
            "trading_days_per_year": annualization,
            "trial_count_status": "unknown",
            "trials": trial_count,
            "n_observations": len(returns),
            "returns": returns.tolist(),
            "trial_sharpes": sharpes.tolist(),
        }
        dsr_values.append(compute_deflated_sharpe(metrics))
    summary["trial_count_for_dsr"] = trial_count
    summary["deflated_sharpe_ratio"] = dsr_values
    portfolio_gate = config["decision_gate"]["portfolio_candidate_requires"]
    baseline_mdd = float(summary.loc[summary["variant"] == "etf_baseline", "net_max_drawdown"].iloc[0])
    lower_ci_gate = float(portfolio_gate["paired_net_difference_block_ci_lower_above_bps_per_day"])
    max_mdd_deterioration = float(portfolio_gate["maximum_net_mdd_deterioration_percentage_points"]) / 100.0
    max_beta_gap = float(portfolio_gate["mean_abs_beta_exposure_error_nav"])
    summary["portfolio_candidate_gate"] = [
        "baseline"
        if name == "etf_baseline"
        else (
            "candidate"
            if row["paired_net_difference_block_bootstrap_ci95"][0] * 10000.0 > lower_ci_gate
            and row["net_max_drawdown"] >= baseline_mdd - max_mdd_deterioration
            and row["mean_beta_exposure_abs_error"] <= max_beta_gap
            else "not_passed"
        )
        for name, (_, row) in zip(strategy_names, summary.iterrows(), strict=True)
    ]

    all_sector = all_sector.copy()
    all_sector["year"] = pd.to_datetime(all_sector["trade_date"]).dt.year
    annual_rows = []
    for (variant, year), frame in all_daily.assign(year=pd.to_datetime(all_daily["trade_date"]).dt.year).groupby(["variant", "year"]):
        gross = _series_metrics(frame["gross_return"].to_numpy(), annualization)
        net = _series_metrics(frame["net_return"].to_numpy(), annualization)
        base_frame = all_daily[
            (all_daily["variant"] == "etf_baseline")
            & (pd.to_datetime(all_daily["trade_date"]).dt.year == year)
        ].sort_values("trade_date")
        candidate = frame.sort_values("trade_date")
        diff = candidate["net_return"].to_numpy() - base_frame["net_return"].to_numpy()
        annual_rows.append({
            "variant": variant,
            "year": int(year),
            "observations": len(frame),
            "gross_sharpe": gross["annualized_sharpe"],
            "net_sharpe": net["annualized_sharpe"],
            "gross_total_return": gross["total_return"],
            "net_total_return": net["total_return"],
            "net_max_drawdown": net["max_drawdown"],
            "paired_mean_daily_net_difference_vs_etf_bps": float(diff.mean() * 10000.0),
        })
    annual = pd.DataFrame(annual_rows)

    sector_rows = []
    sector_gate = config["decision_gate"]["sector_classification"]
    sector_lower = float(sector_gate["improvement_if_paired_net_difference_block_ci_lower_above_bps_per_day"])
    sector_upper = float(sector_gate["deterioration_if_paired_net_difference_block_ci_upper_below_bps_per_day"])
    sector_error_gate = float(sector_gate["maximum_mean_abs_beta_exposure_error_nav"])
    core_sector = all_sector[all_sector["variant"].isin(PRIMARY_NAMES)]
    for (variant, etf), frame in core_sector.groupby(["variant", "topix17_etf"]):
        baseline_frame = core_sector[
            (core_sector["variant"] == "etf_baseline")
            & (core_sector["topix17_etf"] == etf)
        ].sort_values("trade_date")
        frame = frame.sort_values("trade_date")
        base_net = baseline_frame["net_pnl_contribution"].to_numpy()
        this_net = frame["net_pnl_contribution"].to_numpy()
        paired_difference = this_net - base_net
        beta_error = frame["beta_exposure_error"].abs().to_numpy()
        positive_target = np.maximum(frame["sector_target_weight"].to_numpy(), 0.0)
        stock_beta_proxy = np.maximum(frame["stock_beta_exposure"].to_numpy(), 0.0)
        denominator = float(positive_target.sum())
        sector_ci = (
            [0.0, 0.0]
            if variant == "etf_baseline"
            else _block_bootstrap_ci(
                paired_difference,
                block,
                bootstrap,
                seed + 1000 + JP_TICKERS.index(etf) + PRIMARY_NAMES.index(variant) * 100,
            )
        )
        mean_beta_error = float(beta_error.mean())
        sector_status = (
            "baseline"
            if variant == "etf_baseline"
            else "improvement"
            if sector_ci[0] * 10000.0 > sector_lower and mean_beta_error <= sector_error_gate
            else "deterioration"
            if sector_ci[1] * 10000.0 < sector_upper and mean_beta_error <= sector_error_gate
            else "inconclusive"
        )
        sector_rows.append({
            "variant": variant,
            "topix17_etf": etf,
            "net_pnl_contribution_sum": float(this_net.sum()),
            "gross_pnl_contribution_sum": float(frame["gross_pnl_contribution"].sum()),
            "net_pnl_difference_vs_etf_sum": float((this_net - base_net).sum()),
            "net_pnl_difference_vs_etf_bps_nav": float((this_net - base_net).sum() * 10000.0),
            "paired_mean_daily_net_difference_vs_etf_bps_nav": float(paired_difference.mean() * 10000.0),
            "paired_net_difference_block_ci95_bps_nav_per_day": [sector_ci[0] * 10000.0, sector_ci[1] * 10000.0],
            "sector_result": sector_status,
            "estimated_cost_return_sum": float(frame["estimated_cost_return"].sum()),
            "average_one_way_turnover": float(frame["one_way_turnover"].mean()),
            "mean_abs_beta_exposure_error": mean_beta_error,
            "p95_abs_beta_exposure_error": float(np.quantile(beta_error, 0.95)),
            "nominal_exposure_error_mean_abs": float(frame["nominal_exposure_error"].abs().mean()),
            "long_target_stock_beta_exposure_share": (
                float(np.maximum(stock_beta_proxy, 0.0).sum() / denominator) if denominator > 0 else 0.0
            ),
            "mean_sector_signal": float(frame["mu_gap_sector"].mean()),
            "mean_long_gross_contribution": float(frame["long_gross_contribution"].mean()),
            "mean_short_gross_contribution": float(frame["short_gross_contribution"].mean()),
        })
    sector_summary = pd.DataFrame(sector_rows)
    return summary, pd.DataFrame(paired_rows), annual, sector_summary


def _related_prior_names() -> list[str]:
    tokens = ("nightmanager_etf_stock_proxy", "subsector", "jpx33_direct")
    registry = ExperimentRegistry(default_registry_path())
    names: set[str] = set()
    for record in registry:
        search = f"{record.name} {record.hypothesis} {record.report_path or ''}".lower()
        if any(token in search for token in tokens):
            names.add(record.name)
    return sorted(names)


def _write_report(
    inputs: dict[str, Any],
    specs: list[dict[str, Any]],
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    annual: pd.DataFrame,
    sector_summary: pd.DataFrame,
    allocation_rows: pd.DataFrame,
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULT_DIR / "portfolio_summary.csv", index=False)
    paired.to_csv(RESULT_DIR / "daily_portfolio_pnl_and_paired_differences.csv", index=False)
    annual.to_csv(RESULT_DIR / "annual_portfolio_metrics.csv", index=False)
    sector_summary.to_csv(RESULT_DIR / "sector_metrics.csv", index=False)
    allocation_rows.to_csv(RESULT_DIR / "daily_stock_allocations.csv", index=False)

    core = summary.set_index("variant")
    target_summary = inputs["input_summary"]
    result_payload = {
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config_path": str(CONFIG_PATH.relative_to(ROOT)),
        "config_sha256": _sha256_file(CONFIG_PATH),
        "input_summary": target_summary,
        "mu_gap_audit": inputs["mu_audit"],
        "stock_0910_price_available": False,
        "return_interval": "open_to_close proxy for both individual stocks and TOPIX-17 ETFs",
        "active_side_leverage": inputs["side_leverage"],
        "rolling_selection": {
            "training_end_excluded_trade_date": True,
            "beta_definition": "OLS covariance(stock open-close, parent ETF open-close) / variance(parent ETF open-close), on strictly prior sessions",
            "liquidity_definition": "252-session average close times volume shifted one session; same TOPIX-17 parent only",
            "stock_expected_return": "rolling beta times stored NightManager mu_gap for that TOPIX-17 sector",
            "portfolio_mapping": "positive sector exposure is allocated to long stocks so sum(stock_weight * beta) matches the sector exposure where cap allows; residual is ETF; negative sector exposure remains ETF",
            "cost_aware_objective": "estimated one-way trading cost plus lambda-turnover bps times one-way target turnover; exact enumeration of same-sector candidate subsets of size 1 to 3",
            "specifications": specs,
            "individual_stock_weight_cap_nav": "cap applies after side_leverage=1.30",
        },
        "portfolio_metrics": summary.to_dict("records"),
        "related_prior_experiment_names": _related_prior_names(),
        "production_modified": False,
        "known_limits": [
            "survivorship bias: only the 1620 currently present stocks in the expanded mapping have local history",
            "classification is the 2026-07-31 parent mapping held fixed across the sample, not point-in-time",
            "individual stock 09:10 prices are unavailable; primary PnL uses open-to-close proxy",
            "costs are estimates, ETF quote spread is unavailable, ETF gets only 5 bps per side while stocks get prior estimated half-spread plus 5 bps",
            "commission, exchange fees, taxes, market impact, borrow, financing, reverse, and intraday implementation shortfall are excluded",
            "portfolio period was already inspected in prior ETF-basket work; daily decisions are rolling historical, but this is retrospective pseudo-OOS, not an untouched forward test",
            "stored NightManager sector weights are re-priced from the saved exact-production artifact; this study applies current production side_leverage 1.30 without regenerating the full model decisions",
        ],
    }
    (RESULT_DIR / "summary.json").write_text(
        json.dumps(_clean(result_payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    base = core.loc["etf_baseline"]
    lines = [
        "# NightManager sector alpha: direct mapping to high-liquidity stocks",
        "",
        "- 作成日: 2026-10-07（Asia/Tokyo）",
        "- 実験コード: src/research/scripts/experiments/experiment_nightmanager_direct_stock_alpha_20261007.py",
        "- 設定: configs/research/nightmanager_direct_stock_alpha_20261007.yaml",
        "- 結果: var/results/20261007_nightmanager_direct_stock_alpha/",
        "- production code/config: 未変更",
        "",
        "## 仮説と今回の問い",
        "",
        "既存NightManagerのTOPIX-17 sector signalを同業種内の流動性が高くbetaの安定した個別株へ写像し、ETFを直接使う場合と費用控除後PnLを比較した。株の選択・配分はETFリターンへの追随精度を最適化していない。",
        "",
        "各個別株の予測リターンは「trade-dateのmu_gap × 前日までに推定した親ETF beta」。ロングの各sector exposureはこのbeta-equivalent exposureを維持するようにstockへ配分し、3銘柄×上限で届かない分はETFに残した。shortは貸株データがないためETFを維持した。",
        "",
        "| Variant | 方法 |",
        "|---|---|",
        "| etf_baseline | 現行TOPIX-17 ETF portfolio |",
        "| single_stock_hybrid | 各long sectorで流動性上位かつbeta安定条件を満たす1銘柄。shortはETF |",
        "| stock3_hybrid | 同業種の上位3銘柄へbeta-equivalent exposure配分。shortはETF |",
        "| cost_aware_stock3_hybrid | 上位K候補から最大3銘柄を列挙し、推定売買費用＋turnover penaltyを最小化。shortはETF |",
        "",
        "前提: candidate K=10、beta 252日/安定性比較126日、beta範囲0.50〜1.50、2期間beta差0.25以内、ADV 1億円以上、個別株上限はNAVの10%、Cost-aware turnover penalty 5bp。感度は各連続値±20%、K=8/12を1因子ずつ試した。beta/流動性filterと銘柄選択は常にtrade dateより前に確定した情報に限る。",
        "",
        "## 結果",
        "",
        f"対象期間は{inputs['input_summary']['target_weights_start']}〜{inputs['input_summary']['target_weights_end']}の{len(inputs['eval_dates'])}日。全候補の日次PnLを同じ日付でbaselineと対にして比較した。side_leverageは現行production設定から解決した{inputs['side_leverage']:.2f}を適用した。",
        "",
        "| Variant | Gross Sharpe | Net Sharpe | Gross total return | Net total return | Gross MDD | Net MDD | One-way turnover/day | Cost bp NAV/day | Long net contribution | Short net contribution | ΔNet bp/day vs ETF | Paired block CI95 bp/day | DSR | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, item in summary.iterrows():
        ci = item["paired_net_difference_block_bootstrap_ci95"]
        ci_text = f"[{ci[0] * 10000:.2f}, {ci[1] * 10000:.2f}]"
        dsr = item["deflated_sharpe_ratio"]
        dsr_text = "—" if dsr is None or not np.isfinite(dsr) else f"{dsr:.3f}"
        lines.append(
            f"| {item['variant']} | {item['gross_sharpe']:.3f} | {item['net_sharpe']:.3f} | "
            f"{item['gross_total_return']:.2%} | {item['net_total_return']:.2%} | "
            f"{item['gross_max_drawdown']:.2%} | {item['net_max_drawdown']:.2%} | "
            f"{item['average_one_way_turnover']:.3f} | "
            f"{item['average_estimated_cost_bps_nav_per_day']:.2f} | "
            f"{item['long_net_contribution_sum']:.3f} | {item['short_net_contribution_sum']:.3f} | "
            f"{item['paired_mean_daily_net_difference_vs_etf_bps']:.2f} | {ci_text} | {dsr_text} | {item['portfolio_candidate_gate']} |"
        )
    lines.extend([
        "",
        "費用はポートフォリオの目標weight変化に片道単価を掛けて控除した。ETFは5bp/side、株は過去OHLCから推定したhalf-spread＋同じ5bp/side。取引turnoverはweight差分の0.5×L1。コスト計算は実際の変更weight全量へper-side rateを適用。欠損した選択株の当日open-close returnはゼロとして扱い、該当weightを別途出力した。",
        "",
        f"ETF baseline: net Sharpe {base['net_sharpe']:.3f}, net MDD {base['net_max_drawdown']:.2%}, 平均turnover {base['average_one_way_turnover']:.3f}, 日次平均推定費用 {base['average_estimated_cost_bps_nav_per_day']:.2f}bp。",
        "",
        "## Sector exposure・Long/Short",
        "",
        "sector beta exposure errorは各日の「ETF weight＋stock weight×rolling beta」とNightManager target sector weightとの差。Beta-equivalent exposureは配分式で意図的に維持するが、stockの名目weight合計はbeta次第でずれるため両方を別に計測した。日次・ETF別データはsector_metrics.csvとdaily_portfolio_pnl_and_paired_differences.csv。",
        "",
        "| Variant | ETF/sector | Net PnL Δ vs ETF (bp NAV) | ΔNet bp/day | Paired block CI95 bp/day | 判定 | Cost total (return) | Turnover/day | Mean abs beta exposure gap | Nominal sector gap | Long stock beta exposure share |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ])
    core_sector = sector_summary.sort_values(["variant", "topix17_etf"])
    for _, item in core_sector.iterrows():
        lines.append(
            f"| {item['variant']} | {item['topix17_etf']} | {item['net_pnl_difference_vs_etf_bps_nav']:.1f} | "
            f"{item['paired_mean_daily_net_difference_vs_etf_bps_nav']:.2f} | "
            f"[{item['paired_net_difference_block_ci95_bps_nav_per_day'][0]:.2f}, {item['paired_net_difference_block_ci95_bps_nav_per_day'][1]:.2f}] | "
            f"{item['sector_result']} | {item['estimated_cost_return_sum']:.4f} | {item['average_one_way_turnover']:.3f} | "
            f"{item['mean_abs_beta_exposure_error']:.6f} | {item['nominal_exposure_error_mean_abs']:.6f} | "
            f"{item['long_target_stock_beta_exposure_share']:.1%} |"
        )
    lines.extend([
        "",
        "ETF別判定数:",
    ])
    for name in PRIMARY_NAMES[1:]:
        counts = sector_summary.loc[sector_summary["variant"] == name, "sector_result"].value_counts()
        lines.append(
            f"- {name}: 改善 {int(counts.get('improvement', 0))}、悪化 {int(counts.get('deterioration', 0))}、"
            f"区間が0を含む/保留 {int(counts.get('inconclusive', 0))}。"
        )
    lines.extend([
        "",
        "## 年別OOS安定性・不確実性",
        "",
        f"この期間はrolling historical simulationであり、過去に閲覧された日次データを含むretrospective pseudo-OOSである。未使用forward holdoutではない。日次Net差の95%区間は20営業日circular moving-block bootstrap、5,000 resamples、seed=20261007。DSR trial familyは事前固定した4比較と{len(specs) - 4}個の1因子感度条件、計{len(specs)}系列。DSRはportfolioの絶対Net Sharpeに対する選択補正であり、ETF比の改善を示す値ではない。未知の未登録試行を含む完全な補正でもない。",
        "",
        "年次指標はannual_portfolio_metrics.csvに、日次対応PnLとNet差はdaily_portfolio_pnl_and_paired_differences.csvに保存した。ETF別の成績を同じ時間順で確認し、特定の17セクター一律置換は想定しない。",
        "",
        "## Data / PIT / 実行条件",
        "",
        f"- 個別株は{inputs['input_summary']['stock_count']}銘柄、現行分類mapping {inputs['input_summary']['mapping_path']}。分類metadataは{inputs['input_summary']['mapping_snapshot']}。",
        f"- NightManager target weights: {inputs['input_summary']['target_weights_path']}。保存summary上のside leverageは{inputs['input_summary']['saved_side_leverage']:.2f}だが、この再価格実験ではactive production configの1.30を再適用した。weight生成そのものは再実行していない。",
        f"- signal source: {inputs['mu_audit']['matrix_dir']}。mu_gap vectors were checked against {inputs['mu_audit']['long_csv']}; max absolute difference={inputs['mu_audit']['matrix_vs_long_csv_max_abs_difference']}。利用可能予測日{inputs['mu_audit']['mu_available_days']}/{inputs['mu_audit']['weight_days']}。欠損日は株式化せずETF baseline weightsを使用。realized_target_return columnは読み込んでいない。",
        "- 個別株の09:10価格はローカルcacheにないため、株・ETF双方のprimary PnLはopen→close proxy。モデルsignal自体はNightManagerの9:10→close予測だが株側の執行時間を厳密に再現しない。",
        "- survivorship: expanded mappingは2026年現行1620銘柄で上場廃止銘柄の全履歴を網羅していない。classification PIT: 現行親TOPIX-17を全履歴に固定し、歴史的セクター変更を再構成していない。",
        "- cost limitation: ETF quote spreadは未取得で、ETFは5bp/sideだけ。stockは日足OHLC推定spread＋5bp/side。委託手数料、税、impact、borrow、financing、逆日歩は含めない。shortはETFであるため個別株貸株availabilityを仮定しない。",
        "- baseline saved weightsはproduction exact artifactを再価格したもの。gross/net returnの絶対値は既存BacktestEngine保存系列の完全な再現ではなく、同じopen-close proxy上のpaired comparison。",
        "",
        "## 前回実験・JPX33/Subsectorとの差",
        "",
        "前回のTOPIX-17 ETF stock-proxyはETF close returnとの相関/TEを最小化するETF replication testで、全ETFを一括またはtracking quality gateで置換する設計だった。今回の目的関数にETF tracking errorはなく、元の17次元NightManager signalとportfolio weightsを保持し、銘柄へβで移す。ShortはETFへ残し、cost/turnoverを下げる候補選択を評価した。",
        "",
        "Subsector/JPX33実験は予測空間・ラベルを17次元から79/33次元へ拡張し、細分予測の質や17-sectorへの集約を検証した。今回の予測モデル・信号次元は変更せず、既存17-sectorの執行instrumentだけを個別株とETFのhybridにする。したがって既存実験の再実行ではない。",
        "",
        "## 判定",
        "",
        "採用条件は、ETFより推定costが減るだけでなく、paired cost-adjusted PnL差の95%区間とDD/sector exposureを確認し、株式化した方が改善するETF/sectorに限定すること。retrospective proxy、ETF spread欠落、survivorship/PIT制約があるため、結果はresearch判断に限りproduction変更には使わない。",
        "",
        "## 再現",
        "",
        "1. repository rootで既存.venvを有効化。",
        "2. 実行: timeout -k 20 1800 .venv/bin/python src/research/scripts/experiments/experiment_nightmanager_direct_stock_alpha_20261007.py",
        "3. report: reports/20261007_nightmanager_direct_stock_alpha/report.md。tables: var/results/20261007_nightmanager_direct_stock_alpha/。",
        "4. strategy/one-factor trial summaries are appended to var/experiments/registry.jsonl.",
    ])
    (REPORT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _record_trials(
    summary: pd.DataFrame,
    specs: list[dict[str, Any]],
    all_daily: pd.DataFrame,
    started: datetime,
    ended: datetime,
) -> None:
    registry = ExperimentRegistry(default_registry_path())
    current_by_name = {item.name: item for item in registry.iter_current_records()}
    report_path = str((REPORT_DIR / "report.md").relative_to(ROOT))
    all_sharpes = summary["net_sharpe"].to_numpy(dtype=float).tolist()
    trial_count = len(all_sharpes)
    spec_by_name = {item["name"]: item for item in specs}
    for _, row in summary.iterrows():
        variant = str(row["variant"])
        daily = all_daily.loc[all_daily["variant"] == variant].sort_values("trade_date")
        returns = daily["net_return"].to_numpy(dtype=np.float64)
        metrics = {
            "gross_sharpe": float(row["gross_sharpe"]),
            "net_sharpe": float(row["net_sharpe"]),
            "metric_schema_version": "daily-v1",
            "net_sharpe_frequency": "annual",
            "trading_days_per_year": int(_read_config()["evaluation"]["annualization_sessions"]),
            "trial_count_status": "unknown",
            "max_dd": float(row["net_max_drawdown"]),
            "total_return": float(row["net_total_return"]),
            "n_observations": len(returns),
            "trials": trial_count,
            "trial_sharpes": all_sharpes,
            "returns": returns.tolist(),
            "one_way_turnover_mean": float(row["average_one_way_turnover"]),
            "estimated_cost_bps_nav_per_day": float(row["average_estimated_cost_bps_nav_per_day"]),
            "paired_mean_daily_net_difference_vs_etf_bps": float(row["paired_mean_daily_net_difference_vs_etf_bps"]),
            "paired_net_difference_block_bootstrap_ci95": row["paired_net_difference_block_bootstrap_ci95"],
            "portfolio_candidate_gate": row["portfolio_candidate_gate"],
            "decision_reason": (
                "事前portfolio gate（paired Net差CI、MDD、beta exposure）を全て満たさない"
                if variant != "etf_baseline" and row["portfolio_candidate_gate"] == "not_passed"
                else None
            ),
            "deflated_sharpe_ratio": row["deflated_sharpe_ratio"],
            "n_observations_expected": len(returns),
            "missing_return_count": 0,
            "metric_status": "valid",
        }
        spec = spec_by_name[variant]
        record = ExperimentRecord(
            name=f"{EXPERIMENT_ID}:{variant}",
            hypothesis=(
                "Map the existing NightManager TOPIX-17 sector alpha directly to liquid same-sector stocks, "
                "preserving beta-equivalent long exposure and keeping ETF shorts; ETF tracking error is not optimized."
            ),
            start_time=started,
            end_time=ended,
            study_id=EXPERIMENT_ID,
            parameters={
                "variant": variant,
                "rolling_selection": spec,
                "active_side_leverage": 1.30,
                "return_interval": "matched open-to-close proxy",
                "config_sha256": _sha256_file(CONFIG_PATH),
                "source_market_loader_config_sha256": _sha256_file(
                    ROOT / _read_config()["reuse"]["prior_market_loader_config"]
                ),
                "production_modified": False,
            },
            metrics=metrics,
            decision=(
                Decision.PENDING
                if variant == "etf_baseline"
                else Decision.REJECTED
                if row["portfolio_candidate_gate"] == "not_passed"
                else Decision.PENDING
            ),
            report_path=report_path,
            related_records=_related_prior_names(),
        )
        record.metrics["deflated_sharpe_ratio"] = compute_deflated_sharpe(record.metrics)
        existing = current_by_name.get(record.name)
        if existing is None:
            saved_record = registry.record(record)
        else:
            saved_record = registry.record_correction(existing.record_id, record)
        current_by_name[record.name] = saved_record


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    started = datetime.now(UTC)
    config = _read_config()
    inputs = _load_inputs(config)
    LOG.info(
        "Loaded %d stocks, %d evaluation dates, active side_leverage=%.2f, mu_gap available %d/%d",
        len(inputs["tickers"]), len(inputs["eval_dates"]), inputs["side_leverage"],
        inputs["mu_audit"]["mu_available_days"], inputs["mu_audit"]["weight_days"],
    )
    specs = _strategy_specs(config)
    beta_cache = _build_beta_cache(inputs, specs)
    LOG.info("Precomputed %d rolling beta-window pairs", len(beta_cache))

    outputs = []
    for spec in specs:
        LOG.info("Running strategy %s", spec["name"])
        outputs.append(_one_strategy(inputs, spec, beta_cache))
    all_daily = pd.concat([item["daily"] for item in outputs], ignore_index=True)
    all_sector = pd.concat([item["sector_daily"] for item in outputs], ignore_index=True)
    allocation_rows = pd.concat(
        [item["allocations"] for item in outputs if not item["allocations"].empty],
        ignore_index=True,
    )
    summary, paired, annual, sector_summary = _summarize(all_daily, all_sector, config)
    _write_report(inputs, specs, summary, paired, annual, sector_summary, allocation_rows)
    ended = datetime.now(UTC)
    _record_trials(summary, specs, all_daily, started, ended)
    elapsed = (ended - started).total_seconds()
    LOG.info("Wrote report/results and %d registry trials in %.1f seconds", len(specs), elapsed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
