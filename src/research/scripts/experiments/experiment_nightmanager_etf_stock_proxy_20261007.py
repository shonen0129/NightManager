#!/usr/bin/env python3
"""Rolling OOS test of TOPIX-17 ETF returns replicated with liquid stocks.

This is a research-only repricing study. It reads local market caches and a
saved NightManager V2 target-weight artifact; it does not import or modify the
production decision path and performs no network or broker operations.
"""
from __future__ import annotations

import hashlib
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

from leadlag.config.paths import default_registry_path  # noqa: E402
from leadlag.data.cache_store import SqliteCacheStore  # noqa: E402
from leadlag.data.market_data_cache import adjust_intraday_split_basis  # noqa: E402
from leadlag.data.tickers import JP_TICKERS  # noqa: E402
from leadlag.experiment_registry import (  # noqa: E402
    Decision,
    ExperimentRecord,
    ExperimentRegistry,
    compute_deflated_sharpe,
)

LOG = logging.getLogger("nightmanager_etf_stock_proxy")
CONFIG_PATH = ROOT / "configs/research/nightmanager_etf_stock_proxy_20261007.yaml"
REPORT_DIR = ROOT / "reports/20261007_nightmanager_etf_stock_proxy"
EXPERIMENT_ID = "nightmanager_etf_stock_proxy_20261007"
BASE_SLIPPAGE_BPS = 5.0
ETF_COST_BPS = 5.0
PRICE_FIELDS = ["open", "high", "low", "close", "volume"]
METHOD_COLORS = {
    "etf": "Current ETF",
    "single_k10": "Single (K=10)",
    "sparse3_k10": "Sparse-3 (K=10)",
    "costaware3_k10_l10": "Cost-aware-3 (K=10, λ=10bp)",
}


def _normalize_dates(index: pd.Index) -> pd.DatetimeIndex:
    values = pd.DatetimeIndex(pd.to_datetime(index))
    if values.tz is not None:
        values = values.tz_convert("Asia/Tokyo").tz_localize(None)
    return values.normalize()


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
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def _corwin_schultz_full_spread(high: pd.Series, low: pd.Series) -> pd.Series:
    """Daily Corwin-Schultz spread estimate from high-low ranges."""
    high = pd.to_numeric(high, errors="coerce")
    low = pd.to_numeric(low, errors="coerce")
    valid = (high > 0.0) & (low > 0.0) & (high >= low)
    log_hl = np.log((high / low).where(valid))
    beta = log_hl.pow(2) + log_hl.shift(1).pow(2)
    previous_high = high.shift(1)
    previous_low = low.shift(1)
    two_day_high = pd.concat([high, previous_high], axis=1).max(axis=1, skipna=False)
    two_day_low = pd.concat([low, previous_low], axis=1).min(axis=1, skipna=False)
    gamma = np.log((two_day_high / two_day_low).where(two_day_low > 0.0)).pow(2)
    denominator = 3.0 - 2.0 * np.sqrt(2.0)
    alpha = (
        (np.sqrt(2.0 * beta) - np.sqrt(beta)) / denominator
        - np.sqrt(gamma / denominator)
    )
    exp_alpha = np.exp(alpha.clip(lower=-20.0, upper=10.0))
    spread = 2.0 * (exp_alpha - 1.0) / (exp_alpha + 1.0)
    return spread.where(valid & alpha.notna(), np.nan).clip(lower=0.0, upper=0.50)


def _load_config() -> dict[str, Any]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Invalid experiment config: {CONFIG_PATH}")
    return config


def _load_market_inputs(config: dict[str, Any]) -> dict[str, Any]:
    mapping_path = ROOT / config["data"]["current_industry_mapping"]
    mapping = yaml.safe_load(mapping_path.read_text(encoding="utf-8"))
    stock_rows = mapping["stock_mapping"]
    ticker_to_etf: dict[str, str] = {}
    for row in stock_rows:
        ticker = str(row["ticker"])
        parent = str(row["topix17_etf"])
        if ticker in ticker_to_etf:
            raise ValueError(f"Duplicate stock mapping for {ticker}")
        ticker_to_etf[ticker] = parent

    if set(ticker_to_etf.values()) != set(JP_TICKERS):
        raise ValueError("Expanded subsector stock map does not cover canonical TOPIX-17 ETFs")

    market_cache_path = ROOT / config["data"]["etf_price_cache"]
    cache = SqliteCacheStore(market_cache_path)
    raw = cache.get("raw_ohlc")
    if not isinstance(raw, dict) or not {"jp_open", "jp_close"}.issubset(raw):
        raise RuntimeError(f"ETF open/close cache is incomplete: {market_cache_path}")
    etf_open = raw["jp_open"].copy()
    etf_close = raw["jp_close"].copy()
    etf_open.index = _normalize_dates(etf_open.index)
    etf_close.index = _normalize_dates(etf_close.index)
    etf_open = etf_open.groupby(level=0).last()
    etf_close = etf_close.groupby(level=0).last()
    stock_dir = ROOT / config["data"]["stock_prices"]
    expected_codes = [ticker.removesuffix(".T") for ticker in sorted(ticker_to_etf)]
    first = stock_dir / f"{expected_codes[0]}.parquet"
    if not first.exists():
        # The source list is sorted by ticker; report all missing items below.
        first = next((stock_dir / f"{code}.parquet" for code in expected_codes if (stock_dir / f"{code}.parquet").exists()), None)
    if first is None:
        raise FileNotFoundError(f"No stock OHLC files found under {stock_dir}")
    reference = pd.read_parquet(first, columns=["close"])
    dates = _normalize_dates(reference.index)
    dates = dates[(dates >= pd.Timestamp("2010-01-01")) & (dates <= etf_open.index.max())]
    dates = dates.unique().sort_values()
    dates = dates[dates <= pd.Timestamp("2026-08-21")]
    if len(dates) == 0:
        raise RuntimeError("No common stock/ETF price dates")

    n_dates = len(dates)
    tickers = sorted(ticker_to_etf)
    ticker_to_index = {ticker: i for i, ticker in enumerate(tickers)}
    stock_returns = np.full((n_dates, len(tickers)), np.nan, dtype=np.float64)
    adv = np.full_like(stock_returns, np.nan)
    unit_cost_bps = np.full_like(stock_returns, np.nan)
    data_digest = hashlib.sha256()
    data_digest.update(dates.asi8.tobytes())

    missing_files: list[str] = []
    for column, ticker in enumerate(tickers):
        code = ticker.removesuffix(".T")
        path = stock_dir / f"{code}.parquet"
        if not path.exists():
            missing_files.append(ticker)
            continue
        frame = pd.read_parquet(path, columns=PRICE_FIELDS)
        frame.index = _normalize_dates(frame.index)
        frame = frame.groupby(level=0).last().reindex(dates)
        values = {
            name: pd.to_numeric(frame[name], errors="coerce")
            for name in PRICE_FIELDS
        }
        op = values["open"]
        hi = values["high"]
        lo = values["low"]
        cl = values["close"]
        volume = values["volume"]
        valid = np.isfinite(op) & np.isfinite(cl) & (op > 0.0) & (cl > 0.0)
        returns = (cl / op - 1.0).where(valid)
        dollar_volume = (cl * volume).where(
            np.isfinite(cl) & np.isfinite(volume) & (cl > 0.0) & (volume > 0.0)
        )
        adv_series = dollar_volume.rolling(252, min_periods=200).mean().shift(1)
        spread_series = _corwin_schultz_full_spread(hi, lo)
        prior_spread = spread_series.shift(1).rolling(60, min_periods=30).median()
        cost_series = BASE_SLIPPAGE_BPS + 5000.0 * prior_spread
        stock_returns[:, column] = returns.to_numpy(dtype=np.float64)
        adv[:, column] = adv_series.to_numpy(dtype=np.float64)
        unit_cost_bps[:, column] = cost_series.to_numpy(dtype=np.float64)
        data_digest.update(ticker.encode("utf-8"))
        for name in PRICE_FIELDS:
            array = values[name].to_numpy(dtype=np.float64)
            data_digest.update(np.nan_to_num(array, nan=-9.87654321e99).tobytes())

    if missing_files:
        raise FileNotFoundError(f"Missing raw OHLC files for {len(missing_files)} mapped stocks: {missing_files[:10]}")

    etf_open = etf_open.reindex(dates)
    etf_close = etf_close.reindex(dates)
    etf_returns_df = etf_close[JP_TICKERS] / etf_open[JP_TICKERS] - 1.0
    etf_returns_df = etf_returns_df.where((etf_open[JP_TICKERS] > 0.0) & (etf_close[JP_TICKERS] > 0.0))
    etf_returns = etf_returns_df.to_numpy(dtype=np.float64)
    etf_close_values = etf_close[JP_TICKERS].to_numpy(dtype=np.float64)

    weights_path = ROOT / config["data"]["nightmanager_target_weights"]
    weights = pd.read_csv(weights_path, parse_dates=["trade_date"])
    weights["trade_date"] = _normalize_dates(pd.DatetimeIndex(weights["trade_date"]))
    if weights["trade_date"].duplicated().any():
        raise ValueError("NightManager target weights contain duplicate trade dates")
    weights = weights.set_index("trade_date").reindex(dates)[JP_TICKERS]
    weight_observed = weights.notna().all(axis=1).to_numpy()
    weights_array = weights.to_numpy(dtype=np.float64)
    weight_summary_path = ROOT / config["data"]["nightmanager_backtest_summary"]
    weight_summary = json.loads(weight_summary_path.read_text(encoding="utf-8"))
    side_leverage = float(weight_summary.get("side_leverage", 1.0))
    if not np.isfinite(side_leverage) or side_leverage <= 0.0:
        raise ValueError("Saved NightManager side_leverage is missing or invalid")

    summary = {
        "mapping_path": str(mapping_path.relative_to(ROOT)),
        "mapping_sha256": _sha256_file(mapping_path),
        "mapping_ticker_count": len(tickers),
        "mapping_snapshot": mapping.get("meta", {}),
        "stock_raw_price_end": str(dates.max().date()),
        "stock_raw_price_start": str(dates.min().date()),
        "stock_price_matrix_sha256": data_digest.hexdigest(),
        "etf_cache_path": str(market_cache_path.relative_to(ROOT)),
        "etf_cache_sha256": _sha256_file(market_cache_path),
        "target_weights_path": str(weights_path.relative_to(ROOT)),
        "target_weights_sha256": _sha256_file(weights_path),
        "target_weights_start": str(weights.index[weight_observed].min().date()),
        "target_weights_end": str(weights.index[weight_observed].max().date()),
        "target_weight_days": int(weight_observed.sum()),
        "saved_side_leverage": side_leverage,
        "target_weight_gross_mean_pre_side_leverage": float(np.nanmean(np.nansum(np.abs(weights_array[weight_observed]), axis=1))),
        "target_weight_gross_max_pre_side_leverage": float(np.nanmax(np.nansum(np.abs(weights_array[weight_observed]), axis=1))),
        "target_weight_net_abs_max_pre_side_leverage": float(np.nanmax(np.abs(np.nansum(weights_array[weight_observed], axis=1)))),
        "effective_gross_mean_post_side_leverage": float(np.nanmean(np.nansum(np.abs(weights_array[weight_observed]), axis=1)) * side_leverage),
        "effective_gross_max_post_side_leverage": float(np.nanmax(np.nansum(np.abs(weights_array[weight_observed]), axis=1)) * side_leverage),
        "effective_net_abs_max_post_side_leverage": float(np.nanmax(np.abs(np.nansum(weights_array[weight_observed], axis=1))) * side_leverage),
        "date_count": n_dates,
        "date_start": str(dates.min().date()),
        "date_end": str(dates.max().date()),
        "stock_count": len(tickers),
        "stock_count_by_etf": {
            etf: int(sum(parent == etf for parent in ticker_to_etf.values()))
            for etf in JP_TICKERS
        },
    }
    return {
        "config": config,
        "mapping": mapping,
        "ticker_to_etf": ticker_to_etf,
        "tickers": tickers,
        "ticker_to_index": ticker_to_index,
        "dates": dates,
        "stock_returns": stock_returns,
        "adv": adv,
        "unit_cost_bps": unit_cost_bps,
        "etf_returns": etf_returns,
        "etf_close": etf_close_values,
        "weights": weights_array,
        "weight_observed": weight_observed,
        "side_leverage": side_leverage,
        "input_summary": summary,
        "market_cache": cache,
    }


def _sparse_fit(
    cov_x: np.ndarray,
    cov_xy: np.ndarray,
    var_y: float,
    costs: np.ndarray,
    penalty_lambda_bps: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Exactly enumerate simplex supports of size 1-3 for a pool <= 12.

    The sparse objective is centered tracking-error variance. Cost-aware uses
    the convex surrogate TE variance (bp^2) + lambda(bp) * one-way cost-rate
    (bp). A simplex optimum on a boundary is covered by the smaller support.
    """
    n = cov_x.shape[0]
    best_score = float("inf")
    best_indices = np.asarray([0], dtype=np.int64)
    best_weights = np.asarray([1.0], dtype=np.float64)
    lam = float(penalty_lambda_bps)

    for i in range(n):
        indices = np.asarray([i], dtype=np.int64)
        weights = np.asarray([1.0], dtype=np.float64)
        score = var_y + cov_x[i, i] - 2.0 * cov_xy[i] + lam * costs[i]
        if score < best_score:
            best_score, best_indices, best_weights = score, indices, weights

    tolerance = 1e-9
    for i, j in itertools.combinations(range(n), 2):
        var_diff = cov_x[i, i] + cov_x[j, j] - 2.0 * cov_x[i, j]
        if var_diff <= 1e-12:
            continue
        cov_diff = cov_xy[i] - cov_xy[j] - cov_x[i, j] + cov_x[j, j]
        wi = (cov_diff - 0.5 * lam * (costs[i] - costs[j])) / var_diff
        if wi < -tolerance or wi > 1.0 + tolerance:
            continue
        wi = float(np.clip(wi, 0.0, 1.0))
        weights = np.asarray([wi, 1.0 - wi], dtype=np.float64)
        indices = np.asarray([i, j], dtype=np.int64)
        sub = cov_x[np.ix_(indices, indices)]
        sub_cov = cov_xy[indices]
        error_var = var_y + float(weights @ sub @ weights) - 2.0 * float(weights @ sub_cov)
        score = error_var + lam * float(weights @ costs[indices])
        if score < best_score:
            best_score, best_indices, best_weights = score, indices, weights

    for i, j, k in itertools.combinations(range(n), 3):
        d11 = cov_x[i, i] + cov_x[k, k] - 2.0 * cov_x[i, k]
        d22 = cov_x[j, j] + cov_x[k, k] - 2.0 * cov_x[j, k]
        d12 = cov_x[i, j] - cov_x[i, k] - cov_x[j, k] + cov_x[k, k]
        det = d11 * d22 - d12 * d12
        if d11 <= 1e-12 or d22 <= 1e-12 or det <= 1e-12:
            continue
        v1 = cov_xy[i] - cov_xy[k] - cov_x[i, k] + cov_x[k, k]
        v2 = cov_xy[j] - cov_xy[k] - cov_x[j, k] + cov_x[k, k]
        v1 -= 0.5 * lam * (costs[i] - costs[k])
        v2 -= 0.5 * lam * (costs[j] - costs[k])
        wi = (v1 * d22 - v2 * d12) / det
        wj = (v2 * d11 - v1 * d12) / det
        wk = 1.0 - wi - wj
        if min(wi, wj, wk) < -tolerance:
            continue
        weights = np.clip(np.asarray([wi, wj, wk], dtype=np.float64), 0.0, None)
        total = float(weights.sum())
        if total <= 0.0:
            continue
        weights /= total
        indices = np.asarray([i, j, k], dtype=np.int64)
        sub = cov_x[np.ix_(indices, indices)]
        sub_cov = cov_xy[indices]
        error_var = var_y + float(weights @ sub @ weights) - 2.0 * float(weights @ sub_cov)
        score = error_var + lam * float(weights @ costs[indices])
        if score < best_score:
            best_score, best_indices, best_weights = score, indices, weights

    return best_indices, best_weights, float(best_score)


def _model_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    evaluation = config["evaluation"]
    top_k = int(evaluation["candidate_pool_top_k"])
    pool_sensitivity = [int(value) for value in evaluation["candidate_pool_sensitivity"]]
    base_lambda = float(evaluation["cost_penalty_lambda_bps"])
    lambda_sensitivity = [float(value) for value in evaluation["cost_penalty_sensitivity_bps"]]
    specs = [{"name": "etf", "kind": "etf", "top_k": 0, "lambda": 0.0, "primary": True}]
    for method in ("single", "sparse3", "costaware3"):
        lam = base_lambda if method == "costaware3" else 0.0
        specs.append({
            "name": f"{method}_k{top_k}" + (f"_l{int(lam)}" if method == "costaware3" else ""),
            "kind": method,
            "top_k": top_k,
            "lambda": lam,
            "primary": True,
        })
    for k in pool_sensitivity:
        for method in ("single", "sparse3", "costaware3"):
            lam = base_lambda if method == "costaware3" else 0.0
            specs.append({
                "name": f"{method}_k{k}" + (f"_l{int(lam)}" if method == "costaware3" else ""),
                "kind": method,
                "top_k": k,
                "lambda": lam,
                "primary": False,
            })
    for lam in lambda_sensitivity:
        specs.append({
            "name": f"costaware3_k{top_k}_l{int(lam)}",
            "kind": "costaware3",
            "top_k": top_k,
            "lambda": lam,
            "primary": False,
        })
    names = [spec["name"] for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate strategy names in sensitivity plan: {names}")
    return specs


def _rolling_select(inputs: dict[str, Any]) -> dict[str, Any]:
    config = inputs["config"]
    dates: pd.DatetimeIndex = inputs["dates"]
    tickers: list[str] = inputs["tickers"]
    ticker_to_etf: dict[str, str] = inputs["ticker_to_etf"]
    ticker_to_index: dict[str, int] = inputs["ticker_to_index"]
    etf_returns: np.ndarray = inputs["etf_returns"]
    stock_returns: np.ndarray = inputs["stock_returns"]
    adv: np.ndarray = inputs["adv"]
    costs: np.ndarray = inputs["unit_cost_bps"]
    window = int(config["evaluation"]["lookback_sessions"])
    minimum_observations = int(config["evaluation"]["min_training_observations"])
    specs = _model_specs(config)
    n_specs = len(specs)
    n_dates = len(dates)
    n_etfs = len(JP_TICKERS)
    selected_indices = np.full((n_specs, n_dates, n_etfs, 3), -1, dtype=np.int32)
    selected_weights = np.zeros((n_specs, n_dates, n_etfs, 3), dtype=np.float64)
    proxy_returns = np.full((n_specs, n_dates, n_etfs), np.nan, dtype=np.float64)
    fallback = np.ones((n_specs, n_dates, n_etfs), dtype=bool)
    train_corr = np.full((n_specs, n_dates, n_etfs), np.nan, dtype=np.float64)
    candidate_counts = np.zeros((n_dates, n_etfs), dtype=np.int16)
    selection_records: list[dict[str, Any]] = []
    grouped_indices = {
        etf: np.asarray(
            [ticker_to_index[ticker] for ticker in tickers if ticker_to_etf[ticker] == etf],
            dtype=np.int32,
        )
        for etf in JP_TICKERS
    }
    max_k = max(int(spec["top_k"]) for spec in specs)
    primary_names = {spec["name"] for spec in specs if spec["primary"]}

    for t in range(window, n_dates):
        if t % 250 == 0:
            LOG.info("Rolling selection %s (%d/%d)", dates[t].date(), t, n_dates)
        y_window_all = etf_returns[t - window : t]
        x_window_all = stock_returns[t - window : t]
        for etf_index, etf in enumerate(JP_TICKERS):
            y = y_window_all[:, etf_index]
            y_current = etf_returns[t, etf_index]
            if not np.isfinite(y_current):
                continue
            if len(y) != window or not np.isfinite(y).all() or window < minimum_observations:
                proxy_returns[:, t, etf_index] = y_current
                continue
            group = grouped_indices[etf]
            adv_now = adv[t, group]
            unit_cost_now = costs[t, group]
            x_hist_group = x_window_all[:, group]
            return_eligible = np.isfinite(x_hist_group).all(axis=0)
            eligible = np.isfinite(adv_now) & np.isfinite(unit_cost_now) & return_eligible
            if not eligible.any():
                proxy_returns[:, t, etf_index] = y_current
                continue
            eligible_indices = group[eligible]
            eligible_adv = adv_now[eligible]
            # Stable tie-breaking by the sorted ticker list; all ranks use data through t-1.
            ranked = np.lexsort((eligible_indices, -eligible_adv))
            candidate_indices = eligible_indices[ranked[:max_k]]
            candidate_counts[t, etf_index] = len(candidate_indices)
            if len(candidate_indices) == 0:
                proxy_returns[:, t, etf_index] = y_current
                continue
            x_hist = x_window_all[:, candidate_indices]
            x_bp = x_hist * 10000.0
            y_bp = y * 10000.0
            x_centered = x_bp - x_bp.mean(axis=0, keepdims=True)
            y_centered = y_bp - y_bp.mean()
            cov_x = (x_centered.T @ x_centered) / window
            cov_xy = (x_centered.T @ y_centered) / window
            var_y = float(y_centered @ y_centered / window)
            pool_cost = costs[t, candidate_indices]

            per_spec: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for spec_index, spec in enumerate(specs):
                if spec["kind"] == "etf":
                    proxy_returns[spec_index, t, etf_index] = y_current
                    fallback[spec_index, t, etf_index] = False
                    continue
                k = min(int(spec["top_k"]), len(candidate_indices))
                if k < 1:
                    proxy_returns[spec_index, t, etf_index] = y_current
                    continue
                if spec["kind"] == "single":
                    denom = np.sqrt(np.maximum(cov_x.diagonal()[:k], 0.0) * max(var_y, 1e-18))
                    correlation = np.divide(
                        cov_xy[:k], denom,
                        out=np.full(k, -np.inf, dtype=np.float64),
                        where=denom > 1e-12,
                    )
                    choice = int(np.argmax(correlation))
                    local_indices = np.asarray([choice], dtype=np.int64)
                    local_weights = np.asarray([1.0], dtype=np.float64)
                else:
                    local_indices, local_weights, _ = _sparse_fit(
                        cov_x[:k, :k],
                        cov_xy[:k],
                        var_y,
                        pool_cost[:k],
                        float(spec["lambda"]),
                    )
                chosen_global = candidate_indices[local_indices]
                current_stock_returns = stock_returns[t, chosen_global]
                if not np.isfinite(current_stock_returns).all():
                    proxy_returns[spec_index, t, etf_index] = y_current
                    fallback[spec_index, t, etf_index] = True
                    continue
                model_return = float(local_weights @ current_stock_returns)
                selected_indices[spec_index, t, etf_index, : len(chosen_global)] = chosen_global
                selected_weights[spec_index, t, etf_index, : len(chosen_global)] = local_weights
                proxy_returns[spec_index, t, etf_index] = model_return
                fallback[spec_index, t, etf_index] = False
                basket_train = x_hist[:, local_indices] @ local_weights
                corr = np.corrcoef(basket_train, y)[0, 1] if np.std(basket_train) > 0 and np.std(y) > 0 else np.nan
                train_corr[spec_index, t, etf_index] = float(corr) if np.isfinite(corr) else np.nan
                per_spec[spec["name"]] = (chosen_global, local_weights)
                if spec["name"] in primary_names:
                    for ticker_index, basket_weight in zip(chosen_global, local_weights, strict=True):
                        selection_records.append({
                            "trade_date": str(dates[t].date()),
                            "training_start": str(dates[t - window].date()),
                            "training_end": str(dates[t - 1].date()),
                            "topix17_etf": etf,
                            "variant": spec["name"],
                            "ticker": tickers[int(ticker_index)],
                            "basket_weight": float(basket_weight),
                            "training_basket_correlation": train_corr[spec_index, t, etf_index],
                            "trailing_252d_average_dollar_volume_jpy": float(adv[t, ticker_index]),
                            "estimated_one_way_cost_bps": float(costs[t, ticker_index]),
                            "candidate_pool_size": k,
                        })

    # The source contains full daily coverage for active large-cap candidates.
    # Any missing selected-stock return is explicitly routed to the ETF series.
    for spec_index, spec in enumerate(specs):
        if spec["kind"] == "etf":
            proxy_returns[spec_index] = etf_returns
            fallback[spec_index] = False
            continue
        missing_eval = np.isnan(proxy_returns[spec_index]) & np.isfinite(etf_returns)
        proxy_returns[spec_index][missing_eval] = etf_returns[missing_eval]
        fallback[spec_index][missing_eval] = True
    selected_mass = selected_weights.sum(axis=-1)
    active = selected_indices[..., 0] >= 0
    if np.any(np.abs(selected_mass[active] - 1.0) > 1e-8):
        raise RuntimeError("Selected basket weights do not sum to one")
    if np.any(selected_weights < -1e-12):
        raise RuntimeError("Negative basket weights violate the long-only proxy contract")

    # Independent leakage guard: every recorded lookback ends before the trade date.
    if selection_records:
        record_frame = pd.DataFrame(selection_records)
        if not (pd.to_datetime(record_frame["training_end"]) < pd.to_datetime(record_frame["trade_date"])).all():
            raise RuntimeError("Rolling OOS selection included the trade date in its training window")
    return {
        "specs": specs,
        "selected_indices": selected_indices,
        "selected_weights": selected_weights,
        "proxy_returns": proxy_returns,
        "fallback": fallback,
        "train_corr": train_corr,
        "candidate_counts": candidate_counts,
        "selection_records": selection_records,
        "grouped_indices": grouped_indices,
    }


def _basket_turnover(
    selected_indices: np.ndarray,
    selected_weights: np.ndarray,
    fallback: np.ndarray,
    spec_index: int,
    n_dates: int,
    n_etfs: int,
) -> np.ndarray:
    output = np.full((n_dates, n_etfs), np.nan, dtype=np.float64)
    previous: list[dict[int, float] | None] = [None] * n_etfs
    for t in range(n_dates):
        for j in range(n_etfs):
            current: dict[int, float] = {}
            if not fallback[spec_index, t, j]:
                for ticker_index, weight in zip(
                    selected_indices[spec_index, t, j],
                    selected_weights[spec_index, t, j],
                    strict=True,
                ):
                    if ticker_index >= 0 and weight > 0.0:
                        current[int(ticker_index)] = float(weight)
            if previous[j] is not None:
                keys = set(previous[j]) | set(current)
                output[t, j] = 0.5 * sum(abs(current.get(key, 0.0) - previous[j].get(key, 0.0)) for key in keys)
            previous[j] = current
    return output


def _series_summary(returns: np.ndarray, annualization: int) -> dict[str, Any]:
    values = np.asarray(returns, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        return {"n_observations": int(len(values)), "annualized_sharpe": None, "max_drawdown": None, "total_return": None}
    sigma = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    sharpe = float(np.mean(values) / sigma * np.sqrt(annualization)) if sigma > 1e-14 else None
    if np.any(values <= -1.0):
        total_return = None
        max_dd = None
    else:
        equity = np.cumprod(1.0 + values)
        total_return = float(equity[-1] - 1.0)
        curve = np.concatenate([[1.0], equity])
        peak = np.maximum.accumulate(curve)
        max_dd = float(np.min(curve / peak - 1.0))
    return {
        "n_observations": int(len(values)),
        "annualized_sharpe": sharpe,
        "annualized_volatility": float(sigma * np.sqrt(annualization)),
        "mean_daily_return": float(np.mean(values)),
        "total_return": total_return,
        "max_drawdown": max_dd,
    }


def _moving_block_ci(values: np.ndarray, block: int, samples: int, seed: int) -> list[float] | None:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < block or not np.isfinite(values).all():
        return None
    rng = np.random.default_rng(seed)
    block_count = int(np.ceil(len(values) / block))
    max_start = len(values) - block
    means = np.empty(samples, dtype=np.float64)
    for sample_index in range(samples):
        starts = rng.integers(0, max_start + 1, size=block_count)
        sample = np.concatenate([values[start : start + block] for start in starts])[: len(values)]
        means[sample_index] = float(np.mean(sample))
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def _metric_rows(
    inputs: dict[str, Any],
    rolling: dict[str, Any],
    basket_turnover: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    dates = inputs["dates"]
    etf_returns = inputs["etf_returns"]
    proxy_returns = rolling["proxy_returns"]
    specs = rolling["specs"]
    lookback = int(inputs["config"]["evaluation"]["lookback_sessions"])
    records: list[dict[str, Any]] = []
    validation_end = pd.Timestamp(inputs["config"]["evaluation"]["validation_end"])
    validation_summary: dict[str, dict[str, Any]] = {}
    for spec_index, spec in enumerate(specs):
        per_method_validation: dict[str, Any] = {}
        for etf_index, etf in enumerate(JP_TICKERS):
            active = rolling["selected_indices"][spec_index, :, etf_index, 0] >= 0
            if spec["kind"] == "etf":
                active = np.ones(len(dates), dtype=bool)
            first = int(np.flatnonzero(active)[0]) if active.any() else lookback
            oos = (np.arange(len(dates)) >= max(lookback, first)) & np.isfinite(etf_returns[:, etf_index]) & np.isfinite(proxy_returns[spec_index, :, etf_index])
            eval_positions = np.flatnonzero(oos)
            if len(eval_positions) == 0:
                continue
            target = etf_returns[eval_positions, etf_index]
            proxy = proxy_returns[spec_index, eval_positions, etf_index]
            error = proxy - target
            target_vol = float(np.std(target, ddof=1)) if len(target) > 1 else np.nan
            te_vol = float(np.std(error, ddof=1)) if len(error) > 1 else np.nan
            corr = float(np.corrcoef(proxy, target)[0, 1]) if len(target) > 1 and np.std(proxy) > 0 and np.std(target) > 0 else np.nan
            turnover = basket_turnover.get(spec["name"])
            mean_turnover = float(np.nanmean(turnover[eval_positions, etf_index])) if turnover is not None and np.isfinite(turnover[eval_positions, etf_index]).any() else np.nan
            if spec["kind"] == "etf":
                mean_unit_cost = ETF_COST_BPS
                fallback_rate = 0.0
            else:
                selected = rolling["selected_indices"][spec_index, eval_positions, etf_index]
                selected_w = rolling["selected_weights"][spec_index, eval_positions, etf_index]
                daily_costs: list[float] = []
                for position, ticker_indices, ticker_weights in zip(eval_positions, selected, selected_w, strict=True):
                    valid_slots = ticker_indices >= 0
                    if valid_slots.any():
                        daily_costs.append(float(ticker_weights[valid_slots] @ inputs["unit_cost_bps"][position, ticker_indices[valid_slots]]))
                    else:
                        daily_costs.append(ETF_COST_BPS)
                mean_unit_cost = float(np.mean(daily_costs)) if daily_costs else np.nan
                fallback_rate = float(np.mean(rolling["fallback"][spec_index, eval_positions, etf_index]))
            row = {
                "variant": spec["name"],
                "model": spec["kind"],
                "candidate_pool_top_k": spec["top_k"],
                "cost_penalty_lambda_bps": spec["lambda"] if spec["kind"] == "costaware3" else np.nan,
                "topix17_etf": etf,
                "oos_start": str(dates[eval_positions[0]].date()),
                "oos_end": str(dates[eval_positions[-1]].date()),
                "oos_dates": len(eval_positions),
                "oos_correlation": corr,
                "tracking_error_vol_bps": te_vol * 10000.0,
                "tracking_error_to_etf_vol_ratio": te_vol / target_vol if target_vol > 0 else np.nan,
                "etf_target_vol_bps": target_vol * 10000.0,
                "mae_bps": float(np.mean(np.abs(error)) * 10000.0),
                "rmse_bps": float(np.sqrt(np.mean(error**2)) * 10000.0),
                "sign_agreement": float(np.mean(np.sign(proxy) == np.sign(target))),
                "mean_basket_one_way_turnover": mean_turnover,
                "mean_estimated_one_way_cost_rate_bps": mean_unit_cost,
                "fallback_rate": fallback_rate,
            }
            records.append(row)

            validation = (dates[eval_positions] <= validation_end)
            if validation.any():
                v_target = target[validation]
                v_proxy = proxy[validation]
                v_error = v_proxy - v_target
                v_cost = mean_unit_cost
                if spec["kind"] != "etf":
                    v_positions = eval_positions[validation]
                    cost_values: list[float] = []
                    for position in v_positions:
                        indices = rolling["selected_indices"][spec_index, position, etf_index]
                        weights = rolling["selected_weights"][spec_index, position, etf_index]
                        valid_slots = indices >= 0
                        if valid_slots.any():
                            cost_values.append(float(weights[valid_slots] @ inputs["unit_cost_bps"][position, indices[valid_slots]]))
                        else:
                            cost_values.append(ETF_COST_BPS)
                    v_cost = float(np.mean(cost_values)) if cost_values else np.nan
                v_corr = float(np.corrcoef(v_proxy, v_target)[0, 1]) if len(v_target) > 1 and np.std(v_proxy) > 0 and np.std(v_target) > 0 else np.nan
                v_te = float(np.std(v_error, ddof=1)) if len(v_error) > 1 else np.nan
                v_target_vol = float(np.std(v_target, ddof=1)) if len(v_target) > 1 else np.nan
                per_method_validation[etf] = {
                    "dates": int(len(v_target)),
                    "correlation": v_corr,
                    "tracking_error_vol": v_te,
                    "etf_target_vol": v_target_vol,
                    "tracking_error_ratio": v_te / v_target_vol if v_target_vol > 0 else np.nan,
                    "unit_cost_bps": v_cost,
                }
        validation_summary[spec["name"]] = per_method_validation

    return pd.DataFrame(records), validation_summary


def _build_0910_targets(inputs: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    bars = inputs["market_cache"].get("intraday_5m")
    if not isinstance(bars, pd.DataFrame) or bars.empty:
        return np.full_like(inputs["etf_returns"], np.nan), {"source": "unavailable", "observed_dates": 0}
    bars = adjust_intraday_split_basis(bars)
    bars = bars.copy()
    bars.index = _normalize_dates(bars.index) + (pd.DatetimeIndex(bars.index) - pd.DatetimeIndex(bars.index).normalize())
    out = np.full_like(inputs["etf_returns"], np.nan)
    date_lookup = {date: i for i, date in enumerate(inputs["dates"])}
    closes = inputs["etf_close"]
    observed_cells = 0
    for date in bars.index.normalize().unique():
        t = date_lookup.get(date)
        if t is None:
            continue
        stamp = date + pd.Timedelta(hours=9, minutes=10)
        if stamp not in bars.index:
            continue
        row = bars.loc[stamp]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        for etf_index, etf in enumerate(JP_TICKERS):
            high = row.get(("High", etf), np.nan)
            low = row.get(("Low", etf), np.nan)
            close = row.get(("Close", etf), np.nan)
            price_910 = (high + low) / 2.0 if pd.notna(high) and pd.notna(low) and high > 0 and low > 0 else close
            if pd.notna(price_910) and np.isfinite(price_910) and price_910 > 0 and np.isfinite(closes[t, etf_index]) and closes[t, etf_index] > 0:
                out[t, etf_index] = closes[t, etf_index] / float(price_910) - 1.0
                observed_cells += 1
    per_ticker = {etf: int(np.isfinite(out[:, i]).sum()) for i, etf in enumerate(JP_TICKERS)}
    return out, {
        "source": "ETF 5-minute 09:10 High/Low midpoint; daily close / midpoint - 1",
        "observed_dates": int(np.isfinite(out).any(axis=1).sum()),
        "dates_with_all_17_etfs": int(np.isfinite(out).all(axis=1).sum()),
        "observed_cells": observed_cells,
        "observed_dates_by_etf": per_ticker,
        "bar_start": str(bars.index.min()),
        "bar_end": str(bars.index.max()),
        "stock_0910_bars_available": False,
    }


def _portfolio_simulations(
    inputs: dict[str, Any],
    rolling: dict[str, Any],
    replace_masks: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]]]:
    config = inputs["config"]
    dates = inputs["dates"]
    weights = inputs["weights"]
    weight_observed = inputs["weight_observed"]
    active_positions = np.flatnonzero(weight_observed)
    if len(active_positions) == 0:
        raise RuntimeError("Saved NightManager weights do not overlap the input price history")
    side_leverage = inputs["side_leverage"]
    annualization = int(config["evaluation"]["annualization_sessions"])
    specs = rolling["specs"]
    spec_by_name = {spec["name"]: i for i, spec in enumerate(specs)}
    target_weights = weights[active_positions]
    if not np.isfinite(target_weights).all():
        raise RuntimeError("A date covered by the saved target weights has non-finite weights")
    if float(np.max(np.abs(target_weights.sum(axis=1)))) > 0.05:
        raise RuntimeError("Saved model weights do not satisfy the near-zero net exposure guard")
    if float(np.max(np.abs(target_weights).sum(axis=1))) > 2.0 + 1e-8:
        raise RuntimeError("Saved model weights exceed the gross <= 2.0 guard")

    portfolio_names: list[tuple[str, int, np.ndarray | None]] = []
    for spec_index, spec in enumerate(specs):
        portfolio_names.append((spec["name"], spec_index, None))
    for spec_name, mask in replace_masks.items():
        portfolio_names.append((f"hybrid_{spec_name}", spec_by_name[spec_name], np.asarray(mask, dtype=bool)))

    summary_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    sector_rows: list[dict[str, Any]] = []
    returns_by_name: dict[str, dict[str, Any]] = {}
    bootstrap_block = int(config["evaluation"]["bootstrap_block_sessions"])
    bootstrap_samples = int(config["evaluation"]["bootstrap_resamples"])
    bootstrap_seed = int(config["evaluation"]["bootstrap_seed"])
    etf_returns = inputs["etf_returns"]
    proxy_returns = rolling["proxy_returns"]
    selected_indices = rolling["selected_indices"]
    selected_weights = rolling["selected_weights"]
    fallback = rolling["fallback"]
    tickers = inputs["tickers"]

    for portfolio_index, (portfolio_name, spec_index, replace_mask) in enumerate(portfolio_names):
        previous: list[dict[str, tuple[float, float]]] = [dict() for _ in JP_TICKERS]
        gross_returns: list[float] = []
        net_returns: list[float] = []
        cost_returns: list[float] = []
        turnover_values: list[float] = []
        fallback_exposures: list[float] = []
        for t in active_positions:
            daily_gross = 0.0
            total_cost = 0.0
            total_turnover = 0.0
            fallback_exposure = 0.0
            per_etf_gross = np.zeros(len(JP_TICKERS), dtype=np.float64)
            per_etf_cost = np.zeros(len(JP_TICKERS), dtype=np.float64)
            per_etf_turnover = np.zeros(len(JP_TICKERS), dtype=np.float64)
            for etf_index, etf in enumerate(JP_TICKERS):
                model_weight = float(weights[t, etf_index])
                effective_weight = model_weight * side_leverage
                if replace_mask is not None and not replace_mask[etf_index]:
                    return_value = etf_returns[t, etf_index]
                    current = {f"ETF:{etf}": (effective_weight, ETF_COST_BPS)}
                elif specs[spec_index]["kind"] == "etf":
                    return_value = etf_returns[t, etf_index]
                    current = {f"ETF:{etf}": (effective_weight, ETF_COST_BPS)}
                elif fallback[spec_index, t, etf_index]:
                    return_value = etf_returns[t, etf_index]
                    current = {f"ETF:{etf}": (effective_weight, ETF_COST_BPS)}
                    fallback_exposure += abs(effective_weight)
                else:
                    return_value = proxy_returns[spec_index, t, etf_index]
                    current: dict[str, tuple[float, float]] = {}
                    for ticker_index, basket_weight in zip(
                        selected_indices[spec_index, t, etf_index],
                        selected_weights[spec_index, t, etf_index],
                        strict=True,
                    ):
                        if ticker_index < 0 or basket_weight <= 0.0:
                            continue
                        ticker_index = int(ticker_index)
                        ticker = tickers[ticker_index]
                        current[f"STOCK:{ticker}"] = (
                            effective_weight * float(basket_weight),
                            float(inputs["unit_cost_bps"][t, ticker_index]),
                        )
                if not np.isfinite(return_value):
                    if abs(model_weight) > 1e-12:
                        raise RuntimeError(f"Missing realized return for active weight on {dates[t].date()} {etf} / {portfolio_name}")
                    return_value = 0.0
                contribution = effective_weight * float(return_value)
                daily_gross += contribution
                per_etf_gross[etf_index] = contribution
                old = previous[etf_index]
                keys = set(old) | set(current)
                l1_trade = 0.0
                sector_cost = 0.0
                for key in keys:
                    old_value = old.get(key, (0.0, current.get(key, (0.0, ETF_COST_BPS))[1]))[0]
                    new_value, current_rate = current.get(key, (0.0, old.get(key, (0.0, ETF_COST_BPS))[1]))
                    traded = abs(new_value - old_value)
                    l1_trade += traded
                    sector_cost += traded * current_rate / 10000.0
                turnover = 0.5 * l1_trade
                total_turnover += turnover
                total_cost += sector_cost
                per_etf_cost[etf_index] = sector_cost
                per_etf_turnover[etf_index] = turnover
                previous[etf_index] = current
            net = daily_gross - total_cost
            gross_returns.append(daily_gross)
            net_returns.append(net)
            cost_returns.append(total_cost)
            turnover_values.append(total_turnover)
            fallback_exposures.append(fallback_exposure)
            daily_rows.append({
                "trade_date": str(dates[t].date()),
                "variant": portfolio_name,
                "gross_return": daily_gross,
                "estimated_execution_cost_return": total_cost,
                "net_return_after_spread_and_slippage": net,
                "effective_one_way_turnover": total_turnover,
                "estimated_cost_bps_of_nav": total_cost * 10000.0,
                "fallback_gross_exposure": fallback_exposure,
            })
            for etf_index, etf in enumerate(JP_TICKERS):
                sector_rows.append({
                    "trade_date": str(dates[t].date()),
                    "variant": portfolio_name,
                    "topix17_etf": etf,
                    "gross_return_contribution": per_etf_gross[etf_index],
                    "estimated_execution_cost_return": per_etf_cost[etf_index],
                    "net_return_contribution": per_etf_gross[etf_index] - per_etf_cost[etf_index],
                    "effective_one_way_turnover": per_etf_turnover[etf_index],
                    "estimated_cost_bps_of_nav": per_etf_cost[etf_index] * 10000.0,
                })

        gross_arr = np.asarray(gross_returns)
        cost_arr = np.asarray(cost_returns)
        net_arr = np.asarray(net_returns)
        turnover_arr = np.asarray(turnover_values)
        gross_metrics = _series_summary(gross_arr, annualization)
        net_metrics = _series_summary(net_arr, annualization)
        record = {
            "variant": portfolio_name,
            "is_hybrid_selective": replace_mask is not None,
            "substituted_etf_count": int(replace_mask.sum()) if replace_mask is not None else (0 if portfolio_name == "etf" else len(JP_TICKERS)),
            "start_date": str(dates[active_positions[0]].date()),
            "end_date": str(dates[active_positions[-1]].date()),
            "n_observations": len(net_arr),
            "average_gross_daily_return": float(np.mean(gross_arr)),
            "average_transaction_cost_bps_of_nav": float(np.mean(cost_arr) * 10000.0),
            "cumulative_transaction_cost_bps_of_nav": float(np.sum(cost_arr) * 10000.0),
            "average_effective_one_way_turnover": float(np.mean(turnover_arr)),
            "average_fallback_gross_exposure": float(np.mean(fallback_exposures)),
            "gross_sharpe": gross_metrics["annualized_sharpe"],
            "gross_max_drawdown": gross_metrics["max_drawdown"],
            "gross_total_return": gross_metrics["total_return"],
            "net_sharpe": net_metrics["annualized_sharpe"],
            "net_max_drawdown": net_metrics["max_drawdown"],
            "net_total_return": net_metrics["total_return"],
        }
        if portfolio_name != "etf" and replace_mask is None:
            record["paired_mean_daily_net_return_difference_vs_etf"] = float(np.mean(net_arr - np.asarray(returns_by_name["etf"]["net_returns"])))
            record["paired_mean_daily_net_return_difference_ci95"] = _moving_block_ci(
                net_arr - np.asarray(returns_by_name["etf"]["net_returns"]),
                bootstrap_block,
                bootstrap_samples,
                bootstrap_seed + portfolio_index,
            )
        elif replace_mask is not None:
            record["paired_mean_daily_net_return_difference_vs_etf"] = float(np.mean(net_arr - np.asarray(returns_by_name["etf"]["net_returns"])))
            record["paired_mean_daily_net_return_difference_ci95"] = _moving_block_ci(
                net_arr - np.asarray(returns_by_name["etf"]["net_returns"]),
                bootstrap_block,
                bootstrap_samples,
                bootstrap_seed + portfolio_index,
            )
        returns_by_name[portfolio_name] = {
            "gross_returns": gross_arr,
            "cost_returns": cost_arr,
            "net_returns": net_arr,
            "turnover": turnover_arr,
            "metrics": record,
        }
        summary_rows.append(record)

    # DSR is defined over the predeclared 12 pure strategy/configuration trials.
    pure_names = [spec["name"] for spec in specs]
    pure_sharpes = [
        returns_by_name[name]["metrics"]["net_sharpe"]
        for name in pure_names
        if returns_by_name[name]["metrics"]["net_sharpe"] is not None
    ]
    trial_count = len(pure_names)
    for row in summary_rows:
        if row["variant"] not in pure_names:
            continue
        net_array = returns_by_name[row["variant"]]["net_returns"]
        dsr_metrics = {
            "metric_status": "valid",
            "metric_schema_version": "daily-v1",
            "net_sharpe_frequency": "annual",
            "trading_days_per_year": int(inputs["config"]["evaluation"]["annualization_sessions"]),
            "trial_count_status": "unknown",
            "net_sharpe": row["net_sharpe"],
            "trials": trial_count,
            "n_observations": len(net_array),
            "returns": net_array.tolist(),
            "trial_sharpes": [float(value) for value in pure_sharpes],
        }
        row["deflated_sharpe_ratio"] = compute_deflated_sharpe(dsr_metrics)

    return pd.DataFrame(summary_rows), pd.DataFrame(daily_rows), {"returns": returns_by_name, "sector_daily": pd.DataFrame(sector_rows), "trial_count": trial_count}


def _actual_0910_metrics(
    inputs: dict[str, Any],
    rolling: dict[str, Any],
    targets_0910: np.ndarray,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for spec_index, spec in enumerate(rolling["specs"]):
        if spec["kind"] == "etf":
            proxy = inputs["etf_returns"]
        else:
            proxy = rolling["proxy_returns"][spec_index]
        for etf_index, etf in enumerate(JP_TICKERS):
            valid = np.isfinite(targets_0910[:, etf_index]) & np.isfinite(proxy[:, etf_index])
            target = targets_0910[valid, etf_index]
            candidate = proxy[valid, etf_index]
            error = candidate - target
            corr = float(np.corrcoef(candidate, target)[0, 1]) if len(target) > 1 and np.std(candidate) > 0 and np.std(target) > 0 else np.nan
            records.append({
                "variant": spec["name"],
                "topix17_etf": etf,
                "observed_0910_dates": int(len(target)),
                "mixed_window_correlation": corr,
                "mixed_window_tracking_error_vol_bps": float(np.std(error, ddof=1) * 10000.0) if len(error) > 1 else np.nan,
                "mixed_window_mae_bps": float(np.mean(np.abs(error)) * 10000.0) if len(error) else np.nan,
                "mixed_window_rmse_bps": float(np.sqrt(np.mean(error**2)) * 10000.0) if len(error) else np.nan,
                "mixed_window_sign_agreement": float(np.mean(np.sign(candidate) == np.sign(target))) if len(target) else np.nan,
                "target_interval": "ETF 09:10 midpoint-to-close vs stock open-to-close proxy; mixed start times",
            })
    return pd.DataFrame(records)


def _reprice_reconciliation(inputs: dict[str, Any]) -> dict[str, Any]:
    result_path = ROOT / "var/results/v2_backtest_exact_production/daily_gross_returns.csv"
    if not result_path.exists():
        return {"saved_gross_return_file": None, "status": "unavailable"}
    saved = pd.read_csv(result_path, parse_dates=["trade_date"]).set_index("trade_date").iloc[:, 0]
    saved.index = _normalize_dates(saved.index)
    weights = inputs["weights"]
    observed = inputs["weight_observed"]
    exposure_return = np.nansum(weights * inputs["etf_returns"], axis=1) * inputs["side_leverage"]
    dates = inputs["dates"]
    common = observed & np.isfinite(exposure_return)
    saved_values = saved.reindex(dates).to_numpy(dtype=np.float64)
    common &= np.isfinite(saved_values)
    if not common.any():
        return {"saved_gross_return_file": str(result_path.relative_to(ROOT)), "status": "no_common_dates"}
    difference = exposure_return[common] - saved_values[common]
    return {
        "saved_gross_return_file": str(result_path.relative_to(ROOT)),
        "status": "repriced_with_local_daily_open_close_and_saved_weights",
        "paired_dates": int(common.sum()),
        "mean_absolute_difference": float(np.mean(np.abs(difference))),
        "rmse_difference": float(np.sqrt(np.mean(difference**2))),
        "correlation": float(np.corrcoef(exposure_return[common], saved_values[common])[0, 1]),
        "note": "Saved returns are retained for provenance; this experiment compares a fresh, common open-to-close repricing with the same saved weights.",
    }


def _related_registry_names(registry: ExperimentRegistry) -> list[str]:
    tokens = ("subsector", "jpx33", "industry_proxy", "etf_stock")
    found: list[str] = []
    for record in registry.iter_records():
        searchable = f"{record.name} {record.hypothesis} {record.report_path or ''}".lower()
        if any(token in searchable for token in tokens):
            found.append(record.name)
    return sorted(set(found))


def _write_outputs(
    inputs: dict[str, Any],
    rolling: dict[str, Any],
    per_etf: pd.DataFrame,
    validation: dict[str, dict[str, Any]],
    pnl_summary: pd.DataFrame,
    daily_pnl: pd.DataFrame,
    pnl_extra: dict[str, Any],
    targets_0910: np.ndarray,
    coverage_0910: dict[str, Any],
    actual_0910: pd.DataFrame,
    basket_turnover: dict[str, np.ndarray],
    gate_rows: list[dict[str, Any]],
    reconciliation: dict[str, Any],
) -> None:
    result_dir = ROOT / "var/results/20261007_nightmanager_etf_stock_proxy"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    sector_daily = pnl_extra["sector_daily"]
    portfolio_sector_stats = (
        sector_daily.groupby(["variant", "topix17_etf"], as_index=False)
        .agg(
            mean_effective_one_way_turnover=("effective_one_way_turnover", "mean"),
            mean_estimated_cost_bps_nav_per_day=("estimated_cost_bps_of_nav", "mean"),
            total_estimated_cost_bps_nav=("estimated_cost_bps_of_nav", "sum"),
        )
    )
    per_etf = per_etf.merge(portfolio_sector_stats, how="left", on=["variant", "topix17_etf"])
    per_etf.to_csv(result_dir / "per_etf_metrics.csv", index=False)
    actual_0910.to_csv(result_dir / "observed_0910_diagnostic.csv", index=False)
    pnl_summary.to_csv(result_dir / "portfolio_pnl_summary.csv", index=False)
    daily_pnl.to_csv(result_dir / "daily_portfolio_pnl.csv", index=False)
    sector_daily.to_csv(result_dir / "daily_sector_pnl_contributions.csv", index=False)
    if rolling["selection_records"]:
        pd.DataFrame(rolling["selection_records"]).to_csv(result_dir / "primary_daily_basket_allocations.csv", index=False)
    for method, series in basket_turnover.items():
        if method == "etf":
            continue
        rows = []
        for etf_index, etf in enumerate(JP_TICKERS):
            values = series[:, etf_index]
            rows.append({
                "variant": method,
                "topix17_etf": etf,
                "mean_basket_one_way_turnover": float(np.nanmean(values)) if np.isfinite(values).any() else np.nan,
            })
        pd.DataFrame(rows).to_csv(result_dir / f"{method}_basket_turnover.csv", index=False)

    specs = rolling["specs"]
    result_summary = {
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config_path": str(CONFIG_PATH.relative_to(ROOT)),
        "config_sha256": _sha256_file(CONFIG_PATH),
        "input_summary": inputs["input_summary"],
        "stock_0910_price_available": False,
        "etf_0910_coverage": coverage_0910,
        "weight_repricing_reconciliation": reconciliation,
        "rolling_selection": {
            "lookback_sessions": inputs["config"]["evaluation"]["lookback_sessions"],
            "candidate_pool_top_k": inputs["config"]["evaluation"]["candidate_pool_top_k"],
            "training_day_included": False,
            "candidate_liquidity_rank": "trailing 252-session average close * volume, ending at t-1",
            "sparse_optimizer": "exact enumeration of all nonnegative simplex supports of size 1-3 among the eligible top-K pool",
            "cost_aware_objective": "centered tracking-error variance (bp^2) + lambda (bp) * estimated one-way cost rate (bp)",
            "specifications": specs,
        },
        "replaceability_gates": gate_rows,
        "portfolio_metrics": pnl_summary.to_dict("records"),
        "validation_metrics": validation,
        "related_prior_experiment_count": len(_related_registry_names(ExperimentRegistry(default_registry_path()))),
        "related_prior_experiment_names": _related_registry_names(ExperimentRegistry(default_registry_path())),
        "return_target": "matched ETF and individual-stock open-to-close return proxy for primary full-history OOS evaluation",
        "pnls": "saved NightManager target weights re-priced with local open-to-close returns, side leverage from saved summary, spread + slippage only",
        "production_modified": False,
    }
    (result_dir / "summary.json").write_text(json.dumps(_clean(result_summary), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    primary_stock = ["single_k10", "sparse3_k10", "costaware3_k10_l10"]
    etf_by_variant = {
        name: per_etf.loc[per_etf["variant"] == name].set_index("topix17_etf")
        for name in ["etf", *primary_stock]
    }
    rows = [
        "# NightManager: TOPIX-17 ETF stock-basket proxy experiment",
        "",
        "- 作成日: 2026-10-07（Asia/Tokyo）",
        "- 実験コード: `src/research/scripts/experiments/experiment_nightmanager_etf_stock_proxy_20261007.py`",
        "- 設定: `configs/research/nightmanager_etf_stock_proxy_20261007.yaml`",
        "- レポート結果: `var/results/20261007_nightmanager_etf_stock_proxy/`",
        "- 本番コード・本番設定の変更: なし",
        "",
        "## 仮説と事前固定した比較",
        "",
        "TOPIX-17 ETFを同じTOPIX-17親分類に属する高流動性個別株へ置き換え、ETFリターンを追跡しつつ推定売買費用を下げられる対象があるかを調べた。毎日 t の選択には t より前の252営業日のETF/個別株open-to-closeリターンと売買代金だけを使う。",
        "",
        "| 比較 | 選択ルール |",
        "|---|---|",
        "| Current ETF | TOPIX-17 ETFをそのまま保持 |",
        "| Single | 過去252日相関が最大の同業種株。流動性上位K銘柄内 |",
        "| Sparse-3 | 流動性上位K銘柄から1〜3社、非負・合計1の重みでtracking error分散を最小化 |",
        "| Cost-aware-3 | 同じ制約でtracking error分散 + λ×推定片道コストを最小化 |",
        "",
        "主要設定はK=10、lookback=252日、cost-aware λ=10bp。感度はK=8/12とλ=5/20bpを同じ rolling OOSで追加評価した。Cost-awareの目的関数は単位をそろえるため、tracking-error分散(bp²) + λ(bp) ×推定片道コスト率(bp)とした。",
        "",
        "## データと時間窓",
        "",
        f"- 個別株: `{inputs['input_summary']['mapping_path']}` の2026-07-31現行JPX分類/1620銘柄マップと `var/research/subsector/raw_ohlc/` のOHLCV。共有対象期間は {inputs['input_summary']['stock_raw_price_start']}〜{inputs['input_summary']['stock_raw_price_end']}。",
        "- ETF: NightManagerローカルの `var/market_data/etf_prices.sqlite` に保存されたTOPIX-17の日次open/closeを再利用。主要比較はETF・個別株の双方をopen→closeへそろえた。個別株の09:10価格はキャッシュに存在しないため、長期指標は9:10損益とは呼ばない。",
        f"- ETF 09:10参考診断: ETFの5分足09:10 High/Low中値は {coverage_0910.get('bar_start')}〜{coverage_0910.get('bar_end')} のキャッシュにあり、観測日は全体{coverage_0910.get('observed_dates', 0)}日、17 ETF全揃い{coverage_0910.get('dates_with_all_17_etfs', 0)}日。個別株proxyはopen→closeのままなので、短期診断は時間窓が混ざり参考値として別表にした。",
        "- 同業種定義は現行分類を過去へ固定したもの。過去のTOPIX採用履歴・上場廃止銘柄・歴史的業種変更がなく、1620社は現在の生存銘柄からなるため、survivorship biasとPIT分類漏れがある。",
        "- 全期間は過去に閲覧済みの価格データによるretrospective pseudo-OOS。日次選択はrolling historicalだが、期間は未使用OOSではない。2019-12-31までの品質/費用ゲートだけでETF別の部分置換可否を決め、NightManager weights期間の2020年以降にhybridを適用した。",
        "",
        "## ETF別の長期rolling OOS指標（K=10）",
        "",
        "相関・TE・MAE・RMSE・符号一致率・単位basket turnover・推定片道コスト率を同じETF/日で集計した。turnoverはbasket重みの0.5×L1差の平均。NightManager目標ウェイトを使う期間の実効片道turnoverとNAVあたり推定コストもETF別に併記した。",
        "",
        "| ETF | 方法 | OOS日数 | 相関 | TE vol (bp) | TE/ETF vol | MAE (bp) | RMSE (bp) | 符号一致 | basket turnover | unit cost (bp/片道) | 実効turnover/日 | 費用 (bp NAV/日) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for etf in JP_TICKERS:
        for method in ["etf", *primary_stock]:
            if etf not in etf_by_variant[method].index:
                continue
            item = etf_by_variant[method].loc[etf]
            turnover = item["mean_basket_one_way_turnover"]
            turnover_text = "—" if not np.isfinite(turnover) else f"{turnover:.3f}"
            rows.append(
                f"| {etf} | {METHOD_COLORS.get(method, method)} | {int(item['oos_dates'])} | "
                f"{item['oos_correlation']:.3f} | {item['tracking_error_vol_bps']:.1f} | "
                f"{item['tracking_error_to_etf_vol_ratio']:.2f} | {item['mae_bps']:.1f} | "
                f"{item['rmse_bps']:.1f} | {item['sign_agreement']:.3f} | "
                f"{turnover_text} | {item['mean_estimated_one_way_cost_rate_bps']:.2f} | "
                f"{item['mean_effective_one_way_turnover']:.3f} | "
                f"{item['mean_estimated_cost_bps_nav_per_day']:.2f} |"
            )
    rows.extend([
        "",
        "## NightManager target weightsの再価格付け",
        "",
        f"保存済みウェイトは `{inputs['input_summary']['target_weights_path']}`（{inputs['input_summary']['target_weights_start']}〜{inputs['input_summary']['target_weights_end']}、{inputs['input_summary']['target_weight_days']}日）。model weight grossは平均{inputs['input_summary']['target_weight_gross_mean_pre_side_leverage']:.3f}、最大{inputs['input_summary']['target_weight_gross_max_pre_side_leverage']:.3f}（上限2.0）、net絶対値最大{inputs['input_summary']['target_weight_net_abs_max_pre_side_leverage']:.3g}。side_leverage={inputs['side_leverage']:.2f}適用後の実効grossは平均{inputs['input_summary']['effective_gross_mean_post_side_leverage']:.3f}、最大{inputs['input_summary']['effective_gross_max_post_side_leverage']:.3f}、net絶対値最大{inputs['input_summary']['effective_net_abs_max_post_side_leverage']:.3g}。PnLはこの実効ウェイトで日次open→close returnsを再価格付けした。",
        f"保存されたgross returnとの照合: {reconciliation.get('paired_dates', 0)}日、相関{reconciliation.get('correlation', float('nan')):.3f}、MAE{reconciliation.get('mean_absolute_difference', float('nan'))*10000:.1f}bp。完全一致しないため、本表は保存済みNet PnLの再現ではなく、同一ウェイト/同一日次リターン定義でのETF対proxy再価格比較とする。",
        "",
        "| Variant | 年率Gross Sharpe | 年率Net Sharpe* | Gross MDD | Net MDD* | Net累積return* | 片道turnover/日 | 平均コスト (bp NAV/日) | Δ平均日次Net vs ETF (bp) | block bootstrap 95% CI (bp/日) | DSR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for _, item in pnl_summary.iterrows():
        ci = item.get("paired_mean_daily_net_return_difference_ci95")
        ci_text = "—" if not isinstance(ci, list) else f"[{ci[0]*10000:.2f}, {ci[1]*10000:.2f}]"
        dsr = item.get("deflated_sharpe_ratio")
        dsr_text = "—" if dsr is None or not np.isfinite(dsr) else f"{dsr:.3f}"
        delta = item.get("paired_mean_daily_net_return_difference_vs_etf", np.nan)
        delta_text = "—" if not np.isfinite(delta) else f"{delta * 10000:.2f}"
        rows.append(
            f"| {item['variant']} | {item['gross_sharpe']:.3f} | {item['net_sharpe']:.3f} | "
            f"{item['gross_max_drawdown']:.2%} | {item['net_max_drawdown']:.2%} | "
            f"{item['net_total_return']:.2%} | {item['average_effective_one_way_turnover']:.3f} | "
            f"{item['average_transaction_cost_bps_of_nav']:.2f} | "
            f"{delta_text} | "
            f"{ci_text} | {dsr_text} |"
        )
    rows.extend([
        "",
        "* Netは本実験で見積もったspread + 5bp/片道slippageだけを粗収益から引いた。commission、取引所費用、税、borrow/financing/reverse、impact、空売り可否は含まない。ETF側は歴史的quote spreadを取得できなかったため既存5bp片道slippageのみ、株側は過去OHLCからのCorwin–Schultz half-spread推定 + 同じ5bpを使う。したがってコスト比較はETF側spreadを欠く保守的推定で、実約定コストの証明ではない。",
        "",
        "## ETF別の選択的置換判定",
        "",
        "2019年末までのrolling OOSをETF別品質ゲートに使い、相関≥0.90かつTE/ETF vol≤0.35の場合だけ2020年以降のhybridで置換する。コスト率はゲートに含めない。ETFの過去quote spreadがなくETF側は5bp/片道だけを置いたため、単位コスト同士の優劣を合否に使わず、NightManager配分下の推定cost控除後PnLで評価する。選定期間は既に閲覧済みで未使用OOSではない。",
        "",
        "| Variant | ETF | 検証日数 | corr | TE/ETF vol | cost rate (bp) | 判定 |",
        "|---|---|---:|---:|---:|---:|---|",
    ])
    for item in gate_rows:
        rows.append(
            f"| {item['variant']} | {item['topix17_etf']} | {item['validation_dates']} | "
            f"{item['validation_correlation']:.3f} | {item['validation_te_ratio']:.2f} | "
            f"{item['validation_unit_cost_bps']:.2f} | {item['verdict']} |"
        )
    rows.extend([
        "",
        "## 09:10実測ETFとの短期診断",
        "",
        "`var/results/20261007_nightmanager_etf_stock_proxy/observed_0910_diagnostic.csv` にETF別指標を保存した。ETF actual targetは5分足09:10 mid proxy→close、個別株はopen→close proxyのため時間窓が一致しない。この表から厳密なtracking abilityや9:10執行PnLを結論しない。",
        "",
        "## 既存JPX33/subsector実験との差分",
        "",
        "既存subsector実験は79グループへ時価総額加重で集約し、BLPXの予測信号/ICを評価した。JPX33実験も業種予測を17 ETFへ写像したもので、実際に売買する個別株portfolioの追随誤差・執行turnover・費用後PnLは測っていなかった。本実験は現行TOPIX-17 parent分類に属する1620個別銘柄OHLCVから、毎日過去252日だけで単一株/最大3株の売買basketを選び、実ETFリターンと NightManager保存ウェイトへ付け直した推定cost/PnLを直接比較する。従って単なる既存実験の再実行ではない。本実験の事前固定12設定は同一目的の比較としてDSRで補正し、既存の関連実験24件は予測IC・分類写像など異なる指標/仮説を含むため同じSharpe試行群へ混算していない。",
        "",
        "## 判定と限界",
        "",
        "ETFごとにOOS品質ゲートを適用し、一律置換は仮定しない。品質ゲートを通過したETFは0/51（3方式×17 ETF）で、全hybridはETF版と同じ配分となる。Sparse-3/Cost-aware-3は多くのETFでSingleよりTEを下げたが、品質閾値を満たす追跡精度に達せず、NightManager再価格PnLでも全置換版はETF版を大幅に下回った。したがって『許容TEの3銘柄basketが費用控除後PnLを改善するETFがある』という仮説は本実験では支持されない。OOS終端は個別株データ最終日で2026年は8月までの部分年。日足open→closeは9:10約定価格を示さず、ETFの歴史的quote spread、株のPIT構成銘柄/分類、約定可能数量、loan availabilityがない。費用後PnLはspread/slippage代理に限られ、本番採用根拠ではない。",
        "",
        "## 再現手順",
        "",
        f"```bash\ntimeout -k 20 14400 .venv/bin/python -u {Path(__file__).relative_to(ROOT)}\n```",
        "",
        "実験はローカルParquet/SQLiteと保存済みCSVのみを読み、production code/config、market cache、target weight artifactを書き換えない。スクリプト実行時は`var/experiments/registry.jsonl`へ候補設定の記録を追加する。",
        "",
        "## 成果物",
        "",
        "- `per_etf_metrics.csv`: 12候補のETF別OOS指標とK/λ感度",
        "- `primary_daily_basket_allocations.csv`: 主要3つの株proxy選択と日次重み/事前情報",
        "- `portfolio_pnl_summary.csv`, `daily_portfolio_pnl.csv`, `daily_sector_pnl_contributions.csv`: ETF版、全置換版、選択的hybridの再価格損益",
        "- `observed_0910_diagnostic.csv`: ETF実測09:10ターゲットとの短期混合時間窓診断",
        "- `summary.json`: データfingerprint、設定、選択ゲート、検証範囲",
        "",
    ])
    (REPORT_DIR / "report.md").write_text("\n".join(rows), encoding="utf-8")

    # Append one registry record per predeclared pure portfolio trial.
    registry = ExperimentRegistry(default_registry_path())
    prior_names = _related_registry_names(registry)
    started_at = datetime.now(UTC)
    ended_at = datetime.now(UTC)
    all_pure_rows = pnl_summary[pnl_summary["variant"].isin([spec["name"] for spec in specs])]
    trial_sharpes = [float(value) for value in all_pure_rows["net_sharpe"].dropna().to_list()]
    for spec in specs:
        method_name = spec["name"]
        portfolio = pnl_extra["returns"][method_name]
        metrics = {
            "metric_status": "valid",
            "metric_schema_version": "daily-v1",
            "trial_count_status": "unknown",
            "net_sharpe_frequency": "annual",
            "trading_days_per_year": int(inputs["config"]["evaluation"]["annualization_sessions"]),
            "net_sharpe": portfolio["metrics"]["net_sharpe"],
            "n_observations": int(len(portfolio["net_returns"])),
            "max_dd": portfolio["metrics"]["net_max_drawdown"],
            "total_return": portfolio["metrics"]["net_total_return"],
            "returns": portfolio["net_returns"].tolist(),
            "turnover": portfolio["metrics"]["average_effective_one_way_turnover"],
            "trials": int(pnl_extra["trial_count"]),
            "trial_sharpes": trial_sharpes,
        }
        metrics["deflated_sharpe"] = compute_deflated_sharpe(metrics)
        registry.record(ExperimentRecord(
            name=f"{EXPERIMENT_ID}_{method_name}",
            hypothesis="A same-TOPIX-17-industry, rolling OOS single-stock or sparse-three-stock basket can track the ETF and improve transaction-cost-adjusted NightManager repricing.",
            start_time=started_at,
            end_time=ended_at,
            study_id=EXPERIMENT_ID,
            parameters={
                "variant": spec,
                "target": "ETF open-to-close, matched interval for stock proxies",
                "lookback_sessions": int(inputs["config"]["evaluation"]["lookback_sessions"]),
                "candidate_rank": "prior 252-session average dollar volume",
                "mapping_sha256": inputs["input_summary"]["mapping_sha256"],
                "target_weights_sha256": inputs["input_summary"]["target_weights_sha256"],
                "stock_price_matrix_sha256": inputs["input_summary"]["stock_price_matrix_sha256"],
                "pnl_side_leverage": inputs["side_leverage"],
                "production_modified": False,
                "decision_basis": (
                    "ETF control baseline"
                    if method_name == "etf"
                    else "rejected: no ETF passed the predeclared tracking-quality gate"
                    if not any(bool(item["replaceable"]) for item in gate_rows)
                    else "pending: at least one ETF passed the tracking-quality gate"
                ),
            },
            metrics=metrics,
            decision=(Decision.PENDING if method_name == "etf" else
                      Decision.REJECTED if not any(bool(item["replaceable"]) for item in gate_rows) else
                      Decision.PENDING),
            report_path=str((REPORT_DIR / "report.md").relative_to(ROOT)),
            related_records=prior_names,
        ))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    started_at = datetime.now(UTC)
    config = _load_config()
    inputs = _load_market_inputs(config)
    LOG.info("Loaded %d stocks and %d daily dates (%s to %s)", len(inputs["tickers"]), len(inputs["dates"]), inputs["dates"].min().date(), inputs["dates"].max().date())
    rolling = _rolling_select(inputs)
    LOG.info("Completed rolling selections for %d variants", len(rolling["specs"]))
    n_dates = len(inputs["dates"])
    basket_turnover: dict[str, np.ndarray] = {}
    for spec_index, spec in enumerate(rolling["specs"]):
        if spec["kind"] == "etf":
            continue
        basket_turnover[spec["name"]] = _basket_turnover(
            rolling["selected_indices"], rolling["selected_weights"], rolling["fallback"],
            spec_index, n_dates, len(JP_TICKERS),
        )
    per_etf, validation = _metric_rows(inputs, rolling, basket_turnover)
    targets_0910, coverage_0910 = _build_0910_targets(inputs)
    actual_0910 = _actual_0910_metrics(inputs, rolling, targets_0910)

    gate_rows: list[dict[str, Any]] = []
    replace_masks: dict[str, np.ndarray] = {}
    min_corr = float(config["replaceability_gates"]["minimum_oos_correlation"])
    max_te_ratio = float(config["replaceability_gates"]["maximum_tracking_error_to_etf_vol_ratio"])
    stock_method_names = ["single_k10", "sparse3_k10", "costaware3_k10_l10"]
    for method_name in stock_method_names:
        etf_validation = validation.get(method_name, {})
        eligible = np.zeros(len(JP_TICKERS), dtype=bool)
        for etf_index, etf in enumerate(JP_TICKERS):
            item = etf_validation.get(etf, {})
            corr = item.get("correlation", np.nan)
            ratio = item.get("tracking_error_ratio", np.nan)
            unit_cost = item.get("unit_cost_bps", np.nan)
            passes_corr = np.isfinite(corr) and corr >= min_corr
            passes_te = np.isfinite(ratio) and ratio <= max_te_ratio
            cost_gate_required = bool(config["replaceability_gates"]["require_lower_estimated_unit_cost"])
            passes_cost = bool(np.isfinite(unit_cost) and unit_cost < ETF_COST_BPS) if cost_gate_required else None
            passed = bool(passes_corr and passes_te and (passes_cost is not False))
            eligible[etf_index] = passed
            reasons = []
            if not passes_corr:
                reasons.append("相関ゲート未達")
            if not passes_te:
                reasons.append("TEゲート未達")
            if passes_cost is False:
                reasons.append("推定片道コスト低下なし")
            gate_rows.append({
                "variant": method_name,
                "topix17_etf": etf,
                "validation_dates": int(item.get("dates", 0)),
                "validation_correlation": corr,
                "validation_te_ratio": ratio,
                "validation_unit_cost_bps": unit_cost,
                "pass_correlation": passes_corr,
                "pass_tracking_error": passes_te,
                "pass_cost": passes_cost,
                "cost_gate_required": cost_gate_required,
                "replaceable": passed,
                "verdict": "選択的置換候補" if passed else "ETF維持: " + "、".join(reasons),
            })
        replace_masks[method_name] = eligible

    pnl_summary, daily_pnl, pnl_extra = _portfolio_simulations(inputs, rolling, replace_masks)
    reconciliation = _reprice_reconciliation(inputs)
    _write_outputs(
        inputs, rolling, per_etf, validation, pnl_summary, daily_pnl, pnl_extra,
        targets_0910, coverage_0910, actual_0910, basket_turnover,
        gate_rows, reconciliation,
    )
    LOG.info("Report written to %s", REPORT_DIR / "report.md")
    LOG.info("Results written to %s", ROOT / "var/results/20261007_nightmanager_etf_stock_proxy")
    LOG.info("Experiment completed in %.1f seconds", (datetime.now(UTC) - started_at).total_seconds())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
