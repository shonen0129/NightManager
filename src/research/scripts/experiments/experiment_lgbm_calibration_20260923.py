#!/usr/bin/env python3
"""Walk-forward calibration diagnostic for the saved LGBM order forecasts.

This is a development-only follow-up to order 11.  It fits an expanding
affine calibrator only from strictly earlier ticker-days, then re-evaluates the
same fixed 12.5 bps/side gate.  It does not create a fresh holdout or model
actual broker inventory/fills.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "research" / "scripts" / "experiments"))

from experiment_lgbm_order_cost_20260923 import (  # noqa: E402
    _cost_params,
    _order_cost_gate,
)

from leadlag.core.pnl import simulate_daily_pnl
from leadlag.data.intraday_inputs import build_open_910_returns  # noqa: E402
from leadlag.data.tickers import JP_TICKERS  # noqa: E402
from leadlag.execution.backtester import BacktestEngine  # noqa: E402
from leadlag.execution.config import load_config_from_yaml  # noqa: E402
from leadlag.experiment_registry import (  # noqa: E402
    Decision,
    ExperimentRecord,
    ExperimentRegistry,
    compute_deflated_sharpe,
)
from leadlag.reporting.metrics import (  # noqa: E402
    MetricsSpec,
    calculate_metrics,
    compute_drawdown_series,
)

OUTPUT_DIR = ROOT / "reports" / "20260923_profitability_order_12"
REGISTRY_PATH = ROOT / "var" / "experiments" / "registry.jsonl"
SOURCE_ORDERS = ROOT / "reports" / "20260923_profitability_order_11" / "daily_orders.csv"
SOURCE_GATE_WEIGHTS = (
    ROOT / "reports" / "20260923_profitability_order_11" / "daily_gate_weights.csv"
)
SOURCE_BASELINE = (
    ROOT / "reports" / "20260923_profitability_order_5" / "v2_current_overlay_oos"
)
EXEC_DF_SOURCE = (
    ROOT / "var/results/20260920_structural_completion/r3_capture_gap/exact_df_exec.pkl"
)
PRICE_CACHE_SOURCE = ROOT / "var" / "market_data" / "etf_prices.sqlite"
CONFIG_PATH = ROOT / "configs" / "production" / "production.yaml"
CENTRAL_BPS = 12.5
WARMUP_DAYS = (50, 63, 76)  # primary=63; neighboring values are +/- about 20%.
PRIMARY_WARMUP = 63
BLOCK_DAYS = 20
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 42
ANNUALIZATION_DAYS = 252


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _normalize_frame_index(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.index = pd.DatetimeIndex(pd.to_datetime(result.index)).normalize()
    return result


def _load_observations() -> tuple[
    pd.DatetimeIndex,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    targets = pd.read_csv(
        SOURCE_BASELINE / "daily_weights.csv", index_col=0, parse_dates=True
    )
    targets = _normalize_frame_index(targets).loc[:, JP_TICKERS].astype(float)
    dates = pd.DatetimeIndex(targets.index)

    orders = pd.read_csv(SOURCE_ORDERS, parse_dates=["trade_date"])
    central = orders.loc[np.isclose(orders["cost_bps_per_side"], CENTRAL_BPS)]
    predictions = central.pivot(
        index="trade_date", columns="ticker", values="expected_return"
    )
    predictions = _normalize_frame_index(predictions).reindex(index=dates, columns=JP_TICKERS)
    if predictions.isna().any().any():
        raise ValueError("saved order forecasts do not cover the canonical target dates")

    df_exec = _load_pickle(EXEC_DF_SOURCE)
    if not isinstance(df_exec, pd.DataFrame):
        raise TypeError("exact df_exec snapshot must be a pandas DataFrame")
    df_exec = _normalize_frame_index(df_exec)
    open_910 = _normalize_frame_index(
        build_open_910_returns(df_exec, JP_TICKERS)
    ).reindex(columns=JP_TICKERS)
    if not dates.isin(df_exec.index).all():
        raise ValueError("acceptance df_exec does not cover the saved forecast dates")

    realized_array, gap_array = BacktestEngine._compute_target_and_gap_returns(
        df_exec,
        pd.DatetimeIndex(df_exec.index),
        dates,
        open_910_returns=open_910,
    )
    realized = pd.DataFrame(realized_array, index=dates, columns=JP_TICKERS)
    gap_returns = pd.DataFrame(gap_array, index=dates, columns=JP_TICKERS)

    fallback = pd.read_csv(
        SOURCE_BASELINE / "daily_fallback.csv", parse_dates=["trade_date"]
    ).set_index("trade_date")["fallback"]
    fallback.index = pd.DatetimeIndex(fallback.index).normalize()
    fallback = fallback.reindex(dates).fillna(True).astype(bool)

    saved_gate_weights = pd.read_csv(
        SOURCE_GATE_WEIGHTS, index_col=0, parse_dates=True
    )
    saved_gate_weights = _normalize_frame_index(saved_gate_weights)
    saved_returns = pd.read_csv(
        SOURCE_BASELINE / "daily_net_returns.csv", index_col=0, parse_dates=True
    )
    saved_returns = _normalize_frame_index(saved_returns)
    return (
        dates,
        targets,
        predictions,
        realized,
        fallback,
        df_exec,
        open_910,
        gap_returns,
        saved_gate_weights,
        saved_returns,
    )


def _fit_expanding_calibration(
    predictions: pd.DataFrame,
    realized: pd.DataFrame,
    warmup_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.DatetimeIndex(predictions.index)
    calibrated = predictions.copy()
    coefficients: list[dict[str, Any]] = []
    for index, date in enumerate(dates):
        if index < warmup_days:
            coefficients.append(
                {
                    "trade_date": date,
                    "history_days": index,
                    "history_ticker_days": 0,
                    "intercept": 0.0,
                    "slope": 1.0,
                    "calibration_applied": False,
                }
            )
            continue
        history_dates = dates[:index]
        x = predictions.loc[history_dates].to_numpy(dtype=float).reshape(-1)
        y = realized.loc[history_dates].to_numpy(dtype=float).reshape(-1)
        valid = np.isfinite(x) & np.isfinite(y)
        design = np.column_stack((np.ones(int(valid.sum())), x[valid]))
        intercept, slope = np.linalg.lstsq(design, y[valid], rcond=None)[0]
        calibrated.loc[date] = float(intercept) + float(slope) * predictions.loc[date]
        coefficients.append(
            {
                "trade_date": date,
                "history_days": index,
                "history_ticker_days": int(valid.sum()),
                "intercept": float(intercept),
                "slope": float(slope),
                "calibration_applied": True,
            }
        )
    return calibrated, pd.DataFrame(coefficients).set_index("trade_date")


def _forecast_metrics(
    prediction: pd.DataFrame,
    realized: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> dict[str, Any]:
    x = prediction.loc[dates].to_numpy(dtype=float).reshape(-1)
    y = realized.loc[dates].to_numpy(dtype=float).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    design = np.column_stack((np.ones(len(x)), x))
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    return {
        "trading_days": int(len(dates)),
        "ticker_days": int(len(x)),
        "predicted_mean": float(np.mean(x)),
        "realized_mean": float(np.mean(y)),
        "mae": float(np.mean(np.abs(x - y))),
        "rmse": float(np.sqrt(np.mean((x - y) ** 2))),
        "pearson": float(np.corrcoef(x, y)[0, 1]),
        "spearman": float(spearmanr(x, y).statistic),
        "directional_accuracy": float(np.mean(np.sign(x) == np.sign(y))),
        "calibration_intercept": float(intercept),
        "calibration_slope": float(slope),
    }


def _reliability_table(
    raw: pd.DataFrame,
    calibrated: pd.DataFrame,
    realized: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "prediction": raw.loc[dates].to_numpy(dtype=float).reshape(-1),
            "calibrated_prediction": calibrated.loc[dates].to_numpy(dtype=float).reshape(-1),
            "realized": realized.loc[dates].to_numpy(dtype=float).reshape(-1),
        }
    )
    frame["decile"] = pd.qcut(
        frame["prediction"], 10, labels=False, duplicates="drop"
    )
    return frame.groupby("decile", observed=True).agg(
        n=("realized", "size"),
        predicted_mean=("prediction", "mean"),
        calibrated_mean=("calibrated_prediction", "mean"),
        realized_mean=("realized", "mean"),
        realized_win_rate=("realized", lambda values: float((values > 0).mean())),
    )


def _paired_block_ci(
    lhs: np.ndarray,
    rhs: np.ndarray,
    *,
    block_days: int = BLOCK_DAYS,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    delta = np.asarray(lhs, dtype=float) - np.asarray(rhs, dtype=float)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(len(delta) / block_days))
    boot = np.empty(samples, dtype=float)
    for sample in range(samples):
        starts = rng.integers(0, len(delta), n_blocks)
        indices = np.concatenate(
            [(start + np.arange(block_days)) % len(delta) for start in starts]
        )[: len(delta)]
        boot[sample] = float(delta[indices].mean())
    return {
        "mean_daily_delta": float(delta.mean()),
        "sum_daily_delta": float(delta.sum()),
        "block_days": block_days,
        "samples": samples,
        "seed": seed,
        "ci95": [float(value) for value in np.quantile(boot, [0.025, 0.975])],
    }


def _simulate(
    weights: pd.DataFrame,
    dates: pd.DatetimeIndex,
    df_exec: pd.DataFrame,
    open_910: pd.DataFrame,
    config: Any,
    bps_per_side: float,
    fallback: pd.Series,
) -> dict[str, Any]:
    params = _cost_params(config, bps_per_side)
    targets, gaps = BacktestEngine._compute_target_and_gap_returns(
        df_exec,
        pd.DatetimeIndex(df_exec.index),
        dates,
        open_910_returns=open_910,
    )
    pnl = simulate_daily_pnl(
        weights=weights.loc[dates, JP_TICKERS].to_numpy(dtype=float),
        target_returns=targets,
        gap_returns=gaps,
        sim_dates=dates,
        slip=float(params["slip_bps"]) / 10000.0,
        financing_daily=float(params["fin_annual"]) / 365.0,
        borrow_daily=float(params["borrow_annual"]) / 365.0,
        reverse_daily=float(params["rev_bps"]) / 10000.0,
        alpha_long=float(params["alpha_long"]),
        alpha_short=float(params["alpha_short"]),
        side_leverage=float(params["side_leverage"]),
    )
    return BacktestEngine._assemble_v2_results(
        pnl,
        weights.loc[dates, JP_TICKERS],
        fallback.reindex(dates).to_numpy(dtype=bool),
        [{"trade_date": str(date.date())} for date in dates],
        dates,
        float(params["alpha_long"]),
        float(params["alpha_short"]),
        float(params["side_leverage"]),
    )


def _slice_metrics(result: dict[str, Any], dates: pd.DatetimeIndex) -> dict[str, Any]:
    net = result["daily_returns"].loc[dates].astype(float)
    gross = result["daily_returns_gross"].loc[dates].astype(float)
    spec = MetricsSpec(
        frequency="daily", annualization_periods=ANNUALIZATION_DAYS, include_flat_days=True
    )
    net_stats = calculate_metrics(net, spec=spec)
    gross_stats = calculate_metrics(gross, spec=spec)
    cost_parts = {
        part: float(result[f"daily_{part}_costs"].loc[dates].astype(float).sum())
        for part in ("slip", "financing", "borrow", "reverse")
    }
    return {
        "days": int(len(dates)),
        "start": str(dates[0].date()),
        "end": str(dates[-1].date()),
        "net_sharpe": float(net_stats["Sharpe"]),
        "gross_sharpe": float(gross_stats["Sharpe"]),
        "net_sum_simple": float(net.sum()),
        "gross_sum_simple": float(gross.sum()),
        "max_drawdown": float(compute_drawdown_series(net).min()),
        "mean_turnover": float(result["daily_turnover"].loc[dates].mean()),
        "mean_model_gross": float(result["daily_gross_exps"].loc[dates].mean()),
        "mean_effective_gross": float(result["daily_gross_exps"].loc[dates].mean())
        * float(result["side_leverage"]),
        "cost_components": cost_parts,
        "total_cost_sum": float(sum(cost_parts.values())),
        "returns": net.tolist(),
    }


def _main() -> None:
    started = datetime.now(UTC)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (
        dates,
        targets,
        raw_predictions,
        realized,
        fallback,
        df_exec,
        open_910,
        _gap_returns,
        saved_gate_weights,
        saved_returns,
    ) = _load_observations()
    config = load_config_from_yaml(CONFIG_PATH, strict=True)

    baseline_daily = pd.read_csv(
        SOURCE_BASELINE / "daily_net_returns.csv", index_col=0, parse_dates=True
    )
    baseline_daily = _normalize_frame_index(baseline_daily).reindex(dates)
    baseline_5bps = _simulate(
        targets, dates, df_exec, open_910, config, 5.0, fallback
    )
    reproduction_error = float(
        np.max(
            np.abs(
                baseline_5bps["daily_returns"].to_numpy(dtype=float)
                - baseline_daily["net_return"].to_numpy(dtype=float)
            )
        )
    )
    if reproduction_error > 1e-10:
        raise RuntimeError(
            f"acceptance input snapshot does not reproduce the source baseline: {reproduction_error:.3e}"
        )

    forecast_map_raw = {
        date.strftime("%Y-%m-%d"): {"mu": raw_predictions.loc[date].to_numpy(dtype=float)}
        for date in dates
    }
    cost_params = _cost_params(config, CENTRAL_BPS)
    raw_gate_weights, _raw_decisions, _raw_orders = _order_cost_gate(
        targets,
        forecast_map_raw,
        fallback,
        bps_per_side=CENTRAL_BPS,
        cost_params=cost_params,
        dates=dates,
    )
    saved_central = saved_gate_weights.loc[
        dates, [f"{ticker}_bps_{CENTRAL_BPS:.1f}" for ticker in JP_TICKERS]
    ].to_numpy(dtype=float)
    raw_gate_reproduction_error = float(
        np.max(np.abs(raw_gate_weights.to_numpy(dtype=float) - saved_central))
    )
    if raw_gate_reproduction_error > 1e-9:
        raise RuntimeError(
            f"saved raw gate weights did not reproduce: {raw_gate_reproduction_error:.3e}"
        )

    calibrated_predictions: dict[int, pd.DataFrame] = {}
    coefficient_frames: list[pd.DataFrame] = []
    prediction_rows = pd.DataFrame(
        {
            "raw_prediction": raw_predictions.stack(future_stack=True),
            "realized_return": realized.stack(future_stack=True),
        }
    )
    prediction_rows.index.names = ["trade_date", "ticker"]
    calibration_summaries: dict[str, Any] = {}
    loss_bootstrap: dict[str, Any] = {}

    for warmup in WARMUP_DAYS:
        calibrated, coefficients = _fit_expanding_calibration(
            raw_predictions, realized, warmup
        )
        calibrated_predictions[warmup] = calibrated
        coefficients["warmup_days"] = warmup
        coefficient_frames.append(coefficients.reset_index())
        test_dates = dates[warmup:]
        raw_stats = _forecast_metrics(raw_predictions, realized, test_dates)
        calibrated_stats = _forecast_metrics(calibrated, realized, test_dates)
        daily_raw_mae = (raw_predictions.loc[test_dates] - realized.loc[test_dates]).abs().mean(axis=1)
        daily_cal_mae = (calibrated.loc[test_dates] - realized.loc[test_dates]).abs().mean(axis=1)
        calibration_summaries[str(warmup)] = {
            "raw": raw_stats,
            "calibrated": calibrated_stats,
            "mean_daily_mae_delta_cal_minus_raw": float((daily_cal_mae - daily_raw_mae).mean()),
            "first_test_date": str(test_dates[0].date()),
            "test_days": int(len(test_dates)),
        }
        loss_bootstrap[str(warmup)] = _paired_block_ci(
            daily_cal_mae.to_numpy(dtype=float),
            daily_raw_mae.to_numpy(dtype=float),
        )
        prediction_rows[f"calibrated_{warmup}d"] = calibrated.stack(future_stack=True)

    common_dates = dates[max(WARMUP_DAYS) :]
    reliability = _reliability_table(
        raw_predictions,
        calibrated_predictions[PRIMARY_WARMUP],
        realized,
        common_dates,
    )
    top_decile = reliability.iloc[-1]

    all_weights: dict[str, pd.DataFrame] = {"raw": raw_gate_weights}
    all_results: dict[str, dict[str, Any]] = {
        "baseline_12.5": _simulate(targets, dates, df_exec, open_910, config, CENTRAL_BPS, fallback),
        "raw_gate_12.5": _simulate(
            raw_gate_weights, dates, df_exec, open_910, config, CENTRAL_BPS, fallback
        ),
    }
    gate_summaries: dict[str, Any] = {}
    for warmup in WARMUP_DAYS:
        prediction_map = {
            date.strftime("%Y-%m-%d"): {"mu": calibrated_predictions[warmup].loc[date].to_numpy(dtype=float)}
            for date in dates
        }
        weights, decisions, orders = _order_cost_gate(
            targets,
            prediction_map,
            fallback,
            bps_per_side=CENTRAL_BPS,
            cost_params=cost_params,
            dates=dates,
        )
        key = f"calibrated_{warmup}d"
        all_weights[key] = weights
        all_results[key] = _simulate(
            weights, dates, df_exec, open_910, config, CENTRAL_BPS, fallback
        )
        gate_summaries[key] = {
            "common_eval": _slice_metrics(all_results[key], common_dates),
            "primary_startup_eval": _slice_metrics(
                all_results[key], dates[warmup:]
            ),
            "mean_turnover_all_dates": float(all_results[key]["daily_turnover"].mean()),
            "mean_model_gross_all_dates": float(all_results[key]["daily_gross_exps"].mean()),
            "executed_order_days": int((decisions["executed_order_count"] > 0).sum()),
            "executed_order_count": int(decisions["executed_order_count"].sum()),
            "selected_estimated_cost_sum": float(decisions["estimated_order_cost"].sum()),
        }
        decisions.to_csv(OUTPUT_DIR / f"daily_gate_{key}.csv", index_label="trade_date")
        orders.to_csv(OUTPUT_DIR / f"orders_{key}.csv", index=False)
        weights.to_csv(OUTPUT_DIR / f"weights_{key}.csv", index_label="trade_date")

    daily_returns = pd.DataFrame(
        {
            key: result["daily_returns"].astype(float).reindex(dates)
            for key, result in all_results.items()
        },
        index=dates,
    )
    daily_returns.to_csv(OUTPUT_DIR / "daily_portfolio_returns.csv", index_label="trade_date")
    prediction_rows.to_csv(OUTPUT_DIR / "daily_forecast_calibration.csv", index_label=["trade_date", "ticker"])
    pd.concat(coefficient_frames, ignore_index=True).to_csv(
        OUTPUT_DIR / "calibration_coefficients.csv", index=False
    )
    reliability.to_csv(OUTPUT_DIR / "raw_reliability_deciles.csv")

    # Keep every candidate Sharpe on the same common post-warmup dates.
    common_results = {
        key: _slice_metrics(result, common_dates)
        for key, result in all_results.items()
    }
    trial_sharpes = [float(common_results[key]["net_sharpe"]) for key in common_results]
    primary_key = f"calibrated_{PRIMARY_WARMUP}d"
    primary_returns = common_results[primary_key]["returns"]
    dsr_metrics = {
        "net_sharpe": common_results[primary_key]["net_sharpe"],
        "net_sharpe_frequency": "annual",
        "trading_days_per_year": ANNUALIZATION_DAYS,
        "n_observations": len(common_dates),
        "returns": primary_returns,
        "trial_sharpes": trial_sharpes,
        "trials": len(trial_sharpes),
    }
    nominal_dsr = compute_deflated_sharpe(dsr_metrics)
    primary_delta = _paired_block_ci(
        np.asarray(common_results[primary_key]["returns"], dtype=float),
        np.asarray(common_results["baseline_12.5"]["returns"], dtype=float),
    )

    # Count prior tested strategy candidates at all three previously reported costs.
    prior = json.loads((SOURCE_ORDERS.parent / "results.json").read_text(encoding="utf-8"))
    prior_scenarios = prior["scenarios"]
    prior_trial_sharpes = [
        float(prior_scenarios[cost][variant]["net_sharpe"])
        for cost in ("10.0", "12.5", "15.0")
        for variant in ("baseline", "gated")
    ]
    # DSR above is same-window for the current calibration candidates. The broader
    # known family below is recorded separately because old-window Sharpes differ.
    registry = ExperimentRegistry(REGISTRY_PATH)
    known_trials_before = registry.count_trials()
    metrics: dict[str, Any] = {
        "common_evaluation_period": [str(common_dates[0].date()), str(common_dates[-1].date())],
        "common_evaluation_days": int(len(common_dates)),
        "is_fresh_holdout": False,
        "prediction_calibration": calibration_summaries,
        "prediction_mae_paired_block_ci_cal_minus_raw": loss_bootstrap,
        "order_gate_common_eval": common_results,
        "primary_63d_gate_delta_vs_baseline": primary_delta,
        "nominal_dsr_same_window_current_candidates": nominal_dsr,
        "same_window_trial_sharpes": trial_sharpes,
        "prior_order11_trial_sharpes_different_window": prior_trial_sharpes,
        "raw_gate_reproduction_max_weight_error": raw_gate_reproduction_error,
        "baseline_5bps_reproduction_max_return_error": reproduction_error,
        "open_910_source_coverage": {
            "evaluation_days": int(len(dates)),
            "complete_17_ticker_days": int(open_910.loc[dates, JP_TICKERS].notna().all(axis=1).sum()),
            "any_910_value_days": int(open_910.loc[dates, JP_TICKERS].notna().any(axis=1).sum()),
            "nonmissing_ticker_day_fraction": float(open_910.loc[dates, JP_TICKERS].notna().to_numpy().mean()),
            "label_fallback": "missing 09:10 values use the backtest target calculator's open-to-close fallback",
        },
        "input_provenance": {
            "forecast_file_sha256": _sha256(SOURCE_ORDERS),
            "exact_df_exec_sha256": _sha256(EXEC_DF_SOURCE),
            "etf_price_cache_sha256": _sha256(PRICE_CACHE_SOURCE),
            "model_historical_provider_available_at_proven": False,
        },
        "registry_records_before_run": known_trials_before,
        "frozen_parameters": {
            "central_bps_per_side": CENTRAL_BPS,
            "expanding_calibration_warmups_days": list(WARMUP_DAYS),
            "primary_warmup_days": PRIMARY_WARMUP,
            "bootstrap_block_days": BLOCK_DAYS,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
    }
    (OUTPUT_DIR / "results.json").write_text(
        json.dumps(_jsonable(metrics), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = [
        "# 収益改善順序12：LGBM予測の時系列校正診断",
        "",
        f"実行日時: {started.isoformat()}。既存予測を再利用し、モデル・本番設定・注文経路は変更していない。",
        "",
        "## 仮説と固定設計",
        "",
        "既存LGBMの予測順位に情報があるなら、過去日だけで推定した切片・傾きによる拡大型校正で、未使用の次日予測の誤差と固定コスト注文ゲートが改善する。予測ラベル・価格データ・中心コスト12.5bps/片道は順序11と揃える。校正候補は過去50/63/76営業日の最低履歴で開始し、63日を事前主候補とする。校正は銘柄日をまとめたOLS、当日より前の予測・実現値のみ使用。",
        "",
        f"- 対象: {dates[0].date()}〜{dates[-1].date()}、{len(dates)}営業日。順序11で既に確認済みで、fresh holdoutではない。",
        "- 価格・ラベル: `20260920_structural_completion/r3_capture_gap/exact_df_exec.pkl` と `var/market_data/etf_prices.sqlite` の5分足を読み、順序11と同じターゲット計算を再現。5分足1629.Tはloaderの読込時分割調整を適用。ラベルは欠損9:10入力で9:00→大引けにフォールバックする。",
        "- 保有: 前日のモデルウェイトを実在庫proxyとして継続する既存研究実装。約定・板厚・手数料・impactは今回も実測されない。",
        "- 9:10入力の評価期間内カバレッジ: 完全17銘柄 {}/{}日、少なくとも1銘柄の値あり {}/{}日、セル充足率 {:.1%}。".format(
            metrics["open_910_source_coverage"]["complete_17_ticker_days"],
            len(dates),
            metrics["open_910_source_coverage"]["any_910_value_days"],
            len(dates),
            metrics["open_910_source_coverage"]["nonmissing_ticker_day_fraction"],
        ),
        "- 学習artifact metadataはhistorical provider `available_at`の証跡を持たない。リークの証明ではないが、完全なPIT証明でもない。",
        "",
        "## 再現確認",
        "",
        f"- exact_df_execと5分足cacheによるcanonical 5bps baselineの日次リターン最大差: {reproduction_error:.3e}。",
        f"- 保存済み12.5bps raw gateとのweight最大差: {raw_gate_reproduction_error:.3e}。",
        "",
        "## 予測校正結果",
        "",
        "| 最低履歴日数 | 評価日数 | raw MAE | 校正MAE | raw slope | 校正後slope | 校正−raw 日次MAE差 | 95% block CI |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for warmup in WARMUP_DAYS:
        summary = calibration_summaries[str(warmup)]
        report.append(
            "| {} | {} | {:.5f} | {:.5f} | {:.3f} | {:.3f} | {:+.6f} | [{:+.6f}, {:+.6f}] |".format(
                warmup,
                summary["test_days"],
                summary["raw"]["mae"],
                summary["calibrated"]["mae"],
                summary["raw"]["calibration_slope"],
                summary["calibrated"]["calibration_slope"],
                summary["mean_daily_mae_delta_cal_minus_raw"],
                loss_bootstrap[str(warmup)]["ci95"][0],
                loss_bootstrap[str(warmup)]["ci95"][1],
            )
        )
    report += [
        "",
        "MAE block CIは日付を単位にした20営業日circular block bootstrap、1000回、seed=42。63日主候補のMAE差は{:.2f}bp/銘柄日で、95%区間は0をまたぐ。共通期間のraw最上位十分位は予測{:+.1f}bp、実現{:+.1f}bp、校正後予測{:+.1f}bpで、過大さが残る。全候補期間は過去に閲覧済みであり、区間は独立OOSの採否根拠ではない。分位診断は `raw_reliability_deciles.csv`、係数系列は `calibration_coefficients.csv`.".format(
            calibration_summaries[str(PRIMARY_WARMUP)]["mean_daily_mae_delta_cal_minus_raw"] * 10000.0,
            float(top_decile["predicted_mean"]) * 10000.0,
            float(top_decile["realized_mean"]) * 10000.0,
            float(top_decile["calibrated_mean"]) * 10000.0,
        ),
        "",
        "## 固定コストの注文ゲート（共通期間）",
        "",
        f"校正startup感度を同じ共通期間 {common_dates[0].date()}〜{common_dates[-1].date()} ({len(common_dates)}日) で比較。",
        "",
        "| 方法 | net Sharpe | gross Sharpe | max DD | turnover | cost sum | mean model gross |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in [
        ("baseline_12.5", "canonical baseline"),
        ("raw_gate_12.5", "raw forecast gate"),
        ("calibrated_50d", "calibrated, warmup 50d"),
        ("calibrated_63d", "calibrated, warmup 63d (primary)"),
        ("calibrated_76d", "calibrated, warmup 76d"),
    ]:
        stats = common_results[key]
        report.append(
            f"| {label} | {stats['net_sharpe']:.3f} | {stats['gross_sharpe']:.3f} | {stats['max_drawdown']:.2%} | {stats['mean_turnover']:.3f} | {stats['total_cost_sum']:.4f} | {stats['mean_model_gross']:.3f} |"
        )
    report += [
        "",
        "Primary 63d gate minus same-cost baseline paired mean daily net difference: {:.6f}, 20-day block-bootstrap 95% CI [{:.6f}, {:.6f}]。名目同期間DSR={:.4f}、比較候補Sharpe数={}。過去の試行と既知OOSの再閲覧を完全には補正できないため、DSRは採否に使わない。".format(
            primary_delta["mean_daily_delta"],
            primary_delta["ci95"][0],
            primary_delta["ci95"][1],
            nominal_dsr if nominal_dsr is not None else float("nan"),
            len(trial_sharpes),
        ),
        "",
        "## 判定と制約",
        "",
        "**保留（開発診断のみ）**。予測校正・同じ既知期間内の注文ゲートの変化は観察できたが、データを既に閲覧済みであり、9:10値は全期間の多くでopen-to-close fallback、実在庫と実約定費用もないため、新方式の採用可否は判定しない。未使用OOSの独立検証は、PIT取得時刻証拠・9:10入力・約定台帳を整え、校正方式を固定した後に行う。",
        "",
        "## 成果物",
        "",
        "- `results.json`: 全指標と固定条件",
        "- `daily_forecast_calibration.csv`: 日次・銘柄別予測と実現ラベル",
        "- `calibration_coefficients.csv`: 当日より前で推定した校正係数",
        "- `raw_reliability_deciles.csv`: raw予測分位と実現リターン",
        "- `daily_portfolio_returns.csv`: baseline/raw/calibrated gateの日次netリターン",
        "- `weights_calibrated_*.csv`, `daily_gate_calibrated_*.csv`, `orders_calibrated_*.csv`: gate経路再現",
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    record = ExperimentRecord(
        "20260923_lgbm_expanding_affine_calibration_diagnostic",
        "An expanding affine calibration fitted only from prior ticker-days improves the usable accuracy of the existing LGBM return forecast and the fixed-cost order gate.",
        start_time=started,
        end_time=datetime.now(UTC),
        parameters={
            "source_period": [str(dates[0].date()), str(dates[-1].date())],
            "source_order11_forecasts_sha256": _sha256(SOURCE_ORDERS),
            "exact_df_exec_sha256": _sha256(EXEC_DF_SOURCE),
            "etf_price_cache_sha256": _sha256(PRICE_CACHE_SOURCE),
            "calibration": "pooled OLS: realized return ~ intercept + slope * prior raw prediction, expanding; strict date lag",
            "warmup_days_sensitivity": list(WARMUP_DAYS),
            "primary_warmup_days": PRIMARY_WARMUP,
            "cost_bps_per_side": CENTRAL_BPS,
            "bootstrap": {"block_days": BLOCK_DAYS, "samples": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED},
            "is_fresh_holdout": False,
            "provider_available_at_proven": False,
            "open_910_complete_days": metrics["open_910_source_coverage"]["complete_17_ticker_days"],
            "historical_registry_records_before_run": known_trials_before,
        },
        metrics={
            "calibration_primary": calibration_summaries[str(PRIMARY_WARMUP)],
            "calibration_mae_bootstrap_ci": loss_bootstrap[str(PRIMARY_WARMUP)],
            "primary_common_window_gate": common_results[primary_key],
            "primary_gate_vs_baseline_paired_ci": primary_delta,
            "nominal_dsr_same_window_current_candidates": nominal_dsr,
            "same_window_trial_sharpes": trial_sharpes,
            "current_gate_reproduction_max_weight_error": raw_gate_reproduction_error,
            "canonical_5bps_reproduction_max_return_error": reproduction_error,
            "fresh_holdout": False,
        },
        decision=Decision.PENDING,
        report_path="reports/20260923_profitability_order_12/report.md",
        related_records=["20260923_lgbm_order_cost_primary"],
    )
    registry.record(record)


if __name__ == "__main__":
    _main()
