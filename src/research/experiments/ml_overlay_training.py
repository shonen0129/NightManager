"""Research-only training entry point for the production ML overlay.

Training has research dependencies and may read a large historical data set;
it is intentionally outside the ``leadlag`` production wheel.  The fitted
object is the production ``MLOrderOverlayModel`` so that the inference side
keeps its stable pickle class path.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from leadlag.config.schemas import AppConfig, ProductionV2RunConfig, parse_run_config
from leadlag.data import macro as macro_data
from leadlag.data.adr_features import (
    DEFAULT_ADR_FEATURES_PATH,
    load_adr_features,
    normalize_adr_features,
)
from leadlag.data.intraday_inputs import build_open_910_returns, compute_jp_target_returns
from leadlag.data.pit_lake import PITDataLake
from leadlag.data.rank_reversal import load_rank_reversal_frame
from leadlag.data.tickers import JP_TICKERS
from leadlag.domain.inputs import DecisionInputs, HistoricalInputs
from leadlag.models.ml_order_overlay import MLOrderOverlayModel
from leadlag.models.ml_overlay_artifact import save_overlay_model
from leadlag.models.ml_overlay_features import (
    _precompute_market_vol,
    _safe,
    overlay_continuous_columns,
)
from leadlag.models.production_v2 import ProductionV2Model
from leadlag.models.v2.pit import load_pit_ir_history
from leadlag.runner.model_factory import build_blpx_model
from leadlag.utils.dataframe_fingerprint import dataframe_fingerprint
from leadlag.utils.timestamps import normalize_jst_date, normalize_jst_index

logger = logging.getLogger(__name__)

TRADING_DAYS = 245
SLIPPAGE_BPS_PER_SIDE = 5.0
ROUND_TRIP_COST = 2.0 * SLIPPAGE_BPS_PER_SIDE / 10000.0
DEFAULT_LGBM_KWARGS: dict[str, Any] = {
    "n_estimators": 100,
    "max_depth": 3,
    "num_leaves": 20,
    "learning_rate": 0.05,
    "min_child_samples": 300,
    "reg_alpha": 0.5,
    "reg_lambda": 1.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}


def _normalize_training_frame(df_exec: pd.DataFrame) -> pd.DataFrame:
    """Own an execution frame whose date keys are on the JST contract."""
    if not isinstance(df_exec, pd.DataFrame) or df_exec.empty:
        raise ValueError("df_exec must be a non-empty DataFrame")
    frame = df_exec.copy(deep=True)
    frame.index = normalize_jst_index(frame.index)
    if frame.index.has_duplicates:
        raise ValueError("df_exec contains duplicate JST trade dates after normalization")
    return frame.sort_index()


def _normalize_training_dates(values: pd.DatetimeIndex | list[Any]) -> pd.DatetimeIndex:
    """Normalize caller-supplied training dates to unique JST date keys."""
    dates = pd.DatetimeIndex([normalize_jst_date(value) for value in values])
    if dates.has_duplicates:
        raise ValueError("training dates contain duplicate JST trade dates")
    return dates.sort_values()


def _build_training_decision_model(run_cfg: ProductionV2RunConfig) -> ProductionV2Model:
    """Build the same BLPX-backed V2 model used by live and backtest runs."""
    parsed = parse_run_config(run_cfg)
    # ``build_blpx_model`` is the canonical production factory.  Constructing
    # a small AppConfig wrapper keeps this research-only entry point on the
    # same dependency wiring without importing runner lifecycle code.
    app_config = AppConfig(v2=parsed)
    return ProductionV2Model(parsed, blpx_model=build_blpx_model(app_config))


def _collect_training_data(
    train_dates: pd.DatetimeIndex,
    df_exec: pd.DataFrame,
    y_target: np.ndarray,
    gap_input_dir: Path,
    run_cfg: ProductionV2RunConfig,
    market_vol: pd.DataFrame,
    per_ticker_interactions: bool = False,
    adr_df: pd.DataFrame | None = None,
    target_type: str = "raw",
    open_910_returns: pd.DataFrame | None = None,
    historical_inputs: HistoricalInputs | None = None,
    decision_model: ProductionV2Model | None = None,
) -> pd.DataFrame:
    """Collect point-in-time ticker rows from the canonical V2 decision path."""
    rows: list[dict[str, Any]] = []
    df_exec = _normalize_training_frame(df_exec)
    train_dates = _normalize_training_dates(train_dates)
    if open_910_returns is None:
        open_910_returns = build_open_910_returns(df_exec, JP_TICKERS)
    if historical_inputs is None:
        historical_inputs = HistoricalInputs(
            df_exec,
            open_910_returns=open_910_returns,
            adr_features_frame=adr_df,
            source="ml_overlay_training",
        )
    if decision_model is None:
        decision_model = _build_training_decision_model(run_cfg)
    lake = PITDataLake(df_exec)
    for date in train_dates:
        date_str = date.strftime("%Y-%m-%d")
        try:
            decision_as_of = date + pd.Timedelta(hours=9, minutes=10)
            snapshot = (
                lake.get_execution_snapshot(decision_as_of, historical_inputs.open_910_returns)
                if historical_inputs.open_910_returns is not None
                and all(f"jp_open_trade_{ticker}" in df_exec for ticker in JP_TICKERS)
                else lake.get_snapshot(decision_as_of)
            )
            sig_date = df_exec.loc[date].get("sig_date") if "sig_date" in df_exec.columns else None
            if sig_date is not None and pd.isna(sig_date):
                sig_date = None
            inputs = DecisionInputs(
                known=snapshot.to_known_inputs(
                    sig_date=sig_date,
                    observed_at={
                        "us_returns": f"{date.date()} 09:00",
                        "jp_gap_returns": f"{date.date()} 09:10",
                        "jp_betas": f"{date.date()} 09:10",
                        "topix_night_return": f"{date.date()} 09:10",
                        "current_prices": f"{date.date()} 09:10",
                        "prev_closes": f"{date.date()} 09:10",
                    },
                    source="ml_overlay_training",
                ),
                historical=historical_inputs,
                gap_input_dir=gap_input_dir,
                use_file_cache=True,
            )
            v2 = decision_model.decide(
                inputs=inputs,
                overlay_enabled=False,
                use_file_cache=True,
            )
        except Exception as exc:
            logger.warning("[%s] Skipping V2 generation: %s", date_str, exc)
            continue
        fallback = v2.fallback
        if fallback.get("gap_data_missing", False) or fallback.get("audit_failure", False):
            continue

        topix_night = float(snapshot.topix_night_return)
        scores = v2.scores
        for j, ticker in enumerate(JP_TICKERS):
            score = float(scores[j])
            side = 1.0 if score > 0 else -1.0
            realized = float(y_target[df_exec.index.get_loc(date), j])
            if not np.isfinite(realized):
                logger.warning(
                    "[%s] Skipping %s: realized training target is not finite",
                    date_str,
                    ticker,
                )
                continue
            target = side * realized - ROUND_TRIP_COST
            gap = float(snapshot.jp_gap_returns[j])
            beta = float(snapshot.jp_betas[j])
            gap_idio = gap - beta * topix_night
            market_vol_value = float(market_vol.loc[date, ticker])
            score_x_gap = score * gap
            score_x_gap_idio = score * gap_idio
            mu_gap_value = float(v2.mu_gap[j])
            adr_return = 0.0
            if adr_df is not None:
                try:
                    adr_return = float(adr_df.loc[date, f"adr_{ticker}"])
                except Exception:
                    logger.warning(
                        "[%s] ADR feature missing for %s during training/collection; "
                        "using 0.0 fallback.",
                        date,
                        ticker,
                    )
            adr_return = float(_safe(np.array([adr_return]))[0])
            record: dict[str, Any] = {
                "trade_date": date,
                "ticker": ticker,
                "score": score,
                "mu_gap": mu_gap_value,
                "sigma_gap": float(v2.sigma_gap[j]),
                "gap": gap,
                "gap_idio": gap_idio,
                "topix_night": topix_night,
                "market_vol_20d": market_vol_value,
                "score_x_gap": score_x_gap,
                "score_x_gap_idio": score_x_gap_idio,
                "abs_score": abs(score),
                "abs_gap": abs(gap),
                "target": target,
                "adr_return": adr_return,
                "adr_x_score": adr_return * score,
                "adr_x_gap": adr_return * gap,
                "adr_x_gap_idio": adr_return * gap_idio,
                "adr_x_mu_gap": adr_return * mu_gap_value,
                "abs_adr": abs(adr_return),
            }
            if per_ticker_interactions:
                for ticker2 in JP_TICKERS:
                    current = int(ticker2 == ticker)
                    record[f"ticker_{ticker2}_score"] = current * score
                    record[f"ticker_{ticker2}_gap"] = current * gap
                    record[f"ticker_{ticker2}_score_x_gap"] = current * score_x_gap
            rows.append(record)

    train_df = pd.DataFrame(rows)
    if train_df.empty:
        return train_df
    y_raw = _safe(train_df["target"].to_numpy())
    x = _safe(train_df["score"].to_numpy())
    if target_type == "residual":
        b, a = np.polyfit(x, y_raw, 1) if np.var(x) > 1e-12 else (0.0, 0.0)
        train_df["target"] = y_raw - (a + b * x)
    elif target_type == "residual_sign":
        b, a = np.polyfit(x, y_raw, 1) if np.var(x) > 1e-12 else (0.0, 0.0)
        train_df["target"] = (y_raw - (a + b * x) > 0.0).astype(int)
    elif target_type == "classification":
        train_df["target"] = (y_raw > 0.0).astype(int)
    return train_df


def _train_overlay_lgbm(
    train_df: pd.DataFrame,
    lgbm_kwargs: dict[str, Any] | None = None,
    use_ticker: bool = True,
    use_classification: bool = False,
    per_ticker_interactions: bool = False,
    p_trade_scale: float = 1.0,
) -> MLOrderOverlayModel:
    """Fit LightGBM using the exact feature order consumed by production."""
    import lightgbm as lgb

    cont_cols = overlay_continuous_columns(per_ticker_interactions)
    feature_cols = cont_cols + (["ticker"] if use_ticker else [])
    fit_df = train_df[feature_cols + ["target"]].copy()
    if use_ticker:
        fit_df["ticker"] = pd.Categorical(fit_df["ticker"], categories=JP_TICKERS)
    raw_target = fit_df["target"].to_numpy(dtype=float)
    if not np.isfinite(raw_target).all():
        raise ValueError("training target contains non-finite values")
    target = raw_target
    target_std = float(np.nanstd(target)) if not np.all(target == 0.0) else 1.0
    target_std = max(target_std, 1e-8)
    if use_classification:
        fit_df["label"] = (fit_df["target"] > 0).astype(int)
        y = fit_df["label"].values
        fitted = lgb.LGBMClassifier(**(lgbm_kwargs or DEFAULT_LGBM_KWARGS))
    else:
        y = target
        fitted = lgb.LGBMRegressor(**(lgbm_kwargs or DEFAULT_LGBM_KWARGS))
    fitted.fit(fit_df[feature_cols], _safe(y))
    return MLOrderOverlayModel(
        lgbm=fitted,
        cont_cols=cont_cols,
        target_std=target_std,
        use_ticker=use_ticker,
        use_classification=use_classification,
        per_ticker_interactions=per_ticker_interactions,
        p_trade_scale=p_trade_scale,
    )


def _train_overlay_model_impl(
    df_exec: pd.DataFrame,
    gap_input_dir: Path,
    run_cfg: ProductionV2RunConfig,
    train_start: str,
    train_end: str,
    output_dir: Path,
    lgbm_kwargs: dict[str, Any] | None = None,
    use_ticker: bool = True,
    use_classification: bool = False,
    per_ticker_interactions: bool = False,
    p_trade_scale: float = 1.0,
    target_type: str = "raw",
) -> MLOrderOverlayModel:
    """Train, provenance-stamp, and publish an overlay artifact."""
    minimum_start = pd.Timestamp("2015-01-05")
    train_start_ts = normalize_jst_date(train_start)
    train_end_ts = normalize_jst_date(train_end)
    if train_start_ts < minimum_start:
        raise ValueError(
            f"ML overlay training must start on or after {minimum_start.date()}"
        )
    df_exec = _normalize_training_frame(df_exec)
    market_vol = _precompute_market_vol(df_exec)
    # Keep the intraday source explicit.  With a supplied frame the adapter
    # does not reopen the process-global 5-minute cache; missing cells retain
    # the documented open-to-close fallback in the pure target arithmetic.
    open_910_returns = build_open_910_returns(df_exec, JP_TICKERS)
    # Training dates are contract dates in the JST clock.  Normalize explicit
    # timezone offsets before filtering any run-owned frame.
    open_910_returns = open_910_returns.loc[open_910_returns.index <= train_end_ts].copy()
    y_target = compute_jp_target_returns(
        df_exec,
        JP_TICKERS,
        open_910_returns=open_910_returns,
        allow_implicit_io=True,
    )
    adr_df = load_adr_features()
    if adr_df is not None:
        # Keep injected/run-owned adapters on the same JST date contract as
        # the filesystem loader before slicing the historical frame.
        adr_df = normalize_adr_features(adr_df)
    if adr_df is not None:
        adr_df = adr_df.loc[adr_df.index <= train_end_ts].copy()
        logger.info("Loaded ADR features for overlay training: %s", DEFAULT_ADR_FEATURES_PATH)
    dates = df_exec.index
    train_dates = dates[(dates >= train_start_ts) & (dates <= train_end_ts)]
    logger.info("Training overlay model on %d dates (%s -> %s)", len(train_dates), train_start, train_end)
    macro_prices = None
    if run_cfg.macro_kappa_enabled or run_cfg.macro_direction_enabled:
        try:
            macro_prices = macro_data.load_macro_prices(
                start=df_exec.index.min().strftime("%Y-%m-%d"),
                end=(train_end_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                period="max",
            )
        except Exception as exc:
            logger.warning("Failed to load run-owned macro prices for training: %s", exc)

    rank_reversal_signals = None
    if run_cfg.cs_overlay_enabled:
        rank_reversal_signals = load_rank_reversal_frame(
            Path(gap_input_dir),
            train_dates,
            file_pattern=run_cfg.cs_rank_reversal_file_pattern,
        )

    pit_ir_history: dict[str, np.ndarray] = {}
    pit_history_trade_dates: dict[str, np.ndarray] = {}
    for date in train_dates:
        date_str = date.strftime("%Y-%m-%d")
        history_ir, _alerts, history_dates = load_pit_ir_history(
            Path(gap_input_dir), date_str
        )
        pit_ir_history[date_str] = history_ir
        pit_history_trade_dates[date_str] = history_dates

    historical_inputs = HistoricalInputs(
        df_exec,
        open_910_returns=open_910_returns,
        macro_prices=macro_prices,
        adr_features_frame=adr_df,
        rank_reversal_signals=rank_reversal_signals,
        pit_ir_history=pit_ir_history,
        pit_history_trade_dates=pit_history_trade_dates,
        source="ml_overlay_training",
        observed_at_by_date={
            date.strftime("%Y-%m-%d"): {
                "open_910_returns": f"{date.date()} 09:10",
                "macro_prices": f"{date.date()} 09:00",
                "adr_features": f"{date.date()} 09:00",
                "rank_reversal_signals": f"{date.date()} 09:00",
                "pit_ir_history": f"{date.date()} 09:10",
            }
            for date in train_dates
        },
    )
    decision_model = _build_training_decision_model(run_cfg)
    train_df = _collect_training_data(
        train_dates,
        df_exec,
        y_target,
        Path(gap_input_dir),
        run_cfg,
        market_vol,
        per_ticker_interactions=per_ticker_interactions,
        adr_df=adr_df,
        target_type=target_type,
        open_910_returns=open_910_returns,
        historical_inputs=historical_inputs,
        decision_model=decision_model,
    )
    if target_type in ("classification", "residual_sign"):
        use_classification = True
    if train_df.empty:
        raise ValueError("No training samples collected.")
    model = _train_overlay_lgbm(
        train_df,
        lgbm_kwargs=lgbm_kwargs,
        use_ticker=use_ticker,
        use_classification=use_classification,
        per_ticker_interactions=per_ticker_interactions,
        p_trade_scale=p_trade_scale,
    )
    # Include column labels, dtypes, index metadata, and values so ticker
    # reordering cannot silently reuse an artifact trained on another schema.
    training_frame = df_exec.loc[df_exec.index <= train_end_ts]
    data_hash = dataframe_fingerprint(training_frame)
    config_hash = hashlib.sha256(
        json.dumps(run_cfg.model_dump(), sort_keys=True, default=str).encode()
    ).hexdigest()
    training_metadata = {
        "metadata_status": "verified",
        "train_start": train_start_ts.strftime("%Y-%m-%d"),
        "train_end": train_end_ts.strftime("%Y-%m-%d"),
        "label_asof_end": train_end_ts.strftime("%Y-%m-%d"),
        "data_hash": data_hash,
        "config_hash": config_hash,
        "open_910_hash": dataframe_fingerprint(open_910_returns),
        "adr_features_hash": None if adr_df is None else dataframe_fingerprint(adr_df),
        "macro_prices_hash": None if macro_prices is None else dataframe_fingerprint(macro_prices),
        "rank_reversal_hash": (
            None
            if rank_reversal_signals is None
            else dataframe_fingerprint(rank_reversal_signals)
        ),
        "pit_history_dates": sorted(pit_history_trade_dates),
        "historical_inputs_hash": historical_inputs.fingerprint,
        "input_snapshot": (
            "df_exec+open_910_returns+macro_prices+adr_features+"
            "rank_reversal+pit_history"
        ),
        "code_revision": os.environ.get("GIT_COMMIT", "unknown"),
    }
    object.__setattr__(model, "metadata", training_metadata)
    save_overlay_model(model, Path(output_dir), training_metadata=training_metadata)
    return model


def _record_training_event(
    *,
    status: str,
    parameters: dict[str, Any],
    metrics: dict[str, Any],
    registry_path: Path | None,
) -> None:
    """Append a completion/interruption record without masking training errors."""
    try:
        from research.experiment_utils import record_simple_experiment

        record_simple_experiment(
            name="ml_overlay_training",
            hypothesis="Train the production ML order overlay on a PIT historical window.",
            parameters=parameters,
            metrics={"status": status, **metrics},
            registry_path=registry_path,
        )
    except Exception as exc:  # pragma: no cover - registry failure is observational
        logger.warning("Could not record ML overlay training %s event: %s", status, exc)


def train_overlay_model(
    df_exec: pd.DataFrame,
    gap_input_dir: Path,
    run_cfg: ProductionV2RunConfig,
    train_start: str,
    train_end: str,
    output_dir: Path,
    lgbm_kwargs: dict[str, Any] | None = None,
    use_ticker: bool = True,
    use_classification: bool = False,
    per_ticker_interactions: bool = False,
    p_trade_scale: float = 1.0,
    target_type: str = "raw",
    registry_path: Path | None = None,
) -> MLOrderOverlayModel:
    """Train the overlay and record either completion or interruption."""
    parameters: dict[str, Any] = {
        "train_start": train_start,
        "train_end": train_end,
        "gap_input_dir": str(gap_input_dir),
        "output_dir": str(output_dir),
        "run_config": run_cfg.model_dump(mode="json"),
        "use_ticker": use_ticker,
        "use_classification": use_classification,
        "per_ticker_interactions": per_ticker_interactions,
        "p_trade_scale": p_trade_scale,
        "target_type": target_type,
    }
    try:
        model = _train_overlay_model_impl(
            df_exec=df_exec,
            gap_input_dir=gap_input_dir,
            run_cfg=run_cfg,
            train_start=train_start,
            train_end=train_end,
            output_dir=output_dir,
            lgbm_kwargs=lgbm_kwargs,
            use_ticker=use_ticker,
            use_classification=use_classification,
            per_ticker_interactions=per_ticker_interactions,
            p_trade_scale=p_trade_scale,
            target_type=target_type,
        )
    except BaseException as exc:
        _record_training_event(
            status="interrupted",
            parameters=parameters,
            metrics={"error_type": type(exc).__name__, "error": str(exc)[:500]},
            registry_path=registry_path,
        )
        raise

    _record_training_event(
        status="completed",
        parameters=parameters,
        metrics={
            "n_observations": int(len(df_exec)),
            "data_hash": model.metadata.get("data_hash", ""),
            "config_hash": model.metadata.get("config_hash", ""),
            "artifact_version": model.metadata.get("artifact_version", ""),
        },
        registry_path=registry_path,
    )
    return model


__all__ = ["train_overlay_model"]
