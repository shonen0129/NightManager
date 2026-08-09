"""Phase 2: LightGBM order decision overlay.

Uses the same per-ticker feature construction as Phase 1, but replaces the
Ridge regression with a ``lightgbm.LGBMRegressor``.  The predicted
contribution is still converted to ``p_trade`` via a sigmoid.

The overlay is again applied by monkey-patching ``generate_v2_production_portfolio``
and running ``BacktestEngine.run_v2_backtest`` for both baseline and overlay.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit

from experiments.ml_order_decision.phase1 import (
    _build_ticker_features,
    _collect_training_data,
    _compute_metrics,
    _precompute_market_vol,
    _recompute_w_pre,
)
from leadlag.data.tickers import JP_TICKERS
from leadlag.execution.backtester import BacktestEngine
from leadlag.models.sre import compute_jp_target_returns

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LightGBMModel:
    """Container for a fitted LightGBM overlay model."""

    lgbm: lgb.LGBMRegressor | lgb.LGBMClassifier
    cont_cols: list[str]
    target_std: float
    use_ticker: bool
    use_classification: bool


def _sigmoid(x: np.ndarray, scale: float) -> np.ndarray:
    return expit(x / max(scale, 1e-8))


def _train_overlay_lgbm(
    train_df: pd.DataFrame,
    lgbm_kwargs: dict | None = None,
    use_ticker: bool = True,
    use_classification: bool = False,
    per_ticker_interactions: bool = False,
) -> LightGBMModel:
    """Fit a LightGBM regressor on the collected training data."""
    # Derive continuous columns from the actual feature DataFrame; this allows
    # optional VIX features to be included without hard-coding column names.
    cont_cols = [c for c in train_df.columns if c not in ("target", "ticker")]
    if per_ticker_interactions:
        # Avoid collinearity between raw per-ticker features and their base forms.
        base_to_drop = {"score", "mu_gap", "gap", "score_x_gap", "score_x_gap_idio"}
        cont_cols = [c for c in cont_cols if c not in base_to_drop]

    # Optionally include ticker as categorical
    feature_cols = cont_cols + (["ticker"] if use_ticker else [])
    train_df = train_df[feature_cols + ["target"]].copy()
    if use_ticker:
        train_df["ticker"] = pd.Categorical(train_df["ticker"], categories=JP_TICKERS)

    if use_classification:
        train_df["label"] = (train_df["target"] > 0).astype(int)

    # Time-ordered split: last 10% as validation for early stopping
    n_val = max(1, int(0.1 * len(train_df)))
    train_part = train_df.iloc[:-n_val]
    val_part = train_df.iloc[-n_val:]

    target_std = float(np.std(train_part["target"].values, ddof=1)) if not use_classification and len(train_part) > 1 else 1.0

    default_kwargs = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 4,
        "num_leaves": 31,
        "min_child_samples": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    if lgbm_kwargs:
        default_kwargs.update(lgbm_kwargs)

    target_col = "label" if use_classification else "target"
    model_cls = lgb.LGBMClassifier if use_classification else lgb.LGBMRegressor
    model = model_cls(**default_kwargs)

    logger.info(
        "Training LightGBM %s: train=%d, validation=%d, features=%d",
        "classifier" if use_classification else "regressor",
        len(train_part),
        len(val_part),
        len(feature_cols),
    )

    model.fit(
        train_part[feature_cols],
        train_part[target_col],
        categorical_feature=["ticker"] if use_ticker else [],
        eval_set=[(val_part[feature_cols], val_part[target_col])],
        callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(period=0)],
    )

    logger.info(
        "LightGBM fitted: best_iter=%d, target_std=%.4f",
        model.best_iteration_,
        target_std,
    )

    return LightGBMModel(
        lgbm=model,
        cont_cols=cont_cols,
        target_std=target_std,
        use_ticker=use_ticker,
        use_classification=use_classification,
    )


def _predict_p_trade_lgbm(
    features: pd.DataFrame,
    model: LightGBMModel,
) -> np.ndarray:
    """Predict ``p_trade`` from a feature DataFrame using LightGBM."""
    feature_cols = model.cont_cols + (["ticker"] if model.use_ticker else [])
    x = features[feature_cols].copy()
    if model.use_ticker:
        x["ticker"] = pd.Categorical(x["ticker"], categories=JP_TICKERS)
    if model.use_classification:
        y_hat = model.lgbm.predict_proba(x)[:, 1]
        return y_hat
    y_hat = model.lgbm.predict(x)
    return _sigmoid(y_hat, model.target_std)


def make_overlay_generator_lgbm(
    df_exec: pd.DataFrame,
    market_vol: pd.DataFrame,
    model: LightGBMModel,
    original_generate: callable,
    p_trade_ema_span: float | None = None,
    use_ticker: bool = True,
    per_ticker_interactions: bool = False,
    vix_features: pd.DataFrame | None = None,
) -> callable:
    """Return a wrapped ``generate_v2_production_portfolio`` for LightGBM overlay.

    If ``p_trade_ema_span`` is set, a per-ticker exponential moving average of
    ``p_trade`` is applied to smooth day-to-day fluctuations.  This requires the
    overlay backtest to run sequentially (``n_jobs=1``) so that the EMA history is
    updated in calendar order.
    """

    ema_state: dict[str, float] = {}
    ema_dates: dict[str, pd.Timestamp] = {}
    ema_alpha = 2.0 / (p_trade_ema_span + 1.0) if p_trade_ema_span and p_trade_ema_span > 0 else 1.0

    def _wrapped(trade_date: str, gap_input_dir: Path, cfg: dict) -> dict:
        result = original_generate(trade_date, gap_input_dir, cfg)

        if result["fallback"]["gap_data_missing"]:
            return result

        date = pd.Timestamp(trade_date)
        features = _build_ticker_features(
            df_exec, result, date, market_vol,
            per_ticker_interactions=per_ticker_interactions,
            vix_features=vix_features,
        )
        p_trade = _predict_p_trade_lgbm(features, model)

        # Apply optional EMA smoothing
        if p_trade_ema_span and p_trade_ema_span > 0:
            for i, tk in enumerate(JP_TICKERS):
                prev = ema_state.get(tk)
                prev_date = ema_dates.get(tk)
                if prev is None or prev_date is None or (date - prev_date).days > 5:
                    ema_state[tk] = float(p_trade[i])
                else:
                    ema_state[tk] = ema_alpha * float(p_trade[i]) + (1.0 - ema_alpha) * prev
                ema_dates[tk] = date
            p_trade = np.array([ema_state[tk] for tk in JP_TICKERS], dtype=float)

        score_adjusted = result["scores"] * p_trade

        w_pre_overlay = _recompute_w_pre(
            score_adjusted, result["Omega_gap"], result["run_config"]
        )
        mult = result["pit_binning"]["multiplier"]
        w_final_overlay = w_pre_overlay * mult

        w_final_overlay[np.abs(w_final_overlay) < 1e-8] = 0.0

        result = dict(result)
        result["w_final"] = w_final_overlay
        summary = dict(result.get("summary", {}))
        summary["overlay_applied"] = 2
        summary["p_trade_mean"] = float(np.mean(p_trade))
        summary["p_trade_std"] = float(np.std(p_trade))
        result["summary"] = summary
        return result

    return _wrapped


DEFAULT_VIX_CACHE = Path(__file__).resolve().parents[3] / "market_data" / "vix_regime_overlay" / "vix_cache.csv"


def _prepare_vix_features(
    df_exec: pd.DataFrame,
    vix_cache_path: Path | str = DEFAULT_VIX_CACHE,
    window: int = 60,
    min_periods: int = 30,
) -> pd.DataFrame:
    """Build PIT 60-day log VIX z-scores and JP-US spread z-scores.

    The VIX value for a trade date is the previous business day's close
    (``vix_df.shift(1).reindex(..., method='ffill')``) so that no same-day
    VIX close leaks into the 9:10 decision.
    """
    vix_cache_path = Path(vix_cache_path)
    if not vix_cache_path.exists():
        raise FileNotFoundError(f"VIX cache not found: {vix_cache_path}")

    vix_df = pd.read_csv(vix_cache_path, index_col=0, parse_dates=True)
    vix_df.index = pd.to_datetime(vix_df.index).tz_localize(None).normalize()
    vix_df = vix_df[["us_vix", "jp_vix"]].astype(float)
    vix_df["vix_spread"] = vix_df["jp_vix"] - vix_df["us_vix"]

    # Use previous day's close and align to df_exec business days
    aligned = vix_df.shift(1).reindex(df_exec.index, method="ffill")

    out = pd.DataFrame(index=aligned.index)
    # z-scores on log VIX; rolling window ends at the previous close (PIT)
    for col in ("us_vix", "jp_vix"):
        log_vix = np.log(aligned[col].replace(0.0, np.nan)).ffill()
        roll_mean = log_vix.rolling(window=window, min_periods=min_periods).mean()
        roll_std = log_vix.rolling(window=window, min_periods=min_periods).std().replace(0.0, 1.0)
        out[f"{col}_z"] = ((log_vix - roll_mean) / roll_std).fillna(0.0)

    # z-score on raw spread (JP VIX - US VIX)
    spread = aligned["vix_spread"].ffill()
    spread_mean = spread.rolling(window=window, min_periods=min_periods).mean()
    spread_std = spread.rolling(window=window, min_periods=min_periods).std().replace(0.0, 1.0)
    out["vix_spread_z"] = ((spread - spread_mean) / spread_std).fillna(0.0)

    return out


def run_phase2_experiment(
    df_exec: pd.DataFrame,
    gap_input_dir: Path,
    cfg: dict,
    output_dir: Path,
    train_start: str = "2020-01-06",
    train_end: str = "2022-12-31",
    test_start: str = "2023-01-01",
    test_end: str = "2024-12-31",
    side_leverage: float = 1.5,
    n_jobs: int = -1,
    lgbm_kwargs: dict | None = None,
    p_trade_ema_span: float | None = None,
    use_ticker: bool = True,
    use_classification: bool = False,
    per_ticker_interactions: bool = False,
    vix_cache_path: Path | str | None = None,
) -> dict:
    """Run the full Phase 2 experiment and save artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    market_vol = _precompute_market_vol(df_exec)
    y_target = compute_jp_target_returns(df_exec, JP_TICKERS)

    vix_features = _prepare_vix_features(df_exec, vix_cache_path) if vix_cache_path is not None else None

    all_dates = df_exec.index
    train_dates = all_dates[(all_dates >= train_start) & (all_dates <= train_end)]
    test_dates = all_dates[(all_dates >= test_start) & (all_dates <= test_end)]

    if len(train_dates) == 0 or len(test_dates) == 0:
        raise ValueError("No dates in train or test range.")

    logger.info(
        "Train: %s -> %s (%d days)",
        train_dates[0].date(),
        train_dates[-1].date(),
        len(train_dates),
    )
    logger.info(
        "Test:  %s -> %s (%d days)",
        test_dates[0].date(),
        test_dates[-1].date(),
        len(test_dates),
    )

    # 1. Collect training data and fit LightGBM
    train_df = _collect_training_data(
        train_dates, df_exec, y_target, gap_input_dir, cfg, market_vol,
        per_ticker_interactions=per_ticker_interactions,
        vix_features=vix_features,
    )
    if train_df.empty:
        raise ValueError("No training samples collected.")

    model = _train_overlay_lgbm(
        train_df,
        lgbm_kwargs=lgbm_kwargs,
        use_ticker=use_ticker,
        use_classification=use_classification,
        per_ticker_interactions=per_ticker_interactions,
    )

    # 2. Baseline V2 backtest
    logger.info("Running baseline V2 backtest...")
    baseline_result = BacktestEngine.run_v2_backtest(
        cfg,
        gap_input_dir,
        df_exec,
        start_date=test_start,
        end_date=test_end,
        side_leverage=side_leverage,
        n_jobs=n_jobs,
    )

    # 3. Overlay V2 backtest via monkey-patch
    import leadlag.models.production_v2 as pv2

    original_generate = pv2.generate_v2_production_portfolio
    wrapped_generate = make_overlay_generator_lgbm(
        df_exec, market_vol, model, original_generate,
        p_trade_ema_span=p_trade_ema_span,
        use_ticker=use_ticker,
        per_ticker_interactions=per_ticker_interactions,
        vix_features=vix_features,
    )
    pv2.generate_v2_production_portfolio = wrapped_generate

    # EMA smoothing requires sequential execution so the per-ticker EMA history
    # is updated in calendar order.
    overlay_n_jobs = 1 if p_trade_ema_span and p_trade_ema_span > 0 else n_jobs

    try:
        logger.info("Running overlay V2 backtest (n_jobs=%s, EMA span=%s)...", overlay_n_jobs, p_trade_ema_span)
        overlay_result = BacktestEngine.run_v2_backtest(
            cfg,
            gap_input_dir,
            df_exec,
            start_date=test_start,
            end_date=test_end,
            side_leverage=side_leverage,
            n_jobs=overlay_n_jobs,
        )
    finally:
        pv2.generate_v2_production_portfolio = original_generate

    # 4. Metrics
    baseline_metrics = _compute_metrics(baseline_result["daily_returns"])
    overlay_metrics = _compute_metrics(overlay_result["daily_returns"])

    # 5. Save artifacts
    baseline_result["weights"].to_csv(output_dir / "baseline_weights.csv")
    overlay_result["weights"].to_csv(output_dir / "overlay_weights.csv")
    baseline_result["daily_returns"].to_csv(output_dir / "baseline_daily_returns.csv")
    overlay_result["daily_returns"].to_csv(output_dir / "overlay_daily_returns.csv")
    baseline_result["daily_turnover"].to_csv(output_dir / "baseline_daily_turnover.csv")
    overlay_result["daily_turnover"].to_csv(output_dir / "overlay_daily_turnover.csv")
    baseline_result["daily_gross_exps"].to_csv(output_dir / "baseline_daily_gross_exps.csv")
    overlay_result["daily_gross_exps"].to_csv(output_dir / "overlay_daily_gross_exps.csv")

    # Feature importance
    imp = pd.DataFrame(
        {
            "feature": model.lgbm.feature_name_,
            "importance": model.lgbm.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    imp.to_csv(output_dir / "lgbm_feature_importance.csv", index=False)

    return {
        "model": model,
        "baseline_result": baseline_result,
        "overlay_result": overlay_result,
        "baseline_metrics": baseline_metrics,
        "overlay_metrics": overlay_metrics,
        "output_dir": output_dir,
    }
