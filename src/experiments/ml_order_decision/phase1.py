"""Phase 1: per-ticker gap-coefficient Ridge regression overlay.

This module trains a simple Ridge regression that predicts the post-cost
ticker-level return (``side * realized - cost``) from ticker, score, gap and
auxiliary features.  The predicted value is mapped to a ``p_trade`` probability
which is then used to re-scale V2 ``mu_gap / sigma_gap`` scores.

To keep the cost/execution model identical between baseline and overlay, the
overlay is applied by monkey-patching ``generate_v2_production_portfolio`` and
running ``BacktestEngine.run_v2_backtest`` a second time.

This is an *experiment* module; it lives outside ``src/leadlag/`` to avoid
contaminating the production path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import Ridge

from leadlag.config.schemas import AppConfig, ProductionV2RunConfig
from leadlag.core.portfolio import solve_baseline_style
from leadlag.core.signal import build_weights_minvar
from leadlag.data.tickers import JP_TICKERS
from leadlag.execution.backtester import BacktestEngine
from leadlag.execution.config import build_app_config_from_dict
from leadlag.models.production_v2 import generate_v2_production_portfolio
from leadlag.models.sre import compute_jp_target_returns

logger = logging.getLogger(__name__)


def _resolve_app_config(cfg: AppConfig | dict) -> AppConfig:
    """Return a validated ``AppConfig`` from either a Pydantic model or a raw dict."""
    if isinstance(cfg, AppConfig):
        return cfg
    return build_app_config_from_dict(cfg)


def _resolve_run_cfg(cfg: AppConfig | dict) -> ProductionV2RunConfig:
    """Return a validated ``ProductionV2RunConfig`` from an ``AppConfig`` or a raw dict."""
    if isinstance(cfg, AppConfig):
        return cfg.v2
    return ProductionV2RunConfig.model_validate(cfg)


SLIPPAGE_BPS_PER_SIDE = 5.0
ROUND_TRIP_COST = 2.0 * SLIPPAGE_BPS_PER_SIDE / 10000.0
TRADING_DAYS = 245

# VIX feature names (market-level z-scores and interactions with per-ticker signals)
VIX_BASE_COLS = ["us_vix_z", "jp_vix_z", "vix_spread_z"]
VIX_INTERACTION_SUFFIXES = ["x_score", "x_gap", "x_score_x_gap", "x_score_x_gap_idio"]
VIX_FEATURE_COLS = VIX_BASE_COLS + [
    f"{base}_{suffix}"
    for base in VIX_BASE_COLS
    for suffix in VIX_INTERACTION_SUFFIXES
]


@dataclass(frozen=True)
class FitResult:
    """Container for a fitted overlay model."""

    ridge: Ridge
    feature_cols: list[str]
    cont_cols: list[str]
    ticker_cols: list[str]
    train_mean: pd.Series
    train_std: pd.Series
    target_std: float
    per_ticker_interactions: bool


def _safe(arr: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf with 0.0."""
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _sigmoid(x: np.ndarray, scale: float) -> np.ndarray:
    """Numerically stable sigmoid centered at 0 with scale."""
    return expit(x / max(scale, 1e-8))


def _add_vix_to_rec(
    rec: dict,
    vix_features: pd.DataFrame | None,
    date: pd.Timestamp,
    score: float,
    gap: float,
    score_x_gap: float,
    score_x_gap_idio: float,
) -> None:
    """Append VIX z-score features and interactions to a record.

    VIX is used as of the previous close (already shifted by the caller).
    Missing or NaN values are filled with 0.0 so the model is not exposed
    to future information.
    """
    if vix_features is None:
        return
    if date in vix_features.index:
        v = vix_features.loc[date]
        us_z = float(np.nan_to_num(v.get("us_vix_z", 0.0), nan=0.0))
        jp_z = float(np.nan_to_num(v.get("jp_vix_z", 0.0), nan=0.0))
        spread_z = float(np.nan_to_num(v.get("vix_spread_z", 0.0), nan=0.0))
    else:
        us_z = jp_z = spread_z = 0.0
    for z, name in [(us_z, "us_vix_z"), (jp_z, "jp_vix_z"), (spread_z, "vix_spread_z")]:
        rec[name] = z
        rec[f"{name}_x_score"] = z * score
        rec[f"{name}_x_gap"] = z * gap
        rec[f"{name}_x_score_x_gap"] = z * score_x_gap
        rec[f"{name}_x_score_x_gap_idio"] = z * score_x_gap_idio


def _recompute_w_pre(
    scores: np.ndarray,
    Omega_gap: np.ndarray,
    run_cfg: object,
) -> np.ndarray:
    """Recompute ``w_pre`` from (possibly adjusted) scores using V2 logic."""
    n_j = len(JP_TICKERS)
    sorted_idx = np.argsort(scores)
    short_idx = sorted_idx[: run_cfg.short_count]
    long_idx = sorted_idx[-run_cfg.long_count :]

    if run_cfg.minvar_enabled:
        w_minvar = build_weights_minvar(
            signal=scores,
            q=float(run_cfg.long_count) / n_j,
            n_j=n_j,
            Sigma_YY=Omega_gap,
            alpha=run_cfg.minvar_alpha,
            enforce_sign=False,
        )
        w_pre = w_minvar * (run_cfg.baseline_gross / 2.0)
    else:
        w_pre = solve_baseline_style(
            scores,
            long_idx,
            short_idx,
            baseline_gross=run_cfg.baseline_gross,
        )
    return w_pre


def _precompute_market_vol(df_exec: pd.DataFrame) -> pd.DataFrame:
    """Pre-compute PIT 20-day rolling mean absolute |r_oc| per ticker."""
    oc_cols = [f"jp_oc_{tk}" for tk in JP_TICKERS]
    oc_df = df_exec[oc_cols].copy()
    oc_df.columns = JP_TICKERS
    vol = oc_df.abs().rolling(window=20, min_periods=5).mean().shift(1)
    return vol


def _build_ticker_features(
    df_exec: pd.DataFrame,
    v2_result: dict,
    date: pd.Timestamp,
    market_vol: pd.DataFrame,
    per_ticker_interactions: bool = False,
    vix_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a per-ticker feature DataFrame for one date."""
    scores = v2_result["scores"]
    mu_gap = v2_result["mu_gap"]
    sigma_gap = v2_result["sigma_gap"]

    row = df_exec.loc[date]
    topix_night = float(row["topix_night_return"])

    records = []
    for i, tk in enumerate(JP_TICKERS):
        gap = float(row[f"jp_gap_{tk}"])
        beta = float(row[f"jp_beta_{tk}"])
        gap_idio = gap - beta * topix_night
        market_vol_val = (
            float(market_vol.loc[date, tk]) if market_vol is not None else 0.0
        )
        score = float(scores[i])
        score_x_gap = score * gap
        score_x_gap_idio = score * gap_idio

        rec = {
            "ticker": tk,
            "score": score,
            "mu_gap": float(mu_gap[i]),
            "sigma_gap": float(sigma_gap[i]),
            "gap": gap,
            "gap_idio": gap_idio,
            "topix_night": topix_night,
            "market_vol_20d": market_vol_val,
            "score_x_gap": score_x_gap,
            "score_x_gap_idio": score_x_gap_idio,
            "abs_score": abs(score),
            "abs_gap": abs(gap),
        }
        if per_ticker_interactions:
            for tk2 in JP_TICKERS:
                is_current = int(tk2 == tk)
                rec[f"ticker_{tk2}_score"] = is_current * score
                rec[f"ticker_{tk2}_gap"] = is_current * gap
                rec[f"ticker_{tk2}_gap_idio"] = is_current * gap_idio
                rec[f"ticker_{tk2}_score_x_gap"] = is_current * score_x_gap
                rec[f"ticker_{tk2}_score_x_gap_idio"] = is_current * score_x_gap_idio
        _add_vix_to_rec(
            rec, vix_features, date, score, gap, score_x_gap, score_x_gap_idio
        )
        records.append(rec)

    return pd.DataFrame(records)


def _prepare_design_matrix(
    df: pd.DataFrame,
    cont_cols: list[str],
    ticker_cols: list[str],
) -> pd.DataFrame:
    """One-hot encode tickers and align columns."""
    x = pd.get_dummies(df[["ticker"] + cont_cols], columns=["ticker"], drop_first=False)
    for col in ticker_cols:
        if col not in x.columns:
            x[col] = 0.0
    return x[cont_cols + ticker_cols]


def _standardize(
    x: pd.DataFrame,
    cont_cols: list[str],
    train_mean: pd.Series | None = None,
    train_std: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Standardize continuous columns; leave one-hot tickers untouched."""
    if train_mean is None:
        train_mean = x[cont_cols].mean()
    if train_std is None:
        train_std = x[cont_cols].std().replace(0.0, 1.0)
    x = x.copy()
    x[cont_cols] = (x[cont_cols] - train_mean) / train_std
    return x, train_mean, train_std


def _collect_training_data(
    train_dates: pd.DatetimeIndex,
    df_exec: pd.DataFrame,
    y_target: np.ndarray,
    gap_input_dir: Path,
    cfg: AppConfig | dict,
    market_vol: pd.DataFrame,
    per_ticker_interactions: bool = False,
    vix_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Loop over training dates and collect per-ticker features + target."""
    run_cfg = _resolve_run_cfg(cfg)

    gap_cols = [f"jp_gap_{tk}" for tk in JP_TICKERS]
    beta_cols = [f"jp_beta_{tk}" for tk in JP_TICKERS]

    rows = []
    for i, date in enumerate(train_dates):
        if i % 100 == 0:
            logger.info(
                "Building training data: %d/%d (%s)",
                i + 1,
                len(train_dates),
                date.date(),
            )

        date_str = date.strftime("%Y-%m-%d")
        try:
            v2 = generate_v2_production_portfolio(date_str, gap_input_dir, cfg=run_cfg)
        except Exception as e:
            logger.warning("[%s] V2 generation failed: %s", date_str, e)
            continue

        if v2["fallback"]["gap_data_missing"]:
            continue

        scores = v2["scores"]
        row = df_exec.loc[date]
        topix_night = float(row["topix_night_return"])

        for j, tk in enumerate(JP_TICKERS):
            score = float(scores[j])
            side = 1.0 if score > 0 else -1.0
            realized = float(y_target[df_exec.index.get_loc(date), j])
            target = side * realized - ROUND_TRIP_COST

            gap = float(row[gap_cols[j]])
            beta = float(row[beta_cols[j]])
            gap_idio = gap - beta * topix_night
            market_vol_val = float(market_vol.loc[date, tk])
            score_x_gap = score * gap
            score_x_gap_idio = score * gap_idio

            rec = {
                "ticker": tk,
                "score": score,
                "mu_gap": float(v2["mu_gap"][j]),
                "sigma_gap": float(v2["sigma_gap"][j]),
                "gap": gap,
                "gap_idio": gap_idio,
                "topix_night": topix_night,
                "market_vol_20d": market_vol_val,
                "score_x_gap": score_x_gap,
                "score_x_gap_idio": score_x_gap_idio,
                "abs_score": abs(score),
                "abs_gap": abs(gap),
                "target": target,
            }
            if per_ticker_interactions:
                for tk2 in JP_TICKERS:
                    is_current = int(tk2 == tk)
                    rec[f"ticker_{tk2}_score"] = is_current * score
                    rec[f"ticker_{tk2}_gap"] = is_current * gap
                    rec[f"ticker_{tk2}_gap_idio"] = is_current * gap_idio
                    rec[f"ticker_{tk2}_score_x_gap"] = is_current * score_x_gap
                    rec[f"ticker_{tk2}_score_x_gap_idio"] = is_current * score_x_gap_idio
            _add_vix_to_rec(
                rec, vix_features, date, score, gap, score_x_gap, score_x_gap_idio
            )
            rows.append(rec)

    return pd.DataFrame(rows)


def _train_overlay(
    train_df: pd.DataFrame,
    ridge_alpha: float = 1.0,
    per_ticker_interactions: bool = False,
) -> FitResult:
    """Fit a Ridge regression on the collected training data."""
    cont_cols = [
        "score",
        "mu_gap",
        "sigma_gap",
        "gap",
        "gap_idio",
        "topix_night",
        "market_vol_20d",
        "score_x_gap",
        "score_x_gap_idio",
        "abs_score",
        "abs_gap",
    ]
    if per_ticker_interactions:
        # Replace score/gap/score_x_gap/mu_gap with per-ticker versions; drop score_x_gap_idio to avoid near-collinearity
        cont_cols = [c for c in cont_cols if c not in ("score", "mu_gap", "gap", "score_x_gap", "score_x_gap_idio")]
        per_ticker_cols = [
            f"ticker_{tk}_{suffix}"
            for tk in JP_TICKERS
            for suffix in ("score", "gap", "score_x_gap")
        ]
        cont_cols = cont_cols + per_ticker_cols
    ticker_cols = [f"ticker_{tk}" for tk in JP_TICKERS]

    x = _prepare_design_matrix(train_df, cont_cols, ticker_cols)
    y = train_df["target"].values

    x, train_mean, train_std = _standardize(x, cont_cols)

    ridge = Ridge(alpha=ridge_alpha)
    ridge.fit(_safe(x.values), _safe(y))

    target_std = float(np.std(y, ddof=1)) if len(y) > 1 else 1.0

    logger.info(
        "Ridge fitted: n=%d, features=%d, target_std=%.4f, per_ticker_interactions=%s",
        len(y),
        len(x.columns),
        target_std,
        per_ticker_interactions,
    )

    return FitResult(
        ridge=ridge,
        feature_cols=cont_cols + ticker_cols,
        cont_cols=cont_cols,
        ticker_cols=ticker_cols,
        train_mean=train_mean,
        train_std=train_std,
        target_std=target_std,
        per_ticker_interactions=per_ticker_interactions,
    )


def _predict_p_trade(
    features: pd.DataFrame,
    model: FitResult,
) -> np.ndarray:
    """Predict ``p_trade`` from a feature DataFrame."""
    x = _prepare_design_matrix(features, model.cont_cols, model.ticker_cols)
    x, _, _ = _standardize(x, model.cont_cols, model.train_mean, model.train_std)
    y_hat = model.ridge.predict(_safe(x.values))
    return _sigmoid(y_hat, model.target_std)


def make_overlay_generator(
    df_exec: pd.DataFrame,
    market_vol: pd.DataFrame,
    model: FitResult,
    original_generate: callable,
) -> callable:
    """Return a wrapped ``generate_v2_production_portfolio`` that applies the overlay."""

    def _wrapped(trade_date: str, gap_input_dir: Path, cfg: ProductionV2RunConfig | dict) -> dict:
        result = original_generate(trade_date, gap_input_dir, cfg)

        if result["fallback"]["gap_data_missing"]:
            return result

        date = pd.Timestamp(trade_date)
        features = _build_ticker_features(
            df_exec, result, date, market_vol,
            per_ticker_interactions=model.per_ticker_interactions,
        )
        p_trade = _predict_p_trade(features, model)
        score_adjusted = result["scores"] * p_trade

        w_pre_overlay = _recompute_w_pre(
            score_adjusted, result["Omega_gap"], result["run_config"]
        )
        mult = result["pit_binning"]["multiplier"]
        w_final_overlay = w_pre_overlay * mult

        # Zero out tiny weights and preserve sign
        w_final_overlay[np.abs(w_final_overlay) < 1e-8] = 0.0

        result = dict(result)
        result["w_final"] = w_final_overlay
        # Update summary so downstream diagnostics are not misleading
        summary = dict(result.get("summary", {}))
        summary["overlay_applied"] = 1
        summary["p_trade_mean"] = float(np.mean(p_trade))
        summary["p_trade_std"] = float(np.std(p_trade))
        result["summary"] = summary
        return result

    return _wrapped


def _compute_metrics(daily_returns: pd.Series) -> dict:
    """Compute annualized Sharpe, AR, Vol, MDD."""
    dr = daily_returns.dropna()
    if len(dr) < 10:
        return {"sharpe": np.nan, "ar": np.nan, "vol": np.nan, "mdd": np.nan, "n": len(dr)}
    ar = float(dr.mean() * TRADING_DAYS)
    vol = float(dr.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe = ar / vol if vol > 0 else np.nan
    wealth = (1.0 + dr).cumprod()
    mdd = float(((wealth / wealth.cummax()) - 1.0).min())
    return {"sharpe": sharpe, "ar": ar, "vol": vol, "mdd": mdd, "n": len(dr)}


def run_phase1_experiment(
    df_exec: pd.DataFrame,
    gap_input_dir: Path,
    cfg: AppConfig | dict,
    output_dir: Path,
    train_start: str = "2020-01-06",
    train_end: str = "2022-12-31",
    test_start: str = "2023-01-01",
    test_end: str = "2024-12-31",
    side_leverage: float = 1.5,
    ridge_alpha: float = 1.0,
    per_ticker_interactions: bool = False,
    n_jobs: int = -1,
) -> dict:
    """Run the full Phase 1 experiment and save artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    app_config = _resolve_app_config(cfg)

    market_vol = _precompute_market_vol(df_exec)
    y_target = compute_jp_target_returns(df_exec, JP_TICKERS)

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

    # 1. Collect training data and fit overlay
    train_df = _collect_training_data(
        train_dates, df_exec, y_target, gap_input_dir, app_config, market_vol,
        per_ticker_interactions=per_ticker_interactions,
    )
    if train_df.empty:
        raise ValueError("No training samples collected.")

    model = _train_overlay(
        train_df,
        ridge_alpha=ridge_alpha,
        per_ticker_interactions=per_ticker_interactions,
    )

    # 2. Baseline V2 backtest
    logger.info("Running baseline V2 backtest...")
    baseline_result = BacktestEngine.run_v2_backtest(
        app_config,
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
    wrapped_generate = make_overlay_generator(
        df_exec, market_vol, model, original_generate
    )
    pv2.generate_v2_production_portfolio = wrapped_generate

    try:
        logger.info("Running overlay V2 backtest...")
        overlay_result = BacktestEngine.run_v2_backtest(
            app_config,
            gap_input_dir,
            df_exec,
            start_date=test_start,
            end_date=test_end,
            side_leverage=side_leverage,
            n_jobs=n_jobs,
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

    coef_df = pd.DataFrame(
        {"feature": model.feature_cols, "coef": model.ridge.coef_.tolist()}
    ).sort_values("coef", key=lambda s: s.abs(), ascending=False)
    coef_df.to_csv(output_dir / "ridge_coefficients.csv", index=False)

    return {
        "model": model,
        "baseline_result": baseline_result,
        "overlay_result": overlay_result,
        "baseline_metrics": baseline_metrics,
        "overlay_metrics": overlay_metrics,
        "output_dir": output_dir,
    }
