"""Production ML order-decision overlay for Residual-BLPX-RA v2.

Provides a LightGBM overlay that re-scales the V2 ``mu_gap / sigma_gap``
score vector using per-ticker interaction features:

    ticker × score
    ticker × gap
    ticker × score × gap

The model is trained once on historical PIT data and loaded at runtime.
If the model is missing, the overlay is disabled, or required market data is
unavailable, the function returns the original V2 result unchanged.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit

from leadlag.compliance.v2_auditor import run_numerical_audit
from leadlag.core.portfolio import solve_baseline_style
from leadlag.core.signal import build_weights_minvar
from leadlag.data.tickers import JP_TICKERS
from leadlag.models.production_v2 import generate_v2_production_portfolio
from leadlag.models.sre import compute_jp_target_returns

logger = logging.getLogger(__name__)

TRADING_DAYS = 245
SLIPPAGE_BPS_PER_SIDE = 5.0
ROUND_TRIP_COST = 2.0 * SLIPPAGE_BPS_PER_SIDE / 10000.0

DEFAULT_LGBM_KWARGS = {
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


@dataclass(frozen=True)
class MLOrderOverlayModel:
    """Container for a fitted LightGBM overlay model."""

    lgbm: Any  # lightgbm.LGBMRegressor | LGBMClassifier
    cont_cols: list[str]
    target_std: float
    use_ticker: bool
    use_classification: bool
    per_ticker_interactions: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe(arr: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf in place with finite values."""
    arr = np.asarray(arr, dtype=float)
    arr[np.isnan(arr) | np.isinf(arr)] = 0.0
    return arr


def _sigmoid(x: np.ndarray, scale: float) -> np.ndarray:
    """Numerically stable sigmoid centered at 0 with scale."""
    return expit(x / max(scale, 1e-8))


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
    trade_date: pd.Timestamp,
    market_vol: pd.DataFrame | None,
    per_ticker_interactions: bool = False,
) -> pd.DataFrame:
    """Build a per-ticker feature DataFrame for one trade date."""
    scores = v2_result["scores"]
    mu_gap = v2_result["mu_gap"]
    sigma_gap = v2_result["sigma_gap"]

    if trade_date not in df_exec.index:
        raise KeyError(f"Trade date {trade_date} not in df_exec")

    row = df_exec.loc[trade_date]
    topix_night = float(row["topix_night_return"])

    records = []
    for i, tk in enumerate(JP_TICKERS):
        gap = float(row[f"jp_gap_{tk}"])
        beta = float(row[f"jp_beta_{tk}"])
        gap_idio = gap - beta * topix_night
        market_vol_val = (
            float(market_vol.loc[trade_date, tk]) if market_vol is not None else 0.0
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
                rec[f"ticker_{tk2}_score_x_gap"] = is_current * score_x_gap
        records.append(rec)

    return pd.DataFrame(records)


def _predict_p_trade(
    features: pd.DataFrame,
    model: MLOrderOverlayModel,
) -> np.ndarray:
    """Predict ``p_trade`` from a feature DataFrame."""
    feature_cols = model.cont_cols + (["ticker"] if model.use_ticker else [])
    x = features[feature_cols].copy()
    if model.use_ticker:
        x["ticker"] = pd.Categorical(x["ticker"], categories=JP_TICKERS)

    for col in model.cont_cols:
        if col not in x.columns:
            x[col] = 0.0

    if model.use_classification:
        pred = model.lgbm.predict_proba(x)[:, 1]
    else:
        pred = model.lgbm.predict(x)

    return _sigmoid(pred, model.target_std)


def _recompute_w_pre(
    scores: np.ndarray,
    Omega_gap: np.ndarray,
    run_cfg: Any,
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


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------


def save_overlay_model(model: MLOrderOverlayModel, output_dir: Path) -> None:
    """Save overlay model artifact and metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "cont_cols": model.cont_cols,
        "target_std": float(model.target_std),
        "use_ticker": model.use_ticker,
        "use_classification": model.use_classification,
        "per_ticker_interactions": model.per_ticker_interactions,
        "n_tickers": len(JP_TICKERS),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    with open(output_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    logger.info("Overlay model saved to %s", output_dir)


def load_overlay_model(model_dir: Path) -> MLOrderOverlayModel:
    """Load overlay model artifact."""
    model_dir = Path(model_dir)
    model_path = model_dir / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Overlay model not found: {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    if not isinstance(model, MLOrderOverlayModel):
        raise TypeError(f"Loaded model is not an MLOrderOverlayModel: {type(model)}")

    logger.info("Overlay model loaded from %s", model_dir)
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _collect_training_data(
    train_dates: pd.DatetimeIndex,
    df_exec: pd.DataFrame,
    y_target: np.ndarray,
    gap_input_dir: Path,
    cfg: dict,
    market_vol: pd.DataFrame,
    per_ticker_interactions: bool = False,
) -> pd.DataFrame:
    """Collect per-ticker training rows for one or more train dates."""

    gap_cols = [f"jp_gap_{tk}" for tk in JP_TICKERS]
    beta_cols = [f"jp_beta_{tk}" for tk in JP_TICKERS]

    rows: list[dict] = []
    for date in train_dates:
        date_str = date.strftime("%Y-%m-%d")
        try:
            v2 = generate_v2_production_portfolio(
                trade_date=date_str,
                gap_input_dir=gap_input_dir,
                cfg=cfg,
            )
        except Exception as e:
            logger.warning("[%s] Skipping V2 generation: %s", date_str, e)
            continue

        if v2["fallback"]["gap_data_missing"]:
            continue

        row = df_exec.loc[date]
        topix_night = float(row["topix_night_return"])
        scores = v2["scores"]

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
                    rec[f"ticker_{tk2}_score_x_gap"] = is_current * score_x_gap
            rows.append(rec)

    return pd.DataFrame(rows)


def _train_overlay_lgbm(
    train_df: pd.DataFrame,
    lgbm_kwargs: dict | None = None,
    use_ticker: bool = True,
    use_classification: bool = False,
    per_ticker_interactions: bool = False,
) -> MLOrderOverlayModel:
    """Fit a LightGBM overlay model on the collected training data."""
    import lightgbm as lgb

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
        cont_cols = [
            c for c in cont_cols
            if c not in ("score", "mu_gap", "gap", "score_x_gap", "score_x_gap_idio")
        ]
        cont_cols = cont_cols + [
            f"ticker_{tk}_{suffix}"
            for tk in JP_TICKERS
            for suffix in ("score", "gap", "score_x_gap")
        ]

    feature_cols = cont_cols + (["ticker"] if use_ticker else [])
    train_df = train_df[feature_cols + ["target"]].copy()
    if use_ticker:
        train_df["ticker"] = pd.Categorical(train_df["ticker"], categories=JP_TICKERS)

    target = _safe(train_df["target"].values)
    target_std = float(np.nanstd(target)) if not np.all(target == 0.0) else 1.0
    target_std = max(target_std, 1e-8)

    if use_classification:
        train_df["label"] = (train_df["target"] > 0).astype(int)
        y = train_df["label"].values
        model = lgb.LGBMClassifier(**(lgbm_kwargs or DEFAULT_LGBM_KWARGS))
    else:
        y = target
        model = lgb.LGBMRegressor(**(lgbm_kwargs or DEFAULT_LGBM_KWARGS))

    x = train_df[feature_cols].copy()
    model.fit(x, _safe(y))

    return MLOrderOverlayModel(
        lgbm=model,
        cont_cols=cont_cols,
        target_std=target_std,
        use_ticker=use_ticker,
        use_classification=use_classification,
        per_ticker_interactions=per_ticker_interactions,
    )


def train_overlay_model(
    df_exec: pd.DataFrame,
    gap_input_dir: Path,
    cfg: dict,
    train_start: str,
    train_end: str,
    output_dir: Path,
    lgbm_kwargs: dict | None = None,
    use_ticker: bool = True,
    use_classification: bool = False,
    per_ticker_interactions: bool = False,
) -> MLOrderOverlayModel:
    """Train and save an ML order-decision overlay model."""
    gap_input_dir = Path(gap_input_dir)
    output_dir = Path(output_dir)

    market_vol = _precompute_market_vol(df_exec)
    y_target = compute_jp_target_returns(df_exec, JP_TICKERS)

    all_dates = df_exec.index
    train_dates = all_dates[
        (all_dates >= train_start) & (all_dates <= train_end)
    ]
    logger.info(
        "Training overlay model on %d dates (%s -> %s)",
        len(train_dates),
        train_start,
        train_end,
    )

    train_df = _collect_training_data(
        train_dates,
        df_exec,
        y_target,
        gap_input_dir,
        cfg,
        market_vol,
        per_ticker_interactions=per_ticker_interactions,
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

    save_overlay_model(model, output_dir)
    return model


# ---------------------------------------------------------------------------
# Runtime application
# ---------------------------------------------------------------------------


def apply_overlay(
    result: dict,
    df_exec: pd.DataFrame,
    overlay_model: MLOrderOverlayModel,
    trade_date: str,
) -> dict:
    """Apply the ML overlay to a V2 production result.

    Returns the original result unchanged if the overlay cannot be applied
    (missing market data, fallback triggered, etc.).
    """
    if result["fallback"]["gap_data_missing"]:
        logger.info("[%s] V2 fallback active; skipping overlay.", trade_date)
        return result

    date = pd.to_datetime(trade_date)
    if date not in df_exec.index:
        logger.warning("[%s] Trade date not in df_exec; skipping overlay.", trade_date)
        return result

    market_vol = _precompute_market_vol(df_exec)

    try:
        features = _build_ticker_features(
            df_exec,
            result,
            date,
            market_vol,
            per_ticker_interactions=overlay_model.per_ticker_interactions,
        )
    except Exception as e:
        logger.warning("[%s] Feature build failed: %s; skipping overlay.", trade_date, e)
        return result

    p_trade = _predict_p_trade(features, overlay_model)
    p_trade = _safe(p_trade)

    score_adjusted = _safe(result["scores"]) * p_trade
    w_pre = _recompute_w_pre(score_adjusted, result["Omega_gap"], result["run_config"])
    mult = result["pit_binning"]["multiplier"]
    w_final = w_pre * mult
    w_final[np.abs(w_final) < 1e-8] = 0.0

    # Re-run numerical audit on the new weights
    numerical = run_numerical_audit(w_final, score_adjusted, result["Omega_gap"])
    if numerical["status"] == "FAILED":
        logger.warning(
            "[%s] Overlay numerical audit failed; returning original V2 result.",
            trade_date,
        )
        return result

    result = dict(result)
    result["scores_overlay"] = score_adjusted
    result["w_final"] = w_final
    result["numerical"] = numerical

    summary = dict(result.get("summary", {}))
    run_cfg = result["run_config"]
    gross = float(np.sum(np.abs(w_final)))
    cost_bps = gross * run_cfg.cost_bps_per_gross
    p_mean = float(np.dot(w_final, result["mu_gap"]))
    p_var = float(np.dot(w_final, np.dot(result["Omega_gap"], w_final)))
    p_vol = float(np.sqrt(max(0.0, p_var)))
    p_ir = (p_mean - cost_bps / 10000.0) / p_vol if p_vol > 1e-6 else 0.0

    summary["overlay_applied"] = 1
    summary["p_trade_mean"] = float(np.mean(p_trade))
    summary["p_trade_std"] = float(np.std(p_trade))
    summary["target_gross"] = gross
    summary["expected_cost_bps"] = cost_bps
    summary["predicted_portfolio_mean"] = p_mean
    summary["predicted_portfolio_vol"] = p_vol
    summary["predicted_portfolio_ir"] = p_ir
    result["summary"] = summary

    logger.info(
        "[%s] ML overlay applied. p_trade mean=%.4f std=%.4f gross=%.4f",
        trade_date,
        float(np.mean(p_trade)),
        float(np.std(p_trade)),
        float(np.sum(np.abs(w_final))),
    )
    return result


def generate_v2_production_portfolio_with_overlay(
    trade_date: str,
    gap_input_dir: Path | None,
    cfg: dict,
    df_exec: pd.DataFrame | None,
    overlay_model: MLOrderOverlayModel | None,
) -> dict:
    """Run V2 production and optionally apply the ML overlay.

    This is a convenience wrapper for callers that have the overlay model
    already loaded.  It runs the base V2 pipeline and then calls
    ``apply_overlay`` when a model and df_exec are supplied.
    """
    result = generate_v2_production_portfolio(
        trade_date=trade_date,
        gap_input_dir=gap_input_dir,
        cfg=cfg,
    )

    if overlay_model is None or df_exec is None:
        return result

    return apply_overlay(result, df_exec, overlay_model, trade_date)
