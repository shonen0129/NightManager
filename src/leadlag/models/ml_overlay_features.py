"""Feature construction and deterministic inference helpers for the ML overlay.

This module is deliberately independent from artifact persistence and from the
training entry point.  It contains the feature schema used by production
inference; the training package imports the same helpers so that column order
and missing-value handling cannot drift between the two paths.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.special import expit

from leadlag.core.portfolio import solve_baseline_style
from leadlag.core.signal import build_weights_minvar
from leadlag.data.pit_lake import MarketSnapshot
from leadlag.data.tickers import JP_TICKERS
from leadlag.domain.portfolio import PortfolioDecision

# Keep the historical logger name so operational filters remain stable after
# the implementation moves out of ``ml_order_overlay``.
logger = logging.getLogger("leadlag.models.ml_order_overlay")


def overlay_continuous_columns(per_ticker_interactions: bool = False) -> list[str]:
    """Return the canonical continuous feature order used at fit and inference."""
    columns = [
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
        "adr_return",
        "adr_x_score",
        "adr_x_gap",
        "adr_x_gap_idio",
        "adr_x_mu_gap",
        "abs_adr",
    ]
    if not per_ticker_interactions:
        return columns
    return [
        column
        for column in columns
        if column not in ("score", "mu_gap", "gap", "score_x_gap", "score_x_gap_idio")
    ] + [
        f"ticker_{ticker}_{suffix}"
        for ticker in JP_TICKERS
        for suffix in ("score", "gap", "score_x_gap")
    ]


def _safe(arr: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf with finite values."""
    result = np.asarray(arr, dtype=float).copy()
    result[np.isnan(result) | np.isinf(result)] = 0.0
    return result


def _sigmoid(x: np.ndarray, scale: float) -> np.ndarray:
    """Numerically stable sigmoid centered at zero with the fitted scale."""
    return cast(np.ndarray, expit(x / max(scale, 1e-8)))


def _precompute_market_vol(df_exec: pd.DataFrame) -> pd.DataFrame:
    """Compute the strictly historical 20-day mean absolute close return."""
    oc_cols = [f"jp_oc_{tk}" for tk in JP_TICKERS]
    oc_df = df_exec[oc_cols].copy()
    oc_df.columns = JP_TICKERS
    return oc_df.abs().rolling(window=20, min_periods=5).mean().shift(1)


def _build_ticker_features(
    df_exec: pd.DataFrame,
    v2_result: PortfolioDecision,
    trade_date: pd.Timestamp,
    market_vol: pd.DataFrame | None,
    per_ticker_interactions: bool = False,
    adr_df: pd.DataFrame | None = None,
    snapshot: MarketSnapshot | None = None,
) -> pd.DataFrame:
    """Build one row per TOPIX ticker using the point-in-time snapshot."""
    scores = v2_result.scores
    mu_gap = v2_result.mu_gap
    sigma_gap = v2_result.sigma_gap

    if snapshot is not None:
        topix_night = float(snapshot.topix_night_return)
    else:
        if trade_date not in df_exec.index:
            raise KeyError(f"Trade date {trade_date} not in df_exec")
        row = df_exec.loc[trade_date]
        topix_night = float(row["topix_night_return"])

    records: list[dict[str, Any]] = []
    for i, ticker in enumerate(JP_TICKERS):
        if snapshot is not None:
            gap = float(snapshot.jp_gap_returns[i])
            beta = float(snapshot.jp_betas[i])
        else:
            gap = float(row[f"jp_gap_{ticker}"])
            beta = float(row[f"jp_beta_{ticker}"])
        gap_idio = gap - beta * topix_night
        market_vol_value = (
            float(market_vol.loc[trade_date, ticker]) if market_vol is not None else 0.0
        )
        score = float(scores[i])
        score_x_gap = score * gap
        score_x_gap_idio = score * gap_idio
        mu_gap_value = float(mu_gap[i])

        adr_return = 0.0
        if adr_df is not None:
            try:
                adr_return = float(adr_df.loc[trade_date, f"adr_{ticker}"])
            except Exception:
                logger.warning(
                    "[%s] ADR feature missing for %s; using 0.0 fallback.",
                    trade_date,
                    ticker,
                )
        adr_return = float(_safe(np.array([adr_return]))[0])

        record: dict[str, Any] = {
            "ticker": ticker,
            "score": score,
            "mu_gap": mu_gap_value,
            "sigma_gap": float(sigma_gap[i]),
            "gap": gap,
            "gap_idio": gap_idio,
            "topix_night": topix_night,
            "market_vol_20d": market_vol_value,
            "score_x_gap": score_x_gap,
            "score_x_gap_idio": score_x_gap_idio,
            "abs_score": abs(score),
            "abs_gap": abs(gap),
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
        records.append(record)
    return pd.DataFrame(records)


def _predict_p_trade(features: pd.DataFrame, model: Any) -> np.ndarray:
    """Predict the per-ticker trade multiplier from a fitted overlay."""
    feature_cols = list(model.cont_cols) + (["ticker"] if model.use_ticker else [])
    x = features[feature_cols].copy()
    if model.use_ticker:
        x["ticker"] = pd.Categorical(x["ticker"], categories=JP_TICKERS)
    for column in model.cont_cols:
        if column not in x.columns:
            x[column] = 0.0
    if model.use_classification:
        prediction = model.lgbm.predict_proba(x)[:, 1]
    else:
        prediction = model.lgbm.predict(x)
    scale = getattr(model, "p_trade_scale", 1.0)
    return _sigmoid(prediction, model.target_std) * scale


def _recompute_w_pre(scores: np.ndarray, omega_gap: np.ndarray, run_cfg: Any) -> np.ndarray:
    """Recompute pre-RuleD weights from adjusted scores using V2 rules."""
    n_j = len(JP_TICKERS)
    sorted_idx = np.argsort(scores)
    short_idx = sorted_idx[: run_cfg.short_count]
    long_idx = sorted_idx[-run_cfg.long_count :]
    if run_cfg.minvar_enabled:
        w_minvar = build_weights_minvar(
            signal=scores,
            q=float(run_cfg.long_count) / n_j,
            n_j=n_j,
            Sigma_YY=omega_gap,
            alpha=run_cfg.minvar_alpha,
            enforce_sign=False,
        )
        return cast(np.ndarray, w_minvar * (run_cfg.baseline_gross / 2.0))
    return solve_baseline_style(
        scores,
        long_idx,
        short_idx,
        baseline_gross=run_cfg.baseline_gross,
    )


__all__ = [
    "_build_ticker_features",
    "_predict_p_trade",
    "_precompute_market_vol",
    "_recompute_w_pre",
    "_safe",
    "_sigmoid",
    "overlay_continuous_columns",
]
