"""Production application of a fitted ML order overlay."""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from leadlag.compliance.v2_auditor import run_numerical_audit
from leadlag.config import safe_config_copy
from leadlag.config.schemas import ProductionV2RunConfig, parse_run_config
from leadlag.data.adr_features import load_adr_features, validate_adr_features
from leadlag.data.pit_lake import MarketSnapshot
from leadlag.domain.portfolio import PortfolioDecision
from leadlag.models.ml_overlay_artifact import _validate_overlay_provenance
from leadlag.models.ml_overlay_features import (
    _build_ticker_features,
    _precompute_market_vol,
    _predict_p_trade,
    _safe,
)
from leadlag.utils.timestamps import normalize_jst_date

logger = logging.getLogger("leadlag.models.ml_order_overlay")


def apply_overlay(
    result: PortfolioDecision,
    df_exec: pd.DataFrame,
    overlay_model: Any,
    trade_date: str,
    snapshot: MarketSnapshot | None = None,
    adr_features: pd.DataFrame | None = None,
    allow_implicit_io: bool = True,
) -> PortfolioDecision:
    """Apply a fitted overlay while preserving V2 fallback and audit rules."""
    fallback = result.fallback
    if fallback.get("gap_data_missing", False) or fallback.get("audit_failure", False):
        logger.info("[%s] V2 fallback active; skipping overlay.", trade_date)
        return result

    metadata = _validate_overlay_provenance(
        getattr(overlay_model, "metadata", {}) or {},
        f"Overlay application for {trade_date}",
    )
    date = normalize_jst_date(trade_date)
    if date <= pd.Timestamp(metadata["train_end"]):
        raise ValueError(
            f"Overlay was trained through {metadata['train_end']} and cannot be applied in-sample "
            f"on {trade_date}. Supply a fold-specific artifact or run an explicit "
            "ML-disabled comparison."
        )

    if date not in df_exec.index:
        logger.warning("[%s] Trade date not in df_exec; skipping overlay.", trade_date)
        return result

    market_vol = _precompute_market_vol(df_exec)
    adr_df = adr_features
    if any(column.startswith("adr_") for column in overlay_model.cont_cols):
        if adr_df is None and allow_implicit_io:
            adr_df = load_adr_features()
        elif adr_df is None:
            logger.warning("[%s] ADR features were not supplied in the decision snapshot; skipping overlay", trade_date)
            return result
        adr_df = validate_adr_features(adr_df, date)
        if adr_df is None:
            logger.warning("[%s] ADR features are unavailable for the trade date; skipping overlay", trade_date)
            return result
        if adr_df is not None:
            logger.debug("[%s] Loaded ADR features for overlay application", trade_date)

    try:
        features = _build_ticker_features(
            df_exec,
            result,
            date,
            market_vol,
            per_ticker_interactions=overlay_model.per_ticker_interactions,
            adr_df=adr_df,
            snapshot=snapshot,
        )
    except Exception as exc:
        logger.warning("[%s] Feature build failed: %s; skipping overlay.", trade_date, exc)
        return result

    p_trade = _safe(_predict_p_trade(features, overlay_model))
    multiplier = result.pit_binning["multiplier"]
    if multiplier < 1e-12:
        logger.warning(
            "[%s] RuleD multiplier is %.6g; overlay needs a non-zero scale. "
            "Returning original V2 result.",
            trade_date,
            float(multiplier),
        )
        return result

    w_pre = result.w_final / multiplier
    w_scaled = _safe(w_pre) * p_trade
    w_scaled[np.abs(w_scaled) < 1e-8] = 0.0
    baseline_gross = float(result.run_config.baseline_gross)
    long_mask = w_scaled > 0
    short_mask = w_scaled < 0
    long_sum = float(w_scaled[long_mask].sum())
    short_sum = float(w_scaled[short_mask].sum())
    if long_sum > 0 and short_sum < 0:
        w_scaled[long_mask] *= (baseline_gross / 2.0) / long_sum
        w_scaled[short_mask] *= (-baseline_gross / 2.0) / short_sum
    else:
        logger.warning("[%s] Overlay collapsed one side; returning original V2 result.", trade_date)
        return result

    w_final = w_scaled * multiplier
    w_final[np.abs(w_final) < 1e-8] = 0.0
    score_adjusted = _safe(result.scores) * p_trade
    numerical = run_numerical_audit(w_final, score_adjusted, result.Omega_gap)
    if numerical["status"] == "FAILED":
        logger.warning("[%s] Overlay numerical audit failed; returning original V2 result.", trade_date)
        return result

    summary = dict(result.summary)
    run_cfg = result.run_config
    gross = float(np.sum(np.abs(w_final)))
    cost_bps = gross * run_cfg.cost_bps_per_gross
    p_mean = float(np.dot(w_final, result.mu_gap))
    p_var = float(np.dot(w_final, np.dot(result.Omega_gap, w_final)))
    p_vol = float(np.sqrt(max(0.0, p_var)))
    p_ir = (p_mean - cost_bps / 10000.0) / p_vol if p_vol > 1e-6 else 0.0
    summary.update(
        {
            "overlay_applied": 1,
            "p_trade_mean": float(np.mean(p_trade)),
            "p_trade_std": float(np.std(p_trade)),
            "target_gross": gross,
            "expected_cost_bps": cost_bps,
            "predicted_portfolio_mean": p_mean,
            "predicted_portfolio_vol": p_vol,
            "predicted_portfolio_ir": p_ir,
        }
    )
    logger.info(
        "[%s] ML overlay applied. p_trade mean=%.4f std=%.4f gross=%.4f",
        trade_date,
        float(np.mean(p_trade)),
        float(np.std(p_trade)),
        gross,
    )
    return replace(
        result,
        w_final=w_final,
        scores_overlay=score_adjusted,
        numerical=numerical,
        summary=summary,
    )


def generate_v2_production_portfolio_with_overlay(
    trade_date: str,
    gap_input_dir: Path | None,
    cfg: ProductionV2RunConfig | dict,
    df_exec: pd.DataFrame | None,
    overlay_model: Any | None,
) -> PortfolioDecision:
    """Run V2 and optionally apply an already loaded overlay model."""
    cfg_copy = safe_config_copy(cfg)
    run_cfg = cfg_copy if isinstance(cfg_copy, ProductionV2RunConfig) else parse_run_config(cfg_copy)
    # Lazy import avoids a production_v2 -> overlay -> production_v2 cycle.
    from leadlag.models.production_v2 import ProductionV2Model

    result = ProductionV2Model(parse_run_config(run_cfg)).decide(trade_date=trade_date, gap_input_dir=gap_input_dir, overlay_enabled=False, use_file_cache=True)
    if overlay_model is None or df_exec is None:
        return result
    return apply_overlay(result, df_exec, overlay_model, trade_date)


__all__ = ["apply_overlay", "generate_v2_production_portfolio_with_overlay"]
