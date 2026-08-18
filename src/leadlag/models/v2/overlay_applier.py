"""V2 overlay helpers: multi-horizon blending, rank-reversal, and ML overlay."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from leadlag.data.pit_lake import MarketSnapshot
from leadlag.domain.portfolio import PortfolioDecision
from leadlag.models.signal_enhancement import apply_rank_reversal_overlay

logger = logging.getLogger(__name__)


def _apply_rank_reversal_overlay(
    scores: np.ndarray,
    gap_input_dir: Any,
    date_str: str,
    run_cfg: Any,
    alerts: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Apply cross-sectional rank-reversal overlay if enabled."""
    if not run_cfg.cs_overlay_enabled:
        return scores, alerts

    scores, cs_alerts = apply_rank_reversal_overlay(
        scores=scores,
        gap_input_dir=gap_input_dir,
        date_str=date_str,
        weight=run_cfg.cs_overlay_weight,
        file_pattern=run_cfg.cs_rank_reversal_file_pattern,
    )
    alerts.extend(cs_alerts)
    if not any("not found" in a or "None" in a for a in cs_alerts):
        logger.info(
            "[%s] Rank reversal overlay applied: weight=%.2f",
            date_str, run_cfg.cs_overlay_weight,
        )
    return scores, alerts


def _multi_horizon_scores(
    model: Any,
    trade_date: str,
    df_exec: Any,
    current_prices: dict[str, float],
    use_file_cache: bool = True,
    snapshot: MarketSnapshot | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Blend per-horizon (mu_gap, Omega_gap) into a single score series.

    If ``snapshot`` is supplied, it is passed to ``compute_distribution`` as the
    point-in-time source for gap, betas, and TOPIX night return.
    """
    from leadlag.models.v2.distribution_resolver import compute_distribution

    n_j = model.n_j
    h1_scores = None
    weighted_sum = np.zeros(n_j)
    total_weight = 0.0
    mu_h1 = None
    omega_h1 = None

    for h, w in zip(model.run_config.mh_horizons, model.run_config.mh_weights):
        mu_h, omega_h = compute_distribution(
            model,
            trade_date=trade_date,
            df_exec=df_exec,
            current_prices=current_prices,
            horizon=h,
            use_file_cache=use_file_cache,
            snapshot=snapshot,
        )
        sigma_h = np.sqrt(np.maximum(np.diag(omega_h), model.run_config.sigma_floor))
        score_h = mu_h / sigma_h

        if h == 1:
            mu_h1, omega_h1, h1_scores = mu_h, omega_h, score_h

        weighted_sum += w * score_h
        total_weight += w

    if total_weight < 1e-8:
        if h1_scores is None:
            raise RuntimeError("Multi-horizon blend produced no valid horizon.")
        assert mu_h1 is not None and omega_h1 is not None and h1_scores is not None
        return mu_h1, omega_h1, h1_scores

    blended = weighted_sum / total_weight
    blended_std = np.std(blended)
    h1_std = np.std(h1_scores) if h1_scores is not None else blended_std
    if blended_std > 1e-8 and h1_std > 1e-8:
        blended = blended * (h1_std / blended_std)

    scores = (blended - np.median(blended))
    score_std = np.std(scores)
    if score_std > 1e-8:
        scores = scores / score_std

    assert mu_h1 is not None and omega_h1 is not None
    return mu_h1, omega_h1, scores


def _apply_overlay(
    model: Any,
    result: PortfolioDecision,
    trade_date: str,
    df_exec: Any,
    overlay_enabled: bool,
    snapshot: MarketSnapshot | None = None,
) -> PortfolioDecision:
    """Apply the ML order overlay if enabled and available.

    If ``snapshot`` is supplied, it is passed to ``apply_overlay`` as the
    point-in-time source for per-ticker gap, beta, and TOPIX night return.
    """
    if not overlay_enabled:
        return result
    if model._overlay_model is None:
        return result
    if not getattr(model.run_config, "ml_overlay_enabled", False):
        return result
    if df_exec is None:
        logger.warning("[%s] Overlay requested but df_exec is None; skipping.", trade_date)
        return result

    from leadlag.models.ml_order_overlay import apply_overlay
    return PortfolioDecision.from_dict(
        apply_overlay(result, df_exec, model._overlay_model, trade_date, snapshot=snapshot)
    )
