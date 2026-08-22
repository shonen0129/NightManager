"""V2 audit and summary helpers."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from leadlag.compliance.v2_auditor import run_leakage_audit, run_numerical_audit
from leadlag.config.schemas import ProductionV2RunConfig
from leadlag.data.tickers import JP_TICKERS
from leadlag.domain.portfolio import PortfolioDecision

logger = logging.getLogger(__name__)


def _build_summary(
    w: np.ndarray,
    date_str: str,
    mult: float,
    assigned_bin: str,
    lo_thresh: float,
    hi_thresh: float,
    run_cfg: ProductionV2RunConfig,
    *,
    fallback: bool,
    candidate: str,
    version: str,
    scores: np.ndarray | None = None,
    mu_gap: np.ndarray | None = None,
    Omega_gap: np.ndarray | None = None,
) -> dict:
    """Build a one-row performance summary dict.

    Uses ``run_cfg.cost_bps_per_gross`` for IR calculation so that the
    cost assumption always comes from the YAML config.
    """
    n_j = len(JP_TICKERS)
    if scores is None:
        scores = np.zeros(n_j)
    if mu_gap is None:
        mu_gap = np.zeros(n_j)
    if Omega_gap is None:
        Omega_gap = np.eye(n_j) * 0.01

    long_idx = np.where(w > 1e-8)[0]
    short_idx = np.where(w < -1e-8)[0]
    gross = float(np.sum(np.abs(w)))
    net = float(np.sum(w))

    p_mean = float(np.dot(w, mu_gap))
    p_var = float(np.dot(w, np.dot(Omega_gap, w)))
    p_vol = float(np.sqrt(max(0.0, p_var)))
    # Cost in decimal: cost_bps_per_gross / 10000 × gross
    ex_ante_cost = gross * (run_cfg.cost_bps_per_gross / 10000.0)
    p_ir = float((p_mean - ex_ante_cost) / p_vol) if p_vol > 1e-6 else 0.0

    w_l = w[w > 0]
    hhi = float(np.sum((w_l / np.sum(w_l)) ** 2)) if len(w_l) > 0 else 0.0

    return {
        "trade_date": date_str,
        "candidate": candidate,
        "version": version,
        "long_count": int(len(long_idx)),
        "short_count": int(len(short_idx)),
        "target_gross": gross,
        "target_net": net,
        "gross_multiplier": float(mult),
        "pit_bin": assigned_bin,
        "pit_threshold_low": lo_thresh,
        "pit_threshold_high": hi_thresh,
        "predicted_portfolio_mean": p_mean,
        "predicted_portfolio_vol": p_vol,
        "predicted_portfolio_ir": p_ir,
        "expected_cost_bps": gross * run_cfg.cost_bps_per_gross,
        "herfindahl": hhi,
        "fallback_triggered": int(fallback),
    }


def _run_safety_audits(
    w_final: np.ndarray,
    scores: np.ndarray,
    mu_gap: np.ndarray,
    Omega_gap: np.ndarray,
    sigma_gap: np.ndarray,
    gap_input_dir: Any,
    date_str: str,
    signal_date: str,
    run_cfg: ProductionV2RunConfig,
    fallback: dict,
    pit_binning: dict,
    alerts: list[str],
    pit_history_trade_dates: np.ndarray | None,
    candidate: str,
    version: str,
) -> PortfolioDecision:
    """Run leakage/numerical audits and assemble the final result."""
    # Combine the two distinct fallback reasons into a single trigger flag.
    fallback_triggered = (
        fallback.get("gap_data_missing", False)
        or fallback.get("audit_failure", False)
    )

    if fallback_triggered:
        # Flat or audit-fallback means no signal was computed, so there is no leakage.
        # Return a clearly distinguished status to avoid false FAILED alerts.
        leakage = {
            "status": "FLAT",
            "signal_date_strictly_before_trade_date": True,
            "post_open_timing_respected": True,
            "realized_returns_not_used_in_signal": True,
            "pit_binning_strictly_historical": True,
            "gap_data_freshness_ok": True,
        }
    else:
        leakage = run_leakage_audit(
            signal_date,
            date_str,
            gap_data_loaded=not fallback.get("gap_data_missing", False),
            pit_history_trade_dates=pit_history_trade_dates,
        )

    numerical = run_numerical_audit(w_final, scores, Omega_gap)
    if numerical["status"] == "FAILED" and run_cfg.fallback_on_audit_failure:
        alerts.append(f"Numerical audit FAILED: {numerical}. Falling back to flat position.")
        fallback["audit_failure"] = True
        fallback_triggered = True
        w_final = np.zeros_like(w_final)
        numerical = run_numerical_audit(w_final, scores, Omega_gap)
    elif numerical["status"] == "FAILED":
        alerts.append(
            f"Numerical audit FAILED: {numerical}. fallback_on_audit_failure=False; "
            "keeping v2 weights."
        )

    summary = _build_summary(
        w_final, date_str, pit_binning["multiplier"], pit_binning["assigned_bin"],
        pit_binning["threshold_low"], pit_binning["threshold_high"], run_cfg,
        fallback=fallback_triggered, candidate=candidate,
        version=version,
        scores=scores, mu_gap=mu_gap, Omega_gap=Omega_gap,
    )

    return PortfolioDecision.from_dict({
        "w_final": w_final,
        "scores": scores,
        "mu_gap": mu_gap,
        "sigma_gap": sigma_gap,
        "Omega_gap": Omega_gap,
        "fallback": fallback,
        "pit_binning": pit_binning,
        "leakage": leakage,
        "numerical": numerical,
        "alerts": alerts,
        "summary": summary,
        "run_config": run_cfg,
    })


def _compare_distribution(
    label: str,
    mu_file: np.ndarray,
    omega_file: np.ndarray,
    mu_ondemand: np.ndarray,
    omega_ondemand: np.ndarray,
) -> None:
    """Compare file cache to on-demand and log divergence warnings."""
    max_abs_mu = float(np.max(np.abs(mu_file - mu_ondemand)))
    mu_scale = float(np.max(np.abs(mu_file))) + 1e-8
    rel_mu = max_abs_mu / mu_scale

    frob_omega = float(np.linalg.norm(omega_file - omega_ondemand, "fro"))
    omega_scale = float(np.linalg.norm(omega_file, "fro")) + 1e-8
    rel_omega = frob_omega / omega_scale

    logger.info(
        "[%s] Shadow on-demand vs file cache: "
        "max|dmu|=%.6g (rel=%.4g), frob|dOmega|=%.6g (rel=%.4g)",
        label, max_abs_mu, rel_mu, frob_omega, rel_omega,
    )
    if rel_mu > 0.01 or rel_omega > 0.01:
        logger.warning(
            "[%s] On-demand distribution differs from file cache by "
            ">1%% (rel_mu=%.4g, rel_omega=%.4g). "
            "File cache is used; investigate model/dataset drift.",
            label, rel_mu, rel_omega,
        )
