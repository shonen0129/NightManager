"""V2 fallback helpers: matrix repair, macro adjustment, and PIT/RuleD."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from leadlag.config.schemas import ProductionV2RunConfig
from leadlag.core.macro import (
    MACRO_NAMES,
    MACRO_SENS_MATRIX,
    compute_macro_direction_adjustment,
    compute_macro_surprise,
    compute_sigma_yy_inflation,
)
from leadlag.core.portfolio import get_rolling_pit_bin
from leadlag.models.v2.pit import load_pit_ir_history

logger = logging.getLogger(__name__)


def _repair_and_adjust(
    mu_gap: np.ndarray,
    Omega_gap: np.ndarray,
    run_cfg: ProductionV2RunConfig,
    date_str: str,
    n_j: int,
    alerts: list[str],
    cache: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Ensure Omega_gap is symmetric and PSD; optionally apply macro adjustments."""
    # Symmetrize before any eigenvalue or quadratic-form operation.
    sym_err = float(np.max(np.abs(Omega_gap - Omega_gap.T)))
    if sym_err > 1e-8:
        Omega_gap = 0.5 * (Omega_gap + Omega_gap.T)
        alerts.append(
            f"Omega_gap was non-symmetric (err={sym_err:.3e}); symmetrized before PSD repair."
        )

    # Ensure Omega_gap is PSD
    min_eig = np.min(np.linalg.eigvalsh(Omega_gap))
    if min_eig < 0.0:
        Omega_gap = Omega_gap + (abs(min_eig) + 1e-8) * np.eye(n_j)
        alerts.append("Omega_gap repaired to PSD.")

    # Macro adjustments (Omega_gap inflation and/or directional mu_gap adjustment)
    if run_cfg.macro_kappa_enabled or run_cfg.macro_direction_enabled:
        try:
            # Download the full available history once and reuse it across backtest
            # dates (cache key is stable regardless of the trade date). The PIT cut
            # and a rolling 2-year window are applied below so no look-ahead occurs.
            import leadlag.models.production_v2 as _pv2
            close_prices = _pv2.download_macro_prices(period="max", cache=cache)
            if close_prices is not None:
                # PIT cut: the close for the trade date itself is not known at 9:10.
                close_prices = close_prices[close_prices.index < pd.to_datetime(date_str)]
                # Match the original 2-year EWMA horizon and reduce compute.
                macro_start_dt = pd.to_datetime(date_str) - pd.Timedelta(days=365 * 2)
                close_prices = close_prices[close_prices.index >= macro_start_dt]
            if close_prices is not None and len(close_prices) >= 30:
                macro_returns = close_prices.pct_change()
                macro_returns = macro_returns.replace([np.inf, -np.inf], np.nan)
                macro_returns = macro_returns.fillna(0.0)
                macro_returns = macro_returns[MACRO_NAMES]

                surprise = compute_macro_surprise(
                    macro_returns,
                    halflife_mean=run_cfg.macro_surprise_halflife_mean,
                    halflife_vol=run_cfg.macro_surprise_halflife_vol,
                )
                surprise_t = surprise[-1:]  # (1, n_macro) — use only the latest day
                kappas_arr = np.array(run_cfg.macro_kappas, dtype=float)

                # Apply kappa and direction atomically: if either step fails,
                # roll back to the pre-macro state to avoid a half-applied
                # distribution (e.g. inflated Omega without adjusted mu).
                original_omega = Omega_gap.copy()
                original_mu = mu_gap.copy()
                try:
                    # Factor-Kappa: inflate Omega_gap (|surprise| × |sensitivity|)
                    if run_cfg.macro_kappa_enabled:
                        scales_t = compute_sigma_yy_inflation(
                            surprise_t, kappas_arr, MACRO_SENS_MATRIX,
                        )  # (1, n_j)
                        d = np.sqrt(scales_t[0])  # (n_j,)
                        Omega_gap = Omega_gap * np.outer(d, d)
                        alerts.append(
                            f"Macro kappa Omega_gap inflation applied: "
                            f"scales_mean={float(np.mean(scales_t[0])):.3f}, "
                            f"scales_max={float(np.max(scales_t[0])):.3f}"
                        )
                        logger.info(
                            "[%s] Macro kappa: Omega_gap inflated. "
                            "surprise=%s, scales_mean=%.3f, scales_max=%.3f",
                            date_str,
                            np.round(surprise_t[0], 3),
                            float(np.mean(scales_t[0])),
                            float(np.max(scales_t[0])),
                        )

                    # Directional adjustment: signed surprise × signed sensitivity on mu_gap
                    if run_cfg.macro_direction_enabled:
                        dir_adj_t = compute_macro_direction_adjustment(
                            surprise_t, kappas_arr, MACRO_SENS_MATRIX,
                        )  # (1, n_j)
                        mu_gap = mu_gap * dir_adj_t[0]
                        alerts.append(
                            f"Macro direction adjustment applied: "
                            f"adj_mean={float(np.mean(dir_adj_t[0])):.3f}, "
                            f"adj_std={float(np.std(dir_adj_t[0])):.3f}"
                        )
                        logger.info(
                            "[%s] Macro direction: mu_gap adjusted. "
                            "adj_mean=%.3f, adj_std=%.3f",
                            date_str,
                            float(np.mean(dir_adj_t[0])),
                            float(np.std(dir_adj_t[0])),
                        )
                except Exception as macro_e:
                    Omega_gap = original_omega
                    mu_gap = original_mu
                    alerts.append(f"Macro adjustment failed: {macro_e}")
                    logger.warning("[%s] Macro adjustment failed: %s", date_str, macro_e)
            else:
                alerts.append("Macro enabled but data insufficient; skipping.")
                n_rows = len(close_prices) if close_prices is not None else 0
                logger.warning("[%s] Macro: data insufficient (%d rows).", date_str, n_rows)
        except Exception as e:
            alerts.append(f"Macro adjustment failed: {e}")
            logger.warning("[%s] Macro adjustment failed: %s", date_str, e)

    return mu_gap, Omega_gap, alerts


def _apply_pit_ruleD(
    w_pre: np.ndarray,
    mu_gap: np.ndarray,
    Omega_gap: np.ndarray,
    gap_input_dir: Path | None,
    date_str: str,
    run_cfg: ProductionV2RunConfig,
    alerts: list[str],
) -> tuple[np.ndarray, dict, list[str], np.ndarray]:
    """PIT binning and RuleD gross multiplier."""
    # PIT binning for RuleD — load history, compute current IR
    history_ir = np.array([])
    pit_history_dates = np.array([])
    if gap_input_dir is not None:
        history_ir, pit_alerts, pit_history_dates = load_pit_ir_history(gap_input_dir, date_str)
        alerts.extend(pit_alerts)

    # For PIT binning, use the baseline style weights as reference
    p_mean_baseline = np.dot(w_pre, mu_gap)
    p_var_baseline = np.dot(w_pre, np.dot(Omega_gap, w_pre))
    p_vol_baseline = np.sqrt(max(0.0, p_var_baseline))
    # Ex-ante cost in decimal units
    ex_ante_cost = run_cfg.baseline_gross * (run_cfg.cost_bps_per_gross / 10000.0)
    current_ir = (p_mean_baseline - ex_ante_cost) / p_vol_baseline if p_vol_baseline > 1e-6 else 0.0

    # Use PIT parameters from run_cfg (not hardcoded)
    assigned_bin, lo_thresh, hi_thresh, mult = get_rolling_pit_bin(
        history_ir,
        current_ir,
        rolling_window=run_cfg.pit_rolling_window,
        low_pct=run_cfg.tertile_low_pct,
        high_pct=run_cfg.tertile_high_pct,
        mult_low=run_cfg.mult_low,
        mult_mid=run_cfg.mult_mid,
        mult_high=run_cfg.mult_high,
    )
    history_count = int(np.sum(np.isfinite(history_ir)))
    pit_fallback = history_count < run_cfg.pit_rolling_window

    if pit_fallback:
        alerts.append(
            f"PIT history insufficient ({history_count} < {run_cfg.pit_rolling_window}). "
            f"Using {assigned_bin}/{mult:.2f} multiplier."
        )

    pit_binning = {
        "assigned_bin": assigned_bin,
        "threshold_low": lo_thresh,
        "threshold_high": hi_thresh,
        "multiplier": float(mult),
        "current_ir": float(current_ir),
        "history_count": history_count,
        "fallback_flag": pit_fallback,
    }
    logger.info(
        "[%s] PIT bin=%s (IR=%.4f), mult=%.2f, history=%dd",
        date_str, assigned_bin, current_ir, mult, history_count,
    )

    # Apply RuleD multiplier
    w_final = w_pre * mult

    return w_final, pit_binning, alerts, pit_history_dates
