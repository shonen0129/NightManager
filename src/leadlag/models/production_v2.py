"""Production v2 portfolio construction module.

Implements the daily mu_over_sigma + baseline_style + RuleD pipeline.

All runtime parameters are read from the YAML ``cfg`` dict at call time via
``parse_run_config(cfg)`` → ``ProductionV2RunConfig``.  Default values are
defined in the Pydantic schema (leadlag.config.schemas.ProductionV2RunConfig).

Public API
----------
parse_run_config(cfg)
    Convert a YAML cfg dict to a validated ``ProductionV2RunConfig``.

generate_v2_production_portfolio(trade_date, gap_input_dir, cfg)
    Core orchestrator.  Returns a result dict including ``run_config`` for the
    writer layer.

load_gap_matrices(gap_input_dir, date_str)
    Load mu_gap / Omega_gap ``.npy`` files.

load_pit_ir_history(gap_input_dir, trade_date)
    Load historical ex-ante IR series for PIT binning.

NOTE: v1 fallback mechanism was DEPRECATED on 2026-07-09 due to circular dependency.
Gap data missing now results in flat position (w_final=0).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from leadlag.compliance.v2_auditor import run_leakage_audit, run_numerical_audit
from leadlag.config.schemas import ProductionV2RunConfig
from leadlag.core.macro import (
    MACRO_NAMES,
    MACRO_SENS_MATRIX,
    compute_macro_direction_adjustment,
    compute_macro_surprise,
    compute_sigma_yy_inflation,
    download_macro_prices,
)
from leadlag.core.portfolio import get_rolling_pit_bin, solve_baseline_style
from leadlag.core.signal import build_weights_minvar
from leadlag.data.tickers import JP_TICKERS
from leadlag.models.signal_enhancement import apply_multi_horizon_blend, apply_rank_reversal_overlay

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module identifier (used by production_v2_writer.py)
# ---------------------------------------------------------------------------
VERSION = "production_residual_blpx_v2"

# ---------------------------------------------------------------------------
# Default constants (mirror ProductionV2RunConfig Pydantic defaults)
# Exported for tests and tools that need quick access without parsing cfg.
# ---------------------------------------------------------------------------
BASELINE_GROSS = 2.0
COST_BPS_PER_GROSS = 10.0
LONG_COUNT = 5
SHORT_COUNT = 5


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


def parse_run_config(cfg: dict) -> ProductionV2RunConfig:
    """Convert a raw YAML cfg dict to a validated ``ProductionV2RunConfig``.

    Reads the following top-level YAML sections:
      - ``portfolio``     → ``long_count``, ``short_count``
      - ``gross_scaling`` → ``baseline_gross``, PIT binning params, RuleD multipliers
      - ``costs``         → ``cost_bps_per_gross``
      - ``fallback``      → fallback behavior flags

    Any missing key falls back to the Pydantic field default so the function
    is safe to call with an empty dict (e.g. in tests).

    Args:
        cfg: Raw dict loaded from the production YAML file.

    Returns:
        Validated, frozen ``ProductionV2RunConfig`` instance.
    """
    portfolio = cfg.get("portfolio", {})
    gross_scaling = cfg.get("gross_scaling", {})
    costs = cfg.get("costs", {})
    fallback_cfg = cfg.get("fallback", {})
    multipliers = gross_scaling.get("multipliers", {})

    # Phase 2A: Multi-horizon blend config
    mh_cfg = cfg.get("multi_horizon_blend", {})
    # Phase 2D: CS feature overlay config
    cs_cfg = cfg.get("cs_feature_overlay", {})
    # MinVar weight optimization config
    portfolio_cfg = portfolio

    return ProductionV2RunConfig(
        long_count=portfolio.get("long_count", 5),
        short_count=portfolio.get("short_count", 5),
        baseline_gross=gross_scaling.get("baseline_gross", 2.0),
        cost_bps_per_gross=costs.get("cost_bps_per_gross", 10.0),
        pit_rolling_window=gross_scaling.get("pit_rolling_window", 252),
        tertile_low_pct=gross_scaling.get("tertile_low_pct", 33.3333),
        tertile_high_pct=gross_scaling.get("tertile_high_pct", 66.6667),
        mult_low=multipliers.get("Low", 0.75),
        mult_mid=multipliers.get("Medium", 1.00),
        mult_high=multipliers.get("High", 1.00),
        fallback_multiplier=gross_scaling.get("fallback_multiplier", 1.00),
        fallback_on_gap_data_missing=fallback_cfg.get("fallback_on_gap_data_missing", True),
        fallback_on_audit_failure=fallback_cfg.get("fallback_on_audit_failure", True),
        mh_blend_enabled=mh_cfg.get("enabled", False),
        mh_horizons=tuple(mh_cfg.get("horizons", [1, 3, 5])),
        mh_weights=tuple(mh_cfg.get("weights", [0.8, 0.1, 0.1])),
        cs_overlay_enabled=cs_cfg.get("enabled", False),
        cs_overlay_weight=cs_cfg.get("weight", 0.05),
        minvar_enabled=portfolio_cfg.get("minvar_enabled", False),
        minvar_alpha=portfolio_cfg.get("minvar_alpha", 0.5),
        macro_kappa_enabled=portfolio_cfg.get("macro_kappa_enabled", False),
        macro_kappas=tuple(portfolio_cfg.get("macro_kappas", [3.0, 0.5, 0.5])),
        macro_surprise_halflife_mean=float(portfolio_cfg.get("macro_surprise_halflife_mean", 20.0)),
        macro_surprise_halflife_vol=float(portfolio_cfg.get("macro_surprise_halflife_vol", 60.0)),
        macro_direction_enabled=portfolio_cfg.get("macro_direction_enabled", False),
    )


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def load_gap_matrices(
    gap_input_dir: Path,
    date_str: str,
) -> tuple[np.ndarray | None, np.ndarray | None, list[str]]:
    """Load mu_gap and Omega_gap matrices for the given trade date.

    Args:
        gap_input_dir: Directory that contains a ``matrices/`` subdirectory.
        date_str: Trade date in any format parseable by ``pd.to_datetime``.

    Returns:
        Tuple of (mu_gap, Omega_gap, alerts).
        Both arrays are ``None`` when files are missing.
    """
    alerts: list[str] = []
    date_numeric = pd.to_datetime(date_str).strftime("%Y%m%d")

    mu_file = gap_input_dir / "matrices" / f"mu_gap_{date_numeric}.npy"
    omega_file = gap_input_dir / "matrices" / f"omega_gap_{date_numeric}.npy"

    if not mu_file.exists():
        alerts.append(f"mu_gap file missing: {mu_file}")
        return None, None, alerts
    if not omega_file.exists():
        alerts.append(f"omega_gap file missing: {omega_file}")
        return None, None, alerts

    mu_gap = np.load(mu_file)
    Omega_gap = np.load(omega_file)
    return mu_gap, Omega_gap, alerts


def load_pit_ir_history(
    gap_input_dir: Path,
    trade_date: str,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Load historical ex-ante IR series for PIT binning.

    Reads ``portfolio_gap_distribution_diagnostics.csv`` and returns only
    rows strictly before *trade_date* to preserve point-in-time integrity.

    Args:
        gap_input_dir: Root directory of the gap distribution output.
        trade_date: Trade execution date (rows >= this date are excluded).

    Returns:
        Tuple of (history_ir array, alerts list, history_trade_dates array).
    """
    alerts: list[str] = []
    diag_file = gap_input_dir / "portfolio_gap_distribution_diagnostics.csv"

    if not diag_file.exists():
        alerts.append(
            f"Diagnostics file missing: {diag_file}. PIT binning falls back to Medium/1.0."
        )
        return np.array([]), alerts, np.array([])

    df = pd.read_csv(diag_file)
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d")
    df_hist = df[df["trade_date"] < trade_date]

    # Prefer pred_ir_gap_baseline_cost (computed with same weight construction
    # and cost formula as current_ir) over pred_ir_gap_exante_cost (legacy,
    # uses different weights and rolling realized cost).
    ir_col = "pred_ir_gap_baseline_cost"
    if ir_col not in df_hist.columns:
        ir_col = "pred_ir_gap_exante_cost"
        alerts.append(
            "pred_ir_gap_baseline_cost not found in diagnostics CSV, falling back to "
            "pred_ir_gap_exante_cost. Historical IR may be inconsistent with current_ir. "
            "Regenerate diagnostics with updated compute_gap_adjusted_distribution.py."
        )

    if ir_col not in df_hist.columns:
        alerts.append(
            "No IR column found in diagnostics. PIT binning falls back to Medium/1.0."
        )
        return np.array([]), alerts, np.array([])

    history_ir = df_hist[ir_col].values
    history_dates = pd.to_datetime(df_hist["trade_date"]).values
    return history_ir, alerts, history_dates


def _derive_signal_date(gap_input_dir: Path | None, trade_date: str) -> str:
    """Derive signal_date from gap matrix file naming or trade_date - 1 day.

    Gap matrices are computed from data up to (and including) the signal date,
    which is the US close day before trade_date.  The ``mu_gap_{YYYYMMDD}.npy``
    filename encodes the signal date.  We find the most-recent matrix file
    dated strictly before *trade_date* to use as signal_date.

    Falls back to trade_date minus 1 calendar day when the directory or files
    are not available.
    """
    trade_dt = pd.to_datetime(trade_date).normalize()
    fallback_sig_date = (trade_dt - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    if gap_input_dir is None:
        return fallback_sig_date

    matrices_dir = gap_input_dir / "matrices"
    if not matrices_dir.exists():
        return fallback_sig_date

    import re
    candidate_dates = []
    for f in matrices_dir.glob("mu_gap_*.npy"):
        stem = f.stem  # e.g. "mu_gap_20260612"
        match = re.search(r"(\d{8})", stem)
        if match:
            date_str_candidate = match.group(1)
            try:
                cdt = pd.to_datetime(date_str_candidate, format="%Y%m%d").normalize()
                if cdt < trade_dt:
                    candidate_dates.append(cdt)
            except ValueError:
                pass

    if not candidate_dates:
        return fallback_sig_date

    latest_signal_dt = max(candidate_dates)
    return latest_signal_dt.strftime("%Y-%m-%d")


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
        "version": VERSION,
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


# ---------------------------------------------------------------------------
# Core orchestrator
# ---------------------------------------------------------------------------


def generate_v2_production_portfolio(
    trade_date: str,
    gap_input_dir: Path | None,
    cfg: dict,
) -> dict:
    """Core v2 production portfolio construction.

    All runtime parameters are resolved from *cfg* via ``parse_run_config()``,
    so the function itself contains no magic numbers.

    Pipeline:
      1. Parse ``cfg`` → ``ProductionV2RunConfig`` (single source of truth).
      2. Load gap-adjusted distribution matrices (mu_gap, Omega_gap).
      3. If gap data missing → return flat position (w_final=0).
      4. Ensure Omega_gap is PSD (eigenvalue repair if needed).
      5. Compute mu_over_sigma scores.
      6. Select top-``long_count`` long / bottom-``short_count`` short by score.
      7. Compute pre-gross weights via ``solve_baseline_style(baseline_gross)``.
      8. PIT binning (RuleD) using strictly historical IR history.
      9. Apply RuleD gross multiplier (from ``run_cfg`` multiplier table).
      10. Safety audits; if numerical audit fails → return flat position.

    Args:
        trade_date: Execution date in 'YYYY-MM-DD' format.
        gap_input_dir: Directory containing gap distribution outputs, or None.
        cfg: Raw YAML config dict.  Parsed into ``ProductionV2RunConfig``
             internally; all runtime values come from here.

    Returns:
        Dict with keys:
          w_final, scores, mu_gap, sigma_gap, Omega_gap,
          fallback, pit_binning, leakage, numerical, alerts, summary,
          run_config (ProductionV2RunConfig — passed to writer layer)
    """
    # 1. Parse cfg → single source of truth for all runtime parameters
    run_cfg = parse_run_config(cfg)

    n_j = len(JP_TICKERS)
    date_str = pd.to_datetime(trade_date).strftime("%Y-%m-%d")
    alerts: list[str] = []
    fallback = {
        "gap_data_missing": False,
    }

    # 2. Load gap-adjusted distribution matrices
    mu_gap: np.ndarray | None = None
    Omega_gap: np.ndarray | None = None
    if gap_input_dir is not None:
        mu_gap, Omega_gap, gap_alerts = load_gap_matrices(gap_input_dir, date_str)
        alerts.extend(gap_alerts)
    else:
        alerts.append("--gap-input-dir not specified.")

    # 3. Fallback: gap data missing → flat position (no trading)
    if mu_gap is None or Omega_gap is None:
        fallback["gap_data_missing"] = True
        logger.error(
            "[%s] Gap data missing. Returning flat position (w_final=0). No trading today.",
            date_str,
        )
        alerts.append("Gap data missing. Flat position (w_final=0) returned.")

        dummy_scores = np.zeros(n_j)
        dummy_Omega = np.eye(n_j) * 0.01
        leakage = run_leakage_audit(
            date_str, date_str,
            gap_data_loaded=False,
            pit_history_trade_dates=None,
        )
        numerical = run_numerical_audit(np.zeros(n_j), dummy_scores, dummy_Omega)

        return {
            "w_final": np.zeros(n_j),
            "scores": dummy_scores,
            "mu_gap": np.zeros(n_j),
            "sigma_gap": np.ones(n_j) * 0.1,
            "Omega_gap": dummy_Omega,
            "fallback": fallback,
            "pit_binning": {
                "assigned_bin": "Medium",
                "threshold_low": float("nan"),
                "threshold_high": float("nan"),
                "multiplier": run_cfg.fallback_multiplier,
                "history_count": 0,
                "fallback_flag": True,
            },
            "leakage": leakage,
            "numerical": numerical,
            "alerts": alerts,
            "summary": _build_summary(
                np.zeros(n_j), date_str, run_cfg.fallback_multiplier, "Medium",
                float("nan"), float("nan"), run_cfg,
                fallback=True, candidate="flat_position",
            ),
            "run_config": run_cfg,
        }

    # 5. Ensure Omega_gap is PSD
    min_eig = np.min(np.linalg.eigvalsh(Omega_gap))
    if min_eig < 0.0:
        Omega_gap = Omega_gap + (abs(min_eig) + 1e-8) * np.eye(n_j)
        alerts.append("Omega_gap repaired to PSD.")

    # 5a. Macro adjustments (Omega_gap inflation and/or directional mu_gap adjustment)
    if run_cfg.macro_kappa_enabled or run_cfg.macro_direction_enabled:
        try:
            macro_start = (pd.to_datetime(date_str) - pd.Timedelta(days=365 * 2)).strftime("%Y-%m-%d")
            macro_end = date_str
            close_prices = download_macro_prices(start=macro_start, end=macro_end)
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
            else:
                alerts.append("Macro enabled but data insufficient; skipping.")
                logger.warning("[%s] Macro: data insufficient (%d rows).", date_str, len(close_prices) if close_prices is not None else 0)
        except Exception as e:
            alerts.append(f"Macro adjustment failed: {e}")
            logger.warning("[%s] Macro adjustment failed: %s", date_str, e)

    # 6. Compute mu_over_sigma scores
    sigma_gap = np.sqrt(np.maximum(np.diag(Omega_gap), 1e-6))
    scores = mu_gap / sigma_gap

    # 6a. Phase 2A: Multi-horizon signal blending
    if run_cfg.mh_blend_enabled and len(run_cfg.mh_horizons) > 1:
        mh_cfg = cfg.get("multi_horizon_blend", {})
        mu_pattern = mh_cfg.get(
            "mu_file_pattern_h", "matrices/mu_gap_h{h}_{date}.npy"
        )
        omega_pattern = mh_cfg.get(
            "omega_file_pattern_h", "matrices/omega_gap_h{h}_{date}.npy"
        )
        scores, mh_alerts = apply_multi_horizon_blend(
            scores_h1=scores,
            gap_input_dir=gap_input_dir,
            date_str=date_str,
            horizons=run_cfg.mh_horizons,
            weights=run_cfg.mh_weights,
            mu_pattern=mu_pattern,
            omega_pattern=omega_pattern,
        )
        alerts.extend(mh_alerts)
        if not any("not found" in a for a in mh_alerts):
            logger.info("[%s] Multi-horizon blend applied: horizons=%s, weights=%s",
                        date_str, run_cfg.mh_horizons, run_cfg.mh_weights)

    # 6b. Phase 2D: Cross-sectional rank reversal overlay
    if run_cfg.cs_overlay_enabled:
        cs_cfg = cfg.get("cs_feature_overlay", {})
        rr_pattern = cs_cfg.get(
            "rank_reversal_file_pattern", "matrices/rank_reversal_{date}.npy"
        )
        scores, cs_alerts = apply_rank_reversal_overlay(
            scores=scores,
            gap_input_dir=gap_input_dir,
            date_str=date_str,
            weight=run_cfg.cs_overlay_weight,
            file_pattern=rr_pattern,
        )
        alerts.extend(cs_alerts)
        if not any("not found" in a or "None" in a for a in cs_alerts):
            logger.info("[%s] Rank reversal overlay applied: weight=%.2f",
                        date_str, run_cfg.cs_overlay_weight)

    # 7. Select longs / shorts by score using run_cfg counts
    sorted_idx = np.argsort(scores)
    short_idx = sorted_idx[:run_cfg.short_count]
    long_idx = sorted_idx[-run_cfg.long_count:]

    # 8. Compute pre-gross weights
    if run_cfg.minvar_enabled:
        # MinVar: use Omega_gap as predicted covariance for weight optimization
        w_minvar = build_weights_minvar(
            signal=scores,
            q=float(run_cfg.long_count) / n_j,
            n_j=n_j,
            Sigma_YY=Omega_gap,
            alpha=run_cfg.minvar_alpha,
            enforce_sign=False,
        )
        # Scale to baseline_gross (build_weights_minvar normalizes each side to 1)
        w_pre = w_minvar * (run_cfg.baseline_gross / 2.0)
        logger.info(
            "[%s] MinVar weights applied: alpha=%.2f, gross=%.4f",
            date_str, run_cfg.minvar_alpha, float(np.sum(np.abs(w_pre))),
        )
    else:
        w_pre = solve_baseline_style(
            scores, long_idx, short_idx, baseline_gross=run_cfg.baseline_gross
        )

    # 9. PIT binning for RuleD — load history, compute current IR
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

    # 10. Apply RuleD multiplier
    w_final = w_pre * mult

    # 11. Safety audits
    signal_date_str = _derive_signal_date(gap_input_dir, date_str)
    leakage = run_leakage_audit(
        signal_date_str, date_str,
        gap_data_loaded=(mu_gap is not None and Omega_gap is not None),
        pit_history_trade_dates=pit_history_dates if gap_input_dir is not None else None,
    )

    numerical = run_numerical_audit(w_final, scores, Omega_gap)
    if numerical["status"] == "FAILED" and run_cfg.fallback_on_audit_failure:
        alerts.append(f"Numerical audit FAILED: {numerical}. Falling back to flat position.")
        fallback["gap_data_missing"] = True
        w_final = np.zeros(n_j)
        numerical = run_numerical_audit(w_final, scores, Omega_gap)
    elif numerical["status"] == "FAILED":
        alerts.append(
            f"Numerical audit FAILED: {numerical}. fallback_on_audit_failure=False; "
            "keeping v2 weights."
        )

    summary = _build_summary(
        w_final, date_str, mult, assigned_bin, lo_thresh, hi_thresh, run_cfg,
        fallback=fallback["gap_data_missing"], candidate="primary_ruleD",
        scores=scores, mu_gap=mu_gap, Omega_gap=Omega_gap,
    )

    return {
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
    }
