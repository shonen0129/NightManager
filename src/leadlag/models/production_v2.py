"""Production v2 portfolio construction module.

Implements the daily mu_over_sigma + baseline_style + RuleD pipeline.

All runtime parameters are read from the YAML ``cfg`` dict at call time via
``parse_run_config(cfg)`` → ``ProductionV2RunConfig``.  Module-level constants
(``LONG_COUNT``, ``BASELINE_GROSS``, etc.) are kept only as documentation of
the Pydantic-schema defaults — they are **not** used inside any function.

Public API
----------
parse_run_config(cfg)
    Convert a YAML cfg dict to a validated ``ProductionV2RunConfig``.

generate_v2_production_portfolio(trade_date, gap_input_dir, v1_weights_file, cfg)
    Core orchestrator.  Returns a result dict including ``run_config`` for the
    writer layer.

load_gap_matrices(gap_input_dir, date_str)
    Load mu_gap / Omega_gap ``.npy`` files.

load_v1_fallback_weights(v1_weights_file, date_str, n_j)
    Load v1 Residual-BLPX baseline weights.

load_pit_ir_history(gap_input_dir, trade_date)
    Load historical ex-ante IR series for PIT binning.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from leadlag.config.schemas import ProductionV2RunConfig
from leadlag.core.portfolio import get_rolling_pit_bin, solve_baseline_style
from leadlag.compliance.v2_auditor import run_leakage_audit, run_numerical_audit
from leadlag.data.tickers import JP_TICKERS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants — documentation only.
# These are the Pydantic field *defaults*; actual runtime values always come
# from ProductionV2RunConfig parsed from the YAML cfg dict.
# ---------------------------------------------------------------------------
VERSION = "production_p8p3_v2"
LONG_COUNT = 5          # ProductionV2RunConfig.long_count default
SHORT_COUNT = 5         # ProductionV2RunConfig.short_count default
BASELINE_GROSS = 2.0    # ProductionV2RunConfig.baseline_gross default
COST_BPS_PER_GROSS = 10.0  # ProductionV2RunConfig.cost_bps_per_gross default


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


def load_v1_fallback_weights(
    v1_weights_file: Path,
    date_str: str,
    n_j: int,
) -> tuple[np.ndarray, list[str]]:
    """Load v1 Residual-BLPX baseline weights as fallback.

    Args:
        v1_weights_file: Path to ``v1_baseline_weights.csv``.
        date_str: Expected trade date in the file (for alignment check).
        n_j: Number of JP tickers (determines output array length).

    Returns:
        Tuple of (w_v1 array, alerts list).
    """
    alerts: list[str] = []
    w_v1 = np.zeros(n_j)

    if not v1_weights_file.exists():
        alerts.append(f"v1 weights file not found: {v1_weights_file}. Using zero weights.")
        return w_v1, alerts

    df = pd.read_csv(v1_weights_file)
    if len(df) == 0:
        alerts.append("v1 weights file is empty. Using zero weights.")
        return w_v1, alerts

    # Verify date alignment
    file_date = str(df.iloc[0].get("trade_date", ""))
    if file_date != date_str:
        alerts.append(
            f"v1 weights date mismatch: file has {file_date}, expected {date_str}. "
            "Using stale v1 weights (caution)."
        )

    for _, row in df.iterrows():
        tk = str(row.get("ticker", ""))
        if tk in JP_TICKERS:
            idx = JP_TICKERS.index(tk)
            w_v1[idx] = float(row.get("weight", 0.0))

    if np.sum(np.abs(w_v1)) < 1e-8:
        alerts.append("v1 weights loaded but all zero. Using zero weights.")

    return w_v1, alerts


def load_pit_ir_history(
    gap_input_dir: Path,
    trade_date: str,
) -> tuple[np.ndarray, list[str]]:
    """Load historical ex-ante IR series for PIT binning.

    Reads ``portfolio_gap_distribution_diagnostics.csv`` and returns only
    rows strictly before *trade_date* to preserve point-in-time integrity.

    Args:
        gap_input_dir: Root directory of the gap distribution output.
        trade_date: Trade execution date (rows >= this date are excluded).

    Returns:
        Tuple of (history_ir array, alerts list).
    """
    alerts: list[str] = []
    diag_file = gap_input_dir / "portfolio_gap_distribution_diagnostics.csv"

    if not diag_file.exists():
        alerts.append(
            f"Diagnostics file missing: {diag_file}. PIT binning falls back to Medium/1.0."
        )
        return np.array([]), alerts

    df = pd.read_csv(diag_file)
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d")
    df_hist = df[df["trade_date"] < trade_date]

    if "pred_ir_gap_exante_cost" not in df_hist.columns:
        alerts.append(
            "pred_ir_gap_exante_cost column missing. PIT binning falls back to Medium/1.0."
        )
        return np.array([]), alerts

    history_ir = df_hist["pred_ir_gap_exante_cost"].values
    return history_ir, alerts


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

    candidate_dates = []
    for f in matrices_dir.glob("mu_gap_*.npy"):
        stem = f.stem  # e.g. "mu_gap_20260612"
        parts = stem.split("_")
        if len(parts) >= 3:
            date_str_candidate = parts[2]
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
    v1_weights_file: Path,
    cfg: dict,
) -> dict:
    """Core v2 production portfolio construction.

    All runtime parameters are resolved from *cfg* via ``parse_run_config()``,
    so the function itself contains no magic numbers.

    Pipeline:
      1. Parse ``cfg`` → ``ProductionV2RunConfig`` (single source of truth).
      2. Load v1 baseline weights (fallback source + PIT IR base).
      3. Load gap-adjusted distribution matrices (mu_gap, Omega_gap).
      4. If gap data missing and ``run_cfg.fallback_on_gap_data_missing`` → return v1.
      5. Ensure Omega_gap is PSD (eigenvalue repair if needed).
      6. Compute mu_over_sigma scores.
      7. Select top-``long_count`` long / bottom-``short_count`` short by score.
      8. Compute pre-gross weights via ``solve_baseline_style(baseline_gross)``.
      9. PIT binning (RuleD) using strictly historical IR history.
      10. Apply RuleD gross multiplier (from ``run_cfg`` multiplier table).
      11. Safety audits; if numerical audit fails and ``fallback_on_audit_failure`` → v1.

    Args:
        trade_date: Execution date in 'YYYY-MM-DD' format.
        gap_input_dir: Directory containing gap distribution outputs, or None.
        v1_weights_file: Path to ``v1_baseline_weights.csv``.
        cfg: Raw YAML config dict.  Parsed into ``ProductionV2RunConfig``
             internally; all runtime values come from here.

    Returns:
        Dict with keys:
          w_final, w_v1, scores, mu_gap, sigma_gap, Omega_gap,
          fallback, pit_binning, leakage, numerical, alerts, summary,
          run_config (ProductionV2RunConfig — passed to writer layer)
    """
    # 1. Parse cfg → single source of truth for all runtime parameters
    run_cfg = parse_run_config(cfg)

    n_j = len(JP_TICKERS)
    date_str = str(trade_date)
    alerts: list[str] = []
    fallback = {
        "gap_data_missing": False,
        "v1_fallback_used": False,
        "sre_fallback_used": False,
    }

    # 2. Load v1 baseline weights (always needed for PIT IR + fallback)
    w_v1, v1_alerts = load_v1_fallback_weights(v1_weights_file, date_str, n_j)
    alerts.extend(v1_alerts)

    # 3. Load gap-adjusted distribution matrices
    mu_gap: np.ndarray | None = None
    Omega_gap: np.ndarray | None = None
    if gap_input_dir is not None:
        mu_gap, Omega_gap, gap_alerts = load_gap_matrices(gap_input_dir, date_str)
        alerts.extend(gap_alerts)
    else:
        alerts.append("--gap-input-dir not specified. Using v1 fallback.")

    # 4. Fallback: gap data missing → use v1 weights (if configured)
    if mu_gap is None or Omega_gap is None:
        fallback["gap_data_missing"] = True
        if run_cfg.fallback_on_gap_data_missing:
            fallback["v1_fallback_used"] = True
            logger.warning("[%s] Gap data missing — activating v1 fallback.", date_str)
        else:
            logger.error(
                "[%s] Gap data missing and fallback_on_gap_data_missing=False. "
                "Returning zero weights.",
                date_str,
            )

        dummy_scores = np.zeros(n_j)
        dummy_Omega = np.eye(n_j) * 0.01
        leakage = run_leakage_audit(date_str, date_str)
        numerical = run_numerical_audit(w_v1, dummy_scores, dummy_Omega)

        return {
            "w_final": w_v1.copy(),
            "w_v1": w_v1.copy(),
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
                w_v1, date_str, run_cfg.fallback_multiplier, "Medium",
                float("nan"), float("nan"), run_cfg,
                fallback=True, candidate="v1_fallback",
            ),
            "run_config": run_cfg,
        }

    # 5. Ensure Omega_gap is PSD
    min_eig = np.min(np.linalg.eigvalsh(Omega_gap))
    if min_eig < 0.0:
        Omega_gap = Omega_gap + (abs(min_eig) + 1e-8) * np.eye(n_j)
        alerts.append("Omega_gap repaired to PSD.")

    # 6. Compute mu_over_sigma scores
    sigma_gap = np.sqrt(np.maximum(np.diag(Omega_gap), 1e-6))
    scores = mu_gap / sigma_gap

    # 7. Select longs / shorts by score using run_cfg counts
    sorted_idx = np.argsort(scores)
    short_idx = sorted_idx[:run_cfg.short_count]
    long_idx = sorted_idx[-run_cfg.long_count:]

    # 8. Compute pre-gross weights (baseline_style) using run_cfg gross
    w_pre = solve_baseline_style(
        scores, long_idx, short_idx, baseline_gross=run_cfg.baseline_gross
    )

    # 9. PIT binning for RuleD — load history, compute current IR
    history_ir = np.array([])
    if gap_input_dir is not None:
        history_ir, pit_alerts = load_pit_ir_history(gap_input_dir, date_str)
        alerts.extend(pit_alerts)

    if np.sum(np.abs(w_v1)) > 1e-8:
        p_mean_v1 = np.dot(w_v1, mu_gap)
        p_var_v1 = np.dot(w_v1, np.dot(Omega_gap, w_v1))
        p_vol_v1 = np.sqrt(max(0.0, p_var_v1))
        # Ex-ante cost in decimal units
        ex_ante_cost = run_cfg.baseline_gross * (run_cfg.cost_bps_per_gross / 10000.0)
        current_ir = (p_mean_v1 - ex_ante_cost) / p_vol_v1 if p_vol_v1 > 1e-6 else 0.0
    else:
        current_ir = 0.0

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
    leakage = run_leakage_audit(signal_date_str, date_str)

    numerical = run_numerical_audit(w_final, scores, Omega_gap)
    if numerical["status"] == "FAILED" and run_cfg.fallback_on_audit_failure:
        alerts.append(f"Numerical audit FAILED: {numerical}. Falling back to v1.")
        fallback["v1_fallback_used"] = True
        w_final = w_v1.copy()
        numerical = run_numerical_audit(w_final, scores, Omega_gap)
    elif numerical["status"] == "FAILED":
        alerts.append(
            f"Numerical audit FAILED: {numerical}. fallback_on_audit_failure=False; "
            "keeping v2 weights."
        )

    summary = _build_summary(
        w_final, date_str, mult, assigned_bin, lo_thresh, hi_thresh, run_cfg,
        fallback=fallback["v1_fallback_used"], candidate="primary_ruleD",
        scores=scores, mu_gap=mu_gap, Omega_gap=Omega_gap,
    )

    return {
        "w_final": w_final,
        "w_v1": w_v1,
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
