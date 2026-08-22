"""V2 gap-distribution I/O and on-demand helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from leadlag.config.schemas import ProductionV2RunConfig
from leadlag.core.gap_adjustment import build_raw_distribution, compute_gap_adjusted_distribution
from leadlag.data.pit_lake import MarketSnapshot
from leadlag.data.tickers import JP_TICKERS
from leadlag.data.validation import DataValidationError
from leadlag.models.v2.audit_comparator import _compare_distribution, _run_safety_audits
from leadlag.utils.gap_matrix_io import load_gap_matrices

logger = logging.getLogger(__name__)


def _build_current_prices_from_df_exec(
    df_exec: pd.DataFrame,
    trade_date: str,
) -> dict[str, float]:
    """Extract 9:10 opens for JP tickers from ``jp_open_trade_*`` columns.
    Returns a ``ticker -> open price`` dict.  Missing or non-positive prices
    are omitted so callers can decide whether to use them.
    """
    if trade_date not in df_exec.index:
        return {}
    row = df_exec.loc[trade_date]
    prices: dict[str, float] = {}
    for tk in JP_TICKERS:
        col = f"jp_open_trade_{tk}"
        if col in row and pd.notna(row[col]) and float(row[col]) > 0:
            prices[tk] = float(row[col])
    return prices


def _extract_gap_inputs(
    df_exec: pd.DataFrame,
    trade_date: str,
    current_prices: dict[str, float],
    snapshot: MarketSnapshot | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Extract the opening gap override, per-ticker betas, and TOPIX night return.

    If ``snapshot`` is supplied, it is the single point-in-time source of truth
    and its ``jp_gap_returns``, ``jp_betas``, and ``topix_night_return`` are
    used directly.  Otherwise the values are read from ``df_exec.loc[trade_date]``
    for backward compatibility.

    ``gap_override[j] = current_prices[ticker] / previous_close - 1``.
    Missing prices or previous closes are replaced with 0.0.
    """
    n_j = len(JP_TICKERS)
    if snapshot is not None:
        gap_override = np.array(snapshot.jp_gap_returns, dtype=float).copy()
        betas_t = np.array(snapshot.jp_betas, dtype=float).copy()
        topix_night_t = float(snapshot.topix_night_return)
        if not np.isfinite(topix_night_t):
            topix_night_t = 0.0
        return gap_override, betas_t, topix_night_t

    if trade_date not in df_exec.index:
        return np.zeros(n_j), np.zeros(n_j), 0.0
    row = df_exec.loc[trade_date]
    gap_override = np.zeros(n_j)
    betas_t = np.zeros(n_j)
    for j, tk in enumerate(JP_TICKERS):
        prev_close = row.get(f"jp_close_sig_{tk}")
        price = current_prices.get(tk)
        if (
            price is not None
            and np.isfinite(price)
            and float(price) > 0
            and pd.notna(prev_close)
            and float(prev_close) > 0
        ):
            gap_override[j] = float(price) / float(prev_close) - 1.0
        else:
            gap_override[j] = 0.0
        beta_val = row.get(f"jp_beta_{tk}")
        betas_t[j] = float(beta_val) if pd.notna(beta_val) else 0.0
    topix_night_t = float(row.get("topix_night_return", 0.0))
    if not np.isfinite(topix_night_t):
        topix_night_t = 0.0
    return gap_override, betas_t, topix_night_t


def _gap_alerts_fatal(gap_alerts: list[str]) -> bool:
    """Return True if loaded gap matrices must not be used.

    Alerts are now produced with explicit ``[FATAL]`` / ``[REPAIRABLE]``
    severity prefixes by ``validate_gap_matrices``.  ``[REPAIRABLE]``
    issues (symmetry / PSD) are fixed downstream, so only ``[FATAL]``
    alerts, missing-matrix placeholders, or any explicit DataValidationError
    are considered fatal here.
    """
    for alert in gap_alerts:
        if alert.startswith("[FATAL]"):
            return True
    return False


def _load_gap_or_flat(
    gap_input_dir: Path | None,
    run_cfg: ProductionV2RunConfig,
    n_j: int,
    date_str: str,
) -> dict:
    """Load gap matrices or return a flat-position result.
    Returns a dict with keys:
      - is_flat (bool): whether the flat fallback was triggered.
      - result (PortfolioDecision | None): final decision when is_flat is True.
      - mu_gap / Omega_gap: loaded matrices when is_flat is False.
      - alerts (list[str]): alerts from this stage.
    """
    alerts: list[str] = []
    gap_alerts: list[str] = []
    fallback = {"gap_data_missing": False}
    mu_gap: np.ndarray | None = None
    Omega_gap: np.ndarray | None = None
    if gap_input_dir is not None:
        try:
            mu_gap, Omega_gap, gap_alerts = load_gap_matrices(
                gap_input_dir, date_str, strict=True
            )
        except DataValidationError as exc:
            gap_alerts = [str(exc)]
        alerts.extend(gap_alerts)
    else:
        alerts.append("--gap-input-dir not specified.")
    if mu_gap is None or Omega_gap is None or _gap_alerts_fatal(gap_alerts):
        fallback["gap_data_missing"] = True
        logger.error(
            "[%s] Gap data missing or invalid. "
            "Returning flat position (w_final=0). No trading today.",
            date_str,
        )
        alerts.append("Gap data missing or invalid. Flat position (w_final=0) returned.")
        dummy_scores = np.zeros(n_j)
        dummy_Omega = np.eye(n_j) * 0.01
        pit_binning = {
            "assigned_bin": "Medium",
            "threshold_low": float("nan"),
            "threshold_high": float("nan"),
            "multiplier": run_cfg.fallback_multiplier,
            "current_ir": 0.0,
            "history_count": 0,
            "fallback_flag": True,
        }
        from leadlag.models.v2 import VERSION
        result = _run_safety_audits(
            w_final=np.zeros(n_j),
            scores=dummy_scores,
            mu_gap=np.zeros(n_j),
            Omega_gap=dummy_Omega,
            sigma_gap=np.ones(n_j) * 0.1,
            gap_input_dir=gap_input_dir,
            date_str=date_str,
            signal_date=date_str,
            run_cfg=run_cfg,
            fallback=fallback,
            pit_binning=pit_binning,
            alerts=alerts,
            pit_history_trade_dates=None,
            candidate="flat_position",
            version=VERSION,
        )
        return {
            "is_flat": True,
            "result": result,
            "mu_gap": None,
            "Omega_gap": None,
            "alerts": alerts,
        }
    return {
        "is_flat": False,
        "result": None,
        "mu_gap": mu_gap,
        "Omega_gap": Omega_gap,
        "alerts": alerts,
    }


def _resolve_current_index(df_exec: pd.DataFrame, trade_date: str) -> int:
    """Return the integer position of *trade_date* in *df_exec*."""
    return int(df_exec.index.get_loc(trade_date))


def _compute_ondemand(
    model: Any,
    trade_date: str,
    df_exec: pd.DataFrame,
    current_prices: dict[str, float],
    *,
    horizon: int = 1,
    snapshot: MarketSnapshot | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute on-demand gap-adjusted distribution.

    If ``snapshot`` is supplied, its point-in-time market data is used as the
    single source of truth for the as-of gap, betas, and TOPIX night return.
    """
    if model._blpx_model is None:
        raise RuntimeError("_compute_ondemand requires a blpx_model")
    inputs = model._blpx_model._prepare_common_inputs(df_exec, horizon=horizon)
    current_index = _resolve_current_index(df_exec, trade_date)
    blpx_result = model._blpx_model.compute_blp_signal(
        all_returns=inputs["jp_res_returns_p3"],
        current_index=current_index,
        v0_static=inputs["v0_static"],
        c_full=inputs["c_full_p3"],
        is_residual=True,
        return_matrices=True,
    )
    # Determine US market direction from the BLPX z-score of US returns.
    us_negative = float(np.nanmean(blpx_result["z_U_t"])) < 0.0
    # Select gap correction coefficients based on US direction.
    gap_open_coef = model._blpx_model.gap_open_coef
    if us_negative and getattr(model._blpx_model, "gap_open_coef_neg", None) is not None:
        gap_open_coef = model._blpx_model.gap_open_coef_neg
    topix_beta_coef = model._blpx_model.topix_beta_coef
    if us_negative and getattr(model._blpx_model, "topix_beta_coef_neg", None) is not None:
        topix_beta_coef = model._blpx_model.topix_beta_coef_neg
    # Build gap-adjusted distribution.
    gap_override, betas_t, topix_night_t = _extract_gap_inputs(
        df_exec, trade_date, current_prices, snapshot=snapshot
    )
    mu_raw, omega_raw = build_raw_distribution(
        blpx_result,
        vol_adjusted_target=getattr(model._blpx_model, "vol_adjusted_target", False),
    )
    mu_gap, omega_gap = compute_gap_adjusted_distribution(
        mu_raw,
        omega_raw,
        gap_override,
        betas_t,
        topix_night_t,
        gap_open_coef=gap_open_coef,
        topix_beta_coef=topix_beta_coef,
    )
    return mu_gap, omega_gap


def compute_distribution(
    model: Any,
    trade_date: str,
    df_exec: pd.DataFrame,
    current_prices: dict[str, float],
    *,
    horizon: int = 1,
    mu_pattern: str | None = None,
    omega_pattern: str | None = None,
    use_file_cache: bool = True,
    snapshot: MarketSnapshot | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (mu_gap, Omega_gap) for trade_date and horizon.

    If ``snapshot`` is supplied, its point-in-time data is used as the single
    source of truth for as-of gap, betas, and TOPIX night return.  Otherwise
    the per-date values are read from ``df_exec.loc[trade_date]``.

    1. The validated Step 2 file cache is the primary, trusted path.
    2. If the cache is missing and ``ondemand_fallback_enabled`` is True,
       fall back to on-demand BLPX computation.
    3. If ``shadow_ondemand_validation`` is True, also compute on-demand
       when the file cache exists and compare the two distributions.
    """
    if model._blpx_model is None:
        raise RuntimeError("compute_distribution requires a blpx_model")
    run_cfg = model.run_config
    gap_input_dir = getattr(model, "_current_gap_input_dir", None) or getattr(
        run_cfg, "gap_input_dir", None
    )
    ondemand_fallback = getattr(run_cfg, "ondemand_fallback_enabled", True)
    shadow_validation = getattr(run_cfg, "shadow_ondemand_validation", False)
    file_mu: np.ndarray | None = None
    file_omega: np.ndarray | None = None
    if use_file_cache and gap_input_dir is not None:
        if horizon == 1:
            _mu_pattern = mu_pattern or "matrices/mu_gap_{date}.npy"
            _omega_pattern = omega_pattern or "matrices/omega_gap_{date}.npy"
            _pattern_kwargs = None
        else:
            _mu_pattern = mu_pattern or run_cfg.mh_mu_file_pattern_h
            _omega_pattern = omega_pattern or run_cfg.mh_omega_file_pattern_h
            _pattern_kwargs = {"h": horizon}
        file_mu, file_omega, file_alerts = load_gap_matrices(
            gap_input_dir,
            trade_date,
            mu_pattern=_mu_pattern,
            omega_pattern=_omega_pattern,
            pattern_kwargs=_pattern_kwargs,
            n_j=model.n_j,
            strict=False,
        )
        if _gap_alerts_fatal(file_alerts):
            logger.error(
                "[%s] File cache gap matrices are invalid (%s); falling back to on-demand.",
                trade_date, ", ".join(file_alerts),
            )
            file_mu, file_omega = None, None
    if file_mu is not None and file_omega is not None:
        if shadow_validation:
            mu_ondemand, omega_ondemand = _compute_ondemand(
                model,
                trade_date=trade_date,
                df_exec=df_exec,
                current_prices=current_prices,
                horizon=horizon,
                snapshot=snapshot,
            )
            label = f"{trade_date}:h{horizon}"
            _compare_distribution(label, file_mu, file_omega, mu_ondemand, omega_ondemand)
        return file_mu, file_omega
    if ondemand_fallback:
        logger.warning(
            "[%s] Gap file cache missing (h=%d); computing on-demand.",
            trade_date, horizon,
        )
        return _compute_ondemand(
            model,
            trade_date=trade_date,
            df_exec=df_exec,
            current_prices=current_prices,
            horizon=horizon,
            snapshot=snapshot,
        )
    raise RuntimeError(
        f"Gap matrices missing for {trade_date} (h={horizon}) and on-demand fallback is disabled."
    )
