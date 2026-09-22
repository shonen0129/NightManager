"""V2 gap-distribution I/O and on-demand helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from leadlag.config.schemas import ProductionV2RunConfig
from leadlag.data.horizon_returns import compute_cumulative_returns
from leadlag.data.pit_lake import MarketSnapshot
from leadlag.data.tickers import JP_TICKERS
from leadlag.data.validation import DataValidationError
from leadlag.models.v2.audit_comparator import _run_safety_audits
from leadlag.pipeline.gap_distribution import compute_gap_distribution, select_gap_coefficients
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


def _extract_horizon_snapshot_inputs(
    historical_frame: pd.DataFrame,
    trade_date: str,
    horizon: int,
    snapshot: MarketSnapshot,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Rebuild a cumulative h-day gap from historical rows and today's snapshot.

    ``compute_cumulative_returns`` produces the h-day value from rows ending at
    the trade date.  For live decisions the current row's gap and TOPIX night
    return must come from the PIT snapshot, while the preceding h-1 rows stay
    on the historical execution frame.  Returning ``None`` preserves the
    existing zero/missing-data behavior when the window is incomplete.
    """
    if horizon <= 1 or trade_date not in historical_frame.index:
        return None
    try:
        snapshot_gap = np.asarray(snapshot.jp_gap_returns, dtype=float)
        snapshot_betas = np.asarray(snapshot.jp_betas, dtype=float)
    except (TypeError, ValueError):
        return None
    if snapshot_gap.shape != (len(JP_TICKERS),) or snapshot_betas.shape != (len(JP_TICKERS),):
        return None

    window = historical_frame.loc[:trade_date].tail(horizon)
    if len(window) != horizon:
        return None

    def _compound(values: np.ndarray, current: float) -> float:
        values = np.asarray(values, dtype=float).copy()
        # A missing live observation follows the one-day snapshot fallback of
        # treating that component as zero.  Missing historical rows still
        # invalidate the cumulative window, matching the existing rolling
        # frame behavior.
        if not np.isfinite(values[:-1]).all():
            return 0.0
        values[-1] = current if np.isfinite(current) else 0.0
        return float(np.prod(1.0 + values) - 1.0)

    gap_override = np.zeros(len(JP_TICKERS), dtype=float)
    for index, ticker in enumerate(JP_TICKERS):
        column = f"jp_gap_{ticker}"
        if column not in window.columns:
            return None
        gap_override[index] = _compound(window[column].to_numpy(dtype=float), snapshot_gap[index])

    if "topix_night_return" not in window.columns:
        return None
    topix_night_t = _compound(
        window["topix_night_return"].to_numpy(dtype=float),
        float(snapshot.topix_night_return),
    )
    if not np.isfinite(topix_night_t):
        topix_night_t = 0.0
    betas_t = np.nan_to_num(snapshot_betas, nan=0.0, posinf=0.0, neginf=0.0)
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
    open_910_returns: pd.DataFrame | None = None,
    allow_implicit_io: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute on-demand gap-adjusted distribution.

    If ``snapshot`` is supplied, its point-in-time market data is used as the
    single source of truth for the as-of gap, betas, and TOPIX night return.
    """
    if model._blpx_model is None:
        raise RuntimeError("_compute_ondemand requires a blpx_model")
    # The feature frame uses h-day cumulative returns, while the JP target is
    # the corresponding 9:10-to-close return over the original execution
    # frame.  Passing the target explicitly keeps this aligned with the
    # research gap generator and prevents accidentally treating cumulative
    # ``jp_oc`` columns as intraday target inputs.
    model_frame = compute_cumulative_returns(df_exec, horizon)
    from leadlag.data.intraday_inputs import compute_jp_target_returns

    y_jp_target = compute_jp_target_returns(
        df_exec,
        JP_TICKERS,
        horizon=horizon,
        open_910_returns=open_910_returns,
        allow_implicit_io=allow_implicit_io,
        required_index=[trade_date],
    )
    inputs = model._blpx_model._prepare_common_inputs(
        model_frame,
        horizon=horizon,
        y_jp_target=y_jp_target,
        open_910_returns=open_910_returns,
        allow_implicit_io=allow_implicit_io,
    )
    current_index = _resolve_current_index(model_frame, trade_date)
    blpx_result = model._blpx_model.compute_blp_signal(
        all_returns=inputs["jp_res_returns_p3"],
        current_index=current_index,
        v0_static=inputs["v0_static"],
        c_full=inputs["c_full_p3"],
        is_residual=True,
        return_matrices=True,
    )
    gap_open_coef, topix_beta_coef = select_gap_coefficients(model._blpx_model, blpx_result)
    # Build gap-adjusted distribution.
    # The research cache uses the horizon-specific cumulative gap and
    # overnight columns.  Reconstruct those same columns for h>1; a one-day
    # snapshot remains the authoritative point-in-time source for h=1.
    if horizon == 1 and snapshot is not None:
        gap_override, betas_t, topix_night_t = _extract_gap_inputs(
            df_exec, trade_date, current_prices, snapshot=snapshot
        )
    else:
        snapshot_inputs = (
            _extract_horizon_snapshot_inputs(df_exec, trade_date, horizon, snapshot)
            if horizon > 1 and snapshot is not None
            else None
        )
        if snapshot_inputs is not None:
            gap_override, betas_t, topix_night_t = snapshot_inputs
        else:
            gap_override, betas_t, topix_night_t = _extract_gap_inputs(
                df_exec, trade_date, current_prices, snapshot=None
            )
            if horizon > 1 and trade_date in model_frame.index:
                row = model_frame.loc[trade_date]
                gap_override = np.asarray(
                    [row.get(f"jp_gap_{ticker}", 0.0) for ticker in JP_TICKERS],
                    dtype=float,
                )
                gap_override = np.nan_to_num(gap_override, nan=0.0, posinf=0.0, neginf=0.0)
                if "topix_night_return" in row:
                    topix_night_t = float(row["topix_night_return"])
                    if not np.isfinite(topix_night_t):
                        topix_night_t = 0.0
    computation = compute_gap_distribution(
        blpx_result,
        gap_override=gap_override,
        betas_t=betas_t,
        topix_night_t=topix_night_t,
        vol_adjusted_target=getattr(model._blpx_model, "vol_adjusted_target", False),
        gap_open_coef=gap_open_coef,
        topix_beta_coef=topix_beta_coef,
    )
    return computation.mu_gap, computation.omega_gap


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
    from leadlag.domain.distribution import DistributionReason, DistributionResolutionError
    from leadlag.models.v2.fallback_policy import FallbackPolicy

    result = FallbackPolicy.default(
        model, use_file_cache=use_file_cache, mu_pattern=mu_pattern, omega_pattern=omega_pattern,
    ).resolve(trade_date, df_exec, current_prices, horizon=horizon, snapshot=snapshot)
    if result.mu_gap is None or result.Omega_gap is None or not result.is_available:
        raise DistributionResolutionError(
            result.reason or DistributionReason.POLICY_EXHAUSTED,
            "; ".join(result.alerts or []), attempts=result.attempts,
        )
    return result.mu_gap, result.Omega_gap
