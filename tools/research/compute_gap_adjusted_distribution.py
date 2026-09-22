#!/usr/bin/env python
"""Compute, Save, and Diagnose Gap-Adjusted Prediction Distribution (Step 2).

Transforms pre-gap raw distribution parameters (mu_raw, Omega_raw) to Japanese gap-adjusted
9:10-to-close distribution parameters (mu_gap, Omega_gap) using delta-approximation.
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Add src/ to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.broker.tachibana.session_cache import save_open_prices_cache
from leadlag.core.portfolio import solve_baseline_style
from leadlag.core.signal import build_weights_minvar
from leadlag.data.decision_cache import save_decision_cache
from leadlag.data.gap_store import GapStore
from leadlag.data.horizon_returns import (
    compute_cumulative_returns as _shared_compute_cumulative_returns,
)
from leadlag.data.intraday_inputs import (
    build_5m_910_prices,
    build_open_910_returns,
    compute_jp_target_returns,
)
from leadlag.data.market_data_cache import save_df_exec_to_local_cache
from leadlag.data.pit_lake import PITDataLake
from leadlag.data.tickers import JP_TICKERS, TOPIX_TICKER
from leadlag.execution.config import load_config_from_yaml
from leadlag.models.signal_enhancement import apply_multi_horizon_blend, apply_rank_reversal_overlay
from leadlag.models.v2.gap_io import _extract_horizon_snapshot_inputs
from leadlag.pipeline.gap_distribution import compute_gap_distribution, select_gap_coefficients
from leadlag.pipeline.gap_reporting import (
    build_gap_diagnostic_frames,
    write_gap_diagnostic_frames,
)
from leadlag.runner.model_factory import build_blpx_model
from leadlag.utils.gap_matrix_io import save_gap_matrices
from leadlag.utils.gap_provenance import (
    config_version,
    gap_inputs_version,
    input_version,
    model_version,
    open_910_version,
)
from research.diagnostics.gap_inputs import (
    attach_topix_trade_returns,
    build_gap_historical_inputs,
    inject_tachibana_realtime_prices,
    load_gap_execution_inputs,
    mask_future_jp_labels,
    prepare_gap_model_inputs,
)
from research.diagnostics.gap_outputs import (
    compute_pit_bins,
    prepare_portfolio_output_frame,
    render_gap_diagnostics,
)
from research.diagnostics.gap_portfolio import (
    add_baseline_diagnostics,
    build_portfolio_diagnostic_record,
    evaluate_gap_portfolio,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("GapAdjustedDistribution")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Residual-BLPX Step 2 Gap-Adjusted Distribution Diagnostics")
    parser.add_argument("--distribution-input-dir", default="var/live/pipeline_data/distribution_diagnostics/20260708_184812", help="Step 1 diagnostics input directory")
    parser.add_argument("--validation-input-dir", default="var/live/pipeline_data/distribution_validation/20260614_235912", help="Step 1 validation input directory")
    parser.add_argument("--vol-state-panel", default="var/live/pipeline_data/vol_state_diagnostics/20260614_115821/state_panel.csv", help="US Vol State Panel CSV path")
    parser.add_argument("--config", default="configs/production/production.yaml", help="Path to production YAML config")
    parser.add_argument("--model", default="production_residual_blpx", help="Model identifier")
    parser.add_argument("--results-dir", default="var/live/pipeline_data/diagnostics_weights", help="Validation/weights folder")
    parser.add_argument("--output-dir", default="var/live/pipeline_data/gap_adjusted_distribution", help="Output directory")
    parser.add_argument("--start", default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2026-06-14", help="End date (YYYY-MM-DD)")
    parser.add_argument("--bin-method", choices=["tertile", "quintile"], default="tertile", help="Binning method")
    parser.add_argument("--rolling-bin-window", type=int, default=252, help="Rolling window for PIT binning")
    parser.add_argument("--expanding-min-window", type=int, default=252, help="Expanding min window for PIT binning")
    parser.add_argument("--save-daily-matrices", type=str, default="true", help="Save daily matrices (true/false)")
    parser.add_argument("--save-to-gap-store", type=str, default="true", help="Also write daily gap matrices to the canonical GapStore (true/false)")
    parser.add_argument("--compare-pre-gap", type=str, default="true", help="Compare with pre-gap metrics (true/false)")
    parser.add_argument("--save-multi-horizon", type=str, default="true", help="Save h=3/h=5 gap matrices for multi-horizon blend (true/false)")
    parser.add_argument("--cumulative-method", choices=["sum", "cumprod"], default="cumprod", help="Method for multi-horizon cumulative returns")
    parser.add_argument("--save-rank-reversal", type=str, default="true", help="Save daily rank reversal signal for CS overlay (true/false)")
    parser.add_argument("--mh-horizons", type=str, default="3,5", help="Comma-separated multi-horizon days to compute")
    parser.add_argument("--use-tachibana-prices", type=str, default="false", help="Inject real-time prices from Tachibana API for today's gap computation (true/false)")
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of parallel workers for per-date computation. 1 = sequential, -1 = all cores")
    parser.add_argument("--self-test", action="store_true", help="Run self-tests and exit")
    return parser.parse_args()


def str_to_bool(val: str) -> bool:
    """Convert string to boolean."""
    return str(val).lower() in ("true", "1", "yes", "t", "y")


def compute_cumulative_returns(
    df_exec: pd.DataFrame,
    horizon: int,
    method: str = "cumprod",
) -> pd.DataFrame:
    """Create a modified df_exec with cumulative h-day returns for US and JP.

    Used for multi-horizon signal blending (Phase 2A).
    For ``method="cumprod"`` (default) the exact compounded return is used:
        (1 + r_1) * (1 + r_2) * ... * (1 + r_h) - 1
    For ``method="sum"`` the legacy simple-sum approximation is retained for
    comparison/backward compatibility.
    """
    return _shared_compute_cumulative_returns(df_exec, horizon, method=method)


def compute_rank_reversal_for_date(df_exec: pd.DataFrame, i: int) -> np.ndarray:
    """Compute cross-sectional rank reversal signal for date index *i*.

    Rank reversal = -(rank(t-1) - rank(t-2)) for JP open-close returns.
    Shifted by 1 day to avoid lookahead. Returns NaN-filled array for
    indices where the signal cannot be computed (i < 2).

    Args:
        df_exec: Full execution DataFrame with jp_oc_{ticker} columns.
        i: Current date index in df_exec.

    Returns:
        Array of shape (n_j,) with rank reversal values (negated rank change).
    """
    n_j = len(JP_TICKERS)
    if i < 2:
        return np.full(n_j, np.nan)

    jp_oc_cols = [f"jp_oc_{tk}" for tk in JP_TICKERS]
    df_oc = df_exec[jp_oc_cols].copy()
    df_oc.columns = JP_TICKERS

    # Shift by 1 day for lookahead safety
    ranks = df_oc.shift(1).rank(axis=1)
    rank_change = ranks.diff()

    if i >= len(rank_change):
        return np.full(n_j, np.nan)

    vals = -rank_change.iloc[i].values.astype(float)
    return vals


def run_self_tests() -> int:
    """Run validation self-tests."""
    logger.info("=== Running Self-Tests ===")
    # 1. Dimension Check for D_gap @ Omega @ D_gap
    n_j = 17
    Omega_raw = np.eye(n_j)
    denom = np.ones(n_j)
    denom[0] = 0.5
    denom[1] = 0.05  # below floor
    denom_floored = np.maximum(denom, 0.1)

    # Check floor behavior
    assert denom_floored[1] == 0.1, "Denominator floor not working"

    D_gap = np.diag(1.0 / denom_floored)
    Omega_gap = D_gap @ Omega_raw @ D_gap
    Omega_gap = 0.5 * (Omega_gap + Omega_gap.T)
    assert Omega_gap.shape == (17, 17), "Dimensions are not 17x17"
    assert np.allclose(Omega_gap, Omega_gap.T, atol=1e-15), "Omega_gap is not symmetric"

    # 2. Reconstruct check (GapOpen_filt = 0 => mu_gap = mu_raw, Omega_gap = Omega_raw)
    mu_raw = np.random.randn(n_j)
    denom_zero = np.ones(n_j)  # meaning GapOpen_filt = 0
    mu_gap = (1.0 + mu_raw) / denom_zero - 1.0
    assert np.allclose(mu_gap, mu_raw), "GapOpen_filt=0 mu adjustment failed"
    Omega_gap_zero = np.diag(1.0/denom_zero) @ Omega_raw @ np.diag(1.0/denom_zero)
    assert np.allclose(Omega_gap_zero, Omega_raw), "GapOpen_filt=0 Omega adjustment failed"

    # 3. PIT bin boundaries test
    dates = pd.date_range("2026-01-01", periods=30)
    ir_series = pd.Series(list(range(1, 31)), index=dates)
    bins = compute_pit_bins(ir_series, "tertile", rolling_window=15)
    assert bins.iloc[0:15].isna().all(), "PIT bin should be NaN for index < window"
    assert bins.iloc[15] == "High", f"PIT bin failed, expected High, got {bins.iloc[15]}"

    # Changing day t value should not affect day t boundary assignment
    ir_series_leak = ir_series.copy()
    ir_series_leak.iloc[15] = -999.0
    bins_leak = compute_pit_bins(ir_series_leak, "tertile", rolling_window=15)
    assert bins_leak.iloc[15] == "Low", "PIT bin with different day t val failed"

    # Ticker order, floor logic, realized cost diagnostic verification
    assert JP_TICKERS[0] == "1617.T", "JP TICKERS standard check failed"

    logger.info("=== All Self-Tests Passed ===")
    return 0


# ---------------------------------------------------------------------------
# Module-level _process_date implementation (extracted from main() for testing)
# ---------------------------------------------------------------------------

@dataclass
class GapDistContext:
    """Read-only context for _process_date_impl."""
    df_exec: pd.DataFrame
    model: Any
    jp_gap: Any
    jp_beta: Any
    topix_night: Any
    jp_res_returns_p3: np.ndarray
    v0_static: np.ndarray
    c_full_p3: np.ndarray
    dist_in_dir: Path
    out_dir: Path
    save_daily_m: bool
    save_mh: bool
    mh_horizons: list
    mh_models: dict
    mh_inputs: dict
    save_rr: bool
    y_jp_target: np.ndarray
    weights_df: pd.DataFrame
    turnover_map: dict
    cfg: dict
    c: float
    b: float
    bl_mh_enabled: bool
    bl_mh_horizons: tuple
    bl_mh_weights: tuple
    bl_mh_mu_pattern: str
    bl_mh_omega_pattern: str
    bl_cs_overlay_enabled: bool
    bl_cs_overlay_weight: float
    bl_rr_pattern: str
    bl_long_count: int
    bl_short_count: int
    bl_minvar_enabled: bool
    bl_minvar_alpha: float
    bl_baseline_gross: float
    bl_cost_bps_per_gross: float
    gap_store: Any | None = None
    save_to_gap_store: bool = False
    bundle_model_version: str = model_version()
    bundle_config_version: str = ""
    bundle_ticker_order: tuple[str, ...] = tuple(JP_TICKERS)
    open_910_returns: pd.DataFrame | None = None
    historical_inputs: Any | None = None
    historical_inputs_by_horizon: dict[int, Any] = field(default_factory=dict)


@dataclass
class GapDistAccumulators:
    """Mutable accumulators updated by _process_date_impl."""
    dropped_count: int = 0
    missing_data_count: int = 0
    nan_inf_count: int = 0
    neg_eigen_days_gap: int = 0
    days_with_min_eigen_lt_neg_1e_8_gap: int = 0
    days_with_diag_le_zero_gap: int = 0
    symmetry_max_err_gap: float = 0.0
    denominator_min_overall: float = 1e9
    denominator_floor_hit_count_overall: int = 0
    leakage_violation: bool = False
    all_dates_audit: bool = True
    gap_long_records: list = field(default_factory=list)
    gap_daily_records: list = field(default_factory=list)
    dist_long_records: list = field(default_factory=list)
    dist_daily_records: list = field(default_factory=list)
    omega_gap_daily_records: list = field(default_factory=list)
    omega_gap_ticker_records: dict = field(default_factory=dict)
    portfolio_diagnostics_records: list = field(default_factory=list)


def _process_date_impl(dt: pd.Timestamp, ctx: GapDistContext, acc: GapDistAccumulators) -> None:
    """Process a single date for gap-adjusted distribution computation.

    This is a module-level function to enable unit testing with synthetic data.
    Mutates *acc* in place with diagnostic counters and record lists.
    """
    i = ctx.df_exec.index.get_indexer([dt])[0]
    if i == -1:
        logger.warning(f"Trade date {dt.strftime('%Y-%m-%d')} not found in df_exec index; treating as missing data.")
        acc.missing_data_count += 1
        return
    if i < ctx.model.corr_window:
        acc.dropped_count += 1
        return

    sig_date = ctx.df_exec["sig_date"].values[i]
    sig_date_dt = pd.to_datetime(sig_date, format="ISO8601").tz_localize(None).normalize()
    trade_date_dt = pd.to_datetime(dt, format="ISO8601").tz_localize(None).normalize()

    # 1. Leakage check
    if not (sig_date_dt < trade_date_dt):
        acc.all_dates_audit = False
        acc.leakage_violation = True

    date_str = dt.strftime("%Y-%m-%d")
    dt_str = dt.strftime("%Y%m%d")
    sig_dt_str = sig_date_dt.strftime("%Y%m%d")

    # Step 1 DistributionDiagnostics names omega_struct by sig_date (the last US
    # trading day in the window). Load that exact matrix first; only fall back to
    # matrices dated on or before the signal date to avoid using a covariance
    # computed from data observed after the signal was generated.
    omega_struct_file = ctx.dist_in_dir / "matrices" / f"omega_struct_{sig_dt_str}.npy"
    use_step1_covariance = omega_struct_file.exists()
    if not omega_struct_file.exists():
        # Fallback: most recent available omega_struct on or before sig_date.
        # Restrict to omega_struct_YYYYMMDD.npy; ignore omega_struct_psd_YYYYMMDD.npy.
        import re

        fallback_files = sorted(
            f for f in (ctx.dist_in_dir / "matrices").glob("omega_struct_*.npy")
            if re.fullmatch(r"omega_struct_(\d{8})", f.stem)
        )
        fallback_files = [
            f for f in fallback_files
            if f.stem.split("_")[-1] <= sig_dt_str
        ]
        if fallback_files:
            omega_struct_file = fallback_files[-1]
            logger.warning(
                f"omega_struct for signal date {sig_date_dt.date()} not found, "
                f"found fallback {omega_struct_file.name}; using BLPX covariance "
                "to preserve cache/on-demand parity"
            )
            # A matrix computed for another signal date is not the same
            # distribution as the on-demand BLPX result.  Keep the historical
            # file discovery for diagnostics, but do not use stale covariance
            # as the h=1 output.
            Omega_struct = None
            use_step1_covariance = False
        else:
            # Backward compatibility: some historical runs / test fixtures named
            # the matrix by trade date. This branch is a last-resort safety net
            # and should not be exercised in normal live operation.
            omega_struct_file = ctx.dist_in_dir / "matrices" / f"omega_struct_{dt_str}.npy"
            if not omega_struct_file.exists():
                logger.warning(f"Step 1 omega matrix file missing on {date_str}: {omega_struct_file}")
                acc.missing_data_count += 1
                return
            Omega_struct = np.load(omega_struct_file)
            # Legacy fixtures/runs used the trade-date filename.  Preserve
            # that explicit compatibility path; only an older *fallback* file
            # is excluded from the h=1 covariance calculation above.
            use_step1_covariance = True
    else:
        Omega_struct = np.load(omega_struct_file)

    if Omega_struct is not None and not np.isfinite(Omega_struct).all():
        acc.nan_inf_count += 1
        logger.warning(f"NaN or Inf detected in Omega_struct on {date_str}")
        return

    # Daily parameters
    gap_override = np.nan_to_num(ctx.jp_gap[i], nan=0.0) if ctx.jp_gap is not None else np.zeros(ctx.model.n_j)
    betas_t = np.asarray(ctx.jp_beta[i], dtype=float) if ctx.jp_beta is not None else np.zeros(ctx.model.n_j)
    # Clip NaN/Inf betas to 0.0 (no systematic gap exposure) to avoid NaN
    # propagating into the denominator and gap-adjusted covariance.
    if not np.isfinite(betas_t).all():
        logger.warning(f"Non-finite betas_t on {date_str}; clipping NaN/Inf to 0.0")
        betas_t = np.nan_to_num(betas_t, nan=0.0, posinf=0.0, neginf=0.0)
    topix_night_t = float(ctx.topix_night[i]) if ctx.topix_night is not None else 0.0
    execution_snapshot = None
    if ctx.open_910_returns is not None and all(
        f"jp_open_trade_{ticker}" in ctx.df_exec for ticker in JP_TICKERS
    ):
        try:
            execution_snapshot = PITDataLake(ctx.df_exec).get_execution_snapshot(
                trade_date_dt + pd.Timedelta(hours=9, minutes=10), ctx.open_910_returns
            )
        except ValueError as exc:
            acc.missing_data_count += 1
            logger.warning("Invalid execution prices on %s: %s", date_str, exc)
            return
        gap_override = np.asarray(execution_snapshot.jp_gap_returns)
        betas_t = np.nan_to_num(execution_snapshot.jp_betas, nan=0.0, posinf=0.0, neginf=0.0)
        topix_night_t = float(execution_snapshot.topix_night_return)

    # Run model to get raw std scaling and standardized predictions
    try:
        jp_res_returns_for_date = ctx.jp_res_returns_p3
        if ctx.historical_inputs is not None:
            # Validate the same 09:10 calculation boundary used by production
            # and mask all JP labels whose close was not available at that
            # boundary.  ``y_jp_target[i]`` below remains evaluation-only.
            ctx.historical_inputs.calculation_frame(
                trade_date_dt + pd.Timedelta(hours=9, minutes=10)
            )
            jp_res_returns_for_date = mask_future_jp_labels(
                ctx.jp_res_returns_p3,
                ctx.df_exec.index,
                trade_date_dt + pd.Timedelta(hours=9, minutes=10),
                n_u=ctx.model.n_u,
            )
        residual_blpx_res = ctx.model.compute_blp_signal(
            jp_res_returns_for_date,
            i,
            gap_override=gap_override,
            betas_t=betas_t,
            topix_night_t=topix_night_t,
            rolling_std=None,
            v0_static=ctx.v0_static,
            c_full=ctx.c_full_p3,
            is_residual=True,
            return_matrices=True,
        )
    except Exception as e:
        acc.missing_data_count += 1
        logger.warning(f"Error calling compute_blp_signal on {date_str}: {e}")
        return

    # Pure one-day reconstruction is shared with the production on-demand path.
    # An exact signal-date Step 1 covariance remains an explicit override.  A
    # stale fallback is intentionally ignored so cache and on-demand use the
    # same BLPX covariance source.
    gap_open_coef, topix_beta_coef = select_gap_coefficients(ctx.model, residual_blpx_res)
    computation = compute_gap_distribution(
        residual_blpx_res,
        gap_override=gap_override,
        betas_t=betas_t,
        topix_night_t=topix_night_t,
        vol_adjusted_target=ctx.model.vol_adjusted_target,
        gap_open_coef=gap_open_coef,
        topix_beta_coef=topix_beta_coef,
        omega_struct=Omega_struct if use_step1_covariance else None,
    )
    mu_raw = computation.mu_raw
    Omega_raw = computation.omega_raw
    mu_gap = computation.mu_gap
    Omega_gap = computation.omega_gap
    gap_filt = computation.gap_filt
    denominator = computation.denominator
    denominator_floored = computation.denominator_floored
    gap_syst = betas_t * topix_night_t
    gap_idio = gap_override - gap_syst
    floor_hit_flags = (denominator < 0.1).astype(int)

    acc.denominator_min_overall = min(acc.denominator_min_overall, float(np.min(denominator)))
    acc.denominator_floor_hit_count_overall += int(np.sum(floor_hit_flags))

    # Check matching with production model's signal
    signal_prod = residual_blpx_res["signal"]
    diff_mu = np.max(np.abs(mu_gap - signal_prod))
    if diff_mu > 1e-12:
        logger.warning(f"Difference in mu_gap and production signal on {date_str}: {diff_mu:.2e}")

    # Symmetrize
    sym_err = float(np.max(np.abs(Omega_gap - Omega_gap.T)))
    acc.symmetry_max_err_gap = max(acc.symmetry_max_err_gap, sym_err)
    Omega_gap = 0.5 * (Omega_gap + Omega_gap.T)

    # Check finitude of transformed parameters
    if not (np.isfinite(mu_raw).all() and np.isfinite(Omega_raw).all() and np.isfinite(mu_gap).all() and np.isfinite(Omega_gap).all()):
        acc.nan_inf_count += 1
        logger.warning(f"NaN or Inf detected in transformed variables on {date_str}")
        return

    # Eigenvalue decomposition
    try:
        eigvals_gap, _ = np.linalg.eigh(Omega_gap)
    except np.linalg.LinAlgError as e:
        logger.warning(f"Eigenvalues did not converge for Omega_gap on {date_str}: {e}")
        acc.missing_data_count += 1
        return

    min_eig_gap = float(np.min(eigvals_gap))
    max_eig_gap = float(np.max(eigvals_gap))
    neg_count_gap = int(np.sum(eigvals_gap < 0))
    if neg_count_gap > 0:
        acc.neg_eigen_days_gap += 1
    if min_eig_gap < -1e-8:
        acc.days_with_min_eigen_lt_neg_1e_8_gap += 1

    diag_gap = np.diag(Omega_gap)
    if np.any(diag_gap <= 0):
        acc.days_with_diag_le_zero_gap += 1

    cond_num_gap = float(max_eig_gap / min_eig_gap) if min_eig_gap > 0 else np.nan
    trace_gap = float(np.trace(Omega_gap))

    # Off-diagonal correlation gap
    std_gap = np.sqrt(np.maximum(diag_gap, 1e-10))
    R_gap = Omega_gap / np.outer(std_gap, std_gap)
    avg_offdiag_gap = float((np.sum(R_gap) - ctx.model.n_j) / (ctx.model.n_j * (ctx.model.n_j - 1)))
    frob_gap = float(np.linalg.norm(Omega_gap, "fro"))

    # Raw covariance diagnostics (for comparison)
    try:
        eigvals_raw, _ = np.linalg.eigh(Omega_raw)
    except np.linalg.LinAlgError as e:
        logger.warning(f"Eigenvalues did not converge for Omega_raw on {date_str}: {e}")
        acc.missing_data_count += 1
        return

    min_eig_raw = float(np.min(eigvals_raw))
    max_eig_raw = float(np.max(eigvals_raw))
    neg_count_raw = int(np.sum(eigvals_raw < 0))
    diag_raw = np.diag(Omega_raw)
    cond_num_raw = float(max_eig_raw / min_eig_raw) if min_eig_raw > 0 else np.nan
    trace_raw = float(np.trace(Omega_raw))
    std_raw = np.sqrt(np.maximum(diag_raw, 1e-10))
    R_raw = Omega_raw / np.outer(std_raw, std_raw)
    avg_offdiag_raw = float((np.sum(R_raw) - ctx.model.n_j) / (ctx.model.n_j * (ctx.model.n_j - 1)))
    frob_raw = float(np.linalg.norm(Omega_raw, "fro"))

    # Frobenius norm ratio and trace scale ratio
    fro_ratio = frob_gap / frob_raw if frob_raw > 0 else 1.0
    mean_diag_ratio = float(np.mean(diag_gap / diag_raw))

    bundle_metadata = {
        "sig_date": str(sig_date),
        "trade_date": date_str,
        "horizon": 1,
        "source": "compute_gap_adjusted_distribution",
        "input_version": input_version(ctx.df_exec, dt),
        "model_version": ctx.bundle_model_version,
        "config_version": ctx.bundle_config_version,
        "ticker_order": list(ctx.bundle_ticker_order),
        "calculation_as_of": f"{date_str}T09:10:00+09:00",
        "label_available_at": "trade_date+15:30",
        "observed_at": (
            dict(ctx.historical_inputs.observed_at_for(dt))
            if ctx.historical_inputs is not None else None
        ),
        "observed_at_source": "session_boundary_contract",
        "historical_inputs_fingerprint": (
            ctx.historical_inputs.fingerprint if ctx.historical_inputs is not None else None
        ),
        "execution_price_sources": (
            dict(execution_snapshot.price_sources) if execution_snapshot is not None else {}
        ),
    }
    if ctx.open_910_returns is not None:
        bundle_metadata["open_910_version"] = open_910_version(ctx.open_910_returns, dt)
    bundle_metadata["gap_inputs_version"] = gap_inputs_version(
        date_str, gap_override, betas_t, topix_night_t, horizon=1
    )

    # Save matrices if requested
    if ctx.save_daily_m:
        if not save_gap_matrices(
            ctx.out_dir,
            date_str,
            mu_gap,
            Omega_gap,
            metadata=bundle_metadata,
        ):
            logger.warning("Failed to save h=1 gap bundle on %s", date_str)

    if ctx.gap_store is not None and ctx.save_to_gap_store:
        try:
            ctx.gap_store.save(
                date_str,
                mu_gap,
                Omega_gap,
                metadata=bundle_metadata,
            )
        except Exception as e:
            logger.warning(f"Failed to save gap pair to GapStore on {date_str}: {e}")

    # --- Phase 2A: Save multi-horizon gap matrices ---
    if ctx.save_mh and ctx.mh_horizons:
        for h in ctx.mh_horizons:
            try:
                model_h = ctx.mh_models[h]
                inputs_h = ctx.mh_inputs[h]
                gap_h = inputs_h["jp_gap"]
                beta_h = inputs_h["jp_beta"]
                topix_night_h = inputs_h["topix_night"]
                jp_res_h = inputs_h["jp_res_returns_p3"]
                c_full_h = inputs_h["c_full_p3"]
                v0_h = inputs_h["v0_static"]
                jp_res_h_for_date = jp_res_h
                history_h = ctx.historical_inputs_by_horizon.get(h)
                if history_h is not None:
                    history_h.calculation_frame(
                        trade_date_dt + pd.Timedelta(hours=9, minutes=10)
                    )
                    jp_res_h_for_date = mask_future_jp_labels(
                        jp_res_h,
                        ctx.df_exec.index,
                        trade_date_dt + pd.Timedelta(hours=9, minutes=10),
                        n_u=model_h.n_u,
                    )

                gap_override_h = np.nan_to_num(gap_h[i], nan=0.0) if gap_h is not None else np.zeros(model_h.n_j)
                betas_t_h = np.asarray(beta_h[i], dtype=float) if beta_h is not None else np.zeros(model_h.n_j)
                # Clip NaN/Inf betas to 0.0 (no systematic gap exposure) for multi-horizon too.
                if not np.isfinite(betas_t_h).all():
                    logger.warning(f"Non-finite betas_t on {date_str} (h={h}); clipping NaN/Inf to 0.0")
                    betas_t_h = np.nan_to_num(betas_t_h, nan=0.0, posinf=0.0, neginf=0.0)
                topix_night_t_h = float(topix_night_h[i]) if topix_night_h is not None else 0.0
                if execution_snapshot is not None:
                    horizon_snapshot_inputs = _extract_horizon_snapshot_inputs(
                        ctx.df_exec, date_str, h, execution_snapshot
                    )
                    if horizon_snapshot_inputs is None:
                        raise ValueError(f"Incomplete execution price window for h={h}")
                    gap_override_h, betas_t_h, topix_night_t_h = horizon_snapshot_inputs

                res_h = model_h.compute_blp_signal(
                    jp_res_h_for_date, i,
                    gap_override=gap_override_h,
                    betas_t=betas_t_h,
                    topix_night_t=topix_night_t_h,
                    rolling_std=None,
                    v0_static=v0_h,
                    c_full=c_full_h,
                    is_residual=True,
                    return_matrices=True,
                )

                gap_open_coef_h, topix_beta_coef_h = select_gap_coefficients(model_h, res_h)
                computation_h = compute_gap_distribution(
                    res_h,
                    gap_override=gap_override_h,
                    betas_t=betas_t_h,
                    topix_night_t=topix_night_t_h,
                    vol_adjusted_target=model_h.vol_adjusted_target,
                    gap_open_coef=gap_open_coef_h,
                    topix_beta_coef=topix_beta_coef_h,
                )
                mu_gap_h = computation_h.mu_gap
                Omega_gap_h = computation_h.omega_gap

                if not save_gap_matrices(
                    ctx.out_dir,
                    date_str,
                    mu_gap_h,
                    Omega_gap_h,
                    mu_pattern=f"matrices/mu_gap_h{h}_{{date}}.npy",
                    omega_pattern=f"matrices/omega_gap_h{h}_{{date}}.npy",
                    pattern_kwargs={"h": h},
                    metadata={
                        **bundle_metadata,
                        "horizon": h,
                        "historical_inputs_fingerprint": (
                            history_h.fingerprint if history_h is not None else None
                        ),
                        "observed_at": (
                            dict(history_h.observed_at_for(dt))
                            if history_h is not None else None
                        ),
                        "observed_at_source": "session_boundary_contract",
                        "gap_inputs_version": gap_inputs_version(
                            date_str, gap_override_h, betas_t_h, topix_night_t_h, horizon=h
                        ),
                    },
                ):
                    logger.warning("Failed to save h=%d gap bundle on %s", h, date_str)

                if ctx.gap_store is not None and ctx.save_to_gap_store:
                    try:
                        ctx.gap_store.save_horizon(
                            date_str,
                            mu_gap_h,
                            Omega_gap_h,
                            metadata={
                                **bundle_metadata,
                                "horizon": h,
                                "gap_inputs_version": gap_inputs_version(
                                    date_str, gap_override_h, betas_t_h, topix_night_t_h, horizon=h
                                ),
                            },
                            horizon=h,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to save h={h} gap pair to GapStore on {date_str}: {e}")
            except Exception as e:
                logger.warning(f"Multi-horizon h={h} failed on {date_str}: {e}")

    # --- Phase 2D: Save rank reversal signal ---
    if ctx.save_rr:
        rr_signal = compute_rank_reversal_for_date(ctx.df_exec, i)
        if np.isfinite(rr_signal).any():
            np.save(ctx.out_dir / "matrices" / f"rank_reversal_{dt_str}.npy", rr_signal)
            if ctx.gap_store is not None and ctx.save_to_gap_store:
                try:
                    ctx.gap_store.put(date_str, "rank_reversal", rr_signal)
                except Exception as e:
                    logger.warning(f"Failed to save rank reversal to GapStore on {date_str}: {e}")

    # Daily records
    acc.omega_gap_daily_records.append({
        "trade_date": date_str,
        "min_eigenvalue": min_eig_gap,
        "max_eigenvalue": max_eig_gap,
        "negative_eigen_count": neg_count_gap,
        "condition_number": cond_num_gap,
        "trace": trace_gap,
        "avg_offdiag_corr": avg_offdiag_gap,
        "frob_norm": frob_gap,
    })

    acc.dist_daily_records.append({
        "trade_date": date_str,
        "min_eigenvalue_raw": min_eig_raw,
        "min_eigenvalue_gap": min_eig_gap,
        "negative_eigen_count_raw": neg_count_raw,
        "negative_eigen_count_gap": neg_count_gap,
        "diag_nonpositive_count_raw": int(np.sum(diag_raw <= 0)),
        "diag_nonpositive_count_gap": int(np.sum(diag_gap <= 0)),
        "trace_raw": trace_raw,
        "trace_gap": trace_gap,
        "condition_number_raw": cond_num_raw,
        "condition_number_gap": cond_num_gap,
        "avg_offdiag_corr_raw": avg_offdiag_raw,
        "avg_offdiag_corr_gap": avg_offdiag_gap,
        "fro_norm_raw": frob_raw,
        "fro_norm_gap": frob_gap,
        "fro_norm_ratio_gap_vs_raw": fro_ratio,
        "mean_diag_ratio_gap_vs_raw": mean_diag_ratio,
    })

    # Daily Japanese gap stats
    mean_abs_gap = float(np.mean(np.abs(gap_override)))
    max_abs_gap = float(np.max(np.abs(gap_override)))
    disp_gap = float(np.std(gap_override))
    mean_abs_gap_filt = float(np.mean(np.abs(gap_filt)))
    mean_abs_gap_syst = float(np.mean(np.abs(gap_syst)))
    mean_abs_gap_idio = float(np.mean(np.abs(gap_idio)))
    max_abs_gap_filt = float(np.max(np.abs(gap_filt)))
    denom_min_t = float(np.min(denominator))
    floor_hits_t = int(np.sum(floor_hit_flags))

    acc.gap_daily_records.append({
        "trade_date": date_str,
        "mean_abs_GapOpen": mean_abs_gap,
        "max_abs_GapOpen": max_abs_gap,
        "dispersion_GapOpen": disp_gap,
        "mean_abs_GapOpen_filt": mean_abs_gap_filt,
        "mean_abs_GapOpen_syst": mean_abs_gap_syst,
        "mean_abs_GapOpen_idio": mean_abs_gap_idio,
        "max_abs_GapOpen_filt": max_abs_gap_filt,
        "denominator_min": denom_min_t,
        "denominator_floor_hit_count": floor_hits_t,
    })

    # Retrieve realized JP target returns
    realized_jp_returns = ctx.y_jp_target[i]

    # Stock-level long records
    for idx, tk in enumerate(JP_TICKERS):
        acc.gap_long_records.append({
            "signal_date": sig_date,
            "trade_date": date_str,
            "ticker": tk,
            "GapOpen": float(gap_override[idx]),
            "TOPIXNight": float(topix_night_t),
            "beta": float(betas_t[idx]),
            "GapOpen_syst": float(gap_syst[idx]),
            "GapOpen_idio": float(gap_idio[idx]),
            "GapOpen_filt": float(gap_filt[idx]),
            "denominator": float(denominator[idx]),
            "denominator_floored": float(denominator_floored[idx]),
            "availability": "POST_OPEN",
        })

        acc.dist_long_records.append({
            "signal_date": sig_date,
            "trade_date": date_str,
            "ticker": tk,
            "mu_raw": float(mu_raw[idx]),
            "mu_gap": float(mu_gap[idx]),
            "omega_diag_raw": float(diag_raw[idx]),
            "omega_std_raw": float(std_raw[idx]),
            "omega_diag_gap": float(diag_gap[idx]),
            "omega_std_gap": float(std_gap[idx]),
            "denominator": float(denominator[idx]),
            "denominator_floored": float(denominator_floored[idx]),
            "floor_hit": int(floor_hit_flags[idx]),
            "realized_target_return": float(realized_jp_returns[idx]),
        })

        acc.omega_gap_ticker_records.setdefault(tk, []).append({
            "date": date_str,
            "omega_raw_diag": float(diag_raw[idx]),
            "omega_gap_diag": float(diag_gap[idx]),
        })

    # Portfolio Weights
    w_t = np.zeros(ctx.model.n_j)
    if dt in ctx.weights_df.index:
        w_t = ctx.weights_df.loc[dt, JP_TICKERS].values

    turnover = ctx.turnover_map.get(dt, 0.0)
    portfolio_evaluation = evaluate_gap_portfolio(
        weights=w_t,
        realized_returns=realized_jp_returns,
        mu_raw=mu_raw,
        omega_raw=Omega_raw,
        mu_gap=mu_gap,
        omega_gap=Omega_gap,
        slippage_bps_per_side=ctx.cfg["costs"]["slippage_bps_per_side"],
        turnover=turnover,
    )

    # --- Baseline IR (consistent with production_v2.py current_ir) ---
    sigma_gap_bl = np.sqrt(np.maximum(np.diag(Omega_gap), 1e-6))
    scores_bl = mu_gap / sigma_gap_bl

    if ctx.bl_mh_enabled and len(ctx.bl_mh_horizons) > 1:
        scores_bl, _ = apply_multi_horizon_blend(
            scores_h1=scores_bl,
            gap_input_dir=ctx.out_dir,
            date_str=date_str,
            horizons=ctx.bl_mh_horizons,
            weights=ctx.bl_mh_weights,
            mu_pattern=ctx.bl_mh_mu_pattern,
            omega_pattern=ctx.bl_mh_omega_pattern,
            expected_identity={
                "input_version": input_version(ctx.df_exec, dt),
                "model_version": ctx.bundle_model_version,
                "config_version": ctx.bundle_config_version,
                "ticker_order": list(ctx.bundle_ticker_order),
            },
        )

    if ctx.bl_cs_overlay_enabled:
        scores_bl, _ = apply_rank_reversal_overlay(
            scores=scores_bl,
            gap_input_dir=ctx.out_dir,
            date_str=date_str,
            weight=ctx.bl_cs_overlay_weight,
            file_pattern=ctx.bl_rr_pattern,
        )

    sorted_idx_bl = np.argsort(scores_bl)
    short_idx_bl = sorted_idx_bl[:ctx.bl_short_count]
    long_idx_bl = sorted_idx_bl[-ctx.bl_long_count:]

    if ctx.bl_minvar_enabled:
        w_minvar_bl = build_weights_minvar(
            signal=scores_bl,
            q=float(ctx.bl_long_count) / ctx.model.n_j,
            n_j=ctx.model.n_j,
            Sigma_YY=Omega_gap,
            alpha=ctx.bl_minvar_alpha,
            enforce_sign=False,
        )
        w_baseline = w_minvar_bl * (ctx.bl_baseline_gross / 2.0)
    else:
        w_baseline = solve_baseline_style(
            scores_bl, long_idx_bl, short_idx_bl, baseline_gross=ctx.bl_baseline_gross
        )

    baseline_metrics = add_baseline_diagnostics(
        {},
        weights=w_baseline,
        mu_gap=mu_gap,
        omega_gap=Omega_gap,
        baseline_cost_bps_per_gross=ctx.bl_cost_bps_per_gross,
        baseline_gross=ctx.bl_baseline_gross,
    )
    gap_metrics = {
        "mean_abs_GapOpen": mean_abs_gap,
        "max_abs_GapOpen": max_abs_gap,
        "dispersion_GapOpen": disp_gap,
        "mean_abs_GapOpen_filt": mean_abs_gap_filt,
        "mean_abs_GapOpen_syst": mean_abs_gap_syst,
        "mean_abs_GapOpen_idio": mean_abs_gap_idio,
        "max_abs_GapOpen_filt": max_abs_gap_filt,
        "denominator_min": denom_min_t,
        "denominator_floor_hit_count": floor_hits_t,
    }
    acc.portfolio_diagnostics_records.append(
        build_portfolio_diagnostic_record(
            signal_date=sig_date,
            trade_date=date_str,
            evaluation=portfolio_evaluation,
            gap_metrics=gap_metrics,
            baseline_metrics=baseline_metrics,
        )
    )


def _merge_accumulators(dst: GapDistAccumulators, src: GapDistAccumulators) -> None:
    """Merge partial accumulator *src* into *dst* (thread-safe when called sequentially)."""
    dst.dropped_count += src.dropped_count
    dst.missing_data_count += src.missing_data_count
    dst.nan_inf_count += src.nan_inf_count
    dst.neg_eigen_days_gap += src.neg_eigen_days_gap
    dst.days_with_min_eigen_lt_neg_1e_8_gap += src.days_with_min_eigen_lt_neg_1e_8_gap
    dst.days_with_diag_le_zero_gap += src.days_with_diag_le_zero_gap
    dst.symmetry_max_err_gap = max(dst.symmetry_max_err_gap, src.symmetry_max_err_gap)
    dst.denominator_min_overall = min(dst.denominator_min_overall, src.denominator_min_overall)
    dst.denominator_floor_hit_count_overall += src.denominator_floor_hit_count_overall
    dst.leakage_violation = dst.leakage_violation or src.leakage_violation
    dst.all_dates_audit = dst.all_dates_audit and src.all_dates_audit
    dst.gap_long_records.extend(src.gap_long_records)
    dst.gap_daily_records.extend(src.gap_daily_records)
    dst.dist_long_records.extend(src.dist_long_records)
    dst.dist_daily_records.extend(src.dist_daily_records)
    dst.omega_gap_daily_records.extend(src.omega_gap_daily_records)
    dst.portfolio_diagnostics_records.extend(src.portfolio_diagnostics_records)
    for tk, records in src.omega_gap_ticker_records.items():
        dst.omega_gap_ticker_records.setdefault(tk, []).extend(records)


def _process_date_worker(dt: pd.Timestamp, ctx: GapDistContext) -> GapDistAccumulators:
    """Thread-safe worker: creates a local accumulator, processes one date, returns it."""
    local_acc = GapDistAccumulators()
    _process_date_impl(dt, ctx, local_acc)
    return local_acc


def main():
    args = parse_arguments()

    if args.self_test:
        sys.exit(run_self_tests())

    save_daily_m = str_to_bool(args.save_daily_matrices)
    save_to_gap_store = str_to_bool(args.save_to_gap_store)
    compare_pre = str_to_bool(args.compare_pre_gap)
    save_mh = str_to_bool(args.save_multi_horizon)
    save_rr = str_to_bool(args.save_rank_reversal)
    mh_horizons = [int(h) for h in args.mh_horizons.split(",")] if save_mh else []

    # Setup output paths
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / run_timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    # Canonical SQLite gap store (accumulates all historical dates).
    gap_store = None
    if save_to_gap_store:
        gap_store_path = Path(args.output_dir) / "gap_store.sqlite"
        gap_store = GapStore(gap_store_path)
        logger.info("Canonical gap store: %s", gap_store.path)

    if save_daily_m or (save_mh and mh_horizons):
        (out_dir / "matrices").mkdir(exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    logger.info(f"Establishing Step 2 output directory: {out_dir}")

    # 1. Load config
    cfg_path = ROOT / args.config
    logger.info(f"Loading config from {cfg_path}")
    app_config = load_config_from_yaml(cfg_path, strict=True)
    v2_cfg = app_config.v2
    # Keep the small context contract used by the per-date worker while all
    # values come from the validated inherited config.
    cfg = {"costs": {"slippage_bps_per_side": v2_cfg.costs.slippage_bps_per_side}}

    results_dir = Path(args.results_dir) if args.results_dir.startswith("results") else ROOT / args.results_dir
    dist_in_dir = Path(args.distribution_input_dir) if args.distribution_input_dir.startswith("results") else ROOT / args.distribution_input_dir

    # Extract config params for baseline IR (must match production_v2.py exactly)
    bl_long_count = v2_cfg.long_count
    bl_short_count = v2_cfg.short_count
    bl_baseline_gross = v2_cfg.baseline_gross
    bl_cost_bps_per_gross = v2_cfg.cost_bps_per_gross
    bl_minvar_enabled = v2_cfg.minvar_enabled
    bl_minvar_alpha = v2_cfg.minvar_alpha
    bl_mh_enabled = v2_cfg.mh_blend_enabled
    bl_mh_horizons = tuple(v2_cfg.mh_horizons)
    bl_mh_weights = tuple(v2_cfg.mh_weights)
    bl_mh_mu_pattern = v2_cfg.mh_mu_file_pattern_h
    bl_mh_omega_pattern = v2_cfg.mh_omega_file_pattern_h
    bl_cs_overlay_enabled = v2_cfg.cs_overlay_enabled
    bl_cs_overlay_weight = v2_cfg.cs_overlay_weight
    bl_rr_pattern = v2_cfg.cs_rank_reversal_file_pattern
    logger.info(
        "Baseline IR config: long=%d short=%d gross=%.1f cost_bps=%.1f minvar=%s alpha=%.2f mh=%s cs=%s",
        bl_long_count, bl_short_count, bl_baseline_gross, bl_cost_bps_per_gross,
        bl_minvar_enabled, bl_minvar_alpha, bl_mh_enabled, bl_cs_overlay_enabled,
    )

    # 2. Load raw market data and reuse Step 1 preprocessing when valid.
    logger.info("Loading market data...")
    market_inputs = load_gap_execution_inputs(beta_window=60)
    raw_data = market_inputs.raw_data
    df_exec = market_inputs.df_exec

    # Inject Tachibana real-time prices for today if enabled
    api_client = None
    opens_executor = None
    opens_future = None
    if str_to_bool(args.use_tachibana_prices):
        today = pd.Timestamp.now().tz_localize(None).normalize()
        from leadlag.execution.broker_ops import build_api_client

        api_client = build_api_client(api_url=None, api_token=None, api_dry_run=False)
        df_exec, api_client = inject_tachibana_realtime_prices(df_exec, raw_data, today, api_client=api_client)

        # Start open-price fetch in parallel with the main computation.
        # The decision step can then read these cached opens and skip an API call.
        tickers_for_opens = JP_TICKERS + [TOPIX_TICKER]

        def _fetch_and_cache_opens() -> None:
            try:
                opens = api_client.fetch_open_prices(tickers_for_opens, allow_missing=True)
                topix_open = opens.pop(TOPIX_TICKER, None)
                save_open_prices_cache(opens, topix_open, today.strftime("%Y%m%d"))
            except Exception as e:
                logger.warning("Parallel open-price fetch failed: %s", e)

        opens_executor = ThreadPoolExecutor(max_workers=1)
        opens_future = opens_executor.submit(_fetch_and_cache_opens)

    # Compute TOPIX returns (guard against zero TOPIX open prices).
    df_exec = attach_topix_trade_returns(
        df_exec,
        raw_data,
        preserve_today_placeholder=str_to_bool(args.use_tachibana_prices),
    )

    # Setup model
    logger.info("Instantiating Residual-BLPX model...")
    model = build_blpx_model(app_config)
    # Keep the exact 09:10 input frame used for targets and bundle provenance.
    # This prevents a later 5-minute correction from being hidden behind an
    # unchanged df_exec fingerprint.
    open_910_returns = build_open_910_returns(df_exec, JP_TICKERS)
    historical_inputs = build_gap_historical_inputs(
        df_exec,
        open_910_returns=open_910_returns,
        horizon=1,
        source="gap_generation_h1",
    )
    inputs = prepare_gap_model_inputs(
        model,
        df_exec,
        open_910_returns=open_910_returns,
        historical_inputs=historical_inputs,
    )

    # Fetch weights
    weights_file = results_dir / "daily_positions_Residual-BLPX_only.csv"
    if not weights_file.exists():
        logger.error(f"Weights file not found at {weights_file}")
        sys.exit(1)
    logger.info(f"Loading weights from {weights_file}")
    weights_df = pd.read_csv(weights_file, index_col=0)
    weights_df.index = pd.to_datetime(weights_df.index, format="ISO8601").tz_localize(None).normalize()

    # Model inputs
    y_jp_target = inputs.y_jp_target
    jp_gap = inputs.jp_gap
    jp_beta = inputs.jp_beta
    topix_night = inputs.topix_night
    jp_res_returns_p3 = inputs.jp_res_returns_p3
    c_full_p3 = inputs.c_full_p3
    v0_static = inputs.v0_static

    # --- Phase 2A: Multi-horizon model setup ---
    mh_models = {}   # {h: model_instance}
    mh_inputs = {}   # {h: inputs_dict}
    historical_inputs_by_horizon: dict[int, Any] = {}

    if save_mh and mh_horizons:
        # 5分足 09:10 価格は h-day target 計算で使う。h=3/h=5 両方に共有。
        p_910_df = build_5m_910_prices(df_exec, JP_TICKERS)

        for h in mh_horizons:
            logger.info(f"Setting up multi-horizon model for h={h}...")
            df_exec_h = compute_cumulative_returns(df_exec, h, method=args.cumulative_method)
            # h-day 9:10→大引け target を元の h=1 df_exec から直接計算。
            # df_exec_h 内の jp_oc は h 日積上げなので target 計算に使わない。
            y_jp_target_h = compute_jp_target_returns(
                df_exec, JP_TICKERS, horizon=h, p_910_df=p_910_df
            )
            model_h = build_blpx_model(app_config)
            historical_inputs_h = build_gap_historical_inputs(
                df_exec,
                open_910_returns=open_910_returns,
                horizon=h,
                source=f"gap_generation_h{h}",
            )
            inputs_h = prepare_gap_model_inputs(
                model_h,
                df_exec_h,
                y_jp_target=y_jp_target_h,
                horizon=h,
                open_910_returns=open_910_returns,
                historical_inputs=historical_inputs_h,
            )
            mh_models[h] = model_h
            mh_inputs[h] = inputs_h
            historical_inputs_by_horizon[h] = historical_inputs_h
            logger.info(f"  h={h} model ready (corr_window={model_h.corr_window})")

    if save_rr:
        logger.info("Rank reversal signal will be saved daily.")

    sim_dates = df_exec.index
    sim_dates_slice = sim_dates[sim_dates >= args.start]
    if args.end != "latest":
        sim_dates_slice = sim_dates_slice[sim_dates_slice <= args.end]

    logger.info(f"Diagnostics window: {sim_dates_slice[0].strftime('%Y-%m-%d')} to {sim_dates_slice[-1].strftime('%Y-%m-%d')} ({len(sim_dates_slice)} days)")

    # Cache and parameters
    c = model.gap_open_coef
    b = model.topix_beta_coef
    logger.info(f"Model parameters: gap_open_coef c={c:.2f}, topix_beta_coef b={b:.2f}")

    # Pre-compute turnovers from weights_df (removes sequential w_prev dependency for parallelization)
    turnover_map: dict[pd.Timestamp, float] = {}
    _w_prev = np.zeros(model.n_j)
    for _dt in sim_dates_slice:
        _w_t = np.zeros(model.n_j)
        if _dt in weights_df.index:
            _w_t = weights_df.loc[_dt, JP_TICKERS].values
        turnover_map[_dt] = float(np.sum(np.abs(_w_t - _w_prev)) / 2.0)
        _w_prev = _w_t.copy()

    n_jobs = args.n_jobs

    # Build context and accumulators for _process_date_impl
    acc = GapDistAccumulators(
        omega_gap_ticker_records={tk: [] for tk in JP_TICKERS},
    )
    ctx = GapDistContext(
        df_exec=df_exec,
        model=model,
        jp_gap=jp_gap,
        jp_beta=jp_beta,
        topix_night=topix_night,
        jp_res_returns_p3=jp_res_returns_p3,
        v0_static=v0_static,
        c_full_p3=c_full_p3,
        dist_in_dir=dist_in_dir,
        out_dir=out_dir,
        gap_store=gap_store,
        save_to_gap_store=save_to_gap_store,
        save_daily_m=save_daily_m,
        save_mh=save_mh,
        mh_horizons=mh_horizons,
        mh_models=mh_models,
        mh_inputs=mh_inputs,
        save_rr=save_rr,
        y_jp_target=y_jp_target,
        weights_df=weights_df,
        turnover_map=turnover_map,
        cfg=cfg,
        c=c,
        b=b,
        bl_mh_enabled=bl_mh_enabled,
        bl_mh_horizons=bl_mh_horizons,
        bl_mh_weights=bl_mh_weights,
        bl_mh_mu_pattern=bl_mh_mu_pattern,
        bl_mh_omega_pattern=bl_mh_omega_pattern,
        bl_cs_overlay_enabled=bl_cs_overlay_enabled,
        bl_cs_overlay_weight=bl_cs_overlay_weight,
        bl_rr_pattern=bl_rr_pattern,
        bl_long_count=bl_long_count,
        bl_short_count=bl_short_count,
        bl_minvar_enabled=bl_minvar_enabled,
        bl_minvar_alpha=bl_minvar_alpha,
        bl_baseline_gross=bl_baseline_gross,
        bl_cost_bps_per_gross=bl_cost_bps_per_gross,
        bundle_model_version=model_version(model),
        bundle_config_version=config_version(app_config.v2),
        bundle_ticker_order=tuple(JP_TICKERS),
        open_910_returns=open_910_returns,
        historical_inputs=historical_inputs,
        historical_inputs_by_horizon=historical_inputs_by_horizon,
    )

    def _process_date(dt):
        """Thin wrapper delegating to module-level _process_date_impl."""
        _process_date_impl(dt, ctx, acc)

    if n_jobs == 1 or len(sim_dates_slice) <= 1:
        for dt in sim_dates_slice:
            _process_date(dt)
    else:
        from joblib import Parallel, delayed

        partial_results = Parallel(n_jobs=n_jobs, backend="threading", verbose=10)(
            delayed(_process_date_worker)(dt, ctx) for dt in sim_dates_slice
        )
        for partial_acc in partial_results:
            _merge_accumulators(acc, partial_acc)

    logger.info("Reconstruction loops completed.")

    # Persist Tachibana-injected df_exec so downstream production steps
    # (e.g. ML overlay) see today's real 9:10 gaps instead of stale placeholders.
    if str_to_bool(args.use_tachibana_prices):
        try:
            save_df_exec_to_local_cache(df_exec)
            save_decision_cache(df_exec)
            logger.info("Persisted Tachibana-injected df_exec to local cache.")
        except Exception as e:
            logger.warning("Failed to persist Tachibana-injected df_exec: %s", e)

    diagnostic_frames = build_gap_diagnostic_frames(acc, tickers=JP_TICKERS)
    write_gap_diagnostic_frames(diagnostic_frames, out_dir)
    df_gap_long = diagnostic_frames.gap_long
    df_gap_daily = diagnostic_frames.gap_daily
    df_dist_long = diagnostic_frames.distribution_long
    df_dist_daily = diagnostic_frames.distribution_daily
    df_omega_daily = diagnostic_frames.omega_daily

    df_port = pd.DataFrame(acc.portfolio_diagnostics_records)
    df_port = prepare_portfolio_output_frame(df_port)

    # Merge Vol State panel if available
    vol_state_merged = False
    vol_cols = []
    if args.vol_state_panel:
        state_file = Path(args.vol_state_panel)
        if state_file.exists():
            logger.info(f"Merging US Vol State panel from {state_file}...")
            df_states = pd.read_csv(state_file)
            df_states["trade_date"] = pd.to_datetime(df_states["trade_date"], format="ISO8601").dt.strftime("%Y-%m-%d")

            # Select target vol states to merge
            target_cols = [
                "trade_date",
                "US_ret_dispersion_z_60",
                "US_absret_avg_z_60",
                "US_avg_corr_60",
                "US_pc1_share_60",
                "VIX_z_60",
                "VIX_level",
            ]
            # Match columns available
            avail_cols = [col for col in target_cols if col in df_states.columns]
            df_states_sub = df_states[avail_cols]

            df_port = df_port.merge(df_states_sub, on="trade_date", how="left")
            vol_cols = [c for c in avail_cols if c != "trade_date"]
            vol_state_merged = True
            logger.info(f"Vol state merged successfully. Columns: {vol_cols}")
        else:
            logger.warning(f"Vol state panel file not found: {state_file}")

    df_port.to_csv(out_dir / "portfolio_gap_distribution_diagnostics.csv", index=False)

    # --- Cleanup: parallel open fetch, session persistence, API client close ---
    # Do this early because the following return may short-circuit the rest of main().
    if opens_future is not None:
        try:
            opens_future.result(timeout=30.0)
        except Exception as e:
            logger.warning("Open-price fetch did not complete in time: %s", e)
    if opens_executor is not None:
        opens_executor.shutdown(wait=False, cancel_futures=True)

    if api_client is not None:
        release_fn = getattr(api_client, "release_session", None)
        try:
            if release_fn:
                release_fn()
            else:
                api_client.close()
        except Exception as e:
            logger.warning("Failed to release broker session: %s", e)

    if not compare_pre or len(df_port) < 10:
        logger.info("Skipping diagnostic comparison and plotting due to compare_pre_gap=False or insufficient data points.")
        logger.info("Report and diagnostic suite executed successfully (skipped plotting/reporting).")
        print(f"Diagnostics files written to output directory: {out_dir}")
        return

    render_gap_diagnostics(
        df_port=df_port,
        out_dir=out_dir,
        df_gap_long=df_gap_long,
        df_gap_daily=df_gap_daily,
        df_dist_long=df_dist_long,
        df_dist_daily=df_dist_daily,
        df_omega_daily=df_omega_daily,
        acc=acc,
        model=model,
        app_config=app_config,
        args=args,
        c=c,
        b=b,
        vol_state_merged=vol_state_merged,
        vol_cols=vol_cols,
        save_daily_m=save_daily_m,
        compare_pre=compare_pre,
        plots_dir=plots_dir,
    )

if __name__ == "__main__":
    main()
