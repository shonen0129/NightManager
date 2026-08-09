"""Pipeline infrastructure for composition-based signal computation.

Step 2: Extracts CommonInputs dataclass and build_common_inputs pure function.
Step 3: Adds SignalComponent protocol, PCAComponent, and related infrastructure.

The existing model methods remain as thin delegates to preserve backward compatibility.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, cast, runtime_checkable

import numpy as np
import pandas as pd

from leadlag.core import signal as signals
from leadlag.core.correlation import (
    build_base_vectors,
    build_v3_static,
    compute_baseline_correlation,
)
from leadlag.core.residualize import compute_rolling_ols_betas
from leadlag.data.preprocessor import compute_us_residualized_returns
from leadlag.data.tickers import JP_TICKERS, US_TICKERS

logger = logging.getLogger(__name__)


@dataclass
class P4Inputs:
    """US residualized returns for P4 signal computation."""
    all_returns_p4: np.ndarray
    r_us_adj: np.ndarray
    spy_returns: np.ndarray


@dataclass
class CommonInputs:
    """Output of build_common_inputs — pure data, no I/O dependencies.

    All arrays are aligned with df_exec rows. JP columns are in JP_TICKERS order,
    US columns in US_TICKERS order.
    """
    all_returns_raw: np.ndarray
    c_full: np.ndarray
    c_full_p3: np.ndarray
    v0_static: np.ndarray
    v1: np.ndarray
    v2: np.ndarray
    jp_gap: np.ndarray
    jp_beta: np.ndarray | None
    topix_night: np.ndarray | None
    y_jp_oc_df: pd.DataFrame
    jp_res_returns_p3: np.ndarray
    y_jp_target: np.ndarray
    n_u: int
    n_j: int
    dates: pd.DatetimeIndex
    p4: P4Inputs | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict format compatible with existing _prepare_common_inputs callers."""
        out = {
            "all_returns_raw": self.all_returns_raw,
            "c_full": self.c_full,
            "c_full_p3": self.c_full_p3,
            "v0_static": self.v0_static,
            "v1": self.v1,
            "v2": self.v2,
            "jp_gap": self.jp_gap,
            "jp_beta": self.jp_beta,
            "topix_night": self.topix_night,
            "y_jp_oc_df": self.y_jp_oc_df,
            "jp_res_returns_p3": self.jp_res_returns_p3,
            "y_jp_target": self.y_jp_target,
        }
        if self.p4 is not None:
            out["all_returns_p4"] = self.p4.all_returns_p4
            out["r_us_adj"] = self.p4.r_us_adj
            out["spy_returns"] = self.p4.spy_returns
        return out


def _load_us_returns(df_exec: pd.DataFrame) -> np.ndarray:
    """Load raw US close-to-close returns from df_exec."""
    return np.asarray(
        df_exec[[f"us_cc_{tk}" for tk in US_TICKERS]].values, dtype=float
    )


def _apply_fractional_differencing(
    us_returns_raw: np.ndarray,
    df_exec: pd.DataFrame,
    *,
    frac_diff_enabled: bool,
    frac_diff_d: float,
    frac_diff_threshold: float,
    frac_diff_window: int,
    frac_diff_normalize: str | None,
) -> np.ndarray:
    """Apply optional fractional differencing to US returns."""
    if not (frac_diff_enabled and frac_diff_d > 0.0):
        return us_returns_raw

    # Lazy import: this module is used only when fractional differencing is on.
    from leadlag.features.fractional_diff import fractional_diff_df

    us_cols = [f"us_cc_{tk}" for tk in US_TICKERS]
    us_df = pd.DataFrame(us_returns_raw, columns=us_cols, index=df_exec.index)
    fd_df = fractional_diff_df(
        us_df,
        d=frac_diff_d,
        threshold=frac_diff_threshold,
        window=frac_diff_window,
        normalize=frac_diff_normalize,
    )
    nan_count = fd_df.isna().sum().sum()
    if nan_count > 0:
        logger.warning(
            "build_common_inputs: %d NaN values in fractional diff output "
            "will be filled with 0.0; check upstream US return data.",
            nan_count,
        )
        fd_df = fd_df.fillna(0.0)
    return np.asarray(fd_df.values, dtype=float)


def _build_all_returns_raw(
    us_returns_raw: np.ndarray,
    y_jp_target: np.ndarray,
) -> np.ndarray:
    """Stack US and JP target returns into the full returns matrix."""
    return np.column_stack([us_returns_raw, y_jp_target])


def _compute_baseline_correlation_matrix(
    returns: np.ndarray,
    sim_dates: pd.DatetimeIndex,
    ewma_half_life: int,
) -> np.ndarray:
    """Compute the baseline correlation matrix for a returns series."""
    return compute_baseline_correlation(returns, sim_dates.values, ewma_half_life)


def _build_static_vectors(
    n_u: int,
    n_j: int,
    include_v4_prior: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the static prior and base vectors."""
    v0_static = build_v3_static(n_u, n_j, include_v4_prior)
    base_vectors = build_base_vectors(n_u, n_j)
    return v0_static, base_vectors["v1"], base_vectors["v2"]


def _extract_jp_inputs(
    df_exec: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, pd.DataFrame]:
    """Extract JP gap, beta, overnight and open-to-close inputs."""
    jp_gap = df_exec[[f"jp_gap_{tk}" for tk in JP_TICKERS]].values
    jp_beta = (
        df_exec[[f"jp_beta_{tk}" for tk in JP_TICKERS]].values
        if any(c.startswith("jp_beta_") for c in df_exec.columns)
        else None
    )
    topix_night = (
        df_exec["topix_night_return"].values
        if "topix_night_return" in df_exec.columns
        else None
    )
    y_jp_oc_df = df_exec[[f"jp_oc_{tk}" for tk in JP_TICKERS]].rename(
        columns=lambda c: c.replace("jp_oc_", "")
    )
    return jp_gap, jp_beta, topix_night, y_jp_oc_df


def _select_topix_for_beta(df_exec: pd.DataFrame) -> np.ndarray:
    """Select the TOPIX benchmark used for rolling beta residualization."""
    if "topix_oc_return" in df_exec.columns:
        return np.asarray(df_exec["topix_oc_return"].values, dtype=float)
    if "topix_cc_trade" in df_exec.columns:
        return np.asarray(df_exec["topix_cc_trade"].values, dtype=float)
    return np.asarray(
        df_exec["topix_night_return"].values + df_exec["topix_oc_return"].values,
        dtype=float,
    )


def _compute_jp_topix_residual_returns(
    y_jp_target: np.ndarray,
    topix_for_beta: np.ndarray,
    beta_window: int,
    all_returns_raw: np.ndarray,
    n_u: int,
) -> np.ndarray:
    """Residualize JP target returns against TOPIX and build the P3 returns matrix."""
    betas_jp_p3 = compute_rolling_ols_betas(
        y_jp_target, topix_for_beta.reshape(-1, 1), beta_window
    )
    y_residuals_p3 = y_jp_target - betas_jp_p3[:, :, 0] * topix_for_beta.reshape(-1, 1)

    jp_res_returns_p3 = all_returns_raw.copy()
    jp_res_returns_p3[:, n_u:] = y_residuals_p3
    return jp_res_returns_p3


def _build_p4_inputs(
    all_returns_raw: np.ndarray,
    jp_res_returns_p3: np.ndarray,
    df_exec: pd.DataFrame,
    n_u: int,
    us_res_enabled: bool,
    us_res_gamma: float,
    us_res_beta_window: int,
) -> P4Inputs | None:
    """Build the P4 (US/SPY residualized) inputs when enabled."""
    if not us_res_enabled:
        return None

    spy_col = None
    for col in ["spy_cc", "SPY_cc", "SPY", "spy", "r_US_MKT"]:
        if col in df_exec.columns:
            spy_col = col
            break
    if spy_col is None:
        raise ValueError("SPY benchmark return column not found in df_exec")
    spy_returns = df_exec[spy_col].values

    us_returns = all_returns_raw[:, :n_u]
    r_us_adj = compute_us_residualized_returns(
        us_returns,
        spy_returns,
        beta_window=us_res_beta_window,
        gamma=us_res_gamma,
    )

    all_returns_p4 = jp_res_returns_p3.copy()
    all_returns_p4[:, :n_u] = r_us_adj

    return P4Inputs(
        all_returns_p4=all_returns_p4,
        r_us_adj=r_us_adj,
        spy_returns=spy_returns,
    )


def _assemble_common_inputs(
    all_returns_raw: np.ndarray,
    c_full: np.ndarray,
    c_full_p3: np.ndarray,
    v0_static: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    jp_gap: np.ndarray,
    jp_beta: np.ndarray | None,
    topix_night: np.ndarray | None,
    y_jp_oc_df: pd.DataFrame,
    jp_res_returns_p3: np.ndarray,
    y_jp_target: np.ndarray,
    n_u: int,
    n_j: int,
    dates: pd.DatetimeIndex,
    p4: P4Inputs | None,
) -> CommonInputs:
    """Assemble the final CommonInputs dataclass."""
    return CommonInputs(
        all_returns_raw=all_returns_raw,
        c_full=c_full,
        c_full_p3=c_full_p3,
        v0_static=v0_static,
        v1=v1,
        v2=v2,
        jp_gap=jp_gap,
        jp_beta=jp_beta,
        topix_night=topix_night,
        y_jp_oc_df=y_jp_oc_df,
        jp_res_returns_p3=jp_res_returns_p3,
        y_jp_target=y_jp_target,
        n_u=n_u,
        n_j=n_j,
        dates=dates,
        p4=p4,
    )


def build_common_inputs(
    df_exec: pd.DataFrame,
    y_jp_target: np.ndarray,
    *,
    n_u: int,
    n_j: int,
    ewma_half_life: int,
    beta_window: int,
    include_v4_prior: bool,
    us_res_enabled: bool = False,
    us_res_gamma: float = 0.5,
    us_res_beta_window: int = 60,
    frac_diff_enabled: bool = False,
    frac_diff_d: float = 0.5,
    frac_diff_threshold: float = 1e-5,
    frac_diff_window: int = 100,
    frac_diff_normalize: str | None = None,
) -> CommonInputs:
    """Build CommonInputs from df_exec and pre-computed y_jp_target.

    This is a pure computation function — no I/O (5-min cache, etc.).
    y_jp_target must be pre-computed by the caller (e.g. via compute_jp_target_returns).

    Args:
        df_exec: Execution DataFrame with US/JP columns.
        y_jp_target: Pre-computed 9:10-to-close JP target returns, shape (T, n_j).
        n_u: Number of US tickers.
        n_j: Number of JP tickers.
        ewma_half_life: EWMA half-life for baseline correlation.
        beta_window: Rolling OLS window for TOPIX residualization.
        include_v4_prior: Whether to include v4 prior in static vectors.
        us_res_enabled: Whether to compute US residualized (P4) inputs.
        us_res_gamma: Gamma for US residualization.
        us_res_beta_window: Beta window for US residualization.
        frac_diff_enabled: Whether to apply fractional differencing to US returns.
        frac_diff_d: Fractional differencing order (0 < d < 1).
        frac_diff_threshold: Weight cutoff for binomial expansion.
        frac_diff_window: Maximum lookback for fractional diff filter.
        frac_diff_normalize: Optional weight normalization for the truncated
            window (see fractional_diff.normalize).

    Returns:
        CommonInputs dataclass instance.
    """
    sim_dates = df_exec.index

    us_returns_raw = _load_us_returns(df_exec)
    us_returns_raw = _apply_fractional_differencing(
        us_returns_raw,
        df_exec,
        frac_diff_enabled=frac_diff_enabled,
        frac_diff_d=frac_diff_d,
        frac_diff_threshold=frac_diff_threshold,
        frac_diff_window=frac_diff_window,
        frac_diff_normalize=frac_diff_normalize,
    )

    all_returns_raw = _build_all_returns_raw(us_returns_raw, y_jp_target)
    c_full = _compute_baseline_correlation_matrix(
        all_returns_raw, sim_dates, ewma_half_life
    )

    v0_static, v1, v2 = _build_static_vectors(n_u, n_j, include_v4_prior)
    jp_gap, jp_beta, topix_night, y_jp_oc_df = _extract_jp_inputs(df_exec)
    topix_for_beta = _select_topix_for_beta(df_exec)

    jp_res_returns_p3 = _compute_jp_topix_residual_returns(
        y_jp_target, topix_for_beta, beta_window, all_returns_raw, n_u
    )
    c_full_p3 = _compute_baseline_correlation_matrix(
        jp_res_returns_p3, sim_dates, ewma_half_life
    )

    p4_inputs = _build_p4_inputs(
        all_returns_raw,
        jp_res_returns_p3,
        df_exec,
        n_u,
        us_res_enabled,
        us_res_gamma,
        us_res_beta_window,
    )

    return _assemble_common_inputs(
        all_returns_raw=all_returns_raw,
        c_full=c_full,
        c_full_p3=c_full_p3,
        v0_static=v0_static,
        v1=v1,
        v2=v2,
        jp_gap=jp_gap,
        jp_beta=jp_beta,
        topix_night=topix_night,
        y_jp_oc_df=y_jp_oc_df,
        jp_res_returns_p3=jp_res_returns_p3,
        y_jp_target=y_jp_target,
        n_u=n_u,
        n_j=n_j,
        dates=sim_dates,
        p4=p4_inputs,
    )
# ---------------------------------------------------------------------------
# Step 3: Component protocol and PCAComponent
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunContext:
    """Immutable context for a single predict_signals run."""
    dates: pd.DatetimeIndex
    n_u: int
    n_j: int
    start_idx: int
    start_idx_raw: int


@dataclass(frozen=True)
class StepContext:
    """Context for a single step (row i) within a run."""
    run: RunContext
    inputs: CommonInputs
    i: int


@dataclass
class ComponentResult:
    """Result of a single component computation."""
    signal: np.ndarray
    diagnostics: dict[str, Any] = field(default_factory=dict)
    covariance: np.ndarray | None = None


@runtime_checkable
class SignalComponent(Protocol):
    """Protocol for stateless or stateful signal components."""
    name: str

    def begin_run(self, context: RunContext) -> None: ...
    def compute(self, context: StepContext) -> ComponentResult: ...
    def end_run(self) -> dict[str, Any]: ...


@runtime_checkable
class EnsembleCombiner(Protocol):
    """Protocol for combining multiple component results into a final signal."""
    def begin_run(self, context: RunContext) -> None: ...
    def combine(
        self,
        context: StepContext,
        components: Mapping[str, ComponentResult],
    ) -> ComponentResult: ...
    def end_run(self) -> dict[str, Any]: ...


class PCAComponent:
    """Stateless PCA signal component (Raw-PCA or Residual-PCA).

    Wraps signals.compute_signal with fixed configuration parameters.
    Supports optional c0_override and k_override for P4/residual-prior variants.
    """

    def __init__(
        self,
        *,
        name: str,
        n_u: int,
        n_j: int,
        corr_window: int,
        k: int,
        lambda_reg: float,
        lambda_lw: float,
        lw_target: str,
        ewma_half_life: int,
        gap_open_coef: float,
        topix_beta_coef: float,
        vol_adjusted_target: bool,
        min_raw_weight: float = 0.0,
        k_override: int | None = None,
        use_c0_override: bool = False,
    ):
        self.name = name
        self._n_u = n_u
        self._n_j = n_j
        self._corr_window = corr_window
        self._k = k
        self._lambda_reg = lambda_reg
        self._lambda_lw = lambda_lw
        self._lw_target = lw_target
        self._ewma_half_life = ewma_half_life
        self._gap_open_coef = gap_open_coef
        self._topix_beta_coef = topix_beta_coef
        self._vol_adjusted_target = vol_adjusted_target
        self._min_raw_weight = min_raw_weight
        self._k_override = k_override
        self._use_c0_override = use_c0_override

    def begin_run(self, context: RunContext) -> None:
        pass

    def end_run(self) -> dict[str, Any]:
        return {}

    def compute(
        self,
        context: StepContext,
        *,
        all_returns: np.ndarray,
        c_full: np.ndarray,
        v0_static: np.ndarray,
        v1: np.ndarray,
        v2: np.ndarray,
        c0_override: np.ndarray | None = None,
    ) -> ComponentResult:
        """Compute PCA signal at step i using StepContext for gap/beta/topix_night.

        Args:
            context: Step context with run info and inputs.
            all_returns: Returns array (raw or residualized) to use.
            c_full: Correlation matrix to use.
            v0_static: Static prior vectors.
            v1, v2: Base vectors.
            c0_override: Optional C0 override for P4/residual-prior.

        Returns:
            ComponentResult with signal and empty diagnostics.
        """
        inputs = context.inputs
        return self._compute_core(
            i=context.i,
            all_returns=all_returns,
            c_full=c_full,
            v0_static=v0_static,
            v1=v1,
            v2=v2,
            jp_gap=inputs.jp_gap,
            jp_beta=inputs.jp_beta,
            topix_night=inputs.topix_night,
            c0_override=c0_override,
        )

    def compute_standalone(
        self,
        *,
        i: int,
        all_returns: np.ndarray,
        c_full: np.ndarray,
        v0_static: np.ndarray,
        v1: np.ndarray,
        v2: np.ndarray,
        jp_gap: np.ndarray | None,
        jp_beta: np.ndarray | None,
        topix_night: np.ndarray | None,
        c0_override: np.ndarray | None = None,
    ) -> ComponentResult:
        """Compute PCA signal without requiring a StepContext.

        Used by existing model methods during the strangler migration period.
        """
        return self._compute_core(
            i=i,
            all_returns=all_returns,
            c_full=c_full,
            v0_static=v0_static,
            v1=v1,
            v2=v2,
            jp_gap=jp_gap,
            jp_beta=jp_beta,
            topix_night=topix_night,
            c0_override=c0_override,
        )

    def _compute_core(
        self,
        *,
        i: int,
        all_returns: np.ndarray,
        c_full: np.ndarray,
        v0_static: np.ndarray,
        v1: np.ndarray,
        v2: np.ndarray,
        jp_gap: np.ndarray | None,
        jp_beta: np.ndarray | None,
        topix_night: np.ndarray | None,
        c0_override: np.ndarray | None = None,
    ) -> ComponentResult:
        """Core computation shared by compute() and compute_standalone()."""
        k_eff = self._k_override if self._k_override is not None else self._k

        gap_t1 = np.nan_to_num(jp_gap[i], nan=0.0) if jp_gap is not None else np.zeros(self._n_j)
        betas_t = np.asarray(jp_beta[i], dtype=float) if jp_beta is not None else None
        topix_night_t = float(topix_night[i]) if topix_night is not None else None

        sig_kwargs = dict(
            all_returns=all_returns,
            current_index=i,
            n_u=self._n_u,
            corr_window=self._corr_window,
            c_full=c_full,
            v0_static=v0_static,
            v1=v1,
            v2=v2,
            k=k_eff,
            lambda_reg=self._lambda_reg,
            lambda_lw=self._lambda_lw,
            lw_target=self._lw_target,
            ewma_half_life=self._ewma_half_life,
            v3_dynamic=False,
            gap_override=gap_t1,
            gap_open_coef=self._gap_open_coef,
            topix_beta_coef=self._topix_beta_coef,
            betas_t=betas_t,
            topix_night_t=topix_night_t,
            vol_adjusted_target=self._vol_adjusted_target,
            min_raw_weight=self._min_raw_weight,
        )
        if self._use_c0_override and c0_override is not None:
            sig_kwargs["c0_override"] = c0_override

        sig_res = signals.compute_signal(**cast(Any, sig_kwargs))
        sig = np.asarray(sig_res["signal"], dtype=float)
        return ComponentResult(signal=sig)


# ---------------------------------------------------------------------------
# Step 4: SignalPipeline
# ---------------------------------------------------------------------------

class SignalPipeline:
    """Runs components and combiner over a range of indices.

    The pipeline calls begin_run on all components and the combiner,
    loops from start_idx to T calling compute/combine, then calls end_run.
    Results are collected as arrays and returned for OutputAdapter formatting.
    """

    def __init__(
        self,
        components: list[SignalComponent],
        combiner: EnsembleCombiner,
    ):
        self._components = components
        self._combiner = combiner

    def run(
        self,
        inputs: CommonInputs,
        start_idx: int,
        T: int,
        start_idx_raw: int | None = None,
        n_jobs: int = 1,
    ) -> dict[str, Any]:
        """Run the pipeline and return dict of signal arrays.

        Args:
            inputs: CommonInputs dataclass with all required data.
            start_idx: Start index for the simulation loop.
            T: Total number of rows.
            start_idx_raw: Raw start index (defaults to start_idx).
            n_jobs: Number of parallel workers. 1 = sequential. -1 = all cores.
                Uses joblib threading backend (numpy releases GIL for heavy ops).

        Returns:
            Dict with keys: component names → (T, n_j) arrays,
            "combined" → (T, n_j) array,
            "normalized" → (T, n_j) array.
        """
        if start_idx_raw is None:
            start_idx_raw = start_idx
        run_ctx = RunContext(
            dates=inputs.dates,
            n_u=inputs.n_u,
            n_j=inputs.n_j,
            start_idx=start_idx,
            start_idx_raw=start_idx_raw,
        )

        for comp in self._components:
            comp.begin_run(run_ctx)
        self._combiner.begin_run(run_ctx)

        n_j = inputs.n_j
        results: dict[str, Any] = {}
        for comp in self._components:
            results[comp.name] = np.zeros((T, n_j))
        results["combined"] = np.zeros((T, n_j))
        results["normalized"] = np.zeros((T, n_j))
        diagnostics_list: list[dict] = []

        indices = list(range(start_idx, T))

        if n_jobs == 1 or len(indices) <= 1:
            step_results = self._run_sequential(indices, run_ctx, inputs)
        else:
            step_results = self._run_parallel(indices, run_ctx, inputs, n_jobs)

        for i, comp_signals, combined_signal, normalized_signal, step_diag in step_results:
            for comp in self._components:
                if comp.name in comp_signals:
                    results[comp.name][i] = comp_signals[comp.name]
            results["combined"][i] = combined_signal
            if normalized_signal is not None:
                results["normalized"][i] = normalized_signal
            if step_diag:
                diagnostics_list.append(step_diag)

        for comp in self._components:
            end = comp.end_run()
            if end:
                results.setdefault("_end_diagnostics", {}).setdefault(comp.name, end)
        combiner_end = self._combiner.end_run()
        if combiner_end:
            results.setdefault("_end_diagnostics", {})["_combiner"] = combiner_end

        if diagnostics_list:
            results["_step_diagnostics"] = diagnostics_list

        return results

    def _run_sequential(
        self,
        indices: list[int],
        run_ctx: RunContext,
        inputs: CommonInputs,
    ) -> list[tuple[int, dict[str, np.ndarray], np.ndarray, np.ndarray | None, dict]]:
        """Run the pipeline sequentially for the given indices."""
        out = []
        for i in indices:
            step_ctx = StepContext(run=run_ctx, inputs=inputs, i=i)
            comp_signals: dict[str, np.ndarray] = {}
            comp_results: dict[str, ComponentResult] = {}
            for comp in self._components:
                cr = comp.compute(step_ctx)
                comp_results[comp.name] = cr
                comp_signals[comp.name] = cr.signal
            combined = self._combiner.combine(step_ctx, comp_results)
            normalized = combined.diagnostics.get("normalized")
            step_diag = {k: v for k, v in combined.diagnostics.items() if k != "normalized"}
            out.append((i, comp_signals, combined.signal, normalized, step_diag))
        return out

    def _run_parallel(
        self,
        indices: list[int],
        run_ctx: RunContext,
        inputs: CommonInputs,
        n_jobs: int,
    ) -> list[tuple[int, dict[str, np.ndarray], np.ndarray, np.ndarray | None, dict]]:
        """Run the pipeline in parallel using joblib threading backend."""
        from joblib import Parallel, delayed

        def _compute_step(
            i: int,
        ) -> tuple[int, dict[str, np.ndarray], np.ndarray, np.ndarray | None, dict]:
            step_ctx = StepContext(run=run_ctx, inputs=inputs, i=i)
            comp_signals: dict[str, np.ndarray] = {}
            comp_results: dict[str, ComponentResult] = {}
            for comp in self._components:
                cr = comp.compute(step_ctx)
                comp_results[comp.name] = cr
                comp_signals[comp.name] = cr.signal
            combined = self._combiner.combine(step_ctx, comp_results)
            normalized = combined.diagnostics.get("normalized")
            step_diag = {k: v for k, v in combined.diagnostics.items() if k != "normalized"}
            return (i, comp_signals, combined.signal, normalized, step_diag)

        results = Parallel(n_jobs=n_jobs, backend="threading", verbose=10)(
            delayed(_compute_step)(i) for i in indices
        )
        return cast(list[tuple[int, dict[str, np.ndarray], np.ndarray, np.ndarray | None, dict]], results)



# ---------------------------------------------------------------------------
# Step 5: Generic component wrapper and BLPX
# ---------------------------------------------------------------------------

class CallableComponent:
    """Wraps a model method as a pipeline component.

    Transitional design for the strangler migration period.
    The compute_fn receives a StepContext and returns a dict with 'signal' key.
    """

    def __init__(self, name: str, compute_fn: Callable[[StepContext], dict[str, Any]]):
        self.name = name
        self._compute_fn = compute_fn

    def begin_run(self, context: RunContext) -> None:
        pass

    def end_run(self) -> dict[str, Any]:
        return {}

    def compute(self, context: StepContext) -> ComponentResult:
        result = self._compute_fn(context)
        signal = np.asarray(result["signal"], dtype=float)
        diagnostics = {k: v for k, v in result.items() if k != "signal"}
        return ComponentResult(signal=signal, diagnostics=diagnostics)


# ---------------------------------------------------------------------------
# Step 6: BLPX components, stateful combiner, and output adapter
# ---------------------------------------------------------------------------

class BLPXCombiner:
    """Stateful combiner for BLPX model.

    Handles static ensemble, meta-learning weight prediction, and macro confidence
    scaling. Tracks IC history, US dispersions, condition numbers, and VIX values
    across steps for meta-learning training.
    """

    def __init__(
        self,
        raw_pca_weight: float,
        residual_pca_weight: float,
        raw_blpx_weight: float,
        residual_blpx_weight: float,
        normalization_method: str,
        n_j: int,
        n_u: int,
        normalize_fn: Any,
        meta_enabled: bool,
        meta_train_window: int,
        meta_smooth_factor: float,
        corr_window: int,
        meta_predict_fn: Any | None = None,
        macro_confidence_enabled: bool = False,
        macro_scales: np.ndarray | None = None,
        macro_direction_adj: np.ndarray | None = None,
        vix_series: Any | None = None,
        y_jp_target: np.ndarray | None = None,
        all_returns_raw: np.ndarray | None = None,
    ):
        self._w0 = raw_pca_weight
        self._w3 = residual_pca_weight
        self._w_blpx = raw_blpx_weight
        self._w_blpx_p3 = residual_blpx_weight
        self._norm = normalization_method
        self._n_j = n_j
        self._n_u = n_u
        self._normalize_fn = normalize_fn
        self._meta_enabled = meta_enabled
        self._meta_train_window = meta_train_window
        self._meta_smooth_factor = meta_smooth_factor
        self._corr_window = corr_window
        self._meta_predict_fn = meta_predict_fn
        self._macro_confidence_enabled = macro_confidence_enabled
        self._macro_scales = macro_scales
        self._macro_direction_adj = macro_direction_adj
        self._vix_series = vix_series
        self._y_jp_target = y_jp_target
        self._all_returns_raw = all_returns_raw

        # State arrays (initialized in begin_run)
        self._us_dispersions: list[float] = []
        self._cond_nums: list[float] = []
        self._vix_vals: list[float] = []
        self._ic_blpx_vals: list[float] = []
        self._ic_pca_vals: list[float] = []
        self._meta_weights: list[float] = []
        self._raw_pca_signals: np.ndarray | None = None
        self._raw_blpx_signals: np.ndarray | None = None
        self._start_idx: int = 0

    def begin_run(self, context: RunContext) -> None:
        T = len(context.dates)
        self._us_dispersions = [0.0] * T
        self._cond_nums = [0.0] * T
        self._vix_vals = [20.0] * T
        self._ic_blpx_vals = [0.0] * T
        self._ic_pca_vals = [0.0] * T
        self._meta_weights = [0.8] * T
        self._start_idx = context.start_idx

    def end_run(self) -> dict[str, Any]:
        return {}

    def combine(
        self,
        context: StepContext,
        components: Mapping[str, ComponentResult],
    ) -> ComponentResult:
        from scipy.stats import spearmanr

        i = context.i
        inp = context.inputs

        raw_pca_sig = components["raw_pca"].signal
        residual_pca_sig = components["residual_pca"].signal
        raw_blpx_res = components["raw_blpx"]
        residual_blpx_res = components["residual_blpx"]

        z0 = self._normalize_fn(raw_pca_sig, self._norm)
        z3 = self._normalize_fn(residual_pca_sig, self._norm)
        z_raw_blpx = self._normalize_fn(raw_blpx_res.signal, self._norm)
        z_residual_blpx = self._normalize_fn(residual_blpx_res.signal, self._norm)

        # Track diagnostics
        self._us_dispersions[i] = float(np.nanvar(inp.all_returns_raw[i, :self._n_u]))
        self._cond_nums[i] = float(raw_blpx_res.diagnostics.get("cond_num", 0.0))
        self._vix_vals[i] = float(self._vix_series.iloc[i]) if self._vix_series is not None else 20.0

        # Store signals for IC calculation
        if self._raw_pca_signals is not None:
            self._raw_pca_signals[i] = raw_pca_sig
        if self._raw_blpx_signals is not None:
            self._raw_blpx_signals[i] = raw_blpx_res.signal

        # Calculate ICs for row i-1
        if i - 1 >= self._start_idx and self._y_jp_target is not None:
            y_prev = self._y_jp_target[i - 1]
            if self._raw_blpx_signals is not None:
                sig_blpx_prev = self._raw_blpx_signals[i - 1]
                valid_blpx = np.isfinite(sig_blpx_prev) & np.isfinite(y_prev)
                if np.sum(valid_blpx) >= 5 and np.std(sig_blpx_prev[valid_blpx]) > 1e-8 and np.std(y_prev[valid_blpx]) > 1e-8:
                    self._ic_blpx_vals[i - 1] = float(spearmanr(sig_blpx_prev[valid_blpx], y_prev[valid_blpx])[0])
                else:
                    self._ic_blpx_vals[i - 1] = 0.0

            if self._raw_pca_signals is not None:
                sig_pca_prev = self._raw_pca_signals[i - 1]
                valid_pca = np.isfinite(sig_pca_prev) & np.isfinite(y_prev)
                if np.sum(valid_pca) >= 5 and np.std(sig_pca_prev[valid_pca]) > 1e-8 and np.std(y_prev[valid_pca]) > 1e-8:
                    self._ic_pca_vals[i - 1] = float(spearmanr(sig_pca_prev[valid_pca], y_prev[valid_pca])[0])
                else:
                    self._ic_pca_vals[i - 1] = 0.0

        # Predict meta weight
        w_t = 0.8
        if self._meta_enabled:
            if i >= self._start_idx + self._meta_train_window and self._meta_predict_fn is not None:
                w_t = self._meta_predict_fn(
                    i, self._us_dispersions, self._cond_nums,
                    self._vix_vals, self._ic_blpx_vals, self._ic_pca_vals,
                )
                if self._meta_smooth_factor < 1.0 and i - 1 >= self._start_idx:
                    w_prev_meta = self._meta_weights[i - 1]
                    w_t = self._meta_smooth_factor * w_t + (1.0 - self._meta_smooth_factor) * w_prev_meta
            self._meta_weights[i] = w_t

        # Combine
        if self._meta_enabled:
            pca_denom = self._w0 + self._w3
            if pca_denom > 0.0:
                s_pca = (self._w0 * z0 + self._w3 * z3) / pca_denom
            else:
                s_pca = 0.5 * z0 + 0.5 * z3

            blpx_denom = self._w_blpx + self._w_blpx_p3
            if blpx_denom > 0.0:
                s_blpx = (self._w_blpx * z_raw_blpx + self._w_blpx_p3 * z_residual_blpx) / blpx_denom
            else:
                s_blpx = 0.5 * z_raw_blpx + 0.5 * z_residual_blpx

            s_ens = (1.0 - w_t) * s_pca + w_t * s_blpx
        else:
            s_ens = (
                self._w0 * z0 + self._w3 * z3
                + self._w_blpx * z_raw_blpx + self._w_blpx_p3 * z_residual_blpx
            )

        # Macro confidence scaling
        if self._macro_confidence_enabled and self._macro_scales is not None:
            scale_t = self._macro_scales[i]
            s_ens = s_ens / scale_t
            s_ens = np.nan_to_num(s_ens, nan=0.0, posinf=0.0, neginf=0.0)

            if self._macro_direction_adj is not None:
                dir_adj_t = self._macro_direction_adj[i]
                s_ens = s_ens * dir_adj_t
                s_ens = np.nan_to_num(s_ens, nan=0.0, posinf=0.0, neginf=0.0)

        s_norm = self._normalize_fn(s_ens, self._norm)

        date_str = context.run.dates[i].strftime("%Y-%m-%d")
        step_diag = {
            "date": date_str,
            "raw_blpx_cond_num": raw_blpx_res.diagnostics.get("cond_num", 0.0),
            "raw_blpx_b_norm": raw_blpx_res.diagnostics.get("b_norm", 0.0),
            "raw_blpx_b_pca_norm": raw_blpx_res.diagnostics.get("b_pca_norm", 0.0),
            "raw_blpx_b_sector_norm": raw_blpx_res.diagnostics.get("b_sector_norm", 0.0),
            "raw_blpx_b_struct_norm": raw_blpx_res.diagnostics.get("b_struct_norm", 0.0),
            "raw_blpx_sigma_xx_trace": raw_blpx_res.diagnostics.get("sigma_xx_trace", 0.0),
            "raw_blpx_sigma_yx_norm": raw_blpx_res.diagnostics.get("sigma_yx_norm", 0.0),
            "raw_blpx_sigma_yy_trace": raw_blpx_res.diagnostics.get("sigma_yy_trace", 0.0),
            "raw_blpx_min_pred_var": raw_blpx_res.diagnostics.get("min_pred_var", 0.0),
            "raw_blpx_max_pred_var": raw_blpx_res.diagnostics.get("max_pred_var", 0.0),
            "raw_blpx_num_pred_var_floored": raw_blpx_res.diagnostics.get("num_pred_var_floored", 0),
            "raw_blpx_pinv_fallback": int(raw_blpx_res.diagnostics.get("pinv_fallback", 0)),
            "raw_blpx_num_training_samples": raw_blpx_res.diagnostics.get("num_training_samples", 0),
            "meta_ensemble_weight": w_t,
        }

        return ComponentResult(signal=s_ens, diagnostics={"normalized": s_norm, **step_diag})


class BLPXOutputAdapter:
    """Converts pipeline output arrays to BLPX model's dict-of-DataFrames format."""

    def __init__(self, n_j: int, jp_tickers: list[str]):
        self._n_j = n_j
        self._jp_tickers = jp_tickers

    def adapt(
        self,
        pipeline_results: dict[str, np.ndarray],
        inputs: CommonInputs,
        sigma_yy: np.ndarray | None = None,
    ) -> dict[str, Any]:
        sim_dates = inputs.dates
        T = len(sim_dates)
        jp = self._jp_tickers

        raw_pca_df = pd.DataFrame(pipeline_results["raw_pca"], index=sim_dates, columns=jp)
        residual_pca_df = pd.DataFrame(pipeline_results["residual_pca"], index=sim_dates, columns=jp)
        p4_df = pd.DataFrame(np.zeros((T, self._n_j)), index=sim_dates, columns=jp)
        raw_blpx_df = pd.DataFrame(pipeline_results["raw_blpx"], index=sim_dates, columns=jp)
        residual_blpx_df = pd.DataFrame(pipeline_results["residual_blpx"], index=sim_dates, columns=jp)
        combined_df = pd.DataFrame(pipeline_results["combined"], index=sim_dates, columns=jp)
        normalized_df = pd.DataFrame(pipeline_results["normalized"], index=sim_dates, columns=jp)

        blp_diag_df = (
            pd.DataFrame(pipeline_results["_step_diagnostics"]).set_index("date")
            if "_step_diagnostics" in pipeline_results
            else pd.DataFrame()
        )

        out = {
            "raw_pca_signals": raw_pca_df,
            "residual_pca_signals": residual_pca_df,
            "p4_signals": p4_df,
            "raw_blpx_signals": raw_blpx_df,
            "residual_blpx_signals": residual_blpx_df,
            "signals": combined_df,
            "normalized_signals": normalized_df,
            "y_jp_oc_df": inputs.y_jp_oc_df,
            "blp_diagnostics": blp_diag_df,
        }
        if sigma_yy is not None:
            out["sigma_yy"] = sigma_yy
        return out


