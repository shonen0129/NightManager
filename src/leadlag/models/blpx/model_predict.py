"""BLPX signal prediction orchestration helpers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from leadlag.core.pipeline import (
    CallableComponent,
    CommonInputs,
    SignalPipeline,
)
from leadlag.core.pipeline_blpx import (
    BLPXCombiner,
    BLPXOutputAdapter,
)
from leadlag.data.tickers import JP_TICKERS
from leadlag.domain.signal import SignalPackage

if TYPE_CHECKING:
    from leadlag.models.blpx.model import ProductionBLPXModel

logger = logging.getLogger("leadlag.models.blpx")


def predict_signals(self: ProductionBLPXModel, df_exec: pd.DataFrame, n_jobs: int = 1) -> SignalPackage:
    """Generate component and ensemble signals for all rows in df_exec."""

    # Clear per-run signal caches to prevent cross-run contamination
    self._raw_pca_cache.clear()
    self._residual_pca_cache.clear()
    self._blp_corr_cache.clear()

    T = len(df_exec)
    sim_dates = df_exec.index

    inputs = self._prepare_common_inputs(df_exec)
    all_returns_raw = inputs["all_returns_raw"]
    c_full = inputs["c_full"]
    c_full_p3 = inputs["c_full_p3"]
    v0_static = inputs["v0_static"]
    v1 = inputs["v1"]
    v2 = inputs["v2"]
    jp_gap = inputs["jp_gap"]
    jp_beta = inputs["jp_beta"]
    topix_night = inputs["topix_night"]
    jp_res_returns_p3 = inputs["jp_res_returns_p3"]
    y_jp_target = inputs["y_jp_target"]

    # Precompute target standard deviations if target cost adjustment is vol_scale
    rolling_std = None
    if self.exec_adjustment == "vol_scale":
        df_y = pd.DataFrame(y_jp_target)
        rolling_std = df_y.rolling(20).std(ddof=1).values
        overall_std = np.std(y_jp_target, axis=0, ddof=1)
        overall_std = np.maximum(overall_std, 1e-8)
        for col_idx in range(self.n_j):
            nan_mask = np.isnan(rolling_std[:, col_idx])
            rolling_std[nan_mask, col_idx] = overall_std[col_idx]
        rolling_std = np.maximum(rolling_std, 1e-8)

    # Precompute macro confidence scales if enabled
    self._prepare_macro_confidence(df_exec)

    # Load VIX if meta-learning is enabled
    vix_series = self._load_vix_series(df_exec)

    # Optimize: skip loop iterations before warmup if _start_date is specified
    start_date_str = getattr(self, "_start_date", None)
    if start_date_str is not None:
        start_dt = pd.to_datetime(start_date_str)
        start_idx_raw = df_exec.index.searchsorted(start_dt)
        start_idx = max(self.corr_window, start_idx_raw - self.blp_window)
    else:
        start_idx = self.corr_window
        start_idx_raw = self.corr_window

    # Determine which components to compute (skip zero-weight for speed)
    need_raw_pca = (self.raw_pca_weight > 0.0) or self.meta_enabled
    need_residual_pca = self.residual_pca_weight > 0.0
    need_raw_blpx = (self.raw_blpx_weight > 0.0) or self.meta_enabled
    need_residual_blpx = self.residual_blpx_weight > 0.0

    cache_key = (
        len(df_exec),
        df_exec.index[0],
        df_exec.index[-1],
        self.corr_window,
        self.k,
        self.lambda_reg,
        self.ewma_half_life,
        self.lambda_lw,
        self.lw_target,
        self.gap_open_coef,
        self.topix_beta_coef,
        self.vol_adjusted_target,
    )

    # PCA caching
    raw_pca_cached = False
    raw_pca_cache_arr = None
    if need_raw_pca:
        if cache_key in self._raw_pca_cache:
            raw_pca_cache_arr = self._raw_pca_cache[cache_key]
            raw_pca_cached = True
    residual_pca_cached = False
    residual_pca_cache_arr = None
    if need_residual_pca:
        if cache_key in self._residual_pca_cache:
            residual_pca_cache_arr = self._residual_pca_cache[cache_key]
            residual_pca_cached = True

    # Build CommonInputs
    common_inputs = CommonInputs(
        all_returns_raw=all_returns_raw,
        c_full=c_full,
        c_full_p3=c_full_p3,
        v0_static=v0_static,
        v1=v1,
        v2=v2,
        jp_gap=jp_gap,
        jp_beta=jp_beta,
        topix_night=topix_night,
        y_jp_oc_df=inputs["y_jp_oc_df"],
        jp_res_returns_p3=jp_res_returns_p3,
        y_jp_target=y_jp_target,
        n_u=self.n_u,
        n_j=self.n_j,
        dates=sim_dates,
        p4=None,
    )

    # Build component closures
    def _raw_pca_fn(ctx: Any) -> dict:
        i = ctx.i
        if not need_raw_pca:
            return {"signal": np.zeros(self.n_j)}
        if raw_pca_cached and raw_pca_cache_arr is not None:
            return {"signal": raw_pca_cache_arr[i]}
        inp = ctx.inputs
        sig = self.compute_production_signal(
            i,
            inp.c_full,
            inp.v0_static,
            inp.v1,
            inp.v2,
            inp.all_returns_raw,
            inp.jp_gap,
            inp.jp_beta,
            inp.topix_night,
        )
        return {"signal": sig}

    def _residual_pca_fn(ctx: Any) -> dict:
        i = ctx.i
        if not need_residual_pca:
            return {"signal": np.zeros(self.n_j)}
        if residual_pca_cached and residual_pca_cache_arr is not None:
            return {"signal": residual_pca_cache_arr[i]}
        inp = ctx.inputs
        sig = self.compute_residual_signal(
            inp.jp_res_returns_p3,
            i,
            inp.c_full_p3,
            inp.v0_static,
            inp.v1,
            inp.v2,
            inp.jp_gap,
            inp.jp_beta,
            inp.topix_night,
        )
        return {"signal": sig}

    def _raw_blpx_fn(ctx: Any) -> dict:
        i = ctx.i
        if not need_raw_blpx or i < start_idx_raw:
            return {**self._ZERO_BLP_DIAGNOSTICS, "signal": np.zeros(self.n_j)}
        inp = ctx.inputs
        gap_override = np.nan_to_num(inp.jp_gap[i], nan=0.0) if inp.jp_gap is not None else None
        betas_t = np.asarray(inp.jp_beta[i], dtype=float) if inp.jp_beta is not None else None
        topix_night_t = float(inp.topix_night[i]) if inp.topix_night is not None else None
        return self.compute_blp_signal(
            inp.all_returns_raw,
            i,
            gap_override=gap_override,
            betas_t=betas_t,
            topix_night_t=topix_night_t,
            rolling_std=rolling_std,
            v0_static=inp.v0_static,
            c_full=inp.c_full,
            is_residual=False,
        )

    def _residual_blpx_fn(ctx: Any) -> dict:
        i = ctx.i
        if not need_residual_blpx or i < start_idx_raw:
            return {**self._ZERO_BLP_DIAGNOSTICS, "signal": np.zeros(self.n_j)}
        inp = ctx.inputs
        gap_override = np.nan_to_num(inp.jp_gap[i], nan=0.0) if inp.jp_gap is not None else None
        betas_t = np.asarray(inp.jp_beta[i], dtype=float) if inp.jp_beta is not None else None
        topix_night_t = float(inp.topix_night[i]) if inp.topix_night is not None else None
        result = self.compute_blp_signal(
            inp.jp_res_returns_p3,
            i,
            gap_override=gap_override,
            betas_t=betas_t,
            topix_night_t=topix_night_t,
            rolling_std=rolling_std,
            v0_static=inp.v0_static,
            c_full=inp.c_full_p3,
            is_residual=True,
        )
        if self.minvar_enabled and "sigma_Y_cov" in result:
            sigma_yy_array[i] = result["sigma_Y_cov"]
        return result

    components = [
        CallableComponent("raw_pca", _raw_pca_fn),
        CallableComponent("residual_pca", _residual_pca_fn),
        CallableComponent("raw_blpx", _raw_blpx_fn),
        CallableComponent("residual_blpx", _residual_blpx_fn),
    ]

    # Prepare signal arrays for IC tracking
    raw_pca_signals_arr = np.zeros((T, self.n_j))
    raw_blpx_signals_arr = np.zeros((T, self.n_j))
    sigma_yy_array = np.zeros((T, self.n_j, self.n_j))

    combiner = BLPXCombiner(
        raw_pca_weight=self.raw_pca_weight,
        residual_pca_weight=self.residual_pca_weight,
        raw_blpx_weight=self.raw_blpx_weight,
        residual_blpx_weight=self.residual_blpx_weight,
        normalization_method=self.normalization_method,
        n_j=self.n_j,
        n_u=self.n_u,
        normalize_fn=self.normalize_signals,
        meta_enabled=self.meta_enabled,
        meta_train_window=self.meta_train_window,
        meta_smooth_factor=self.meta_smooth_factor,
        corr_window=self.corr_window,
        meta_predict_fn=self._predict_meta_weight if self.meta_enabled else None,
        macro_confidence_enabled=self.macro_confidence_enabled,
        macro_scales=self._macro_scales,
        macro_direction_adj=self._macro_direction_adj,
        vix_series=vix_series,
        y_jp_target=y_jp_target,
        all_returns_raw=all_returns_raw,
    )
    combiner._raw_pca_signals = raw_pca_signals_arr
    combiner._raw_blpx_signals = raw_blpx_signals_arr

    pipeline = SignalPipeline(components=components, combiner=combiner)
    pipeline_results = pipeline.run(
        common_inputs,
        start_idx=start_idx,
        T=T,
        start_idx_raw=start_idx_raw,
        n_jobs=n_jobs,
    )

    # Update PCA caches
    if need_raw_pca and not raw_pca_cached:
        self._raw_pca_cache[cache_key] = pipeline_results["raw_pca"].copy()
    if need_residual_pca and not residual_pca_cached:
        self._residual_pca_cache[cache_key] = pipeline_results["residual_pca"].copy()

    # Extension E: Inflate Sigma_YY based on macro surprise
    if (
        self.macro_confidence_enabled
        and self.macro_sigma_yy_inflation_enabled
        and self._macro_surprise_raw is not None
        and self.macro_kappas is not None
        and np.any(sigma_yy_array)
    ):
        from leadlag.core.macro import compute_sigma_yy_inflation

        sigma_yy_array = compute_sigma_yy_inflation(
            self._macro_surprise_raw,
            cast(np.ndarray, self.macro_kappas),
            self._macro_sens_matrix,
            sigma_yy_base=sigma_yy_array,
        )

    adapter = BLPXOutputAdapter(n_j=self.n_j, jp_tickers=JP_TICKERS)
    return adapter.adapt(pipeline_results, common_inputs, sigma_yy=sigma_yy_array)


class BLPXPredictMixin:
    """Mixin providing the full predict_signals orchestration."""

    predict_signals = predict_signals
