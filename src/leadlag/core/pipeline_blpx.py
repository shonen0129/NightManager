"""BLPX-specific pipeline combiner and output adapter."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from leadlag.core.pipeline import (
    CommonInputs,
    ComponentResult,
    RunContext,
    StepContext,
)
from leadlag.domain.signal import SignalPackage

logger = logging.getLogger(__name__)
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
        self._raw_pca_signals = np.zeros((T, self._n_j))
        self._raw_blpx_signals = np.zeros((T, self._n_j))
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
    ) -> SignalPackage:
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
        return SignalPackage.from_dict(out)


