"""Bayesian Online Update BLPX Model.

B_bayes_t = (1 - eta_t) * B_bayes_{t-1} + eta_t * B_struct_t

Three update modes:
  - 'ic':      Rolling Rank IC adaptive eta (original)
  - 'cs_var':  Cross-sectional prediction error variance adaptive eta
  - 'kalman':  Per-asset Kalman gain using prediction error variance vs B_struct change variance
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from leadlag.data.tickers import JP_TICKERS, US_TICKERS
from leadlag.models.sector_relative_ensemble_blp_enhanced import (
    SectorRelativeEnsembleBLPEnhancedModel,
)

logger = logging.getLogger(__name__)


class BayesianBLPXModel(SectorRelativeEnsembleBLPEnhancedModel):
    """BLPX with Bayesian online update on prediction matrix B_struct."""

    def __init__(self, config: dict | object):
        super().__init__(config)
        self.bayesian_enabled = bool(self._resolve_val("bayesian_enabled", True))
        self.bayesian_mode = str(self._resolve_val("bayesian_mode", "ic"))
        self.bayesian_eta_base = float(self._resolve_val("bayesian_eta_base", 0.3))
        self.bayesian_ic_window = int(self._resolve_val("bayesian_ic_window", 63))
        self.bayesian_ic_amplifier = float(self._resolve_val("bayesian_ic_amplifier", 5.0))
        self.bayesian_eta_min = float(self._resolve_val("bayesian_eta_min", 0.05))
        self.bayesian_eta_max = float(self._resolve_val("bayesian_eta_max", 0.80))
        self.bayesian_warmup = int(self._resolve_val("bayesian_warmup", 21))
        # cs_var mode params
        self.bayesian_cs_var_window = int(self._resolve_val("bayesian_cs_var_window", 63))
        self.bayesian_cs_var_scale = float(self._resolve_val("bayesian_cs_var_scale", 1.0))
        # kalman mode params
        self.bayesian_kalman_window = int(self._resolve_val("bayesian_kalman_window", 63))
        self.bayesian_kalman_q_scale = float(self._resolve_val("bayesian_kalman_q_scale", 1.0))
        self.bayesian_kalman_r_floor = float(self._resolve_val("bayesian_kalman_r_floor", 1e-6))
        # State
        self._B_bayes_prev: np.ndarray | None = None
        self._ic_history: list[float] = []
        self._eta_history: list[float] = []
        self._prev_predicted_z = np.zeros(self.n_j)
        # cs_var state
        self._cs_var_history: deque = deque(maxlen=self.bayesian_cs_var_window)
        # kalman state
        self._error_history: deque = deque(maxlen=self.bayesian_kalman_window)
        self._B_struct_history: deque = deque(maxlen=self.bayesian_kalman_window + 1)

    def _compute_adaptive_eta_ic(self) -> float:
        """IC-based adaptive eta (original mode)."""
        if len(self._ic_history) < self.bayesian_warmup:
            return self.bayesian_eta_base
        w = min(self.bayesian_ic_window, len(self._ic_history))
        rolling_ic = float(np.mean(self._ic_history[-w:]))
        eta = self.bayesian_eta_base + self.bayesian_ic_amplifier * max(0.0, rolling_ic)
        return float(np.clip(eta, self.bayesian_eta_min, self.bayesian_eta_max))

    def _compute_adaptive_eta_cs_var(self, current_error: np.ndarray | None) -> float:
        """Cross-sectional variance based adaptive eta.

        When cross-sectional error variance is low relative to baseline,
        the model is predicting consistently → increase eta (trust new data).
        When high → decrease eta (rely on prior).
        """
        if current_error is None:
            return self.bayesian_eta_base

        valid = ~np.isnan(current_error)
        if valid.sum() < 3:
            return self.bayesian_eta_base

        e = current_error[valid]
        e_centered = e - np.mean(e)
        cs_var_t = float(np.var(e_centered, ddof=1))
        self._cs_var_history.append(cs_var_t)

        if len(self._cs_var_history) < self.bayesian_warmup:
            return self.bayesian_eta_base

        baseline_var = float(np.mean(list(self._cs_var_history)))
        if baseline_var < 1e-12:
            return self.bayesian_eta_base

        ratio = baseline_var / (cs_var_t + 1e-12)
        eta = self.bayesian_eta_base * (ratio ** self.bayesian_cs_var_scale)
        return float(np.clip(eta, self.bayesian_eta_min, self.bayesian_eta_max))

    def _compute_kalman_gain(self, B_struct: np.ndarray) -> np.ndarray:
        """Per-asset Kalman gain for B_struct update.

        K_j = Q_j / (Q_j + R_j)
        where Q_j = variance of B_struct changes (process noise)
              R_j = variance of prediction errors (observation noise)

        Q estimation: uses non-overlapping subsamples of B_struct changes to avoid
        autocorrelation bias from overlapping rolling window estimates.
        R estimation: bias-corrects total prediction error variance by subtracting
        the Q contribution (Var(ΔB̂) ≈ Q + 2R/window → R = (Var(ΔB̂) - Q) * window / 2).

        Returns vector of shape (n_j,) — one gain per JP asset.
        """
        n_j = self.n_j

        if len(self._error_history) < self.bayesian_warmup:
            return np.full(n_j, self.bayesian_eta_base)

        if len(self._B_struct_history) < 3:
            return np.full(n_j, self.bayesian_eta_base)

        B_arr = np.array(list(self._B_struct_history))  # (T_hist, n_j, n_u)
        blp_window = getattr(self, "blp_window", 252)

        # --- Process noise Q: non-overlapping subsampling ---
        # Use every blp_window-th sample to avoid autocorrelation from overlapping rolling estimates
        step = max(blp_window, 2)
        B_sub = B_arr[::step]
        if len(B_sub) < 3:
            # Fallback: use all but with larger stride
            step = max(len(B_arr) // 3, 2)
            B_sub = B_arr[::step]
        if len(B_sub) < 2:
            return np.full(n_j, self.bayesian_eta_base)

        delta_B_nonoverlap = np.diff(B_sub, axis=0)  # (n_sub-1, n_j, n_u)
        Q = np.var(delta_B_nonoverlap, axis=0, ddof=1) if len(delta_B_nonoverlap) >= 2 else np.zeros((n_j, B_arr.shape[2]))
        Q_per_asset = np.mean(Q, axis=1) * self.bayesian_kalman_q_scale  # (n_j,)
        Q_per_asset = np.maximum(Q_per_asset, 1e-12)

        # --- Observation noise R: bias-corrected ---
        # Total prediction error variance = R_true + model_residual_var
        # Model residual is irreducible; we estimate R_true from B_struct changes:
        #   Var(ΔB̂_consecutive) ≈ Q + 2*R/blp_window
        # So R ≈ (Var(ΔB̂_consecutive) - Q) * blp_window / 2
        delta_B_all = np.diff(B_arr, axis=0)  # (T-1, n_j, n_u) — consecutive (overlapping)
        var_delta_all = np.var(delta_B_all, axis=0, ddof=1) if len(delta_B_all) >= 2 else np.zeros_like(Q)
        var_delta_per_asset = np.mean(var_delta_all, axis=1)  # (n_j,)
        R_theoretical = np.maximum((var_delta_per_asset - Q_per_asset) * blp_window / 2.0, self.bayesian_kalman_r_floor)

        # Use theoretical R (not raw error variance) for Kalman gain
        R = R_theoretical

        K = Q_per_asset / (Q_per_asset + R)  # (n_j,)
        return np.clip(K, self.bayesian_eta_min, self.bayesian_eta_max)

    @staticmethod
    def _compute_rank_ic(predicted: np.ndarray, actual: np.ndarray) -> float:
        valid = ~(np.isnan(predicted) | np.isnan(actual))
        if valid.sum() < 3:
            return 0.0
        rho, _ = stats.spearmanr(predicted[valid], actual[valid])
        return float(rho) if np.isfinite(rho) else 0.0

    def compute_blp_signal_bayesian(
        self, all_returns: np.ndarray, current_index: int,
        gap_override: np.ndarray | None = None,
        betas_t: np.ndarray | None = None,
        topix_night_t: float | None = None,
        rolling_std: np.ndarray | None = None,
        v0_static: np.ndarray | None = None,
        c_full: np.ndarray | None = None,
        is_residual: bool = False,
        y_actual: np.ndarray | None = None,
    ) -> dict[str, Any]:
        result = self.compute_blp_signal(
            all_returns=all_returns, current_index=current_index,
            gap_override=gap_override, betas_t=betas_t,
            topix_night_t=topix_night_t, rolling_std=rolling_std,
            v0_static=v0_static, c_full=c_full,
            is_residual=is_residual, return_matrices=True,
        )
        if not self.bayesian_enabled:
            return result

        B_struct = result.get("B_struct")
        if B_struct is None:
            return result

        # Track prediction error and IC from previous step
        current_error = None
        if y_actual is not None and self._B_bayes_prev is not None:
            current_error = y_actual - self._prev_predicted_z
            ic_val = self._compute_rank_ic(self._prev_predicted_z, y_actual)
            self._ic_history.append(ic_val)
            self._error_history.append(current_error.copy())

        # Track B_struct history for Kalman process noise estimation
        self._B_struct_history.append(B_struct.copy())

        # Compute adaptive eta / Kalman gain based on mode
        if self.bayesian_mode == "cs_var":
            eta_t = self._compute_adaptive_eta_cs_var(current_error)
            self._eta_history.append(eta_t)
        elif self.bayesian_mode == "kalman":
            eta_vec = self._compute_kalman_gain(B_struct)
            eta_t = float(np.mean(eta_vec))  # scalar for diagnostics
            self._eta_history.append(eta_t)
        else:  # "ic"
            eta_t = self._compute_adaptive_eta_ic()
            self._eta_history.append(eta_t)

        # Bayesian update
        if self._B_bayes_prev is not None and self._B_bayes_prev.shape == B_struct.shape:
            if self.bayesian_mode == "kalman":
                # Per-asset Kalman gain: each row of B updated independently
                eta_col = eta_vec.reshape(-1, 1)  # (n_j, 1)
                B_bayes = (1.0 - eta_col) * self._B_bayes_prev + eta_col * B_struct
            else:
                B_bayes = (1.0 - eta_t) * self._B_bayes_prev + eta_t * B_struct
        else:
            B_bayes = B_struct.copy()

        # Recompute signal with B_bayes
        z_U_t = result.get("z_U")
        if z_U_t is not None:
            z_hat_j_t1 = B_bayes @ z_U_t
            z_hat_j_t1 = np.nan_to_num(z_hat_j_t1, nan=0.0, posinf=0.0, neginf=0.0)

            Sigma_YY_reg = result.get("Sigma_YY")
            Sigma_YX_reg = result.get("Sigma_YX")
            inv_A_tikh = result.get("inv_A")
            if Sigma_YY_reg is not None and Sigma_YX_reg is not None and inv_A_tikh is not None:
                z_hat_j_t1, pred_var, num_floored = self._apply_confidence_weighting(
                    z_hat_j_t1, Sigma_YY_reg, Sigma_YX_reg, inv_A_tikh, self.beta_conf)
            else:
                pred_var = np.ones(self.n_j)

            mu = result.get("mu_Y", np.zeros(self.n_j))
            mu_X = result.get("mu_X", np.zeros(self.n_u))
            sigma = result.get("sigma_Y", np.ones(self.n_j))
            sigma_X = result.get("sigma_X", np.ones(self.n_u))
            full_mu = np.concatenate([mu_X, mu])
            full_sigma = np.concatenate([sigma_X, sigma])

            r_hat_jp_cc = self._denormalize_signal(
                z_hat_j_t1, full_mu, full_sigma,
                all_returns, current_index, self.n_u, self.vol_adjusted_target)
            signal = self._apply_gap_adjustment(
                r_hat_jp_cc, z_hat_j_t1, gap_override, betas_t, topix_night_t)

            result["signal"] = signal
            result["z_hat_j_t1"] = z_hat_j_t1
            result["B_bayes"] = B_bayes
            result["eta_t"] = eta_t
            self._prev_predicted_z = z_hat_j_t1.copy()
        else:
            self._prev_predicted_z = result.get("z_hat_j_t1", np.zeros(self.n_j)).copy()

        self._B_bayes_prev = B_bayes.copy()
        return result

    def predict_signals(self, df_exec: pd.DataFrame) -> dict[str, Any]:
        if not self.bayesian_enabled:
            return super().predict_signals(df_exec)

        from leadlag.core.pipeline import (
            BayesianCombiner,
            BayesianOutputAdapter,
            CallableComponent,
            CommonInputs,
            SignalPipeline,
        )

        self._B_bayes_prev = None
        self._ic_history = []
        self._eta_history = []
        self._prev_predicted_z = np.zeros(self.n_j)
        self._cs_var_history = deque(maxlen=self.bayesian_cs_var_window)
        self._error_history = deque(maxlen=self.bayesian_kalman_window)
        self._B_struct_history = deque(maxlen=self.bayesian_kalman_window + 1)

        T = len(df_exec)
        sim_dates = df_exec.index

        inputs = self._prepare_common_inputs(df_exec)
        jp_res_returns_p3 = inputs["jp_res_returns_p3"]
        c_full_p3 = inputs["c_full_p3"]
        v0_static = inputs["v0_static"]
        v1 = inputs["v1"]
        v2 = inputs["v2"]
        jp_gap = inputs["jp_gap"]
        jp_beta = inputs["jp_beta"]
        topix_night = inputs["topix_night"]
        y_jp_target = inputs["y_jp_target"]

        rolling_std = None
        if self.exec_adjustment == "vol_scale":
            df_y = pd.DataFrame(y_jp_target)
            rolling_std = df_y.rolling(20).std(ddof=1).values
            overall_std = np.maximum(np.std(y_jp_target, axis=0, ddof=1), 1e-8)
            for c in range(self.n_j):
                nan_mask = np.isnan(rolling_std[:, c])
                rolling_std[nan_mask, c] = overall_std[c]
            rolling_std = np.maximum(rolling_std, 1e-8)

        common_inputs = CommonInputs(
            all_returns_raw=inputs["all_returns_raw"],
            c_full=inputs["c_full"],
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

        start_idx = self.corr_window
        sigma_yy_array = np.zeros((T, self.n_j, self.n_j))

        def _bayesian_fn(ctx):
            i = ctx.i
            inp = ctx.inputs
            gap_override = np.nan_to_num(inp.jp_gap[i], nan=0.0) if inp.jp_gap is not None else None
            betas_t = np.asarray(inp.jp_beta[i], dtype=float) if inp.jp_beta is not None else None
            topix_night_t = float(inp.topix_night[i]) if inp.topix_night is not None else None
            y_actual_prev = inp.y_jp_target[i - 1] if i > start_idx + 1 else None

            result = self.compute_blp_signal_bayesian(
                all_returns=inp.jp_res_returns_p3, current_index=i,
                gap_override=gap_override, betas_t=betas_t,
                topix_night_t=topix_night_t, rolling_std=rolling_std,
                v0_static=inp.v0_static, c_full=inp.c_full_p3,
                is_residual=True, y_actual=y_actual_prev,
            )

            if self.minvar_enabled and "sigma_Y_cov" in result:
                sigma_yy_array[i] = result["sigma_Y_cov"]

            result["ic"] = self._ic_history[-1] if self._ic_history else 0.0
            result["rolling_ic"] = float(np.mean(self._ic_history[-self.bayesian_ic_window:])) if self._ic_history else 0.0
            result["cs_var"] = float(self._cs_var_history[-1]) if self._cs_var_history else 0.0
            result["mode"] = self.bayesian_mode
            return result

        components = [
            CallableComponent("residual_blpx_bayesian", _bayesian_fn),
        ]

        combiner = BayesianCombiner(
            normalization_method=self.normalization_method,
            n_j=self.n_j,
            normalize_fn=self.normalize_signals,
        )

        pipeline = SignalPipeline(components=components, combiner=combiner)
        pipeline_results = pipeline.run(common_inputs, start_idx=start_idx, T=T)

        adapter = BayesianOutputAdapter(n_j=self.n_j, jp_tickers=JP_TICKERS)
        return adapter.adapt(pipeline_results, common_inputs, sigma_yy=sigma_yy_array)
