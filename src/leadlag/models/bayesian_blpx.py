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

        Returns vector of shape (n_j,) — one gain per JP asset.
        """
        n_j = self.n_j

        # Observation noise: per-asset rolling prediction error variance
        if len(self._error_history) < self.bayesian_warmup:
            return np.full(n_j, self.bayesian_eta_base)

        errors = np.array(list(self._error_history))  # (T, n_j)
        R = np.var(errors, axis=0, ddof=1)  # (n_j,)
        R = np.maximum(R, self.bayesian_kalman_r_floor)

        # Process noise: per-asset variance of B_struct row changes
        if len(self._B_struct_history) < 3:
            return np.full(n_j, self.bayesian_eta_base)

        B_arr = np.array(list(self._B_struct_history))  # (T, n_j, n_u)
        delta_B = np.diff(B_arr, axis=0)  # (T-1, n_j, n_u)
        Q = np.var(delta_B, axis=0)  # (n_j, n_u) — variance across time of each element
        Q_per_asset = np.mean(Q, axis=1) * self.bayesian_kalman_q_scale  # (n_j,) — average over US factors
        Q_per_asset = np.maximum(Q_per_asset, 1e-12)

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

        residual_blpx_signals = np.zeros((T, self.n_j))
        combined_signals = np.zeros((T, self.n_j))
        normalized_combined_signals = np.zeros((T, self.n_j))
        sigma_yy_array = np.zeros((T, self.n_j, self.n_j))
        eta_records = []

        start_idx = self.corr_window
        for i in range(start_idx, T):
            gap_override = np.nan_to_num(jp_gap[i], nan=0.0) if jp_gap is not None else None
            betas_t = np.asarray(jp_beta[i], dtype=float) if jp_beta is not None else None
            topix_night_t = float(topix_night[i]) if topix_night is not None else None
            y_actual_prev = y_jp_target[i] if i > start_idx else None

            result = self.compute_blp_signal_bayesian(
                all_returns=jp_res_returns_p3, current_index=i,
                gap_override=gap_override, betas_t=betas_t,
                topix_night_t=topix_night_t, rolling_std=rolling_std,
                v0_static=v0_static, c_full=c_full_p3,
                is_residual=True, y_actual=y_actual_prev,
            )
            residual_blpx_signals[i] = result["signal"]
            if self.minvar_enabled and "sigma_Y_cov" in result:
                sigma_yy_array[i] = result["sigma_Y_cov"]

            z_res = self.normalize_signals(result["signal"], self.normalization_method)
            combined_signals[i] = z_res
            normalized_combined_signals[i] = z_res

            eta_records.append({
                "date": sim_dates[i].strftime("%Y-%m-%d"),
                "eta": result.get("eta_t", 0.0),
                "ic": self._ic_history[-1] if self._ic_history else 0.0,
                "rolling_ic": float(np.mean(self._ic_history[-self.bayesian_ic_window:])) if self._ic_history else 0.0,
                "cs_var": float(self._cs_var_history[-1]) if self._cs_var_history else 0.0,
                "mode": self.bayesian_mode,
            })

        sim_dates_idx = sim_dates
        residual_blpx_df = pd.DataFrame(residual_blpx_signals, index=sim_dates_idx, columns=JP_TICKERS)
        combined_df = pd.DataFrame(combined_signals, index=sim_dates_idx, columns=JP_TICKERS)
        normalized_df = pd.DataFrame(normalized_combined_signals, index=sim_dates_idx, columns=JP_TICKERS)
        eta_df = pd.DataFrame(eta_records).set_index("date") if eta_records else pd.DataFrame()

        return {
            "signals": combined_df,
            "normalized_signals": normalized_df,
            "residual_blpx_signals": residual_blpx_df,
            "raw_pca_signals": residual_blpx_df,
            "residual_pca_signals": residual_blpx_df,
            "p4_signals": residual_blpx_df,
            "raw_blpx_signals": residual_blpx_df,
            "sigma_yy": sigma_yy_array,
            "y_jp_oc_df": inputs["y_jp_oc_df"],
            "blp_diagnostics": eta_df,
            "bayesian_diagnostics": eta_df,
        }
