"""Shared base class for BLP-based ensemble models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from leadlag.core import signal as signals
from leadlag.core.correlation import (
    build_base_vectors,
    build_v3_static,
    compute_baseline_correlation,
)
from leadlag.core.residualize import compute_rolling_ols_betas
from leadlag.data.tickers import JP_TICKERS, US_TICKERS
from leadlag.models.base import BaseModel
from leadlag.models.sre import compute_jp_target_returns


class _BLPBase(BaseModel):
    """Intermediate base class for BLP ensemble models.

    Provides shared implementations of _prepare_common_inputs, compute_production_signal,
    compute_residual_signal, _apply_gap_adjustment, and _denormalize_signal.
    """

    def _prepare_common_inputs(self, df_exec: pd.DataFrame) -> dict:
        """Prepare inputs common to signal computation (raw targets, residuals, betas)."""
        sim_dates = df_exec.index

        y_jp_target = compute_jp_target_returns(df_exec, JP_TICKERS)

        us_returns_raw = df_exec[[f"us_cc_{tk}" for tk in US_TICKERS]].values
        all_returns_raw = np.column_stack([us_returns_raw, y_jp_target])

        c_full = compute_baseline_correlation(
            all_returns_raw, sim_dates.values, self.ewma_half_life
        )
        v0_static = build_v3_static(self.n_u, self.n_j, self.include_v4_prior)
        base_vectors = build_base_vectors(self.n_u, self.n_j)
        v1, v2 = base_vectors["v1"], base_vectors["v2"]

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

        topix_cc_trade = (
            df_exec["topix_cc_trade"].values
            if "topix_cc_trade" in df_exec.columns
            else df_exec["topix_night_return"].values + df_exec["topix_oc_return"].values
        )

        betas_jp_p3 = compute_rolling_ols_betas(
            y_jp_target, topix_cc_trade.reshape(-1, 1), self.beta_window
        )
        y_residuals_p3 = y_jp_target - betas_jp_p3[:, :, 0] * topix_cc_trade.reshape(-1, 1)

        jp_res_returns_p3 = all_returns_raw.copy()
        jp_res_returns_p3[:, self.n_u :] = y_residuals_p3

        c_full_p3 = compute_baseline_correlation(
            jp_res_returns_p3, sim_dates.values, self.ewma_half_life
        )

        return {
            "all_returns_raw": all_returns_raw,
            "c_full": c_full,
            "c_full_p3": c_full_p3,
            "v0_static": v0_static,
            "v1": v1,
            "v2": v2,
            "jp_gap": jp_gap,
            "jp_beta": jp_beta,
            "topix_night": topix_night,
            "y_jp_oc_df": y_jp_oc_df,
            "jp_res_returns_p3": jp_res_returns_p3,
            "y_jp_target": y_jp_target,
        }

    def _compute_pca_signal(
        self,
        all_returns: np.ndarray,
        i: int,
        c_full: np.ndarray,
        v0_static: np.ndarray,
        v1: np.ndarray,
        v2: np.ndarray,
        jp_gap: np.ndarray,
        jp_beta: np.ndarray | None,
        topix_night: np.ndarray | None,
    ) -> np.ndarray:
        """Compute a PCA-based signal (Raw-PCA or Residual-PCA) at index i."""
        gap_t1 = np.nan_to_num(jp_gap[i], nan=0.0) if jp_gap is not None else np.zeros(self.n_j)
        betas_t = np.asarray(jp_beta[i], dtype=float) if jp_beta is not None else None
        topix_night_t = float(topix_night[i]) if topix_night is not None else None

        sig_res = signals.compute_signal(
            all_returns=all_returns,
            current_index=i,
            n_u=self.n_u,
            corr_window=self.corr_window,
            c_full=c_full,
            v0_static=v0_static,
            v1=v1,
            v2=v2,
            k=self.k,
            lambda_reg=self.lambda_reg,
            lambda_lw=self.lambda_lw,
            lw_target=self.lw_target,
            ewma_half_life=self.ewma_half_life,
            v3_dynamic=False,
            gap_override=gap_t1,
            gap_open_coef=self.gap_open_coef,
            topix_beta_coef=self.topix_beta_coef,
            betas_t=betas_t,
            topix_night_t=topix_night_t,
            vol_adjusted_target=self.vol_adjusted_target,
            min_raw_weight=getattr(self, "min_raw_weight", 0.0),
        )
        return np.asarray(sig_res["signal"], dtype=float)

    def compute_production_signal(
        self,
        i: int,
        c_full: np.ndarray,
        v0_static: np.ndarray,
        v1: np.ndarray,
        v2: np.ndarray,
        all_returns: np.ndarray,
        jp_gap: np.ndarray,
        jp_beta: np.ndarray | None,
        topix_night: np.ndarray | None,
    ) -> np.ndarray:
        """Compute the Raw-PCA (Production PCA) signal at index i."""
        return self._compute_pca_signal(
            all_returns, i, c_full, v0_static, v1, v2, jp_gap, jp_beta, topix_night
        )

    def compute_residual_signal(
        self,
        jp_res_returns_p3: np.ndarray,
        i: int,
        c_full_p3: np.ndarray,
        v0_static: np.ndarray,
        v1: np.ndarray,
        v2: np.ndarray,
        jp_gap: np.ndarray,
        jp_beta: np.ndarray | None,
        topix_night: np.ndarray | None,
    ) -> np.ndarray:
        """Compute the Residual-PCA (Residual target PCA) signal at index i."""
        return self._compute_pca_signal(
            jp_res_returns_p3, i, c_full_p3, v0_static, v1, v2, jp_gap, jp_beta, topix_night
        )

    @staticmethod
    def _denormalize_signal(
        z_hat_j_t1: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
        all_returns: np.ndarray,
        current_index: int,
        n_u: int,
        vol_adjusted_target: bool,
    ) -> np.ndarray:
        """Denormalize standardized JP return predictions to raw return space."""
        mu_jp = mu[n_u:]
        sigma_jp = sigma[n_u:]
        if vol_adjusted_target:
            if current_index >= 20:
                jp_returns_20 = all_returns[current_index - 20 : current_index, n_u:]
                jp_returns_20 = np.nan_to_num(jp_returns_20, nan=0.0, posinf=0.0, neginf=0.0)
                sigma_j_t = np.std(jp_returns_20, axis=0, ddof=1)
                sigma_j_t = np.maximum(sigma_j_t, 1e-8)
            else:
                sigma_j_t = sigma_jp
            r_hat_jp_cc = z_hat_j_t1 * sigma_j_t
        else:
            r_hat_jp_cc = mu_jp + sigma_jp * z_hat_j_t1
        return np.nan_to_num(r_hat_jp_cc, nan=0.0, posinf=0.0, neginf=0.0)

    def _apply_gap_adjustment(
        self,
        r_hat_jp_cc: np.ndarray,
        z_hat_j_t1: np.ndarray,
        gap_override: np.ndarray | None,
        betas_t: np.ndarray | None,
        topix_night_t: float | None,
    ) -> np.ndarray:
        """Apply gap override adjustment to the predicted signal."""
        if gap_override is not None:
            gap_vec = np.asarray(gap_override, dtype=float).reshape(-1)
            use_topix = False
            if betas_t is not None and topix_night_t is not None:
                betas_vec = np.asarray(betas_t, dtype=float).reshape(-1)
                if (
                    betas_vec.shape == gap_vec.shape
                    and np.all(np.isfinite(betas_vec))
                    and np.isfinite(float(topix_night_t))
                ):
                    use_topix = True

            if use_topix:
                gap_syst = betas_vec * float(topix_night_t)
                gap_idio = gap_vec - gap_syst
                gap_filt = (
                    self.gap_open_coef * gap_idio
                    + (self.gap_open_coef - self.topix_beta_coef) * gap_syst
                )
                denom = np.maximum(1.0 + gap_filt, 0.1)
                signal = (1.0 + r_hat_jp_cc) / denom - 1.0
            else:
                signal = r_hat_jp_cc - self.gap_open_coef * gap_vec
        else:
            signal = z_hat_j_t1
        return signal
