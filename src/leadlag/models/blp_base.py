"""Shared base class for BLP-based ensemble models."""

from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
import pandas as pd

from leadlag.data.tickers import JP_TICKERS
from leadlag.models.base import BaseModel

if TYPE_CHECKING:
    from leadlag.core.pipeline import PCAComponent


class _BLPBase(BaseModel):
    """Intermediate base class for BLP ensemble models.

    Provides shared implementations of _prepare_common_inputs, compute_production_signal,
    compute_residual_signal, _apply_gap_adjustment, and _denormalize_signal.
    """

    def _prepare_common_inputs(self, df_exec: pd.DataFrame) -> dict:
        """Prepare inputs common to signal computation (raw targets, residuals, betas)."""
        from leadlag.core.pipeline import build_common_inputs
        from leadlag.models.sre import compute_jp_target_returns

        y_jp_target = compute_jp_target_returns(df_exec, JP_TICKERS)

        # Read fractional diff config from the features section
        frac_cfg = {}
        if isinstance(self.config, dict):
            frac_cfg = self.config.get("features", {}).get("fractional_diff", {})
        frac_diff_enabled = bool(frac_cfg.get("enabled", False))
        frac_diff_d = float(frac_cfg.get("d", 0.5))
        frac_diff_threshold = float(frac_cfg.get("threshold", 1e-5))
        frac_diff_window = int(frac_cfg.get("window", 100))

        inputs = build_common_inputs(
            df_exec,
            y_jp_target,
            n_u=self.n_u,
            n_j=self.n_j,
            ewma_half_life=self.ewma_half_life,
            beta_window=self.beta_window,
            include_v4_prior=self.include_v4_prior,
            frac_diff_enabled=frac_diff_enabled,
            frac_diff_d=frac_diff_d,
            frac_diff_threshold=frac_diff_threshold,
            frac_diff_window=frac_diff_window,
        )
        out = inputs.to_dict()
        out["y_jp_target"] = y_jp_target
        return out

    def _get_pca_component(self) -> "PCAComponent":
        """Lazily create and cache a PCAComponent for PCA signal computation."""
        if not hasattr(self, "_pca_component"):
            from leadlag.core.pipeline import PCAComponent
            self._pca_component = PCAComponent(
                name="pca",
                n_u=self.n_u,
                n_j=self.n_j,
                corr_window=self.corr_window,
                k=self.k,
                lambda_reg=self.lambda_reg,
                lambda_lw=self.lambda_lw,
                lw_target=self.lw_target,
                ewma_half_life=self.ewma_half_life,
                gap_open_coef=self.gap_open_coef,
                topix_beta_coef=self.topix_beta_coef,
                vol_adjusted_target=self.vol_adjusted_target,
                min_raw_weight=getattr(self, "min_raw_weight", 0.0),
            )
        return self._pca_component

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
        comp = self._get_pca_component()
        result = comp.compute_standalone(
            i=i,
            all_returns=all_returns,
            c_full=c_full,
            v0_static=v0_static,
            v1=v1,
            v2=v2,
            jp_gap=jp_gap,
            jp_beta=jp_beta,
            topix_night=topix_night,
        )
        return result.signal

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
        gap_open_coef_override: float | None = None,
        topix_beta_coef_override: float | None = None,
    ) -> np.ndarray:
        """Apply gap override adjustment to the predicted signal."""
        gap_coef = gap_open_coef_override if gap_open_coef_override is not None else self.gap_open_coef
        beta_coef = topix_beta_coef_override if topix_beta_coef_override is not None else self.topix_beta_coef

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
                    gap_coef * gap_idio
                    + (gap_coef - beta_coef) * gap_syst
                )
                denom = np.maximum(1.0 + gap_filt, 0.1)
                signal = (1.0 + r_hat_jp_cc) / denom - 1.0
            else:
                signal = r_hat_jp_cc - gap_coef * gap_vec
        else:
            signal = z_hat_j_t1
        return signal
