"""Sector Relative Ensemble with Regularized Block BLP Model.

Implements the PCA-Ensemble-BLP model which integrates standard Production signals (Raw-PCA, Residual-PCA)
with Regularized Block BLP signals (P5, P5P3).
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from typing import Any

from leadlag.core import signal as signals
from leadlag.core.correlation import (
    build_base_vectors,
    build_v3_static,
    compute_baseline_correlation,
    compute_correlation,
)
from leadlag.core.residualize import compute_rolling_ols_betas
from leadlag.data.tickers import JP_TICKERS, US_TICKERS
from leadlag.models.base import BaseModel
from leadlag.models.sre import compute_jp_target_returns

logger = logging.getLogger(__name__)


class SectorRelativeEnsembleBLPModel(BaseModel):
    """Sector Relative Ensemble with Regularized Block BLP (PCA-Ensemble-BLP) Model.

    Ensembles standard PCA signals (Raw-PCA, Residual-PCA) and Regularized Block BLP signals (P5, P5P3).
    """

    def __init__(self, config: dict | object):
        """Initialize SectorRelativeEnsembleBLPModel.

        Args:
            config: Dict or object containing configuration options.
        """
        self.config = config
        self.n_u = len(US_TICKERS)
        self.n_j = len(JP_TICKERS)

        # Config resolution
        self.model_name = self._resolve_val("model_name", "sector_relative_ensemble_blp")
        self.k = self._resolve_val("k", 6)
        self.lambda_reg = self._resolve_val("lambda_reg", 0.75)
        self.q = self._resolve_val("q", 0.3)
        self.weight_mode = self._resolve_val("weight_mode", "signal")
        self.ewma_half_life = self._resolve_val("ewma_half_life", 45)
        self.lambda_lw = self._resolve_val("lambda_lw", 0.5)
        self.lw_target = self._resolve_val("lw_target", "equicorrelation")
        self.corr_window = self._resolve_val("corr_window", 60)
        self.include_v4_prior = self._resolve_val("include_v4_prior", True)
        self.gap_open_coef = self._resolve_val("gap_open_coef", 0.70)
        self.topix_beta_coef = self._resolve_val("topix_beta_coef", 0.6)
        self.beta_window = self._resolve_val("beta_window", 60)
        self.vol_adjusted_target = self._resolve_val("vol_adjusted_target", True)
        self.normalization_method = self._resolve_val("normalization", "zscore")

        # BLP Parameters (with standard defaults)
        self.blp_window = int(self._resolve_val("blp_window", 252))
        self.blp_ewma_halflife = self._resolve_val("blp_ewma_halflife", 45)
        self.alpha_xx = float(self._resolve_val("alpha_xx", 0.5))
        self.alpha_yx = float(self._resolve_val("alpha_yx", 0.25))
        self.rho = float(self._resolve_val("rho", 0.03))
        self.rank = self._resolve_val("rank", "full")

        # Ensemble weights
        self.p0_weight = float(self._resolve_val("p0_weight", 0.4))
        self.p3_weight = float(self._resolve_val("p3_weight", 0.4))
        self.p5_weight = float(self._resolve_val("p5_weight", 0.1))
        self.p5p3_weight = float(self._resolve_val("p5p3_weight", 0.1))

        # Slippage cost parameter resolution
        self.slippage_bps = self._resolve_val("slippage_bps", 5.0)
        if isinstance(config, dict):
            if "costs" in config and "slippage_bps_per_side" in config["costs"]:
                self.slippage_bps = float(config["costs"]["slippage_bps_per_side"])

    def _resolve_val(self, key: str, default: any) -> any:
        """Resolve value from config object or dict."""
        if hasattr(self.config, key):
            return getattr(self.config, key)
        if isinstance(self.config, dict):
            if key in self.config:
                return self.config[key]
            for section in ["model", "ensemble", "portfolio", "costs", "residualization"]:
                if section in self.config and key in self.config[section]:
                    return self.config[section][key]
            # Translations
            if key == "model_name" and "name" in self.config.get("model", {}):
                return self.config["model"]["name"]
            if key == "k" and "k" in self.config.get("model", {}):
                return self.config["model"]["k"]
            if key == "q" and "long_short_frac" in self.config.get("portfolio", {}):
                return self.config["portfolio"]["long_short_frac"]
        return default

    def _prepare_common_inputs(self, df_exec: pd.DataFrame) -> dict:
        """Prepare inputs common to signal computation (raw targets, residuals, betas)."""
        sim_dates = df_exec.index

        # Target returns for JP on trade_date (D_t+1)
        y_jp_target = compute_jp_target_returns(df_exec, JP_TICKERS)

        # Build all_returns_raw: US columns are us_cc_* (on D_t), JP columns are y_jp_target (on D_t+1)
        us_returns_raw = df_exec[[f"us_cc_{tk}" for tk in US_TICKERS]].values
        all_returns_raw = np.column_stack([us_returns_raw, y_jp_target])

        # PCA baseline correlation
        c_full = compute_baseline_correlation(
            all_returns_raw, sim_dates.values, self.ewma_half_life
        )
        v0_static = build_v3_static(self.n_u, self.n_j, self.include_v4_prior)
        base_vectors = build_base_vectors(self.n_u, self.n_j)
        v1, v2 = base_vectors["v1"], base_vectors["v2"]

        # Parse gap, beta, and topix night
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

        # TOPIX trade return
        topix_cc_trade = (
            df_exec["topix_cc_trade"].values
            if "topix_cc_trade" in df_exec.columns
            else df_exec["topix_night_return"].values + df_exec["topix_oc_return"].values
        )

        # Rolling OLS residualization for Residual-PCA/P5P3
        betas_jp_p3 = compute_rolling_ols_betas(
            y_jp_target, topix_cc_trade.reshape(-1, 1), self.beta_window
        )
        y_residuals_p3 = y_jp_target - betas_jp_p3[:, :, 0] * topix_cc_trade.reshape(-1, 1)

        # Replace JP columns with residuals for TOPIX-residualized returns
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
        }

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
        )
        return np.asarray(sig_res["signal"], dtype=float)

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
        gap_t1 = np.nan_to_num(jp_gap[i], nan=0.0) if jp_gap is not None else np.zeros(self.n_j)
        betas_t = np.asarray(jp_beta[i], dtype=float) if jp_beta is not None else None
        topix_night_t = float(topix_night[i]) if topix_night is not None else None

        sig_res = signals.compute_signal(
            all_returns=jp_res_returns_p3,
            current_index=i,
            n_u=self.n_u,
            corr_window=self.corr_window,
            c_full=c_full_p3,
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
        )
        return np.asarray(sig_res["signal"], dtype=float)

    def compute_blp_signal(
        self,
        all_returns: np.ndarray,
        current_index: int,
        gap_override: np.ndarray | None = None,
        betas_t: np.ndarray | None = None,
        topix_night_t: float | None = None,
    ) -> dict[str, Any]:
        """Compute the Regularized Block BLP signal for a single time step.

        Ensure Y_date <= signal_date by slicing up to current_index - 1.
        """
        window_start = max(0, current_index - self.blp_window)
        # Slices from window_start to current_index-1 (contains historical X and Y)
        window_returns = all_returns[window_start:current_index]
        window_returns = np.nan_to_num(window_returns, nan=0.0, posinf=0.0, neginf=0.0)

        # Estimate rolling mean, std, and correlation
        mu, sigma, corr = compute_correlation(window_returns, self.blp_ewma_halflife)
        mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
        sigma = np.nan_to_num(sigma, nan=1.0, posinf=1.0, neginf=1.0)
        corr = np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0)
        np.fill_diagonal(corr, 1.0)

        # Partition correlation matrix (scale-standardized covariance)
        C_XX = corr[: self.n_u, : self.n_u]
        C_YX = corr[self.n_u :, : self.n_u]

        # Regularize Sigma_XX and Sigma_YX
        Sigma_XX_reg = (1.0 - self.alpha_xx) * C_XX + self.alpha_xx * np.eye(self.n_u)
        Sigma_YX_reg = (1.0 - self.alpha_yx) * C_YX

        # Scale-type Ridge matrix
        diag_mean = float(np.mean(np.diag(Sigma_XX_reg)))
        ridge_matrix = self.rho * diag_mean * np.eye(self.n_u)
        A = Sigma_XX_reg + ridge_matrix

        # Solve for B_t with pseudo-inverse fallback
        try:
            singular_values = np.linalg.svd(A, compute_uv=False)
            cond_num = float(singular_values[0] / np.maximum(singular_values[-1], 1e-12))
        except Exception:
            cond_num = np.nan

        pinv_fallback = False
        try:
            if not np.isfinite(A).all():
                raise ValueError("A contains NaNs or Infs")
            inv_A = np.linalg.inv(A)
            B_t = Sigma_YX_reg @ inv_A
        except Exception:
            pinv_fallback = True
            try:
                inv_A = np.linalg.pinv(A)
                B_t = Sigma_YX_reg @ inv_A
            except Exception:
                B_t = np.zeros((self.n_j, self.n_u))

        # Low rank SVD projection
        if self.rank != "full" and self.rank is not None:
            rank_val = int(self.rank)
            if rank_val < min(B_t.shape):
                try:
                    U, S, Vt = np.linalg.svd(B_t, full_matrices=False)
                    B_t = U[:, :rank_val] @ np.diag(S[:rank_val]) @ Vt[:rank_val, :]
                except Exception as e:
                    logger.warning(f"SVD rank reduction failed: {e}")

        # Standardize current US input
        X_t = all_returns[current_index, : self.n_u]
        X_t = np.nan_to_num(X_t, nan=0.0, posinf=0.0, neginf=0.0)
        mu_X = mu[: self.n_u]
        sigma_X = sigma[: self.n_u]
        sigma_X_safe = np.where(sigma_X > 1e-8, sigma_X, 1.0)
        z_U_t = (X_t - mu_X) / sigma_X_safe

        # Predict standardized JP returns
        z_hat_j_t1 = B_t @ z_U_t
        z_hat_j_t1 = np.nan_to_num(z_hat_j_t1, nan=0.0, posinf=0.0, neginf=0.0)

        # Vol adjustment / denormalization
        mu_jp = mu[self.n_u :]
        sigma_jp = sigma[self.n_u :]
        if self.vol_adjusted_target:
            if current_index >= 20:
                jp_returns_20 = all_returns[current_index - 20 : current_index, self.n_u :]
                jp_returns_20 = np.nan_to_num(jp_returns_20, nan=0.0, posinf=0.0, neginf=0.0)
                sigma_j_t = np.std(jp_returns_20, axis=0, ddof=1)
                sigma_j_t = np.maximum(sigma_j_t, 1e-8)
            else:
                sigma_j_t = sigma_jp
            r_hat_jp_cc = z_hat_j_t1 * sigma_j_t
        else:
            r_hat_jp_cc = mu_jp + sigma_jp * z_hat_j_t1
        r_hat_jp_cc = np.nan_to_num(r_hat_jp_cc, nan=0.0, posinf=0.0, neginf=0.0)

        # Apply gap override
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

        b_norm = float(np.linalg.norm(B_t))
        sigma_xx_trace = float(np.trace(C_XX))
        sigma_yx_norm = float(np.linalg.norm(C_YX))

        return {
            "signal": signal,
            "z_hat_j_t1": z_hat_j_t1,
            "cond_num": cond_num,
            "b_norm": b_norm,
            "sigma_xx_trace": sigma_xx_trace,
            "sigma_yx_norm": sigma_yx_norm,
            "pinv_fallback": pinv_fallback,
            "num_training_samples": len(window_returns),
        }

    def normalize_signals(self, sig: np.ndarray, method: str = "zscore") -> np.ndarray:
        """Cross-sectionally normalize the signal values."""
        if method == "identity":
            return sig
        centered = sig - np.median(sig)
        if method == "zscore":
            std = np.std(centered)
            # Safe std division: if std is zero or extremely small, use 1.0
            std_safe = std if std > 1e-8 else 1.0
            return centered / std_safe
        elif method == "rank_normalize":
            ranks = pd.Series(sig).rank(pct=True).values
            return (ranks - 0.5) * 2.0
        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def combine_signals(
        self, z0: np.ndarray, z3: np.ndarray, z5: np.ndarray, z5p3: np.ndarray
    ) -> np.ndarray:
        """Combine component signals with ensemble weights."""
        return (
            self.p0_weight * z0
            + self.p3_weight * z3
            + self.p5_weight * z5
            + self.p5p3_weight * z5p3
        )

    def build_weights(self, signal: np.ndarray, q: float | None = None) -> np.ndarray:
        """Construct portfolio weights from combined signal."""
        q_val = q if q is not None else self.q
        return signals.build_weights(
            signal=signal,
            q=q_val,
            n_j=self.n_j,
            weight_mode=self.weight_mode,
            enforce_sign=False,
        )

    def predict_signals(self, df_exec: pd.DataFrame) -> dict[str, Any]:
        """Generate component and ensemble signals for all rows in df_exec."""
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

        # Setup output arrays
        p0_signals = np.zeros((T, self.n_j))
        p3_signals = np.zeros((T, self.n_j))
        p5_signals = np.zeros((T, self.n_j))
        p5p3_signals = np.zeros((T, self.n_j))
        combined_signals = np.zeros((T, self.n_j))
        normalized_combined_signals = np.zeros((T, self.n_j))

        # Track BLP diagnostics
        blp_diagnostics = []

        start_idx = self.corr_window
        for i in range(start_idx, T):
            # 1. Raw-PCA (Production PCA)
            p0_sig = self.compute_production_signal(
                i, c_full, v0_static, v1, v2, all_returns_raw, jp_gap, jp_beta, topix_night
            )
            p0_signals[i] = p0_sig

            # 2. Residual-PCA (Residual target PCA)
            p3_sig = self.compute_residual_signal(
                jp_res_returns_p3, i, c_full_p3, v0_static, v1, v2, jp_gap, jp_beta, topix_night
            )
            p3_signals[i] = p3_sig

            # 3. P5 (Raw target BLP)
            gap_override = np.nan_to_num(jp_gap[i], nan=0.0) if jp_gap is not None else None
            betas_t = np.asarray(jp_beta[i], dtype=float) if jp_beta is not None else None
            topix_night_t = float(topix_night[i]) if topix_night is not None else None

            p5_res = self.compute_blp_signal(
                all_returns_raw,
                i,
                gap_override=gap_override,
                betas_t=betas_t,
                topix_night_t=topix_night_t,
            )
            p5_signals[i] = p5_res["signal"]

            # 4. P5P3 (Residual target BLP)
            p5p3_res = self.compute_blp_signal(
                jp_res_returns_p3,
                i,
                gap_override=gap_override,
                betas_t=betas_t,
                topix_night_t=topix_night_t,
            )
            p5p3_signals[i] = p5p3_res["signal"]

            # Standard Z-score normalization of component signals
            z0 = self.normalize_signals(p0_sig, self.normalization_method)
            z3 = self.normalize_signals(p3_sig, self.normalization_method)
            z5 = self.normalize_signals(p5_res["signal"], self.normalization_method)
            z5p3 = self.normalize_signals(p5p3_res["signal"], self.normalization_method)

            # Combined PCA-Ensemble-BLP signal
            s_ens = self.combine_signals(z0, z3, z5, z5p3)
            combined_signals[i] = s_ens
            normalized_combined_signals[i] = self.normalize_signals(
                s_ens, self.normalization_method
            )

            # Diagnostics recording (e.g. log daily BLP states)
            date_str = sim_dates[i].strftime("%Y-%m-%d")
            blp_diagnostics.append(
                {
                    "date": date_str,
                    "p5_cond_num": p5_res["cond_num"],
                    "p5_b_norm": p5_res["b_norm"],
                    "p5_sigma_xx_trace": p5_res["sigma_xx_trace"],
                    "p5_sigma_yx_norm": p5_res["sigma_yx_norm"],
                    "p5_pinv_fallback": int(p5_res["pinv_fallback"]),
                    "p5_num_training_samples": p5_res["num_training_samples"],
                    "p5p3_cond_num": p5p3_res["cond_num"],
                    "p5p3_b_norm": p5p3_res["b_norm"],
                    "p5p3_sigma_xx_trace": p5p3_res["sigma_xx_trace"],
                    "p5p3_sigma_yx_norm": p5p3_res["sigma_yx_norm"],
                    "p5p3_pinv_fallback": int(p5p3_res["pinv_fallback"]),
                    "p5p3_num_training_samples": p5p3_res["num_training_samples"],
                }
            )

        # Build DataFrames
        p0_df = pd.DataFrame(p0_signals, index=sim_dates, columns=JP_TICKERS)
        p3_df = pd.DataFrame(p3_signals, index=sim_dates, columns=JP_TICKERS)
        p4_signals = np.zeros((T, self.n_j))
        p4_df = pd.DataFrame(p4_signals, index=sim_dates, columns=JP_TICKERS)
        p5_df = pd.DataFrame(p5_signals, index=sim_dates, columns=JP_TICKERS)
        p5p3_df = pd.DataFrame(p5p3_signals, index=sim_dates, columns=JP_TICKERS)
        combined_df = pd.DataFrame(combined_signals, index=sim_dates, columns=JP_TICKERS)
        normalized_df = pd.DataFrame(
            normalized_combined_signals, index=sim_dates, columns=JP_TICKERS
        )

        return {
            "p0_signals": p0_df,
            "p3_signals": p3_df,
            "p4_signals": p4_df,
            "p5_signals": p5_df,
            "p5p3_signals": p5p3_df,
            "signals": combined_df,
            "normalized_signals": normalized_df,
            "y_jp_oc_df": inputs["y_jp_oc_df"],
            "blp_diagnostics": pd.DataFrame(blp_diagnostics).set_index("date"),
        }
