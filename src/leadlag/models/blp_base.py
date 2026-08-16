"""Shared base classes for production BLPX models.

This module consolidates the legacy ``src/research/models/base.py`` and
``src/research/models/blp_base.py`` into ``leadlag``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from leadlag.compliance.auditor import AuditContext
from leadlag.core import signal as signals
from leadlag.data.tickers import JP_TICKERS

if TYPE_CHECKING:
    from leadlag.core.pipeline import PCAComponent


class BLPModelBase(ABC):
    """Minimal base class for config-driven production models."""

    _config_sections: list[str] = [
        "model", "ensemble", "portfolio", "costs", "residualization", "blpx"
    ]
    _config_aliases: dict[str, list[str]] = {}

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg

    def _resolve_val(self, key: str, default: Any) -> Any:
        """Resolve value from the nested config dict.

        Searches the dict's top-level keys, then nested section keys
        (as defined by ``_config_sections``), and finally applies any
        alias translations (as defined by ``_config_aliases``).
        """
        aliases = self._config_aliases.get(key, [])
        keys_to_try = [key] + aliases
        for k in keys_to_try:
            if k in self.cfg:
                return self.cfg[k]
            for section in self._config_sections:
                if section in self.cfg and isinstance(self.cfg[section], dict) and k in self.cfg[section]:
                    return self.cfg[section][k]
            # Legacy translations
            if k == "model_name" and "name" in self.cfg.get("model", {}):
                return self.cfg["model"]["name"]
            if k == "k" and "k" in self.cfg.get("model", {}):
                return self.cfg["model"]["k"]
            if k == "q" and "long_short_frac" in self.cfg.get("portfolio", {}):
                return self.cfg["portfolio"]["long_short_frac"]
        return default

    def _resolve_nested(self, key: str, default: Any) -> Any:
        """Resolve dotted nested keys or fall back to _resolve_val."""
        parts = key.split(".")
        val = self._resolve_val(parts[-1], None)
        if val is not None:
            return val
        curr = self.cfg
        for part in parts:
            if isinstance(curr, dict) and part in curr:
                curr = curr[part]
            else:
                return default
        return curr

    def _resolve_slippage_bps(self) -> float:
        """Resolve slippage bps from config, checking costs section."""
        slippage = self._resolve_val("slippage_bps", 5.0)
        if "costs" in self.cfg and "slippage_bps_per_side" in self.cfg["costs"]:
            slippage = float(self.cfg["costs"]["slippage_bps_per_side"])
        return float(slippage)

    def normalize_signals(self, sig: np.ndarray, method: str = "zscore") -> np.ndarray:
        """Cross-sectionally normalize the signal values."""
        if method == "identity":
            return sig
        centered = sig - np.median(sig)
        if method == "zscore":
            std = np.std(centered)
            std_safe = std if std > 1e-8 else 1.0
            return centered / std_safe
        elif method == "rank_normalize":
            ranks = pd.Series(sig).rank(pct=True).values
            return (ranks - 0.5) * 2.0
        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def build_weights(
        self,
        signal: np.ndarray,
        q: float | None = None,
        Sigma_YY: np.ndarray | None = None,
    ) -> np.ndarray:
        """Construct portfolio weights from combined signal."""
        q_val = q if q is not None else getattr(self, "q", 0.5)

        if getattr(self, "minvar_enabled", False) and Sigma_YY is not None:
            from leadlag.core.signal import build_weights_minvar
            return build_weights_minvar(
                signal=signal,
                q=q_val,
                n_j=getattr(self, "n_j", len(JP_TICKERS)),
                Sigma_YY=Sigma_YY,
                alpha=getattr(self, "minvar_alpha", 0.5),
                enforce_sign=False,
            )

        return signals.build_weights(
            signal=signal,
            q=q_val,
            n_j=getattr(self, "n_j", len(JP_TICKERS)),
            weight_mode=getattr(self, "weight_mode", "signal"),
            enforce_sign=False,
        )

    def get_audit_context(self) -> AuditContext:
        """Return metadata required by ComplianceAuditor."""
        n_u = getattr(self, "n_u", 15)
        n_j = getattr(self, "n_j", 17)
        return AuditContext(
            n_u=n_u,
            n_j=n_j,
            us_res_enabled=getattr(self, "us_res_enabled", False),
            us_res_beta_shift=getattr(self, "us_res_beta_shift", 1),
            us_res_beta_window=getattr(self, "us_res_beta_window", 60),
            us_res_gamma=getattr(self, "us_res_gamma", 0.5),
            prior_variant=getattr(self, "prior_variant", None),
            raw_pca_weight=getattr(self, "raw_pca_weight", 0.5),
            residual_pca_weight=getattr(self, "residual_pca_weight", 0.5),
            p4_weight=getattr(self, "p4_weight", 0.0),
            raw_blpx_weight=getattr(self, "raw_blpx_weight", 0.0),
            residual_blpx_weight=getattr(self, "residual_blpx_weight", 0.0),
        )

    @abstractmethod
    def predict_signals(self, df_exec: pd.DataFrame, n_jobs: int = 1) -> dict[str, np.ndarray]:
        """Generate raw signals from the execution dataset."""
        pass


class _BLPBase(BLPModelBase):
    """Intermediate base class for BLPX production models."""

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        # Core build_common_inputs parameters
        self.ewma_halflife = int(self._resolve_val("ewma_halflife", 120))
        self.beta_window = int(self._resolve_val("beta_window", 60))
        self.include_v4_prior = bool(self._resolve_val("include_v4_prior", False))
        self.us_res_enabled = bool(self._resolve_val("us_res_enabled", False))
        self.us_res_gamma = float(self._resolve_val("us_res_gamma", 0.5))
        self.us_res_beta_window = int(self._resolve_val("us_res_beta_window", 252))
        self.frac_diff_enabled = bool(self._resolve_val("frac_diff_enabled", False))
        self.frac_diff_d = float(self._resolve_val("frac_diff_d", 0.1))
        self.frac_diff_threshold = float(self._resolve_val("frac_diff_threshold", 1e-5))
        self.frac_diff_window = int(self._resolve_val("frac_diff_window", 100))
        self.frac_diff_normalize = self._resolve_val("frac_diff_normalize", None)

    def _prepare_common_inputs(
        self,
        df_exec: pd.DataFrame,
        *,
        horizon: int = 1,
        p_910_df: pd.DataFrame | None = None,
        y_jp_target: np.ndarray | None = None,
    ) -> dict:
        """Build and cache CommonInputs for the given df_exec and horizon."""
        from leadlag.core.pipeline import build_common_inputs
        from leadlag.data.preprocessor import compute_jp_target_returns
        from leadlag.data.tickers import US_TICKERS

        cache_key = (id(df_exec), horizon)
        if not hasattr(self, "_common_inputs_cache"):
            self._common_inputs_cache: dict = {}
        if cache_key in self._common_inputs_cache:
            return self._common_inputs_cache[cache_key]

        if y_jp_target is None:
            y_jp_target = compute_jp_target_returns(
                df_exec, JP_TICKERS, horizon=horizon, p_910_df=p_910_df
            )

        inputs = build_common_inputs(
            df_exec,
            y_jp_target,
            n_u=getattr(self, "n_u", len(US_TICKERS)),
            n_j=getattr(self, "n_j", len(JP_TICKERS)),
            ewma_half_life=self.ewma_halflife,
            beta_window=self.beta_window,
            include_v4_prior=self.include_v4_prior,
            us_res_enabled=self.us_res_enabled,
            us_res_gamma=self.us_res_gamma,
            us_res_beta_window=self.us_res_beta_window,
            frac_diff_enabled=self.frac_diff_enabled,
            frac_diff_d=self.frac_diff_d,
            frac_diff_threshold=self.frac_diff_threshold,
            frac_diff_window=self.frac_diff_window,
            frac_diff_normalize=self.frac_diff_normalize,
        )
        out = inputs.to_dict()
        out["y_jp_target"] = y_jp_target
        self._common_inputs_cache[cache_key] = out
        return out

    def clear_caches(self) -> None:
        """Clear per-instance caches before pickling."""
        for attr in (
            "_production_signal_cache",
            "_residual_signal_cache",
            "_raw_pca_cache",
            "_residual_pca_cache",
            "_blp_corr_cache",
            "_common_inputs_cache",
            "_macro_price_cache",
        ):
            if hasattr(self, attr):
                getattr(self, attr).clear()

    def _get_pca_component(self) -> PCAComponent:
        """Lazily create and cache a PCAComponent for PCA signal computation."""
        if not hasattr(self, "_pca_component"):
            from leadlag.core.pipeline import PCAComponent
            self._pca_component = PCAComponent(
                name="pca",
                n_u=self.n_u,
                n_j=self.n_j,
                corr_window=getattr(self, "corr_window", 252),
                k=getattr(self, "k", 5),
                lambda_reg=getattr(self, "lambda_reg", 0.1),
                lambda_lw=getattr(self, "lambda_lw", 0.1),
                lw_target=getattr(self, "lw_target", "identity"),
                ewma_half_life=self.ewma_halflife,
                gap_open_coef=getattr(self, "gap_open_coef", 0.7),
                topix_beta_coef=getattr(self, "topix_beta_coef", 0.6),
                vol_adjusted_target=getattr(self, "vol_adjusted_target", False),
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
        gap_coef = gap_open_coef_override if gap_open_coef_override is not None else getattr(self, "gap_open_coef", 0.7)
        beta_coef = topix_beta_coef_override if topix_beta_coef_override is not None else getattr(self, "topix_beta_coef", 0.6)

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
