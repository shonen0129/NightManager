"""Shared base class for production BLPX models.

Consolidates the legacy ``src/research/models/base.py`` and
``src/research/models/blp_base.py`` into ``leadlag``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from leadlag.compliance.auditor import AuditContext
from leadlag.core import signal as signals
from leadlag.core.gap_adjustment import apply_gap_adjustment, denormalize_signal
from leadlag.data.tickers import JP_TICKERS
from leadlag.domain.signal import SignalPackage
from leadlag.utils.cache_manager import CacheManager

if TYPE_CHECKING:
    from leadlag.core.pipeline import PCAComponent


class _BLPBase:
    """Base class for BLPX production models."""

    def __init__(self, cfg: Any) -> None:
        from leadlag.config.schemas import BLPXConfig
        if isinstance(cfg, dict):
            cfg = BLPXConfig.model_validate(cfg)
        self.cfg: BLPXConfig = cfg

        # Core build_common_inputs parameters are read from the Pydantic config
        # via __getattr__; explicit attributes kept for type narrowing.
        self.ewma_halflife = int(self.cfg.ewma_halflife)
        self.beta_window = int(self.cfg.beta_window)
        self.include_v4_prior = bool(self.cfg.include_v4_prior)
        self.us_res_enabled = bool(self.cfg.us_res_enabled)
        self.us_res_gamma = float(self.cfg.us_res_gamma)
        self.us_res_beta_window = int(self.cfg.us_res_beta_window)
        self.frac_diff_enabled = bool(self.cfg.frac_diff_enabled)
        self.frac_diff_d = float(self.cfg.frac_diff_d)
        self.frac_diff_threshold = float(self.cfg.frac_diff_threshold)
        self.frac_diff_window = int(self.cfg.frac_diff_window)
        self.frac_diff_normalize = self.cfg.frac_diff_normalize

        self._cache_manager = CacheManager(
            CacheManager.config_hash_from_pydantic(self.cfg),
            maxsize=128,
        )

    def __getattr__(self, name: str) -> Any:
        if name != "cfg" and hasattr(self, "cfg") and hasattr(self.cfg, name):
            return getattr(self.cfg, name)
        raise AttributeError(f"{type(self).__name__} has no attribute {name!r}")

    def _resolve_slippage_bps(self) -> float:
        """Resolve slippage bps from config, checking the nested costs section."""
        slippage: float = getattr(self.cfg, "slippage_bps", 5.0) or 5.0
        costs = self.cfg.costs
        if costs is not None and hasattr(costs, "slippage_bps_per_side"):
            slippage = float(costs.slippage_bps_per_side)
        return slippage

    def normalize_signals(self, sig: np.ndarray, method: str = "zscore") -> np.ndarray:
        """Cross-sectionally normalize the signal values."""
        if method == "identity":
            return sig
        centered = sig - np.median(sig)
        if method == "zscore":
            std = np.std(centered)
            std_safe = std if std > 1e-8 else 1.0
            return cast(np.ndarray, centered / std_safe)
        elif method == "rank_normalize":
            ranks = pd.Series(sig).rank(pct=True).values
            return cast(np.ndarray, (ranks - 0.5) * 2.0)
        raise ValueError(f"Unknown normalization method: {method}")

    def build_weights(
        self,
        signal: np.ndarray,
        q: float | None = None,
        Sigma_YY: np.ndarray | None = None,
    ) -> np.ndarray:
        """Construct portfolio weights from a combined signal."""
        q_val = q if q is not None else getattr(self, "q", 0.5)
        n_j = getattr(self, "n_j", len(JP_TICKERS))
        if getattr(self, "minvar_enabled", False) and Sigma_YY is not None:
            from leadlag.core.signal import build_weights_minvar
            return build_weights_minvar(
                signal=signal,
                q=q_val,
                n_j=n_j,
                Sigma_YY=Sigma_YY,
                alpha=getattr(self, "minvar_alpha", 0.5),
                enforce_sign=False,
            )
        return signals.build_weights(
            signal=signal,
            q=q_val,
            n_j=n_j,
            weight_mode=getattr(self, "weight_mode", "signal"),
            enforce_sign=False,
        )

    def get_audit_context(self) -> AuditContext:
        """Return metadata required by ComplianceAuditor."""
        return AuditContext(
            n_u=getattr(self, "n_u", 15),
            n_j=getattr(self, "n_j", 17),
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

    def predict_signals(self, df_exec: pd.DataFrame, n_jobs: int = 1) -> SignalPackage:
        """Generate raw signals from the execution dataset (subclasses override)."""
        raise NotImplementedError

    def _prepare_common_inputs(
        self,
        df_exec: pd.DataFrame,
        *,
        horizon: int = 1,
        p_910_df: pd.DataFrame | None = None,
        y_jp_target: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Build and cache CommonInputs for the given df_exec and horizon."""
        from leadlag.core.pipeline import build_common_inputs
        from leadlag.data.preprocessor import compute_jp_target_returns
        from leadlag.data.tickers import US_TICKERS

        # Stable cache key based on the (immutable) df_exec shape and index
        # range, not object identity.  This lets different PITDataLake objects
        # holding identical history share the common-inputs cache while still
        # invalidating when the history window changes.
        cache_key = (
            df_exec.index[0],
            df_exec.index[-1],
            df_exec.shape,
            hash(tuple(df_exec.columns)),
            horizon,
        )
        common_inputs_cache = self._cache_manager.namespace("common_inputs")
        if cache_key in common_inputs_cache:
            return cast(dict[str, Any], common_inputs_cache[cache_key])

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
        common_inputs_cache[cache_key] = out
        return out

    def clear_caches(self) -> None:
        """Clear per-instance caches before pickling."""
        if hasattr(self, "_cache_manager"):
            self._cache_manager.clear()

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
        """Compute a PCA-based signal at index i."""
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
        """Compute the Raw-PCA signal at index i."""
        return self._compute_pca_signal(all_returns, i, c_full, v0_static, v1, v2, jp_gap, jp_beta, topix_night)

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
        """Compute the Residual-PCA signal at index i."""
        return self._compute_pca_signal(jp_res_returns_p3, i, c_full_p3, v0_static, v1, v2, jp_gap, jp_beta, topix_night)

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
        return denormalize_signal(z_hat_j_t1, mu, sigma, all_returns, current_index, n_u, vol_adjusted_target)

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
        return apply_gap_adjustment(r_hat_jp_cc, z_hat_j_t1, gap_override, betas_t, topix_night_t, gap_coef, beta_coef)


# Backward-compatible alias for research models and historical references.
BLPModelBase = _BLPBase
