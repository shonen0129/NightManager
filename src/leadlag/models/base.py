"""Abstract Base Class for strategy models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from leadlag.core import signal as signals


@dataclass
class AuditContext:
    """Metadata required by ComplianceAuditor for safety checks.

    Models override ``BaseModel.get_audit_context()`` to expose their
    internal state without the auditor relying on duck-typed ``getattr`` calls.
    """

    n_u: int
    """US ETF 銘柄数 (N_US)."""

    n_j: int
    """JP ETF 銘柄数 (N_JP)."""

    us_res_enabled: bool = False
    """True if US return residualization (P4 variant) is active."""

    us_res_beta_shift: int = 1
    """Beta window shift used for US residualization (must be 1 for no-lookahead)."""

    us_res_beta_window: int = 60
    """Beta estimation window length for US residualization."""

    us_res_gamma: float = 0.5
    """Blend coefficient for US residualization."""

    prior_variant: str | None = None
    """Prior subspace variant identifier (e.g. 'resid_v2_removed'), or None."""

    raw_pca_weight: float = 0.5
    """Ensemble weight for Raw-PCA (Production PCA) signal."""

    residual_pca_weight: float = 0.5
    """Ensemble weight for Residual-PCA (Residual target PCA) signal."""

    p4_weight: float = 0.0
    """Ensemble weight for P4 (US-residualized) signal."""

    raw_blpx_weight: float = 0.0
    """Ensemble weight for Raw-BLPX signal."""

    residual_blpx_weight: float = 0.0
    """Ensemble weight for Residual-BLPX signal."""

    extra: dict = field(default_factory=dict)
    """Model-specific auxiliary metadata (arbitrary key-value pairs)."""


class BaseModel(ABC):
    """Abstract Base Class for Lead-Lag strategy models."""

    _config_sections: list[str] = ["model", "ensemble", "portfolio", "costs", "residualization"]
    _config_aliases: dict[str, list[str]] = {}

    @abstractmethod
    def predict_signals(self, df_exec: pd.DataFrame, n_jobs: int = 1) -> dict[str, np.ndarray]:
        """Generate raw signals from the execution dataset.

        Args:
            df_exec: Execution DataFrame.
            n_jobs: Number of parallel workers for signal computation. 1 = sequential.

        Returns:
            Dict containing the final signal array (shape N_J,) and other component signals.
        """
        pass

    def _resolve_val(self, key: str, default: any) -> any:
        """Resolve value from config object or dict.

        Searches the config object's attributes, then the dict's top-level keys,
        then nested section keys (as defined by ``_config_sections``), and finally
        applies any alias translations (as defined by ``_config_aliases``).
        """
        aliases = self._config_aliases.get(key, [])
        keys_to_try = [key] + aliases
        for k in keys_to_try:
            if hasattr(self.config, k):
                return getattr(self.config, k)
            if isinstance(self.config, dict):
                if k in self.config:
                    return self.config[k]
                for section in self._config_sections:
                    if section in self.config and isinstance(self.config[section], dict) and k in self.config[section]:
                        return self.config[section][k]
                # Translations
                if k == "model_name" and "name" in self.config.get("model", {}):
                    return self.config["model"]["name"]
                if k == "k" and "k" in self.config.get("model", {}):
                    return self.config["model"]["k"]
                if k == "q" and "long_short_frac" in self.config.get("portfolio", {}):
                    return self.config["portfolio"]["long_short_frac"]
        return default

    def _resolve_nested(self, key: str, default: any) -> any:
        """Resolve dotted nested keys or fall back to _resolve_val."""
        parts = key.split(".")
        val = self._resolve_val(parts[-1], None)
        if val is not None:
            return val
        if isinstance(self.config, dict):
            curr = self.config
            for part in parts:
                if isinstance(curr, dict) and part in curr:
                    curr = curr[part]
                else:
                    return default
            return curr
        return default

    def _resolve_slippage_bps(self) -> float:
        """Resolve slippage bps from config, checking costs section."""
        slippage = self._resolve_val("slippage_bps", 5.0)
        if isinstance(self.config, dict):
            if "costs" in self.config and "slippage_bps_per_side" in self.config["costs"]:
                slippage = float(self.config["costs"]["slippage_bps_per_side"])
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
        self, signal: np.ndarray, q: float | None = None,
        Sigma_YY: np.ndarray | None = None,
    ) -> np.ndarray:
        """Construct portfolio weights from combined signal."""
        q_val = q if q is not None else self.q

        if getattr(self, "minvar_enabled", False) and Sigma_YY is not None:
            from leadlag.core.signal import build_weights_minvar
            return build_weights_minvar(
                signal=signal,
                q=q_val,
                n_j=self.n_j,
                Sigma_YY=Sigma_YY,
                alpha=getattr(self, "minvar_alpha", 0.5),
                enforce_sign=False,
            )

        return signals.build_weights(
            signal=signal,
            q=q_val,
            n_j=self.n_j,
            weight_mode=self.weight_mode,
            enforce_sign=False,
        )

    def get_audit_context(self) -> AuditContext:
        """Return metadata required by ComplianceAuditor.

        Default implementation infers ``n_u`` / ``n_j`` from instance attributes
        (``self.n_u``, ``self.n_j``) if present.  Subclasses should override
        this method to expose model-specific audit information accurately.

        Returns:
            AuditContext populated with available model metadata.
        """
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
