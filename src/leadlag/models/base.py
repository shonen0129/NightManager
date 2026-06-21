"""Abstract Base Class for strategy models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


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

    p0_weight: float = 0.5
    """Ensemble weight for Raw-PCA (Production PCA) signal."""

    p3_weight: float = 0.5
    """Ensemble weight for Residual-PCA (Residual target PCA) signal."""

    p4_weight: float = 0.0
    """Ensemble weight for P4 (US-residualized) signal."""

    extra: dict = field(default_factory=dict)
    """Model-specific auxiliary metadata (arbitrary key-value pairs)."""


class BaseModel(ABC):
    """Abstract Base Class for Lead-Lag strategy models."""

    @abstractmethod
    def predict_signals(self, df_exec: pd.DataFrame) -> dict[str, np.ndarray]:
        """Generate raw signals from the execution dataset.

        Args:
            df_exec: Execution DataFrame.

        Returns:
            Dict containing the final signal array (shape N_J,) and other component signals.
        """
        pass

    @abstractmethod
    def build_weights(self, signals: np.ndarray) -> np.ndarray:
        """Construct portfolio weights from signals.

        Args:
            signals: Signal array of shape (n_j,).

        Returns:
            Weight array of shape (n_j,).
        """
        pass

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
            p0_weight=getattr(self, "p0_weight", 0.5),
            p3_weight=getattr(self, "p3_weight", 0.5),
            p4_weight=getattr(self, "p4_weight", 0.0),
        )
