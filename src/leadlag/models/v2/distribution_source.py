"""V2 distribution sources for the FallbackPolicy chain.

A ``DistributionSource`` is a single step in the ``on-demand -> file cache -> flat position``
resolution chain.  The ``FallbackPolicy`` iterates through the configured sources and
returns the first successful ``DistributionResult``.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from leadlag.config.schemas import ProductionV2RunConfig
from leadlag.data.pit_lake import MarketSnapshot
from leadlag.domain.portfolio import PortfolioDecision
from leadlag.models.v2 import VERSION
from leadlag.models.v2.audit_comparator import _compare_distribution, _run_safety_audits
from leadlag.models.v2.gap_io import _compute_ondemand, _gap_alerts_fatal
from leadlag.utils.gap_matrix_io import load_gap_matrices

logger = logging.getLogger(__name__)



@dataclass(frozen=True)
class DistributionResult:
    """Outcome of a ``DistributionSource.resolve`` call.

    Exactly one of the following must be true for a usable result:
      - ``is_flat`` is True and ``flat_decision`` is set.
      - ``mu_gap`` and ``Omega_gap`` are set (normal distribution acquired).

    ``is_available=False`` means the source could not provide a distribution and
    the next source in the chain should be tried.
    """

    mu_gap: np.ndarray | None = None
    Omega_gap: np.ndarray | None = None
    is_flat: bool = False
    flat_decision: PortfolioDecision | None = None
    source: str = ""
    alerts: list[str] | None = None
    is_available: bool = False

    def __post_init__(self) -> None:
        if self.alerts is None:
            object.__setattr__(self, "alerts", [])


class DistributionSource(ABC):
    """Abstract step in the V2 distribution resolution chain."""

    name: str = "abstract"

    def __init__(self, model: Any) -> None:
        self.model = model
        self.run_cfg: ProductionV2RunConfig = model.run_config
        self.n_j: int = model.n_j

    @abstractmethod
    def resolve(
        self,
        trade_date: str,
        df_exec: pd.DataFrame | None,
        current_prices: dict[str, float] | None,
        *,
        horizon: int = 1,
        snapshot: MarketSnapshot | None = None,
    ) -> DistributionResult:
        """Try to acquire a distribution for *trade_date*.

        Returns a ``DistributionResult``.  If the source is unavailable,
        ``result.is_available`` must be False so the policy can continue.
        """


class FileCacheDistributionSource(DistributionSource):
    """Load pre-computed gap matrices from the Step 2 file/SQLite cache."""

    name = "file_cache"

    def _gap_input_dir(self) -> Path | None:
        return getattr(self.model, "_current_gap_input_dir", None) or getattr(
            self.run_cfg, "gap_input_dir", None
        )

    def _pattern_kwargs(self, horizon: int) -> dict | None:
        if horizon == 1:
            return None
        return {"h": horizon}

    def resolve(
        self,
        trade_date: str,
        df_exec: pd.DataFrame | None,
        current_prices: dict[str, float] | None,
        *,
        horizon: int = 1,
        snapshot: MarketSnapshot | None = None,
    ) -> DistributionResult:
        gap_input_dir = self._gap_input_dir()
        if gap_input_dir is None:
            return DistributionResult(
                source=self.name,
                alerts=["gap_input_dir not specified."],
                is_available=False,
            )

        mu_pattern = "matrices/mu_gap_{date}.npy"
        omega_pattern = "matrices/omega_gap_{date}.npy"
        pattern_kwargs = self._pattern_kwargs(horizon)
        if horizon != 1:
            mu_pattern = self.run_cfg.mh_mu_file_pattern_h
            omega_pattern = self.run_cfg.mh_omega_file_pattern_h

        try:
            mu_gap, omega_gap, alerts = load_gap_matrices(
                gap_input_dir,
                trade_date,
                mu_pattern=mu_pattern,
                omega_pattern=omega_pattern,
                pattern_kwargs=pattern_kwargs,
                n_j=self.n_j,
                strict=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] File cache load raised %s", trade_date, exc)
            return DistributionResult(
                source=self.name,
                alerts=[f"File cache load raised {exc}"],
                is_available=False,
            )

        if mu_gap is None or omega_gap is None or _gap_alerts_fatal(alerts):
            return DistributionResult(
                source=self.name,
                alerts=alerts or ["Gap file cache missing/invalid."],
                is_available=False,
            )

        # Optional shadow validation against on-demand.
        shadow = getattr(self.run_cfg, "shadow_ondemand_validation", False)
        if (
            shadow
            and self.model._blpx_model is not None
            and df_exec is not None
            and current_prices is not None
        ):
            try:
                mu_ondemand, omega_ondemand = _compute_ondemand(
                    self.model,
                    trade_date=trade_date,
                    df_exec=df_exec,
                    current_prices=current_prices,
                    horizon=horizon,
                    snapshot=snapshot,
                )
                _compare_distribution(
                    f"{trade_date}:h{horizon}",
                    mu_gap,
                    omega_gap,
                    mu_ondemand,
                    omega_ondemand,
                )
            except Exception as exc:  # noqa: BLE001
                alerts.append(f"Shadow on-demand validation failed: {exc}")
                logger.warning("[%s] Shadow validation failed: %s", trade_date, exc)

        return DistributionResult(
            mu_gap=mu_gap,
            Omega_gap=omega_gap,
            source=self.name,
            alerts=alerts or [],
            is_available=True,
        )


class OnDemandDistributionSource(DistributionSource):
    """Compute ``(mu_gap, Omega_gap)`` on-demand from 9:10 prices and BLPX."""

    name = "on_demand"

    def resolve(
        self,
        trade_date: str,
        df_exec: pd.DataFrame | None,
        current_prices: dict[str, float] | None,
        *,
        horizon: int = 1,
        snapshot: MarketSnapshot | None = None,
    ) -> DistributionResult:
        if getattr(self.run_cfg, "ondemand_fallback_enabled", True) is False:
            return DistributionResult(
                source=self.name,
                alerts=["ondemand_fallback_enabled is False."],
                is_available=False,
            )

        if self.model._blpx_model is None:
            return DistributionResult(
                source=self.name,
                alerts=["blpx_model not available for on-demand computation."],
                is_available=False,
            )

        if df_exec is None or current_prices is None:
            return DistributionResult(
                source=self.name,
                alerts=["df_exec and current_prices are required for on-demand computation."],
                is_available=False,
            )

        try:
            mu_gap, omega_gap = _compute_ondemand(
                self.model,
                trade_date=trade_date,
                df_exec=df_exec,
                current_prices=current_prices,
                horizon=horizon,
                snapshot=snapshot,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] On-demand computation failed: %s", trade_date, exc)
            return DistributionResult(
                source=self.name,
                alerts=[f"On-demand computation failed: {exc}"],
                is_available=False,
            )

        return DistributionResult(
            mu_gap=mu_gap,
            Omega_gap=omega_gap,
            source=self.name,
            alerts=[],
            is_available=True,
        )


class FlatPositionSource(DistributionSource):
    """Terminal source: return a flat (zero-weight) portfolio decision."""

    name = "flat_position"

    def _build_flat(
        self,
        trade_date: str,
        gap_input_dir: Path | None,
        prior_alerts: list[str] | None,
    ) -> PortfolioDecision:
        n_j = self.n_j
        run_cfg = self.run_cfg
        alerts = list(prior_alerts or [])
        fallback = {"gap_data_missing": True}
        dummy_scores = np.zeros(n_j)
        dummy_Omega = np.eye(n_j) * 0.01
        pit_binning = {
            "assigned_bin": "Medium",
            "threshold_low": float("nan"),
            "threshold_high": float("nan"),
            "multiplier": run_cfg.fallback_multiplier,
            "current_ir": 0.0,
            "history_count": 0,
            "fallback_flag": True,
        }
        logger.error(
            "[%s] All distribution sources failed. "
            "Returning flat position (w_final=0). No trading today.",
            trade_date,
        )
        alerts.append("All distribution sources failed. Flat position (w_final=0) returned.")
        return _run_safety_audits(
            w_final=np.zeros(n_j),
            scores=dummy_scores,
            mu_gap=np.zeros(n_j),
            Omega_gap=dummy_Omega,
            sigma_gap=np.ones(n_j) * 0.1,
            gap_input_dir=gap_input_dir,
            date_str=trade_date,
            signal_date=trade_date,
            run_cfg=run_cfg,
            fallback=fallback,
            pit_binning=pit_binning,
            alerts=alerts,
            pit_history_trade_dates=None,
            candidate="flat_position",
            version=VERSION,
        )

    def resolve(
        self,
        trade_date: str,
        df_exec: pd.DataFrame | None,
        current_prices: dict[str, float] | None,
        *,
        horizon: int = 1,
        snapshot: MarketSnapshot | None = None,
    ) -> DistributionResult:
        gap_input_dir = getattr(self.model, "_current_gap_input_dir", None)
        return DistributionResult(
            is_flat=True,
            flat_decision=self._build_flat(trade_date, gap_input_dir, []),
            source=self.name,
            alerts=[],
            is_available=True,
        )
