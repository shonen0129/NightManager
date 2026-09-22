"""V2 distribution sources for the FallbackPolicy chain.

A ``DistributionSource`` is a single step in the ``on-demand -> file cache -> flat position``
resolution chain.  The ``FallbackPolicy`` iterates through the configured sources and
returns the first successful ``DistributionResult``.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from leadlag.config.schemas import ProductionV2RunConfig
from leadlag.data.intraday_inputs import has_usable_execution_prices
from leadlag.data.pit_lake import MarketSnapshot
from leadlag.data.tickers import JP_TICKERS
from leadlag.domain.distribution import (
    DistributionAttempt,
    DistributionReason,
    DistributionResult,
    DistributionStatus,
)
from leadlag.domain.portfolio import PortfolioDecision
from leadlag.models.v2 import VERSION
from leadlag.models.v2.audit_comparator import _compare_distribution, _run_safety_audits
from leadlag.models.v2.gap_io import (
    _compute_ondemand,
    _extract_gap_inputs,
    _extract_horizon_snapshot_inputs,
    _gap_alerts_fatal,
)
from leadlag.utils.distribution_provenance import validate_distribution_provenance
from leadlag.utils.gap_matrix_io import load_gap_bundle
from leadlag.utils.gap_provenance import bundle_identity, validate_bundle_identity
from leadlag.utils.timestamps import normalize_jst_date

logger = logging.getLogger(__name__)

__all__ = [
    "DistributionResult",
    "DistributionSource",
    "FileCacheDistributionSource",
    "FlatPositionSource",
    "OnDemandDistributionSource",
]


class DistributionSource(ABC):
    """Abstract step in the V2 distribution resolution chain."""

    name: str = "abstract"

    def __init__(self, model: Any, *, gap_input_dir: Path | None = None, mu_pattern: str | None = None, omega_pattern: str | None = None) -> None:
        self.model = model
        self.run_cfg: ProductionV2RunConfig = model.run_config
        self.n_j: int = model.n_j
        self.gap_input_dir = gap_input_dir if gap_input_dir is not None else self.run_cfg.gap_input_dir
        self.mu_pattern = mu_pattern
        self.omega_pattern = omega_pattern

    @abstractmethod
    def resolve(
        self,
        trade_date: str,
        df_exec: pd.DataFrame | None,
        current_prices: dict[str, float] | None,
        *,
        horizon: int = 1,
        snapshot: MarketSnapshot | None = None,
        open_910_returns: pd.DataFrame | None = None,
        allow_implicit_io: bool = True,
    ) -> DistributionResult:
        """Try to acquire a distribution for *trade_date*.

        Returns a ``DistributionResult``.  If the source is unavailable,
        ``result.is_available`` must be False so the policy can continue.
        """


def _metadata_from_df_exec(
    df_exec: pd.DataFrame | None,
    trade_date: str,
    horizon: int,
) -> dict[str, Any] | None:
    """Build on-demand provenance from the actual PIT execution row."""
    if df_exec is None or trade_date not in df_exec.index:
        return None
    row = df_exec.loc[trade_date]
    if isinstance(row, pd.DataFrame):
        return None
    has_sig_date = "sig_date" in row.index
    has_signal_date = "signal_date" in row.index
    raw_signal_date = row["sig_date"] if has_sig_date else row.get("signal_date")
    if (
        raw_signal_date is None
        or not _is_scalar_provenance(raw_signal_date)
        or _is_missing_scalar(raw_signal_date)
    ):
        return None
    try:
        signal_date = normalize_jst_date(raw_signal_date)
        trade_dt = normalize_jst_date(trade_date)
    except (TypeError, ValueError, OverflowError, OSError):
        return None
    if _is_missing_scalar(signal_date) or _is_missing_scalar(trade_dt):
        return None
    metadata: dict[str, Any] = {
        "sig_date": signal_date.strftime("%Y-%m-%d"),
        "trade_date": trade_dt.strftime("%Y-%m-%d"),
        "horizon": int(horizon),
        "source": "df_exec",
    }
    if has_signal_date:
        metadata["signal_date"] = row["signal_date"]
    return metadata


def _is_missing_scalar(value: Any) -> bool:
    """Return whether *value* is a scalar missing value.

    ``pd.isna`` returns an array for list-like input; converting that result
    directly to ``bool`` raises an ambiguous-truth-value error.  Provenance is
    scalar by contract, so list-like values are treated as invalid by the
    caller rather than allowed to escape into date parsing.
    """
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(missing, (bool, np.bool_)) and bool(missing)


def _is_scalar_provenance(value: Any) -> bool:
    """Return whether a provenance field contains one scalar value."""
    return not isinstance(value, (list, tuple, set, dict, pd.Series, pd.Index, np.ndarray))


def _validate_distribution_metadata(
    metadata: dict[str, Any] | None,
    trade_date: str,
    horizon: int,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Validate signal date and identity fields for one distribution."""
    normalized, error = validate_distribution_provenance(
        metadata,
        trade_date,
        horizon,
        label=f"Distribution provenance for h={horizon}",
    )
    return normalized, [error] if error else []


def _classify_cache_alerts(alerts: list[str]) -> tuple[DistributionStatus, DistributionReason]:
    """Translate cache-loader diagnostics into stable source-level reasons.

    ``load_gap_bundle`` still exposes human-readable alerts for logs and
    compatibility.  The fallback policy must consume this typed classification
    instead of inspecting those strings itself.
    """
    text = "; ".join(alerts).lower()
    if (
        "provenance" in text
        or "signal date" in text
        or "signal_date" in text
        or "gap bundle" in text
        or "identity" in text
        or " mismatch" in text
    ):
        return DistributionStatus.REJECTED, DistributionReason.PROVENANCE_REJECTED
    if "missing" in text or "not found" in text or "not specified" in text:
        return DistributionStatus.UNAVAILABLE, DistributionReason.CACHE_MISSING
    return DistributionStatus.UNAVAILABLE, DistributionReason.CACHE_INVALID


class FileCacheDistributionSource(DistributionSource):
    """Load pre-computed gap matrices from the Step 2 file/SQLite cache."""

    name = "file_cache"

    def _gap_input_dir(self) -> Path | None:
        return self.gap_input_dir

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
        open_910_returns: pd.DataFrame | None = None,
        allow_implicit_io: bool = True,
    ) -> DistributionResult:
        gap_input_dir = self._gap_input_dir()
        if gap_input_dir is None:
            return DistributionResult(
                source=self.name,
                alerts=["gap_input_dir not specified."],
                status=DistributionStatus.UNAVAILABLE,
                reason=DistributionReason.CACHE_NOT_CONFIGURED,
                horizon=horizon,
            )

        mu_pattern = "matrices/mu_gap_{date}.npy"
        omega_pattern = "matrices/omega_gap_{date}.npy"
        pattern_kwargs = self._pattern_kwargs(horizon)
        if horizon != 1:
            mu_pattern = self.run_cfg.mh_mu_file_pattern_h
            omega_pattern = self.run_cfg.mh_omega_file_pattern_h

        mu_pattern = self.mu_pattern or mu_pattern
        omega_pattern = self.omega_pattern or omega_pattern
        if (
            df_exec is not None
            and not allow_implicit_io
            and not has_usable_execution_prices(
                open_910_returns, df_exec, JP_TICKERS, required_index=[trade_date]
            )
        ):
            # Require an owned observation frame and usable prices. Explicit
            # missing observations may use daily opens; their NaNs remain in
            # the fingerprint and cannot match a different observed frame.
            return DistributionResult(
                source=self.name,
                alerts=[
                    f"strict h={horizon} cache resolution requires a complete finite "
                    "explicit open_910_returns or finite positive fallback opens."
                ],
                status=DistributionStatus.UNAVAILABLE,
                reason=DistributionReason.INPUTS_MISSING,
                horizon=horizon,
            )
        expected_identity = None
        expected_gap_identity = None
        if snapshot is not None and df_exec is None:
            # A live snapshot must be tied to the run-owned execution frame so
            # the cache can be checked against the same point-in-time inputs.
            # Without the frame, accepting a cache would silently ignore the
            # supplied snapshot and could reuse stale gap matrices.
            return DistributionResult(
                source=self.name,
                alerts=[
                    "snapshot-backed cache resolution requires the run-owned df_exec "
                    "for gap provenance validation."
                ],
                status=DistributionStatus.REJECTED,
                reason=DistributionReason.PROVENANCE_REJECTED,
                horizon=horizon,
            )
        if df_exec is not None:
            try:
                gap_inputs = None
                if snapshot is not None and horizon > 1:
                    gap_inputs = _extract_horizon_snapshot_inputs(
                        df_exec, trade_date, horizon, snapshot
                    )
                elif snapshot is not None:
                    gap_inputs = _extract_gap_inputs(
                        df_exec, trade_date, current_prices or {}, snapshot=snapshot
                    )
                if snapshot is not None and gap_inputs is None:
                    raise ValueError("point-in-time gap inputs are unavailable")
                expected_identity = bundle_identity(
                    df_exec,
                    trade_date,
                    config=self.run_cfg,
                    model=self.model,
                    open_910_returns=open_910_returns,
                    horizon=horizon,
                )
                if gap_inputs is not None:
                    expected_gap_identity = {
                        **expected_identity,
                        "gap_inputs_version": bundle_identity(
                            df_exec,
                            trade_date,
                            config=self.run_cfg,
                            model=self.model,
                            gap_inputs=gap_inputs,
                            horizon=horizon,
                        )["gap_inputs_version"],
                    }
            except (TypeError, ValueError, KeyError) as exc:
                logger.warning("[%s] Could not build cache identity: %s", trade_date, exc)
                return DistributionResult(
                    source=self.name,
                    alerts=[f"Cache identity unavailable: {exc}"],
                    status=DistributionStatus.REJECTED,
                    reason=DistributionReason.PROVENANCE_REJECTED,
                    horizon=horizon,
                )
        try:
            mu_gap, omega_gap, metadata, alerts = load_gap_bundle(
                gap_input_dir,
                trade_date,
                mu_pattern=mu_pattern,
                omega_pattern=omega_pattern,
                pattern_kwargs=pattern_kwargs,
                n_j=self.n_j,
                strict=False,
                require_metadata=True,
                expected_identity=expected_identity,
                require_identity=expected_identity is not None,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] File cache load raised %s", trade_date, exc)
            status, reason = _classify_cache_alerts([str(exc)])
            return DistributionResult(
                source=self.name,
                alerts=[f"File cache load raised {exc}"],
                status=status,
                reason=reason,
                horizon=horizon,
            )

        if mu_gap is None or omega_gap is None or _gap_alerts_fatal(alerts):
            status, reason = _classify_cache_alerts(alerts or ["Gap file cache missing/invalid."])
            return DistributionResult(
                source=self.name,
                alerts=alerts or ["Gap file cache missing/invalid."],
                status=status,
                reason=reason,
                horizon=horizon,
            )

        # A live snapshot must be represented in the bundle identity.  Legacy
        # bundles remain readable for snapshot-free historical compatibility,
        # but they cannot be used when the caller supplies current PIT inputs.
        if expected_gap_identity is not None:
            if metadata is None or "gap_inputs_version" not in metadata:
                return DistributionResult(
                    source=self.name,
                    alerts=[
                        "Gap bundle gap_inputs_version is missing for a snapshot-backed "
                        "decision."
                    ],
                    status=DistributionStatus.REJECTED,
                    reason=DistributionReason.PROVENANCE_REJECTED,
                    horizon=horizon,
                )
            gap_identity_errors = validate_bundle_identity(
                metadata, expected_gap_identity, require=True
            )
            if gap_identity_errors:
                return DistributionResult(
                    source=self.name,
                    alerts=gap_identity_errors,
                    status=DistributionStatus.REJECTED,
                    reason=DistributionReason.PROVENANCE_REJECTED,
                    horizon=horizon,
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
                    open_910_returns=open_910_returns,
                    allow_implicit_io=allow_implicit_io,
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
            status=DistributionStatus.READY,
            reason=DistributionReason.RESOLVED,
            metadata=metadata,
            horizon=horizon,
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
        open_910_returns: pd.DataFrame | None = None,
        allow_implicit_io: bool = True,
    ) -> DistributionResult:
        if getattr(self.run_cfg, "ondemand_fallback_enabled", True) is False:
            return DistributionResult(
                source=self.name,
                alerts=["ondemand_fallback_enabled is False."],
                status=DistributionStatus.UNAVAILABLE,
                reason=DistributionReason.ON_DEMAND_DISABLED,
                horizon=horizon,
            )

        if self.model._blpx_model is None:
            return DistributionResult(
                source=self.name,
                alerts=["blpx_model not available for on-demand computation."],
                status=DistributionStatus.UNAVAILABLE,
                reason=DistributionReason.MODEL_UNAVAILABLE,
                horizon=horizon,
            )

        if df_exec is None or current_prices is None:
            return DistributionResult(
                source=self.name,
                alerts=["df_exec and current_prices are required for on-demand computation."],
                status=DistributionStatus.UNAVAILABLE,
                reason=DistributionReason.INPUTS_MISSING,
                horizon=horizon,
            )

        if (
            not allow_implicit_io
            and not has_usable_execution_prices(
                open_910_returns, df_exec, JP_TICKERS, required_index=[trade_date]
            )
        ):
            return DistributionResult(
                source=self.name,
                alerts=[
                    f"strict h={horizon} on-demand resolution requires a complete finite "
                    "explicit open_910_returns or finite positive fallback opens."
                ],
                status=DistributionStatus.UNAVAILABLE,
                reason=DistributionReason.INPUTS_MISSING,
                horizon=horizon,
            )

        try:
            mu_gap, omega_gap = _compute_ondemand(
                self.model,
                trade_date=trade_date,
                df_exec=df_exec,
                current_prices=current_prices,
                horizon=horizon,
                snapshot=snapshot,
                open_910_returns=open_910_returns,
                allow_implicit_io=allow_implicit_io,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] On-demand computation failed: %s", trade_date, exc)
            return DistributionResult(
                source=self.name,
                alerts=[f"On-demand computation failed: {exc}"],
                status=DistributionStatus.UNAVAILABLE,
                reason=DistributionReason.COMPUTATION_FAILED,
                horizon=horizon,
            )

        metadata, provenance_alerts = _validate_distribution_metadata(
            _metadata_from_df_exec(df_exec, trade_date, horizon), trade_date, horizon
        )
        if provenance_alerts:
            return DistributionResult(
                source=self.name,
                alerts=provenance_alerts,
                status=DistributionStatus.REJECTED,
                reason=DistributionReason.PROVENANCE_REJECTED,
                horizon=horizon,
            )

        return DistributionResult(
            mu_gap=mu_gap,
            Omega_gap=omega_gap,
            source=self.name,
            alerts=[],
            status=DistributionStatus.READY,
            reason=DistributionReason.RESOLVED,
            metadata=metadata,
            horizon=horizon,
        )


class FlatPositionSource(DistributionSource):
    """Terminal source: return a flat (zero-weight) portfolio decision."""

    name = "flat_position"

    def _build_flat(
        self,
        trade_date: str,
        gap_input_dir: Path | None,
        prior_alerts: list[str] | None,
        *,
        fallback: dict[str, Any] | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> PortfolioDecision:
        n_j = self.n_j
        run_cfg = self.run_cfg
        alerts = list(prior_alerts or [])
        fallback = dict(fallback or {"gap_data_missing": True})
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
            diagnostics=diagnostics,
        )

    def resolve_failure(
        self,
        trade_date: str,
        *,
        alerts: list[str] | None = None,
        fallback: dict[str, Any] | None = None,
        diagnostics: dict[str, Any] | None = None,
        reason: DistributionReason = DistributionReason.FLAT_FALLBACK,
        attempts: tuple[DistributionAttempt, ...] = (),
        horizon: int = 1,
    ) -> DistributionResult:
        """Build a terminal flat result while preserving the failure reason."""
        gap_input_dir = self.gap_input_dir
        return DistributionResult(
            is_flat=True,
            flat_decision=self._build_flat(
                trade_date,
                gap_input_dir,
                alerts,
                fallback=fallback,
                diagnostics=diagnostics,
            ),
            source=self.name,
            alerts=list(alerts or []),
            status=DistributionStatus.FLAT,
            reason=reason,
            attempts=attempts,
            horizon=horizon,
        )

    def resolve(
        self,
        trade_date: str,
        df_exec: pd.DataFrame | None,
        current_prices: dict[str, float] | None,
        *,
        horizon: int = 1,
        snapshot: MarketSnapshot | None = None,
        open_910_returns: pd.DataFrame | None = None,
        allow_implicit_io: bool = True,
    ) -> DistributionResult:
        gap_input_dir = self.gap_input_dir
        return DistributionResult(
            is_flat=True,
            flat_decision=self._build_flat(trade_date, gap_input_dir, []),
            source=self.name,
            alerts=[],
            status=DistributionStatus.FLAT,
            reason=DistributionReason.FLAT_FALLBACK,
            horizon=horizon,
        )
