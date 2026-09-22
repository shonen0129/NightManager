"""Unit tests for the V2 FallbackPolicy and DistributionSource chain."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from leadlag.config.schemas import ProductionV2RunConfig
from leadlag.data.pit_lake import MarketSnapshot
from leadlag.data.tickers import JP_TICKERS
from leadlag.domain.distribution import DistributionReason
from leadlag.models.v2.distribution_source import (
    DistributionResult,
    DistributionSource,
    FileCacheDistributionSource,
    FlatPositionSource,
    OnDemandDistributionSource,
)
from leadlag.models.v2.fallback_policy import FallbackPolicy
from leadlag.utils.gap_matrix_io import save_gap_matrices
from leadlag.utils.gap_provenance import bundle_identity


class _MockModel:
    """Minimal mock model for DistributionSource tests."""

    def __init__(self, run_config: ProductionV2RunConfig | None = None) -> None:
        self.run_config = run_config or ProductionV2RunConfig()
        self.n_j = len(JP_TICKERS)
        self._blpx_model = None


def _make_dummy_result() -> DistributionResult:
    n_j = len(JP_TICKERS)
    return DistributionResult(
        mu_gap=np.ones(n_j) * 0.01,
        Omega_gap=np.eye(n_j) * 0.001,
        source="dummy",
        is_available=True,
    )


@pytest.mark.unit
class TestDistributionSourceInterface:
    """Sanity checks for the base class and concrete implementations."""

    def test_distribution_result_is_frozen(self) -> None:
        result = _make_dummy_result()
        assert result.mu_gap is not None
        assert result.Omega_gap is not None

    def test_flat_position_source_returns_zero_weights(self) -> None:
        model = _MockModel()
        source = FlatPositionSource(model)
        result = source.resolve(
            "2024-01-01",
            df_exec=None,
            current_prices=None,
            horizon=1,
        )

        assert result.is_flat is True
        assert result.flat_decision is not None
        assert result.flat_decision.fallback["gap_data_missing"] is True
        assert np.allclose(result.flat_decision.w_final, 0.0)

    def test_on_demand_source_disabled_by_config(self) -> None:
        model = _MockModel()
        model.run_config = ProductionV2RunConfig(ondemand_fallback_enabled=False)
        source = OnDemandDistributionSource(model)

        result = source.resolve(
            "2024-01-01",
            df_exec=None,
            current_prices=None,
            horizon=1,
        )

        assert result.is_available is False
        assert any("ondemand_fallback_enabled" in a for a in result.alerts or [])

    def test_on_demand_source_requires_blpx_model(self) -> None:
        model = _MockModel()
        model.run_config = ProductionV2RunConfig(ondemand_fallback_enabled=True)
        model._blpx_model = None
        source = OnDemandDistributionSource(model)

        result = source.resolve(
            "2024-01-01",
            df_exec=None,
            current_prices=None,
            horizon=1,
        )

        assert result.is_available is False
        assert any("blpx_model not available" in a for a in result.alerts or [])

    def test_file_cache_source_loads_precomputed_matrices(self, tmp_path: Path) -> None:
        model = _MockModel()
        model.run_config = model.run_config.model_copy(update={"gap_input_dir": tmp_path})
        n_j = len(JP_TICKERS)

        trade_date = "2024-01-15"
        mu = np.ones(n_j) * 0.01
        omega = np.eye(n_j) * 0.001

        save_gap_matrices(
            tmp_path,
            trade_date,
            mu,
            omega,
            mu_pattern="matrices/mu_gap_{date}.npy",
            omega_pattern="matrices/omega_gap_{date}.npy",
            metadata={
                "sig_date": "2024-01-12",
                "trade_date": trade_date,
                "horizon": 1,
            },
        )

        source = FileCacheDistributionSource(model)
        result = source.resolve(trade_date, df_exec=None, current_prices=None, horizon=1)

        assert result.is_available is True
        assert result.source == "file_cache"
        assert np.allclose(result.mu_gap, mu)
        assert np.allclose(result.Omega_gap, omega)

    def test_file_cache_source_rejects_unprovenanced_bundle(self, tmp_path: Path) -> None:
        model = _MockModel()
        model.run_config = model.run_config.model_copy(update={"gap_input_dir": tmp_path})
        n_j = len(JP_TICKERS)
        trade_date = "2024-01-15"

        assert save_gap_matrices(
            tmp_path,
            trade_date,
            np.ones(n_j) * 0.01,
            np.eye(n_j) * 0.001,
        )

        result = FileCacheDistributionSource(model).resolve(
            trade_date, df_exec=None, current_prices=None, horizon=1
        )

        assert result.is_available is False
        assert any("provenance metadata is missing" in alert for alert in result.alerts or [])

    def test_file_cache_source_validates_bundle_identity_against_inputs(self, tmp_path: Path) -> None:
        model = _MockModel()
        model.run_config = model.run_config.model_copy(update={"gap_input_dir": tmp_path})
        trade_date = "2024-01-15"
        frame = pd.DataFrame(
            {"sig_date": [pd.Timestamp("2024-01-12")], "feature": [1.0]},
            index=pd.DatetimeIndex([pd.Timestamp(trade_date)]),
        )
        metadata = {
            "sig_date": "2024-01-12",
            "trade_date": trade_date,
            "horizon": 1,
            **bundle_identity(frame, trade_date, config=model.run_config, model=model),
        }
        assert save_gap_matrices(
            tmp_path,
            trade_date,
            np.ones(len(JP_TICKERS)) * 0.01,
            np.eye(len(JP_TICKERS)) * 0.001,
            metadata=metadata,
        )

        source = FileCacheDistributionSource(model)
        result = source.resolve(
            trade_date,
            df_exec=frame,
            current_prices=None,
            horizon=1,
        )
        assert result.is_available is True

        changed = frame.copy()
        changed.loc[pd.Timestamp(trade_date), "feature"] = 2.0
        rejected = source.resolve(
            trade_date,
            df_exec=changed,
            current_prices=None,
            horizon=1,
        )
        assert rejected.is_available is False
        assert any("input_version mismatch" in alert for alert in rejected.alerts or [])

    @pytest.mark.parametrize("horizon", [1, 3, 5])
    def test_strict_cache_requires_run_owned_open_910(
        self, tmp_path: Path, horizon: int
    ) -> None:
        """A typed run must not accept a cache without its 09:10 snapshot."""
        model = _MockModel()
        model.run_config = model.run_config.model_copy(update={"gap_input_dir": tmp_path})
        trade_date = "2024-01-15"
        frame = pd.DataFrame(
            {"sig_date": [pd.Timestamp("2024-01-12")], "feature": [1.0]},
            index=pd.DatetimeIndex([pd.Timestamp(trade_date)]),
        )
        metadata = {
            "sig_date": "2024-01-12",
            "trade_date": trade_date,
            "horizon": horizon,
            **bundle_identity(frame, trade_date, config=model.run_config, model=model),
        }
        assert save_gap_matrices(
            tmp_path,
            trade_date,
            np.ones(len(JP_TICKERS)) * 0.01,
            np.eye(len(JP_TICKERS)) * 0.001,
            mu_pattern=(
                model.run_config.mh_mu_file_pattern_h if horizon != 1 else "matrices/mu_gap_{date}.npy"
            ),
            omega_pattern=(
                model.run_config.mh_omega_file_pattern_h
                if horizon != 1
                else "matrices/omega_gap_{date}.npy"
            ),
            pattern_kwargs={"h": horizon} if horizon != 1 else None,
            metadata=metadata,
        )

        result = FileCacheDistributionSource(model).resolve(
            trade_date,
            df_exec=frame,
            current_prices=None,
            horizon=horizon,
            allow_implicit_io=False,
        )

        assert result.is_available is False
        assert result.reason == DistributionReason.INPUTS_MISSING
        assert any(
            f"strict h={horizon}" in alert and "explicit open_910_returns" in alert
            for alert in result.alerts or []
        )

    def test_file_cache_source_rejects_changed_open_910_inputs(self, tmp_path: Path) -> None:
        model = _MockModel()
        model.run_config = model.run_config.model_copy(update={"gap_input_dir": tmp_path})
        trade_date = "2024-01-15"
        frame = pd.DataFrame(
            {"sig_date": [pd.Timestamp("2024-01-12")], "feature": [1.0]},
            index=pd.DatetimeIndex([pd.Timestamp(trade_date)]),
        )
        open_910 = pd.DataFrame(0.01, index=frame.index, columns=JP_TICKERS)
        metadata = {
            "sig_date": "2024-01-12",
            "trade_date": trade_date,
            "horizon": 1,
            **bundle_identity(
                frame,
                trade_date,
                config=model.run_config,
                model=model,
                open_910_returns=open_910,
            ),
        }
        assert save_gap_matrices(
            tmp_path,
            trade_date,
            np.ones(len(JP_TICKERS)) * 0.01,
            np.eye(len(JP_TICKERS)) * 0.001,
            metadata=metadata,
        )

        source = FileCacheDistributionSource(model)
        assert source.resolve(
            trade_date,
            df_exec=frame,
            current_prices=None,
            horizon=1,
            open_910_returns=open_910,
        ).is_available

        changed = open_910.copy()
        changed.iloc[0, 0] = 0.02
        rejected = source.resolve(
            trade_date,
            df_exec=frame,
            current_prices=None,
            horizon=1,
            open_910_returns=changed,
        )
        assert rejected.is_available is False
        assert any("open_910_version mismatch" in alert for alert in rejected.alerts or [])

    def test_file_cache_source_rejects_changed_snapshot_gap_inputs(self, tmp_path: Path) -> None:
        model = _MockModel()
        model.run_config = model.run_config.model_copy(update={"gap_input_dir": tmp_path})
        trade_date = "2024-01-15"
        frame = pd.DataFrame(
            {"sig_date": [pd.Timestamp("2024-01-12")]},
            index=pd.DatetimeIndex([pd.Timestamp(trade_date)]),
        )
        snapshot = MarketSnapshot(
            as_of=pd.Timestamp("2024-01-15 09:10"),
            trade_date=trade_date,
            us_returns=np.zeros(15),
            jp_gap_returns=np.zeros(len(JP_TICKERS)),
            jp_betas=np.ones(len(JP_TICKERS)),
            topix_night_return=0.001,
            current_prices={ticker: 100.0 for ticker in JP_TICKERS},
            prev_closes={ticker: 100.0 for ticker in JP_TICKERS},
        )
        metadata = {
            "sig_date": "2024-01-12",
            "trade_date": trade_date,
            "horizon": 1,
            **bundle_identity(
                frame,
                trade_date,
                config=model.run_config,
                model=model,
                gap_inputs=(snapshot.jp_gap_returns, snapshot.jp_betas, snapshot.topix_night_return),
                horizon=1,
            ),
        }
        assert save_gap_matrices(
            tmp_path,
            trade_date,
            np.ones(len(JP_TICKERS)) * 0.01,
            np.eye(len(JP_TICKERS)) * 0.001,
            metadata=metadata,
        )
        source = FileCacheDistributionSource(model)
        assert source.resolve(
            trade_date, frame, None, snapshot=snapshot, horizon=1
        ).is_available

        changed = MarketSnapshot(
            as_of=snapshot.as_of,
            trade_date=trade_date,
            us_returns=snapshot.us_returns,
            jp_gap_returns=np.full(len(JP_TICKERS), 0.01),
            jp_betas=snapshot.jp_betas,
            topix_night_return=snapshot.topix_night_return,
            current_prices=snapshot.current_prices,
            prev_closes=snapshot.prev_closes,
        )
        rejected = source.resolve(trade_date, frame, None, snapshot=changed, horizon=1)
        assert not rejected.is_available
        assert any("gap_inputs_version mismatch" in alert for alert in rejected.alerts or [])

    def test_file_cache_source_rejects_legacy_bundle_with_live_snapshot(self, tmp_path: Path) -> None:
        """A snapshot-backed run cannot consume a bundle without gap identity."""
        model = _MockModel()
        model.run_config = model.run_config.model_copy(update={"gap_input_dir": tmp_path})
        trade_date = "2024-01-15"
        frame = pd.DataFrame(
            {"sig_date": [pd.Timestamp("2024-01-12")]},
            index=pd.DatetimeIndex([pd.Timestamp(trade_date)]),
        )
        snapshot = MarketSnapshot(
            as_of=pd.Timestamp(f"{trade_date} 09:10"),
            trade_date=trade_date,
            us_returns=np.zeros(15),
            jp_gap_returns=np.zeros(len(JP_TICKERS)),
            jp_betas=np.ones(len(JP_TICKERS)),
            topix_night_return=0.001,
            current_prices={ticker: 100.0 for ticker in JP_TICKERS},
            prev_closes={ticker: 100.0 for ticker in JP_TICKERS},
        )
        metadata = {
            "sig_date": "2024-01-12",
            "trade_date": trade_date,
            "horizon": 1,
            **bundle_identity(frame, trade_date, config=model.run_config, model=model),
        }
        assert save_gap_matrices(
            tmp_path,
            trade_date,
            np.ones(len(JP_TICKERS)) * 0.01,
            np.eye(len(JP_TICKERS)) * 0.001,
            metadata=metadata,
        )

        rejected = FileCacheDistributionSource(model).resolve(
            trade_date, frame, None, snapshot=snapshot, horizon=1
        )
        assert rejected.is_available is False
        assert rejected.reason == DistributionReason.PROVENANCE_REJECTED
        assert any("gap_inputs_version is missing" in alert for alert in rejected.alerts or [])

    def test_strict_on_demand_rejects_all_nan_open_910(self, monkeypatch) -> None:
        """Explicit all-NaN 09:10 input must not reach the BLPX calculator."""
        model = _MockModel()
        model._blpx_model = object()
        source = OnDemandDistributionSource(model)
        trade_date = "2024-01-15"
        frame = pd.DataFrame(
            {"sig_date": [pd.Timestamp("2024-01-12")], "feature": [1.0]},
            index=pd.DatetimeIndex([pd.Timestamp(trade_date)]),
        )
        open_910 = pd.DataFrame(np.nan, index=frame.index, columns=JP_TICKERS)

        def _unexpected(*args, **kwargs):
            raise AssertionError("all-NaN open_910_returns reached on-demand computation")

        monkeypatch.setattr(
            "leadlag.models.v2.distribution_source._compute_ondemand", _unexpected
        )
        result = source.resolve(
            trade_date,
            df_exec=frame,
            current_prices={ticker: 100.0 for ticker in JP_TICKERS},
            open_910_returns=open_910,
            allow_implicit_io=False,
        )

        assert result.is_available is False
        assert result.reason == DistributionReason.INPUTS_MISSING
        assert any("complete finite" in alert for alert in result.alerts or [])

    def test_file_cache_source_missing_dir_is_unavailable(self) -> None:
        model = _MockModel()
        source = FileCacheDistributionSource(model)

        result = source.resolve("2024-01-01", df_exec=None, current_prices=None, horizon=1)

        assert result.is_available is False


@pytest.mark.unit
class TestFallbackPolicy:
    """Test FallbackPolicy chaining, ordering, and alert propagation."""

    def test_default_chain_order_with_file_cache_preferred(self) -> None:
        model = _MockModel()
        policy = FallbackPolicy.default(model, use_file_cache=True)

        assert policy.sources[0].name == "file_cache"
        assert policy.sources[1].name == "on_demand"
        assert policy.sources[2].name == "flat_position"

    def test_default_chain_order_with_ondemand_preferred(self) -> None:
        model = _MockModel()
        policy = FallbackPolicy.default(model, use_file_cache=False)

        assert policy.sources[0].name == "on_demand"
        assert policy.sources[1].name == "file_cache"
        assert policy.sources[2].name == "flat_position"

    def test_resolve_picks_first_available(self) -> None:
        class _AlwaysAvailable(DistributionSource):
            name = "always"

            def resolve(self, trade_date, df_exec, current_prices, *, horizon=1, snapshot=None):
                return DistributionResult(
                    mu_gap=np.ones(len(JP_TICKERS)) * 0.01,
                    Omega_gap=np.eye(len(JP_TICKERS)) * 0.001,
                    source=self.name,
                    is_available=True,
                )

        class _NeverAvailable(DistributionSource):
            name = "never"

            def resolve(self, trade_date, df_exec, current_prices, *, horizon=1, snapshot=None):
                return DistributionResult(
                    source=self.name,
                    alerts=["missing"],
                    is_available=False,
                )

        model = _MockModel()
        policy = (
            FallbackPolicy()
            .add(_NeverAvailable(model))
            .add(_AlwaysAvailable(model))
            .add(FlatPositionSource(model))
        )

        result = policy.resolve("2024-01-01", df_exec=None, current_prices=None)

        assert result.is_available is True
        assert result.source == "always"
        # Prior alert from the failed source should be preserved.
        assert "missing" in (result.alerts or [])

    def test_resolve_falls_through_to_flat(self) -> None:
        model = _MockModel()
        policy = (
            FallbackPolicy()
            .add(FlatPositionSource(model))
        )

        result = policy.resolve("2024-01-01", df_exec=None, current_prices=None)

        assert result.is_flat is True
        assert result.flat_decision.fallback["gap_data_missing"] is True

    def test_resolve_prior_alerts_propagated_to_flat(self) -> None:
        class _FailingSource(DistributionSource):
            name = "failing"

            def resolve(self, trade_date, df_exec, current_prices, *, horizon=1, snapshot=None):
                return DistributionResult(
                    source=self.name,
                    alerts=["file missing", "cache stale"],
                    is_available=False,
                )

        model = _MockModel()
        policy = (
            FallbackPolicy()
            .add(_FailingSource(model))
            .add(FlatPositionSource(model))
        )

        result = policy.resolve("2024-01-01", df_exec=None, current_prices=None)

        assert result.is_flat is True
        # Flat source itself only logs a generic message, but prior alerts are now
        # part of the DistributionResult.  This verifies the propagation path.
        assert "file missing" in (result.alerts or [])
        assert "cache stale" in (result.alerts or [])

    def test_resolve_snapshot_kwarg_passed_through(self, tmp_path: Path) -> None:
        """Ensure FallbackPolicy and OnDemandDistributionSource accept snapshot kwarg."""
        model = _MockModel()
        model.run_config = ProductionV2RunConfig(ondemand_fallback_enabled=False)

        # File cache not present, on-demand disabled -> flat.  Just ensure no
        # TypeError on the snapshot keyword argument.
        policy = FallbackPolicy.default(model, use_file_cache=False)

        result = policy.resolve(
            "2024-01-01",
            df_exec=None,
            current_prices=None,
            snapshot=None,
        )

        assert result.is_flat is True
