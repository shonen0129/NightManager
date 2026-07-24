"""Unit tests for _process_date_impl extracted from compute_gap_adjusted_distribution.py.

Tests focus on edge cases that motivated the refactoring:
- Early date dropping (corr_window guard)
- Missing omega_struct file handling
- Leakage check (sig_date >= trade_date)
- NaN/Inf detection in Omega_struct
- Accumulator mutation correctness
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools/production"))

from compute_gap_adjusted_distribution import (
    GapDistAccumulators,
    GapDistContext,
    _merge_accumulators,
    _process_date_impl,
    _process_date_worker,
)
from leadlag.data.tickers import JP_TICKERS


def _make_minimal_ctx(tmp_path: Path, df_exec: pd.DataFrame, model: MagicMock, **overrides) -> GapDistContext:
    """Build a minimal GapDistContext with sensible defaults for testing."""
    n_j = len(JP_TICKERS)
    defaults = dict(
        df_exec=df_exec,
        model=model,
        jp_gap=np.zeros((len(df_exec), n_j)),
        jp_beta=np.ones((len(df_exec), n_j)),
        topix_night=np.zeros(len(df_exec)),
        jp_res_returns_p3=np.zeros((len(df_exec), n_j)),
        v0_static=np.zeros(n_j),
        c_full_p3=np.eye(n_j),
        dist_in_dir=tmp_path / "dist_in",
        out_dir=tmp_path / "out",
        save_daily_m=False,
        save_mh=False,
        mh_horizons=[],
        mh_models={},
        mh_inputs={},
        save_rr=False,
        y_jp_target=np.zeros((len(df_exec), n_j)),
        weights_df=pd.DataFrame(columns=JP_TICKERS),
        turnover_map={},
        cfg={"costs": {"slippage_bps_per_side": 5.0}},
        c=0.7,
        b=0.6,
        bl_mh_enabled=False,
        bl_mh_horizons=(1, 3, 5),
        bl_mh_weights=(0.8, 0.1, 0.1),
        bl_mh_mu_pattern="matrices/mu_gap_h{h}_{date}.npy",
        bl_mh_omega_pattern="matrices/omega_gap_h{h}_{date}.npy",
        bl_cs_overlay_enabled=False,
        bl_cs_overlay_weight=0.05,
        bl_rr_pattern="matrices/rank_reversal_{date}.npy",
        bl_long_count=5,
        bl_short_count=5,
        bl_minvar_enabled=False,
        bl_minvar_alpha=0.5,
        bl_baseline_gross=2.0,
        bl_cost_bps_per_gross=10.0,
    )
    defaults.update(overrides)
    return GapDistContext(**defaults)


def _make_df_exec(n_rows: int = 100, corr_window: int = 60) -> pd.DataFrame:
    """Build a minimal df_exec with sig_date column."""
    dates = pd.bdate_range("2025-01-01", periods=n_rows)
    sig_dates = dates - pd.Timedelta(days=1)
    df = pd.DataFrame(index=dates)
    df["sig_date"] = sig_dates.values
    # Add jp_oc columns for rank reversal
    for tk in JP_TICKERS:
        df[f"jp_oc_{tk}"] = np.random.randn(n_rows) * 0.01
    return df


def _make_mock_model(n_j: int = 17, corr_window: int = 60, vol_adjusted_target: bool = True) -> MagicMock:
    """Build a mock model with attributes used by _process_date_impl."""
    model = MagicMock()
    model.n_j = n_j
    model.corr_window = corr_window
    model.vol_adjusted_target = vol_adjusted_target
    model.gap_open_coef = 0.7
    model.topix_beta_coef = 0.6
    # compute_blp_signal returns a dict with required keys
    model.compute_blp_signal.return_value = {
        "z_hat_j_t1": np.zeros(n_j),
        "sigma_Y_denorm": np.ones(n_j),
        "mu_Y": np.zeros(n_j),
        "sigma_Y": np.ones(n_j),
        "signal": np.zeros(n_j),
    }
    return model


class TestProcessDateDropped:
    """Test early date dropping via corr_window guard."""

    def test_date_before_corr_window_dropped(self, tmp_path):
        """Date at index < corr_window should increment dropped_count and return early."""
        df_exec = _make_df_exec(n_rows=100, corr_window=60)
        model = _make_mock_model(corr_window=60)
        ctx = _make_minimal_ctx(tmp_path, df_exec, model)
        acc = GapDistAccumulators(omega_gap_ticker_records={tk: [] for tk in JP_TICKERS})

        # Date at index 10 (< 60) should be dropped
        early_dt = df_exec.index[10]
        _process_date_impl(early_dt, ctx, acc)

        assert acc.dropped_count == 1
        assert len(acc.gap_long_records) == 0
        assert len(acc.portfolio_diagnostics_records) == 0

    def test_date_at_corr_window_not_dropped(self, tmp_path):
        """Date at index == corr_window should not be dropped (boundary check)."""
        df_exec = _make_df_exec(n_rows=100, corr_window=60)
        model = _make_mock_model(corr_window=60)
        # Create omega_struct file so processing doesn't fail on missing file
        dist_in = tmp_path / "dist_in" / "matrices"
        dist_in.mkdir(parents=True)
        dt = df_exec.index[60]
        dt_str = dt.strftime("%Y%m%d")
        np.save(dist_in / f"omega_struct_{dt_str}.npy", np.eye(17))

        ctx = _make_minimal_ctx(tmp_path, df_exec, model)
        acc = GapDistAccumulators(omega_gap_ticker_records={tk: [] for tk in JP_TICKERS})

        _process_date_impl(dt, ctx, acc)

        assert acc.dropped_count == 0


class TestProcessDateMissingData:
    """Test missing omega_struct file handling."""

    def test_missing_omega_struct_increments_missing_count(self, tmp_path):
        """When omega_struct file is missing and no fallback exists, missing_data_count increments."""
        df_exec = _make_df_exec(n_rows=100, corr_window=60)
        model = _make_mock_model(corr_window=60)
        # No omega_struct file created
        ctx = _make_minimal_ctx(tmp_path, df_exec, model)
        acc = GapDistAccumulators(omega_gap_ticker_records={tk: [] for tk in JP_TICKERS})

        dt = df_exec.index[60]
        _process_date_impl(dt, ctx, acc)

        assert acc.missing_data_count == 1
        assert len(acc.portfolio_diagnostics_records) == 0

    def test_fallback_excludes_psd_files(self, tmp_path):
        """P1 regression: omega_struct_psd_YYYYMMDD.npy must NOT be selected as fallback."""
        df_exec = _make_df_exec(n_rows=100, corr_window=60)
        model = _make_mock_model(corr_window=60)
        dist_in = tmp_path / "dist_in" / "matrices"
        dist_in.mkdir(parents=True)

        dt = df_exec.index[60]
        dt_str = dt.strftime("%Y%m%d")
        prev_str = df_exec.index[59].strftime("%Y%m%d")

        # Save a PSD file for the previous date (should be excluded by re.fullmatch)
        np.save(dist_in / f"omega_struct_psd_{prev_str}.npy", np.eye(17) * 2.0)
        # Save a regular file for the previous date (should be selected as fallback)
        np.save(dist_in / f"omega_struct_{prev_str}.npy", np.eye(17))

        ctx = _make_minimal_ctx(tmp_path, df_exec, model)
        acc = GapDistAccumulators(omega_gap_ticker_records={tk: [] for tk in JP_TICKERS})

        _process_date_impl(dt, ctx, acc)

        # Should NOT increment missing_data_count (fallback found)
        assert acc.missing_data_count == 0
        # Should produce records (fallback was used)
        assert len(acc.portfolio_diagnostics_records) == 1

    def test_date_not_in_df_exec_index_increments_missing_count(self, tmp_path):
        """P3a regression: get_indexer returning -1 must be treated as missing data, not dropped."""
        df_exec = _make_df_exec(n_rows=100, corr_window=60)
        model = _make_mock_model(corr_window=60)
        ctx = _make_minimal_ctx(tmp_path, df_exec, model)
        acc = GapDistAccumulators(omega_gap_ticker_records={tk: [] for tk in JP_TICKERS})

        # A date that doesn't exist in df_exec index
        dt = pd.Timestamp("2099-12-31")
        _process_date_impl(dt, ctx, acc)

        assert acc.missing_data_count == 1
        assert acc.dropped_count == 0
        assert len(acc.portfolio_diagnostics_records) == 0


class TestProcessDateLeakage:
    """Test leakage check: sig_date must be strictly before trade_date."""

    def test_leakage_detected_when_sig_date_equals_trade_date(self, tmp_path):
        """When sig_date == trade_date, leakage flags should be set."""
        df_exec = _make_df_exec(n_rows=100, corr_window=60)
        # Set sig_date equal to trade_date for index 60
        dt = df_exec.index[60]
        df_exec.loc[dt, "sig_date"] = dt  # Same date → leakage

        model = _make_mock_model(corr_window=60)
        dist_in = tmp_path / "dist_in" / "matrices"
        dist_in.mkdir(parents=True)
        dt_str = dt.strftime("%Y%m%d")
        np.save(dist_in / f"omega_struct_{dt_str}.npy", np.eye(17))

        ctx = _make_minimal_ctx(tmp_path, df_exec, model)
        acc = GapDistAccumulators(omega_gap_ticker_records={tk: [] for tk in JP_TICKERS})

        _process_date_impl(dt, ctx, acc)

        assert acc.leakage_violation is True
        assert acc.all_dates_audit is False

    def test_no_leakage_when_sig_date_before_trade_date(self, tmp_path):
        """When sig_date < trade_date, leakage flags should remain clean."""
        df_exec = _make_df_exec(n_rows=100, corr_window=60)
        model = _make_mock_model(corr_window=60)
        dist_in = tmp_path / "dist_in" / "matrices"
        dist_in.mkdir(parents=True)
        dt = df_exec.index[60]
        dt_str = dt.strftime("%Y%m%d")
        np.save(dist_in / f"omega_struct_{dt_str}.npy", np.eye(17))

        ctx = _make_minimal_ctx(tmp_path, df_exec, model)
        acc = GapDistAccumulators(omega_gap_ticker_records={tk: [] for tk in JP_TICKERS})

        _process_date_impl(dt, ctx, acc)

        assert acc.leakage_violation is False
        assert acc.all_dates_audit is True


class TestProcessDateNaNInf:
    """Test NaN/Inf detection in Omega_struct."""

    def test_nan_in_omega_struct_increments_nan_inf_count(self, tmp_path):
        """When Omega_struct contains NaN, nan_inf_count increments and function returns early."""
        df_exec = _make_df_exec(n_rows=100, corr_window=60)
        model = _make_mock_model(corr_window=60)
        dist_in = tmp_path / "dist_in" / "matrices"
        dist_in.mkdir(parents=True)
        dt = df_exec.index[60]
        dt_str = dt.strftime("%Y%m%d")
        omega_bad = np.eye(17)
        omega_bad[0, 0] = np.nan
        np.save(dist_in / f"omega_struct_{dt_str}.npy", omega_bad)

        ctx = _make_minimal_ctx(tmp_path, df_exec, model)
        acc = GapDistAccumulators(omega_gap_ticker_records={tk: [] for tk in JP_TICKERS})

        _process_date_impl(dt, ctx, acc)

        assert acc.nan_inf_count == 1
        assert len(acc.portfolio_diagnostics_records) == 0


class TestProcessDateAccumulators:
    """Test that accumulators are correctly mutated for valid dates."""

    def test_valid_date_produces_records(self, tmp_path):
        """A valid date with all data should produce records in all accumulators."""
        df_exec = _make_df_exec(n_rows=100, corr_window=60)
        model = _make_mock_model(corr_window=60)
        dist_in = tmp_path / "dist_in" / "matrices"
        dist_in.mkdir(parents=True)
        dt = df_exec.index[60]
        dt_str = dt.strftime("%Y%m%d")
        np.save(dist_in / f"omega_struct_{dt_str}.npy", np.eye(17))

        ctx = _make_minimal_ctx(tmp_path, df_exec, model)
        acc = GapDistAccumulators(omega_gap_ticker_records={tk: [] for tk in JP_TICKERS})

        _process_date_impl(dt, ctx, acc)

        assert acc.dropped_count == 0
        assert acc.missing_data_count == 0
        assert acc.nan_inf_count == 0
        assert len(acc.gap_daily_records) == 1
        assert len(acc.dist_daily_records) == 1
        assert len(acc.omega_gap_daily_records) == 1
        assert len(acc.portfolio_diagnostics_records) == 1
        # 17 tickers → 17 long records per date
        assert len(acc.gap_long_records) == 17
        assert len(acc.dist_long_records) == 17
        # Ticker records
        for tk in JP_TICKERS:
            assert len(acc.omega_gap_ticker_records[tk]) == 1

    def test_multiple_dates_accumulate(self, tmp_path):
        """Processing multiple dates should accumulate records correctly."""
        df_exec = _make_df_exec(n_rows=100, corr_window=60)
        model = _make_mock_model(corr_window=60)
        dist_in = tmp_path / "dist_in" / "matrices"
        dist_in.mkdir(parents=True)

        ctx = _make_minimal_ctx(tmp_path, df_exec, model)
        acc = GapDistAccumulators(omega_gap_ticker_records={tk: [] for tk in JP_TICKERS})

        for idx in [60, 61, 62]:
            dt = df_exec.index[idx]
            dt_str = dt.strftime("%Y%m%d")
            np.save(dist_in / f"omega_struct_{dt_str}.npy", np.eye(17))
            _process_date_impl(dt, ctx, acc)

        assert len(acc.portfolio_diagnostics_records) == 3
        assert len(acc.gap_daily_records) == 3
        assert len(acc.gap_long_records) == 3 * 17


class TestMergeAccumulators:
    """Test _merge_accumulators correctly combines two partial accumulators."""

    def test_merge_counters(self):
        """Numeric counters should add up."""
        a = GapDistAccumulators(dropped_count=3, missing_data_count=2, nan_inf_count=1)
        b = GapDistAccumulators(dropped_count=1, missing_data_count=5, nan_inf_count=0)
        _merge_accumulators(a, b)
        assert a.dropped_count == 4
        assert a.missing_data_count == 7
        assert a.nan_inf_count == 1

    def test_merge_min_values(self):
        """denominator_min_overall should take the minimum."""
        a = GapDistAccumulators(denominator_min_overall=0.5, symmetry_max_err_gap=1e-10)
        b = GapDistAccumulators(denominator_min_overall=0.3, symmetry_max_err_gap=5e-10)
        _merge_accumulators(a, b)
        assert a.denominator_min_overall == 0.3
        assert a.symmetry_max_err_gap == 5e-10

    def test_merge_booleans(self):
        """leakage_violation should OR, all_dates_audit should AND."""
        a = GapDistAccumulators(leakage_violation=False, all_dates_audit=True)
        b = GapDistAccumulators(leakage_violation=True, all_dates_audit=True)
        _merge_accumulators(a, b)
        assert a.leakage_violation is True
        assert a.all_dates_audit is True

        a2 = GapDistAccumulators(leakage_violation=False, all_dates_audit=True)
        b2 = GapDistAccumulators(leakage_violation=False, all_dates_audit=False)
        _merge_accumulators(a2, b2)
        assert a2.leakage_violation is False
        assert a2.all_dates_audit is False

    def test_merge_lists(self):
        """Record lists should extend."""
        a = GapDistAccumulators(
            gap_long_records=[{"a": 1}],
            portfolio_diagnostics_records=[{"x": 1}],
        )
        b = GapDistAccumulators(
            gap_long_records=[{"a": 2}, {"a": 3}],
            portfolio_diagnostics_records=[{"x": 2}],
        )
        _merge_accumulators(a, b)
        assert len(a.gap_long_records) == 3
        assert len(a.portfolio_diagnostics_records) == 2

    def test_merge_ticker_records(self):
        """omega_gap_ticker_records should merge per-ticker lists."""
        a = GapDistAccumulators(omega_gap_ticker_records={JP_TICKERS[0]: [{"d": 1}]})
        b = GapDistAccumulators(omega_gap_ticker_records={JP_TICKERS[0]: [{"d": 2}], JP_TICKERS[1]: [{"d": 3}]})
        _merge_accumulators(a, b)
        assert len(a.omega_gap_ticker_records[JP_TICKERS[0]]) == 2
        assert len(a.omega_gap_ticker_records[JP_TICKERS[1]]) == 1


class TestProcessDateWorker:
    """Test _process_date_worker returns an isolated accumulator."""

    def test_worker_returns_local_accumulator(self, tmp_path):
        """Worker should return a GapDistAccumulators with results for one date."""
        df_exec = _make_df_exec(n_rows=100, corr_window=60)
        model = _make_mock_model(corr_window=60)
        dist_in = tmp_path / "dist_in" / "matrices"
        dist_in.mkdir(parents=True)
        dt = df_exec.index[60]
        dt_str = dt.strftime("%Y%m%d")
        np.save(dist_in / f"omega_struct_{dt_str}.npy", np.eye(17))

        ctx = _make_minimal_ctx(tmp_path, df_exec, model)
        result = _process_date_worker(dt, ctx)

        assert isinstance(result, GapDistAccumulators)
        assert result.dropped_count == 0
        assert len(result.portfolio_diagnostics_records) == 1

    def test_worker_does_not_mutate_shared_acc(self, tmp_path):
        """Worker should not mutate any external accumulator."""
        df_exec = _make_df_exec(n_rows=100, corr_window=60)
        model = _make_mock_model(corr_window=60)
        dist_in = tmp_path / "dist_in" / "matrices"
        dist_in.mkdir(parents=True)
        dt = df_exec.index[60]
        dt_str = dt.strftime("%Y%m%d")
        np.save(dist_in / f"omega_struct_{dt_str}.npy", np.eye(17))

        ctx = _make_minimal_ctx(tmp_path, df_exec, model)
        shared_acc = GapDistAccumulators(omega_gap_ticker_records={tk: [] for tk in JP_TICKERS})

        result = _process_date_worker(dt, ctx)

        # Shared acc should be untouched
        assert shared_acc.dropped_count == 0
        assert len(shared_acc.portfolio_diagnostics_records) == 0
        # Result should have the data
        assert len(result.portfolio_diagnostics_records) == 1

    def test_worker_merge_roundtrip_matches_serial(self, tmp_path):
        """Processing 3 dates via worker+merge should match serial processing."""
        df_exec = _make_df_exec(n_rows=100, corr_window=60)
        model = _make_mock_model(corr_window=60)
        dist_in = tmp_path / "dist_in" / "matrices"
        dist_in.mkdir(parents=True)

        ctx = _make_minimal_ctx(tmp_path, df_exec, model)

        # Serial
        serial_acc = GapDistAccumulators(omega_gap_ticker_records={tk: [] for tk in JP_TICKERS})
        for idx in [60, 61, 62]:
            dt = df_exec.index[idx]
            dt_str = dt.strftime("%Y%m%d")
            np.save(dist_in / f"omega_struct_{dt_str}.npy", np.eye(17))
            _process_date_impl(dt, ctx, serial_acc)

        # Worker + merge
        parallel_acc = GapDistAccumulators(omega_gap_ticker_records={tk: [] for tk in JP_TICKERS})
        for idx in [60, 61, 62]:
            dt = df_exec.index[idx]
            partial = _process_date_worker(dt, ctx)
            _merge_accumulators(parallel_acc, partial)

        assert parallel_acc.dropped_count == serial_acc.dropped_count
        assert parallel_acc.missing_data_count == serial_acc.missing_data_count
        assert len(parallel_acc.portfolio_diagnostics_records) == len(serial_acc.portfolio_diagnostics_records)
        assert len(parallel_acc.gap_long_records) == len(serial_acc.gap_long_records)
        assert len(parallel_acc.gap_daily_records) == len(serial_acc.gap_daily_records)
