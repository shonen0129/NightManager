"""Integration tests for the production v2 module.

These tests exercise the full pipeline end-to-end using synthetic data,
verifying that:
  - All public functions are importable from the src package
  - The pipeline produces numerically correct output
  - Fallback paths work correctly
  - The tools/ entry-point self-test passes
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make the package importable without installation
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.compliance.v2_auditor import run_leakage_audit, run_numerical_audit
from leadlag.config.schemas import ProductionV2RunConfig
from leadlag.core.portfolio import get_rolling_pit_bin, solve_baseline_style
from leadlag.data.tickers import JP_TICKERS
from leadlag.models.production_v2 import (
    BASELINE_GROSS,
    COST_BPS_PER_GROSS,
    LONG_COUNT,
    SHORT_COUNT,
    VERSION,
    generate_v2_production_portfolio,
    load_gap_matrices,
    load_pit_ir_history,
    parse_run_config,
)

N_J = len(JP_TICKERS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_gap_data(n_j: int = N_J, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Return a synthetic (mu_gap, Omega_gap) pair for testing."""
    rng = np.random.default_rng(seed)
    mu_gap = rng.normal(0.0, 0.01, n_j)
    A = rng.normal(0.0, 1.0, (n_j, n_j))
    Omega_gap = (A @ A.T) / n_j + np.eye(n_j) * 0.01
    return mu_gap, Omega_gap


# ---------------------------------------------------------------------------
# solve_baseline_style
# ---------------------------------------------------------------------------

class TestSolveBaselineStyle:
    def test_net_zero(self):
        scores = np.linspace(-1.0, 1.0, N_J)
        longs = np.argsort(scores)[-LONG_COUNT:]
        shorts = np.argsort(scores)[:SHORT_COUNT]
        w = solve_baseline_style(scores, longs, shorts, baseline_gross=BASELINE_GROSS)
        assert abs(np.sum(w)) < 1e-12

    def test_gross_equals_baseline(self):
        scores = np.linspace(-1.0, 1.0, N_J)
        longs = np.argsort(scores)[-LONG_COUNT:]
        shorts = np.argsort(scores)[:SHORT_COUNT]
        w = solve_baseline_style(scores, longs, shorts, baseline_gross=BASELINE_GROSS)
        assert abs(np.sum(np.abs(w)) - BASELINE_GROSS) < 1e-10

    def test_long_positive_short_negative(self):
        scores = np.linspace(-1.0, 1.0, N_J)
        longs = np.argsort(scores)[-LONG_COUNT:]
        shorts = np.argsort(scores)[:SHORT_COUNT]
        w = solve_baseline_style(scores, longs, shorts, baseline_gross=BASELINE_GROSS)
        assert (w[longs] >= 0.0).all()
        assert (w[shorts] <= 0.0).all()

    def test_non_selected_are_zero(self):
        scores = np.linspace(-1.0, 1.0, N_J)
        sorted_idx = np.argsort(scores)
        longs = sorted_idx[-LONG_COUNT:]
        shorts = sorted_idx[:SHORT_COUNT]
        w = solve_baseline_style(scores, longs, shorts, baseline_gross=BASELINE_GROSS)
        neither = np.setdiff1d(np.arange(N_J), np.concatenate([longs, shorts]))
        assert np.allclose(w[neither], 0.0)


# ---------------------------------------------------------------------------
# get_rolling_pit_bin
# ---------------------------------------------------------------------------

class TestGetRollingPitBin:
    def test_low_bin_returns_0_75(self):
        hist = np.linspace(0.0, 3.0, 500)
        b, lo, hi, m = get_rolling_pit_bin(hist, 0.5, rolling_window=252)
        assert b == "Low"
        assert abs(m - 0.75) < 1e-9

    def test_medium_bin_returns_1_0(self):
        hist = np.linspace(0.0, 3.0, 500)
        b, _, _, m = get_rolling_pit_bin(hist, 2.2, rolling_window=252)
        assert b == "Medium"
        assert abs(m - 1.0) < 1e-9

    def test_high_bin_returns_1_0(self):
        hist = np.linspace(0.0, 3.0, 500)
        b, _, _, m = get_rolling_pit_bin(hist, 3.5, rolling_window=252)
        assert b == "High"
        assert abs(m - 1.0) < 1e-9

    def test_insufficient_history_fallback(self):
        hist = np.linspace(0.0, 3.0, 500)
        b, lo, _, m = get_rolling_pit_bin(hist, 2.0, rolling_window=600)
        assert b == "Medium"
        assert np.isnan(lo)
        assert abs(m - 1.0) < 1e-9

    def test_empty_history_fallback(self):
        b, lo, hi, m = get_rolling_pit_bin(np.array([]), 1.0, rolling_window=252)
        assert b == "Medium"
        assert np.isnan(lo) and np.isnan(hi)


# ---------------------------------------------------------------------------
# load_pit_ir_history
# ---------------------------------------------------------------------------

class TestLoadPitIrHistory:
    """Tests for load_pit_ir_history column preference and PIT filtering."""

    def _write_diag_csv(self, tmp_path, rows):
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_csv(tmp_path / "portfolio_gap_distribution_diagnostics.csv", index=False)

    def test_prefers_baseline_cost_column(self, tmp_path):
        """When pred_ir_gap_baseline_cost exists, it is used over legacy column."""
        rows = []
        for i in range(300):
            d = f"2025-01-{i+1:02d}" if i < 30 else f"2025-02-{i-29+1:02d}"
            rows.append({
                "trade_date": f"2025-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}",
                "pred_ir_gap_exante_cost": float(i + 1) * 0.01,
                "pred_ir_gap_baseline_cost": float(i + 1) * 0.02,
            })
        self._write_diag_csv(tmp_path, rows)
        hist, alerts, dates = load_pit_ir_history(tmp_path, "2026-01-01")
        assert len(alerts) == 0
        # Should use baseline_cost column (0.02 multiplier)
        assert abs(hist[0] - 0.02) < 1e-9
        assert abs(hist[1] - 0.04) < 1e-9

    def test_falls_back_to_legacy_column_with_alert(self, tmp_path):
        """When pred_ir_gap_baseline_cost is absent, falls back to exante_cost and alerts."""
        rows = []
        for i in range(300):
            rows.append({
                "trade_date": f"2025-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}",
                "pred_ir_gap_exante_cost": float(i + 1) * 0.01,
            })
        self._write_diag_csv(tmp_path, rows)
        hist, alerts, dates = load_pit_ir_history(tmp_path, "2026-01-01")
        assert len(hist) > 0
        assert any("pred_ir_gap_baseline_cost" in a for a in alerts)
        assert abs(hist[0] - 0.01) < 1e-9

    def test_missing_file_returns_empty(self, tmp_path):
        """Missing diagnostics file returns empty array and alert."""
        hist, alerts, dates = load_pit_ir_history(tmp_path, "2026-01-01")
        assert len(hist) == 0
        assert any("missing" in a.lower() for a in alerts)

    def test_filters_future_dates(self, tmp_path):
        """Rows >= trade_date are excluded (PIT integrity)."""
        rows = [
            {"trade_date": "2025-06-01", "pred_ir_gap_baseline_cost": 0.1},
            {"trade_date": "2025-06-02", "pred_ir_gap_baseline_cost": 0.2},
            {"trade_date": "2025-06-03", "pred_ir_gap_baseline_cost": 0.3},
        ]
        self._write_diag_csv(tmp_path, rows)
        hist, alerts, dates = load_pit_ir_history(tmp_path, "2025-06-02")
        assert len(hist) == 1
        assert abs(hist[0] - 0.1) < 1e-9


# ---------------------------------------------------------------------------
# run_leakage_audit
# ---------------------------------------------------------------------------

class TestLeakageAudit:
    def test_valid_dates_pass(self):
        result = run_leakage_audit("2026-06-15", "2026-06-16")
        assert result["status"] == "PASSED"
        assert result["signal_date_strictly_before_trade_date"] is True

    def test_same_date_fails(self):
        result = run_leakage_audit("2026-06-16", "2026-06-16")
        assert result["status"] == "FAILED"
        assert result["signal_date_strictly_before_trade_date"] is False

    def test_future_signal_fails(self):
        result = run_leakage_audit("2026-06-17", "2026-06-16")
        assert result["status"] == "FAILED"


# ---------------------------------------------------------------------------
# run_numerical_audit
# ---------------------------------------------------------------------------

class TestNumericalAudit:
    def test_valid_inputs_pass(self):
        w = np.array([0.2] * 5 + [-0.2] * 5)
        scores = np.ones(10)
        Omega = np.eye(10) * 0.01
        result = run_numerical_audit(w, scores, Omega)
        assert result["status"] == "PASSED"
        assert result["weights_finite"] is True
        assert result["net_exposure_near_zero"] is True

    def test_nan_weights_fail(self):
        w = np.array([np.nan] * 10)
        result = run_numerical_audit(w, np.ones(10), np.eye(10))
        assert result["status"] == "FAILED"
        assert result["weights_finite"] is False

    def test_non_zero_net_fails(self):
        w = np.ones(10) * 0.1  # net = 1.0
        result = run_numerical_audit(w, np.ones(10), np.eye(10))
        assert result["status"] == "FAILED"
        assert result["net_exposure_near_zero"] is False

    def test_asymmetric_omega_fails(self):
        Omega = np.eye(10) * 0.01
        Omega[0, 1] = 999.0  # break symmetry
        w = np.array([0.2] * 5 + [-0.2] * 5)
        result = run_numerical_audit(w, np.ones(10), Omega)
        assert result["status"] == "FAILED"
        assert result["covariance_symmetric"] is False


# ---------------------------------------------------------------------------
# generate_v2_production_portfolio (integration, no real files)
# ---------------------------------------------------------------------------

class TestGenerateV2Portfolio:
    def test_flat_position_when_no_gap_dir(self, tmp_path):
        """When gap_input_dir is None the function returns flat position (w_final=0)."""
        result = generate_v2_production_portfolio(
            trade_date="2026-06-16",
            gap_input_dir=None,
            cfg={},
        )
        assert result["fallback"]["gap_data_missing"] is True
        assert np.allclose(result["w_final"], 0.0)

    def test_full_pipeline_with_synthetic_data(self, tmp_path):
        """Full pipeline runs with synthetic gap matrices and produces valid weights."""
        mu_gap, Omega_gap = _make_synthetic_gap_data()

        # Write synthetic gap matrices
        matrices_dir = tmp_path / "matrices"
        matrices_dir.mkdir()
        np.save(matrices_dir / "mu_gap_20260616.npy", mu_gap)
        np.save(matrices_dir / "omega_gap_20260616.npy", Omega_gap)

        result = generate_v2_production_portfolio(
            trade_date="2026-06-16",
            gap_input_dir=tmp_path,
            cfg={},
        )

        assert result["fallback"]["gap_data_missing"] is False
        w = result["w_final"]
        assert len(w) == N_J
        # Weights should be market-neutral
        assert abs(np.sum(w)) < 1e-8
        # Gross must be <= BASELINE_GROSS (may be reduced by RuleD mult)
        assert np.sum(np.abs(w)) <= BASELINE_GROSS + 1e-10
        # Exactly LONG_COUNT longs and SHORT_COUNT shorts
        assert int(np.sum(w > 1e-8)) == LONG_COUNT
        assert int(np.sum(w < -1e-8)) == SHORT_COUNT

    def test_summary_fields_present(self, tmp_path):
        """Summary dict contains all expected fields."""
        mu_gap, Omega_gap = _make_synthetic_gap_data()
        matrices_dir = tmp_path / "matrices"
        matrices_dir.mkdir()
        np.save(matrices_dir / "mu_gap_20260616.npy", mu_gap)
        np.save(matrices_dir / "omega_gap_20260616.npy", Omega_gap)

        result = generate_v2_production_portfolio(
            trade_date="2026-06-16",
            gap_input_dir=tmp_path,
            cfg={},
        )
        s = result["summary"]
        for key in [
            "trade_date", "candidate", "version",
            "long_count", "short_count",
            "target_gross", "target_net",
            "gross_multiplier", "pit_bin",
            "predicted_portfolio_ir",
            "expected_cost_bps", "herfindahl",
            "fallback_triggered",
        ]:
            assert key in s, f"Missing summary key: {key}"

    def test_pit_binning_keys_present(self, tmp_path):
        """pit_binning result contains required keys."""
        mu_gap, Omega_gap = _make_synthetic_gap_data()
        matrices_dir = tmp_path / "matrices"
        matrices_dir.mkdir()
        np.save(matrices_dir / "mu_gap_20260616.npy", mu_gap)
        np.save(matrices_dir / "omega_gap_20260616.npy", Omega_gap)

        result = generate_v2_production_portfolio(
            trade_date="2026-06-16",
            gap_input_dir=tmp_path,
            cfg={},
        )
        pit = result["pit_binning"]
        for key in [
            "assigned_bin", "threshold_low", "threshold_high",
            "multiplier", "history_count", "fallback_flag",
        ]:
            assert key in pit, f"Missing pit_binning key: {key}"

    def test_leakage_audit_passes_with_valid_gap_files(self, tmp_path):
        """Leakage audit passes when gap matrices are dated before trade_date.

        The pipeline:
        1. Loads gap data for trade_date=2026-06-16 (mu_gap_20260616.npy)
        2. _derive_signal_date() scans matrices/ for the most-recent file
           strictly before trade_date → finds mu_gap_20260615.npy → signal_date=2026-06-15
        3. run_leakage_audit("2026-06-15", "2026-06-16") → PASSED
        """
        import pandas as pd

        mu_gap, Omega_gap = _make_synthetic_gap_data()
        matrices_dir = tmp_path / "matrices"
        matrices_dir.mkdir()

        # The pipeline loads trade_date's file (20260616)
        np.save(matrices_dir / "mu_gap_20260616.npy", mu_gap)
        np.save(matrices_dir / "omega_gap_20260616.npy", Omega_gap)

        # A prior-day file exists so _derive_signal_date returns 2026-06-15
        # (strictly before trade_date 2026-06-16 → leakage PASSED)
        np.save(matrices_dir / "mu_gap_20260615.npy", mu_gap)
        np.save(matrices_dir / "omega_gap_20260615.npy", Omega_gap)


        result = generate_v2_production_portfolio(
            trade_date="2026-06-16",
            gap_input_dir=tmp_path,
            cfg={},
        )
        assert result["fallback"]["gap_data_missing"] is False, \
            "Gap data should have been loaded for trade_date"
        assert result["leakage"]["status"] == "PASSED", \
            f"Leakage audit should pass; signal_date < trade_date. Got: {result['leakage']}"


# ---------------------------------------------------------------------------
# parse_run_config
# ---------------------------------------------------------------------------

class TestParseRunConfig:
    def test_empty_cfg_returns_defaults(self):
        """parse_run_config({}) should return all Pydantic defaults."""
        rc = parse_run_config({})
        assert isinstance(rc, ProductionV2RunConfig)
        assert rc.long_count == 5
        assert rc.short_count == 5
        assert abs(rc.baseline_gross - 2.0) < 1e-12
        assert abs(rc.cost_bps_per_gross - 10.0) < 1e-12
        assert rc.pit_rolling_window == 252
        assert abs(rc.mult_low - 0.75) < 1e-12
        assert abs(rc.mult_mid - 1.00) < 1e-12
        assert abs(rc.mult_high - 1.00) < 1e-12
        assert rc.fallback_on_gap_data_missing is True
        assert rc.fallback_on_audit_failure is True

    def test_custom_portfolio_counts(self):
        cfg = {"portfolio": {"long_count": 3, "short_count": 3}}
        rc = parse_run_config(cfg)
        assert rc.long_count == 3
        assert rc.short_count == 3

    def test_custom_gross_scaling(self):
        cfg = {
            "gross_scaling": {
                "baseline_gross": 1.5,
                "pit_rolling_window": 100,
                "multipliers": {"Low": 0.5, "Medium": 1.0, "High": 1.2},
            }
        }
        rc = parse_run_config(cfg)
        assert abs(rc.baseline_gross - 1.5) < 1e-12
        assert rc.pit_rolling_window == 100
        assert abs(rc.mult_low - 0.5) < 1e-12
        assert abs(rc.mult_high - 1.2) < 1e-12

    def test_custom_cost_bps(self):
        cfg = {"costs": {"cost_bps_per_gross": 15.0}}
        rc = parse_run_config(cfg)
        assert abs(rc.cost_bps_per_gross - 15.0) < 1e-12

    def test_fallback_flags_from_cfg(self):
        cfg = {
            "fallback": {
                "fallback_on_gap_data_missing": False,
                "fallback_on_audit_failure": False,
            }
        }
        rc = parse_run_config(cfg)
        assert rc.fallback_on_gap_data_missing is False
        assert rc.fallback_on_audit_failure is False

    def test_result_is_frozen(self):
        rc = parse_run_config({})
        with pytest.raises(Exception):
            rc.long_count = 99  # type: ignore


# ---------------------------------------------------------------------------
# cfg propagation into generate_v2_production_portfolio
# ---------------------------------------------------------------------------

class TestCfgPropagation:
    """Verify that cfg values are actually used in the pipeline output."""

    def _make_files(self, tmp_path, trade_date="2026-06-16"):
        import pandas as pd
        mu_gap, Omega_gap = _make_synthetic_gap_data()
        matrices_dir = tmp_path / "matrices"
        matrices_dir.mkdir(exist_ok=True)
        date_num = trade_date.replace("-", "")
        np.save(matrices_dir / f"mu_gap_{date_num}.npy", mu_gap)
        np.save(matrices_dir / f"omega_gap_{date_num}.npy", Omega_gap)

    def test_run_config_in_result(self, tmp_path):
        """result['run_config'] is a ProductionV2RunConfig instance."""
        self._make_files(tmp_path)
        result = generate_v2_production_portfolio(
            trade_date="2026-06-16",
            gap_input_dir=tmp_path,
            cfg={},
        )
        assert "run_config" in result
        assert isinstance(result["run_config"], ProductionV2RunConfig)

    def test_custom_long_short_count_respected(self, tmp_path):
        """Custom long_count=3/short_count=3 produces exactly 3 longs and 3 shorts."""
        self._make_files(tmp_path)
        cfg = {"portfolio": {"long_count": 3, "short_count": 3}}
        result = generate_v2_production_portfolio(
            trade_date="2026-06-16",
            gap_input_dir=tmp_path,
            cfg=cfg,
        )
        w = result["w_final"]
        assert int(np.sum(w > 1e-8)) == 3
        assert int(np.sum(w < -1e-8)) == 3

    def test_custom_baseline_gross_respected(self, tmp_path):
        """Custom baseline_gross=1.5 yields gross <= 1.5 (RuleD may reduce further)."""
        self._make_files(tmp_path)
        cfg = {"gross_scaling": {"baseline_gross": 1.5, "multipliers": {"Low": 1.0, "Medium": 1.0, "High": 1.0}}}
        result = generate_v2_production_portfolio(
            trade_date="2026-06-16",
            gap_input_dir=tmp_path,
            cfg=cfg,
        )
        gross = float(np.sum(np.abs(result["w_final"])))
        assert gross <= 1.5 + 1e-10

    def test_cost_bps_in_summary(self, tmp_path):
        """Custom cost_bps_per_gross=20.0 is reflected in summary expected_cost_bps."""
        self._make_files(tmp_path)
        cfg = {
            "costs": {"cost_bps_per_gross": 20.0},
            "gross_scaling": {"multipliers": {"Low": 1.0, "Medium": 1.0, "High": 1.0}},
        }
        result = generate_v2_production_portfolio(
            trade_date="2026-06-16",
            gap_input_dir=tmp_path,
            cfg=cfg,
        )
        gross = result["summary"]["target_gross"]
        expected_cost = result["summary"]["expected_cost_bps"]
        assert abs(expected_cost - gross * 20.0) < 1e-9

    def test_fallback_flag_respected(self, tmp_path):
        """fallback_on_gap_data_missing=True (default) returns flat position when gap dir is None."""
        result = generate_v2_production_portfolio(
            trade_date="2026-06-16",
            gap_input_dir=None,
            cfg={},
        )
        assert result["fallback"]["gap_data_missing"] is True


# ---------------------------------------------------------------------------
# Macro Factor-Kappa (Omega_gap inflation)
# ---------------------------------------------------------------------------

class TestMacroKappaOmegaGapInflation:
    """Test that macro kappa inflates Omega_gap and affects scores/weights."""

    def _make_files(self, tmp_path, trade_date="2026-06-16"):
        import pandas as pd
        mu_gap, Omega_gap = _make_synthetic_gap_data()
        matrices_dir = tmp_path / "matrices"
        matrices_dir.mkdir(exist_ok=True)
        date_num = trade_date.replace("-", "")
        np.save(matrices_dir / f"mu_gap_{date_num}.npy", mu_gap)
        np.save(matrices_dir / f"omega_gap_{date_num}.npy", Omega_gap)
        return mu_gap, Omega_gap

    def test_macro_kappa_disabled_by_default(self):
        rc = parse_run_config({})
        assert rc.macro_kappa_enabled is False

    def test_macro_kappa_enabled_from_cfg(self):
        rc = parse_run_config({"portfolio": {"macro_kappa_enabled": True}})
        assert rc.macro_kappa_enabled is True

    def test_macro_kappas_parsed_from_cfg(self):
        rc = parse_run_config({"portfolio": {"macro_kappas": [5.0, 1.0, 2.0]}})
        assert rc.macro_kappas == (5.0, 1.0, 2.0)

    def test_inflation_changes_omega_gap(self, tmp_path, monkeypatch):
        """When macro kappa is enabled, Omega_gap is inflated (diagonal increases)."""
        import pandas as pd
        from leadlag.models import production_v2 as pv2_mod

        mu_gap, Omega_gap_orig = self._make_files(tmp_path)

        # Mock download_macro_prices to return synthetic data
        dates = pd.date_range("2025-01-01", "2026-06-16", freq="B")
        rng = np.random.default_rng(123)
        close_data = {
            "USDJPY": 150.0 + rng.normal(0, 0.5, len(dates)).cumsum(),
            "CLF": 70.0 + rng.normal(0, 0.3, len(dates)).cumsum(),
            "TNX": 4.0 + rng.normal(0, 0.05, len(dates)).cumsum(),
        }
        mock_prices = pd.DataFrame(close_data, index=dates, columns=["USDJPY", "CLF", "TNX"])
        monkeypatch.setattr(pv2_mod, "download_macro_prices", lambda **kw: mock_prices.copy())

        cfg = {"portfolio": {"macro_kappa_enabled": True, "macro_kappas": [3.0, 0.5, 0.5]}}
        result = generate_v2_production_portfolio(
            trade_date="2026-06-16",
            gap_input_dir=tmp_path,
            cfg=cfg,
        )

        omega_result = result["Omega_gap"]
        orig_diag = np.diag(Omega_gap_orig)
        result_diag = np.diag(omega_result)
        # At least some diagonal entries should be inflated
        assert np.any(result_diag > orig_diag + 1e-10), \
            "Omega_gap diagonal should be inflated by macro kappa"
        # Check alert was added
        assert any("Macro kappa" in a for a in result["alerts"])

    def test_inflation_preserves_psd(self, tmp_path, monkeypatch):
        """Inflated Omega_gap remains PSD."""
        import pandas as pd
        from leadlag.models import production_v2 as pv2_mod

        _, _ = self._make_files(tmp_path)

        dates = pd.date_range("2025-01-01", "2026-06-16", freq="B")
        rng = np.random.default_rng(456)
        close_data = {
            "USDJPY": 150.0 + rng.normal(0, 0.5, len(dates)).cumsum(),
            "CLF": 70.0 + rng.normal(0, 0.3, len(dates)).cumsum(),
            "TNX": 4.0 + rng.normal(0, 0.05, len(dates)).cumsum(),
        }
        mock_prices = pd.DataFrame(close_data, index=dates, columns=["USDJPY", "CLF", "TNX"])
        monkeypatch.setattr(pv2_mod, "download_macro_prices", lambda **kw: mock_prices.copy())

        cfg = {"portfolio": {"macro_kappa_enabled": True, "macro_kappas": [3.0, 0.5, 0.5]}}
        result = generate_v2_production_portfolio(
            trade_date="2026-06-16",
            gap_input_dir=tmp_path,
            cfg=cfg,
        )

        omega = result["Omega_gap"]
        min_eig = np.min(np.linalg.eigvalsh(omega))
        assert min_eig > -1e-10, "Inflated Omega_gap should remain PSD"

    def test_disabled_does_not_inflate(self, tmp_path):
        """When macro kappa is disabled, Omega_gap is unchanged from input."""
        mu_gap, Omega_gap_orig = self._make_files(tmp_path)
        result = generate_v2_production_portfolio(
            trade_date="2026-06-16",
            gap_input_dir=tmp_path,
            cfg={},
        )
        assert np.allclose(result["Omega_gap"], Omega_gap_orig, atol=1e-10)


# ---------------------------------------------------------------------------
# Self-test parity (entry-point self-test exits 0)
# ---------------------------------------------------------------------------

class TestSelfTestParity:
    def test_entry_point_self_test_exits_zero(self):
        """The tools/ entry-point self-test function should return 0."""
        # Import from entry-point directly
        import importlib.util
        tools_script = ROOT / "tools" / "production" / "run_daily_production_v2.py"
        spec = importlib.util.spec_from_file_location("run_daily_v2", tools_script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert mod.run_self_tests() == 0
