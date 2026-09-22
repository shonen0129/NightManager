"""Structural contracts for the S2a input adapters."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from leadlag.compliance.v2_auditor import run_leakage_audit
from leadlag.config.schemas import ProductionV2RunConfig
from leadlag.data import intraday_inputs
from leadlag.data.rank_reversal import load_rank_reversal_frame
from leadlag.data.tickers import JP_TICKERS
from leadlag.models.signal_enhancement import apply_multi_horizon_blend
from leadlag.models.v2.fallback import _apply_pit_ruleD, _repair_and_adjust
from leadlag.models.v2.overlay_applier import _apply_rank_reversal_overlay
from leadlag.utils.gap_matrix_io import save_gap_matrices
from leadlag.utils.gap_provenance import bundle_identity, gap_inputs_version

ROOT = Path(__file__).resolve().parents[2]


def test_core_macro_has_no_network_or_cache_io() -> None:
    """Macro math must not import yfinance or own a download cache."""
    source = (ROOT / "src/leadlag/core/macro.py").read_text()
    tree = ast.parse(source)
    imported = {
        node.names[0].name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import) and node.names
    }
    assert "yfinance" not in imported
    assert "run_with_timeout" not in source
    assert "download_macro_prices" not in source


def test_model_overlay_has_no_adr_filesystem_loader() -> None:
    """ADR artifact ownership belongs to the data adapter."""
    source = (ROOT / "src/leadlag/models/ml_order_overlay.py").read_text()
    assert "def _load_adr_features" not in source
    assert "read_pickle" not in source


def test_intraday_adapter_exposes_explicit_h1_input() -> None:
    """The legacy h=1 correction is an explicit adapter output."""
    source = (ROOT / "src/leadlag/data/intraday_inputs.py").read_text()
    assert "def build_open_910_returns" in source
    assert "def compute_jp_target_returns" in source


def test_intraday_adapter_normalizes_timezone_aware_execution_and_bars() -> None:
    """09:10 extraction uses the JST date even when sources carry offsets."""
    ticker = JP_TICKERS[0]
    jst_date = pd.Timestamp("2026-09-16")
    execution_index = pd.DatetimeIndex([jst_date.tz_localize("Asia/Tokyo").tz_convert("UTC")])
    frame = pd.DataFrame(
        {f"jp_open_trade_{ticker}": [100.0]},
        index=execution_index,
    )
    bars_index = pd.DatetimeIndex(
        [jst_date + pd.Timedelta(hours=9), jst_date + pd.Timedelta(hours=9, minutes=10)]
    ).tz_localize("Asia/Tokyo").tz_convert("UTC")
    bars = pd.DataFrame(
        {
            ("Open", ticker): [90.0, 90.0],
            ("High", ticker): [90.0, 99.0],
            ("Low", ticker): [90.0, 99.0],
            ("Close", ticker): [90.0, 99.0],
        },
        index=bars_index,
    )

    p_910 = intraday_inputs.build_5m_910_prices(frame, [ticker], df_5m=bars)
    open_returns = intraday_inputs.build_open_910_returns(frame, [ticker], df_5m=bars)
    assert p_910.iloc[0, 0] == pytest.approx(99.0)
    assert open_returns.iloc[0, 0] == pytest.approx(-0.01)


def test_rank_reversal_normalizes_timezone_aware_trade_date(tmp_path: Path) -> None:
    """Rank-reversal date files use the strategy's JST calendar date."""
    matrices = tmp_path / "matrices"
    matrices.mkdir()
    np.save(matrices / "rank_reversal_20260916.npy", np.ones(len(JP_TICKERS)))

    frame = load_rank_reversal_frame(
        tmp_path,
        [pd.Timestamp("2026-09-15 15:00", tz="UTC")],
    )

    assert frame is not None
    assert list(frame.index) == [pd.Timestamp("2026-09-16")]


def test_strict_h1_target_does_not_reopen_5m_cache(monkeypatch) -> None:
    """A strict typed target must fail closed when its 09:10 frame is absent."""
    def _unexpected(*args, **kwargs):
        raise AssertionError("implicit intraday source access")

    monkeypatch.setattr(intraday_inputs, "load_intraday_cache", _unexpected)
    ticker = JP_TICKERS[0]
    frame = pd.DataFrame(
        {
            f"jp_open_trade_{ticker}": [100.0],
            f"jp_oc_{ticker}": [0.01],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2026-09-16")]),
    )

    with pytest.raises(ValueError, match="explicit open_910_returns"):
        intraday_inputs.compute_jp_target_returns(
            frame,
            [ticker],
            horizon=1,
            allow_implicit_io=False,
        )


@pytest.mark.parametrize("horizon", [3, 5])
def test_strict_multi_day_target_requires_run_owned_open_910(horizon: int) -> None:
    """Strict multi-day target arithmetic cannot silently use start-day open."""
    ticker = JP_TICKERS[0]
    frame = pd.DataFrame(
        {
            f"jp_open_trade_{ticker}": [100.0],
            f"jp_oc_{ticker}": [0.01],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2026-09-16")]),
    )

    with pytest.raises(ValueError, match=f"strict h={horizon}.*explicit open_910_returns"):
        intraday_inputs.compute_jp_target_returns(
            frame,
            [ticker],
            horizon=horizon,
            allow_implicit_io=False,
        )


@pytest.mark.parametrize("horizon", [1, 3, 5])
def test_strict_target_uses_explicit_open_fallback(horizon: int) -> None:
    """Missing observations use actual daily opens, without inventing 09:10 data."""
    ticker = JP_TICKERS[0]
    dates = pd.bdate_range("2026-09-09", periods=6)
    frame = pd.DataFrame(
        {
            f"jp_open_trade_{ticker}": [100.0] * len(dates),
            f"jp_oc_{ticker}": [0.01] * len(dates),
        },
        index=dates,
    )
    open_returns = pd.DataFrame(np.nan, index=frame.index, columns=[ticker])

    result = intraday_inputs.compute_jp_target_returns(
        frame, [ticker], horizon=horizon, open_910_returns=open_returns,
        allow_implicit_io=False,
    )
    assert np.isnan(result[:horizon - 1]).all()
    np.testing.assert_allclose(result[horizon - 1:], 0.01)
    assert open_returns.isna().all().all()


@pytest.mark.parametrize("horizon", [1, 3, 5])
@pytest.mark.parametrize("invalid_open", [0.0, -1.0, np.nan, np.inf])
def test_strict_target_rejects_missing_observation_and_invalid_open(horizon, invalid_open):
    ticker = JP_TICKERS[0]
    frame = pd.DataFrame(
        {f"jp_open_trade_{ticker}": [invalid_open], f"jp_oc_{ticker}": [0.01]},
        index=pd.DatetimeIndex(["2026-09-16"]),
    )
    with pytest.raises(ValueError, match="fallback opens"):
        intraday_inputs.compute_jp_target_returns(
            frame, [ticker], horizon=horizon,
            open_910_returns=pd.DataFrame(np.nan, index=frame.index, columns=[ticker]),
            allow_implicit_io=False,
        )


def test_strict_target_checks_only_current_date_when_history_is_sparse() -> None:
    """Historical 09:10 gaps may fall back while today's input is complete."""
    dates = pd.DatetimeIndex([pd.Timestamp("2026-09-15"), pd.Timestamp("2026-09-16")])
    frame = pd.DataFrame(
        {
            **{f"jp_open_trade_{ticker}": [100.0, 100.0] for ticker in JP_TICKERS},
            **{f"jp_oc_{ticker}": [0.01, 0.02] for ticker in JP_TICKERS},
        },
        index=dates,
    )
    open_returns = pd.DataFrame(np.nan, index=dates, columns=JP_TICKERS)
    open_returns.loc[dates[-1], :] = 0.001

    assert not intraday_inputs.has_valid_open_910_returns(open_returns, frame, JP_TICKERS)
    assert intraday_inputs.has_valid_open_910_returns(
        open_returns, frame, JP_TICKERS, required_index=[dates[-1]]
    )
    target = intraday_inputs.compute_jp_target_returns(
        frame,
        JP_TICKERS,
        open_910_returns=open_returns,
        allow_implicit_io=False,
        required_index=[dates[-1]],
    )
    assert target.shape == (len(frame), len(JP_TICKERS))


def test_strict_typed_fallback_does_not_reopen_macro_or_pit_sources(monkeypatch) -> None:
    """Run-owned decision inputs must prevent fallback helpers from doing I/O."""
    import leadlag.models.v2.fallback as fallback

    def _unexpected(*args, **kwargs):
        raise AssertionError("implicit source access")

    monkeypatch.setattr(fallback.macro_data, "load_macro_prices", _unexpected)
    monkeypatch.setattr(fallback, "load_pit_ir_history", _unexpected)
    cfg = ProductionV2RunConfig(
        macro_kappa_enabled=True,
        macro_direction_enabled=False,
        pit_rolling_window=10,
    )
    mu, omega, alerts = _repair_and_adjust(
        np.zeros(17),
        np.eye(17),
        cfg,
        "2026-09-16",
        17,
        [],
        allow_implicit_io=False,
    )
    assert mu.shape == (17,)
    assert omega.shape == (17, 17)
    assert any("explicit macro_prices" in alert for alert in alerts)
    w_final, pit, pit_alerts, _ = _apply_pit_ruleD(
        np.ones(17),
        mu,
        omega,
        Path("/tmp/absent-gap-store"),
        "2026-09-16",
        cfg,
        [],
        allow_implicit_io=False,
    )
    assert np.isfinite(w_final).all()
    assert pit["fallback_flag"] is True
    assert any("run snapshot" in alert for alert in pit_alerts)


def test_explicit_pit_history_keeps_dates_for_leakage_audit() -> None:
    """Run-owned PIT values must carry their source dates into the audit."""
    cfg = ProductionV2RunConfig(pit_rolling_window=1)
    history_dates = np.array(["2026-09-16"], dtype="datetime64[ns]")
    w_final, pit, _alerts, audited_dates = _apply_pit_ruleD(
        np.ones(17),
        np.zeros(17),
        np.eye(17),
        None,
        "2026-09-16",
        cfg,
        [],
        pit_ir_history=np.array([0.1]),
        pit_history_trade_dates=history_dates,
        allow_implicit_io=False,
    )
    assert pit["history_count"] == 1
    np.testing.assert_array_equal(audited_dates, history_dates)
    audit = run_leakage_audit(
        "2026-09-15", "2026-09-16", pit_history_trade_dates=audited_dates
    )
    assert audit["status"] == "FAILED"
    assert audit["pit_binning_strictly_historical"] is False

    _w_final, _pit, missing_alerts, missing_dates = _apply_pit_ruleD(
        np.ones(17),
        np.zeros(17),
        np.eye(17),
        None,
        "2026-09-16",
        cfg,
        [],
        pit_ir_history=np.array([0.1]),
        allow_implicit_io=False,
    )
    assert len(missing_dates) == 1
    assert any("trade dates" in alert for alert in missing_alerts)


def test_strict_rank_reversal_does_not_reopen_gap_file(monkeypatch) -> None:
    """A typed run must skip a missing snapshot signal without implicit I/O."""
    import leadlag.models.signal_enhancement as signal_enhancement

    def _unexpected(*args, **kwargs):
        raise AssertionError("implicit rank-reversal source access")

    monkeypatch.setattr(signal_enhancement, "load_gap_npy", _unexpected)
    cfg = ProductionV2RunConfig(cs_overlay_enabled=True, cs_overlay_weight=0.05)
    scores = np.linspace(-1.0, 1.0, 17)
    enhanced, alerts = _apply_rank_reversal_overlay(
        scores,
        Path("/tmp/snapshot-gap"),
        "2026-09-16",
        cfg,
        [],
        rank_reversal_signal=None,
        allow_implicit_io=False,
    )

    np.testing.assert_array_equal(enhanced, scores)
    assert any("explicit signal missing" in alert for alert in alerts)


def test_multi_horizon_blend_requires_shared_bundle_identity(tmp_path: Path) -> None:
    """Research multi-horizon loading must use the shared provenance contract."""
    date = "2026-09-16"
    frame = pd.DataFrame(
        {"sig_date": [pd.Timestamp("2026-09-15")]},
        index=pd.DatetimeIndex([pd.Timestamp(date)]),
    )

    class Model:
        run_config = ProductionV2RunConfig()

    model = Model()
    identity = bundle_identity(frame, date, config=model.run_config, model=model)
    assert save_gap_matrices(
        tmp_path,
        date,
        np.ones(len(JP_TICKERS)) * 0.01,
        np.eye(len(JP_TICKERS)) * 0.001,
        mu_pattern="matrices/mu_gap_h{h}_{date}.npy",
        omega_pattern="matrices/omega_gap_h{h}_{date}.npy",
        pattern_kwargs={"h": 3},
        metadata={
            "sig_date": "2026-09-15",
            "trade_date": date,
            "horizon": 3,
            **identity,
            "gap_inputs_version": gap_inputs_version(
                date,
                np.zeros(len(JP_TICKERS)),
                np.zeros(len(JP_TICKERS)),
                0.0,
                horizon=3,
            ),
        },
    )

    scores = np.linspace(-1.0, 1.0, len(JP_TICKERS))
    blended, alerts = apply_multi_horizon_blend(
        scores,
        tmp_path,
        date,
        horizons=(1, 3),
        weights=(0.5, 0.5),
        expected_identity=identity,
    )
    assert blended.shape == scores.shape
    assert not any("unprovenanced" in alert for alert in alerts)
