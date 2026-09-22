"""Regression tests for the S2b versioned input boundary."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from leadlag.compliance.v2_auditor import run_leakage_audit
from leadlag.config.schemas import ProductionV2RunConfig
from leadlag.core.pit import PITMatrixView
from leadlag.data.pit_lake import MarketSnapshot, PITDataLake
from leadlag.data.tickers import JP_TICKERS
from leadlag.domain.inputs import DecisionInputs, HistoricalInputs, KnownMarketInputs
from leadlag.execution.config import load_config_from_yaml
from leadlag.execution.v2_bridge import _decision_as_of, _resolve_trade_date
from leadlag.models.production_v2 import ProductionV2Model
from leadlag.runner.production import ProductionRunner
from leadlag.utils.gap_matrix_io import save_gap_matrices


def _known() -> KnownMarketInputs:
    return KnownMarketInputs(
        trade_date=pd.Timestamp("2026-09-16"),
        as_of=pd.Timestamp("2026-09-16 09:10"),
        sig_date=pd.Timestamp("2026-09-15"),
        ticker_order=("A", "B"),
        us_returns=np.array([0.01, -0.02]),
        jp_gap_returns=np.array([0.003, -0.001]),
        jp_betas=np.array([0.8, 1.1]),
        topix_night_return=0.002,
        current_prices={"A": 100.0, "B": 200.0},
        prev_closes={"A": 99.0, "B": 201.0},
        source="test",
    )


def test_known_market_inputs_freezes_arrays_and_mappings() -> None:
    known = _known()
    with pytest.raises(ValueError):
        known.jp_gap_returns[0] = 0.5
    with pytest.raises(TypeError):
        known.current_prices["A"] = 101.0


def test_known_market_inputs_rejects_same_day_signal() -> None:
    with pytest.raises(ValueError, match="strictly earlier"):
        KnownMarketInputs(
            trade_date="2026-09-16",
            as_of="2026-09-16 09:10",
            sig_date="2026-09-16",
            ticker_order=("A",),
            us_returns=[0.0],
            jp_gap_returns=[0.0],
            jp_betas=[1.0],
            topix_night_return=0.0,
        )


def test_historical_inputs_owns_frame_and_versions_content() -> None:
    source = pd.DataFrame({"x": [1.0, 2.0]}, index=pd.to_datetime(["2026-09-16", "2026-09-15"]))
    history = HistoricalInputs(source, source="test")
    source.iloc[0, 0] = 99.0
    assert history.to_frame().iloc[0, 0] == 2.0
    isolated = history.to_frame()
    isolated.iloc[0, 0] = 88.0
    assert history.to_frame().iloc[0, 0] == 2.0
    digest = history.fingerprint
    assert len(digest) == 64


def test_historical_feature_frames_are_defensive_copies() -> None:
    index = pd.to_datetime(["2026-09-15", "2026-09-16"])
    open_910 = pd.DataFrame({"A": [0.01, 0.02]}, index=index)
    macro = pd.DataFrame({"USDJPY=X": [140.0, 141.0]}, index=index)
    adr = pd.DataFrame({"adr_A": [0.03, 0.04]}, index=index)
    history = HistoricalInputs(
        pd.DataFrame({"x": [1.0, 2.0]}, index=index),
        open_910_returns=open_910,
        macro_prices=macro,
        adr_features_frame=adr,
        source="test",
    )
    before = history.fingerprint

    assert history.open_910_returns is not open_910
    assert history.macro_prices is not macro
    assert history.adr_features is not adr
    exposed_open = history.open_910_returns
    exposed_macro = history.macro_prices
    exposed_adr = history.adr_features
    assert exposed_open is not None and exposed_macro is not None and exposed_adr is not None
    exposed_open.iloc[0, 0] = 99.0
    exposed_macro.iloc[0, 0] = 999.0
    exposed_adr.iloc[0, 0] = 999.0

    assert history.fingerprint == before
    assert history.macro_prices.iloc[0, 0] == pytest.approx(140.0)


def test_historical_pit_history_can_be_date_scoped() -> None:
    index = pd.to_datetime(["2026-09-15", "2026-09-16"])
    history = HistoricalInputs(
        pd.DataFrame({"x": [1.0, 2.0]}, index=index),
        pit_ir_history={
            "2026-09-15": np.array([1.0]),
            "2026-09-16": np.array([1.0, 2.0]),
        },
        source="test",
    )
    np.testing.assert_array_equal(history.pit_ir_history_for("2026-09-15"), [1.0])
    np.testing.assert_array_equal(history.pit_ir_history_for("2026-09-16"), [1.0, 2.0])
    assert history.pit_ir_history_for("2026-09-17") is None


def test_historical_pit_history_dates_are_date_scoped_and_versioned() -> None:
    index = pd.to_datetime(["2026-09-15", "2026-09-16"])
    dates = {
        "2026-09-15": np.array(["2026-09-14"], dtype="datetime64[ns]"),
        "2026-09-16": np.array(["2026-09-14", "2026-09-15"], dtype="datetime64[ns]"),
    }
    history = HistoricalInputs(
        pd.DataFrame({"x": [1.0, 2.0]}, index=index),
        pit_ir_history={"2026-09-15": [1.0], "2026-09-16": [1.0, 2.0]},
        pit_history_trade_dates=dates,
        source="test",
    )
    np.testing.assert_array_equal(
        history.pit_history_trade_dates_for("2026-09-16"), dates["2026-09-16"]
    )
    before = history.fingerprint
    assert before != HistoricalInputs(
        pd.DataFrame({"x": [1.0, 2.0]}, index=index),
        pit_ir_history={"2026-09-15": [1.0], "2026-09-16": [1.0, 2.0]},
        pit_history_trade_dates={
            "2026-09-15": ["2026-09-13"],
            "2026-09-16": ["2026-09-14", "2026-09-15"],
        },
        source="test",
    ).fingerprint


def test_historical_pit_history_dates_normalize_timezone_to_jst() -> None:
    """PIT dates must retain their JST calendar day before leakage checks."""
    history = HistoricalInputs(
        pd.DataFrame({"x": [1.0]}, index=pd.to_datetime(["2026-09-17"])),
        pit_ir_history=[0.1],
        pit_history_trade_dates=[pd.Timestamp("2026-09-16 15:00", tz="UTC")],
        source="test",
    )

    dates = history.pit_history_trade_dates_for("2026-09-17")
    assert dates is not None
    np.testing.assert_array_equal(dates, np.array(["2026-09-17"], dtype="datetime64[ns]"))
    audit = run_leakage_audit(
        "2026-09-16",
        "2026-09-17",
        pit_history_trade_dates=[pd.Timestamp("2026-09-16 15:00", tz="UTC")],
    )
    assert audit["pit_binning_strictly_historical"] is False
    assert audit["status"] == "FAILED"


def test_historical_observed_at_can_be_scoped_to_each_trade_date() -> None:
    index = pd.to_datetime(["2026-09-16", "2026-09-17"])
    history = HistoricalInputs(
        pd.DataFrame({"x": [1.0, 2.0]}, index=index),
        observed_at_by_date={
            "2026-09-16": {"macro_prices": "2026-09-16 09:00"},
            "2026-09-17": {"macro_prices": "2026-09-17 09:00"},
        },
        source="test",
    )
    assert history.observed_at_for("2026-09-16")["macro_prices"] == "2026-09-16 09:00"
    assert history.observed_at_for("2026-09-17")["macro_prices"] == "2026-09-17 09:00"


def test_historical_observed_at_scoped_values_merge_with_common_values() -> None:
    history = HistoricalInputs(
        pd.DataFrame({"x": [1.0]}, index=pd.to_datetime(["2026-09-16"])),
        observed_at={"rank_reversal_signals": "2026-09-16 09:20"},
        observed_at_by_date={
            "2026-09-16": {"macro_prices": "2026-09-16 09:00"},
        },
        source="test",
    )
    with pytest.raises(ValueError, match=r"historical observed_at\[rank_reversal_signals\]"):
        DecisionInputs(known=_known(), historical=history)


def test_decision_inputs_reject_future_historical_observation() -> None:
    frame = pd.DataFrame(
        {"rank_reversal": [0.2]},
        index=pd.to_datetime(["2026-09-16"]),
    )
    history = HistoricalInputs(
        frame,
        observed_at_by_date={
            "2026-09-16": {"rank_reversal_signals": "2026-09-16 09:20"},
        },
        source="test",
    )
    with pytest.raises(ValueError, match=r"historical observed_at\[rank_reversal_signals\]"):
        DecisionInputs(known=_known(), historical=history)


def test_historical_calculation_frame_masks_current_and_future_labels() -> None:
    index = pd.to_datetime(["2026-09-15", "2026-09-16", "2026-09-17"])
    source = pd.DataFrame(
        {
            "us_cc_X": [0.01, 0.02, 0.03],
            "jp_gap_A": [0.1, 0.2, 0.3],
            "jp_oc_A": [0.11, 0.22, 0.33],
            "jp_open_trade_A": [100.0, 101.0, 102.0],
        },
        index=index,
    )
    history = HistoricalInputs(source, source="test")

    view = history.calculation_frame("2026-09-16 09:10")
    assert list(view.index) == list(index[:2])
    assert view.loc[index[0], "jp_oc_A"] == pytest.approx(0.11)
    assert pd.isna(view.loc[index[1], "jp_oc_A"])
    assert view.loc[index[1], "jp_gap_A"] == pytest.approx(0.2)
    view.loc[index[0], "jp_gap_A"] = 99.0
    assert history.to_frame().loc[index[0], "jp_gap_A"] == pytest.approx(0.1)


def test_close_labels_become_visible_only_after_jp_close_cutoff() -> None:
    index = pd.to_datetime(["2026-09-16"])
    source = pd.DataFrame({"jp_oc_A": [0.11], "jp_gap_A": [0.02]}, index=index)
    history = HistoricalInputs(source, source="test")

    before_close = history.calculation_frame("2026-09-16 09:10")
    after_close = history.calculation_frame("2026-09-16 15:31")
    assert pd.isna(before_close.loc[index[0], "jp_oc_A"])
    assert after_close.loc[index[0], "jp_oc_A"] == pytest.approx(0.11)


def test_explicit_label_availability_masks_only_current_row_until_available() -> None:
    index = pd.to_datetime(["2026-09-15", "2026-09-16"])
    source = pd.DataFrame({"custom_target": [0.11, 0.22]}, index=index)
    history = HistoricalInputs(
        source,
        label_available_at={"custom_target": "2026-09-16 15:30"},
        source="test",
    )

    before_close = history.calculation_frame("2026-09-16 09:10")
    after_close = history.calculation_frame("2026-09-16 15:31")
    assert before_close.loc[index[0], "custom_target"] == pytest.approx(0.11)
    assert pd.isna(before_close.loc[index[1], "custom_target"])
    assert after_close.loc[index[1], "custom_target"] == pytest.approx(0.22)


def test_explicit_label_availability_overrides_default_close_for_standard_label() -> None:
    index = pd.to_datetime(["2026-09-16"])
    source = pd.DataFrame({"jp_oc_A": [0.11]}, index=index)
    history = HistoricalInputs(
        source,
        label_available_at={"jp_oc_*": "2026-09-16 09:00"},
        source="test",
    )

    view = history.calculation_frame("2026-09-16 09:10")
    assert view.loc[index[0], "jp_oc_A"] == pytest.approx(0.11)


def test_session_relative_label_availability_is_versioned() -> None:
    index = pd.to_datetime(["2026-09-16"])
    history = HistoricalInputs(
        pd.DataFrame({"jp_oc_A": [0.11]}, index=index),
        label_available_at={"jp_oc_*": "trade_date+15:30"},
        observed_at={"close_labels": "2026-09-16 15:30"},
        source="test",
    )
    before_close = history.calculation_frame("2026-09-16 09:10")
    after_close = history.calculation_frame("2026-09-16 15:31")
    assert pd.isna(before_close.loc[index[0], "jp_oc_A"])
    assert after_close.loc[index[0], "jp_oc_A"] == pytest.approx(0.11)
    assert history.fingerprint != HistoricalInputs(
        pd.DataFrame({"jp_oc_A": [0.11]}, index=index),
        label_available_at={"jp_oc_*": "trade_date+15:30"},
        observed_at={"close_labels": "2026-09-16 15:31"},
        source="test",
    ).fingerprint


def test_historical_calculation_frame_requires_a_trade_date() -> None:
    history = HistoricalInputs(
        pd.DataFrame({"jp_oc_A": [0.1]}, index=pd.to_datetime(["2026-09-16"])),
        source="test",
    )
    with pytest.raises(ValueError, match="not present"):
        history.calculation_frame("2026-09-15 09:10")


def test_historical_inputs_normalize_timezone_to_jst() -> None:
    source = pd.DataFrame(
        {"jp_oc_A": [0.1]},
        index=pd.DatetimeIndex(["2026-09-15 15:00",], tz="UTC"),
    )
    history = HistoricalInputs(source, source="test")
    assert history.to_frame().index[0] == pd.Timestamp("2026-09-16 00:00")
    assert history.calculation_frame("2026-09-16 09:10").index[0] == pd.Timestamp("2026-09-16 00:00")


@pytest.mark.parametrize("horizon", [1, 3, 5])
def test_future_target_perturbation_is_invisible_at_decision_time(horizon: int) -> None:
    index = pd.date_range("2026-09-01", periods=8, freq="D")
    frame = pd.DataFrame(
        {
            "us_cc_X": np.linspace(0.01, 0.08, len(index)),
            "jp_oc_A": np.linspace(0.02, 0.16, len(index)),
            "jp_open_trade_A": np.full(len(index), 100.0),
        },
        index=index,
    )
    changed = frame.copy()
    changed.loc[index[index > index[3]], "jp_oc_A"] += 100.0

    visible = HistoricalInputs(frame, horizon=horizon).calculation_frame(index[3])
    visible_changed = HistoricalInputs(changed, horizon=horizon).calculation_frame(index[3])
    pd.testing.assert_frame_equal(visible, visible_changed)


def test_decision_inputs_exposes_content_addressed_version() -> None:
    frame = pd.DataFrame({"x": [1.0]}, index=pd.to_datetime(["2026-09-16"]))
    inputs = DecisionInputs(known=_known(), historical=HistoricalInputs(frame, source="test"))
    assert inputs.trade_date == pd.Timestamp("2026-09-16")
    assert inputs.version.schema_version == "decision-inputs-v1"
    assert len(inputs.version.digest) == 64


def test_model_decision_trade_date_comparison_uses_jst(monkeypatch) -> None:
    """An offset-aware date representing the same JST day must be accepted."""
    import leadlag.models.production_v2 as production_v2

    model = ProductionV2Model(ProductionV2RunConfig())
    inputs = DecisionInputs(
        known=_known(),
        historical=HistoricalInputs(
            pd.DataFrame({"x": [1.0]}, index=pd.to_datetime(["2026-09-16"])),
            source="test",
        ),
    )
    monkeypatch.setattr(production_v2, "_v2_decide", lambda *args, **kwargs: "accepted")

    assert model.decide(trade_date="2026-09-15 15:00+00:00", inputs=inputs) == "accepted"


def test_internal_decision_trade_date_comparison_uses_jst(monkeypatch) -> None:
    """The internal decision boundary must apply the same calendar rule."""
    import leadlag.models.v2.decision_engine as decision_engine

    model = ProductionV2Model(ProductionV2RunConfig())
    inputs = DecisionInputs(
        known=_known(),
        historical=HistoricalInputs(
            pd.DataFrame({"x": [1.0]}, index=pd.to_datetime(["2026-09-16"])),
            source="test",
        ),
    )

    def stop_after_date_validation(*args, **kwargs):
        raise RuntimeError("date validation passed")

    monkeypatch.setattr(decision_engine.FallbackPolicy, "default", stop_after_date_validation)
    with pytest.raises(RuntimeError, match="date validation passed"):
        decision_engine._decide(
            model,
            "2026-09-15 15:00+00:00",
            inputs=inputs,
        )


def test_pit_snapshot_and_view_are_isolated() -> None:
    source = np.array([1.0, 2.0])
    view = PITMatrixView(source, as_of=2)
    source[0] = 99.0
    assert view.historical_range(0, 1)[0] == 1.0
    with pytest.raises(ValueError):
        view.historical_range(0, 3)

    snapshot = MarketSnapshot(
        as_of=pd.Timestamp("2026-09-16 09:10"),
        trade_date="2026-09-16",
        us_returns=np.zeros(15),
        jp_gap_returns=np.zeros(17),
        jp_betas=np.ones(17),
        topix_night_return=0.0,
        current_prices={JP_TICKERS[0]: 100.0},
        prev_closes={JP_TICKERS[0]: 99.0},
    )
    with pytest.raises(ValueError):
        snapshot.us_returns[0] = 1.0
    with pytest.raises(TypeError):
        snapshot.current_prices[JP_TICKERS[0]] = 101.0


def test_pit_lake_normalizes_timezone_aware_trade_dates() -> None:
    frame = pd.DataFrame(
        {"sig_date": [pd.Timestamp("2026-09-15")]},
        index=pd.DatetimeIndex(["2026-09-15 15:00"], tz="UTC"),
    )
    lake = PITDataLake(frame)
    snapshot = lake.get_snapshot("2026-09-16 09:10+09:00")
    assert snapshot.trade_date == "2026-09-16"


def test_pit_lake_builds_known_and_historical_contract() -> None:
    frame = pd.DataFrame(
        {
            "sig_date": [pd.Timestamp("2026-09-15")],
            f"jp_open_trade_{JP_TICKERS[0]}": [101.0],
            f"jp_close_sig_{JP_TICKERS[0]}": [100.0],
        },
        index=pd.to_datetime(["2026-09-16"]),
    )
    inputs = PITDataLake(frame).build_decision_inputs(
        "2026-09-16",
        current_prices={JP_TICKERS[0]: 105.0},
        source="test_lake",
    )
    assert inputs.known.current_prices[JP_TICKERS[0]] == 105.0
    assert inputs.known.jp_gap_returns[0] == pytest.approx(0.05)
    assert inputs.historical.source == "test_lake"
    assert inputs.known.as_of == pd.Timestamp("2026-09-16 09:10")


def test_date_only_pit_lake_request_uses_0910_snapshot() -> None:
    frame = pd.DataFrame(
        {
            f"jp_open_trade_{JP_TICKERS[0]}": [101.0],
            f"jp_close_sig_{JP_TICKERS[0]}": [100.0],
        },
        index=pd.to_datetime(["2026-09-16"]),
    )
    inputs = PITDataLake(frame).build_decision_inputs("2026-09-16")
    assert inputs.known.as_of == pd.Timestamp("2026-09-16 09:10")
    with pytest.raises(ValueError, match="snapshot timestamp"):
        PITDataLake(frame).build_decision_inputs(
            "2026-09-16",
            snapshot=PITDataLake(frame).get_snapshot("2026-09-16 08:00"),
        )


def test_pit_lake_rejects_pre_0910_decision_request() -> None:
    frame = pd.DataFrame(
        {
            f"jp_open_trade_{JP_TICKERS[0]}": [101.0],
            f"jp_close_sig_{JP_TICKERS[0]}": [100.0],
        },
        index=pd.to_datetime(["2026-09-16"]),
    )

    with pytest.raises(ValueError, match="at or after 09:10"):
        PITDataLake(frame).build_decision_inputs("2026-09-16 08:00")


def test_known_market_inputs_rejects_pre_0910_as_of() -> None:
    with pytest.raises(ValueError, match="at or after 09:10"):
        KnownMarketInputs(
            trade_date="2026-09-16",
            as_of="2026-09-16 09:09",
            ticker_order=("A",),
            us_returns=[0.0],
            jp_gap_returns=[0.0],
            jp_betas=[1.0],
            topix_night_return=0.0,
        )


def test_live_bridge_decision_cutoff_matches_observed_input_times() -> None:
    """The live bridge must snapshot at 09:10, after all known observations."""
    cutoff = _decision_as_of("2026-09-16")
    assert cutoff == pd.Timestamp("2026-09-16 09:10")

    frame = pd.DataFrame(
        {
            "sig_date": [pd.Timestamp("2026-09-15")],
            f"jp_open_trade_{JP_TICKERS[0]}": [101.0],
            f"jp_close_sig_{JP_TICKERS[0]}": [100.0],
        },
        index=pd.to_datetime(["2026-09-16"]),
    )
    inputs = PITDataLake(frame).build_decision_inputs(
        cutoff,
        source="v2_bridge_live",
        observed_at={
            "us_returns": "2026-09-16 09:00",
            "jp_gap_returns": "2026-09-16 09:10",
            "current_prices": "2026-09-16 09:10",
        },
    )
    assert inputs.known.as_of == cutoff


def test_live_bridge_decision_cutoff_normalizes_aware_trade_date_to_jst() -> None:
    cutoff = _decision_as_of("2026-09-15 15:00+00:00")
    assert cutoff == pd.Timestamp("2026-09-16 09:10")


def test_resolve_trade_date_normalizes_aware_explicit_input_to_jst(tmp_path: Path) -> None:
    assert (
        _resolve_trade_date("2026-09-15 15:00+00:00", tmp_path)
        == "2026-09-16"
    )


def test_resolve_latest_trade_date_normalizes_aware_csv_value_to_jst(tmp_path: Path) -> None:
    pd.DataFrame({"trade_date": ["2026-09-15T15:00:00+00:00"]}).to_csv(
        tmp_path / "latest_weights.csv", index=False
    )
    assert _resolve_trade_date("latest", tmp_path) == "2026-09-16"


def test_pit_lake_compatibility_frame_is_a_copy() -> None:
    frame = pd.DataFrame({"x": [1.0]}, index=pd.to_datetime(["2026-09-16"]))
    lake = PITDataLake(frame)
    exposed = lake.df_exec
    exposed.iloc[0, 0] = 99.0
    assert lake.df_exec.iloc[0, 0] == 1.0


def test_v2_model_accepts_decision_inputs_contract(tmp_path) -> None:
    cfg = load_config_from_yaml("configs/production/production.yaml", strict=True).v2.model_copy(
        deep=True,
        update={
            "gap_input_dir": str(tmp_path),
            "ml_overlay_enabled": False,
            "cs_overlay_enabled": False,
            "macro_kappa_enabled": False,
            "macro_direction_enabled": False,
        },
    )
    date = "2026-09-16"
    n_j = len(JP_TICKERS)
    assert save_gap_matrices(
        tmp_path,
        date,
        np.linspace(-0.01, 0.01, n_j),
        np.eye(n_j) * 0.01,
        metadata={"sig_date": "2026-09-15", "trade_date": date, "horizon": 1},
    )
    frame = pd.DataFrame(
        {"sig_date": [pd.Timestamp("2026-09-15")]},
        index=pd.to_datetime([date]),
    )
    lake = PITDataLake(frame)
    inputs = lake.build_decision_inputs(
        date,
        current_prices={ticker: 1000.0 for ticker in JP_TICKERS},
        gap_input_dir=tmp_path,
        source="test_model",
    )
    result = ProductionV2Model(cfg, blpx_model=None).decide(inputs=inputs, overlay_enabled=False)
    assert result.w_final.shape == (n_j,)


def test_runner_can_be_constructed_from_one_contract() -> None:
    sentinel = object()

    class CaptureModel:
        def decide(self, **kwargs):
            assert kwargs["inputs"] is inputs
            assert kwargs["use_file_cache"] is False
            return sentinel

    frame = pd.DataFrame({"x": [1.0]}, index=pd.to_datetime(["2026-09-16"]))
    inputs = DecisionInputs(
        known=_known(), historical=HistoricalInputs(frame, source="test"), use_file_cache=False
    )
    runner = object.__new__(ProductionRunner)
    runner._overlay_enabled = False
    runner.model = CaptureModel()
    assert runner.run(inputs) is sentinel


def test_runner_rejects_missing_production_current_price_before_model() -> None:
    frame = pd.DataFrame({"x": [1.0]}, index=pd.to_datetime(["2026-09-16"]))
    known = KnownMarketInputs(
        trade_date="2026-09-16",
        as_of="2026-09-16 09:10",
        sig_date="2026-09-15",
        ticker_order=tuple(JP_TICKERS),
        us_returns=np.zeros(15),
        jp_gap_returns=np.zeros(len(JP_TICKERS)),
        jp_betas=np.ones(len(JP_TICKERS)),
        topix_night_return=0.0,
        current_prices={ticker: 100.0 for ticker in JP_TICKERS[:-1]},
        prev_closes={ticker: 99.0 for ticker in JP_TICKERS[:-1]},
        source="test",
    )
    inputs = DecisionInputs(
        known=known,
        historical=HistoricalInputs(frame, source="test"),
    )
    runner = object.__new__(ProductionRunner)
    runner._overlay_enabled = False
    runner.model = object()
    with pytest.raises(ValueError, match="missing current_prices"):
        runner.run(inputs)
