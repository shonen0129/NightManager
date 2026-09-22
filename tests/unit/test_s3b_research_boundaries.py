"""Contract tests for the S3b research orchestration boundaries."""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.diagnostics.gap_inputs import (
    GapModelInputs,
    attach_topix_trade_returns,
    build_gap_historical_inputs,
    mask_future_jp_labels,
    prepare_gap_model_inputs,
)
from research.diagnostics.gap_outputs import (
    compute_pit_bins,
    prepare_portfolio_output_frame,
)
from research.diagnostics.gap_portfolio import evaluate_gap_portfolio


def _input_mapping() -> dict[str, np.ndarray]:
    return {
        "y_jp_target": np.array([1.0]),
        "jp_gap": np.array([2.0]),
        "jp_beta": np.array([3.0]),
        "topix_night": np.array([4.0]),
        "jp_res_returns_p3": np.array([5.0]),
        "c_full_p3": np.array([6.0]),
        "v0_static": np.array([7.0]),
    }


def test_gap_input_assembly_preserves_h1_compatibility_and_horizon_override() -> None:
    class Model:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def _prepare_common_inputs(self, frame: pd.DataFrame, **kwargs: object) -> dict:
            self.calls.append(kwargs)
            values = _input_mapping()
            if "y_jp_target" in kwargs:
                values["y_jp_target"] = np.asarray(kwargs["y_jp_target"])
            return values

    model = Model()
    frame = pd.DataFrame({"x": [1.0]})
    h1 = prepare_gap_model_inputs(model, frame)
    h3 = prepare_gap_model_inputs(model, frame, horizon=3, y_jp_target=np.array([[8.0]]))

    assert isinstance(h1, GapModelInputs)
    assert model.calls[0] == {}
    assert model.calls[1]["horizon"] == 3
    np.testing.assert_array_equal(h3.y_jp_target, np.array([[8.0]]))


def test_portfolio_evaluation_uses_two_sided_slippage_and_market_exposure() -> None:
    result = evaluate_gap_portfolio(
        weights=np.array([0.5, -0.25]),
        realized_returns=np.array([0.02, -0.01]),
        mu_raw=np.array([0.01, -0.005]),
        omega_raw=np.diag([0.04, 0.01]),
        mu_gap=np.array([0.02, -0.002]),
        omega_gap=np.diag([0.01, 0.04]),
        slippage_bps_per_side=5.0,
        turnover=0.4,
    )

    expected_cost = 2.0 * 5.0 / 10000.0 * 0.75
    assert result["gross_return"] == 0.0125
    assert result["cost"] == expected_cost
    assert result["net_return"] == result["gross_return"] - expected_cost
    assert result["gross_exposure"] == 0.75
    assert result["net_exposure"] == 0.25
    assert result["turnover"] == 0.4


def test_topix_trade_returns_are_attached_without_mutating_execution_frame() -> None:
    index = pd.date_range("2026-01-02", periods=2, freq="D")
    frame = pd.DataFrame({"topix_night_return": [0.01, -0.02]}, index=index)
    close = pd.Series([101.0, 98.0], index=index)
    open_ = pd.Series([100.0, 100.0], index=index)

    output = attach_topix_trade_returns(
        frame,
        {"jp_close": {"1306.T": close}, "jp_open": {"1306.T": open_}},
    )

    np.testing.assert_allclose(output["topix_oc_return"], [0.01, -0.02])
    np.testing.assert_allclose(output["topix_cc_trade"], [0.0201, -0.0396])
    assert "topix_oc_return" not in frame.columns


def test_portfolio_output_cost_estimate_is_shifted() -> None:
    frame = pd.DataFrame(
        {
            "cost": [0.01, 0.02, 0.03],
            "pred_mean_gap": [0.1, 0.2, 0.3],
            "pred_vol_gap": [1.0, 1.0, 1.0],
        }
    )
    output = prepare_portfolio_output_frame(frame)
    np.testing.assert_allclose(output["cost_estimate_exante"], [0.0, 0.01, 0.015])
    np.testing.assert_allclose(output["pred_ir_gap_exante_cost"], [0.1, 0.19, 0.285])


def test_pit_bins_do_not_use_current_observation_for_boundaries() -> None:
    index = pd.date_range("2026-01-01", periods=25)
    series = pd.Series(np.arange(25, dtype=float), index=index)
    changed = series.copy()
    changed.iloc[15] = -999.0

    bins = compute_pit_bins(series, "tertile", rolling_window=15)
    changed_bins = compute_pit_bins(changed, "tertile", rolling_window=15)
    assert bins.iloc[:15].isna().all()
    assert bins.iloc[15] == "High"
    assert changed_bins.iloc[15] == "Low"
    pd.testing.assert_series_equal(bins.iloc[:15], changed_bins.iloc[:15])


def test_gap_historical_inputs_use_session_relative_label_cutoff() -> None:
    index = pd.bdate_range("2026-09-14", periods=2)
    frame = pd.DataFrame(
        {"jp_oc_A": [0.01, 0.02], "jp_gap_A": [0.0, 0.01]},
        index=index,
    )
    history = build_gap_historical_inputs(frame)
    before_close = history.calculation_frame("2026-09-15 09:10")
    after_close = history.calculation_frame("2026-09-15 15:31")
    assert pd.isna(before_close.loc[index[1], "jp_oc_A"])
    assert after_close.loc[index[1], "jp_oc_A"] == 0.02


def test_gap_historical_inputs_record_per_date_observation_boundaries() -> None:
    index = pd.bdate_range("2026-09-14", periods=2)
    frame = pd.DataFrame({"jp_oc_A": [0.01, 0.02]}, index=index)
    history = build_gap_historical_inputs(frame)
    observed = history.observed_at_for(index[1])
    assert observed["us_returns"] == "2026-09-15 09:00"
    assert observed["open_910_returns"] == "2026-09-15 09:10"
    history.validate_observed_at("2026-09-15 09:10")


def test_gap_target_mask_hides_current_and_future_rows() -> None:
    index = pd.bdate_range("2026-09-14", periods=3)
    values = np.ones((3, 4), dtype=float)
    masked = mask_future_jp_labels(
        values,
        index,
        "2026-09-15 09:10",
        n_u=2,
    )
    np.testing.assert_allclose(masked[0, 2:], [1.0, 1.0])
    assert np.isnan(masked[1:, 2:]).all()
