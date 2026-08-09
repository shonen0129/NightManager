"""Unit tests for the ML order decision overlay."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from leadlag.data.tickers import JP_TICKERS
from leadlag.models.ml_order_overlay import (
    MLOrderOverlayModel,
    _build_ticker_features,
    _predict_p_trade,
    _safe,
    _sigmoid,
    apply_overlay,
)

N_J = len(JP_TICKERS)


class _DummyLGBM:
    """Mock LightGBM regressor that returns zeros."""

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(x), dtype=float)


def _run_config() -> SimpleNamespace:
    return SimpleNamespace(
        long_count=5,
        short_count=5,
        baseline_gross=2.0,
        minvar_enabled=False,
        minvar_alpha=0.5,
        cost_bps_per_gross=10.0,
    )


def _base_result(scores: np.ndarray, omega: np.ndarray | None = None) -> dict:
    if omega is None:
        omega = np.eye(N_J)
    return {
        "scores": scores.astype(float),
        "mu_gap": np.zeros(N_J),
        "sigma_gap": np.full(N_J, 0.01),
        "Omega_gap": omega,
        "run_config": _run_config(),
        "pit_binning": {"multiplier": 1.0, "assigned_bin": "Medium"},
        "fallback": {"gap_data_missing": False},
        "summary": {
            "target_gross": 2.0,
            "expected_cost_bps": 20.0,
            "predicted_portfolio_ir": 0.0,
        },
        "w_final": np.zeros(N_J),
        "numerical": {"status": "PASSED"},
    }


def _make_df_exec(trade_date: pd.Timestamp) -> pd.DataFrame:
    """Create a minimal df_exec with enough history for market_vol_20d."""
    dates = pd.date_range(trade_date - pd.Timedelta(days=29), trade_date, freq="B")
    data = {"topix_night_return": np.full(len(dates), 0.001)}
    for tk in JP_TICKERS:
        data[f"jp_gap_{tk}"] = np.full(len(dates), 0.0)
        data[f"jp_beta_{tk}"] = np.full(len(dates), 1.0)
        data[f"jp_oc_{tk}"] = np.full(len(dates), 0.001)
    return pd.DataFrame(data, index=dates)


def test_safe_replaces_nan_and_inf():
    arr = np.array([1.0, np.nan, np.inf, -np.inf, 2.0])
    out = _safe(arr)
    assert np.allclose(out, [1.0, 0.0, 0.0, 0.0, 2.0])


def test_sigmoid_scale():
    x = np.array([0.0, 1.0, -1.0])
    out = _sigmoid(x, 1.0)
    assert np.allclose(out, [0.5, 0.73105858, 0.26894142], atol=1e-6)


def test_build_ticker_features_per_ticker_interactions():
    trade_date = pd.Timestamp("2023-03-27")
    df_exec = _make_df_exec(trade_date)
    v2_result = _base_result(np.linspace(-1.0, 1.0, N_J))
    v2_result["mu_gap"] = np.full(N_J, 0.0)
    v2_result["sigma_gap"] = np.full(N_J, 0.01)
    market_vol = df_exec[[f"jp_oc_{tk}" for tk in JP_TICKERS]].abs().rolling(
        window=20, min_periods=5
    ).mean().shift(1)
    market_vol.columns = JP_TICKERS

    features = _build_ticker_features(
        df_exec,
        v2_result,
        trade_date,
        market_vol,
        per_ticker_interactions=True,
    )
    assert len(features) == N_J
    assert f"ticker_{JP_TICKERS[0]}_score" in features.columns
    # Only the current ticker's interaction column is non-zero.
    for i, tk in enumerate(JP_TICKERS):
        assert features.loc[i, f"ticker_{tk}_score"] == pytest.approx(
            features.loc[i, "score"]
        )


def test_predict_p_trade():
    features = pd.DataFrame(
        {
            "score": [1.0, -1.0],
            "mu_gap": [0.0, 0.0],
            "sigma_gap": [0.01, 0.01],
            "gap": [0.0, 0.0],
            "gap_idio": [0.0, 0.0],
            "topix_night": [0.0, 0.0],
            "market_vol_20d": [0.0, 0.0],
            "score_x_gap": [0.0, 0.0],
            "score_x_gap_idio": [0.0, 0.0],
            "abs_score": [1.0, 1.0],
            "abs_gap": [0.0, 0.0],
        }
    )
    model = MLOrderOverlayModel(
        lgbm=_DummyLGBM(),
        cont_cols=list(features.columns),
        target_std=1.0,
        use_ticker=False,
        use_classification=False,
        per_ticker_interactions=False,
    )
    p_trade = _predict_p_trade(features, model)
    assert np.allclose(p_trade, [0.5, 0.5])


def test_apply_overlay_adjusts_scores_and_weights():
    trade_date = "2023-03-27"
    date = pd.Timestamp(trade_date)
    df_exec = _make_df_exec(date)
    scores = np.linspace(-1.0, 1.0, N_J)
    result = _base_result(scores)
    # Provide a non-zero market-neutral w_final so the overlay has long/short sides to scale.
    long_thr = np.percentile(scores, 100 * (1 - 5 / N_J))
    short_thr = np.percentile(scores, 100 * (5 / N_J))
    w_init = np.zeros(N_J)
    w_init[scores >= long_thr] = 1.0
    w_init[scores <= short_thr] = -1.0
    if np.sum(np.abs(w_init)) > 0:
        w_init *= 2.0 / np.sum(np.abs(w_init))
    result["w_final"] = w_init

    model = MLOrderOverlayModel(
        lgbm=_DummyLGBM(),
        cont_cols=[
            "score",
            "mu_gap",
            "sigma_gap",
            "gap",
            "gap_idio",
            "topix_night",
            "market_vol_20d",
            "score_x_gap",
            "score_x_gap_idio",
            "abs_score",
            "abs_gap",
        ],
        target_std=1.0,
        use_ticker=False,
        use_classification=False,
        per_ticker_interactions=False,
    )

    out = apply_overlay(result, df_exec, model, trade_date)

    assert out is not result
    assert "scores_overlay" in out
    assert np.allclose(out["scores_overlay"], scores * 0.5, atol=1e-6)
    assert out["summary"]["overlay_applied"] == 1
    assert out["numerical"]["status"] == "PASSED"
    assert abs(float(np.sum(np.abs(out["w_final"]))) - 2.0) < 1e-6
    assert abs(float(np.sum(out["w_final"]))) < 1e-6


def test_apply_overlay_skips_when_fallback_active():
    result = _base_result(np.linspace(-1.0, 1.0, N_J))
    result["fallback"]["gap_data_missing"] = True
    df_exec = _make_df_exec(pd.Timestamp("2023-03-27"))
    model = MLOrderOverlayModel(
        lgbm=_DummyLGBM(),
        cont_cols=["score"],
        target_std=1.0,
        use_ticker=False,
        use_classification=False,
        per_ticker_interactions=False,
    )
    out = apply_overlay(result, df_exec, model, "2023-03-27")
    assert out is result


def test_apply_overlay_skips_when_date_missing():
    result = _base_result(np.linspace(-1.0, 1.0, N_J))
    df_exec = _make_df_exec(pd.Timestamp("2023-03-27"))
    model = MLOrderOverlayModel(
        lgbm=_DummyLGBM(),
        cont_cols=["score"],
        target_std=1.0,
        use_ticker=False,
        use_classification=False,
        per_ticker_interactions=False,
    )
    out = apply_overlay(result, df_exec, model, "2023-03-28")
    assert out is result


def test_apply_overlay_falls_back_on_numerical_audit_failure():
    trade_date = "2023-03-27"
    date = pd.Timestamp(trade_date)
    df_exec = _make_df_exec(date)
    scores = np.linspace(-1.0, 1.0, N_J)
    result = _base_result(scores)
    # Make Omega_gap non-symmetric so numerical audit fails.
    result["Omega_gap"] = np.eye(N_J)
    result["Omega_gap"][0, 1] = 1.0

    model = MLOrderOverlayModel(
        lgbm=_DummyLGBM(),
        cont_cols=[
            "score",
            "mu_gap",
            "sigma_gap",
            "gap",
            "gap_idio",
            "topix_night",
            "market_vol_20d",
            "score_x_gap",
            "score_x_gap_idio",
            "abs_score",
            "abs_gap",
        ],
        target_std=1.0,
        use_ticker=False,
        use_classification=False,
        per_ticker_interactions=False,
    )
    out = apply_overlay(result, df_exec, model, trade_date)
    assert out is result
