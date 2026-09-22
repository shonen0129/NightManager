"""Regression tests for horizon-aware on-demand gap inputs."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from leadlag.data.tickers import JP_TICKERS, US_TICKERS
from leadlag.models.v2 import gap_io


def test_ondemand_horizon_uses_cumulative_gap_and_overnight_inputs(monkeypatch) -> None:
    dates = pd.date_range("2026-01-05", periods=3, freq="B")
    frame = pd.DataFrame(index=dates)
    for ticker in US_TICKERS:
        frame[f"us_cc_{ticker}"] = 0.0
    for ticker in JP_TICKERS:
        frame[f"jp_gap_{ticker}"] = [0.01, 0.02, 0.04]
        frame[f"jp_beta_{ticker}"] = 1.0
        frame[f"jp_open_trade_{ticker}"] = 1.0
        frame[f"jp_oc_{ticker}"] = 0.01
        frame[f"jp_close_sig_{ticker}"] = 1.0
    frame["topix_night_return"] = [0.001, 0.002, 0.003]

    captured: dict[str, np.ndarray | float] = {}

    def fake_gap_distribution(result, **kwargs):
        captured["gap"] = np.asarray(kwargs["gap_override"], dtype=float)
        captured["beta"] = np.asarray(kwargs["betas_t"], dtype=float)
        captured["topix"] = float(kwargs["topix_night_t"])
        return SimpleNamespace(mu_gap=np.zeros(len(JP_TICKERS)), omega_gap=np.eye(len(JP_TICKERS)))

    monkeypatch.setattr(gap_io, "compute_gap_distribution", fake_gap_distribution)

    class FakeBLPX:
        gap_open_coef = 0.7
        topix_beta_coef = 0.6
        gap_open_coef_neg = None
        topix_beta_coef_neg = None

        def _prepare_common_inputs(self, *args, **kwargs):
            return {
                "jp_res_returns_p3": np.zeros((len(frame), len(US_TICKERS) + len(JP_TICKERS))),
                "v0_static": np.zeros(len(JP_TICKERS)),
                "c_full_p3": np.eye(len(US_TICKERS) + len(JP_TICKERS)),
            }

        def compute_blp_signal(self, *args, **kwargs):
            return {"z_U_t": np.ones(len(US_TICKERS))}

    model = SimpleNamespace(_blpx_model=FakeBLPX())
    open_910 = pd.DataFrame(0.0, index=dates, columns=JP_TICKERS)

    gap_io._compute_ondemand(
        model,
        dates[-1].strftime("%Y-%m-%d"),
        frame,
        {},
        horizon=3,
        open_910_returns=open_910,
        allow_implicit_io=False,
    )

    expected_gap = (1.01 * 1.02 * 1.04) - 1.0
    expected_topix = (1.001 * 1.002 * 1.003) - 1.0
    np.testing.assert_allclose(captured["gap"], expected_gap)
    np.testing.assert_allclose(captured["beta"], 1.0)
    assert captured["topix"] == expected_topix

    # A live h=3 decision must replace only the current row with the PIT
    # snapshot while retaining the preceding historical rows.
    snapshot = SimpleNamespace(
        jp_gap_returns=np.full(len(JP_TICKERS), 0.10),
        jp_betas=np.full(len(JP_TICKERS), 2.0),
        topix_night_return=0.01,
    )
    gap_io._compute_ondemand(
        model,
        dates[-1].strftime("%Y-%m-%d"),
        frame,
        {},
        horizon=3,
        snapshot=snapshot,
        open_910_returns=open_910,
        allow_implicit_io=False,
    )
    np.testing.assert_allclose(captured["gap"], (1.01 * 1.02 * 1.10) - 1.0)
    np.testing.assert_allclose(captured["beta"], 2.0)
    assert captured["topix"] == (1.001 * 1.002 * 1.01) - 1.0
