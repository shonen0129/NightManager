"""Capture and verify baseline V2 behavior."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from leadlag.data.tickers import JP_TICKERS
from leadlag.execution.config import load_config_from_yaml
from leadlag.models.production_v2 import ProductionV2Model

BASELINE_VERSION = "v20260813"


def _build_current_prices_from_df_exec(
    df_exec: pd.DataFrame,
    trade_date: str,
) -> dict[str, float] | None:
    """Build 09:10 current prices dict from ``jp_open_trade_*`` columns.

    ``preprocessor.py`` writes ``jp_open_trade_{ticker}`` (09:10 midpoint or
    open) for each JP ticker. If the column is missing, fall back to
    ``jp_close_{ticker} * (1 + jp_gap_{ticker})``. Return None when neither
    source is available so that the caller can fall back to the file cache.
    """
    if trade_date not in df_exec.index:
        return None
    row = df_exec.loc[trade_date]
    prices = {}
    for t in JP_TICKERS:
        open_col = f"jp_open_trade_{t}"
        gap_col = f"jp_gap_{t}"
        close_col = f"jp_close_{t}"
        if open_col in row.index and not pd.isna(row[open_col]):
            prices[t] = float(row[open_col])
        elif gap_col in row.index and not pd.isna(row[gap_col]) \
                and close_col in row.index and not pd.isna(row[close_col]):
            prices[t] = float(row[close_col]) * (1.0 + float(row[gap_col]))
    if len(prices) == len(JP_TICKERS):
        return prices
    return None


def _capture_v2_snapshot(
    df_exec: pd.DataFrame,
    trade_date: str,
    gap_input_dir: Path,
    current_prices: dict[str, float] | None = None,
    config_path: str = "configs/production/production.yaml",
) -> dict:
    app_config = load_config_from_yaml(config_path)
    model = ProductionV2Model(app_config.v2)
    result = model.decide(
        trade_date=trade_date,
        gap_input_dir=gap_input_dir,
        df_exec=df_exec,
        current_prices=current_prices,
    )
    return {
        "w_final": result["w_final"].tolist(),
        "scores": result["scores"].tolist(),
        "pit_binning": result["pit_binning"],
        "summary": {k: v for k, v in result["summary"].items() if k not in (
            "trade_date", "version", "candidate"
        )},
    }


def test_v2_snapshot_matches_baseline(
    regression_baseline_dir: Path,
    regression_df_exec: pd.DataFrame,
) -> None:
    """Compare current model output against the captured baseline."""
    baseline_file = regression_baseline_dir / f"v2_snapshot_{BASELINE_VERSION}.json"

    if not baseline_file.exists():
        pytest.skip(f"Baseline file not found: {baseline_file}")

    with open(baseline_file) as f:
        baseline = json.load(f)

    # Use the last available trade date from the fixture for the test.
    trade_date = str(regression_df_exec.index[-1].date())
    current_prices = _build_current_prices_from_df_exec(
        regression_df_exec, trade_date
    )
    snapshot = _capture_v2_snapshot(
        regression_df_exec,
        trade_date,
        regression_baseline_dir,
        current_prices=current_prices,
    )

    np.testing.assert_allclose(
        snapshot["w_final"], baseline["w_final"], atol=1e-12,
        err_msg="w_final mismatch against baseline",
    )
    np.testing.assert_allclose(
        snapshot["scores"], baseline["scores"], atol=1e-12,
        err_msg="scores mismatch against baseline",
    )
    assert snapshot["pit_binning"]["assigned_bin"] == baseline["pit_binning"]["assigned_bin"]
    assert snapshot["pit_binning"]["multiplier"] == pytest.approx(
        baseline["pit_binning"]["multiplier"], abs=1e-12
    )
