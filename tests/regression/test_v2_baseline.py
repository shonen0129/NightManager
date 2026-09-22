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
from leadlag.utils.gap_provenance import bundle_identity

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
    trade_date: str,
    gap_input_dir: Path,
    config_path: str = "configs/production/production.yaml",
) -> dict:
    app_config = load_config_from_yaml(config_path)
    model = ProductionV2Model(app_config.v2)
    result = model.decide(
        trade_date=trade_date,
        gap_input_dir=gap_input_dir,
        # The offline regression bundle is the source of truth for this
        # snapshot.  Passing the mutable typed adapter here would force the
        # test to recompute a platform-sensitive frame fingerprint and can
        # turn a valid legacy bundle into a flat fallback.  Input identity is
        # checked separately against the sidecar below.
    )
    return {
        "w_final": result.w_final.tolist(),
        "scores": result.scores.tolist(),
        "pit_binning": result.pit_binning,
        "summary": {k: v for k, v in result.summary.items() if k not in (
            "trade_date", "version", "candidate"
        )},
    }


def test_v2_snapshot_matches_baseline(
    regression_baseline_dir: Path,
    regression_df_exec: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compare current model output against the captured baseline."""
    # Regression tests are offline and must not depend on a mutable macro data
    # download.  The production path still enables the feature; this bundle
    # records the explicit no-data fallback used at capture time.
    from leadlag.data import macro as macro_data
    monkeypatch.setattr(
        macro_data,
        "load_macro_prices",
        lambda *args, **kwargs: None,
    )
    baseline_file = regression_baseline_dir / f"v2_snapshot_{BASELINE_VERSION}.json"

    if not baseline_file.exists():
        pytest.skip(f"Baseline file not found: {baseline_file}")

    with open(baseline_file) as f:
        baseline = json.load(f)

    # The baseline bundle is captured for this exact trade date.  Selecting
    # the local cache's latest row makes the test depend on mutable data and
    # silently mismatches the bundled matrices.
    trade_date = "2026-08-14"
    if pd.Timestamp(trade_date) not in regression_df_exec.index:
        pytest.skip(f"Regression fixture does not contain fixed date {trade_date}")
    effective_config = load_config_from_yaml("configs/production/production.yaml").v2
    expected_identity = bundle_identity(
        regression_df_exec,
        trade_date,
        config=effective_config,
        model=None,
    )
    with open(regression_baseline_dir / "matrices" / ".mu_gap_20260814.bundle.json") as f:
        bundle_metadata = json.load(f)
    for field in ("input_version", "model_version", "config_version", "ticker_order"):
        assert bundle_metadata[field] == expected_identity[field], f"bundle identity mismatch: {field}"
    snapshot = _capture_v2_snapshot(
        trade_date,
        regression_baseline_dir,
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
