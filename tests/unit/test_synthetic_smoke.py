"""Fast smoke tests using the ``synthetic_df_exec`` fixture.

These tests exercise the production pipeline on synthetic data without
hitting the network or the real-data download cache.  They are meant to catch
regressions in code paths, shape contracts, and fallback behaviour quickly.
"""

from __future__ import annotations

import pandas as pd
import pytest

from leadlag.config.schemas import AppConfig
from leadlag.data.schema import ExecutionFrame, validate_frame
from leadlag.data.tickers import JP_TICKERS
from leadlag.models.production_v2 import ProductionV2Model


def test_synthetic_df_exec_passes_schema_validation(synthetic_df_exec: pd.DataFrame):
    assert validate_frame(synthetic_df_exec, required=False) == []


def test_synthetic_df_exec_can_wrap_as_execution_frame(synthetic_df_exec: pd.DataFrame):
    frame = ExecutionFrame(synthetic_df_exec)
    assert frame.us_cc().shape[0] == len(synthetic_df_exec)
    assert frame.us_cc().shape[1] == frame.n_us
    assert frame.jp_oc().shape[1] == frame.n_jp


def test_production_v2_flat_fallback_smoke(residual_blpx_prod_config: dict):
    cfg = AppConfig.model_validate(residual_blpx_prod_config)
    model = ProductionV2Model(cfg.v2)
    trade_date = "2025-12-30"
    # No gap matrices are provided, so the model should fall back to flat weights.
    result = model.decide(trade_date, gap_input_dir=None)

    w_final = result["w_final"]
    assert len(w_final) == len(JP_TICKERS)
    # Flat fallback: all weights are zero because gap matrices are unavailable.
    assert max(abs(w_final)) == pytest.approx(0.0, abs=1e-12)
    assert result.get("fallback", {}).get("gap_data_missing") is True
