"""Regression test fixtures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture(scope="session")
def regression_baseline_dir() -> Path:
    path = Path(__file__).parent / "baselines"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture(scope="session")
def regression_df_exec() -> pd.DataFrame:
    """Use the cached df_exec so the regression matches the captured baseline."""
    from leadlag.data.market_data_cache import load_df_exec_from_local_cache

    df_exec = load_df_exec_from_local_cache()
    if df_exec is None or df_exec.empty:
        pytest.skip("No local df_exec cache available for regression tests")
    return df_exec
