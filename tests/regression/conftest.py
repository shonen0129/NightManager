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
    """Load the immutable df_exec bundle captured with the baseline."""
    bundle = Path(__file__).parent / "baselines" / "df_exec_20260814.csv.gz"
    if not bundle.exists():
        pytest.skip(f"Regression input bundle not found: {bundle}")
    df_exec = pd.read_csv(bundle, index_col=0, parse_dates=True)
    df_exec.index.name = "trade_date"
    return df_exec
