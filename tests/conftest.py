"""Shared pytest fixtures for leadlag tests."""

from __future__ import annotations

from pathlib import Path
import sys
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.fetcher import download_data
from leadlag.data.preprocessor import preprocess_data
from leadlag.data.tickers import TOPIX_TICKER
from leadlag.models.sre import SectorRelativeEnsembleModel


@pytest.fixture
def sample_config_dict() -> dict:
    """Return default dictionary configuration for testing."""
    return {
        "model": {"name": "sector_relative_ensemble"},
        "portfolio": {"weight_mode": "signal", "long_short_frac": 0.3},
        "ensemble": {"p0_weight": 0.5, "p3_weight": 0.5, "normalization": "zscore"},
        "costs": {"slippage_bps_per_side": 5.0},
        "residualization": {"enabled_for_p3": True, "beta_window": 60}
    }


@pytest.fixture
def sample_df_exec() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and preprocess market data for testing, returning (df_exec, raw_data)."""
    import pandas as pd
    raw_data = download_data(beta_window=60)
    df_exec = preprocess_data(raw_data, beta_window=60)

    # Compute TOPIX returns
    topix_close = raw_data["jp_close"][TOPIX_TICKER].copy()
    topix_open = raw_data["jp_open"][TOPIX_TICKER].copy()
    topix_close.index = pd.to_datetime(topix_close.index).tz_localize(None).normalize()
    topix_open.index = pd.to_datetime(topix_open.index).tz_localize(None).normalize()
    r_topix_oc = topix_close / topix_open - 1.0
    df_exec["topix_oc_return"] = r_topix_oc.reindex(df_exec.index).values
    df_exec["topix_cc_trade"] = (1.0 + df_exec["topix_night_return"]) * (1.0 + df_exec["topix_oc_return"]) - 1.0

    return df_exec, raw_data


@pytest.fixture
def sample_model(sample_config_dict) -> SectorRelativeEnsembleModel:
    """Return a model initialized with the sample config dict."""
    return SectorRelativeEnsembleModel(sample_config_dict)
