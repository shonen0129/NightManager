"""Shared pytest fixtures for leadlag tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.fetcher import download_data
from leadlag.data.preprocessor import preprocess_data
from leadlag.data.tickers import TOPIX_TICKER


@pytest.fixture
def sample_config_dict() -> dict:
    """Return default dictionary configuration for testing."""
    return {
        "model": {"name": "sector_relative_ensemble"},
        "portfolio": {"weight_mode": "signal", "long_short_frac": 0.3},
        "ensemble": {"raw_pca_weight": 0.5, "residual_pca_weight": 0.5, "normalization": "zscore"},
        "costs": {"slippage_bps_per_side": 5.0},
        "residualization": {"enabled_for_p3": True, "beta_window": 60}
    }


@pytest.fixture(scope="session")
def sample_df_exec() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and preprocess market data for testing, returning (df_exec, raw_data)."""
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


@pytest.fixture(scope="session")
def synthetic_df_exec() -> pd.DataFrame:
    """Return a small, deterministic, fully-populated ``df_exec`` for fast unit tests.

    This fixture avoids the expensive ``download_data()`` / ``preprocess_data()``
    path by synthesizing all required columns with random but finite returns.
    It is **not** representative of real market dynamics; use it for smoke tests
    and code-path coverage only.
    """
    from leadlag.data.schema import all_expected_columns
    from leadlag.data.tickers import JP_TICKERS, US_TICKERS

    rng = np.random.RandomState(42)
    n_rows = 252
    dates = pd.bdate_range("2015-01-05", periods=n_rows)

    df = pd.DataFrame(index=dates, columns=all_expected_columns(), dtype=float)
    df["sig_date"] = dates
    df["is_provisional"] = 0

    for family, tickers in [
        ("us_cc", US_TICKERS),
        ("jp_cc", JP_TICKERS),
        ("jp_oc", JP_TICKERS),
        ("jp_gap", JP_TICKERS),
        ("jp_close_sig", JP_TICKERS),
        ("jp_open_trade", JP_TICKERS),
        ("jp_beta", JP_TICKERS),
    ]:
        for tk in tickers:
            df[f"{family}_{tk}"] = rng.normal(0, 0.01, n_rows)

    for col in ["topix_night_return", "topix_oc_return", "topix_cc_trade"]:
        df[col] = rng.normal(0, 0.01, n_rows)

    # Ensure residualization columns exist for tests that expect them.
    df = df[all_expected_columns()].copy()
    return df


@pytest.fixture(scope="session")
def residual_blpx_prod_config() -> dict:
    """Return Residual-BLPX production configuration dict for testing."""
    config_path = ROOT / "configs" / "archive" / "production_residual_blpx.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)
