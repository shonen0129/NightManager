"""Shared pytest fixtures for leadlag tests."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.preprocessor import preprocess_data
from leadlag.data.tickers import JP_TICKERS, JP_TICKERS_WITH_TOPIX, US_TICKERS


def pytest_collection_modifyitems(config: pytest.Config, items: list[Any]) -> None:
    """Automatically assign pytest markers based on test file location.

    This avoids manually decorating every test file and keeps marker categories
    consistent with ``pyproject.toml``:
    - unit: tests/unit, tests/features
    - integration: tests/integration, tests/regression, tests/research
    - slow: research/regression tests and any test in a sprint file
    - leak: tests whose name contains 'leak' or 'leakage'
    """
    for item in items:
        nodeid = item.nodeid
        if "tests/unit" in nodeid or "tests/features" in nodeid:
            item.add_marker("unit")
        if "tests/integration" in nodeid or "tests/regression" in nodeid or "tests/research" in nodeid:
            item.add_marker("integration")
        if (
            "tests/research" in nodeid
            or "tests/regression" in nodeid
            or "/test_sprint" in nodeid
        ):
            item.add_marker("slow")
        if "leak" in nodeid.lower():
            item.add_marker("leak")


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


def _make_synthetic_raw_data(
    seed: int = 42,
    start: str = "2009-01-05",
    end: str = "2026-08-01",
) -> dict:
    """Build deterministic raw OHLC data for all tickers without network calls."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(start, end)
    n = len(dates)

    def _prices(tickers: list[str]) -> pd.DataFrame:
        rets = rng.normal(0.0002, 0.015, (n, len(tickers)))
        prices = 1000.0 * np.exp(np.cumsum(rets, axis=0))
        return pd.DataFrame(prices, index=dates, columns=tickers)

    us_close = _prices(US_TICKERS)
    jp_close = _prices(JP_TICKERS_WITH_TOPIX)
    # Open is prior close plus a tiny noise to avoid zero/negative opens.
    jp_open = jp_close.shift(1).fillna(1000.0) * (
        1.0 + rng.normal(0.0, 0.002, (n, len(JP_TICKERS_WITH_TOPIX)))
    )
    return {"us_close": us_close, "jp_close": jp_close, "jp_open": jp_open}


@pytest.fixture(scope="session")
def sample_df_exec() -> tuple[pd.DataFrame, dict]:
    """Load and preprocess market data for testing, returning (df_exec, raw_data)."""
    raw_data = _make_synthetic_raw_data()
    df_exec = preprocess_data(raw_data, beta_window=60)
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
    from leadlag.data.tickers import US_TICKERS

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
        ("jp_beta", JP_TICKERS),
    ]:
        for tk in tickers:
            df[f"{family}_{tk}"] = rng.normal(0, 0.01, n_rows)

    # Prices must be strictly positive
    for family in ("jp_close_sig", "jp_open_trade"):
        for tk in JP_TICKERS:
            df[f"{family}_{tk}"] = 1000.0 + rng.normal(0, 10.0, n_rows)

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
