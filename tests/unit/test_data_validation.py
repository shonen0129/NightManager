"""Tests for the data validation gates."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from leadlag.data.tickers import JP_TICKERS, US_TICKERS
from leadlag.data.validation import (
    DataValidationError,
    validate_exec_record,
    validate_gap_matrices,
    validate_raw_data_sources,
)


def test_validate_raw_data_sources_missing_key():
    with pytest.raises(DataValidationError, match="missing required keys"):
        validate_raw_data_sources({})


def test_validate_raw_data_sources_missing_tickers():
    us = pd.DataFrame(index=pd.date_range("2024-01-01", periods=3))
    jp = pd.DataFrame(
        {tk: [1.0, 1.0, 1.0] for tk in JP_TICKERS + ["1306.T"]},
        index=pd.date_range("2024-01-01", periods=3),
    )
    jp_open = jp.copy()
    alerts = validate_raw_data_sources({
        "us_close": us,
        "jp_close": jp,
        "jp_open": jp_open,
    })
    assert any("us_close missing tickers" in a for a in alerts)


def test_validate_exec_record_clean():
    record = {f"us_cc_{tk}": 0.01 for tk in US_TICKERS}
    for tk in JP_TICKERS:
        for prefix in ("jp_cc_", "jp_gap_", "jp_open_trade_", "jp_close_sig_"):
            record[f"{prefix}{tk}"] = 0.01
    record["jp_oc_1617.T"] = np.nan
    record["trade_date"] = "2024-01-01"
    assert validate_exec_record(record) == []


def test_validate_exec_record_detects_nan():
    record = {f"us_cc_{tk}": 0.01 for tk in US_TICKERS}
    for tk in JP_TICKERS:
        for prefix in ("jp_cc_", "jp_gap_", "jp_open_trade_", "jp_close_sig_"):
            record[f"{prefix}{tk}"] = 0.01
    record["jp_gap_1617.T"] = np.nan
    record["trade_date"] = "2024-01-01"
    alerts = validate_exec_record(record)
    assert any("jp_gap missing for 1617.T" in a for a in alerts)


def test_validate_gap_matrices_clean():
    n_j = len(JP_TICKERS)
    rng = np.random.default_rng(42)
    cov = rng.standard_normal((n_j, n_j))
    omega = cov @ cov.T + np.eye(n_j) * 1e-4
    mu = rng.standard_normal(n_j)
    assert validate_gap_matrices(mu, omega, n_j=n_j) == []


def test_validate_gap_matrices_missing():
    assert validate_gap_matrices(None, None, n_j=len(JP_TICKERS)) == [
        "Both mu and Omega gap matrices are missing"
    ]


def test_validate_gap_matrices_bad_shape():
    n_j = len(JP_TICKERS)
    mu = np.zeros(n_j - 1)
    omega = np.eye(n_j)
    alerts = validate_gap_matrices(mu, omega, n_j=n_j)
    assert any("mu shape" in a for a in alerts)


def test_validate_gap_matrices_non_psd():
    n_j = len(JP_TICKERS)
    omega = np.eye(n_j)
    omega[0, 0] = -1.0
    alerts = validate_gap_matrices(np.zeros(n_j), omega, n_j=n_j)
    assert any("positive semi-definite" in a for a in alerts)


def test_validate_gap_matrices_non_finite():
    n_j = len(JP_TICKERS)
    mu = np.full(n_j, np.nan)
    omega = np.eye(n_j)
    alerts = validate_gap_matrices(mu, omega, n_j=n_j)
    assert any("mu contains non-finite" in a for a in alerts)
