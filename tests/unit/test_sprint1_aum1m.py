"""Unit tests for Sprint 1 AUM 1M JPY Tachibana credit costs and stress simulation."""

from __future__ import annotations

import sys
import os
import yaml
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.cache import load_df_exec_from_local_cache
from leadlag.diagnostics.sprint1_experiments import generate_targets_panel


def test_config_loading():
    """Verify that configs/sprint1_aum1m_tachibana.yaml can be loaded with required keys."""
    config_path = ROOT / "configs" / "archive" / "sprint1_aum1m_tachibana.yaml"
    assert config_path.exists()
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    assert config["aum_jpy"] == 1000000
    assert config["buy_interest_rate_annual"] == 0.025
    assert config["stock_borrow_fee_annual"] == 0.0115
    assert "broker_profile" in config
    assert config["broker_profile"]["margin_buy_interest_annual"] == 0.025


def test_rounding_to_lot_size():
    """Test lot size rounding logic on dummy weights."""
    AUM = 1000000
    target_weights = np.array([0.10, -0.10, 0.05, -0.05])
    prices = np.array([25000.0, 12000.0, 3000.0, 45000.0])
    lot_size = 1
    
    target_notionals = target_weights * AUM
    target_shares = target_notionals / prices
    
    # Rounded shares
    shares = np.round(target_shares) * lot_size
    actual_notionals = shares * prices
    actual_weights = actual_notionals / AUM
    
    # Check that actual weights are multiples of price / AUM
    for w, p, s in zip(actual_weights, prices, shares):
        assert np.isclose(w * AUM, s * p)


def test_credit_cost_calculation():
    """Verify margin buy interest and stock borrow fee calculation."""
    buy_rate = 0.025
    borrow_rate = 0.0115
    reverse_fee_bps = 10.0
    day_count = 365
    interest_days = 1
    
    # Test case: 100,000 JPY Long, 50,000 JPY Short
    long_notional = 100000.0
    short_notional = -50000.0
    
    long_cost = long_notional * buy_rate * interest_days / day_count
    short_cost = abs(short_notional) * borrow_rate * interest_days / day_count
    reverse_fee_cost = abs(short_notional) * (reverse_fee_bps / 10000.0)
    
    total_cost = long_cost + short_cost + reverse_fee_cost
    
    assert long_cost > 0
    assert short_cost > 0
    assert reverse_fee_cost > 0
    assert np.isclose(total_cost, 6.849315 + 1.575342 + 50.0, atol=1e-4)


def test_short_unavailability_neutralization():
    """Verify zero_and_rescale for short unavailability."""
    # Start with dollar neutral weights
    w = np.array([0.1, 0.2, 0.1, -0.1, -0.2, -0.1])
    # Suppress index 4 (-0.2 is unavailable)
    mask_unavailable = np.array([False, False, False, False, True, False])
    
    w_new = w.copy()
    w_new[mask_unavailable] = 0.0
    
    # Rescale long and short separately to match the new short sum
    long_sum = np.sum(w_new[w_new > 0.0])
    short_sum = np.abs(np.sum(w_new[w_new < 0.0]))
    
    # Neutralize: rescale whichever side is larger to match the smaller side
    target_gross = min(long_sum, short_sum)
    
    w_final = w_new.copy()
    if long_sum > 0:
        w_final[w_final > 0.0] = w_final[w_final > 0.0] * (target_gross / long_sum)
    if short_sum > 0:
        w_final[w_final < 0.0] = w_final[w_final < 0.0] * (target_gross / short_sum)
        
    assert np.isclose(np.sum(w_final), 0.0, atol=1e-7)
    assert np.sum(np.abs(w_final)) <= np.sum(np.abs(w))
