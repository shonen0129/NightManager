import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add tools/ to path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))

from backtest_ensemble_robustness import (
    optimize_soft_beta_penalty_weights,
    apply_rank_hysteresis,
    normalize_signals,
)


def test_optimize_soft_beta_penalty_weights():
    """Verify that soft beta penalty reduces TOPIX beta exposure."""
    np.random.seed(42)
    w0 = np.array([0.4, 0.3, 0.3, 0.0, 0.0, -0.2, -0.3, -0.5, 0.0, 0.0])
    beta = np.array([1.5, 1.2, 1.1, 0.9, 1.0, 1.3, 0.7, 0.9, 1.1, 1.0])
    w_prev = np.zeros(10)

    # Solve with zero penalty
    w_zero, success_zero = optimize_soft_beta_penalty_weights(
        w0, beta, lambda_beta=0.0, lambda_turnover=0.0, w_prev=w_prev
    )
    assert success_zero
    beta_exp_zero = np.sum(w_zero * beta)

    # Solve with high beta penalty
    w_high, success_high = optimize_soft_beta_penalty_weights(
        w0, beta, lambda_beta=100.0, lambda_turnover=0.0, w_prev=w_prev
    )
    assert success_high
    beta_exp_high = np.sum(w_high * beta)

    # High penalty should significantly reduce beta exposure
    assert abs(beta_exp_high) < abs(beta_exp_zero)
    # Constraints check
    assert pytest.approx(np.sum(w_high), abs=1e-6) == 0.0
    assert np.sum(np.abs(w_high)) <= 2.000001


def test_apply_rank_hysteresis():
    """Verify that rank hysteresis maintains exactly 5 longs/shorts and prevents marginal swaps."""
    # 17 assets
    sig1 = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7])
    # Day 1: active longs are indices 0,1,2,3,4. Active shorts are 12,13,14,15,16.
    active_longs = {0, 1, 2, 3, 4}
    active_shorts = {12, 13, 14, 15, 16}

    # Day 2: sig changes slightly. Rank of index 4 (previously long) changes to 6 (so signal rank is 6).
    # Since rank is 6 (which is <= keep_long_rank=7), hysteresis should keep index 4.
    sig2 = np.array([0.9, 0.8, 0.7, 0.6, 0.35, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7])
    # The actual ranks of sig2:
    # idx 0: 0.9 (rank 1)
    # idx 1: 0.8 (rank 2)
    # idx 2: 0.7 (rank 3)
    # idx 3: 0.6 (rank 4)
    # idx 5: 0.4 (rank 5)
    # idx 4: 0.35 (rank 6)
    # Here, rank of idx 4 is 6. If we did standard top-5, we would select {0,1,2,3,5} as longs.
    # With rank hysteresis, idx 4 (previously long) should be kept because rank 6 <= 7.
    # The selected longs should still be {0,1,2,3,4} because idx 4 is kept, and we already have 5 longs.
    weights, new_longs, new_shorts = apply_rank_hysteresis(
        sig2, active_longs, active_shorts, keep_long_rank=7, keep_short_rank=11
    )

    assert len(new_longs) == 5
    assert len(new_shorts) == 5
    assert 4 in new_longs
    assert 5 not in new_longs  # idx 5 had rank 5, but was not previously long, and since we kept idx 4, we have no slots left
    assert pytest.approx(np.sum(weights), abs=1e-6) == 0.0
    assert pytest.approx(np.sum(np.abs(weights)), abs=1e-6) == 2.0


def test_normalize_signals():
    """Verify z-score and rank signal normalization methods."""
    sig = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    # z-score
    norm_z = normalize_signals(sig, "cross_sectional_zscore")
    assert np.mean(norm_z) == pytest.approx(0.0, abs=1e-6)
    assert np.std(norm_z) == pytest.approx(1.0, abs=1e-6)
    
    # rank
    norm_rank = normalize_signals(sig, "rank_normalize")
    assert np.all(norm_rank >= -1.0)
    assert np.all(norm_rank <= 1.0)
    assert np.mean(norm_rank) == pytest.approx(0.2, abs=1e-6)
