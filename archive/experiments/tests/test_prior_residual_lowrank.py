import pytest
import numpy as np
import pandas as pd

from domain.models.prior_residual_lowrank import (
    project_to_subspace,
    solve_factor_propagation,
    ResidualizedPriorSubspaceLowRankGapModel,
)
from domain.signals.lead_lag import build_v3_static


def test_prior_subspace_dimensions():
    """Verify static prior subspace (V0) shapes and slice dimensions."""
    # V0 shape should be 32 x 6 (15 US assets + 17 JP assets)
    V0_full = build_v3_static(15, 17, include_v4=True)
    assert V0_full.shape == (32, 6)

    # Slice to US (first 11 ETFs) and JP (17 JP ETFs)
    # US sector ETFs correspond to rows 0 to 11
    # JP sector ETFs correspond to rows 15 to 32
    V0_U = V0_full[0:11, :]
    V0_J = V0_full[15:32, :]

    assert V0_U.shape == (11, 6)
    assert V0_J.shape == (17, 6)


def test_project_to_subspace_shape():
    """Verify projection maps standard data to K_prior factor coordinate space."""
    np.random.seed(42)
    n_samples = 50
    n_assets = 11
    k_prior = 6

    V0_U = np.random.randn(n_assets, k_prior)
    data = np.random.randn(n_samples, n_assets)

    proj = project_to_subspace(data, V0_U, eps=1e-8)
    assert proj.shape == (n_samples, k_prior)
    assert np.all(np.isfinite(proj))


def test_solve_factor_propagation():
    """Verify Ridge factor propagation optimizer solver and contraction to A0."""
    np.random.seed(42)
    n_samples = 100
    k_prior = 6

    F = np.random.randn(n_samples, k_prior)
    G = np.random.randn(n_samples, k_prior)
    A0 = np.eye(k_prior)

    # Test with zero regularization
    A_zero = solve_factor_propagation(F, G, lambda_A=0.0, lambda_prior=0.0, A0=A0)
    # Ordinary least squares solution: inv(F.T * F) * F.T * G
    OLS = np.linalg.inv(F.T @ F) @ (F.T @ G)
    assert pytest.approx(A_zero, abs=1e-7) == OLS

    # Test contraction to A0 with high lambda_prior
    A_high_prior = solve_factor_propagation(F, G, lambda_A=0.0, lambda_prior=1e6, A0=A0)
    assert pytest.approx(A_high_prior, abs=1e-3) == A0


def test_prior_subspace_gap_model_predict():
    """Verify fitting and prediction dimensions, outputs, and B_eff properties."""
    np.random.seed(42)
    n_samples = 100
    n_us = 11
    n_jp = 17
    k_prior = 6

    X_train = np.random.randn(n_samples, n_us)
    Y_train = np.random.randn(n_samples, n_jp)
    x_predict = np.random.randn(n_us)

    V0_full = build_v3_static(15, 17, include_v4=True)
    V0_U = V0_full[0:11, 0:k_prior]
    V0_J = V0_full[15:32, 0:k_prior]
    A0 = np.eye(k_prior)

    model = ResidualizedPriorSubspaceLowRankGapModel(
        k_prior=k_prior,
        ridge_alpha_a=1.0,
        lambda_prior_a=1.0,
    )

    y_pred, B_eff = model.fit_predict_step(
        X_train, Y_train, x_predict, V0_U, V0_J, A0
    )

    assert y_pred.shape == (n_jp,)
    assert B_eff.shape == (n_us, n_jp)
    assert np.all(np.isfinite(y_pred))
    assert np.all(np.isfinite(B_eff))


def test_portfolio_weights_sign_and_sum():
    """Confirm portfolio weight generation rules: long = +1, short = -1, net = 0, gross = 2."""
    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(ROOT / "tools"))
    from backtest_prior_residual_lowrank_gap import build_portfolio_weights

    np.random.seed(42)
    signals = np.random.randn(17)

    weights = build_portfolio_weights(signals, q=0.3)
    assert len(weights) == 17

    longs = weights[weights > 0]
    shorts = weights[weights < 0]

    # Verify count: 30% of 17 is 5 positions
    assert len(longs) == 5
    assert len(shorts) == 5

    # Verify sum constraints (with epsilon tolerance)
    assert pytest.approx(np.sum(longs), abs=1e-6) == 1.0
    assert pytest.approx(np.sum(shorts), abs=1e-6) == -1.0
    assert pytest.approx(np.sum(weights), abs=1e-6) == 0.0
    assert pytest.approx(np.sum(np.abs(weights)), abs=1e-6) == 2.0


def test_optimize_beta_neutral_weights():
    """Verify scipy SLSQP TOPIX beta neutral optimization satisfies constraints."""
    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(ROOT / "tools"))
    from backtest_prior_model_improvements import optimize_beta_neutral_weights

    np.random.seed(42)
    w0 = np.array([0.4, 0.3, 0.3, 0.0, 0.0, -0.2, -0.3, -0.5, 0.0, 0.0])
    beta = np.array([1.2, 0.8, 1.1, 0.9, 1.0, 1.3, 0.7, 0.9, 1.1, 1.0])

    w_opt = optimize_beta_neutral_weights(w0, beta, gross_limit=2.0)

    # Constraints verification
    assert np.sum(w_opt) == pytest.approx(0.0, abs=1e-6)
    assert np.sum(w_opt * beta) == pytest.approx(0.0, abs=1e-6)

    # Sign consistency
    for i, val in enumerate(w0):
        if val > 0:
            assert w_opt[i] >= -1e-8
        elif val < 0:
            assert w_opt[i] <= 1e-8
        else:
            assert abs(w_opt[i]) <= 1e-8

    # Gross exposure (with sign-consistent weights, sum(abs(w_opt)) should be 2.0)
    assert np.sum(np.abs(w_opt)) == pytest.approx(2.0, abs=1e-6)


def test_normalize_signals():
    """Verify signal normalization methods (z-score, rank, none)."""
    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(ROOT / "tools"))
    from backtest_prior_model_improvements import normalize_signals

    np.random.seed(42)
    sig = np.random.randn(10)

    # zscore
    norm_z = normalize_signals(sig, "cross_sectional_zscore")
    assert np.mean(norm_z) == pytest.approx(0.0, abs=1e-6)
    assert np.std(norm_z) == pytest.approx(1.0, abs=1e-6)

    # rank
    norm_rank = normalize_signals(sig, "rank_normalize")
    assert np.all(norm_rank >= -1.0)
    assert np.all(norm_rank <= 1.0)
    assert np.mean(norm_rank) == pytest.approx(0.1, abs=1e-6)

    # none
    norm_none = normalize_signals(sig, "none")
    assert np.all(norm_none == sig)


def test_volatility_targeting_series():
    """Verify vol targeting on a series of returns."""
    import numpy as np
    import pandas as pd

    np.random.seed(42)
    returns = np.random.randn(100) * 0.01  # 1% daily return volatility
    w_size = 20
    target_vol = 0.15
    max_scale = 1.0

    # Emulate the script's calculation
    realized_vol = pd.Series(returns).rolling(w_size).std().shift(1).fillna(0.01).values * np.sqrt(252.0)
    scales = np.minimum(max_scale, target_vol / np.maximum(realized_vol, 1e-4))

    assert len(scales) == 100
    assert np.all(scales > 0)
    assert np.all(scales <= max_scale)


def test_no_trade_buffer_logic():
    """Verify that no-trade buffer limits turnover and preserves dollar-neutrality and gross constraints."""
    no_trade_buffer = 0.1
    n_targets = 4
    w_prev = np.array([0.5, 0.5, -0.5, -0.5])
    # Target weights change slightly (less than buffer) for all elements
    w_target = np.array([0.55, 0.45, -0.45, -0.55])  # diffs: 0.05, 0.05, 0.05, 0.05

    w_actual = np.zeros(n_targets)
    for j in range(n_targets):
        if abs(w_target[j] - w_prev[j]) < no_trade_buffer:
            w_actual[j] = w_prev[j]
        else:
            w_actual[j] = w_target[j]

    # w_actual should equal w_prev
    assert np.all(w_actual == w_prev)

    # Test case where some assets change by more than buffer
    w_target2 = np.array([0.7, 0.5, -0.5, -0.7])  # diffs: 0.2, 0.0, 0.0, 0.2

    w_actual2 = np.zeros(n_targets)
    for j in range(n_targets):
        if abs(w_target2[j] - w_prev[j]) < no_trade_buffer:
            w_actual2[j] = w_prev[j]
        else:
            w_actual2[j] = w_target2[j]

    assert w_actual2[0] == 0.7
    assert w_actual2[1] == 0.5
    assert w_actual2[2] == -0.5
    assert w_actual2[3] == -0.7

    # Apply dollar-neutral and gross target re-scaling as in simulation
    w_plus = np.maximum(w_actual2, 0.0)
    w_minus = np.minimum(w_actual2, 0.0)
    sum_plus = np.sum(w_plus)
    sum_minus = -np.sum(w_minus)
    if sum_plus > 0:
        w_plus = w_plus / sum_plus
    if sum_minus > 0:
        w_minus = w_minus / sum_minus
    w_final = w_plus + w_minus

    assert np.sum(w_final) == pytest.approx(0.0, abs=1e-6)
    assert np.sum(np.abs(w_final)) == pytest.approx(2.0, abs=1e-6)
