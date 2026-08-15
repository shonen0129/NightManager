"""Prototype & Experiment: Unified Convex Optimization Portfolio Optimizer.

Formulates and solves the single-stage convex optimization problem:
    max_w [ w^T * mu_gap - (lambda_risk / 2) * w^T * Omega_gap * w - lambda_cost * Cost(w, w_prev) ]
    s.t. sum(w) == 0 (Market Neutral)
         sum(|w|) <= Gross_target
         -w_max <= w_j <= w_max

Compares resulting weights against the existing heuristic top-5/bottom-5 ranking.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from leadlag.config.schemas import ProductionV2RunConfig
from leadlag.data.market_data_cache import load_df_exec_from_local_cache
from leadlag.data.tickers import JP_TICKERS
from leadlag.execution.config import load_config_from_yaml
from leadlag.models.blpx import ProductionBLPXModel
from leadlag.models.production_v2 import (
    ProductionV2Model,
    _build_current_prices_from_df_exec,
)


def solve_convex_portfolio(
    mu_gap: np.ndarray,
    omega_gap: np.ndarray,
    w_prev: np.ndarray | None = None,
    gross_target: float = 2.0,
    max_single_weight: float = 0.30,
    lambda_risk: float = 1.0,
    cost_bps: float = 5.0,
    turnover_penalty: float = 0.0001,
) -> np.ndarray:
    """Solve the single-stage convex optimization problem via SLSQP.
    
    Returns:
        w_opt: Optimized weight vector (n_j,)
    """
    n_j = len(mu_gap)
    if w_prev is None:
        w_prev = np.zeros(n_j)
        
    cost_coeff = cost_bps * 1e-4 + turnover_penalty

    # Negative utility function for minimization
    def objective(w: np.ndarray) -> float:
        # Alpha utility
        alpha_term = np.dot(w, mu_gap)
        # Risk penalty (variance)
        risk_term = 0.5 * lambda_risk * np.dot(w, np.dot(omega_gap, w))
        # Transaction cost & turnover penalty
        cost_term = cost_coeff * np.sum(np.abs(w - w_prev))
        # Total utility to maximize -> minimize negative
        return -(alpha_term - risk_term - cost_term)

    # Gradient for faster and more accurate convergence
    def jacobian(w: np.ndarray) -> float:
        d_alpha = mu_gap
        d_risk = lambda_risk * np.dot(omega_gap, w)
        d_cost = cost_coeff * np.sign(w - w_prev)
        return -(d_alpha - d_risk - d_cost)

    # Constraints
    # 1. Market neutrality: sum(w) = 0
    cons = [
        {"type": "eq", "fun": lambda w: np.sum(w)},
        {"type": "ineq", "fun": lambda w: gross_target - np.sum(np.abs(w))},
    ]

    # Bounds: -max_single_weight <= w_j <= max_single_weight
    bounds = [(-max_single_weight, max_single_weight) for _ in range(n_j)]

    # Initial guess: zero or sign of mu_gap
    w0 = np.zeros(n_j)

    res = minimize(
        objective,
        w0,
        jac=jacobian,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"ftol": 1e-9, "maxiter": 200},
    )

    if not res.success:
        # Fallback to zero
        return np.zeros(n_j)

    w_opt = res.x
    # Post-cleanup: zero out tiny weights (< 1e-4) and rebalance sum to 0
    w_opt[np.abs(w_opt) < 1e-4] = 0.0
    net_exposure = np.sum(w_opt)
    if abs(net_exposure) > 1e-6:
        # Adjust active weights to preserve sum = 0
        active = np.abs(w_opt) > 0
        if np.sum(active) > 0:
            w_opt[active] -= net_exposure / np.sum(active)
            
    return w_opt


def main() -> None:
    print("=== Next-Gen Prototype: Unified Convex Optimization vs Heuristic Ranking ===")

    df_exec = load_df_exec_from_local_cache()
    if df_exec is None:
        print("df_exec not found.")
        return

    app_config = load_config_from_yaml("configs/production/production.yaml")
    cfg = app_config.v2
    blpx_model = ProductionBLPXModel(app_config.model_dump())
    v2_model = ProductionV2Model(cfg, blpx_model=blpx_model)

    test_date = str(df_exec.index[-20])
    current_prices = _build_current_prices_from_df_exec(df_exec, test_date)

    mu_gap, omega_gap = v2_model._compute_ondemand(
        trade_date=test_date,
        df_exec=df_exec,
        current_prices=current_prices,
        horizon=1,
    )

    # 1. Heuristic Top-5 / Bottom-5 weights
    sigma = np.sqrt(np.maximum(np.diag(omega_gap), 1e-8))
    scores = mu_gap / sigma
    sorted_idx = np.argsort(scores)
    
    w_heuristic = np.zeros(len(JP_TICKERS))
    w_heuristic[sorted_idx[-5:]] = 1.0 / 5.0  # Top 5 long (gross=1.0)
    w_heuristic[sorted_idx[:5]] = -1.0 / 5.0  # Bottom 5 short (gross=1.0)
    # Total gross = 2.0

    # 2. Convex optimization weights
    w_convex = solve_convex_portfolio(
        mu_gap=mu_gap,
        omega_gap=omega_gap,
        gross_target=2.0,
        max_single_weight=0.25,
        lambda_risk=10.0,
        cost_bps=5.0,
    )

    print(f"\nTrade Date: {test_date}")
    print(f"{'Ticker':<10} | {'mu_gap (bps)':<14} | {'vol (bps)':<10} | {'Score':<8} | {'Heuristic w':<12} | {'Convex w':<12}")
    print("-" * 75)
    for i, tk in enumerate(JP_TICKERS):
        print(f"{tk:<10} | {mu_gap[i]*10000:>12.2f} | {sigma[i]*10000:>8.2f} | {scores[i]:>8.2f} | {w_heuristic[i]:>12.4f} | {w_convex[i]:>12.4f}")

    print("-" * 75)
    print(f"Heuristic: Net={np.sum(w_heuristic):.6f}, Gross={np.sum(np.abs(w_heuristic)):.4f}, Ex-ante Return={np.dot(w_heuristic, mu_gap)*10000:.2f} bps, Ex-ante Vol={np.sqrt(np.dot(w_heuristic, np.dot(omega_gap, w_heuristic)))*10000:.2f} bps")
    print(f"Convex:    Net={np.sum(w_convex):.6f}, Gross={np.sum(np.abs(w_convex)):.4f}, Ex-ante Return={np.dot(w_convex, mu_gap)*10000:.2f} bps, Ex-ante Vol={np.sqrt(np.dot(w_convex, np.dot(omega_gap, w_convex)))*10000:.2f} bps")
    
    # Ex-ante IR
    ir_heur = np.dot(w_heuristic, mu_gap) / np.sqrt(np.dot(w_heuristic, np.dot(omega_gap, w_heuristic)))
    ir_conv = np.dot(w_convex, mu_gap) / np.sqrt(np.dot(w_convex, np.dot(omega_gap, w_convex)))
    print(f"Ex-ante IR: Heuristic={ir_heur:.4f} vs Convex={ir_conv:.4f} (Delta: +{(ir_conv - ir_heur)/ir_heur*100:.1f}%)")

if __name__ == "__main__":
    main()
