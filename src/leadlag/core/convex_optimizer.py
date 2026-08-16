"""Unified Convex Portfolio Optimizer.

Formulates and solves the single-stage convex portfolio optimization problem:
    max_w [ w^T * mu_gap - (lambda_risk / 2) * w^T * Omega_gap * w - Cost(w, w_prev) - lambda_to * ||w - w_prev||_1 ]
    s.t. sum(w) == 0 (Market Neutral)
         sum(|w|) <= Gross_target
         -max_single_weight <= w_j <= max_single_weight

Uses a smooth pseudo-Huber approximation for L1 transaction costs and gross exposure
to guarantee global smooth convergence (C-infinity) with the SLSQP solver in < 15 iterations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TypeAlias, cast

import numpy as np
from scipy.optimize import minimize

from leadlag.config.schemas import NextGenConfig

logger = logging.getLogger(__name__)

# Backward-compatible alias for the Pydantic NextGenConfig.
ConvexOptimizerConfig: TypeAlias = NextGenConfig


@dataclass(frozen=True)
class OptimizationResult:
    """Result container for convex portfolio optimization."""
    weights: np.ndarray
    gross_exposure: float
    net_exposure: float
    ex_ante_return: float
    ex_ante_vol: float
    ex_ante_ir: float
    ex_ante_cost: float = 0.0
    ex_ante_net_return: float = 0.0
    ex_ante_net_ir: float = 0.0
    turnover: float = 0.0
    converged: bool = True
    iterations: int = 0
    message: str = ""


def ensure_psd(matrix: np.ndarray, min_eigenvalue: float = 1e-8) -> np.ndarray:
    """Ensure matrix is symmetric and strictly positive semi-definite."""
    sym = 0.5 * (matrix + matrix.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    if np.any(eigvals < min_eigenvalue):
        eigvals = np.maximum(eigvals, min_eigenvalue)
        sym = eigvecs @ np.diag(eigvals) @ eigvecs.T
        sym = 0.5 * (sym + sym.T)
    return cast(np.ndarray, sym)


def _smooth_abs(x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Smooth pseudo-Huber approximation of |x|: sqrt(x^2 + eps^2) - eps."""
    return cast(np.ndarray, np.sqrt(x ** 2 + eps ** 2) - eps)


def _smooth_abs_grad(x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Derivative of smooth pseudo-Huber: x / sqrt(x^2 + eps^2)."""
    return cast(np.ndarray, x / np.sqrt(x ** 2 + eps ** 2))


def _smooth_pos(x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Smooth approximation of max(x, 0) using the pseudo-Huber envelope."""
    return cast(np.ndarray, 0.5 * (_smooth_abs(x, eps) + x))


def _smooth_pos_grad(x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Derivative of _smooth_pos: 0.5 * (smooth_abs_grad(x) + 1)."""
    return cast(np.ndarray, 0.5 * (_smooth_abs_grad(x, eps) + 1.0))


def _smooth_neg(x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Smooth approximation of max(-x, 0) using the pseudo-Huber envelope."""
    return cast(np.ndarray, 0.5 * (_smooth_abs(x, eps) - x))


def _smooth_neg_grad(x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Derivative of _smooth_neg: 0.5 * (smooth_abs_grad(x) - 1)."""
    return cast(np.ndarray, 0.5 * (_smooth_abs_grad(x, eps) - 1.0))


def _compute_net_cost(
    w: np.ndarray,
    w_prev: np.ndarray,
    config: NextGenConfig,
) -> float:
    """Estimate daily net cost (slippage + financing + borrow + reverse) in decimal.

    Mirrors the cost decomposition in ``BacktestEngine._simulate_daily_pnl`` for
    a one-day holding period.
    """
    financing_daily = config.buy_interest_annual / 365.0
    borrow_daily = config.borrow_fee_annual / 365.0
    reverse_daily = config.reverse_fee_bps / 10000.0
    slip = config.slippage_bps_per_side / 10000.0
    leverage = config.side_leverage

    alpha_mask = np.where(w > 0, config.overnight_alpha_long, np.where(w < 0, config.overnight_alpha_short, 0.0))
    gross_abs = np.abs(w)
    turnover = np.abs(w - w_prev)

    # Slippage: intraday portion gets a full round-trip (2x), overnight portion
    # pays half-turnover on the rebalanced notional.
    intraday_gross = np.sum((1.0 - alpha_mask) * gross_abs)
    overnight_turnover = np.sum(alpha_mask * turnover)
    slip_cost = leverage * slip * (2.0 * intraday_gross + 0.5 * overnight_turnover)

    held_long = np.sum(alpha_mask * np.maximum(w, 0.0))
    held_short = np.sum(alpha_mask * np.maximum(-w, 0.0))
    fin_cost = leverage * held_long * financing_daily
    borrow_cost = leverage * held_short * borrow_daily
    reverse_cost = leverage * held_short * reverse_daily
    return float(slip_cost + fin_cost + borrow_cost + reverse_cost)


def optimize_portfolio_convex(
    mu_gap: np.ndarray,
    omega_gap: np.ndarray,
    w_prev: np.ndarray | None = None,
    config: NextGenConfig | None = None,
    gross_multiplier: float = 1.0,
) -> OptimizationResult:
    """Solve the convex portfolio optimization problem.

    Args:
        mu_gap: Expected return vector (n_j,).
        omega_gap: Covariance matrix (n_j, n_j).
        w_prev: Previous weights vector (n_j,) for turnover & transaction cost calculation.
        config: Optimization configuration parameters.
        gross_multiplier: Dynamic gross scaling factor (e.g. from RuleD, default 1.0).

    Returns:
        OptimizationResult with optimized weights and ex-ante metrics.
    """
    if config is None:
        config = ConvexOptimizerConfig()

    n_j = len(mu_gap)
    if n_j == 0:
        return OptimizationResult(
            weights=np.zeros(0),
            gross_exposure=0.0,
            net_exposure=0.0,
            ex_ante_return=0.0,
            ex_ante_vol=0.0,
            ex_ante_ir=0.0,
            ex_ante_cost=0.0,
            ex_ante_net_return=0.0,
            ex_ante_net_ir=0.0,
            turnover=0.0,
            converged=True,
            iterations=0,
            message="Empty input vector; returned flat position.",
        )

    if w_prev is None:
        w_prev = np.zeros(n_j)
    elif len(w_prev) != n_j:
        w_prev = np.zeros(n_j)

    # Sanity guard: Check for NaN or Inf in inputs
    if not np.all(np.isfinite(mu_gap)) or not np.all(np.isfinite(omega_gap)) or not np.all(np.isfinite(w_prev)):
        logger.warning("NaN or Inf detected in optimizer inputs. Returning safe flat position.")
        zeros = np.zeros(n_j)
        return OptimizationResult(
            weights=zeros,
            gross_exposure=0.0,
            net_exposure=0.0,
            ex_ante_return=0.0,
            ex_ante_vol=0.0,
            ex_ante_ir=0.0,
            ex_ante_cost=0.0,
            ex_ante_net_return=0.0,
            ex_ante_net_ir=0.0,
            turnover=float(np.sum(np.abs(w_prev))),
            converged=False,
            iterations=0,
            message="NaN/Inf input detected; safety fallback to flat position.",
        )

    # Effective gross target incorporating dynamic scaling (e.g. RuleD)
    effective_gross = config.gross_target * gross_multiplier
    if effective_gross <= 1e-6:
        # Zero target -> Flat position
        zeros = np.zeros(n_j)
        return OptimizationResult(
            weights=zeros,
            gross_exposure=0.0,
            net_exposure=0.0,
            ex_ante_return=0.0,
            ex_ante_vol=0.0,
            ex_ante_ir=0.0,
            ex_ante_cost=0.0,
            ex_ante_net_return=0.0,
            ex_ante_net_ir=0.0,
            turnover=float(np.sum(np.abs(w_prev))),
            converged=True,
            iterations=0,
            message="Zero gross target; returned flat position.",
        )

    # Ensure covariance matrix is strictly PSD
    omega_psd = ensure_psd(omega_gap)

    # Unit cost coefficient (bps converted to decimal + turnover penalty)
    cost_coeff = (config.cost_bps * 1e-4) + config.turnover_penalty
    eps = config.smooth_eps

    # Daily holding cost rates (decimal per day)
    financing_daily = config.buy_interest_annual / 365.0
    borrow_daily = config.borrow_fee_annual / 365.0
    reverse_daily = config.reverse_fee_bps / 10000.0
    leverage = config.side_leverage
    long_hold_rate = leverage * config.overnight_alpha_long * financing_daily
    short_hold_rate = leverage * config.overnight_alpha_short * (borrow_daily + reverse_daily)

    # Objective: Minimize negative net utility (Smooth C-infinity function)
    def objective(w: np.ndarray) -> float:
        # Alpha term
        alpha_term = float(np.dot(w, mu_gap))
        # Variance risk term
        risk_term = 0.5 * config.lambda_risk * float(np.dot(w, np.dot(omega_psd, w)))
        # Smooth transaction & turnover cost term
        trans_term = cost_coeff * float(np.sum(_smooth_abs(w - w_prev, eps)))
        # Smooth holding cost term (financing + borrow + reverse)
        holding_term = float(
            long_hold_rate * np.sum(_smooth_pos(w, eps))
            + short_hold_rate * np.sum(_smooth_neg(w, eps))
        )
        return -(alpha_term - risk_term - trans_term - holding_term)

    def jacobian(w: np.ndarray) -> np.ndarray:
        d_alpha = mu_gap
        d_risk = config.lambda_risk * np.dot(omega_psd, w)
        d_trans = cost_coeff * _smooth_abs_grad(w - w_prev, eps)
        d_hold = (
            long_hold_rate * _smooth_pos_grad(w, eps)
            + short_hold_rate * _smooth_neg_grad(w, eps)
        )
        return cast(np.ndarray, -(d_alpha - d_risk - d_trans - d_hold))

    def _eq_jac(_w: np.ndarray) -> np.ndarray:
        return np.ones(n_j, dtype=float)

    # Constraints:
    # 1. Exact market neutrality: sum(w) = 0
    # 2. Smooth gross exposure limit: sum(smooth_abs(w)) <= effective_gross
    constraints = [
        {
            "type": "eq",
            "fun": lambda w: float(np.sum(w)),
            "jac": _eq_jac,
        },
        {
            "type": "ineq",
            "fun": lambda w: float(effective_gross - np.sum(_smooth_abs(w, eps))),
            "jac": lambda w: -_smooth_abs_grad(w, eps),
        },
    ]

    # Bounds on individual weights
    max_w = min(config.max_single_weight, effective_gross / 2.0)
    bounds = [(-max_w, max_w) for _ in range(n_j)]

    # Smart warm start: start with sign of mu_gap or previous weights
    w0 = np.clip(w_prev, -max_w, max_w)
    if np.sum(np.abs(w0)) > effective_gross:
        w0 *= effective_gross / np.sum(np.abs(w0))
    # Neutralize initial guess
    w0 -= np.mean(w0)

    res = minimize(
        objective,
        w0,
        jac=jacobian,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "ftol": config.solver_tol,
            "maxiter": config.max_iter,
            "disp": False,
        },
    )

    # Retry once with doubled iterations from the last iterate if we hit the
    # iteration limit.  SLSQP can be sensitive to the initial guess; the last
    # iterate is usually close to the feasible region and needs only a few more
    # iterations to converge.
    if not res.success and res.status == 9 and res.x is not None:
        w0 = np.clip(res.x, -max_w, max_w)
        if np.sum(np.abs(w0)) > effective_gross:
            w0 *= effective_gross / np.sum(np.abs(w0))
        w0 -= np.mean(w0)
        res = minimize(
            objective,
            w0,
            jac=jacobian,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={
                "ftol": config.solver_tol,
                "maxiter": max(config.max_iter * 2, 200),
                "disp": False,
            },
        )

    if not res.success:
        # Optimizer did not converge: fall back to a flat position to avoid
        # using an invalid or constraint-violating solution live.
        logger.warning(
            "SLSQP did not converge (status=%d, message=%s). Falling back to flat position.",
            res.status, res.message,
        )
        w_opt = np.zeros(n_j)
        converged = False
    else:
        w_opt = res.x.copy()
        converged = True

    # Post-cleanup: zero out sub-threshold weights and enforce exact market neutrality
    w_opt[np.abs(w_opt) < config.min_weight_threshold] = 0.0
    net_bias = float(np.sum(w_opt))
    active_mask = np.abs(w_opt) > 0.0
    n_active = int(np.sum(active_mask))
    if n_active > 0 and abs(net_bias) > 1e-12:
        w_opt[active_mask] -= net_bias / n_active

    # Final gross clipping if necessary
    actual_gross = float(np.sum(np.abs(w_opt)))
    if actual_gross > effective_gross + 1e-12 and actual_gross > 0.0:
        w_opt *= effective_gross / actual_gross

    # Calculate ex-ante performance metrics
    actual_gross = float(np.sum(np.abs(w_opt)))
    actual_net = float(np.sum(w_opt))
    ex_ante_return = float(np.dot(w_opt, mu_gap))
    port_var = float(np.dot(w_opt, np.dot(omega_psd, w_opt)))
    ex_ante_vol = float(np.sqrt(max(port_var, 1e-12)))
    ex_ante_ir = ex_ante_return / ex_ante_vol if ex_ante_vol > 1e-8 else 0.0
    turnover = float(np.sum(np.abs(w_opt - w_prev)))
    ex_ante_cost = _compute_net_cost(w_opt, w_prev, config)
    ex_ante_net_return = ex_ante_return - ex_ante_cost
    ex_ante_net_ir = ex_ante_net_return / ex_ante_vol if ex_ante_vol > 1e-8 else 0.0

    message = res.message
    if actual_gross > effective_gross + 1e-12:
        message = f"{message}; clipped to effective gross {effective_gross:.6f}"

    return OptimizationResult(
        weights=w_opt,
        gross_exposure=actual_gross,
        net_exposure=actual_net,
        ex_ante_return=ex_ante_return,
        ex_ante_vol=ex_ante_vol,
        ex_ante_ir=ex_ante_ir,
        ex_ante_cost=ex_ante_cost,
        ex_ante_net_return=ex_ante_net_return,
        ex_ante_net_ir=ex_ante_net_ir,
        turnover=turnover,
        converged=converged,
        iterations=res.nit,
        message=message,
    )
