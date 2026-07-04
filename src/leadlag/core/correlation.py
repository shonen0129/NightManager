"""Core mathematical and prior subspace calculations for the lead-lag strategy."""

from __future__ import annotations

import numpy as np

# Numeric constants
EPSILON_WEIGHT = 1e-12
EPSILON_SIGMA = 1e-8
EPSILON_VARIANCE = 1e-16
EPSILON_NORM = 1e-10


def _orthogonalize_and_normalize(
    vector: np.ndarray,
    basis: list[np.ndarray],
) -> np.ndarray:
    """Orthogonalize a vector against basis vectors and normalize it."""
    result = vector.astype(float, copy=True)
    for base in basis:
        result = result - (result @ base) * base

    norm = np.linalg.norm(result)
    if norm <= EPSILON_NORM:
        return np.zeros_like(result)
    return result / norm


def build_base_vectors(n_u: int, n_j: int) -> dict[str, np.ndarray]:
    """Build the fixed base vectors v1, v2."""
    if n_u <= 0 or n_j <= 0:
        raise ValueError(f"n_u and n_j must be positive, got n_u={n_u}, n_j={n_j}")

    n = n_u + n_j
    v1 = np.ones(n) / np.sqrt(n)

    # Group-difference vector normalized to unit norm and orthogonal to v1.
    denom = np.sqrt(float(n_u * n_j * n))
    v2 = np.zeros(n)
    v2[:n_u] = n_j / denom
    v2[n_u:] = -n_u / denom
    return {"v1": v1, "v2": v2}


def get_static_sensitivity_labels() -> dict[str, np.ndarray]:
    """Return the hardcoded sensitivity label vectors (single source of truth).

    Values use a 7-level discrete grid: {-1.0, -0.6, -0.3, 0.0, +0.3, +0.6, +1.0}.
    Order: US_TICKERS (15) then JP_TICKERS (17), total 32.

    Returns
    -------
    dict with keys 'w3' (cyclical/defensive), 'w4' (FX),
    'w5' (energy), 'w6' (inflation), each (32,) ndarray.
    """
    return {
        "w3": np.array(
            [
                1.0, 0.3, 0.3, 1.0, 1.0, 0.6, -1.0, 0.3, -1.0, -1.0, 1.0, 0.0, 0.6, -0.3, -0.6,
                -1.0, 0.3, 0.6, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -0.3, -1.0, -0.3, 0.6, -0.6, 1.0, 0.6, 0.6,
            ],
            dtype=float,
        ),
        "w4": np.array(
            [
                0.3, 0.0, 0.0, 0.3, 0.6, 1.0, -0.6, -0.3, -0.6, -0.3, 0.6, 0.3, 0.0, 0.6, -0.3,
                -0.6, 0.3, 0.3, 0.6, -0.3, 1.0, 0.6, 1.0, 1.0, -0.3, -1.0, -0.3, 1.0, -0.6, 0.3, 0.0, -1.0,
            ],
            dtype=float,
        ),
        "w5": np.array(
            [
                0.3, 0.0, 1.0, 0.0, 0.3, 0.0, -0.3, 0.0, -1.0, 0.0, -0.3, 0.0, 0.3, 0.0, -0.3,
                -0.3, 1.0, 0.0, 0.3, 0.0, -0.3, 0.3, 0.0, 0.0, 0.0, -1.0, 0.0, 0.6, -0.3, 0.0, 0.0, 0.0,
            ],
            dtype=float,
        ),
        "w6": np.array(
            [
                1.0, -0.3, 1.0, 0.3, 0.3, -0.6, -0.3, 0.3, -0.6, -0.3, -0.3, 0.0, 0.6, -0.3, -0.3,
                -0.3, 1.0, 0.3, 0.6, -0.3, 0.0, 0.6, 0.3, -0.3, -0.3, -1.0, -0.3, 1.0, -0.6, 0.3, 0.0, 0.3,
            ],
            dtype=float,
        ),
    }


def build_v3_static(
    n_u: int,
    n_j: int,
    include_v4: bool = True,
    w6_override: np.ndarray | None = None,
) -> np.ndarray:
    """Build static V0 matrix with sector and macro sensitivity priors.

    Sensitivity labels use a 7-level discrete grid: {-1.0, -0.6, -0.3, 0.0, +0.3, +0.6, +1.0}.

    Args:
        n_u: Number of US assets (expected 15 for built-in labels)
        n_j: Number of JP assets (expected 17 for built-in labels)
        include_v4: Whether to include v4-v6 priors
        w6_override: Optional override for v6 (inflation sensitivity labels)

    Returns:
        V0 matrix (N x K0)
    """
    base_vectors = build_base_vectors(n_u, n_j)
    v1, v2 = base_vectors["v1"], base_vectors["v2"]

    labels = get_static_sensitivity_labels()
    w3 = labels["w3"]
    if w3.shape[0] != v1.shape[0]:
        raise ValueError(f"w3 length must be {v1.shape[0]}, got {w3.shape[0]}")
    v3 = _orthogonalize_and_normalize(w3, [v1, v2])

    if not include_v4:
        return np.column_stack([v1, v2, v3])

    w4 = labels["w4"]
    w5 = labels["w5"]
    w6 = labels["w6"]

    if w6_override is not None:
        w6_arr = np.asarray(w6_override, dtype=float).reshape(-1)
        if w6_arr.shape != w6.shape:
            raise ValueError(f"w6_override must have shape {w6.shape}, got {w6_arr.shape}")
        w6 = w6_arr

    v4 = _orthogonalize_and_normalize(w4, [v1, v2, v3])
    v5 = _orthogonalize_and_normalize(w5, [v1, v2, v3, v4])
    v6 = _orthogonalize_and_normalize(w6, [v1, v2, v3, v4, v5])

    return np.column_stack([v1, v2, v3, v4, v5, v6])


def build_v3_dynamic(betas: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Build v3 from continuous beta values, orthogonalized against v1, v2."""
    w3 = betas.copy()
    w3 = w3 - (w3 @ v1) * v1
    w3 = w3 - (w3 @ v2) * v2
    norm = np.linalg.norm(w3)
    if norm < 1e-10:
        return np.zeros_like(w3)
    return w3 / norm


def compute_correlation(
    window_returns: np.ndarray,
    ewma_half_life: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute rolling mean, std, and correlation (equal-weight or EWMA).

    Returns:
        mu, sigma, correlation_matrix
    """
    with np.errstate(invalid="ignore"):
        if ewma_half_life is None:
            mu = np.mean(window_returns, axis=0)
            sigma = np.std(window_returns, axis=0, ddof=0)
            sigma[sigma == 0] = 1e-8
            z_window = (window_returns - mu) / sigma
            corr = np.dot(z_window.T, z_window) / window_returns.shape[0]
            np.fill_diagonal(corr, 1.0)
            return mu, sigma, corr

        if ewma_half_life <= 0:
            raise ValueError("ewma_half_life must be positive when provided")

        t = window_returns.shape[0]
        decay = np.power(0.5, 1.0 / float(ewma_half_life))
        weights = np.power(decay, np.arange(t - 1, -1, -1))
        weights = weights / np.sum(weights)

        mu = np.sum(window_returns * weights[:, None], axis=0)
        var = np.sum(((window_returns - mu) ** 2) * weights[:, None], axis=0)
        sigma = np.sqrt(np.maximum(var, 1e-16))
        sigma[sigma == 0] = 1e-8

        z_window = (window_returns - mu) / sigma
        corr = np.dot((z_window * weights[:, None]).T, z_window)
        np.fill_diagonal(corr, 1.0)
        return mu, sigma, corr


_BASELINE_CORR_CACHE: dict = {}


def compute_baseline_correlation(
    all_returns: np.ndarray,
    date_index: np.ndarray,
    ewma_half_life: float | None = None,
    baseline_start: str = "2010-01-01",
    baseline_end: str = "2014-12-31",
) -> np.ndarray:
    """Compute baseline correlation matrix from a specified period.

    Parameters
    ----------
    all_returns : (T, N) array of returns
    date_index : (T,) array of datetime64 dates
    ewma_half_life : optional EWMA half-life for weighting
    baseline_start : start date for baseline period (default 2010-01-01)
    baseline_end : end date for baseline period (default 2014-12-31)
    """
    mask = (date_index >= np.datetime64(baseline_start)) & (date_index <= np.datetime64(baseline_end))
    base_returns = all_returns[mask]
    if base_returns.shape[0] == 0:
        raise ValueError(f"No rows found for baseline period ({baseline_start} to {baseline_end})")

    cache_key = (ewma_half_life, baseline_start, baseline_end, base_returns.shape, hash(base_returns.tobytes()))
    if cache_key in _BASELINE_CORR_CACHE:
        return _BASELINE_CORR_CACHE[cache_key].copy()

    _, _, corr = compute_correlation(base_returns, ewma_half_life)
    _BASELINE_CORR_CACHE[cache_key] = corr
    return corr.copy()


def build_c0_from_v0(v0: np.ndarray, c_full: np.ndarray) -> np.ndarray:
    """Construct target correlation matrix C0 from V0 and C_full."""
    mat = v0.T @ c_full @ v0
    d_vals = np.diag(mat)
    d0 = np.diag(d_vals)
    c0_raw = v0 @ d0 @ v0.T
    delta = np.diag(c0_raw)
    delta = np.maximum(delta, 1e-10)
    delta_inv_sqrt = np.diag(1.0 / np.sqrt(delta))
    c0 = delta_inv_sqrt @ c0_raw @ delta_inv_sqrt
    np.fill_diagonal(c0, 1.0)
    return c0


def build_lw_target_correlation(corr: np.ndarray, target: str = "equicorrelation") -> np.ndarray:
    """Build LW-style target correlation matrix."""
    n = corr.shape[0]
    if target == "identity":
        return np.eye(n)

    upper = np.triu_indices(n, k=1)
    rho_bar = np.mean(corr[upper])
    rho_min = -1.0 / (n - 1)
    rho_bar = np.clip(rho_bar, rho_min, 0.9999)

    result = np.full((n, n), rho_bar)
    np.fill_diagonal(result, 1.0)
    return result


def effective_raw_weight(lambda_lw: float, lambda_reg: float) -> float:
    """Compute the effective weight of c_t in the final regularized matrix.

    With two-stage shrinkage:
        c_reg = (1-lambda_lw)*(1-lambda_reg)*c_t + lambda_lw*(1-lambda_reg)*C_LW + lambda_reg*c_0
    so the raw sample weight is (1-lambda_lw)*(1-lambda_reg).
    """
    return (1.0 - lambda_lw) * (1.0 - lambda_reg)


def regularize_correlation(
    c_t: np.ndarray,
    c_0_t: np.ndarray,
    lambda_reg: float,
    lambda_lw: float,
    lw_target: str,
    min_raw_weight: float = 0.0,
) -> np.ndarray:
    """Two-stage correlation regularization with attenuation guardrail.

    Stage 1 (LW):  c_lw = (1-lambda_lw)*c_t + lambda_lw*C_LW
    Stage 2 (reg): c_reg = (1-lambda_reg)*c_lw + lambda_reg*c_0

    The effective raw weight is (1-lambda_lw)*(1-lambda_reg).  When this
    falls below *min_raw_weight*, lambda_reg is automatically rescaled so
    that at least *min_raw_weight* of the sample correlation survives.
    This prevents the two-stage shrinkage from collapsing the final
    matrix onto the prior.
    """
    raw_w = effective_raw_weight(lambda_lw, lambda_reg)
    if raw_w < min_raw_weight and (1.0 - lambda_lw) > 1e-10:
        lambda_reg = 1.0 - min_raw_weight / (1.0 - lambda_lw)
        lambda_reg = float(np.clip(lambda_reg, 0.0, 1.0))

    lw_target_mat = build_lw_target_correlation(c_t, lw_target)
    c_lw = (1 - lambda_lw) * c_t + lambda_lw * lw_target_mat
    c_reg = (1 - lambda_reg) * c_lw + lambda_reg * c_0_t
    c_reg = 0.5 * (c_reg + c_reg.T)
    np.fill_diagonal(c_reg, 1.0)
    return c_reg
