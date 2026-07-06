"""Core mathematical and prior subspace calculations for the lead-lag strategy."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import kendalltau, t as student_t

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
    use_copula: bool = False,
    copula_blend_weight: float = 0.0,
    copula_nu_init: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute rolling mean, std, and correlation (equal-weight or EWMA).

    When *use_copula* is True and *copula_blend_weight* > 0, the Pearson
    correlation matrix is blended with a t-copula correlation matrix to
    capture tail dependence.

    Args:
        window_returns: (T, N) array of returns.
        ewma_half_life: EWMA half-life for weighting. None = equal weight.
        use_copula: If True, blend Pearson with t-copula correlation.
        copula_blend_weight: Blend weight in [0, 1]. 0 = Pearson only,
            1 = copula only. Ignored when use_copula=False.
        copula_nu_init: Initial degrees-of-freedom for t-copula estimation.

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
        else:
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

        if use_copula and copula_blend_weight > 0.0:
            corr_copula, _nu = estimate_t_copula(
                window_returns, nu_init=copula_nu_init
            )
            corr = blend_correlation(corr, corr_copula, copula_blend_weight)

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


# ---------------------------------------------------------------------------
# Copula-based correlation estimation
# ---------------------------------------------------------------------------


def empirical_cdf_transform(returns: np.ndarray) -> np.ndarray:
    """Transform each column to uniform [0, 1] via empirical CDF (pseudodata).

    Uses the plotting-position formula u = rank / (T + 1) to avoid
    exact 0 or 1 values that would map to ±inf under inverse CDF.

    Args:
        returns: (T, N) array of returns.

    Returns:
        (T, N) array of uniform [0, 1] values.
    """
    T, N = returns.shape
    u = np.zeros_like(returns, dtype=float)
    for k in range(N):
        col = returns[:, k]
        finite_mask = np.isfinite(col)
        if not np.any(finite_mask):
            u[:, k] = 0.5
            continue
        ranks = np.empty(T)
        ranks[finite_mask] = _rank_average(col[finite_mask])
        ranks[~finite_mask] = np.nan
        u[:, k] = ranks / (T + 1.0)
    return u


def _rank_average(values: np.ndarray) -> np.ndarray:
    """Compute average ranks (1-based) for a 1-D array."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_vals = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based average rank
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    return ranks


def _make_psd_correlation(R: np.ndarray) -> np.ndarray:
    """Project a symmetric matrix to the nearest PSD correlation matrix."""
    R = 0.5 * (R + R.T)
    np.fill_diagonal(R, 1.0)
    eigvals = np.linalg.eigvalsh(R)
    min_eig = eigvals.min()
    if min_eig < 0:
        R = R + (abs(min_eig) + 1e-6) * np.eye(R.shape[0])
        d = np.sqrt(np.diag(R))
        R = R / np.outer(d, d)
        np.fill_diagonal(R, 1.0)
    return R


def estimate_t_copula(
    returns: np.ndarray,
    nu_init: float = 5.0,
    max_outer_iter: int = 5,
) -> tuple[np.ndarray, float]:
    """Estimate a multivariate t-copula correlation matrix and degrees of freedom.

    Uses a two-step iterative procedure:
      1. Kendall's tau → initial correlation matrix R
      2. Alternate: optimize nu (R fixed) then update R (nu fixed)

    Args:
        returns: (T, N) array of returns.
        nu_init: Initial guess for degrees of freedom.
        max_outer_iter: Number of outer alternation iterations.

    Returns:
        Tuple of (R_copula, nu): correlation matrix and degrees of freedom.
    """
    T, N = returns.shape
    if T < 10 or N < 2:
        return np.eye(N), float(nu_init)

    u = empirical_cdf_transform(returns)
    u = np.clip(u, 1e-6, 1.0 - 1e-6)

    # Step 1: Kendall's tau → R_init
    R = np.eye(N)
    for i in range(N):
        for j in range(i + 1, N):
            mask = np.isfinite(u[:, i]) & np.isfinite(u[:, j])
            if np.sum(mask) > 3:
                tau, _ = kendalltau(u[mask, i], u[mask, j])
                if np.isfinite(tau):
                    R[i, j] = R[j, i] = np.sin(np.pi * tau / 2.0)
    R = _make_psd_correlation(R)

    nu = float(nu_init)

    for _ in range(max_outer_iter):
        # Step 2: optimize nu (R fixed)
        z = student_t.ppf(u, df=nu)
        z = np.clip(z, -10.0, 10.0)
        z = np.nan_to_num(z, nan=0.0, posinf=10.0, neginf=-10.0)

        def neg_loglik_nu(nu_val: float) -> float:
            return _t_copula_neg_loglik(z, R, nu_val)

        res = minimize_scalar(
            neg_loglik_nu, bounds=(2.5, 30.0), method="bounded"
        )
        if res.success:
            nu = float(res.x)

        # Step 3: update R (nu fixed)
        z = student_t.ppf(u, df=nu)
        z = np.clip(z, -10.0, 10.0)
        z = np.nan_to_num(z, nan=0.0, posinf=10.0, neginf=-10.0)
        try:
            R_new = np.corrcoef(z.T)
        except Exception:
            R_new = R.copy()
        R_new = np.nan_to_num(R_new, nan=0.0, posinf=1.0, neginf=-1.0)
        if not np.all(np.isfinite(R_new)):
            R_new = R.copy()
        np.fill_diagonal(R_new, 1.0)
        R = _make_psd_correlation(R_new)

    return R, nu


def _t_copula_neg_loglik(
    z: np.ndarray, R: np.ndarray, nu: float
) -> float:
    """Negative log-likelihood of a t-copula given t-quantile transformed data.

    log c(u; R, ν) = gammaln((ν+n)/2) + (n-1)*gammaln(ν/2) - n*gammaln((ν+1)/2)
                      - 0.5*log|R| - ((ν+n)/2)*log(1 + z'R^{-1}z/ν)
                      + ((ν+1)/2)*Σ log(1 + z_k²/ν)

    Args:
        z: (T, N) array of t-quantile transformed values.
        R: (N, N) copula correlation matrix.
        nu: Degrees of freedom.

    Returns:
        Negative log-likelihood (scalar).
    """
    T, N = z.shape
    L = None
    jitter = 1e-6
    for _attempt in range(5):
        try:
            L = np.linalg.cholesky(R + jitter * np.eye(N))
            break
        except np.linalg.LinAlgError:
            jitter *= 10.0
    if L is None:
        return 1e15

    log_det_R = 2.0 * np.sum(np.log(np.diag(L)))
    try:
        R_inv = np.linalg.solve(R + jitter * np.eye(N), np.eye(N))
    except np.linalg.LinAlgError:
        R_inv = np.linalg.pinv(R)

    half_nu = nu / 2.0
    half_nu_plus_N = (nu + N) / 2.0
    half_nu_plus_1 = (nu + 1) / 2.0

    from scipy.special import gammaln

    const = (
        gammaln(half_nu_plus_N)
        + (N - 1) * gammaln(half_nu)
        - N * gammaln(half_nu_plus_1)
        - 0.5 * log_det_R
    )

    # Vectorized: compute all T rows at once
    # quad[t] = z[t] @ R_inv @ z[t]  →  np.einsum over rows
    quad_all = np.einsum("ti,ij,tj->t", z, R_inv, z)  # (T,)
    denom_all = np.sum(np.log1p(z ** 2 / nu), axis=1)  # (T,)

    log_c_all = (
        const
        - half_nu_plus_N * np.log1p(quad_all / nu)
        + half_nu_plus_1 * denom_all
    )

    total = float(np.sum(log_c_all))

    if not np.isfinite(total):
        return 1e15
    return -total


def blend_correlation(
    corr_pearson: np.ndarray,
    corr_copula: np.ndarray,
    weight: float,
) -> np.ndarray:
    """Blend Pearson and copula correlation matrices.

    Args:
        corr_pearson: Pearson correlation matrix (N, N).
        corr_copula: Copula correlation matrix (N, N).
        weight: Blend weight in [0, 1]. 0 = Pearson only, 1 = copula only.

    Returns:
        Blended correlation matrix (N, N).
    """
    weight = float(np.clip(weight, 0.0, 1.0))
    corr = (1.0 - weight) * corr_pearson + weight * corr_copula
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, 1.0)
    return corr


def compute_stress_weight(
    window_returns: np.ndarray,
    method: str = "var_ratio",
    recent_window: int = 20,
    threshold: float = 1.5,
    sigmoid_slope: float = 8.0,
) -> float:
    """Compute a stress-regime weight in [0, 1] for dynamic copula blending.

    When the recent volatility is significantly higher than the baseline,
    the weight approaches 1 (favoring copula correlation that captures
    tail dependence). In calm periods the weight approaches 0 (Pearson only).

    Args:
        window_returns: (T, N) array of returns.
        method: Method for stress detection ("var_ratio").
        recent_window: Number of recent days for vol estimation.
        threshold: Vol ratio above which stress is signaled.
        sigmoid_slope: Steepness of the sigmoid transition.

    Returns:
        Stress weight in [0, 1].
    """
    T = window_returns.shape[0]
    if T < recent_window + 10:
        return 0.0

    if method == "var_ratio":
        recent_vol = np.std(window_returns[-recent_window:], axis=0)
        baseline_vol = np.std(window_returns, axis=0)
        ratio = np.median(recent_vol / np.maximum(baseline_vol, 1e-8))
        if not np.isfinite(ratio):
            return 0.0
        w = 1.0 / (1.0 + np.exp(-(ratio - threshold) * sigmoid_slope))
        return float(np.clip(w, 0.0, 1.0))

    return 0.0
