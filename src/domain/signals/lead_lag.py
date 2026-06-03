"""Lead-Lag signal generation.

Pure numerical computation of the lead-lag strategy signals.
No I/O, no pandas DataFrame mutation – just numpy arrays.
"""

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


def build_v3_static(
    n_u: int,
    n_j: int,
    include_v4: bool = True,
    w6_override: np.ndarray | None = None,
) -> np.ndarray:
    """Build static V0 matrix with sector and macro sensitivity priors.

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

    # v3 (w3c): continuous cyclical/defensive sensitivity labels
    w3 = np.array(
        [
            1.0,
            0.3,
            0.2,
            0.8,
            0.9,
            0.7,
            -1.0,
            0.4,
            -0.9,
            -0.8,
            1.0,
            0.0,
            0.6,
            -0.2,
            -0.7,
            -0.9,
            0.3,
            0.6,
            0.9,
            -0.9,
            1.0,
            1.0,
            0.9,
            0.8,
            -0.3,
            -1.0,
            -0.4,
            0.7,
            -0.5,
            0.8,
            0.6,
            0.5,
        ],
        dtype=float,
    )
    if w3.shape[0] != v1.shape[0]:
        raise ValueError(f"w3 length must be {v1.shape[0]}, got {w3.shape[0]}")
    v3 = _orthogonalize_and_normalize(w3, [v1, v2])

    if not include_v4:
        return np.column_stack([v1, v2, v3])

    # v4: FX sensitivity labels
    w4 = np.array(
        [
            0.4,
            0.0,
            0.1,
            0.2,
            0.7,
            0.8,
            -0.5,
            -0.4,
            -0.7,
            -0.4,
            0.6,
            0.3,
            0.1,
            0.6,
            -0.3,
            -0.6,
            0.2,
            0.2,
            0.5,
            -0.2,
            1.0,
            0.6,
            0.8,
            1.0,
            -0.2,
            -0.8,
            -0.4,
            0.8,
            -0.7,
            0.3,
            0.0,
            -0.9,
        ],
        dtype=float,
    )

    # v5: energy-price sensitivity labels
    w5 = np.array(
        [
            0.4,
            0.0,
            1.0,
            0.0,
            0.2,
            0.0,
            -0.3,
            0.0,
            -0.8,
            0.0,
            -0.3,
            0.0,
            0.4,
            -0.1,
            -0.3,
            -0.3,
            1.0,
            -0.1,
            0.3,
            0.0,
            -0.2,
            0.2,
            0.0,
            0.0,
            0.0,
            -0.9,
            -0.1,
            0.7,
            -0.2,
            0.0,
            0.0,
            0.0,
        ],
        dtype=float,
    )

    # v6: inflation sensitivity labels
    w6 = np.array(
        [
            0.8,
            -0.3,
            1.0,
            0.3,
            0.3,
            -0.5,
            -0.2,
            0.4,
            -0.7,
            -0.2,
            -0.4,
            -0.1,
            0.5,
            -0.4,
            -0.3,
            -0.4,
            1.0,
            0.3,
            0.7,
            -0.2,
            -0.1,
            0.6,
            0.2,
            -0.3,
            -0.3,
            -0.8,
            -0.3,
            0.8,
            -0.5,
            0.2,
            0.1,
            0.3,
        ],
        dtype=float,
    )

    if w6_override is not None:
        w6_arr = np.asarray(w6_override, dtype=float).reshape(-1)
        if w6_arr.shape != w6.shape:
            raise ValueError(
                f"w6_override must have shape {w6.shape}, got {w6_arr.shape}"
            )
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


def compute_baseline_correlation(
    all_returns: np.ndarray,
    date_index: np.ndarray,
    ewma_half_life: float | None = None,
) -> np.ndarray:
    """Compute baseline correlation matrix from 2010-2014 data."""
    # date_index should be datetime-like
    mask = (date_index >= np.datetime64("2010-01-01")) & (
        date_index <= np.datetime64("2014-12-31")
    )
    base_returns = all_returns[mask]
    if base_returns.shape[0] == 0:
        raise ValueError("No rows found for baseline period (2010-2014)")
    _, _, corr = compute_correlation(base_returns, ewma_half_life)
    return corr


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


def build_lw_target_correlation(
    corr: np.ndarray, target: str = "equicorrelation"
) -> np.ndarray:
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


def regularize_correlation(
    c_t: np.ndarray,
    c_0_t: np.ndarray,
    lambda_reg: float,
    lambda_lw: float,
    lw_target: str,
) -> np.ndarray:
    """Two-stage correlation regularization."""
    lw_target_mat = build_lw_target_correlation(c_t, lw_target)
    c_lw = (1 - lambda_lw) * c_t + lambda_lw * lw_target_mat
    c_reg = (1 - lambda_reg) * c_lw + lambda_reg * c_0_t
    c_reg = 0.5 * (c_reg + c_reg.T)
    np.fill_diagonal(c_reg, 1.0)
    return c_reg


def compute_signal(
    all_returns: np.ndarray,
    current_index: int,
    n_u: int,
    corr_window: int,
    c_full: np.ndarray,
    v0_static: np.ndarray | None,
    v1: np.ndarray,
    v2: np.ndarray,
    k: int,
    lambda_reg: float,
    lambda_lw: float,
    lw_target: str,
    ewma_half_life: float | None,
    v3_dynamic: bool = False,
    gap_override: np.ndarray | None = None,
    gap_open_coef: float = 0.70,
    topix_beta_coef: float = 1.20,
    betas_t: np.ndarray | None = None,
    topix_night_t: float | None = None,
    vol_adjusted_target: bool = False,
) -> dict[str, np.ndarray | float]:
    """Compute the lead-lag signal for a single time step.

    Args:
        all_returns: Full return matrix (T x N)
        current_index: Index in the return matrix for the current step
        n_u: Number of US assets
        corr_window: Rolling correlation window
        c_full: Baseline correlation matrix
        v0_static: Static prior subspace (None for dynamic)
        v1, v2: Base vectors
        k: Number of eigenvectors
        lambda_reg: Second-stage shrinkage
        lambda_lw: First-stage shrinkage
        lw_target: LW target type
        ewma_half_life: EWMA half-life
        v3_dynamic: If True, use dynamic beta-based v3
        gap_override: Override gap returns
        gap_open_coef: Gap coefficient
        topix_beta_coef: TOPIX beta coefficient
        betas_t: Rolling beta values for JP assets (N_J,)
        topix_night_t: TOPIX night return for trade date
        vol_adjusted_target: If True, use 20-day rolling vol-adjusted target Z-score.

    Returns:
        Dict with: signal (N_J,), sigma_s, r_hat_jp_cc (N_J,)
    """
    n_j = all_returns.shape[1] - n_u
    window_start = max(0, current_index - corr_window)
    window_returns = all_returns[window_start:current_index]

    mu_w, sigma_w, c_t = compute_correlation(window_returns, ewma_half_life)

    # Determine C_0 for this time step
    if v3_dynamic:
        mkt_ret = np.mean(window_returns, axis=1)
        mkt_var = np.var(mkt_ret, ddof=0)
        if mkt_var < 1e-16:
            mkt_var = 1e-16
        betas = np.array(
            [
                np.cov(window_returns[:, j], mkt_ret, ddof=0)[0, 1] / mkt_var
                for j in range(all_returns.shape[1])
            ]
        )
        betas = betas - np.mean(betas)
        v3_dyn = build_v3_dynamic(betas, v1, v2)
        v0_t = np.column_stack([v1, v2, v3_dyn])
        c0_t = build_c0_from_v0(v0_t, c_full)
    else:
        c0_t = build_c0_from_v0(v0_static, c_full)

    c_t_reg = regularize_correlation(c_t, c0_t, lambda_reg, lambda_lw, lw_target)

    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(c_t_reg)
    sort_idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, sort_idx]

    v_t_k = eigvecs[:, :k]
    v_u_t_k = v_t_k[:n_u, :]
    v_j_t_k = v_t_k[n_u:, :]

    r_us_t = all_returns[current_index, :n_u]
    mu_us = mu_w[:n_u]
    sigma_us = sigma_w[:n_u]
    z_u_t = (r_us_t - mu_us) / sigma_us

    f_t = v_u_t_k.T @ z_u_t
    z_hat_j_t1 = v_j_t_k @ f_t

    mu_jp = mu_w[n_u:]
    sigma_jp = sigma_w[n_u:]
    if vol_adjusted_target:
        if current_index >= 19:
            jp_returns_20 = all_returns[current_index - 19 : current_index + 1, n_u:]
            sigma_j_t = np.std(jp_returns_20, axis=0, ddof=1)
            sigma_j_t = np.maximum(sigma_j_t, 1e-8)
        else:
            sigma_j_t = sigma_jp
        r_hat_jp_cc = z_hat_j_t1 * sigma_j_t
    else:
        r_hat_jp_cc = mu_jp + sigma_jp * z_hat_j_t1

    # Apply gap residual adjustment
    if gap_override is not None:
        gap_vec = np.asarray(gap_override, dtype=float).reshape(-1)
        use_topix = False
        if betas_t is not None and topix_night_t is not None:
            betas_vec = np.asarray(betas_t, dtype=float).reshape(-1)
            if (
                betas_vec.shape == gap_vec.shape
                and np.all(np.isfinite(betas_vec))
                and np.isfinite(float(topix_night_t))
            ):
                use_topix = True

        if use_topix:
            gap_syst = betas_vec * float(topix_night_t)
            gap_idio = gap_vec - gap_syst
            gap_filt = (
                gap_open_coef * gap_idio
                + (gap_open_coef - topix_beta_coef) * gap_syst
            )
            denom = np.maximum(1.0 + gap_filt, 0.1)
            signal = (1.0 + r_hat_jp_cc) / denom - 1.0
        else:
            signal = r_hat_jp_cc - gap_open_coef * gap_vec
    else:
        signal = z_hat_j_t1

    sigma_s = float(np.std(z_hat_j_t1))

    return {
        "signal": signal,
        "sigma_s": sigma_s,
        "r_hat_jp_cc": r_hat_jp_cc,
        # Exposed for nonlinear correction layer:
        # f_t: K-dim factor score (V_U^K.T @ z_U), always shape (K,)
        # z_hat_j_t1: raw linear standardized prediction before gap adj, shape (N_J,)
        "f_t": f_t,
        "z_hat_j_t1": z_hat_j_t1,
    }


def select_long_short_indices(
    signal: np.ndarray,
    q: float,
    n_j: int,
    enforce_sign: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Select long and short position indices from signal.

    Returns:
        long_indices, short_indices
    """
    num_positions = int(np.floor(n_j * q))
    if num_positions <= 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    if enforce_sign:
        long_idx = np.where(signal > EPSILON_WEIGHT)[0]
        short_idx = np.where(signal < -EPSILON_WEIGHT)[0]

        if len(long_idx) == 0 or len(short_idx) == 0:
            return np.array([], dtype=int), np.array([], dtype=int)

        long_idx = long_idx[np.argsort(signal[long_idx])][-num_positions:]
        short_idx = short_idx[np.argsort(signal[short_idx])][:num_positions]
    else:
        sort_order = np.argsort(signal)
        short_idx = sort_order[:num_positions]
        long_idx = sort_order[-num_positions:]

    return long_idx, short_idx


def build_weights(
    signal: np.ndarray,
    q: float,
    n_j: int,
    weight_mode: str = "signal",
    enforce_sign: bool = False,
) -> np.ndarray:
    """Build portfolio weights from signal.

    Returns:
        Weight array of shape (n_j,)
    """
    weights = np.zeros(n_j)
    long_idx, short_idx = select_long_short_indices(signal, q, n_j, enforce_sign)

    if len(long_idx) == 0 or len(short_idx) == 0:
        return weights

    if weight_mode == "signal":
        s_centered = signal - np.median(signal)

        long_raw = s_centered[long_idx]
        long_raw = np.maximum(long_raw, EPSILON_SIGMA)
        long_denom = np.sum(long_raw)
        if long_denom > 0:
            weights[long_idx] = long_raw / long_denom

        short_raw = -s_centered[short_idx]
        short_raw = np.maximum(short_raw, EPSILON_SIGMA)
        short_denom = np.sum(short_raw)
        if short_denom > 0:
            weights[short_idx] = -(short_raw / short_denom)
    else:
        if len(long_idx) > 0:
            weights[long_idx] = 1.0 / len(long_idx)
        if len(short_idx) > 0:
            weights[short_idx] = -1.0 / len(short_idx)

    return weights


def compute_dispersion_indicator(
    signal: np.ndarray,
    q: float,
    n_j: int,
    dispersion_metric: str = "long_short_mean_gap",
    enforce_sign: bool = False,
) -> float:
    """Compute the dispersion indicator value."""
    if dispersion_metric == "sigma":
        return float(np.std(signal))

    long_idx, short_idx = select_long_short_indices(signal, q, n_j, enforce_sign)
    if len(long_idx) == 0 or len(short_idx) == 0:
        return 0.0

    s_tilde = signal - np.median(signal)
    long_mean = np.mean(s_tilde[long_idx])
    short_mean = np.mean(s_tilde[short_idx])
    return float(long_mean - short_mean)


def dispersion_scale(
    indicator: float,
    indicator_history: list[float],
    dispersion_filter: bool = True,
    history_window: int = 60,
) -> float:
    """Compute the dispersion scale factor."""
    scale = 1.0
    if dispersion_filter and len(indicator_history) >= history_window:
        hist = np.array(indicator_history[-history_window:])
        p10 = float(np.percentile(hist, 10))
        p25 = float(np.percentile(hist, 25))
        if indicator < p10:
            scale = 0.0
        elif indicator < p25:
            scale = 0.5
    return scale


def apply_gap_tolerant_filter(
    signal: np.ndarray,
    sigma_s: float,
    weights: np.ndarray,
    jp_close_t: np.ndarray,
    jp_open_t1: np.ndarray,
    gamma: float,
    q: float,
    n_j: int,
    weight_mode: str = "signal",
    enforce_sign: bool = False,
) -> tuple[np.ndarray, int, int, np.ndarray]:
    """Apply gap tolerance filter for limit order execution.

    Returns:
        filtered_weights, long_executed_count, short_executed_count, executed_mask
    """
    has_longs = np.any(weights > 1e-12)
    has_shorts = np.any(weights < -1e-12)

    if not has_longs or not has_shorts:
        return np.zeros(n_j), 0, 0, np.array([], dtype=bool)

    if sigma_s < 1e-12:
        sigma_s_used = 1e-12
    else:
        sigma_s_used = sigma_s

    executed = np.zeros(n_j, dtype=bool)
    long_indices = np.where(weights > 1e-12)[0]
    short_indices = np.where(weights < -1e-12)[0]

    for idx in long_indices:
        limit_price = jp_close_t[idx] * (1.0 + gamma * signal[idx] * sigma_s_used)
        open_price = jp_open_t1[idx]
        if np.isfinite(open_price) and open_price <= limit_price:
            executed[idx] = True

    for idx in short_indices:
        limit_price = jp_close_t[idx] * (
            1.0 - gamma * np.abs(signal[idx]) * sigma_s_used
        )
        open_price = jp_open_t1[idx]
        if np.isfinite(open_price) and open_price >= limit_price:
            executed[idx] = True

    long_executed = np.sum(executed[long_indices])
    short_executed = np.sum(executed[short_indices])

    if long_executed < 2 or short_executed < 2:
        return np.zeros(n_j), 0, 0, executed

    # Renormalize
    final_weights = np.zeros(n_j)
    executed_long = long_indices[executed[long_indices]]
    executed_short = short_indices[executed[short_indices]]

    if weight_mode == "signal":
        s_centered = signal - np.median(signal)
        if len(executed_long) > 0:
            long_raw = s_centered[executed_long]
            long_raw = np.maximum(long_raw, EPSILON_SIGMA)
            long_denom = np.sum(long_raw)
            if long_denom > 0:
                final_weights[executed_long] = long_raw / long_denom
        if len(executed_short) > 0:
            short_raw = -s_centered[executed_short]
            short_raw = np.maximum(short_raw, EPSILON_SIGMA)
            short_denom = np.sum(short_raw)
            if short_denom > 0:
                final_weights[executed_short] = -(short_raw / short_denom)
    else:
        if len(executed_long) > 0:
            final_weights[executed_long] = 1.0 / len(executed_long)
        if len(executed_short) > 0:
            final_weights[executed_short] = -1.0 / len(executed_short)

    return final_weights, int(long_executed), int(short_executed), executed
