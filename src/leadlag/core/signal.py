"""Signal generation and portfolio allocation logic for the lead-lag strategy."""

from __future__ import annotations

import numpy as np

from leadlag.core.correlation import (
    EPSILON_SIGMA,
    EPSILON_WEIGHT,
    build_c0_from_v0,
    build_v3_dynamic,
    compute_correlation,
    regularize_correlation,
)


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
    topix_beta_coef: float = 0.6,
    betas_t: np.ndarray | None = None,
    topix_night_t: float | None = None,
    vol_adjusted_target: bool = False,
    min_raw_weight: float = 0.0,
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
    all_returns.shape[1] - n_u
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

    c_t_reg = regularize_correlation(c_t, c0_t, lambda_reg, lambda_lw, lw_target, min_raw_weight)

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
        if current_index >= 20:
            jp_returns_20 = all_returns[current_index - 20 : current_index, n_u:]
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
            gap_filt = gap_open_coef * gap_idio + (gap_open_coef - topix_beta_coef) * gap_syst
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
        limit_price = jp_close_t[idx] * (1.0 - gamma * np.abs(signal[idx]) * sigma_s_used)
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
