"""Experiment: M_sector refinement approaches vs baseline.

Compares 4 theoretically motivated alternatives to the fixed M_sector mapping:
  1. C0 cross-block: Derive M_sector from V0's implied cross-correlation
  2. V0 extension: Add sector-pair vectors to V0
  3. Multi-target Tikhonov: Integrate priors into ridge regression
  4. RRR with structured mean: B = M_sector + low-rank correction

All approaches are tested in isolation against the production baseline.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from leadlag.core.correlation import (
    build_base_vectors,
    build_c0_from_v0,
    build_v3_static,
    compute_baseline_correlation,
    compute_correlation,
    regularize_correlation,
    _orthogonalize_and_normalize,
)
from leadlag.core import signal as signals
from leadlag.core.residualize import compute_rolling_ols_betas
from leadlag.data.fetcher import download_data
from leadlag.data.preprocessor import preprocess_data
from leadlag.data.tickers import JP_TICKERS, US_TICKERS
from leadlag.models.sre import compute_jp_target_returns
from leadlag.models.sector_relative_ensemble_blp_enhanced import (
    SectorRelativeEnsembleBLPEnhancedModel,
)
from leadlag.execution.backtester import BacktestEngine
from leadlag.reporting.metrics import calculate_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Production parameters
PROD_PARAMS = {
    "blp_window": 252,
    "blp_ewma_halflife": 45,
    "alpha_xx": 0.50,
    "alpha_yx": 0.05,
    "alpha_yy": 0.50,
    "rho": 0.01,
    "rank": "full",
    "lambda_pca": 0.10,
    "lambda_sector": 0.40,
    "beta_conf": 0.25,
    "winsor_sigma": 3.0,
    "lambda_reg": 0.75,
    "lambda_lw": 0.5,
    "lw_target": "equicorrelation",
    "k": 6,
    "corr_window": 60,
    "ewma_half_life": 45,
    "include_v4_prior": True,
    "gap_open_coef": 0.70,
    "topix_beta_coef": 0.6,
    "beta_window": 60,
    "vol_adjusted_target": True,
    "normalization": "zscore",
    "q": 0.3,
    "weight_mode": "signal",
}

N_U = len(US_TICKERS)  # 15
N_J = len(JP_TICKERS)  # 17


def build_model_config(overrides: dict | None = None) -> dict:
    """Build model config dict for SectorRelativeEnsembleBLPEnhancedModel."""
    cfg = {
        "model": {"name": "sector_relative_ensemble_blp_enhanced"},
        "portfolio": {"long_short_frac": 0.3, "weight_mode": "signal"},
        "ensemble": {
            "raw_pca_weight": 0.0,
            "residual_pca_weight": 0.0,
            "raw_blpx_weight": 0.5,
            "residual_blpx_weight": 0.5,
        },
        "costs": {"slippage_bps_per_side": 5.0},
    }
    cfg.update(PROD_PARAMS.copy())
    if overrides:
        cfg.update(overrides)
    return cfg


def run_backtest(model: SectorRelativeEnsembleBLPEnhancedModel, df_exec: pd.DataFrame) -> dict:
    """Run backtest and return metrics."""
    results = BacktestEngine.run_backtest(
        model,
        df_exec=df_exec,
        start_date="2015-01-05",
        overnight_alpha_long=0.0,
        overnight_alpha_short=0.0,
        buy_interest_annual=0.025,
        borrow_fee_annual=0.0115,
        reverse_fee_bps=2.0,
    )
    daily_returns = results["daily_returns"]
    metrics = calculate_metrics(daily_returns)

    # Add turnover and cost stats
    metrics["avg_turnover"] = float(results["daily_turnover"].mean())
    metrics["avg_cost"] = float(results["daily_costs"].mean())
    metrics["avg_gross_exp"] = float(results["daily_gross_exps"].mean())

    return metrics


# ---------------------------------------------------------------------------
# Approach 1: C0 cross-block derived M_sector
# ---------------------------------------------------------------------------


class C0CrossBlockModel(SectorRelativeEnsembleBLPEnhancedModel):
    """Approach 1: Replace hand-crafted M_sector with C0's cross-block.

    The prior correlation matrix C0 = V0 @ diag(V0^T C_full V0) @ V0^T
    contains an implicit US-JP cross-correlation block C0_YU (17x15).
    We use this as M_sector instead of the hardcoded mapping.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._c0_cross_cache: np.ndarray | None = None

    def _build_sector_prior(self) -> np.ndarray:
        """Override: return identity, actual M_sector computed dynamically."""
        return np.eye(self.n_j, self.n_u)

    def _get_sector_prior(
        self,
        current_index: int,
        all_returns: np.ndarray,
        corr: np.ndarray,
        B_blp: np.ndarray,
    ) -> np.ndarray:
        """Derive M_sector from C0's cross-block."""
        # Build V0
        v0 = build_v3_static(self.n_u, self.n_j, self.include_v4_prior)
        # Build C_full from long-run baseline (use all available returns up to current_index)
        window_start = max(0, current_index - self.blp_window)
        window_returns = all_returns[window_start:current_index]
        window_returns = np.nan_to_num(window_returns, nan=0.0, posinf=0.0, neginf=0.0)
        _, _, c_full_local = compute_correlation(window_returns, self.blp_ewma_halflife)
        c_full_local = np.nan_to_num(c_full_local, nan=0.0, posinf=1.0, neginf=-1.0)
        np.fill_diagonal(c_full_local, 1.0)

        # Build C0 from V0 and C_full
        c0 = build_c0_from_v0(v0, c_full_local)

        # Extract cross-block: C0_YU = c0[n_u:, :n_u]
        c0_yu = c0[self.n_u:, :self.n_u]

        # Column-normalize like the original M_sector
        col_sums = np.sum(np.abs(c0_yu), axis=0)
        m_sector = np.zeros_like(c0_yu)
        for u_idx in range(self.n_u):
            if col_sums[u_idx] > 1e-10:
                m_sector[:, u_idx] = c0_yu[:, u_idx] / col_sums[u_idx]

        if m_sector.shape == B_blp.shape:
            return m_sector
        return np.zeros(B_blp.shape)


# ---------------------------------------------------------------------------
# Approach 2: V0 extension with sector-pair vectors
# ---------------------------------------------------------------------------


def build_extended_v0(n_u: int, n_j: int, include_v4: bool = True) -> np.ndarray:
    """Build V0 with additional sector-pair vectors.

    For each US ticker, create a 32-dim vector that places weight on the US ticker
    and its mapped JP counterparts. Orthogonalize against v1..v6 and each other.
    """
    # Build base V0 (v1..v6)
    v0_base = build_v3_static(n_u, n_j, include_v4)
    basis = [v0_base[:, k] for k in range(v0_base.shape[1])]

    # Sector mapping (same as original M_sector)
    mapping = {
        "XLB": {"1620.T": 0.5, "1623.T": 0.5},
        "XLC": {"1626.T": 1.0},
        "XLE": {"1618.T": 0.5, "1627.T": 0.5},
        "XLF": {"1631.T": 0.5, "1632.T": 0.5},
        "XLI": {"1624.T": 0.33, "1622.T": 0.33, "1626.T": 0.34},
        "XLK": {"1626.T": 0.5, "1625.T": 0.5},
        "XLP": {"1617.T": 0.5, "1630.T": 0.5},
        "XLRE": {"1633.T": 1.0},
        "XLU": {"1627.T": 1.0},
        "XLV": {"1621.T": 1.0},
        "XLY": {"1630.T": 0.33, "1626.T": 0.33, "1622.T": 0.34},
        "MTUM": {"1625.T": 0.5, "1626.T": 0.5},
        "VLUE": {"1631.T": 0.25, "1632.T": 0.25, "1623.T": 0.25, "1622.T": 0.25},
        "IUSG": {"1626.T": 0.5, "1625.T": 0.5},
        "USMV": {"1617.T": 0.33, "1621.T": 0.33, "1627.T": 0.34},
    }

    pair_vectors = []
    for u_idx, us_tk in enumerate(US_TICKERS):
        if us_tk not in mapping:
            continue
        w = np.zeros(n_u + n_j)
        w[u_idx] = 1.0
        for jp_tk, weight in mapping[us_tk].items():
            if jp_tk in JP_TICKERS:
                j_idx = n_u + JP_TICKERS.index(jp_tk)
                w[j_idx] = weight
        # Orthogonalize against all existing basis vectors
        v = _orthogonalize_and_normalize(w, basis)
        if np.linalg.norm(v) > 1e-8:
            basis.append(v)
            pair_vectors.append(v)

    if len(pair_vectors) == 0:
        return v0_base

    return np.column_stack(basis)


class V0ExtensionModel(SectorRelativeEnsembleBLPEnhancedModel):
    """Approach 2: Extend V0 with sector-pair vectors.

    The sector mapping information is encoded as additional orthogonal vectors
    in V0, rather than as a separate M_sector matrix. This unifies all prior
    information into the subspace regularization framework.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        # Override k to match extended V0
        self._extended_v0 = build_extended_v0(self.n_u, self.n_j, self.include_v4_prior)
        self.k_extended = self._extended_v0.shape[1]
        logger.info(f"V0Extension: V0 extended from {build_v3_static(self.n_u, self.n_j, self.include_v4_prior).shape[1]} to {self.k_extended} dimensions")

    def _build_sector_prior(self) -> np.ndarray:
        """No separate M_sector; sector info is in extended V0."""
        return np.zeros((self.n_j, self.n_u))

    def _get_sector_prior(
        self,
        current_index: int,
        all_returns: np.ndarray,
        corr: np.ndarray,
        B_blp: np.ndarray,
    ) -> np.ndarray:
        """No separate sector prior; all info is in extended V0."""
        return np.zeros(B_blp.shape)

    def _prepare_common_inputs(self, df_exec: pd.DataFrame) -> dict:
        """Override to use extended V0."""
        inputs = super()._prepare_common_inputs(df_exec)
        # Replace v0_static with extended version
        inputs["v0_static"] = self._extended_v0
        return inputs


# ---------------------------------------------------------------------------
# Approach 3: Multi-target Tikhonov regularization
# ---------------------------------------------------------------------------


class TikhonovModel(SectorRelativeEnsembleBLPEnhancedModel):
    """Approach 3: Multi-target Tikhonov regularization.

    Instead of linearly mixing B_blp, B_pca, and B_sector after estimation,
    integrate the priors directly into the ridge regression:

        B = (Sigma_XX + rho*I + lambda_pca*I + lambda_sec*I)^{-1}
            * (Sigma_YX + lambda_pca*B_pca + lambda_sec*M_sector)

    This is the Bayesian posterior mean with Gaussian priors centered at
    B_pca and M_sector respectively.
    """

    def __init__(self, config: dict):
        super().__init__(config)

    def _build_sector_prior(self) -> np.ndarray:
        """Keep the original hand-crafted M_sector for Tikhonov prior mean."""
        M = np.zeros((self.n_j, self.n_u))
        mapping = {
            "XLB": {"1620.T": 0.5, "1623.T": 0.5},
            "XLC": {"1626.T": 1.0},
            "XLE": {"1618.T": 0.5, "1627.T": 0.5},
            "XLF": {"1631.T": 0.5, "1632.T": 0.5},
            "XLI": {"1624.T": 0.33, "1622.T": 0.33, "1626.T": 0.34},
            "XLK": {"1626.T": 0.5, "1625.T": 0.5},
            "XLP": {"1617.T": 0.5, "1630.T": 0.5},
            "XLRE": {"1633.T": 1.0},
            "XLU": {"1627.T": 1.0},
            "XLV": {"1621.T": 1.0},
            "XLY": {"1630.T": 0.33, "1626.T": 0.33, "1622.T": 0.34},
            "MTUM": {"1625.T": 0.5, "1626.T": 0.5},
            "VLUE": {"1631.T": 0.25, "1632.T": 0.25, "1623.T": 0.25, "1622.T": 0.25},
            "IUSG": {"1626.T": 0.5, "1625.T": 0.5},
            "USMV": {"1617.T": 0.33, "1621.T": 0.33, "1627.T": 0.34},
        }
        for u_idx, us_tk in enumerate(US_TICKERS):
            if us_tk in mapping:
                for jp_tk, w in mapping[us_tk].items():
                    if jp_tk in JP_TICKERS:
                        j_idx = JP_TICKERS.index(jp_tk)
                        M[j_idx, u_idx] = w
        col_sums = np.sum(M, axis=0)
        for u_idx in range(self.n_u):
            if col_sums[u_idx] > 0:
                M[:, u_idx] /= col_sums[u_idx]
        return M

    def compute_blp_signal(
        self,
        all_returns: np.ndarray,
        current_index: int,
        gap_override: np.ndarray | None = None,
        betas_t: np.ndarray | None = None,
        topix_night_t: float | None = None,
        rolling_std: np.ndarray | None = None,
        v0_static: np.ndarray | None = None,
        c_full: np.ndarray | None = None,
        is_residual: bool = False,
        return_matrices: bool = False,
    ) -> dict[str, Any]:
        """Override to use Tikhonov regularization instead of linear mixing."""

        window_start = max(0, current_index - self.blp_window)
        window_returns = all_returns[window_start:current_index].copy()

        if self.exec_adjustment == "vol_scale" and rolling_std is not None:
            for idx_local in range(len(window_returns)):
                idx_global = window_start + idx_local
                vol_factor = rolling_std[idx_global]
                window_returns[idx_local, self.n_u:] /= vol_factor

        window_returns = np.nan_to_num(window_returns, nan=0.0, posinf=0.0, neginf=0.0)

        if self.winsor_sigma is not None:
            for c in range(window_returns.shape[1]):
                mu_c = np.mean(window_returns[:, c])
                std_c = np.std(window_returns[:, c])
                if std_c > 1e-8:
                    window_returns[:, c] = np.clip(
                        window_returns[:, c],
                        mu_c - self.winsor_sigma * std_c,
                        mu_c + self.winsor_sigma * std_c,
                    )

        mu, sigma, corr = compute_correlation(window_returns, self.blp_ewma_halflife)
        mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
        sigma = np.nan_to_num(sigma, nan=1.0, posinf=1.0, neginf=1.0)
        corr = np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0)
        np.fill_diagonal(corr, 1.0)

        C_XX = corr[: self.n_u, : self.n_u]
        C_YX = corr[self.n_u :, : self.n_u]
        C_YY = corr[self.n_u :, self.n_u :]

        Sigma_XX_reg = (1.0 - self.alpha_xx) * C_XX + self.alpha_xx * np.eye(self.n_u)
        Sigma_YX_reg = (1.0 - self.alpha_yx) * C_YX
        Sigma_YY_reg = (1.0 - self.alpha_yy) * C_YY + self.alpha_yy * np.eye(self.n_j)

        diag_mean = float(np.mean(np.diag(Sigma_XX_reg)))

        # --- Compute B_pca prior ---
        B_pca = np.zeros((self.n_j, self.n_u))
        if v0_static is not None and c_full is not None and corr.shape == (32, 32):
            c0_t = build_c0_from_v0(v0_static, c_full)
            c_t_reg = regularize_correlation(
                corr, c0_t, self.lambda_reg, self.lambda_lw, self.lw_target
            )
            eigvals, eigvecs = np.linalg.eigh(c_t_reg)
            sort_idx = np.argsort(eigvals)[::-1]
            eigvecs = eigvecs[:, sort_idx]
            v_t_k = eigvecs[:, : self.k]
            v_u_t_k = v_t_k[: self.n_u, :]
            v_j_t_k = v_t_k[self.n_u :, :]
            B_pca = v_j_t_k @ v_u_t_k.T

        # --- Get M_sector ---
        M_sector = self._get_sector_prior(current_index, all_returns, corr, Sigma_YX_reg)

        # --- Multi-target Tikhonov ---
        # B = (Sigma_XX + (rho*diag + lambda_pca + lambda_sec)*I)^{-1}
        #     * (Sigma_YX + lambda_pca*B_pca + lambda_sec*M_sector)
        # Priors are used directly without rescaling — lambda controls pull strength.
        lambda_total = self.rho * diag_mean + self.lambda_pca + self.lambda_sector
        A_tikh = Sigma_XX_reg + lambda_total * np.eye(self.n_u)

        rhs = Sigma_YX_reg + self.lambda_pca * B_pca + self.lambda_sector * M_sector

        try:
            if not np.isfinite(A_tikh).all():
                raise ValueError("A_tikh contains NaNs or Infs")
            inv_A = np.linalg.inv(A_tikh)
            B_struct = rhs @ inv_A
            pinv_fallback = False
        except Exception:
            pinv_fallback = True
            try:
                inv_A = np.linalg.pinv(A_tikh)
                B_struct = rhs @ inv_A
            except Exception:
                B_struct = np.zeros((self.n_j, self.n_u))
                inv_A = np.zeros((self.n_u, self.n_u))

        # For diagnostics
        B_blp = Sigma_YX_reg @ np.linalg.pinv(Sigma_XX_reg + self.rho * diag_mean * np.eye(self.n_u))
        norm_B_blp = np.linalg.norm(B_blp, "fro")
        norm_B_struct = np.linalg.norm(B_struct, "fro")

        # Standardize current US input
        X_t = all_returns[current_index, : self.n_u]
        X_t = np.nan_to_num(X_t, nan=0.0, posinf=0.0, neginf=0.0)
        mu_X = mu[: self.n_u]
        sigma_X = sigma[: self.n_u]
        sigma_X_safe = np.where(sigma_X > 1e-8, sigma_X, 1.0)
        z_U_t = (X_t - mu_X) / sigma_X_safe

        # Predict standardized JP returns
        z_hat_j_t1 = B_struct @ z_U_t
        z_hat_j_t1 = np.nan_to_num(z_hat_j_t1, nan=0.0, posinf=0.0, neginf=0.0)

        # Confidence weighting
        Sigma_XY_reg = Sigma_YX_reg.T
        Sigma_Y_given_X = Sigma_YY_reg - Sigma_YX_reg @ inv_A @ Sigma_XY_reg
        pred_var = np.maximum(np.diag(Sigma_Y_given_X), 0.0)
        pred_var_floored = np.maximum(pred_var, 1e-8)

        if self.beta_conf > 0.0:
            z_hat_j_t1 = z_hat_j_t1 / (pred_var_floored ** self.beta_conf)
            z_hat_j_t1 = np.clip(z_hat_j_t1, -5.0, 5.0)

        # Vol adjustment / denormalization
        mu_jp = mu[self.n_u:]
        sigma_jp = sigma[self.n_u:]
        if self.vol_adjusted_target:
            if current_index >= 20:
                jp_returns_20 = all_returns[current_index - 20: current_index, self.n_u:]
                jp_returns_20 = np.nan_to_num(jp_returns_20, nan=0.0, posinf=0.0, neginf=0.0)
                sigma_j_t = np.std(jp_returns_20, axis=0, ddof=1)
                sigma_j_t = np.maximum(sigma_j_t, 1e-8)
            else:
                sigma_j_t = sigma_jp
            r_hat_jp_cc = z_hat_j_t1 * sigma_j_t
        else:
            r_hat_jp_cc = mu_jp + sigma_jp * z_hat_j_t1
        r_hat_jp_cc = np.nan_to_num(r_hat_jp_cc, nan=0.0, posinf=0.0, neginf=0.0)

        # Apply gap override
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
                    self.gap_open_coef * gap_idio
                    + (self.gap_open_coef - self.topix_beta_coef) * gap_syst
                )
                denom = np.maximum(1.0 + gap_filt, 0.1)
                signal = (1.0 + r_hat_jp_cc) / denom - 1.0
            else:
                signal = r_hat_jp_cc - self.gap_open_coef * gap_vec
        else:
            signal = z_hat_j_t1

        # Condition number
        try:
            sv = np.linalg.svd(A_tikh, compute_uv=False)
            cond_num = float(sv[0] / np.maximum(sv[-1], 1e-12))
        except Exception:
            cond_num = np.nan

        return {
            "signal": signal,
            "z_hat_j_t1": z_hat_j_t1,
            "cond_num": cond_num,
            "b_norm": norm_B_blp,
            "b_pca_norm": float(np.linalg.norm(B_pca)),
            "b_sector_norm": float(np.linalg.norm(M_sector)),
            "b_struct_norm": norm_B_struct,
            "sigma_xx_trace": float(np.trace(C_XX)),
            "sigma_yx_norm": float(np.linalg.norm(C_YX)),
            "sigma_yy_trace": float(np.trace(C_YY)),
            "min_pred_var": float(np.min(pred_var)),
            "max_pred_var": float(np.max(pred_var)),
            "num_pred_var_floored": 0,
            "pinv_fallback": pinv_fallback,
            "num_training_samples": len(window_returns),
        }


# ---------------------------------------------------------------------------
# Approach 6: Continuous M_sector (data-driven weight refinement)
# ---------------------------------------------------------------------------

# Structural mapping: which JP tickers relate to which US tickers
_SECTOR_MAPPING_STRUCTURE = {
    "XLB": ["1620.T", "1623.T"],
    "XLC": ["1626.T"],
    "XLE": ["1618.T", "1627.T"],
    "XLF": ["1631.T", "1632.T"],
    "XLI": ["1624.T", "1622.T", "1626.T"],
    "XLK": ["1626.T", "1625.T"],
    "XLP": ["1617.T", "1630.T"],
    "XLRE": ["1633.T"],
    "XLU": ["1627.T"],
    "XLV": ["1621.T"],
    "XLY": ["1630.T", "1626.T", "1622.T"],
    "MTUM": ["1625.T", "1626.T"],
    "VLUE": ["1631.T", "1632.T", "1623.T", "1622.T"],
    "IUSG": ["1626.T", "1625.T"],
    "USMV": ["1617.T", "1621.T", "1627.T"],
}


class ContinuousSectorModel(SectorRelativeEnsembleBLPEnhancedModel):
    """Approach 6: Continuous M_sector via data-driven weight refinement.

    Preserves the structural mapping (which US ETF maps to which JP tickers)
    but replaces fixed discrete weights with continuous weights derived from
    rolling cross-correlation.

    For each US ticker u with mapped JP tickers {j1, j2, ...}:
      w_ji = corr(u, ji)^gamma / sum_k corr(u, jk)^gamma

    A shrinkage parameter eta blends between the fixed prior and data-driven:
      w_final = (1-eta) * w_fixed + eta * w_data

    This keeps domain knowledge (structural relationships) while making
    the weight magnitudes adaptive to market conditions.
    """

    def __init__(self, config: dict):
        self._fixed_M = None  # placeholder, set after n_j/n_u available
        super().__init__(config)
        self._fixed_M = self._build_fixed_mapping()
        self.sector_eta = float(self._resolve_val("sector_eta", 0.5))
        self.sector_gamma = float(self._resolve_val("sector_gamma", 2.0))
        logger.info(f"ContinuousSector: eta={self.sector_eta}, gamma={self.sector_gamma}")

    def _build_fixed_mapping(self) -> np.ndarray:
        """Build the original fixed mapping for blending."""
        M = np.zeros((self.n_j, self.n_u))
        fixed_weights = {
            "XLB": {"1620.T": 0.5, "1623.T": 0.5},
            "XLC": {"1626.T": 1.0},
            "XLE": {"1618.T": 0.5, "1627.T": 0.5},
            "XLF": {"1631.T": 0.5, "1632.T": 0.5},
            "XLI": {"1624.T": 0.33, "1622.T": 0.33, "1626.T": 0.34},
            "XLK": {"1626.T": 0.5, "1625.T": 0.5},
            "XLP": {"1617.T": 0.5, "1630.T": 0.5},
            "XLRE": {"1633.T": 1.0},
            "XLU": {"1627.T": 1.0},
            "XLV": {"1621.T": 1.0},
            "XLY": {"1630.T": 0.33, "1626.T": 0.33, "1622.T": 0.34},
            "MTUM": {"1625.T": 0.5, "1626.T": 0.5},
            "VLUE": {"1631.T": 0.25, "1632.T": 0.25, "1623.T": 0.25, "1622.T": 0.25},
            "IUSG": {"1626.T": 0.5, "1625.T": 0.5},
            "USMV": {"1617.T": 0.33, "1621.T": 0.33, "1627.T": 0.34},
        }
        for u_idx, us_tk in enumerate(US_TICKERS):
            if us_tk in fixed_weights:
                for jp_tk, w in fixed_weights[us_tk].items():
                    if jp_tk in JP_TICKERS:
                        j_idx = JP_TICKERS.index(jp_tk)
                        M[j_idx, u_idx] = w
        col_sums = np.sum(M, axis=0)
        for u_idx in range(self.n_u):
            if col_sums[u_idx] > 0:
                M[:, u_idx] /= col_sums[u_idx]
        return M

    def _build_sector_prior(self) -> np.ndarray:
        if self._fixed_M is not None:
            return self._fixed_M.copy()
        return np.zeros((self.n_j, self.n_u))

    def _get_sector_prior(
        self,
        current_index: int,
        all_returns: np.ndarray,
        corr: np.ndarray,
        B_blp: np.ndarray,
    ) -> np.ndarray:
        """Build continuous M_sector by blending fixed structure with rolling correlation."""
        # Extract cross-correlation block from corr (32x32)
        # corr[:n_u, n_u:] = C_XY (15x17), corr[n_u:, :n_u] = C_YX (17x15)
        if corr.shape != (self.n_u + self.n_j, self.n_u + self.n_j):
            return self._fixed_M if self._fixed_M.shape == B_blp.shape else np.zeros(B_blp.shape)

        c_xy = corr[: self.n_u, self.n_u:]  # (15, 17) — US vs JP cross-corr

        M_data = np.zeros((self.n_j, self.n_u))
        for u_idx, us_tk in enumerate(US_TICKERS):
            if us_tk not in _SECTOR_MAPPING_STRUCTURE:
                continue
            jp_tickers = _SECTOR_MAPPING_STRUCTURE[us_tk]
            # Get correlations for this US ticker's mapped JP tickers
            weights = []
            for jp_tk in jp_tickers:
                if jp_tk in JP_TICKERS:
                    j_idx = JP_TICKERS.index(jp_tk)
                    raw_corr = c_xy[u_idx, j_idx]
                    # Apply gamma transform: emphasizes stronger correlations
                    weights.append((j_idx, max(0.0, raw_corr) ** self.sector_gamma))
            if not weights:
                continue
            total = sum(w for _, w in weights)
            if total > 1e-10:
                for j_idx, w in weights:
                    M_data[j_idx, u_idx] = w / total

        # Blend fixed and data-driven
        eta = self.sector_eta
        M_blended = (1.0 - eta) * self._fixed_M + eta * M_data

        # Column normalize
        col_sums = np.sum(M_blended, axis=0)
        for u_idx in range(self.n_u):
            if col_sums[u_idx] > 1e-10:
                M_blended[:, u_idx] /= col_sums[u_idx]

        if M_blended.shape == B_blp.shape:
            return M_blended
        return np.zeros(B_blp.shape)


# ---------------------------------------------------------------------------
# Approach 5: C0 cross-block + Multi-target Tikhonov (combined)
# ---------------------------------------------------------------------------


class C0TikhonovModel(TikhonovModel):
    """Approach 5: C0 cross-block derived M_sector + Tikhonov regularization.

    Combines approaches 1 and 3:
    - M_sector is derived from V0's implied cross-correlation (C0 cross-block)
    - The derived M_sector is used as prior mean in Tikhonov regularization

    This eliminates both the hand-crafted mapping AND the post-hoc linear mixing.
    """

    def __init__(self, config: dict):
        super().__init__(config)

    def _build_sector_prior(self) -> np.ndarray:
        """Override: return placeholder, actual M_sector computed dynamically."""
        return np.eye(self.n_j, self.n_u)

    def _get_sector_prior(
        self,
        current_index: int,
        all_returns: np.ndarray,
        corr: np.ndarray,
        B_blp: np.ndarray,
    ) -> np.ndarray:
        """Derive M_sector from C0's cross-block (same as C0CrossBlockModel)."""
        v0 = build_v3_static(self.n_u, self.n_j, self.include_v4_prior)
        window_start = max(0, current_index - self.blp_window)
        window_returns = all_returns[window_start:current_index]
        window_returns = np.nan_to_num(window_returns, nan=0.0, posinf=0.0, neginf=0.0)
        _, _, c_full_local = compute_correlation(window_returns, self.blp_ewma_halflife)
        c_full_local = np.nan_to_num(c_full_local, nan=0.0, posinf=1.0, neginf=-1.0)
        np.fill_diagonal(c_full_local, 1.0)

        c0 = build_c0_from_v0(v0, c_full_local)
        c0_yu = c0[self.n_u:, :self.n_u]

        col_sums = np.sum(np.abs(c0_yu), axis=0)
        m_sector = np.zeros_like(c0_yu)
        for u_idx in range(self.n_u):
            if col_sums[u_idx] > 1e-10:
                m_sector[:, u_idx] = c0_yu[:, u_idx] / col_sums[u_idx]

        if m_sector.shape == B_blp.shape:
            return m_sector
        return np.zeros(B_blp.shape)


# ---------------------------------------------------------------------------
# Approach 4: RRR with structured mean
# ---------------------------------------------------------------------------


class RRRStructuredMeanModel(SectorRelativeEnsembleBLPEnhancedModel):
    """Approach 4: Reduced Rank Regression with structured mean.

    Model: B = M_sector + U @ V^T  where U (17xK), V (15xK), rank K <= k

    Solve via alternating least squares:
      1. Given U,V: residualize Y' = Y - M_sector @ X, fit U,V on Y' vs X
      2. Iterate until convergence

    The M_sector serves as a structured mean, and the low-rank U@V^T captures
    data-driven deviations from it.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.rrr_rank = int(self._resolve_val("rrr_rank", 6))
        self.rrr_lambda = float(self._resolve_val("rrr_lambda", 0.01))
        logger.info(f"RRRStructuredMean: rank={self.rrr_rank}, lambda={self.rrr_lambda}")

    def _build_sector_prior(self) -> np.ndarray:
        """Keep the original M_sector as structured mean."""
        M = np.zeros((self.n_j, self.n_u))
        mapping = {
            "XLB": {"1620.T": 0.5, "1623.T": 0.5},
            "XLC": {"1626.T": 1.0},
            "XLE": {"1618.T": 0.5, "1627.T": 0.5},
            "XLF": {"1631.T": 0.5, "1632.T": 0.5},
            "XLI": {"1624.T": 0.33, "1622.T": 0.33, "1626.T": 0.34},
            "XLK": {"1626.T": 0.5, "1625.T": 0.5},
            "XLP": {"1617.T": 0.5, "1630.T": 0.5},
            "XLRE": {"1633.T": 1.0},
            "XLU": {"1627.T": 1.0},
            "XLV": {"1621.T": 1.0},
            "XLY": {"1630.T": 0.33, "1626.T": 0.33, "1622.T": 0.34},
            "MTUM": {"1625.T": 0.5, "1626.T": 0.5},
            "VLUE": {"1631.T": 0.25, "1632.T": 0.25, "1623.T": 0.25, "1622.T": 0.25},
            "IUSG": {"1626.T": 0.5, "1625.T": 0.5},
            "USMV": {"1617.T": 0.33, "1621.T": 0.33, "1627.T": 0.34},
        }
        for u_idx, us_tk in enumerate(US_TICKERS):
            if us_tk in mapping:
                for jp_tk, w in mapping[us_tk].items():
                    if jp_tk in JP_TICKERS:
                        j_idx = JP_TICKERS.index(jp_tk)
                        M[j_idx, u_idx] = w
        col_sums = np.sum(M, axis=0)
        for u_idx in range(self.n_u):
            if col_sums[u_idx] > 0:
                M[:, u_idx] /= col_sums[u_idx]
        return M

    def _get_sector_prior(
        self,
        current_index: int,
        all_returns: np.ndarray,
        corr: np.ndarray,
        B_blp: np.ndarray,
    ) -> np.ndarray:
        """Return zeros; M_sector is used as structured mean in compute_blp_signal."""
        return np.zeros(B_blp.shape)

    def compute_blp_signal(
        self,
        all_returns: np.ndarray,
        current_index: int,
        gap_override: np.ndarray | None = None,
        betas_t: np.ndarray | None = None,
        topix_night_t: float | None = None,
        rolling_std: np.ndarray | None = None,
        v0_static: np.ndarray | None = None,
        c_full: np.ndarray | None = None,
        is_residual: bool = False,
        return_matrices: bool = False,
    ) -> dict[str, Any]:
        """Override to use RRR with structured mean."""

        window_start = max(0, current_index - self.blp_window)
        window_returns = all_returns[window_start:current_index].copy()

        if self.exec_adjustment == "vol_scale" and rolling_std is not None:
            for idx_local in range(len(window_returns)):
                idx_global = window_start + idx_local
                vol_factor = rolling_std[idx_global]
                window_returns[idx_local, self.n_u:] /= vol_factor

        window_returns = np.nan_to_num(window_returns, nan=0.0, posinf=0.0, neginf=0.0)

        if self.winsor_sigma is not None:
            for c in range(window_returns.shape[1]):
                mu_c = np.mean(window_returns[:, c])
                std_c = np.std(window_returns[:, c])
                if std_c > 1e-8:
                    window_returns[:, c] = np.clip(
                        window_returns[:, c],
                        mu_c - self.winsor_sigma * std_c,
                        mu_c + self.winsor_sigma * std_c,
                    )

        mu, sigma, corr = compute_correlation(window_returns, self.blp_ewma_halflife)
        mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
        sigma = np.nan_to_num(sigma, nan=1.0, posinf=1.0, neginf=1.0)
        corr = np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0)
        np.fill_diagonal(corr, 1.0)

        C_XX = corr[: self.n_u, : self.n_u]
        C_YX = corr[self.n_u :, : self.n_u]
        C_YY = corr[self.n_u :, self.n_u :]

        Sigma_XX_reg = (1.0 - self.alpha_xx) * C_XX + self.alpha_xx * np.eye(self.n_u)
        Sigma_YX_reg = (1.0 - self.alpha_yx) * C_YX
        Sigma_YY_reg = (1.0 - self.alpha_yy) * C_YY + self.alpha_yy * np.eye(self.n_j)

        diag_mean = float(np.mean(np.diag(Sigma_XX_reg)))
        ridge_matrix = self.rho * diag_mean * np.eye(self.n_u)
        A = Sigma_XX_reg + ridge_matrix

        # --- Standard BLP for baseline ---
        try:
            inv_A = np.linalg.inv(A)
            B_blp = Sigma_YX_reg @ inv_A
            pinv_fallback = False
        except Exception:
            pinv_fallback = True
            try:
                inv_A = np.linalg.pinv(A)
                B_blp = Sigma_YX_reg @ inv_A
            except Exception:
                B_blp = np.zeros((self.n_j, self.n_u))
                inv_A = np.zeros((self.n_u, self.n_u))

        # --- RRR with structured mean ---
        # B = M_sector + U @ V^T
        # Residualize: Sigma_YX_res = Sigma_YX - M_sector @ Sigma_XX
        M_sec = self.M_sector
        if M_sec.shape != B_blp.shape:
            M_sec = np.zeros(B_blp.shape)

        Sigma_YX_res = Sigma_YX_reg - M_sec @ Sigma_XX_reg

        # Solve ridge regression on residual: U @ V^T = Sigma_YX_res @ inv(A_ridge)
        A_rrr = Sigma_XX_reg + (self.rho + self.rrr_lambda) * diag_mean * np.eye(self.n_u)
        try:
            inv_A_rrr = np.linalg.inv(A_rrr)
        except Exception:
            inv_A_rrr = np.linalg.pinv(A_rrr)

        B_res = Sigma_YX_res @ inv_A_rrr

        # Low-rank approximation via SVD
        rank_val = min(self.rrr_rank, min(B_res.shape))
        try:
            U_svd, S_svd, Vt_svd = np.linalg.svd(B_res, full_matrices=False)
            B_low_rank = U_svd[:, :rank_val] @ np.diag(S_svd[:rank_val]) @ Vt_svd[:rank_val, :]
        except Exception:
            B_low_rank = np.zeros_like(B_res)

        # Final B = M_sector + low-rank correction
        B_struct = M_sec + B_low_rank

        norm_B_blp = np.linalg.norm(B_blp, "fro")
        norm_B_struct = np.linalg.norm(B_struct, "fro")

        # B_pca for diagnostics
        B_pca = np.zeros((self.n_j, self.n_u))
        if v0_static is not None and c_full is not None and corr.shape == (32, 32):
            c0_t = build_c0_from_v0(v0_static, c_full)
            c_t_reg = regularize_correlation(
                corr, c0_t, self.lambda_reg, self.lambda_lw, self.lw_target
            )
            eigvals, eigvecs = np.linalg.eigh(c_t_reg)
            sort_idx = np.argsort(eigvals)[::-1]
            eigvecs = eigvecs[:, sort_idx]
            v_t_k = eigvecs[:, : self.k]
            v_u_t_k = v_t_k[: self.n_u, :]
            v_j_t_k = v_t_k[self.n_u :, :]
            B_pca = v_j_t_k @ v_u_t_k.T

        # Standardize current US input
        X_t = all_returns[current_index, : self.n_u]
        X_t = np.nan_to_num(X_t, nan=0.0, posinf=0.0, neginf=0.0)
        mu_X = mu[: self.n_u]
        sigma_X = sigma[: self.n_u]
        sigma_X_safe = np.where(sigma_X > 1e-8, sigma_X, 1.0)
        z_U_t = (X_t - mu_X) / sigma_X_safe

        # Predict
        z_hat_j_t1 = B_struct @ z_U_t
        z_hat_j_t1 = np.nan_to_num(z_hat_j_t1, nan=0.0, posinf=0.0, neginf=0.0)

        # Confidence weighting
        Sigma_XY_reg = Sigma_YX_reg.T
        Sigma_Y_given_X = Sigma_YY_reg - Sigma_YX_reg @ inv_A @ Sigma_XY_reg
        pred_var = np.maximum(np.diag(Sigma_Y_given_X), 0.0)
        pred_var_floored = np.maximum(pred_var, 1e-8)

        if self.beta_conf > 0.0:
            z_hat_j_t1 = z_hat_j_t1 / (pred_var_floored ** self.beta_conf)
            z_hat_j_t1 = np.clip(z_hat_j_t1, -5.0, 5.0)

        # Vol adjustment / denormalization
        mu_jp = mu[self.n_u:]
        sigma_jp = sigma[self.n_u:]
        if self.vol_adjusted_target:
            if current_index >= 20:
                jp_returns_20 = all_returns[current_index - 20: current_index, self.n_u:]
                jp_returns_20 = np.nan_to_num(jp_returns_20, nan=0.0, posinf=0.0, neginf=0.0)
                sigma_j_t = np.std(jp_returns_20, axis=0, ddof=1)
                sigma_j_t = np.maximum(sigma_j_t, 1e-8)
            else:
                sigma_j_t = sigma_jp
            r_hat_jp_cc = z_hat_j_t1 * sigma_j_t
        else:
            r_hat_jp_cc = mu_jp + sigma_jp * z_hat_j_t1
        r_hat_jp_cc = np.nan_to_num(r_hat_jp_cc, nan=0.0, posinf=0.0, neginf=0.0)

        # Apply gap override
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
                    self.gap_open_coef * gap_idio
                    + (self.gap_open_coef - self.topix_beta_coef) * gap_syst
                )
                denom = np.maximum(1.0 + gap_filt, 0.1)
                signal = (1.0 + r_hat_jp_cc) / denom - 1.0
            else:
                signal = r_hat_jp_cc - self.gap_open_coef * gap_vec
        else:
            signal = z_hat_j_t1

        # Condition number
        try:
            sv = np.linalg.svd(A, compute_uv=False)
            cond_num = float(sv[0] / np.maximum(sv[-1], 1e-12))
        except Exception:
            cond_num = np.nan

        return {
            "signal": signal,
            "z_hat_j_t1": z_hat_j_t1,
            "cond_num": cond_num,
            "b_norm": norm_B_blp,
            "b_pca_norm": float(np.linalg.norm(B_pca)),
            "b_sector_norm": float(np.linalg.norm(M_sec)),
            "b_struct_norm": norm_B_struct,
            "sigma_xx_trace": float(np.trace(C_XX)),
            "sigma_yx_norm": float(np.linalg.norm(C_YX)),
            "sigma_yy_trace": float(np.trace(C_YY)),
            "min_pred_var": float(np.min(pred_var)),
            "max_pred_var": float(np.max(pred_var)),
            "num_pred_var_floored": 0,
            "pinv_fallback": pinv_fallback,
            "num_training_samples": len(window_returns),
        }


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------


def main():
    logger.info("=" * 70)
    logger.info("M_sector Refinement Experiment")
    logger.info("=" * 70)

    # Download and preprocess data
    logger.info("[1/3] Downloading market data...")
    data = download_data(beta_window=60)

    logger.info("[2/3] Preprocessing data...")
    df_exec = preprocess_data(data, beta_window=60)

    logger.info(f"Data range: {df_exec.index[0]} to {df_exec.index[-1]}, {len(df_exec)} rows")

    # Define experiments
    experiments = {
        "baseline": {
            "model_class": SectorRelativeEnsembleBLPEnhancedModel,
            "config_overrides": {},
            "description": "Production baseline (fixed M_sector, linear mixing)",
        },
        "approach1_c0_cross": {
            "model_class": C0CrossBlockModel,
            "config_overrides": {},
            "description": "C0 cross-block derived M_sector",
        },
        "approach2_v0_ext": {
            "model_class": V0ExtensionModel,
            "config_overrides": {"lambda_sector": 0.0, "lambda_pca": 0.10},
            "description": "V0 extended with sector-pair vectors",
        },
        "approach3_tikhonov": {
            "model_class": TikhonovModel,
            "config_overrides": {},
            "description": "Multi-target Tikhonov regularization",
        },
        "approach4_rrr": {
            "model_class": RRRStructuredMeanModel,
            "config_overrides": {
                "lambda_sector": 0.0,
                "lambda_pca": 0.0,
                "rrr_rank": 6,
                "rrr_lambda": 0.01,
            },
            "description": "RRR with structured mean (M_sector + low-rank)",
        },
        "approach5_c0_tikhonov": {
            "model_class": C0TikhonovModel,
            "config_overrides": {},
            "description": "C0 cross-block M_sector + Tikhonov regularization (no hand-crafted mapping)",
        },
        "approach6_continuous": {
            "model_class": ContinuousSectorModel,
            "config_overrides": {"sector_eta": 0.5, "sector_gamma": 2.0},
            "description": "Continuous M_sector (rolling-corr weighted, eta=0.5, gamma=2.0)",
        },
    }

    results_all = {}

    logger.info("[3/3] Running backtests...")
    for name, exp in experiments.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {name} — {exp['description']}")
        logger.info(f"{'='*50}")

        cfg = build_model_config(exp["config_overrides"])
        model = exp["model_class"](cfg)

        try:
            metrics = run_backtest(model, df_exec)
            results_all[name] = metrics

            logger.info(f"Results for {name}:")
            logger.info(f"  Sharpe:  {metrics.get('Sharpe', float('nan')):.4f}")
            logger.info(f"  AR:      {metrics.get('AR', float('nan'))*100:.2f}%")
            logger.info(f"  RISK:    {metrics.get('RISK', float('nan'))*100:.2f}%")
            logger.info(f"  MDD:     {metrics.get('MDD', float('nan'))*100:.2f}%")
            logger.info(f"  Turnover: {metrics.get('avg_turnover', float('nan')):.4f}")
            logger.info(f"  Cost/day: {metrics.get('avg_cost', float('nan'))*100:.4f}%")
        except Exception as e:
            logger.error(f"Failed for {name}: {e}", exc_info=True)
            results_all[name] = {"error": str(e)}

    # Summary table
    logger.info(f"\n{'='*70}")
    logger.info("SUMMARY: M_sector Refinement Approaches vs Baseline")
    logger.info(f"{'='*70}")

    rows = []
    for name, metrics in results_all.items():
        if "error" in metrics:
            rows.append({"Approach": name, "Sharpe": "ERROR", "AR": "—", "RISK": "—", "MDD": "—", "Turnover": "—", "Cost/day": "—"})
        else:
            rows.append({
                "Approach": name,
                "Sharpe": f"{metrics.get('Sharpe', float('nan')):.4f}",
                "AR": f"{metrics.get('AR', float('nan'))*100:.2f}%",
                "RISK": f"{metrics.get('RISK', float('nan'))*100:.2f}%",
                "MDD": f"{metrics.get('MDD', float('nan'))*100:.2f}%",
                "Turnover": f"{metrics.get('avg_turnover', float('nan')):.4f}",
                "Cost/day": f"{metrics.get('avg_cost', float('nan'))*10000:.2f}bps",
            })

    summary_df = pd.DataFrame(rows)
    logger.info(f"\n{summary_df.to_string(index=False)}")

    # Save results
    out_dir = ROOT / "artifacts" / "m_sector_refinement_experiment"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_dir / "summary.csv", index=False)

    # Save detailed metrics
    detailed = {}
    for name, metrics in results_all.items():
        if "error" not in metrics:
            detailed[name] = metrics
    pd.DataFrame(detailed).T.to_csv(out_dir / "detailed_metrics.csv")

    logger.info(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
