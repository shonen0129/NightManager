"""Legacy Sector Relative Ensemble (PCA-Ensemble) model.

This class was moved from ``src/leadlag/models/sre.py`` during the V1 model cleanup.
It remains importable from ``archive.legacy_src.models.sre`` for archived
experiments and tests.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from leadlag.core.correlation import (
    compute_correlation,
    get_static_sensitivity_labels,
)
from leadlag.data.tickers import JP_TICKERS, US_TICKERS
from leadlag.models.base import BaseModel

logger = logging.getLogger(__name__)


class SectorRelativeEnsembleModel(BaseModel):
    """Sector Relative Ensemble (PCA-Ensemble) Model.

    Ensembles standard Production signal (Raw-PCA), TOPIX-residualized Production signal (Residual-PCA),
    and SPY/TOPIX-residualized signal (P4).
    """

    _config_sections = ["model", "ensemble", "portfolio", "costs", "residualization", "audit", "output", "prior"]

    def __init__(self, config: dict | object):
        """Initialize SectorRelativeEnsembleModel.

        Args:
            config: Dict or object containing configuration options.
        """
        self.config = config
        self.n_u = len(US_TICKERS)
        self.n_j = len(JP_TICKERS)

        # Config resolution
        self.model_name = self._resolve_val("model_name", "sector_relative_ensemble")
        self.k = self._resolve_val("k", 6)
        self.lambda_reg = self._resolve_val("lambda_reg", 0.75)
        self.q = self._resolve_val("q", 0.3)
        self.weight_mode = self._resolve_val("weight_mode", "signal")
        self.ewma_half_life = self._resolve_val("ewma_half_life", 45)
        self.lambda_lw = self._resolve_val("lambda_lw", 0.5)
        self.lw_target = self._resolve_val("lw_target", "equicorrelation")
        self.corr_window = self._resolve_val("corr_window", 60)
        self.include_v4_prior = self._resolve_val("include_v4_prior", True)
        self.gap_open_coef = self._resolve_val("gap_open_coef", 0.70)
        self.topix_beta_coef = self._resolve_val("topix_beta_coef", 0.6)
        self.beta_window = self._resolve_val("beta_window", 60)
        self.vol_adjusted_target = self._resolve_val("vol_adjusted_target", True)
        self.min_raw_weight = self._resolve_val("min_raw_weight", 0.0)

        # Resolve ensemble weights
        self.raw_pca_weight = float(self._resolve_val("raw_pca_weight", 0.5))
        self.residual_pca_weight = float(self._resolve_val("residual_pca_weight", 0.5))
        self.p4_weight = float(self._resolve_val("p4_weight", 0.0))

        # Resolve US residualization config
        self.us_res_enabled = self._resolve_nested("us_residualization.enabled", self.p4_weight > 0.0)
        self.us_res_benchmark = self._resolve_nested("us_residualization.benchmark", "SPY")
        self.us_res_beta_window = int(self._resolve_nested("us_residualization.beta_window", 60))
        self.us_res_beta_shift = int(self._resolve_nested("us_residualization.beta_shift", 1))
        self.us_res_gamma = float(self._resolve_nested("us_residualization.gamma", 0.5))

        # Resolve fractional differencing config (align with _BLPBase models)
        frac_cfg = {}
        if isinstance(self.config, dict):
            frac_cfg = self.config.get("features", {}).get("fractional_diff", {})
        self.frac_diff_enabled = bool(frac_cfg.get("enabled", False))
        self.frac_diff_d = float(frac_cfg.get("d", 0.5))
        self.frac_diff_threshold = float(frac_cfg.get("threshold", 1e-5))
        self.frac_diff_window = int(frac_cfg.get("window", 100))
        self.frac_diff_normalize = frac_cfg.get("normalize", None)
        if self.frac_diff_normalize is not None:
            self.frac_diff_normalize = str(self.frac_diff_normalize)

        # Resolve US Residual Prior config
        self.prior_variant = self._resolve_nested("prior.variant", None)
        self.normalization_method = self._resolve_val("normalization", "zscore")

        # Slippage cost parameter resolution
        self.slippage_bps = self._resolve_slippage_bps()

        # Overnight holding parameters
        self.overnight_alpha_long = self._resolve_val("overnight_alpha_long", 0.0)
        self.overnight_alpha_short = self._resolve_val("overnight_alpha_short", 0.0)
        self.buy_interest_annual = self._resolve_val("buy_interest_annual", 0.0)
        self.borrow_fee_annual = self._resolve_val("borrow_fee_annual", 0.0)
        self.reverse_fee_bps = self._resolve_val("reverse_fee_bps", 0.0)

        # Per-instance caches (replaces module-level globals for thread safety)
        self._common_inputs_cache: dict = {}
        self._production_signal_cache: dict = {}
        self._residual_signal_cache: dict = {}
        self._residual_prior_cache: dict = {}

    def _prepare_common_inputs(self, df_exec: pd.DataFrame) -> dict:
        """Prepare and compute arrays and target matrices common to backtesting and live run."""
        cache_key = (
            len(df_exec),
            df_exec.index[0],
            df_exec.index[-1],
            self.ewma_half_life,
            self.beta_window,
            self.include_v4_prior,
            self.us_res_enabled,
            self.us_res_gamma,
            self.us_res_beta_window,
            self.frac_diff_enabled,
            self.frac_diff_d,
            self.frac_diff_threshold,
            self.frac_diff_window,
            self.frac_diff_normalize,
        )
        if cache_key in self._common_inputs_cache:
            cached_val = self._common_inputs_cache[cache_key]
            self.v0_static_obj = cached_val["v0_static"]
            return cached_val.copy()

        y_jp_target = compute_jp_target_returns(df_exec, JP_TICKERS)

        from leadlag.core.pipeline import build_common_inputs
        inputs = build_common_inputs(
            df_exec,
            y_jp_target,
            n_u=self.n_u,
            n_j=self.n_j,
            ewma_half_life=self.ewma_half_life,
            beta_window=self.beta_window,
            include_v4_prior=self.include_v4_prior,
            us_res_enabled=self.us_res_enabled,
            us_res_gamma=self.us_res_gamma,
            us_res_beta_window=self.us_res_beta_window,
            frac_diff_enabled=self.frac_diff_enabled,
            frac_diff_d=self.frac_diff_d,
            frac_diff_threshold=self.frac_diff_threshold,
            frac_diff_window=self.frac_diff_window,
            frac_diff_normalize=self.frac_diff_normalize,
        )
        self.v0_static_obj = inputs.v0_static
        out = inputs.to_dict()
        out["y_jp_target"] = y_jp_target

        self._common_inputs_cache[cache_key] = out
        return out

    @staticmethod
    def _orthogonalize_and_normalize(
        vector: np.ndarray, basis: list[np.ndarray]
    ) -> np.ndarray:
        """Remove basis components from vector and normalize."""
        result = vector.astype(float, copy=True)
        for base in basis:
            result = result - (result @ base) * base
        norm = np.linalg.norm(result)
        if norm <= 1e-10:
            return np.asarray(np.zeros_like(result), dtype=float)
        return np.asarray(result / norm, dtype=float)

    def _select_residual_window(
        self,
        df_exec: pd.DataFrame,
        *,
        baseline_start: str = "2010-01-01",
        baseline_end: str = "2014-12-31",
        fallback_size: int = 1260,
    ) -> np.ndarray:
        """Select the baseline-period row indices for residual prior estimation.

        Falls back to the first `fallback_size` rows when the baseline period
        is not present, preserving the original fallback behavior.
        """
        sim_dates = df_exec.index
        baseline_mask = (sim_dates >= pd.to_datetime(baseline_start)) & (
            sim_dates <= pd.to_datetime(baseline_end)
        )
        baseline_indices = np.where(baseline_mask)[0]
        if len(baseline_indices) == 0:
            baseline_indices = np.arange(min(len(df_exec), fallback_size))
        return baseline_indices

    def _build_residual_prior_vectors(self) -> tuple:
        """Build V0_resid and related source/scaling metadata."""
        n = self.n_u + self.n_j

        _labels = get_static_sensitivity_labels()
        w3 = _labels["w3"]
        w4 = _labels["w4"]
        w5 = _labels["w5"]
        w6 = _labels["w6"]

        v1_new = np.ones(n) / np.sqrt(n)
        denom = np.sqrt(float(self.n_u * self.n_j * n))
        v2_new = np.zeros(n)
        v2_new[: self.n_u] = self.n_j / denom
        v2_new[self.n_u :] = -self.n_u / denom

        v1_v2_scale: float | None = None
        c0_source = "residualized"
        k_expected = 6

        if self.prior_variant == "raw_v1_to_v6":
            v3_new = self._orthogonalize_and_normalize(w3, [v1_new, v2_new])
            v4_new = self._orthogonalize_and_normalize(w4, [v1_new, v2_new, v3_new])
            v5_new = self._orthogonalize_and_normalize(w5, [v1_new, v2_new, v3_new, v4_new])
            v6_new = self._orthogonalize_and_normalize(w6, [v1_new, v2_new, v3_new, v4_new, v5_new])
            V0_resid = np.column_stack([v1_new, v2_new, v3_new, v4_new, v5_new, v6_new])
            c0_source = "raw_existing"
            k_expected = 6
        elif self.prior_variant == "resid_v2_removed":
            v3_new = self._orthogonalize_and_normalize(w3, [v1_new])
            v4_new = self._orthogonalize_and_normalize(w4, [v1_new, v3_new])
            v5_new = self._orthogonalize_and_normalize(w5, [v1_new, v3_new, v4_new])
            v6_new = self._orthogonalize_and_normalize(w6, [v1_new, v3_new, v4_new, v5_new])
            V0_resid = np.column_stack([v1_new, v3_new, v4_new, v5_new, v6_new])
            k_expected = 5
        elif self.prior_variant == "resid_v1_v2_removed":
            v3_new = w3 / np.linalg.norm(w3)
            v4_new = self._orthogonalize_and_normalize(w4, [v3_new])
            v5_new = self._orthogonalize_and_normalize(w5, [v3_new, v4_new])
            v6_new = self._orthogonalize_and_normalize(w6, [v3_new, v4_new, v5_new])
            V0_resid = np.column_stack([v3_new, v4_new, v5_new, v6_new])
            k_expected = 4
        elif self.prior_variant == "resid_v1_v2_scaled_025":
            v3_new = self._orthogonalize_and_normalize(w3, [v1_new, v2_new])
            v4_new = self._orthogonalize_and_normalize(w4, [v1_new, v2_new, v3_new])
            v5_new = self._orthogonalize_and_normalize(w5, [v1_new, v2_new, v3_new, v4_new])
            v6_new = self._orthogonalize_and_normalize(w6, [v1_new, v2_new, v3_new, v4_new, v5_new])
            V0_resid = np.column_stack([v1_new, v2_new, v3_new, v4_new, v5_new, v6_new])
            v1_v2_scale = 0.25
            k_expected = 6
        elif self.prior_variant == "resid_v1_v2_scaled_050":
            v3_new = self._orthogonalize_and_normalize(w3, [v1_new, v2_new])
            v4_new = self._orthogonalize_and_normalize(w4, [v1_new, v2_new, v3_new])
            v5_new = self._orthogonalize_and_normalize(w5, [v1_new, v2_new, v3_new, v4_new])
            v6_new = self._orthogonalize_and_normalize(w6, [v1_new, v2_new, v3_new, v4_new, v5_new])
            V0_resid = np.column_stack([v1_new, v2_new, v3_new, v4_new, v5_new, v6_new])
            v1_v2_scale = 0.50
            k_expected = 6
        else:
            raise ValueError(f"Unknown prior_variant: {self.prior_variant}")

        return V0_resid, k_expected, c0_source, v1_v2_scale, v1_new, v2_new

    def _compute_c_full_resid(
        self,
        baseline_returns_raw: np.ndarray,
        baseline_returns_p4: np.ndarray,
        c0_source: str,
    ) -> np.ndarray:
        """Compute C_full_resid from the chosen baseline source."""
        if c0_source == "raw_existing":
            _, _, C_full_resid = compute_correlation(baseline_returns_raw, self.ewma_half_life)
        else:
            _, _, C_full_resid = compute_correlation(baseline_returns_p4, self.ewma_half_life)
        return C_full_resid

    def _build_c0_resid(
        self,
        V0_resid: np.ndarray,
        C_full_resid: np.ndarray,
        v1_v2_scale: float | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build C0_resid from V0_resid and C_full_resid."""
        mat = V0_resid.T @ C_full_resid @ V0_resid
        d_vals = np.diag(mat)
        d_vals = np.maximum(d_vals, 1e-10)

        d_vals_orig = d_vals.copy()

        if v1_v2_scale is not None:
            d_vals[0] *= v1_v2_scale
            d_vals[1] *= v1_v2_scale

        d0 = np.diag(d_vals)
        c0_raw = V0_resid @ d0 @ V0_resid.T

        delta = np.diag(c0_raw)
        delta = np.maximum(delta, 1e-10)
        delta_inv_sqrt = np.diag(1.0 / np.sqrt(delta))
        C0_resid = delta_inv_sqrt @ c0_raw @ delta_inv_sqrt
        np.fill_diagonal(C0_resid, 1.0)

        C0_resid = np.nan_to_num(C0_resid, nan=0.0, posinf=1.0, neginf=-1.0)
        np.fill_diagonal(C0_resid, 1.0)
        return C0_resid, d_vals_orig, d_vals

    def _build_residual_summary_dict(
        self,
        V0_resid: np.ndarray,
        C0_resid: np.ndarray,
        C_full_resid: np.ndarray,
        d_vals_orig: np.ndarray,
        d_vals: np.ndarray,
        c0_source: str,
        k_expected: int,
        v1_new: np.ndarray,
        v2_new: np.ndarray,
    ) -> dict:
        """Assemble diagnostics and the final residual prior result dictionary."""
        eigs = np.linalg.eigvalsh(C0_resid)
        cond_num = float(np.max(eigs) / np.maximum(np.min(eigs), 1e-12))
        return {
            "V0_resid": V0_resid,
            "C0_resid": C0_resid,
            "C_full_resid": C_full_resid,
            "d_vals_orig": d_vals_orig,
            "d_vals_scaled": d_vals,
            "min_eig": float(np.min(eigs)),
            "max_eig": float(np.max(eigs)),
            "cond_num": cond_num,
            "c0_source": c0_source,
            "k_expected": k_expected,
            "v1_new": v1_new,
            "v2_new": v2_new,
        }

    def _prepare_residual_prior(self, df_exec: pd.DataFrame, inputs: dict) -> dict:
        """Precompute the V0_resid and C0_resid matrices for the baseline period."""
        baseline_indices = self._select_residual_window(df_exec)

        all_returns_p4 = inputs["all_returns_p4"]
        all_returns_raw = inputs["all_returns_raw"]

        baseline_returns_p4 = all_returns_p4[baseline_indices]
        baseline_returns_raw = all_returns_raw[baseline_indices]

        cache_key = (
            self.prior_variant,
            self.ewma_half_life,
            self.n_u,
            self.n_j,
            baseline_returns_p4.shape,
            hash(baseline_returns_p4.tobytes()),
            baseline_returns_raw.shape,
            hash(baseline_returns_raw.tobytes()),
        )
        if cache_key in self._residual_prior_cache:
            cached = self._residual_prior_cache[cache_key]
            return {
                k: v.copy() if isinstance(v, np.ndarray) else v
                for k, v in cached.items()
            }

        V0_resid, k_expected, c0_source, v1_v2_scale, v1_new, v2_new = (
            self._build_residual_prior_vectors()
        )
        C_full_resid = self._compute_c_full_resid(
            baseline_returns_raw, baseline_returns_p4, c0_source
        )
        C0_resid, d_vals_orig, d_vals = self._build_c0_resid(
            V0_resid, C_full_resid, v1_v2_scale
        )
        result_dict = self._build_residual_summary_dict(
            V0_resid,
            C0_resid,
            C_full_resid,
            d_vals_orig,
            d_vals,
            c0_source,
            k_expected,
            v1_new,
            v2_new,
        )
        self._residual_prior_cache[cache_key] = result_dict
        return {
            k: v.copy() if isinstance(v, np.ndarray) else v
            for k, v in result_dict.items()
        }
    def _get_pca_component(self, k_override: int | None = None, use_c0_override: bool = False):
        """Lazily create and cache a PCAComponent for PCA signal computation."""
        cache_attr = f"_pca_component_{k_override}_{use_c0_override}"
        if not hasattr(self, cache_attr):
            from leadlag.core.pipeline import PCAComponent
            setattr(self, cache_attr, PCAComponent(
                name="pca",
                n_u=self.n_u,
                n_j=self.n_j,
                corr_window=self.corr_window,
                k=self.k,
                lambda_reg=self.lambda_reg,
                lambda_lw=self.lambda_lw,
                lw_target=self.lw_target,
                ewma_half_life=self.ewma_half_life,
                gap_open_coef=self.gap_open_coef,
                topix_beta_coef=self.topix_beta_coef,
                vol_adjusted_target=self.vol_adjusted_target,
                min_raw_weight=getattr(self, "min_raw_weight", 0.0),
                k_override=k_override,
                use_c0_override=use_c0_override,
            ))
        return getattr(self, cache_attr)

    def compute_production_signal(
        self,
        df_exec: pd.DataFrame,
        i: int,
        c_full: np.ndarray,
        v0_static: np.ndarray,
        v1: np.ndarray,
        v2: np.ndarray,
        all_returns: np.ndarray,
        jp_gap: np.ndarray,
        jp_beta: np.ndarray | None,
        topix_night: np.ndarray | None,
    ) -> np.ndarray:
        """Compute the Raw-PCA (Production) signal at index i."""
        cache_key = (
            i,
            self.lambda_reg,
            self.k,
            self.corr_window,
            self.ewma_half_life,
            self.lambda_lw,
            self.lw_target,
            self.gap_open_coef,
            self.topix_beta_coef,
            self.vol_adjusted_target,
        )
        if cache_key in self._production_signal_cache:
            return self._production_signal_cache[cache_key].copy()
        comp = self._get_pca_component()
        result = comp.compute_standalone(
            i=i,
            all_returns=all_returns,
            c_full=c_full,
            v0_static=v0_static,
            v1=v1,
            v2=v2,
            jp_gap=jp_gap,
            jp_beta=jp_beta,
            topix_night=topix_night,
        )
        sig = result.signal
        self._production_signal_cache[cache_key] = sig
        return sig

    def compute_residual_signal(
        self,
        df_exec: pd.DataFrame,
        jp_res_returns_p3: np.ndarray,
        i: int,
        c_full: np.ndarray,
        v0_static: np.ndarray,
        v1: np.ndarray,
        v2: np.ndarray,
        jp_gap: np.ndarray,
        jp_beta: np.ndarray | None,
        topix_night: np.ndarray | None,
        is_p4: bool = False,
        c0_override: np.ndarray | None = None,
        k_override: int | None = None,
    ) -> np.ndarray:
        """Compute the Residual-PCA (JP Residual target) or P4 signal at index i."""
        k_eff = k_override if k_override is not None else self.k
        _signal_params = (
            self.lambda_reg,
            k_eff,
            self.corr_window,
            self.ewma_half_life,
            self.lambda_lw,
            self.lw_target,
            self.gap_open_coef,
            self.topix_beta_coef,
            self.vol_adjusted_target,
        )
        if not is_p4:
            cache_key = ("Residual-PCA", i, _signal_params)
        else:
            cache_key = (
                "P4",
                i,
                getattr(self, "prior_variant", None),
                getattr(self, "us_res_gamma", None),
                _signal_params,
            )
        if cache_key in self._residual_signal_cache:
            return self._residual_signal_cache[cache_key].copy()
        comp = self._get_pca_component(k_override=k_override, use_c0_override=c0_override is not None)
        result = comp.compute_standalone(
            i=i,
            all_returns=jp_res_returns_p3,
            c_full=c_full,
            v0_static=v0_static,
            v1=v1,
            v2=v2,
            jp_gap=jp_gap,
            jp_beta=jp_beta,
            topix_night=topix_night,
            c0_override=c0_override,
        )
        sig = result.signal
        self._residual_signal_cache[cache_key] = sig
        return sig

    def combine_signals(self, z0: np.ndarray, z3: np.ndarray) -> np.ndarray:
        """Combine Raw-PCA and Residual-PCA signals (signal-level 50/50 ensemble)."""
        return 0.5 * z0 + 0.5 * z3

    def predict_signals(self, df_exec: pd.DataFrame, n_jobs: int = 1) -> dict[str, Any]:
        """Generate raw signals for all rows in df_exec from corr_window index onwards.

        Args:
            df_exec: Execution DataFrame containing historical data.
            n_jobs: Number of parallel workers for signal computation. 1 = sequential.

        Returns:
            Dict of DataFrames/arrays.
        """
        from leadlag.core.pipeline import (
            PCAComponent,
            SignalPipeline,
            SRECombiner,
            SREOutputAdapter,
            _SREP4Component,
            _SRERawPCAComponent,
            _SREResidualPCAComponent,
        )

        T = len(df_exec)

        inputs_dict = self._prepare_common_inputs(df_exec)

        # Reconstruct CommonInputs from dict for pipeline
        from leadlag.core.pipeline import CommonInputs, P4Inputs
        inputs = CommonInputs(
            all_returns_raw=inputs_dict["all_returns_raw"],
            c_full=inputs_dict["c_full"],
            c_full_p3=inputs_dict["c_full_p3"],
            v0_static=inputs_dict["v0_static"],
            v1=inputs_dict["v1"],
            v2=inputs_dict["v2"],
            jp_gap=inputs_dict["jp_gap"],
            jp_beta=inputs_dict["jp_beta"],
            topix_night=inputs_dict["topix_night"],
            y_jp_oc_df=inputs_dict["y_jp_oc_df"],
            jp_res_returns_p3=inputs_dict["jp_res_returns_p3"],
            y_jp_target=inputs_dict["y_jp_target"],
            n_u=self.n_u,
            n_j=self.n_j,
            dates=df_exec.index,
            p4=P4Inputs(
                all_returns_p4=inputs_dict["all_returns_p4"],
                r_us_adj=inputs_dict["r_us_adj"],
                spy_returns=inputs_dict["spy_returns"],
            ) if "all_returns_p4" in inputs_dict else None,
        )

        # Optional P4 prior
        prior_info = None
        if self.us_res_enabled or self.p4_weight > 0.0:
            if self.prior_variant is not None:
                prior_info = self._prepare_residual_prior(df_exec, inputs_dict)

        # Clear per-run signal caches to prevent cross-run contamination
        self._production_signal_cache.clear()
        self._residual_signal_cache.clear()
        self._common_inputs_cache.clear()

        # Build PCA components
        raw_pca_comp = PCAComponent(
            name="raw_pca",
            n_u=self.n_u, n_j=self.n_j,
            corr_window=self.corr_window, k=self.k,
            lambda_reg=self.lambda_reg, lambda_lw=self.lambda_lw,
            lw_target=self.lw_target, ewma_half_life=self.ewma_half_life,
            gap_open_coef=self.gap_open_coef, topix_beta_coef=self.topix_beta_coef,
            vol_adjusted_target=self.vol_adjusted_target,
            min_raw_weight=getattr(self, "min_raw_weight", 0.0),
        )
        residual_pca_comp = PCAComponent(
            name="residual_pca",
            n_u=self.n_u, n_j=self.n_j,
            corr_window=self.corr_window, k=self.k,
            lambda_reg=self.lambda_reg, lambda_lw=self.lambda_lw,
            lw_target=self.lw_target, ewma_half_life=self.ewma_half_life,
            gap_open_coef=self.gap_open_coef, topix_beta_coef=self.topix_beta_coef,
            vol_adjusted_target=self.vol_adjusted_target,
            min_raw_weight=getattr(self, "min_raw_weight", 0.0),
        )

        components = [
            _SRERawPCAComponent(raw_pca_comp),
            _SREResidualPCAComponent(residual_pca_comp),
        ]

        # Optional P4 component
        if self.us_res_enabled or self.p4_weight > 0.0:
            if self.prior_variant is not None:
                C0_resid = prior_info["C0_resid"]
                V0_resid = prior_info["V0_resid"]
                k_p4 = prior_info["k_expected"]
                p4_pca = PCAComponent(
                    name="p4",
                    n_u=self.n_u, n_j=self.n_j,
                    corr_window=self.corr_window, k=self.k,
                    lambda_reg=self.lambda_reg, lambda_lw=self.lambda_lw,
                    lw_target=self.lw_target, ewma_half_life=self.ewma_half_life,
                    gap_open_coef=self.gap_open_coef, topix_beta_coef=self.topix_beta_coef,
                    vol_adjusted_target=self.vol_adjusted_target,
                    min_raw_weight=getattr(self, "min_raw_weight", 0.0),
                    k_override=k_p4,
                    use_c0_override=True,
                )
                p4_comp = _SREP4Component(
                    p4_pca,
                    c_full=C0_resid,
                    v0_static=V0_resid,
                    v1=V0_resid[:, 0],
                    v2=V0_resid[:, 1] if V0_resid.shape[1] > 1 else V0_resid[:, 0],
                    all_returns_p4=inputs.p4.all_returns_p4,
                    jp_gap=inputs.jp_gap,
                    jp_beta=inputs.jp_beta,
                    topix_night=inputs.topix_night,
                )
            else:
                p4_pca = PCAComponent(
                    name="p4",
                    n_u=self.n_u, n_j=self.n_j,
                    corr_window=self.corr_window, k=self.k,
                    lambda_reg=self.lambda_reg, lambda_lw=self.lambda_lw,
                    lw_target=self.lw_target, ewma_half_life=self.ewma_half_life,
                    gap_open_coef=self.gap_open_coef, topix_beta_coef=self.topix_beta_coef,
                    vol_adjusted_target=self.vol_adjusted_target,
                    min_raw_weight=getattr(self, "min_raw_weight", 0.0),
                )
                p4_comp = _SREP4Component(
                    p4_pca,
                    c_full=inputs.c_full,
                    v0_static=inputs.v0_static,
                    v1=inputs.v1,
                    v2=inputs.v2,
                    all_returns_p4=inputs.p4.all_returns_p4,
                    jp_gap=inputs.jp_gap,
                    jp_beta=inputs.jp_beta,
                    topix_night=inputs.topix_night,
                )
            components.append(p4_comp)

        combiner = SRECombiner(
            raw_pca_weight=self.raw_pca_weight,
            residual_pca_weight=self.residual_pca_weight,
            p4_weight=self.p4_weight,
            normalization_method=self.normalization_method,
            n_j=self.n_j,
            normalize_fn=self.normalize_signals,
        )

        pipeline = SignalPipeline(components=components, combiner=combiner)
        pipeline_results = pipeline.run(inputs, start_idx=self.corr_window, T=T, n_jobs=n_jobs)

        adapter = SREOutputAdapter(n_j=self.n_j, jp_tickers=JP_TICKERS)
        return adapter.adapt(pipeline_results, inputs, prior_info=prior_info)
