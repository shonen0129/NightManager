"""Legacy V1 pipeline components retained for archive research scripts.

These classes have been removed from ``leadlag.core.pipeline`` and are kept
here so archived research notebooks can continue to import them.  They are not
used by the production V2 path.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from leadlag.core.pipeline import (
    CommonInputs,
    ComponentResult,
    PCAComponent,
    RunContext,
    StepContext,
)

logger = logging.getLogger(__name__)


class _SRERawPCAComponent:
    """Raw-PCA component for SRE pipeline."""

    name = "raw_pca"

    def __init__(self, pca: PCAComponent):
        self._pca = pca

    def begin_run(self, context: RunContext) -> None:
        pass

    def end_run(self) -> dict[str, Any]:
        return {}

    def compute(self, context: StepContext) -> ComponentResult:
        inputs = context.inputs
        return self._pca.compute_standalone(
            i=context.i,
            all_returns=inputs.all_returns_raw,
            c_full=inputs.c_full,
            v0_static=inputs.v0_static,
            v1=inputs.v1,
            v2=inputs.v2,
            jp_gap=inputs.jp_gap,
            jp_beta=inputs.jp_beta,
            topix_night=inputs.topix_night,
        )


class _SREResidualPCAComponent:
    """Residual-PCA component for SRE pipeline."""

    name = "residual_pca"

    def __init__(self, pca: PCAComponent):
        self._pca = pca

    def begin_run(self, context: RunContext) -> None:
        pass

    def end_run(self) -> dict[str, Any]:
        return {}

    def compute(self, context: StepContext) -> ComponentResult:
        inputs = context.inputs
        return self._pca.compute_standalone(
            i=context.i,
            all_returns=inputs.jp_res_returns_p3,
            c_full=inputs.c_full_p3,
            v0_static=inputs.v0_static,
            v1=inputs.v1,
            v2=inputs.v2,
            jp_gap=inputs.jp_gap,
            jp_beta=inputs.jp_beta,
            topix_night=inputs.topix_night,
        )


class _SREP4Component:
    """P4 (US-residualized) component for SRE pipeline."""

    name = "p4"

    def __init__(
        self,
        pca: PCAComponent,
        c_full: np.ndarray,
        v0_static: np.ndarray,
        v1: np.ndarray,
        v2: np.ndarray,
        all_returns_p4: np.ndarray,
        jp_gap: np.ndarray,
        jp_beta: np.ndarray | None,
        topix_night: np.ndarray | None,
    ):
        self._pca = pca
        self._c_full = c_full
        self._v0_static = v0_static
        self._v1 = v1
        self._v2 = v2
        self._all_returns_p4 = all_returns_p4
        self._jp_gap = jp_gap
        self._jp_beta = jp_beta
        self._topix_night = topix_night

    def begin_run(self, context: RunContext) -> None:
        pass

    def end_run(self) -> dict[str, Any]:
        return {}

    def compute(self, context: StepContext) -> ComponentResult:
        return self._pca.compute_standalone(
            i=context.i,
            all_returns=self._all_returns_p4,
            c_full=self._c_full,
            v0_static=self._v0_static,
            v1=self._v1,
            v2=self._v2,
            jp_gap=self._jp_gap,
            jp_beta=self._jp_beta,
            topix_night=self._topix_night,
        )


class SRECombiner:
    """Combines Raw-PCA, Residual-PCA, and optional P4 signals for SRE.

    Normalizes each component signal, applies ensemble weights, and produces
    both the combined signal and its normalized version.
    """

    def __init__(
        self,
        raw_pca_weight: float,
        residual_pca_weight: float,
        p4_weight: float,
        normalization_method: str,
        n_j: int,
        normalize_fn: Any,
    ):
        self._raw_pca_weight = raw_pca_weight
        self._residual_pca_weight = residual_pca_weight
        self._p4_weight = p4_weight
        self._normalization_method = normalization_method
        self._n_j = n_j
        self._normalize_fn = normalize_fn

    def begin_run(self, context: RunContext) -> None:
        pass

    def end_run(self) -> dict[str, Any]:
        return {}

    def combine(
        self,
        context: StepContext,
        components: Mapping[str, ComponentResult],
    ) -> ComponentResult:
        z0 = self._normalize_fn(components["raw_pca"].signal, self._normalization_method)
        z3 = self._normalize_fn(components["residual_pca"].signal, self._normalization_method)

        if "p4" in components and self._p4_weight > 0.0:
            z4 = self._normalize_fn(components["p4"].signal, self._normalization_method)
        else:
            z4 = np.zeros(self._n_j)

        s_ens = (
            self._raw_pca_weight * z0
            + self._residual_pca_weight * z3
            + self._p4_weight * z4
        )
        s_norm = self._normalize_fn(s_ens, self._normalization_method)
        return ComponentResult(signal=s_ens, diagnostics={"normalized": s_norm})


class SREOutputAdapter:
    """Converts pipeline output arrays to SRE's dict-of-DataFrames format."""

    def __init__(self, n_j: int, jp_tickers: list[str]):
        self._n_j = n_j
        self._jp_tickers = jp_tickers

    def adapt(
        self,
        pipeline_results: dict[str, np.ndarray],
        inputs: CommonInputs,
        prior_info: dict | None = None,
    ) -> dict[str, Any]:
        sim_dates = inputs.dates
        T = len(sim_dates)
        jp_tickers = self._jp_tickers

        raw_pca_df = pd.DataFrame(
            pipeline_results["raw_pca"], index=sim_dates, columns=jp_tickers
        )
        residual_pca_df = pd.DataFrame(
            pipeline_results["residual_pca"], index=sim_dates, columns=jp_tickers
        )
        p4_df = pd.DataFrame(
            pipeline_results.get("p4", np.zeros((T, self._n_j))),
            index=sim_dates, columns=jp_tickers,
        )
        sre_df = pd.DataFrame(
            pipeline_results["combined"], index=sim_dates, columns=jp_tickers,
        )

        sre_normalized_df = pd.DataFrame(index=sim_dates, columns=jp_tickers)
        for date in sim_dates:
            idx = sim_dates.get_loc(date)
            sre_normalized_df.loc[date] = pipeline_results["normalized"][idx]

        out = {
            "raw_pca_signals": raw_pca_df,
            "residual_pca_signals": residual_pca_df,
            "p4_signals": p4_df,
            "signals": sre_df,
            "normalized_signals": sre_normalized_df,
            "y_jp_oc_df": inputs.y_jp_oc_df,
        }
        if prior_info is not None:
            out["prior_info"] = prior_info
        return out

class BLPCombiner:
    """Combines Raw-PCA, Residual-PCA, P5, P5P3 signals for BLP model."""

    def __init__(
        self,
        raw_pca_weight: float,
        residual_pca_weight: float,
        p5_weight: float,
        p5p3_weight: float,
        normalization_method: str,
        n_j: int,
        normalize_fn: Any,
    ):
        self._w0 = raw_pca_weight
        self._w3 = residual_pca_weight
        self._w5 = p5_weight
        self._w5p3 = p5p3_weight
        self._norm = normalization_method
        self._n_j = n_j
        self._normalize_fn = normalize_fn

    def begin_run(self, context: RunContext) -> None:
        pass

    def end_run(self) -> dict[str, Any]:
        return {}

    def combine(
        self,
        context: StepContext,
        components: Mapping[str, ComponentResult],
    ) -> ComponentResult:
        z0 = self._normalize_fn(components["raw_pca"].signal, self._norm)
        z3 = self._normalize_fn(components["residual_pca"].signal, self._norm)
        z5 = self._normalize_fn(components["p5"].signal, self._norm)
        z5p3 = self._normalize_fn(components["p5p3"].signal, self._norm)

        s_ens = self._w0 * z0 + self._w3 * z3 + self._w5 * z5 + self._w5p3 * z5p3
        s_norm = self._normalize_fn(s_ens, self._norm)

        date_str = context.run.dates[context.i].strftime("%Y-%m-%d")
        p5_diag = components["p5"].diagnostics
        p5p3_diag = components["p5p3"].diagnostics

        step_diag = {
            "date": date_str,
            "p5_cond_num": p5_diag.get("cond_num"),
            "p5_b_norm": p5_diag.get("b_norm"),
            "p5_sigma_xx_trace": p5_diag.get("sigma_xx_trace"),
            "p5_sigma_yx_norm": p5_diag.get("sigma_yx_norm"),
            "p5_pinv_fallback": int(p5_diag.get("pinv_fallback", 0)),
            "p5_num_training_samples": p5_diag.get("num_training_samples"),
            "p5p3_cond_num": p5p3_diag.get("cond_num"),
            "p5p3_b_norm": p5p3_diag.get("b_norm"),
            "p5p3_sigma_xx_trace": p5p3_diag.get("sigma_xx_trace"),
            "p5p3_sigma_yx_norm": p5p3_diag.get("sigma_yx_norm"),
            "p5p3_pinv_fallback": int(p5p3_diag.get("pinv_fallback", 0)),
            "p5p3_num_training_samples": p5p3_diag.get("num_training_samples"),
        }

        return ComponentResult(signal=s_ens, diagnostics={"normalized": s_norm, **step_diag})


class BLPOutputAdapter:
    """Converts pipeline output arrays to BLP model's dict-of-DataFrames format."""

    def __init__(self, n_j: int, jp_tickers: list[str]):
        self._n_j = n_j
        self._jp_tickers = jp_tickers

    def adapt(
        self,
        pipeline_results: dict[str, np.ndarray],
        inputs: CommonInputs,
    ) -> dict[str, Any]:
        sim_dates = inputs.dates
        T = len(sim_dates)
        jp = self._jp_tickers

        raw_pca_df = pd.DataFrame(pipeline_results["raw_pca"], index=sim_dates, columns=jp)
        residual_pca_df = pd.DataFrame(pipeline_results["residual_pca"], index=sim_dates, columns=jp)
        p4_df = pd.DataFrame(np.zeros((T, self._n_j)), index=sim_dates, columns=jp)
        p5_df = pd.DataFrame(pipeline_results["p5"], index=sim_dates, columns=jp)
        p5p3_df = pd.DataFrame(pipeline_results["p5p3"], index=sim_dates, columns=jp)
        combined_df = pd.DataFrame(pipeline_results["combined"], index=sim_dates, columns=jp)
        normalized_df = pd.DataFrame(pipeline_results["normalized"], index=sim_dates, columns=jp)

        blp_diag_df = (
            pd.DataFrame(pipeline_results["_step_diagnostics"]).set_index("date")
            if "_step_diagnostics" in pipeline_results
            else pd.DataFrame()
        )

        return {
            "raw_pca_signals": raw_pca_df,
            "residual_pca_signals": residual_pca_df,
            "p4_signals": p4_df,
            "p5_signals": p5_df,
            "p5p3_signals": p5p3_df,
            "signals": combined_df,
            "normalized_signals": normalized_df,
            "y_jp_oc_df": inputs.y_jp_oc_df,
            "blp_diagnostics": blp_diag_df,
        }


class RRRCombiner:
    """Combines Raw-PCA, Residual-PCA, P6, P6P3, P7, P7P3 signals for RRR model."""

    def __init__(
        self,
        raw_pca_weight: float,
        residual_pca_weight: float,
        p6_weight: float,
        p6p3_weight: float,
        p7_weight: float,
        p7p3_weight: float,
        normalization_method: str,
        n_j: int,
        normalize_fn: Any,
        variant: str,
        rank,
        lambda_ridge: float,
        lambda_prior: float,
    ):
        self._w0 = raw_pca_weight
        self._w3 = residual_pca_weight
        self._w6 = p6_weight
        self._w6p3 = p6p3_weight
        self._w7 = p7_weight
        self._w7p3 = p7p3_weight
        self._norm = normalization_method
        self._n_j = n_j
        self._normalize_fn = normalize_fn
        self._variant = variant
        self._rank = rank
        self._lambda_ridge = lambda_ridge
        self._lambda_prior = lambda_prior

    def begin_run(self, context: RunContext) -> None:
        pass

    def end_run(self) -> dict[str, Any]:
        return {}

    def combine(
        self,
        context: StepContext,
        components: Mapping[str, ComponentResult],
    ) -> ComponentResult:
        z0 = self._normalize_fn(components["raw_pca"].signal, self._norm)
        z3 = self._normalize_fn(components["residual_pca"].signal, self._norm)
        z6 = self._normalize_fn(components["p6"].signal, self._norm)
        z6p3 = self._normalize_fn(components["p6p3"].signal, self._norm)
        z7 = self._normalize_fn(components["p7"].signal, self._norm)
        z7p3 = self._normalize_fn(components["p7p3"].signal, self._norm)

        s_ens = (
            self._w0 * z0 + self._w3 * z3
            + self._w6 * z6 + self._w6p3 * z6p3
            + self._w7 * z7 + self._w7p3 * z7p3
        )
        s_norm = self._normalize_fn(s_ens, self._norm)

        date_str = context.run.dates[context.i].strftime("%Y-%m-%d")
        p6_diag = components["p6"].diagnostics
        s_vals = p6_diag.get("singular_values", np.zeros(0))
        s_top = float(s_vals[0]) if len(s_vals) > 0 else 0.0

        step_diag = {
            "date": date_str,
            "variant": self._variant,
            "rank": self._rank,
            "effective_rank": p6_diag.get("effective_rank"),
            "singular_values_top": s_top,
            "condition_number": p6_diag.get("cond_num"),
            "b_norm": p6_diag.get("b_norm"),
            "prior_norm": p6_diag.get("prior_norm"),
            "b_minus_prior_norm": p6_diag.get("b_minus_prior_norm"),
            "lambda_ridge": self._lambda_ridge,
            "lambda_prior": self._lambda_prior,
            "num_training_samples": p6_diag.get("num_training_samples"),
            "pinv_fallback": int(p6_diag.get("pinv_fallback", 0)),
        }

        return ComponentResult(signal=s_ens, diagnostics={"normalized": s_norm, **step_diag})


class RRROutputAdapter:
    """Converts pipeline output arrays to RRR model's dict-of-DataFrames format."""

    def __init__(self, n_j: int, jp_tickers: list[str]):
        self._n_j = n_j
        self._jp_tickers = jp_tickers

    def adapt(
        self,
        pipeline_results: dict[str, np.ndarray],
        inputs: CommonInputs,
    ) -> dict[str, Any]:
        sim_dates = inputs.dates
        T = len(sim_dates)
        jp = self._jp_tickers

        raw_pca_df = pd.DataFrame(pipeline_results["raw_pca"], index=sim_dates, columns=jp)
        residual_pca_df = pd.DataFrame(pipeline_results["residual_pca"], index=sim_dates, columns=jp)
        p4_df = pd.DataFrame(np.zeros((T, self._n_j)), index=sim_dates, columns=jp)
        p6_df = pd.DataFrame(pipeline_results["p6"], index=sim_dates, columns=jp)
        p6p3_df = pd.DataFrame(pipeline_results["p6p3"], index=sim_dates, columns=jp)
        p7_df = pd.DataFrame(pipeline_results["p7"], index=sim_dates, columns=jp)
        p7p3_df = pd.DataFrame(pipeline_results["p7p3"], index=sim_dates, columns=jp)
        combined_df = pd.DataFrame(pipeline_results["combined"], index=sim_dates, columns=jp)
        normalized_df = pd.DataFrame(pipeline_results["normalized"], index=sim_dates, columns=jp)

        rrr_diag_df = (
            pd.DataFrame(pipeline_results["_step_diagnostics"]).set_index("date")
            if "_step_diagnostics" in pipeline_results
            else pd.DataFrame()
        )

        return {
            "raw_pca_signals": raw_pca_df,
            "residual_pca_signals": residual_pca_df,
            "p4_signals": p4_df,
            "p6_signals": p6_df,
            "p6p3_signals": p6p3_df,
            "p7_signals": p7_df,
            "p7p3_signals": p7p3_df,
            "signals": combined_df,
            "normalized_signals": normalized_df,
            "y_jp_oc_df": inputs.y_jp_oc_df,
            "rrr_diagnostics": rrr_diag_df,
        }



class BayesianCombiner:
    """Simple passthrough combiner for Bayesian BLPX model.

    The Bayesian model only uses residual_blpx signals with stateful
    Bayesian updates. The combiner just normalizes the signal.
    """

    def __init__(
        self,
        normalization_method: str,
        n_j: int,
        normalize_fn: Any,
    ):
        self._norm = normalization_method
        self._n_j = n_j
        self._normalize_fn = normalize_fn

    def begin_run(self, context: RunContext) -> None:
        pass

    def end_run(self) -> dict[str, Any]:
        return {}

    def combine(
        self,
        context: StepContext,
        components: Mapping[str, ComponentResult],
    ) -> ComponentResult:
        sig = components["residual_blpx_bayesian"].signal
        z = self._normalize_fn(sig, self._norm)

        date_str = context.run.dates[context.i].strftime("%Y-%m-%d")
        bayes_diag = components["residual_blpx_bayesian"].diagnostics

        step_diag = {
            "date": date_str,
            "eta": bayes_diag.get("eta_t", 0.0),
            "ic": bayes_diag.get("ic", 0.0),
            "rolling_ic": bayes_diag.get("rolling_ic", 0.0),
            "cs_var": bayes_diag.get("cs_var", 0.0),
            "mode": bayes_diag.get("mode", "ic"),
        }

        return ComponentResult(signal=z, diagnostics={"normalized": z, **step_diag})


class BayesianOutputAdapter:
    """Converts pipeline output arrays to Bayesian BLPX model's output format."""

    def __init__(self, n_j: int, jp_tickers: list[str]):
        self._n_j = n_j
        self._jp_tickers = jp_tickers

    def adapt(
        self,
        pipeline_results: dict[str, np.ndarray],
        inputs: CommonInputs,
        sigma_yy: np.ndarray | None = None,
    ) -> dict[str, Any]:
        sim_dates = inputs.dates
        jp = self._jp_tickers

        residual_blpx_df = pd.DataFrame(
            pipeline_results["residual_blpx_bayesian"], index=sim_dates, columns=jp,
        )
        combined_df = pd.DataFrame(pipeline_results["combined"], index=sim_dates, columns=jp)
        normalized_df = pd.DataFrame(pipeline_results["normalized"], index=sim_dates, columns=jp)

        eta_df = (
            pd.DataFrame(pipeline_results["_step_diagnostics"]).set_index("date")
            if "_step_diagnostics" in pipeline_results
            else pd.DataFrame()
        )

        out = {
            "signals": combined_df,
            "normalized_signals": normalized_df,
            "residual_blpx_signals": residual_blpx_df,
            "raw_pca_signals": residual_blpx_df,
            "residual_pca_signals": residual_blpx_df,
            "p4_signals": residual_blpx_df,
            "raw_blpx_signals": residual_blpx_df,
            "y_jp_oc_df": inputs.y_jp_oc_df,
            "blp_diagnostics": eta_df,
            "bayesian_diagnostics": eta_df,
        }
        if sigma_yy is not None:
            out["sigma_yy"] = sigma_yy
        return out
