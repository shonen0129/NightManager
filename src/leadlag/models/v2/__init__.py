"""V2 production model sub-package.

Re-exports the helpers used by ``leadlag.models.production_v2``.
"""

from __future__ import annotations

VERSION = "production_residual_blpx_v2"

from leadlag.models.v2.audit_comparator import (
    _build_summary,
    _compare_distribution,
    _run_safety_audits,
)
from leadlag.models.v2.decision_engine import (
    _decide,
    _derive_signal_date,
    _file_cache_or_flat,
    generate_v2_production_portfolio_from_distribution,
)
from leadlag.models.v2.distribution_resolver import (
    _apply_pit_ruleD,
    _build_current_prices_from_df_exec,
    _compute_ondemand,
    _extract_gap_inputs,
    _gap_alerts_fatal,
    _load_gap_or_flat,
    _repair_and_adjust,
    _resolve_current_index,
    compute_distribution,
    load_pit_ir_history,
)
from leadlag.models.v2.distribution_source import (
    DistributionResult,
    DistributionSource,
    FileCacheDistributionSource,
    FlatPositionSource,
    OnDemandDistributionSource,
)
from leadlag.models.v2.fallback_policy import FallbackPolicy
from leadlag.models.v2.overlay_applier import (
    _apply_overlay,
    _apply_rank_reversal_overlay,
    _multi_horizon_scores,
)

__all__ = [
    "VERSION",
    "_apply_overlay",
    "_apply_pit_ruleD",
    "_apply_rank_reversal_overlay",
    "_build_current_prices_from_df_exec",
    "_build_summary",
    "_compare_distribution",
    "_compute_ondemand",
    "_decide",
    "_derive_signal_date",
    "_extract_gap_inputs",
    "_file_cache_or_flat",
    "_gap_alerts_fatal",
    "_load_gap_or_flat",
    "_multi_horizon_scores",
    "_repair_and_adjust",
    "_resolve_current_index",
    "_run_safety_audits",
    "compute_distribution",
    "DistributionResult",
    "DistributionSource",
    "FallbackPolicy",
    "FileCacheDistributionSource",
    "FlatPositionSource",
    "generate_v2_production_portfolio_from_distribution",
    "load_pit_ir_history",
    "OnDemandDistributionSource",
]
