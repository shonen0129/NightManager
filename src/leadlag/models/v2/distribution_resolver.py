"""V2 gap-distribution resolver compatibility re-exports."""

from __future__ import annotations

from leadlag.models.v2.fallback import _apply_pit_ruleD, _repair_and_adjust
from leadlag.models.v2.gap_io import (
    _build_current_prices_from_df_exec,
    _compute_ondemand,
    _extract_gap_inputs,
    _gap_alerts_fatal,
    _load_gap_or_flat,
    _resolve_current_index,
    compute_distribution,
)
from leadlag.models.v2.pit import load_pit_ir_history

__all__ = [
    "_apply_pit_ruleD",
    "_build_current_prices_from_df_exec",
    "_compute_ondemand",
    "_extract_gap_inputs",
    "_gap_alerts_fatal",
    "_load_gap_or_flat",
    "_repair_and_adjust",
    "_resolve_current_index",
    "compute_distribution",
    "load_pit_ir_history",
]
