"""runner/helpers.py — shared utility functions for all runner modules.

Contains pure utility functions that are used by decision.py, fast.py,
close.py, and backtest.py to avoid code duplication.

This module now re-exports the decomposed ``execution.*`` submodules
(pricing, broker_ops, risk_capital, output_ops, post_decision) for
backward compatibility.
"""

from __future__ import annotations

__all__ = [
    "build_api_client",
    "build_output_dir",
    "fetch_current_positions",
    "resolve_wallet_capital",
    "build_risk_config",
    "run_risk_checks",
    "auto_adjust_gross_exposure",
    "allocate_capital",
    "resolve_daily_open_prices",
    "fetch_fill_prices",
    "split_large_orders",
    "submit_orders_via_api",
    "save_decision_output",
    "save_summary_files",
    "execute_post_decision_flow",
    "save_position_snapshot",
    "save_wallet_snapshot",
    "save_daily_journal",
    "get_hist_returns_for_risk",
    "log_decision_summary",
    "print_risk_report",
    "print_text_orders",
]

import logging

from leadlag.data.cache import get_hist_returns_for_risk as _get_hist_returns_for_risk
from leadlag.execution.broker_ops import (
    build_api_client,
    fetch_current_positions,
    resolve_wallet_capital,
    split_large_orders,
    submit_orders_via_api,
)
from leadlag.execution.output_ops import (
    build_output_dir,
    save_daily_journal,
    save_decision_output,
    save_position_snapshot,
    save_summary_files,
    save_wallet_snapshot,
)
from leadlag.execution.post_decision import execute_post_decision_flow
from leadlag.execution.pricing import fetch_fill_prices, resolve_daily_open_prices
from leadlag.execution.risk_capital import (
    allocate_capital,
    auto_adjust_gross_exposure,
    build_risk_config,
    run_risk_checks,
)
from leadlag.reporting.formatter import (
    log_decision_summary as _log_decision_summary,
)
from leadlag.reporting.formatter import (
    print_risk_report as _print_risk_report,
)
from leadlag.reporting.formatter import (
    print_text_orders as _print_text_orders,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------

get_hist_returns_for_risk = _get_hist_returns_for_risk
log_decision_summary = _log_decision_summary
print_risk_report = _print_risk_report
print_text_orders = _print_text_orders
