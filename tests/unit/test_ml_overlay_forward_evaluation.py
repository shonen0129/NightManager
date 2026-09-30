from __future__ import annotations

import pytest

from leadlag.execution.account_risk import REQUIRED_PNL_BASIS
from leadlag.reporting.ml_overlay_forward_evaluation import (
    OUTCOME_SCHEMA,
    evaluate_ml_overlay_forward,
)


def _shadow(*, digest="input-a", trade_date="2026-09-28"):
    return {
        "trade_date": trade_date,
        "status": "complete",
        "record_fingerprint": "shadow-a",
        "quote_snapshot_id": "quote-a",
        "input_version": {"digest": digest},
        "sizing": {"capital_jpy": 100_000.0},
        "variants": {
            "ml_enabled": {
                "weights": {"1617.T": 0.5, "1618.T": -0.5},
                "modeled_costs_return_fraction": {"total": 0.01},
            },
            "ml_disabled": {
                "weights": {"1617.T": 0.3, "1618.T": -0.3},
                "modeled_costs_return_fraction": {"total": 0.005},
            },
        },
    }


def _outcome(*, digest="input-a", trade_date="2026-09-28", target_source="local_official_close_cache"):
    return {
        "schema_version": OUTCOME_SCHEMA,
        "trade_date": trade_date,
        "input_version": {"digest": digest},
        "quote_snapshot_id": "quote-a",
        "label_status": "complete",
        "target_basis": "frozen_0910_quote_mid_to_official_close",
        "target_source": target_source,
        "target_returns": {"1617.T": 0.1, "1618.T": 0.0},
        "execution_reconciliation": {
            "status": "complete",
            "fill_reconciled": True,
            "positions_reconciled": True,
            "cash_reconciled": True,
            "fees_complete": True,
            "pnl_basis": REQUIRED_PNL_BASIS,
            "input_version_digest": digest,
            "candidate_actual_net_pnl_jpy": 2_000.0,
        },
    }


def test_forward_evaluation_pairs_same_snapshot_and_separates_actual_fills():
    result = evaluate_ml_overlay_forward(
        [_shadow(), _shadow(digest="input-b", trade_date="2026-09-29")],
        [
            _outcome(target_source="close_feed_a"),
            _outcome(
                digest="input-b",
                trade_date="2026-09-29",
                target_source="close_feed_b",
            ),
        ],
        expected_trade_dates=["2026-09-28", "2026-09-29", "2026-09-30"],
    )

    day = next(item for item in result["daily"] if item["trade_date"] == "2026-09-28")
    next_day = next(item for item in result["daily"] if item["trade_date"] == "2026-09-29")
    assert day["status"] == "paired_complete"
    assert day["target_source"] == "close_feed_a"
    assert next_day["status"] == "paired_complete"
    assert next_day["target_source"] == "close_feed_b"
    assert day["ml_enabled_modeled_net_return"] == pytest.approx(0.04)
    assert day["ml_disabled_modeled_net_return"] == pytest.approx(0.025)
    assert day["paired_modeled_net_delta"] == pytest.approx(0.015)
    assert day["candidate_actual_net_return"] == pytest.approx(0.02)
    assert day["baseline_evidence"] == "modeled_weights_same_capital; no_lot_or_fill_claim"
    assert result["paired_complete_count"] == 2
    assert result["actual_fill_pnl_complete_count"] == 2
    assert result["decision_missing_count"] == 1
    missing = next(item for item in result["daily"] if item["trade_date"] == "2026-09-30")
    assert missing["status"] == "decision_missing"


def test_forward_evaluation_keeps_cost_or_label_gaps_out_of_actual_pnl():
    outcome = _outcome()
    outcome["execution_reconciliation"]["fees_complete"] = False
    outcome["target_basis"] = "five_minute_proxy"
    result = evaluate_ml_overlay_forward([_shadow()], [outcome])

    assert result["paired_complete_count"] == 0
    assert result["actual_fill_pnl_complete_count"] == 0
    assert result["daily"][0]["status"] == "target_basis_unverified"


def test_forward_evaluation_excludes_ambiguous_multiple_shadow_versions():
    first = _shadow(digest="input-a")
    second = _shadow(digest="input-b")
    second["record_fingerprint"] = "shadow-b"
    result = evaluate_ml_overlay_forward([first, second], [_outcome()])

    assert result["paired_complete_count"] == 0
    assert result["ambiguous_shadow_date_count"] == 1
    assert result["daily"][0]["status"] == "multiple_shadow_input_versions"
