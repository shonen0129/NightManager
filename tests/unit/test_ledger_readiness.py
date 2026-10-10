"""Offline readiness tests; no broker connection or trading action."""
from __future__ import annotations

from copy import deepcopy

import pytest

from leadlag.execution.ledger_readiness import REQUIRED_SOURCES, assess_ledger_readiness


def _manifest():
    return {
        "schema_version": "account-ledger-evidence-v1",
        "account_key": "tachibana:default",
        "session_date": "2026-10-09",
        "observed_at": "2026-10-09T15:30:00+09:00",
        "sources": {
            name: {
                "authority": "broker_official",
                "source_id": f"official:{name}:2026-10-09",
                "sha256": "a" * 64,
                "reconciled_by": "account-ledger-reviewer",
                "verified": True,
            }
            for name in REQUIRED_SOURCES
        },
        "execution_crosswalk": [
            {"broker_execution_id": "fill-1", "local_fill_id": "local-1", "verified": True},
            {"broker_execution_id": "fill-2", "local_fill_id": "local-2", "verified": True},
        ],
        "all_broker_fills_accounted_for": True,
        "all_local_fills_accounted_for": True,
        "cash_position_fee_balances_verified": True,
        "pending_fills": 0,
        "external_cash_flows_reconciled": True,
        "session_mark_policy_verified": True,
    }


def test_complete_fixture_has_no_readiness_blockers():
    assert assess_ledger_readiness(_manifest()) == []


@pytest.mark.parametrize(
    "mutation,expected",
    [
        (lambda m: m["sources"]["fees"].update(verified=False), "fees: not verified"),
        (lambda m: m["sources"].pop("financing"), "missing source: financing"),
        (lambda m: m["sources"]["cash"].update(sha256="bad"), "cash: invalid sha256"),
        (lambda m: m["sources"]["positions"].update(reconciled_by=""), "missing reconciliation owner"),
        (lambda m: m["execution_crosswalk"][1].update(broker_execution_id="fill-1"), "duplicate broker execution ID"),
        (lambda m: m["execution_crosswalk"][1].update(local_fill_id="local-1"), "duplicate local fill ID"),
        (lambda m: m.update(pending_fills=1), "pending fills not proven zero"),
        (lambda m: m.update(pending_fills=True), "pending fills not proven zero"),
        (lambda m: m.update(all_broker_fills_accounted_for=False), "broker fill coverage"),
        (lambda m: m.update(external_cash_flows_reconciled=False), "external cash flows"),
        (lambda m: m.update(session_mark_policy_verified=False), "session end marks"),
        (lambda m: m.update(cash_position_fee_balances_verified=False), "balances not verified"),
    ],
)
def test_incomplete_or_ambiguous_evidence_remains_blocked(mutation, expected):
    manifest = deepcopy(_manifest())
    mutation(manifest)
    assert expected in " | ".join(assess_ledger_readiness(manifest))


def test_observed_at_before_session_date_is_blocked():
    manifest = _manifest()
    manifest["observed_at"] = "2026-10-08T23:59:59+09:00"
    assert "observed_at cannot precede session_date" in assess_ledger_readiness(manifest)


def test_no_source_manifest_is_never_ready():
    assert assess_ledger_readiness({})


def test_readiness_is_not_a_pnl_or_trading_authorization():
    # Readiness describes evidence only, never generates account-risk-snapshot-v1.
    manifest = _manifest()
    assert assess_ledger_readiness(manifest) == []
    assert "daily_return" not in manifest
    assert "month_return" not in manifest
    assert "reconciliation_status" not in manifest
