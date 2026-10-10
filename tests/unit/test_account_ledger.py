from __future__ import annotations

from copy import deepcopy

import pytest

from leadlag.execution.account_ledger import (
    LedgerReconciliationError,
    build_account_risk_snapshot,
    build_reconciled_execution_ledger,
    write_account_risk_snapshot,
)
from leadlag.execution.account_risk import AccountRiskSnapshot
from leadlag.execution.ledger_readiness import REQUIRED_SOURCES


def _manifest(trade_date: str, *, crosswalk=None):
    return {
        "schema_version": "account-ledger-evidence-v1",
        "account_key": "tachibana:default",
        "session_date": trade_date,
        "observed_at": f"{trade_date}T15:30:00+09:00",
        "sources": {
            name: {
                "authority": "broker_official",
                "source_id": f"official:{name}:{trade_date}",
                "sha256": "a" * 64,
                "reconciled_by": "account-ledger-reviewer",
                "verified": True,
            }
            for name in REQUIRED_SOURCES
        },
        "execution_crosswalk": crosswalk or [
            {"broker_execution_id": f"{trade_date}:fill-1", "local_fill_id": f"{trade_date}:local-1", "verified": True},
        ],
        "all_broker_fills_accounted_for": True,
        "all_local_fills_accounted_for": True,
        "cash_position_fee_balances_verified": True,
        "pending_fills": 0,
        "external_cash_flows_reconciled": True,
        "session_mark_policy_verified": True,
    }


def _session(trade_date: str, **changes):
    values = {
        "account_key": "tachibana:default",
        "trade_date": trade_date,
        "opening_equity_jpy": 100_000.0,
        "realized_pnl_jpy": 1_000.0,
        "unrealized_pnl_change_jpy": 200.0,
        "commission_jpy": 100.0,
        "taxes_jpy": 20.0,
        "financing_jpy": 30.0,
        "borrow_jpy": 10.0,
        "reverse_jpy": 40.0,
        "external_cash_flow_jpy": 0.0,
    }
    values.update(changes)
    pnl = (
        values["realized_pnl_jpy"]
        + values["unrealized_pnl_change_jpy"]
        - values["commission_jpy"]
        - values["taxes_jpy"]
        - values["financing_jpy"]
        - values["borrow_jpy"]
        - values["reverse_jpy"]
    )
    values.setdefault(
        "closing_equity_jpy",
        values["opening_equity_jpy"] + pnl + values["external_cash_flow_jpy"],
    )
    return values


def test_normal_and_partial_fill_ledger_matches_hand_calculation():
    manifest = _manifest(
        "2026-10-01",
        crosswalk=[
            {"broker_execution_id": "b1", "local_fill_id": "order1-fill1", "verified": True},
            {"broker_execution_id": "b2", "local_fill_id": "order1-fill2", "verified": True},
        ],
    )
    ledger = build_reconciled_execution_ledger(manifest, _session("2026-10-01"))
    assert ledger["net_pnl_jpy"] == pytest.approx(1_000.0)
    assert ledger["daily_return"] == pytest.approx(0.01)
    assert ledger["execution_crosswalk_count"] == 2


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda m: m["sources"]["fees"].update(verified=False), "fees: not verified"),
        (lambda m: m.update(cash_position_fee_balances_verified=False), "balances not verified"),
        (lambda m: m.update(pending_fills=1), "pending fills"),
        (lambda m: m["sources"]["fills"].update(verified=False, source_id="api-failure"), "fills: not verified"),
    ],
)
def test_missing_fee_position_mismatch_pending_fill_and_api_failure_block(mutation, match):
    manifest = _manifest("2026-10-01")
    mutation(manifest)
    with pytest.raises(LedgerReconciliationError, match=match):
        build_reconciled_execution_ledger(manifest, _session("2026-10-01"))


def test_external_cash_flow_is_in_cash_bridge_but_not_return():
    session = _session("2026-10-01", external_cash_flow_jpy=50_000.0)
    ledger = build_reconciled_execution_ledger(_manifest("2026-10-01"), session)
    assert ledger["closing_equity_jpy"] == pytest.approx(151_000.0)
    assert ledger["net_pnl_jpy"] == pytest.approx(1_000.0)
    assert ledger["daily_return"] == pytest.approx(0.01)


def test_equity_bridge_mismatch_is_rejected():
    session = _session("2026-10-01")
    session["closing_equity_jpy"] += 1.0
    with pytest.raises(LedgerReconciliationError, match="equity bridge mismatch"):
        build_reconciled_execution_ledger(_manifest("2026-10-01"), session)


def test_ledger_observed_next_morning_is_valid_and_snapshot_uses_observation_time(tmp_path):
    manifest = _manifest("2026-10-01")
    manifest["observed_at"] = "2026-10-02T08:00:00+09:00"
    ledger = build_reconciled_execution_ledger(manifest, _session("2026-10-01"))
    snapshot = build_account_risk_snapshot(
        [ledger],
        valid_for_trade_date="2026-10-02",
        account_key="tachibana:default",
    )
    assert snapshot["observed_through"] == "2026-10-01"
    assert snapshot["observed_at"] == "2026-10-02T08:00:00+09:00"

    write_account_risk_snapshot(tmp_path, snapshot)
    loaded = AccountRiskSnapshot.load(
        tmp_path / "latest.json",
        trade_date="2026-10-02",
        decision_as_of="2026-10-02T09:10:00+09:00",
        account_key="tachibana:default",
    )
    assert loaded.observed_at.isoformat() == "2026-10-02T08:00:00+09:00"


def test_snapshot_compounds_complete_month_and_writes_immutable_pointer(tmp_path):
    first = build_reconciled_execution_ledger(
        _manifest("2026-10-01"),
        _session("2026-10-01"),
    )
    second = build_reconciled_execution_ledger(
        _manifest("2026-10-02"),
        _session(
            "2026-10-02",
            opening_equity_jpy=101_000.0,
            realized_pnl_jpy=-1_000.0,
            unrealized_pnl_change_jpy=0.0,
            commission_jpy=10.0,
            taxes_jpy=0.0,
            financing_jpy=0.0,
            borrow_jpy=0.0,
            reverse_jpy=0.0,
        ),
    )
    snapshot = build_account_risk_snapshot(
        [first, second],
        valid_for_trade_date="2026-10-05",
        account_key="tachibana:default",
    )
    expected_month = (1.0 + first["daily_return"]) * (1.0 + second["daily_return"]) - 1.0
    assert snapshot["daily_return"] == pytest.approx(second["daily_return"])
    assert snapshot["month_return"] == pytest.approx(expected_month)

    immutable_path = write_account_risk_snapshot(tmp_path, snapshot)
    loaded = AccountRiskSnapshot.load(
        tmp_path / "latest.json",
        trade_date="2026-10-05",
        decision_as_of="2026-10-05T09:10:00+09:00",
        account_key="tachibana:default",
    )
    assert loaded.snapshot_id == snapshot["snapshot_id"]
    original = immutable_path.read_bytes()
    write_account_risk_snapshot(tmp_path, snapshot)
    assert immutable_path.read_bytes() == original


def test_snapshot_rejects_stale_or_incomplete_month():
    first = build_reconciled_execution_ledger(
        _manifest("2026-10-01"),
        _session("2026-10-01"),
    )
    with pytest.raises(LedgerReconciliationError, match="prior trading session"):
        build_account_risk_snapshot(
            [first],
            valid_for_trade_date="2026-10-05",
            account_key="tachibana:default",
        )

    second = build_reconciled_execution_ledger(
        _manifest("2026-10-02"),
        _session("2026-10-02"),
    )
    with pytest.raises(LedgerReconciliationError, match="month ledger coverage"):
        build_account_risk_snapshot(
            [second],
            valid_for_trade_date="2026-10-05",
            account_key="tachibana:default",
        )


def test_tampered_immutable_snapshot_is_rejected(tmp_path):
    first = build_reconciled_execution_ledger(
        _manifest("2026-10-01"),
        _session("2026-10-01"),
    )
    snapshot = build_account_risk_snapshot(
        [first],
        valid_for_trade_date="2026-10-02",
        account_key="tachibana:default",
    )
    immutable_path = write_account_risk_snapshot(tmp_path, snapshot)
    tampered = deepcopy(snapshot)
    tampered["daily_return"] = 0.5
    immutable_path.write_text(__import__("json").dumps(tampered), encoding="utf-8")
    with pytest.raises(Exception, match="hash mismatch"):
        AccountRiskSnapshot.load(
            tmp_path / "latest.json",
            trade_date="2026-10-02",
            decision_as_of="2026-10-02T09:10:00+09:00",
            account_key="tachibana:default",
        )
