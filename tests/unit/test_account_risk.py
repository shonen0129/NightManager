from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from leadlag.execution.account_risk import (
    ACCOUNT_RISK_SCHEMA,
    REQUIRED_PNL_BASIS,
    AccountRiskPreflight,
    AccountRiskSnapshot,
    AccountRiskSnapshotError,
    evaluate_account_loss,
    snapshot_payload_sha256,
)
from leadlag.execution.risk_capital import run_risk_checks


def _payload(**changes):
    payload = {
        "schema_version": ACCOUNT_RISK_SCHEMA,
        "valid_for_trade_date": "2026-09-29",
        "observed_through": "2026-09-28",
        "observed_at": "2026-09-28T15:30:00+09:00",
        "as_of": "2026-09-28T15:30:00+09:00",
        "account_key": "tachibana:default",
        "daily_return": -0.03,
        "month_return": -0.06,
        "source": "reconciled_execution_ledger",
        "source_ids": ["sha256:" + "a" * 64],
        "source_sha256s": ["a" * 64],
        "pnl_basis": REQUIRED_PNL_BASIS,
        "cash_reconciled": True,
        "positions_reconciled": True,
        "fees_complete": True,
        "reconciliation_status": "complete",
    }
    payload.update(changes)
    if "as_of" in changes and "observed_at" not in changes:
        payload["observed_at"] = payload["as_of"]
    digest = snapshot_payload_sha256(payload)
    payload["snapshot_sha256"] = digest
    payload["snapshot_id"] = f"sha256:{digest}"
    return payload


def _risk_config():
    return SimpleNamespace(
        daily_loss_warning=0.015,
        daily_loss_stop=0.025,
        monthly_loss_stop=0.05,
    )


def test_reconciled_account_loss_uses_actual_values_and_existing_thresholds():
    snapshot = AccountRiskSnapshot.from_payload(
        _payload(),
        trade_date="2026-09-29",
        decision_as_of="2026-09-29T09:10:05+09:00",
        account_key="tachibana:default",
    )

    report = evaluate_account_loss(snapshot, _risk_config())

    assert report["is_blocked"] is True
    assert report["daily_loss"] == pytest.approx(0.03)
    assert report["monthly_loss"] == pytest.approx(0.06)
    assert any(item.startswith("ActualAccountDailyLoss") for item in report["stop_breaches"])


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"fees_complete": False}, "fees_complete"),
        ({"reconciliation_status": "incomplete"}, "incomplete"),
        ({"pnl_basis": "collateral_balance_change"}, "PnL basis"),
        ({"as_of": "2026-09-29T09:10:06+09:00"}, "after the decision cutoff"),
        ({"valid_for_trade_date": "2026-09-30"}, "not valid for this trade_date"),
    ],
)
def test_account_risk_contract_rejects_incomplete_or_future_evidence(changes, message):
    with pytest.raises(AccountRiskSnapshotError, match=message):
        AccountRiskSnapshot.from_payload(
            _payload(**changes),
            trade_date="2026-09-29",
            decision_as_of="2026-09-29T09:10:05+09:00",
            account_key="tachibana:default",
        )


def test_live_risk_check_blocks_when_actual_account_snapshot_is_missing():
    config = SimpleNamespace(
        var_confidence=0.99,
        var_window=250,
        var_method="historical",
        var_warning=0.02,
        var_stop=0.03,
        es_warning=0.03,
        es_stop=0.04,
        daily_loss_warning=0.015,
        daily_loss_stop=0.025,
        monthly_loss_stop=0.05,
        max_net_exposure=0.05,
        max_gross_exposure=2.0,
    )
    report = run_risk_checks(
        decision={"weight": [0.0, 0.0]},
        total_buy_allocated=0.0,
        total_sell_allocated=0.0,
        max_capital=100_000.0,
        hist_daily_returns=pd.Series([0.01]),
        config=config,
        account_risk_preflight=AccountRiskPreflight.unavailable(
            AccountRiskSnapshotError("account-risk snapshot is missing")
        ),
    )

    assert report["is_blocked"] is True
    assert report["actual_account_risk"]["available"] is False
    assert "ActualAccountRiskUnavailable" in report["stop_breaches"][-1]


def test_actual_loss_stop_overrides_positive_replay_return():
    config = SimpleNamespace(
        var_confidence=0.99,
        var_window=250,
        var_method="historical",
        var_warning=0.02,
        var_stop=0.03,
        es_warning=0.03,
        es_stop=0.04,
        daily_loss_warning=0.015,
        daily_loss_stop=0.025,
        monthly_loss_stop=0.05,
        max_net_exposure=0.05,
        max_gross_exposure=2.0,
    )
    snapshot = AccountRiskSnapshot.from_payload(
        _payload(),
        trade_date="2026-09-29",
        decision_as_of="2026-09-29T09:10:05+09:00",
        account_key="tachibana:default",
    )
    report = run_risk_checks(
        decision={"weight": [0.0, 0.0]},
        total_buy_allocated=0.0,
        total_sell_allocated=0.0,
        max_capital=100_000.0,
        hist_daily_returns=pd.Series([0.01]),
        config=config,
        account_risk_preflight=AccountRiskPreflight.verified(snapshot),
    )

    assert report["is_blocked"] is True
    assert report["actual_account_risk"]["daily_loss"] == pytest.approx(0.03)
    assert "ActualAccountDailyLoss" in " ".join(report["stop_breaches"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"required": True},
        {"required": True, "snapshot": object(), "error": AccountRiskSnapshotError("bad")},
        {"required": False, "error": AccountRiskSnapshotError("unexpected")},
    ],
)
def test_account_risk_preflight_rejects_ambiguous_evidence(kwargs):
    with pytest.raises(ValueError, match="preflight"):
        AccountRiskPreflight(**kwargs)
