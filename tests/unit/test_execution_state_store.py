from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from leadlag.core.types import OrderRequest, OrderResult, OrderSide, OrderStatus
from leadlag.execution.broker_ops import OrderExecutionIncomplete, submit_orders_via_api
from leadlag.execution.contracts import ExecutionPlan
from leadlag.execution.state_store import (
    ExecutionRunStatus,
    ExecutionStateConflict,
    ExecutionStateStore,
)


def _plan() -> ExecutionPlan:
    return ExecutionPlan(
        decision_id="decision-1",
        trade_date="2026-09-16",
        close_orders=(),
        new_orders=(OrderRequest("1305.T", OrderSide.BUY, 10),),
    )


def test_intent_is_committed_before_submission_and_ambiguous_run_blocks_retry(tmp_path: Path) -> None:
    store = ExecutionStateStore(tmp_path / "execution.sqlite")
    assert store.schema_version == 2
    run = store.prepare_run(
        account_key="kabu:default",
        strategy_key="production_v2",
        trade_date="2026-09-16",
        job_type="decision",
        decision_id="decision-1",
    )
    intent_ids = store.record_plan(run.run_id, _plan())
    assert len(intent_ids) == 1
    store.mark_submission_started(run.run_id)
    store.record_observation(
        run_id=run.run_id,
        intent_id=intent_ids[0],
        broker_order_id="order-1",
        ticker="1305.T",
        status="SUBMITTED",
        requested_quantity=10,
        filled_quantity=0,
        remaining_quantity=10,
    )
    store.mark_reconciliation_required(run.run_id, "process stopped after submit")

    candidates = store.list_recovery_candidates()
    assert [candidate.run_id for candidate in candidates] == [run.run_id]
    with pytest.raises(ExecutionStateConflict, match="reconcile before retry"):
        store.prepare_run(
            account_key="kabu:default",
            strategy_key="production_v2",
            trade_date="2026-09-16",
            job_type="decision",
            decision_id="decision-1",
        )


def test_result_observation_is_idempotent_and_completed_run_is_single_flight(tmp_path: Path) -> None:
    store = ExecutionStateStore(tmp_path / "execution.sqlite")
    plan = _plan()
    run = store.prepare_run(
        account_key="dry_run",
        strategy_key="production_v2",
        trade_date=plan.trade_date,
        job_type="decision",
        decision_id=plan.decision_id,
    )
    store.record_plan(run.run_id, plan)
    store.mark_submission_started(run.run_id)
    record = {
        "ticker": "1305.T",
        "side": "BUY",
        "quantity": 10,
        "order_id": "sim-1",
        "status": "FILLED",
        "fill_quantity": 10,
        "fill_price": 100.0,
        "fee": 1.5,
    }
    store.record_result_set(run.run_id, plan, [record])
    store.record_result_set(run.run_id, plan, [record])
    reconciliation_id = store.record_reconciliation(
        run.run_id,
        outcome="complete",
        references={"api_execution_log": "api_execution_log.json"},
    )
    assert reconciliation_id > 0
    latest = store.latest_reconciliation(run.run_id)
    assert latest is not None
    assert latest["outcome"] == "complete"
    assert latest["references"]["api_execution_log"] == "api_execution_log.json"
    fills = store.list_observed_fills(run.run_id)
    assert len(fills) == 1
    assert fills[0].price == pytest.approx(100.0)
    assert fills[0].fee == pytest.approx(1.5)
    store.mark_completed(run.run_id)
    with pytest.raises(ExecutionStateConflict):
        store.prepare_run(
            account_key="dry_run",
            strategy_key="production_v2",
            trade_date=plan.trade_date,
            job_type="decision",
            decision_id=plan.decision_id,
        )
    with store._connect() as conn:  # schema-level assertion, not production API use
        assert conn.execute("SELECT COUNT(*) FROM order_observations").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM execution_reconciliations").fetchone()[0] == 1
        assert conn.execute("SELECT schema_version FROM execution_schema").fetchone()[0] == 2
    assert store.get_run(run.run_id).status == ExecutionRunStatus.COMPLETED


def test_resuming_prepared_run_is_blocked_by_another_unresolved_run(tmp_path: Path) -> None:
    store = ExecutionStateStore(tmp_path / "execution.sqlite")
    prepared = store.prepare_run(
        account_key="account",
        strategy_key="production_v2",
        trade_date="2026-09-16",
        job_type="close",
        decision_id="close-1",
    )
    unresolved = store.prepare_run(
        account_key="account",
        strategy_key="production_v2",
        trade_date="2026-09-17",
        job_type="decision",
        decision_id="decision-1",
    )
    store.mark_submission_started(unresolved.run_id)

    with pytest.raises(ExecutionStateConflict, match="unresolved"):
        store.prepare_run(
            account_key="account",
            strategy_key="production_v2",
            trade_date="2026-09-16",
            job_type="close",
            decision_id="close-1",
        )
    assert store.get_run(prepared.run_id).status == ExecutionRunStatus.PREPARED


def test_account_lease_prevents_decision_close_overlap_and_releases(tmp_path: Path) -> None:
    store = ExecutionStateStore(tmp_path / "execution.sqlite")
    lease = store.acquire_lease("live:kabu:default:production_v2", ttl_seconds=10)
    with pytest.raises(ExecutionStateConflict):
        store.acquire_lease("live:kabu:default:production_v2", ttl_seconds=10)
    lease.release()
    second = store.acquire_lease("live:kabu:default:production_v2", ttl_seconds=10)
    second.release()


def test_submitter_persists_run_and_blocks_same_day_resubmission(tmp_path: Path) -> None:
    class FilledBroker:
        def __init__(self) -> None:
            self.calls = 0

        def submit_orders_batch(self, orders, **_kwargs):
            self.calls += 1
            return [
                OrderResult(
                    order_id="broker-1",
                    status=OrderStatus.FILLED,
                    ticker=order.ticker,
                    side=order.side,
                    quantity=order.quantity,
                    order_type=order.order_type,
                )
                for order in orders
            ]

    frame = pd.DataFrame(
        {"ticker": ["1305.T"], "action": ["BUY"], "quantity": [10]}
    )
    frame.attrs["trade_date"] = "2026-09-16"
    broker = FilledBroker()
    store = ExecutionStateStore(tmp_path / "execution.sqlite")

    summary = submit_orders_via_api(frame, broker, tmp_path, state_store=store)
    assert summary["run_id"]
    assert broker.calls == 1
    assert store.get_run(summary["run_id"]).status == ExecutionRunStatus.EXECUTING
    assert store.latest_reconciliation(summary["run_id"])["outcome"] == "pending"
    assert store.list_recovery_candidates()[0].run_id == summary["run_id"]

    with pytest.raises(OrderExecutionIncomplete, match="already executing"):
        submit_orders_via_api(frame, broker, tmp_path, state_store=store)
    assert broker.calls == 1
