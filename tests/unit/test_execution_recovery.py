from dataclasses import replace
from types import SimpleNamespace

import pytest

from leadlag.broker.base import Position, WalletInfo
from leadlag.core.types import OrderRequest, OrderSide, OrderStatus
from leadlag.execution import reconcile
from leadlag.execution.contracts import ExecutionPlan
from leadlag.execution.state_store import ExecutionStateConflict, ExecutionStateStore


def _run(tmp_path, *, known_id=True, account="test", strategy="v2"):
    store = ExecutionStateStore(tmp_path / "state.sqlite")
    plan = ExecutionPlan(decision_id="d", trade_date="2026-09-16", close_orders=(),
                         new_orders=(OrderRequest("1305.T", OrderSide.BUY, 10),))
    run = store.prepare_run(account_key=account, strategy_key=strategy, trade_date=plan.trade_date,
                            job_type="decision", decision_id=plan.decision_id)
    store.record_plan(run.run_id, plan)
    store.mark_submission_started(run.run_id)
    if known_id:
        store.record_result_set(run.run_id, plan, [{"ticker": "1305.T", "side": "BUY", "quantity": 10,
                                                  "order_id": "a", "status": "SUBMITTED"}])
    broker = SimpleNamespace(
        get_order_status=lambda _id: OrderStatus.FILLED,
        get_positions=lambda: [Position("1305.T", "BUY", 10)],
        get_wallet=lambda: WalletInfo(cash_available=1000),
    )
    return store, run, plan, broker


@pytest.mark.parametrize("known_id", [False, True])
def test_pending_cli_reconciles_configured_account_without_submission(tmp_path, monkeypatch, known_id):
    from leadlag.execution import broker_ops, config

    store, run, _, broker = _run(tmp_path, known_id=known_id, account="tachibana:default", strategy="production_v2")
    # A simulated account must never be reconciled against the real broker.
    other = store.prepare_run(account_key="simulation:test", strategy_key="production_v2",
                              trade_date="2026-09-16", job_type="decision", decision_id="other")
    store.mark_submission_started(other.run_id)
    closed = []
    broker.close = lambda: closed.append(True)
    monkeypatch.setattr(config, "load_config_from_yaml", lambda **kw: SimpleNamespace(broker_provider="tachibana"))
    monkeypatch.setattr(broker_ops, "build_api_client", lambda *a: broker)

    def fills(_broker, records):
        for record in records:
            record.update(fill_quantity=10, fill_price=100, fee=1.0)

    monkeypatch.setattr(reconcile, "fetch_fill_prices", fills)
    code = reconcile.main(["--state-db", str(store.path), "--pending", "--output-dir", str(tmp_path / "out")])
    assert code == (0 if known_id else 2)
    assert closed == [True]
    remaining = {item.run_id for item in store.list_recovery_candidates()}
    assert other.run_id in remaining
    assert (run.run_id not in remaining) == known_id


def test_pending_cli_without_candidates_does_not_connect_to_broker(tmp_path, monkeypatch):
    from leadlag.execution import broker_ops, config

    monkeypatch.setattr(config, "load_config_from_yaml", lambda **kw: SimpleNamespace(broker_provider="tachibana"))

    def unexpected(*args):
        raise AssertionError("No broker connection is needed without pending production runs")

    monkeypatch.setattr(broker_ops, "build_api_client", unexpected)
    assert reconcile.main(["--state-db", str(tmp_path / "state.sqlite"), "--pending"]) == 0


def test_account_snapshot_is_read_only_and_persists_wallet_positions(tmp_path):
    broker = SimpleNamespace(
        get_positions=lambda: [Position("1305.T", "BUY", 10)],
        get_wallet=lambda: WalletInfo(cash_available=1000),
    )
    result = reconcile.collect_readonly_account_snapshot(broker, tmp_path, provider="stub")
    assert result["read_only"] is True
    assert result["position_count"] == 1
    assert result["wallet"]["cash_available"] == 1000
    assert result["path"]


def test_account_snapshot_cli_never_submits_orders(tmp_path, monkeypatch):
    from leadlag.execution import broker_ops, config

    calls: list[str] = []
    broker = SimpleNamespace(
        get_positions=lambda: [],
        get_wallet=lambda: WalletInfo(cash_available=1000),
        close=lambda: calls.append("close"),
        submit_order=lambda *_a, **_kw: calls.append("submit"),
    )
    monkeypatch.setattr(
        config,
        "load_config_from_yaml",
        lambda **_kw: SimpleNamespace(broker_provider="tachibana"),
    )
    monkeypatch.setattr(broker_ops, "build_api_client", lambda *_a, **_kw: broker)
    assert reconcile.main(
        ["--state-db", str(tmp_path / "state.sqlite"), "--account-snapshot", "--output-dir", str(tmp_path)]
    ) == 0
    assert calls == ["close"]


def test_recovery_reconciles_without_resubmission_and_persists_fills(tmp_path, monkeypatch):
    store, run, _, broker = _run(tmp_path)

    def fills(_broker, records):
        for record in records:
            record.update(fill_quantity=10, fill_price=100, fee=1.0)

    monkeypatch.setattr(reconcile, "fetch_fill_prices", fills)
    result = reconcile.reconcile_run(store, run.run_id, broker, tmp_path)
    assert result["complete"]
    assert not store.list_recovery_candidates()
    assert store.list_observed_fills(run.run_id)[0].quantity == 10
    assert store.latest_reconciliation(run.run_id)["references"]["recovery_journal"]


@pytest.mark.parametrize("missing_id", [False, True])
def test_recovery_preserves_ambiguous_submission_or_position_mismatch(tmp_path, monkeypatch, missing_id):
    store, run, _, broker = _run(tmp_path, known_id=not missing_id)
    broker.get_positions = lambda: []

    def fills(_broker, records):
        for record in records:
            record.update(fill_quantity=10, fill_price=100, fee=1.0)

    monkeypatch.setattr(reconcile, "fetch_fill_prices", fills)
    result = reconcile.reconcile_run(store, run.run_id, broker, tmp_path)
    assert not result["complete"]
    assert store.list_recovery_candidates()[0].run_id == run.run_id


def test_unresolved_run_blocks_next_day_and_close(tmp_path):
    store, _, plan, _ = _run(tmp_path)
    for date, kind in (("2026-09-17", "decision"), ("2026-09-16", "close")):
        with pytest.raises(ExecutionStateConflict, match="unresolved"):
            store.prepare_run(account_key="test", strategy_key="v2", trade_date=date,
                              job_type=kind, decision_id=replace(plan, trade_date=date).decision_id)


def test_checkpoint_requires_price_and_fee_for_partial_fill(tmp_path):
    import json

    snapshot = tmp_path / "positions.json"
    snapshot.write_text(
        json.dumps({"positions": [{"ticker": "1305.T", "side": "BUY", "quantity": 5}]})
    )
    plan = {"current_positions": {"1305.T": 0}}
    record = {
        "order_id": "order-1",
        "ticker": "1305.T",
        "side": "BUY",
        "quantity": 10,
        "status": "CANCELLED",
        "fill_quantity": 5,
        "fill_price": None,
    }
    errors = reconcile.verify_position_checkpoint(plan, [record], str(snapshot))
    assert any("fill price" in error for error in errors)
    assert any("fill fee" in error for error in errors)


@pytest.mark.parametrize("checkpoint_fails", [False, True])
def test_post_decision_completes_only_after_persisting_enriched_fills(tmp_path, monkeypatch, checkpoint_fails):
    import json

    import pandas as pd

    from leadlag.execution import post_decision

    store, run, _, broker = _run(tmp_path)
    frame = pd.DataFrame({"ticker": ["1305.T"], "action": ["BUY"], "quantity": [10]})
    frame.attrs["trade_date"] = "2026-09-16"
    summary = {"run_id": run.run_id, "expected_orders_count": 1, "buy_results": [
        {"ticker": "1305.T", "side": "BUY", "quantity": 10, "order_id": "a", "status": "FILLED"},
    ]}
    snapshot = tmp_path / "positions.json"
    snapshot.write_text(json.dumps({"positions": [{"ticker": "1305.T", "side": "BUY", "quantity": 10}]}))
    monkeypatch.setattr(post_decision, "save_decision_output", lambda *a, **kw: "decision.csv")
    monkeypatch.setattr(post_decision, "submit_orders_via_api", lambda **kw: summary)
    monkeypatch.setattr(post_decision, "save_position_snapshot", lambda *a, **kw: str(snapshot))
    monkeypatch.setattr(post_decision, "save_wallet_snapshot", lambda *a, **kw: "wallet.json")
    monkeypatch.setattr(post_decision, "save_daily_journal", lambda **kw: "journal.json")

    def fill_prices(_broker, records):
        for record in records:
            record.update(fill_quantity=10, fill_price=100.0, fee=1.0)

    monkeypatch.setattr(post_decision, "fetch_fill_prices", fill_prices)
    if checkpoint_fails:
        def fail(*a, **kw):
            raise OSError("checkpoint disk failure")

        monkeypatch.setattr(store, "record_reconciliation", fail)
        with pytest.raises(RuntimeError, match="checkpoint disk failure"):
            post_decision._write_decision_output_and_submit(
                frame, {"trade_date": "2026-09-16"}, tmp_path, False, broker, {}, store,
            )
        assert store.list_recovery_candidates()[0].run_id == run.run_id
        logged = json.loads((tmp_path / "api_execution_log.json").read_text())
        assert "checkpoint disk failure" in logged["reconciliation_errors"][0]
    else:
        post_decision._write_decision_output_and_submit(
            frame, {"trade_date": "2026-09-16"}, tmp_path, False, broker, {}, store,
        )
        assert not store.list_recovery_candidates()
        assert store.list_observed_fills(run.run_id)[0].price == 100.0
