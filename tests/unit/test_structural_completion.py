"""Behavioral regressions found while checking the S0–S8 completion claims."""

from dataclasses import replace

import pandas as pd
import pytest

from leadlag.core.types import OrderRequest, OrderSide
from leadlag.data.pit_lake import PITDataLake
from leadlag.domain.inputs import HistoricalInputs
from leadlag.execution.contracts import ExecutionPlan
from leadlag.execution.state_store import ExecutionStateConflict, ExecutionStateStore


def test_calculation_frame_cannot_change_versioned_history():
    history = HistoricalInputs(pd.DataFrame({"x": [1.0]}, index=pd.to_datetime(["2026-09-16"])))
    before = history.fingerprint
    exposed = history.calculation_frame()
    exposed.iloc[0, 0] = 9.0
    assert history.fingerprint == before
    assert history.to_frame().iloc[0, 0] == 1.0


def test_known_input_rejects_future_observation_and_protects_array_buffer():
    from leadlag.domain.inputs import KnownMarketInputs

    kwargs = dict(trade_date="2026-09-16", as_of="2026-09-16 00:10Z", ticker_order=("A",),
                  us_returns=[0.1], jp_gap_returns=[0.1], jp_betas=[1.0], topix_night_return=0.0)
    known = KnownMarketInputs(**kwargs)
    assert known.as_of == pd.Timestamp("2026-09-16 09:10")
    with pytest.raises(ValueError):
        known.us_returns.setflags(write=True)
    with pytest.raises(ValueError, match="later than as_of"):
        KnownMarketInputs(**kwargs, observed_at={"price": "2026-09-16 09:11"})


def test_prediction_contract_does_not_accept_evaluation_targets():
    from leadlag.domain.inputs import DecisionInputs

    with pytest.raises(TypeError, match="evaluation"):
        DecisionInputs(known=None, historical=None, evaluation=object())


@pytest.mark.parametrize("horizon", [3, 5])
def test_legacy_regression_bundle_keeps_verified_horizon(horizon):
    from pathlib import Path

    import numpy as np

    from leadlag.utils.gap_matrix_io import load_gap_bundle

    root = Path(__file__).resolve().parents[1] / "regression/baselines"
    mu, omega, metadata, alerts = load_gap_bundle(
        root, "2026-08-14", mu_pattern="matrices/mu_gap_h{h}_{date}.npy",
        omega_pattern="matrices/omega_gap_h{h}_{date}.npy", pattern_kwargs={"h": horizon},
        n_j=17, require_metadata=True,
    )
    assert mu is not None, alerts
    assert metadata["horizon"] == horizon
    np.testing.assert_array_equal(mu, np.load(root / f"matrices/mu_gap_h{horizon}_20260814.npy"))
    assert omega.shape == (17, 17)


def test_snapshot_timestamp_and_requested_date_must_agree():
    lake = PITDataLake(pd.DataFrame({"x": [1.0, 2.0]}, index=pd.to_datetime(["2026-09-15", "2026-09-16"])))
    with pytest.raises(ValueError, match="snapshot.*date"):
        lake.build_decision_inputs("2026-09-16", snapshot=lake.get_snapshot("2026-09-15"))


def test_snapshot_timestamp_must_not_be_after_requested_cutoff():
    lake = PITDataLake(pd.DataFrame({"x": [1.0]}, index=pd.to_datetime(["2026-09-16"])))
    with pytest.raises(ValueError, match="snapshot timestamp"):
        lake.build_decision_inputs(
            "2026-09-16 09:10",
            snapshot=lake.get_snapshot("2026-09-16 15:31"),
        )
    with pytest.raises(ValueError, match="snapshot timestamp"):
        lake.build_decision_inputs(
            "2026-09-16",
            snapshot=lake.get_snapshot("2026-09-16 15:31"),
        )


def test_pit_lake_rejects_empty_or_duplicate_trade_dates():
    with pytest.raises(ValueError, match="at least one"):
        PITDataLake(pd.DataFrame(index=pd.DatetimeIndex([], dtype="datetime64[ns]")))
    with pytest.raises(ValueError, match="valid trade dates"):
        PITDataLake(pd.DataFrame({"x": [1.0]}, index=pd.to_datetime([pd.NaT])))
    with pytest.raises(ValueError, match="unique"):
        PITDataLake(
            pd.DataFrame(
                {"x": [1.0, 2.0]},
                index=pd.to_datetime(["2026-09-16", "2026-09-16"]),
            )
        )


def test_snapshot_accepts_jst_decision_timestamp():
    lake = PITDataLake(pd.DataFrame({"x": [1.0]}, index=pd.to_datetime(["2026-09-16"])))
    snapshot = lake.get_snapshot("2026-09-16 09:10")
    assert snapshot.as_of == pd.Timestamp("2026-09-16 09:10")
    assert snapshot.trade_date == "2026-09-16"


def _prepared(store):
    plan = ExecutionPlan(decision_id="d", trade_date="2026-09-16", close_orders=(),
                         new_orders=(OrderRequest("1305.T", OrderSide.BUY, 10),))
    run = store.prepare_run(account_key="test", strategy_key="v2", trade_date=plan.trade_date,
                            job_type="decision", decision_id=plan.decision_id)
    store.record_plan(run.run_id, plan)
    return run, plan


def test_reprepared_plan_cannot_silently_change_order_quantity(tmp_path):
    store = ExecutionStateStore(tmp_path / "state.sqlite")
    run, plan = _prepared(store)
    changed = replace(plan, new_orders=(OrderRequest("1305.T", OrderSide.BUY, 20),))
    with pytest.raises(ExecutionStateConflict, match="plan|intent"):
        store.record_plan(run.run_id, changed)


def test_reconciliation_failure_can_be_updated_and_then_completed(tmp_path):
    store = ExecutionStateStore(tmp_path / "state.sqlite")
    run, _ = _prepared(store)
    store.mark_submission_started(run.run_id)
    store.mark_reconciliation_required(run.run_id, "interrupted")
    store.mark_reconciliation_required(run.run_id, "positions unavailable")
    assert store.get_run(run.run_id).last_error == "positions unavailable"
    store.record_reconciliation(run.run_id, outcome="complete", references={"positions": "checked"})
    store.mark_completed(run.run_id)
    assert store.list_recovery_candidates() == []


def test_split_orders_keep_intent_when_final_summary_changes_order(tmp_path):
    store = ExecutionStateStore(tmp_path / "state.sqlite")
    run, plan = _prepared(store)
    a = {"ticker": "1305.T", "side": "BUY", "quantity": 4, "order_id": "a", "status": "SUBMITTED"}
    b = {**a, "quantity": 6, "order_id": "b"}
    store.record_result_set(run.run_id, plan, [a, b])
    store.record_result_set(run.run_id, plan, [{**b, "status": "FILLED"}, {**a, "status": "FILLED"}])
    with store._connect() as conn:
        rows = conn.execute("SELECT intent_id, requested_quantity FROM order_observations").fetchall()
    assert len(rows) == 4
    assert len({row["intent_id"] for row in rows}) == 1
    assert rows[0]["intent_id"] is not None
    assert [row["requested_quantity"] for row in rows] == [4, 6, 6, 4]


def test_var_snapshot_includes_canonical_pit_history(tmp_path):
    import sqlite3

    from leadlag.execution.var_inputs import _snapshot_gap_input

    source = tmp_path / "gap.sqlite"
    with sqlite3.connect(source) as conn:
        conn.execute("CREATE TABLE example (value INTEGER)")
    history = tmp_path / "full_history_diagnostics.csv"
    history.write_text("trade_date,pred_ir_gap\n2026-09-15,1.0\n")
    snapshot, before, owner = _snapshot_gap_input(source)
    try:
        assert (snapshot.parent / history.name).read_text() == history.read_text()
        history.write_text("trade_date,pred_ir_gap\n2026-09-15,9.0\n")
        _, after, owner2 = _snapshot_gap_input(source)
        try:
            assert before != after
            assert "1.0" in (snapshot.parent / history.name).read_text()
        finally:
            owner2.cleanup()
    finally:
        owner.cleanup()
