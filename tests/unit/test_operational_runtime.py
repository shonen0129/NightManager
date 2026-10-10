from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scripts.tools.phase_deadline import TIMEOUT_EXIT_CODE, run_phase

from leadlag.data.gap_store import GapStore
from leadlag.data.tickers import JP_TICKERS
from leadlag.domain.gap_bundle import GapBundleRef, canonical_json_bytes
from leadlag.domain.inputs import DecisionInputs
from leadlag.domain.portfolio import PortfolioDecision
from leadlag.execution.account_risk import AccountRiskPreflight, AccountRiskSnapshot
from leadlag.execution.gap_store_check import check_bundle
from leadlag.execution.runtime_manifest import (
    build_decision_manifest,
    update_account_risk_manifest,
    update_decision_manifest,
    update_execution_manifest,
)
from leadlag.reporting.results_format import write_run_manifest


def _script(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def test_phase_deadline_returns_child_code_and_evidence(tmp_path: Path) -> None:
    child = _script(tmp_path / "child.py", "raise SystemExit(7)\n")
    evidence = tmp_path / "phase.json"
    assert run_phase(
        [sys.executable, str(child)],
        label="unit",
        timeout_seconds=2,
        grace_seconds=0.1,
        log_path=evidence,
    ) == 7
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["return_code"] == 7


def test_phase_deadline_stops_a_hanging_child(tmp_path: Path) -> None:
    child = _script(tmp_path / "slow.py", "import time\ntime.sleep(30)\n")
    started = time.monotonic()
    assert run_phase(
        [sys.executable, str(child)],
        label="timeout",
        timeout_seconds=0.1,
        grace_seconds=0.1,
        log_path=tmp_path / "phase.json",
    ) == TIMEOUT_EXIT_CODE
    assert time.monotonic() - started < 2.0


def test_decision_manifest_captures_inputs_failure_reasons_and_code_state(tmp_path: Path) -> None:
    frame = pd.DataFrame({"value": [0.0]}, index=pd.DatetimeIndex(["2026-09-23"]))
    inputs = DecisionInputs.from_parts(
        frame,
        trade_date="2026-09-23",
        as_of="2026-09-23 09:10",
        ticker_order=tuple(JP_TICKERS),
        us_returns=np.zeros(15),
        jp_gap_returns=np.zeros(17),
        jp_betas=np.ones(17),
        topix_night_return=0.0,
        current_prices={ticker: 100.0 for ticker in JP_TICKERS},
        prev_closes={ticker: 99.0 for ticker in JP_TICKERS},
        observed_at={"current_prices": "2026-09-23 09:10"},
        source="unit",
    )
    config = SimpleNamespace(
        v2=SimpleNamespace(model_dump=lambda mode="json": {"fallback": {"enabled": True}}),
        model_dump=lambda mode="json": {"v2": {"fallback": {"enabled": True}}},
    )
    run_config = SimpleNamespace(version="v2", ml_overlay_model_dir="")
    result = PortfolioDecision(
        w_final=np.zeros(17),
        scores=np.zeros(17),
        mu_gap=np.zeros(17),
        sigma_gap=np.ones(17),
        Omega_gap=np.eye(17),
        fallback={"gap_data_missing": True, "audit_failure": False},
        pit_binning={"multiplier": 1.0},
        leakage={"status": "PASSED"},
        numerical={"status": "PASSED"},
        alerts=["cache missing"],
        summary={"target_gross": 0.0},
        run_config=run_config,
        diagnostics={"distribution_resolution": {"source": "flat", "reason": "cache_missing"}},
    )
    manifest = build_decision_manifest(
        app_config=config,
        inputs=inputs,
        result=result,
        config_path="configs/production/production.yaml",
        gap_input_dir="var/live/pipeline_data/gap_adjusted_distribution/gap_store.sqlite",
        model=SimpleNamespace(_overlay_model=None),
    )
    assert manifest["trade_date"] == "2026-09-23"
    assert manifest["input_version"]["digest"]
    assert manifest["decision"]["fallback_reasons"] == ["gap_data_missing"]
    assert manifest["code"]["dirty_diff_hash"]
    assert len(manifest["decision"]["arrays"]["scores"]) == len(JP_TICKERS)
    assert len(manifest["decision"]["arrays"]["Omega_gap"]) == len(JP_TICKERS)

    output_dir = tmp_path / "result"
    output_dir.mkdir()
    write_run_manifest(str(output_dir), "unit")
    update_decision_manifest(output_dir, manifest)
    saved = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert saved["runtime"]["gap"]["reason"] == "cache_missing"


def test_execution_manifest_records_quantities_and_post_position_exposure(tmp_path: Path) -> None:
    output_dir = tmp_path / "result"
    output_dir.mkdir()
    write_run_manifest(str(output_dir), "unit")
    position_snapshot = tmp_path / "positions.json"
    position_snapshot.write_text(
        json.dumps({
            "positions": [
                {"ticker": "1305", "side": "BUY", "quantity": 100},
                {"ticker": "1305", "side": "SELL", "quantity": 20},
            ]
        }),
        encoding="utf-8",
    )
    update_execution_manifest(
        output_dir,
        phase="decision",
        summary={
            "run_id": "run-1",
            "buy_results": [{"ticker": "1305", "side": "BUY", "quantity": 100, "filled_quantity": 80, "status": "FILLED"}],
            "sell_results": [{"ticker": "1305", "side": "SELL", "quantity": 20, "filled_quantity": 20, "status": "FILLED"}],
            "execution_report": {"incomplete": False},
        },
        position_snapshot_path=position_snapshot,
    )
    saved = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    exposure = saved["execution"]["post_execution_exposure"]
    assert exposure["requested_quantity_by_ticker"] == {"1305": {"BUY": 100, "SELL": 20}}
    assert exposure["filled_quantity_by_ticker"] == {"1305": {"BUY": 80, "SELL": 20}}
    assert exposure["signed_quantity_by_ticker"] == {"1305": 80}
    assert exposure["net_quantity"] == 80


def test_decision_manifest_references_exact_account_risk_snapshot(tmp_path: Path) -> None:
    output_dir = tmp_path / "result"
    output_dir.mkdir()
    write_run_manifest(str(output_dir), "unit")
    snapshot = AccountRiskSnapshot(
        valid_for_trade_date="2026-10-02",
        observed_through="2026-10-01",
        observed_at=pd.Timestamp("2026-10-01T15:30:00+09:00"),
        as_of=pd.Timestamp("2026-10-01T15:30:00+09:00"),
        account_key="tachibana:default",
        daily_return=-0.01,
        month_return=-0.01,
        source="reconciled_execution_ledger",
        source_ids=("sha256:" + "a" * 64,),
        source_sha256s=("a" * 64,),
        snapshot_id="sha256:" + "b" * 64,
        snapshot_sha256="b" * 64,
        pnl_basis="daily_realized_plus_unrealized_change_less_observed_fees",
        cash_reconciled=True,
        positions_reconciled=True,
        fees_complete=True,
        reconciliation_status="complete",
    )
    update_account_risk_manifest(output_dir, AccountRiskPreflight.verified(snapshot))
    saved = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert saved["account_risk"]["snapshot_id"] == snapshot.snapshot_id
    assert saved["account_risk"]["snapshot_sha256"] == snapshot.snapshot_sha256
    assert saved["account_risk"]["source_ids"] == list(snapshot.source_ids)


def test_gap_store_check_requires_the_same_date_bundle(tmp_path: Path) -> None:
    store_path = tmp_path / "gap.sqlite"
    store = GapStore(store_path)
    mu = np.array([0.1, 0.2])
    omega = np.eye(2)
    metadata = {"bundle_version": "gap-v1", "ticker_order": ["1305", "1321"]}
    manifest = GapBundleRef.from_payload(
        trade_date="2026-09-23",
        horizon=None,
        storage_format="sqlite",
        mu_bytes=np.ascontiguousarray(mu).tobytes(),
        omega_bytes=np.ascontiguousarray(omega).tobytes(),
        metadata_bytes=canonical_json_bytes(metadata),
        metadata=metadata,
    )
    store.save("2026-09-23", mu, omega, metadata, manifest=manifest)
    result = check_bundle(store_path, "2026-09-23")
    assert result["status"] == "ready"
    assert result["bundle_version"] == "gap-v1"

    wrong_date = check_bundle(store_path, "2026-09-22")
    assert wrong_date["status"] == "missing"
