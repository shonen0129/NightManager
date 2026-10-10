from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from tools.validation import build_readonly_shadow_acceptance_report as acceptance

from leadlag.data.adr_features import publish_adr_features
from leadlag.data.gap_store import GapStore
from leadlag.data.quote_snapshot import freeze_quote_snapshot, load_frozen_quote_snapshot
from leadlag.data.tickers import ADR_SECTOR_MAP, JP_TICKERS, JP_TICKERS_WITH_TOPIX
from leadlag.execution.account_risk import ACCOUNT_RISK_SCHEMA, REQUIRED_PNL_BASIS

TRADE_DATE = "2026-09-29"


def _risk_config() -> SimpleNamespace:
    return SimpleNamespace(
        daily_loss_warning=0.015,
        daily_loss_stop=0.025,
        monthly_loss_stop=0.05,
    )


def _write_adr_bundle(tmp_path: Path, trade_date: str = TRADE_DATE) -> Path:
    trade_ts = pd.Timestamp(trade_date)
    signal_ts = trade_ts - pd.Timedelta(days=1)
    row: dict[str, object] = {"sig_date": signal_ts}
    for ticker in JP_TICKERS:
        if ADR_SECTOR_MAP[ticker]:
            row[f"adr_{ticker}"] = 0.01
            row[f"coverage_{ticker}"] = 1
        else:
            row[f"adr_{ticker}"] = 0.0
            row[f"coverage_{ticker}"] = 0
    frame = pd.DataFrame([row], index=[trade_ts])
    frame.index.name = "trade_date"
    path = tmp_path / "adr_features.zip"
    publish_adr_features(frame, path, required_trade_date=trade_date)
    return path


def _quote_record() -> dict:
    started = f"{TRADE_DATE}T09:10:02+09:00"
    received = f"{TRADE_DATE}T09:10:06+09:00"
    rows = []
    for index, ticker in enumerate(JP_TICKERS_WITH_TOPIX):
        bid = 100.0 + index
        ask = bid + 2.0
        rows.append(
            {
                "ticker": ticker,
                "status": "OBSERVED",
                "request_started_at": started,
                "response_received_at": received,
                "observed_at": received,
                "timestamp_source": "local_response_receipt",
                "source": "tachibana:CLMMfdsGetMarketPrice",
                "bid_price_1": bid,
                "ask_price_1": ask,
                "quote_mid_price": (bid + ask) / 2.0,
                "valid_two_sided_quote": True,
            }
        )
    return {
        "status": "OBSERVED",
        "window_valid": True,
        "request_started_at": started,
        "response_received_at": received,
        "timestamp_source": "local_response_receipt",
        "source": "tachibana:CLMMfdsGetMarketPrice",
        "rows": rows,
    }


def _write_common_artifacts(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path, str]:
    capture_dir = tmp_path / "capture"
    shadow_dir = tmp_path / "shadow"
    log_dir = tmp_path / "logs"
    capture_dir.mkdir()
    shadow_dir.mkdir()
    log_dir.mkdir()

    preflight = {
        "schema_version": "readonly-shadow-acceptance-preflight-v1",
        "status": "READY",
        "checked_at": f"{TRADE_DATE}T08:45:00+09:00",
        "blocking_checks": [],
        "git": {"sha": "abc123", "dirty": False, "changed_path_count": 0},
        "api": {
            "valid": True,
            "scheme": "https",
            "host": "kabuka.e-shiten.jp",
            "path": "/e_api_v4r10",
            "version": "v4r10",
        },
        "capture_output_dir": str(capture_dir),
        "mode": {"shadow_only_env": "1", "capture_0910_env": "1"},
    }
    (capture_dir / "preflight.json").write_text(json.dumps(preflight), encoding="utf-8")

    (capture_dir / "capture_20260929.json").write_text(
        json.dumps({"run_id": "capture-1", "status": "CAPTURED", "attempts": [{"status": "CAPTURED"}]}),
        encoding="utf-8",
    )
    url_states = {
        key: {
            "state": "nonempty",
            "decrypt_attempted": True,
            "decrypt_succeeded": True,
        }
        for key in ("sUrlRequest", "sUrlMaster", "sUrlPrice", "sUrlEvent")
    }
    auth_record = {
        "schema_version": "tachibana-login-diagnostics-v1",
        "run_id": "capture-1",
        "attempt": 1,
        "capture_status": "CAPTURED",
        "diagnostics": {
            "http_status": 200,
            "response_parsed": True,
            "sKinsyouhouMidokuFlg": {"state": "value", "type": "str", "value": "0"},
            "virtual_urls": url_states,
            "login_success": True,
            "stopped_at": None,
        },
    }
    (capture_dir / "auth_diagnostics.jsonl").write_text(
        json.dumps(auth_record) + "\n",
        encoding="utf-8",
    )
    freeze_quote_snapshot(_quote_record(), capture_dir, trade_date=TRADE_DATE)
    frozen = load_frozen_quote_snapshot(capture_dir, trade_date=TRADE_DATE)

    gap_path = tmp_path / "gap_store.sqlite"
    gap_metadata = {
        "trade_date": TRADE_DATE,
        "source": "compute_gap_adjusted_distribution",
        "quote_snapshot_id": frozen.snapshot_id,
        "input_version": "input-v1",
        "model_version": "model-v1",
        "config_version": "config-v1",
        "gap_inputs_version": "gap-input-v1",
    }
    GapStore(gap_path).save(
        TRADE_DATE,
        np.zeros(17),
        np.eye(17),
        metadata=gap_metadata,
    )

    shadow = {
        "schema_version": "ml-overlay-research-paired-shadow-v2",
        "trade_date": TRADE_DATE,
        "as_of": f"{TRADE_DATE}T09:10:06+09:00",
        "quote_snapshot_id": frozen.snapshot_id,
        "input_version": {"digest": "input-digest"},
        "record_fingerprint": "shadow-fingerprint",
        "status": "complete",
        "attempt": 1,
    }
    (shadow_dir / "daily.jsonl").write_text(json.dumps(shadow) + "\n", encoding="utf-8")

    completed = {
        "schema_version": 1,
        "status": "completed",
        "return_code": 0,
        "started_at": f"{TRADE_DATE}T00:10:00+00:00",
        "finished_at": f"{TRADE_DATE}T00:10:10+00:00",
    }
    (log_dir / "microstructure_0910_20260929.json").write_text(
        json.dumps({**completed, "command": ["collect_0910_microstructure.py"]}),
        encoding="utf-8",
    )
    (log_dir / "decision_20260929.json").write_text(
        json.dumps({**completed, "command": ["bash", "run_decision_v2.sh"]}),
        encoding="utf-8",
    )
    (log_dir / "decision_20260929_gap_distribution.json").write_text(
        json.dumps({**completed, "command": ["bash", "run_gap_distribution.sh"]}),
        encoding="utf-8",
    )
    (log_dir / "decision_20260929_decision.json").write_text(
        json.dumps(
            {
                **completed,
                "command": [
                    "python",
                    "-m",
                    "leadlag.cli",
                    "decision",
                    "--shadow-only",
                    "--api-enable",
                ],
            }
        ),
        encoding="utf-8",
    )
    risk_path = tmp_path / "risk.json"
    return capture_dir, gap_path, shadow_dir, risk_path, log_dir, frozen.snapshot_id


def test_market_to_shadow_passes_while_missing_risk_keeps_stage1_blocked(tmp_path):
    capture_dir, gap_path, shadow_dir, risk_path, log_dir, snapshot_id = _write_common_artifacts(tmp_path)

    report = acceptance.build_acceptance_report(
        trade_date=TRADE_DATE,
        capture_dir=capture_dir,
        gap_store=gap_path,
        shadow_dir=shadow_dir,
        risk_path=risk_path,
        preflight_path=capture_dir / "preflight.json",
        job_log_dir=log_dir,
        generated_at=datetime(2026, 9, 29, 1, 0, tzinfo=UTC),
    )

    assert report["market_to_shadow_status"] == acceptance.PASS
    assert report["risk_inclusive_stage1_status"] == acceptance.BLOCKED
    assert report["checks"]["account_risk"]["reason"] == "account_risk_snapshot_missing"
    assert report["propagation"]["all_three_match"] is True
    assert report["propagation"]["snapshot_ids"] == {
        "frozen": snapshot_id,
        "gap": snapshot_id,
        "shadow": snapshot_id,
    }
    assert report["issue_27_overall_status"] == acceptance.DEFERRED


def test_valid_previous_session_risk_allows_stage1_pass(tmp_path):
    capture_dir, gap_path, shadow_dir, risk_path, log_dir, _ = _write_common_artifacts(tmp_path)
    risk_payload = {
        "schema_version": ACCOUNT_RISK_SCHEMA,
        "valid_for_trade_date": TRADE_DATE,
        "observed_through": "2026-09-28",
        "as_of": "2026-09-28T15:30:00+09:00",
        "account_key": "tachibana:default",
        "daily_return": -0.001,
        "month_return": -0.005,
        "source": "reconciled_execution_ledger",
        "pnl_basis": REQUIRED_PNL_BASIS,
        "cash_reconciled": True,
        "positions_reconciled": True,
        "fees_complete": True,
        "reconciliation_status": "complete",
    }
    risk_path.write_text(json.dumps(risk_payload), encoding="utf-8")

    report = acceptance.build_acceptance_report(
        trade_date=TRADE_DATE,
        capture_dir=capture_dir,
        gap_store=gap_path,
        shadow_dir=shadow_dir,
        risk_path=risk_path,
        preflight_path=capture_dir / "preflight.json",
        job_log_dir=log_dir,
        risk_config=_risk_config(),
        generated_at=datetime(2026, 9, 29, 1, 0, tzinfo=UTC),
    )

    assert report["market_to_shadow_status"] == acceptance.PASS
    assert report["checks"]["account_risk"]["status"] == acceptance.PASS
    assert report["checks"]["account_risk"]["gate"]["is_blocked"] is False
    assert report["risk_inclusive_stage1_status"] == acceptance.PASS
    assert report["issue_27_overall_status"] == acceptance.DEFERRED


def test_snapshot_mismatch_fails_market_path(tmp_path):
    capture_dir, gap_path, shadow_dir, risk_path, log_dir, _ = _write_common_artifacts(tmp_path)
    store = GapStore(gap_path)
    store.save(
        TRADE_DATE,
        np.zeros(17),
        np.eye(17),
        metadata={
            "source": "compute_gap_adjusted_distribution",
            "quote_snapshot_id": "wrong-snapshot",
        },
    )

    report = acceptance.build_acceptance_report(
        trade_date=TRADE_DATE,
        capture_dir=capture_dir,
        gap_store=gap_path,
        shadow_dir=shadow_dir,
        risk_path=risk_path,
        preflight_path=capture_dir / "preflight.json",
        job_log_dir=log_dir,
    )

    assert report["checks"]["gap"]["status"] == acceptance.FAIL
    assert report["checks"]["gap"]["reason"] == "gap_snapshot_id_mismatch"
    assert report["market_to_shadow_status"] == acceptance.FAIL
    assert report["risk_inclusive_stage1_status"] == acceptance.FAIL


def test_missing_shadow_only_flag_fails_job_mode_evidence(tmp_path):
    capture_dir, gap_path, shadow_dir, risk_path, log_dir, _ = _write_common_artifacts(tmp_path)
    decision_path = log_dir / "decision_20260929_decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["command"] = ["python", "-m", "leadlag.cli", "decision", "--api-enable"]
    decision_path.write_text(json.dumps(decision), encoding="utf-8")

    report = acceptance.build_acceptance_report(
        trade_date=TRADE_DATE,
        capture_dir=capture_dir,
        gap_store=gap_path,
        shadow_dir=shadow_dir,
        risk_path=risk_path,
        preflight_path=capture_dir / "preflight.json",
        job_log_dir=log_dir,
    )

    assert report["checks"]["jobs"]["status"] == acceptance.FAIL
    assert report["checks"]["jobs"]["reason"] == "decision_phase_not_shadow_only"
    assert report["market_to_shadow_status"] == acceptance.FAIL


def test_preflight_must_be_from_same_trade_date_and_capture_path(tmp_path):
    capture_dir, gap_path, shadow_dir, risk_path, log_dir, _ = _write_common_artifacts(tmp_path)
    preflight_path = capture_dir / "preflight.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    preflight["checked_at"] = "2026-09-28T08:45:00+09:00"
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")

    report = acceptance.build_acceptance_report(
        trade_date=TRADE_DATE,
        capture_dir=capture_dir,
        gap_store=gap_path,
        shadow_dir=shadow_dir,
        risk_path=risk_path,
        preflight_path=preflight_path,
        job_log_dir=log_dir,
    )

    assert report["checks"]["preflight"]["status"] == acceptance.BLOCKED
    assert report["checks"]["preflight"]["reason"] == "preflight_trade_date_mismatch"
    assert report["market_to_shadow_status"] == acceptance.BLOCKED


def test_missing_auth_diagnostics_blocks_market_acceptance(tmp_path):
    capture_dir, gap_path, shadow_dir, risk_path, log_dir, _ = _write_common_artifacts(tmp_path)
    (capture_dir / "auth_diagnostics.jsonl").unlink()

    report = acceptance.build_acceptance_report(
        trade_date=TRADE_DATE,
        capture_dir=capture_dir,
        gap_store=gap_path,
        shadow_dir=shadow_dir,
        risk_path=risk_path,
        preflight_path=capture_dir / "preflight.json",
        job_log_dir=log_dir,
    )

    assert report["checks"]["auth_diagnostics"]["status"] == acceptance.BLOCKED
    assert report["checks"]["auth_diagnostics"]["reason"] == "auth_diagnostics_missing"
    assert report["market_to_shadow_status"] == acceptance.BLOCKED


def test_issue_35_research_input_passes_with_current_valid_adr_bundle(tmp_path):
    capture_dir, gap_path, shadow_dir, risk_path, log_dir, _ = _write_common_artifacts(tmp_path)
    adr_path = _write_adr_bundle(tmp_path)

    report = acceptance.build_acceptance_report(
        trade_date=TRADE_DATE,
        capture_dir=capture_dir,
        gap_store=gap_path,
        shadow_dir=shadow_dir,
        risk_path=risk_path,
        preflight_path=capture_dir / "preflight.json",
        job_log_dir=log_dir,
        adr_bundle=adr_path,
    )

    assert report["checks"]["adr_features"]["status"] == acceptance.PASS
    assert report["checks"]["adr_features"]["latest_trade_date"] == TRADE_DATE
    assert report["issue_35_research_input_status"] == acceptance.PASS


def test_issue_35_research_input_blocks_stale_adr_bundle(tmp_path):
    capture_dir, gap_path, shadow_dir, risk_path, log_dir, _ = _write_common_artifacts(tmp_path)
    adr_path = _write_adr_bundle(tmp_path, trade_date="2026-09-28")

    report = acceptance.build_acceptance_report(
        trade_date=TRADE_DATE,
        capture_dir=capture_dir,
        gap_store=gap_path,
        shadow_dir=shadow_dir,
        risk_path=risk_path,
        preflight_path=capture_dir / "preflight.json",
        job_log_dir=log_dir,
        adr_bundle=adr_path,
    )

    assert report["checks"]["adr_features"]["status"] == acceptance.BLOCKED
    assert report["checks"]["adr_features"]["reason"] == "adr_trade_date_not_current"
    assert report["issue_35_research_input_status"] == acceptance.BLOCKED
