from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from tools.validation import run_issue27_stage1 as stage1

from leadlag.data.quote_snapshot import freeze_quote_snapshot
from leadlag.data.tickers import JP_TICKERS_WITH_TOPIX

TRADE_DATE = "2026-10-13"


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


def test_stage1_environment_forces_shadow_and_canonical_capture(tmp_path):
    capture_dir = tmp_path / "capture"
    env = stage1.stage1_environment(
        capture_dir,
        {
            "LEADLAG_SHADOW_ONLY": "0",
            "LEADLAG_CAPTURE_0910": "0",
            "LEADLAG_CAPTURE_OUTPUT_DIR": "/wrong/path",
            "UNRELATED": "kept",
        },
    )

    assert env["LEADLAG_SHADOW_ONLY"] == "1"
    assert env["LEADLAG_CAPTURE_0910"] == "1"
    assert env["LEADLAG_CAPTURE_OUTPUT_DIR"] == str(capture_dir.resolve())
    assert env["UNRELATED"] == "kept"


def test_execution_plan_disables_duplicate_capture_for_decision(tmp_path):
    plan = stage1.build_execution_plan(
        capture_dir=tmp_path / "capture",
        report_dir=tmp_path / "report",
        trade_date=TRADE_DATE,
    )

    assert plan["read_only"] is True
    assert plan["controlled_live"] is False
    steps = {step["name"]: step for step in plan["steps"]}
    assert steps["preflight"]["env"]["LEADLAG_SHADOW_ONLY"] == "1"
    assert steps["capture"]["source"] == (
        "registered com.leadlag.microstructure-0910 LaunchAgent"
    )
    assert steps["capture"]["action"] == "wait_for_and_verify_artifact"
    assert "command" not in steps["capture"]
    assert steps["gap_and_v2_shadow"]["env"] == {
        "LEADLAG_SHADOW_ONLY": "1",
        "LEADLAG_CAPTURE_0910": "0",
    }


def test_verify_capture_requires_terminal_and_attempt_evidence(tmp_path):
    capture_dir = tmp_path / "capture"
    capture_dir.mkdir()
    freeze_quote_snapshot(_quote_record(), capture_dir, trade_date=TRADE_DATE)
    path = capture_dir / "capture_20261013.json"

    valid = {
        "run_id": "capture-1",
        "status": "CAPTURED",
        "attempts": [{"attempt": 1, "status": "CAPTURED"}],
    }
    path.write_text(json.dumps(valid), encoding="utf-8")

    evidence = stage1.verify_capture(capture_dir, trade_date=TRADE_DATE)
    assert evidence["run_id"] == "capture-1"
    assert evidence["available_at"] == f"{TRADE_DATE}T09:10:06+09:00"
    assert evidence["available_at_semantics"] == "local_response_receipt"
    assert evidence["observed_count"] == len(JP_TICKERS_WITH_TOPIX)

    for invalid in (
        {**valid, "run_id": ""},
        {**valid, "attempts": []},
        {**valid, "attempts": [{"status": "FAILED"}]},
        {**valid, "status": "FAILED"},
    ):
        path.write_text(json.dumps(invalid), encoding="utf-8")
        with pytest.raises(RuntimeError, match="complete success evidence"):
            stage1.verify_capture(capture_dir, trade_date=TRADE_DATE)


def test_execute_refuses_non_trading_day_before_subprocess(tmp_path, monkeypatch):
    saturday = datetime(
        2026,
        10,
        10,
        9,
        0,
        tzinfo=ZoneInfo("Asia/Tokyo"),
    )
    monkeypatch.setattr(stage1, "_jst_now", lambda: saturday)
    monkeypatch.setattr(stage1, "is_trading_day", lambda _value: False)
    monkeypatch.setattr(
        stage1,
        "_run_command",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("subprocess must not run")
        ),
    )

    with pytest.raises(RuntimeError, match="not a trading day"):
        stage1.execute_stage1(
            capture_dir=tmp_path / "capture",
            report_dir=tmp_path / "report",
        )


def test_load_report_requires_json_object(tmp_path):
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    path = report_dir / "report.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(RuntimeError, match="JSON object"):
        stage1._load_report(report_dir)
