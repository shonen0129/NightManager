from __future__ import annotations

import json
import plistlib
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from tools.validation import collect_0910_microstructure as capture
from tools.validation.collect_0910_microstructure import _load_stored_snapshot_summary


def _quote_rows(
    observed_at: str,
    request_started_at: str = "2026-09-28T09:10:02+09:00",
) -> list[dict]:
    return [
        {
            "ticker": ticker,
            "status": "OBSERVED",
            "request_started_at": request_started_at,
            "response_received_at": observed_at,
            "observed_at": observed_at,
            "timestamp_source": "local_response_receipt",
            "source": "tachibana:CLMMfdsGetMarketPrice",
            "bid_price_1": 100.0,
            "ask_price_1": 102.0,
            "quote_mid_price": 101.0,
            "valid_two_sided_quote": True,
        }
        for ticker in capture.CAPTURE_TICKERS
    ]


def test_retired_tachibana_endpoint_is_rejected_before_request():
    current = "https://kabuka.e-shiten.jp/e_api_v4r10/"
    assert capture._validated_api_url(current) == current

    with pytest.raises(ValueError, match="retired v4r9"):
        capture._validated_api_url("https://kabuka.e-shiten.jp/e_api_v4r9/")


def test_stored_snapshot_summary_preserves_true_0910_evidence(tmp_path):
    path = tmp_path / "quote_snapshots.jsonl"
    records = [
        {
            "status": "OBSERVED_OUTSIDE_0910_WINDOW",
            "window_valid": False,
            "observed_at": "2026-09-23T02:58:00+09:00",
            "observed_count": len(capture.CAPTURE_TICKERS),
            "lob_count": len(capture.CAPTURE_TICKERS),
            "spread_bps_summary": {"count": 17, "median": 50.0},
        },
        {
            "status": "OBSERVED",
            "window_valid": True,
            "observed_at": "2026-09-24T09:10:03+09:00",
            "observed_count": len(capture.CAPTURE_TICKERS),
            "lob_count": len(capture.CAPTURE_TICKERS),
            "spread_bps_summary": {"count": 17, "median": 12.0},
        },
    ]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    summary = _load_stored_snapshot_summary(tmp_path)

    assert summary is not None
    assert summary["stored_capture_count"] == 2
    assert summary["true_0910_capture_count"] == 1
    assert summary["true_0910_observed"] is True
    assert summary["latest_observed_at"] == "2026-09-24T09:10:03+09:00"
    assert summary["spread_bps_summary"]["median"] == 12.0


def test_forward_capture_window_is_one_sided_and_limited_to_30_seconds():
    jst = ZoneInfo("Asia/Tokyo")
    assert capture._forward_capture_window(datetime(2026, 9, 24, 9, 10, 0, tzinfo=jst))
    assert capture._forward_capture_window(datetime(2026, 9, 24, 9, 10, 30, tzinfo=jst))
    assert not capture._forward_capture_window(datetime(2026, 9, 24, 9, 9, 59, tzinfo=jst))
    assert not capture._forward_capture_window(datetime(2026, 9, 24, 9, 10, 31, tzinfo=jst))


def test_capture_only_outside_window_records_miss_without_api_call(tmp_path, monkeypatch):
    from leadlag.core import market_calendar

    now = datetime(2026, 9, 28, 9, 10, 31, tzinfo=ZoneInfo("Asia/Tokyo"))
    monkeypatch.setattr(market_calendar, "is_trading_day", lambda _day: True)
    monkeypatch.setattr(capture, "_jst_now", lambda: now)
    monkeypatch.setattr(
        capture,
        "_collect_live_quotes",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("API must not be called")),
    )

    exit_code = capture._run_capture_only(output_dir=tmp_path)

    run = json.loads((tmp_path / "capture_runs.jsonl").read_text(encoding="utf-8"))
    daily = json.loads((tmp_path / "capture_20260928.json").read_text(encoding="utf-8"))
    assert exit_code == 78
    assert run["status"] == "OUTSIDE_CAPTURE_WINDOW"
    assert daily["status"] == "OUTSIDE_CAPTURE_WINDOW"
    assert not (tmp_path / "quote_snapshots.jsonl").exists()


def test_capture_only_persists_request_and_response_times(tmp_path, monkeypatch):
    from leadlag.core import market_calendar

    monkeypatch.setattr(market_calendar, "is_trading_day", lambda _day: True)
    request_at = datetime(2026, 9, 28, 9, 10, 2, tzinfo=ZoneInfo("Asia/Tokyo"))
    response_at = datetime(2026, 9, 28, 9, 10, 7, tzinfo=ZoneInfo("Asia/Tokyo"))
    times = iter((request_at, request_at))
    monkeypatch.setattr(capture, "_jst_now", lambda: next(times))
    monkeypatch.setattr(
        capture,
        "_collect_live_quotes",
        lambda _tickers, started: {
            "status": "OBSERVED",
            "request_started_at": started.isoformat(),
            "response_received_at": response_at.isoformat(),
            "observed_at": response_at.isoformat(),
            "timestamp_source": "local_response_receipt",
            "source": "tachibana:CLMMfdsGetMarketPrice",
            "observed_count": len(capture.CAPTURE_TICKERS),
            "lob_count": len(capture.CAPTURE_TICKERS),
            "_login_diagnostics": {
                "http_status": 200,
                "response_parsed": True,
                "sResultCode": {"state": "value", "type": "str", "value": "0"},
                "sKinsyouhouMidokuFlg": {"state": "value", "type": "str", "value": "0"},
                "virtual_urls": {
                    "sUrlRequest": {
                        "state": "nonempty",
                        "decrypt_attempted": True,
                        "decrypt_succeeded": True,
                    }
                },
                "login_success": True,
                "stopped_at": None,
            },
            "full_five_level_depth_count": 4,
            "spread_bps_summary": {"count": 15, "median": 18.0},
            "rows": _quote_rows(response_at.isoformat()),
        },
    )

    exit_code = capture._run_capture_only(output_dir=tmp_path)

    snapshot = json.loads((tmp_path / "quote_snapshots.jsonl").read_text(encoding="utf-8"))
    daily = json.loads((tmp_path / "capture_20260928.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert snapshot["window_valid"] is True
    assert snapshot["request_started_at"] == request_at.isoformat()
    assert snapshot["response_received_at"] == response_at.isoformat()
    assert daily["status"] == "CAPTURED"
    assert daily["attempts"][0]["two_sided_quote_count"] == len(capture.CAPTURE_TICKERS)
    assert daily["attempts"][0]["full_five_level_depth_count"] == 4
    assert (tmp_path / "frozen_20260928.json").exists()

    auth = json.loads((tmp_path / "auth_diagnostics.jsonl").read_text(encoding="utf-8"))
    assert auth["schema_version"] == "tachibana-login-diagnostics-v1"
    assert auth["capture_status"] == "CAPTURED"
    assert auth["diagnostics"]["login_success"] is True


def test_capture_only_retries_transient_api_error_inside_window(tmp_path, monkeypatch):
    from leadlag.core import market_calendar

    monkeypatch.setattr(market_calendar, "is_trading_day", lambda _day: True)
    jst = ZoneInfo("Asia/Tokyo")
    now_values = iter(
        (
            datetime(2026, 9, 28, 9, 10, 2, tzinfo=jst),
            datetime(2026, 9, 28, 9, 10, 2, tzinfo=jst),
            datetime(2026, 9, 28, 9, 10, 3, tzinfo=jst),
            datetime(2026, 9, 28, 9, 10, 4, tzinfo=jst),
        )
    )
    monkeypatch.setattr(capture, "_jst_now", lambda: next(now_values))
    monkeypatch.setattr(capture.time_module, "sleep", lambda _seconds: None)
    calls = 0

    def collect(_tickers, started):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError(
                "404 Client Error: Not Found for url: "
                "https://kabuka.e-shiten.jp/e_api_v4r10/auth/?"
                "%7B%22sAuthId%22%3A%22SensitiveLoginTokenXYZ%22%7D"
            )
        return {
            "status": "OBSERVED",
            "request_started_at": started.isoformat(),
            "response_received_at": "2026-09-28T09:10:10+09:00",
            "observed_at": "2026-09-28T09:10:10+09:00",
            "timestamp_source": "local_response_receipt",
            "source": "tachibana:CLMMfdsGetMarketPrice",
            "observed_count": len(capture.CAPTURE_TICKERS),
            "lob_count": len(capture.CAPTURE_TICKERS),
            "full_five_level_depth_count": len(capture.CAPTURE_TICKERS),
            "spread_bps_summary": {"count": len(capture.CAPTURE_TICKERS), "median": 10.0},
            "rows": _quote_rows(
                "2026-09-28T09:10:10+09:00",
                request_started_at="2026-09-28T09:10:04+09:00",
            ),
        }

    monkeypatch.setattr(capture, "_collect_live_quotes", collect)

    exit_code = capture._run_capture_only(output_dir=tmp_path, attempts=2)

    runs = [json.loads(line) for line in (tmp_path / "capture_runs.jsonl").read_text().splitlines()]
    assert exit_code == 0
    assert calls == 2
    assert runs[0]["status"] == "ERROR"
    assert "SensitiveLoginTokenXYZ" not in runs[0]["error"]
    assert "[redacted]" in runs[0]["error"]
    assert runs[1]["status"] == "CAPTURED"


def test_capture_only_persists_failed_login_diagnostics(tmp_path, monkeypatch):
    from leadlag.core import market_calendar

    now = datetime(2026, 9, 28, 9, 10, 2, tzinfo=ZoneInfo("Asia/Tokyo"))
    monkeypatch.setattr(market_calendar, "is_trading_day", lambda _day: True)
    monkeypatch.setattr(capture, "_jst_now", lambda: now)

    diagnostics = {
        "http_status": 200,
        "response_parsed": True,
        "p_errno": {"state": "absent", "type": None, "value": None},
        "sResultCode": {"state": "value", "type": "str", "value": "0"},
        "sKinsyouhouMidokuFlg": {"state": "value", "type": "str", "value": "1"},
        "virtual_urls": {
            "sUrlRequest": {
                "state": "empty",
                "decrypt_attempted": False,
                "decrypt_succeeded": False,
            }
        },
        "login_success": False,
        "stopped_at": "missing_virtual_url:sUrlRequest",
    }

    def fail_collect(*_args, **_kwargs):
        exc = ValueError("missing virtual URL")
        setattr(exc, "_tachibana_login_diagnostics", diagnostics)
        raise exc

    monkeypatch.setattr(capture, "_collect_live_quotes", fail_collect)

    assert capture._run_capture_only(output_dir=tmp_path, attempts=1) == 1
    auth = json.loads((tmp_path / "auth_diagnostics.jsonl").read_text(encoding="utf-8"))
    assert auth["capture_status"] == "ERROR"
    assert auth["diagnostics"]["sKinsyouhouMidokuFlg"]["value"] == "1"
    assert auth["diagnostics"]["virtual_urls"]["sUrlRequest"]["state"] == "empty"
    assert "sAuthId" not in json.dumps(auth)
    assert "https://" not in json.dumps(auth)


def test_independent_launchagent_runs_weekdays_at_0910_without_run_at_load():
    plist_path = capture.ROOT / "scripts/batch/com.leadlag.microstructure-0910.plist"
    with plist_path.open("rb") as handle:
        config = plistlib.load(handle)

    schedule = config["StartCalendarInterval"]
    assert config["Label"] == "com.leadlag.microstructure-0910"
    assert [(item["Weekday"], item["Hour"], item["Minute"]) for item in schedule] == [
        (weekday, 9, 10) for weekday in range(1, 6)
    ]
    assert config["RunAtLoad"] is False
    assert config["KeepAlive"] is False
    assert config["ProgramArguments"][1].endswith(
        "/scripts/batch/run_0910_microstructure_capture.sh"
    )
    assert config["EnvironmentVariables"]["LEADLAG_CAPTURE_OUTPUT_DIR"].endswith(
        "/var/shadow_runs/ml_overlay_value/microstructure"
    )
