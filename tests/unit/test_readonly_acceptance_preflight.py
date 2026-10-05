from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from tools.validation import check_readonly_acceptance_preflight as preflight


def test_safe_api_metadata_keeps_only_endpoint_identity():
    metadata = preflight._safe_api_metadata(
        "https://kabuka.e-shiten.jp/e_api_v4r10/?secret=query"
    )

    assert metadata == {
        "valid": True,
        "scheme": "https",
        "host": "kabuka.e-shiten.jp",
        "path": "/e_api_v4r10",
        "version": "v4r10",
    }
    assert "secret" not in json.dumps(metadata)


def test_safe_api_metadata_rejects_retired_version():
    with pytest.raises(ValueError, match="retired v4r9"):
        preflight._safe_api_metadata(
            "https://kabuka.e-shiten.jp/e_api_v4r9/"
        )


def test_build_preflight_ready_without_persisting_secrets(tmp_path):
    env = {
        "TACHIBANA_API_URL": "https://kabuka.e-shiten.jp/e_api_v4r10/",
        "TACHIBANA_AUTH_ID": "SensitiveAuthId",
        "TACHIBANA_SECOND_PASSWORD": "SensitivePassword",
        "TACHIBANA_PRIVATE_KEY_PATH": "/secret/private.pem",
        "LEADLAG_CAPTURE_OUTPUT_DIR": str(tmp_path / "capture"),
        "LEADLAG_SHADOW_ONLY": "1",
        "LEADLAG_CAPTURE_0910": "1",
    }
    payload = preflight.build_preflight(
        env=env,
        git_state={"sha": "abc123", "dirty": False, "changed_path_count": 0},
        scheduler={
            "status": "REGISTERED",
            "path": "/tmp/com.leadlag.microstructure-0910.plist",
            "label": "com.leadlag.microstructure-0910",
            "capture_output_dir": str(tmp_path / "capture"),
        },
        checked_at=datetime(2026, 10, 6, 8, 45, tzinfo=ZoneInfo("Asia/Tokyo")),
    )

    assert payload["status"] == "READY"
    assert payload["blocking_checks"] == []
    assert payload["checks"]["git_clean"] is True
    assert payload["checks"]["api_v4r10"] is True
    assert payload["checks"]["scheduler_registered"] is True
    assert payload["checks"]["scheduler_capture_output_matches"] is True
    assert payload["checks"]["shadow_only_enabled"] is True
    assert payload["checks"]["capture_0910_enabled"] is True
    assert payload["capture_output_dir"] == str((tmp_path / "capture").resolve())

    serialized = json.dumps(payload)
    assert "SensitiveAuthId" not in serialized
    assert "SensitivePassword" not in serialized
    assert "/secret/private.pem" not in serialized


def test_build_preflight_blocks_dirty_or_unregistered_state():
    payload = preflight.build_preflight(
        env={
            "TACHIBANA_API_URL": "https://kabuka.e-shiten.jp/e_api_v4r10/",
            "LEADLAG_SHADOW_ONLY": "1",
        },
        git_state={"sha": "abc123", "dirty": True, "changed_path_count": 2},
        scheduler={
            "status": "PLIST_PRESENT_NOT_REGISTERED",
            "path": "/tmp/com.leadlag.microstructure-0910.plist",
            "label": "com.leadlag.microstructure-0910",
        },
        checked_at=datetime(2026, 10, 6, 8, 45, tzinfo=ZoneInfo("Asia/Tokyo")),
    )

    assert payload["status"] == "BLOCKED"
    assert set(payload["blocking_checks"]) == {"git_clean", "scheduler_registered"}
