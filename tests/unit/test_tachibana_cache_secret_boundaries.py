"""Synthetic-only regressions for best-effort Tachibana cache boundaries."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from leadlag.broker.tachibana import session_cache

SECRET = "SYNTHETIC_CACHE_SECRET_34"
OPERATIONS = {
    "save_session": ("_session_cache_path", "save_session_cache", ({},), "save session"),
    "load_session": ("_session_cache_path", "load_session_cache", (), "load session"),
    "clear_session": ("_session_cache_path", "clear_session_cache", (), "clear session"),
    "save_open": ("_opens_cache_path", "save_open_prices_cache", ({}, None, "20261010"), "save open prices"),
    "load_open": ("_opens_cache_path", "load_open_prices_cache", ("20261010",), "load open prices"),
    "save_current": ("_current_prices_cache_path", "save_current_prices_cache", ({}, None, "20261010"), "save current prices"),
    "load_current": ("_current_prices_cache_path", "load_current_prices_cache", ("20261010",), "load current prices"),
}


def fail_with_secret(*args, **kwargs):
    raise OSError(f"synthetic failure: {SECRET}")


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("failure_stage", ["path", "io"])
def test_cache_boundary_discards_raw_errors(operation, failure_stage, tmp_path, monkeypatch, caplog):
    path = tmp_path / f"{SECRET}.cache"
    path.write_text("{}")
    path_function, function, args, action = OPERATIONS[operation]
    monkeypatch.setattr(session_cache, path_function, lambda *args: path)
    if failure_stage == "path":
        monkeypatch.setattr(session_cache, path_function, fail_with_secret)
    elif operation == "save_session":
        monkeypatch.setattr(session_cache, "_atomic_json_write", fail_with_secret)
    elif operation.startswith("save_"):
        monkeypatch.setattr(session_cache, "_atomic_csv_write", fail_with_secret)
    elif operation == "load_session":
        monkeypatch.setattr(session_cache.json, "load", fail_with_secret)
    elif operation.startswith("load_"):
        monkeypatch.setattr(pd, "read_csv", fail_with_secret)
    else:
        monkeypatch.setattr(Path, "unlink", fail_with_secret)
    with caplog.at_level(logging.DEBUG):
        assert getattr(session_cache, function)(*args) is None
    assert SECRET not in caplog.text
    assert f"Failed to {action} cache" in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


def test_malformed_session_timestamp_is_rejected_without_logging_its_value(tmp_path, monkeypatch, caplog):
    path = tmp_path / "session.json"
    path.write_text(json.dumps({"saved_at": f"https://invalid.test/?token={SECRET}"}))
    monkeypatch.setattr(session_cache, "_session_cache_path", lambda: path)
    with caplog.at_level(logging.DEBUG):
        assert session_cache.load_session_cache() is None
    assert not path.exists()
    assert SECRET not in caplog.text
    assert "Failed to load session cache" in caplog.text


def test_cache_boundary_success_logs_exclude_session_values_and_paths(tmp_path, monkeypatch, caplog):
    path = tmp_path / f"{SECRET}.json"
    monkeypatch.setattr(session_cache, "_session_cache_path", lambda: path)
    urls = {key: f"https://invalid.test/{SECRET}/{key}" for key in session_cache._REQUIRED_URL_KEYS}
    state = {"decrypted_urls": urls, "p_no": 3, "logged_in": True}
    with caplog.at_level(logging.DEBUG):
        session_cache.save_session_cache(state)
        loaded = session_cache.load_session_cache()
    assert loaded is not None and loaded["decrypted_urls"] == urls
    assert datetime.fromisoformat(loaded["saved_at"]) <= datetime.now(UTC)
    assert path.stat().st_mode & 0o077 == 0
    assert SECRET in path.read_text()
    assert SECRET not in caplog.text
