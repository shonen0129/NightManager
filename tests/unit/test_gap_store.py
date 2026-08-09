"""Tests for ``leadlag.data.gap_store``."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from leadlag.data.gap_store import GapStore, is_gap_store_path


def test_gap_store_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "gap.sqlite"
        store = GapStore(path)
        mu = np.array([1.0, 2.0, 3.0])
        omega = np.eye(3)

        store.put("2026-08-10", "mu", mu)
        store.put("2026-08-10", "omega", omega)

        loaded_mu = store.get("2026-08-10", "mu")
        loaded_omega = store.get("2026-08-10", "omega")

        assert loaded_mu is not None
        assert np.allclose(loaded_mu, mu)
        assert loaded_omega is not None
        assert np.allclose(loaded_omega, omega)


def test_gap_store_horizon():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "gap.sqlite"
        store = GapStore(path)
        arr = np.array([0.1, 0.2])

        store.put("2026-08-10", "mu", arr, horizon=3)
        assert store.exists("2026-08-10", "mu", horizon=3)
        assert not store.exists("2026-08-10", "mu", horizon=5)

        loaded = store.get("2026-08-10", "mu", horizon=3)
        assert loaded is not None
        assert np.allclose(loaded, arr)


def test_gap_store_replace_default_horizon():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "gap.sqlite"
        store = GapStore(path)

        store.put("2026-08-10", "mu", np.array([1.0]))
        store.put("2026-08-10", "mu", np.array([2.0]))

        loaded = store.get("2026-08-10", "mu")
        assert loaded is not None
        assert np.allclose(loaded, np.array([2.0]))

        # Ensure there is exactly one row (INSERT OR REPLACE worked).
        import sqlite3

        conn = sqlite3.connect(str(path))
        count = conn.execute("SELECT COUNT(*) FROM gap_matrices").fetchone()[0]
        conn.close()
        assert count == 1


def test_gap_store_missing():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "gap.sqlite"
        store = GapStore(path)
        assert store.get("2026-08-10", "mu") is None


def test_is_gap_store_path():
    assert is_gap_store_path("foo.sqlite")
    assert is_gap_store_path("foo.sqlite3")
    assert is_gap_store_path("foo.db")
    assert not is_gap_store_path("foo.npy")
    assert not is_gap_store_path("/some/dir")
