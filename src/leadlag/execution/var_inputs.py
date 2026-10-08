"""Snapshot acquisition and content identities for a single VaR calculation."""
from __future__ import annotations

import hashlib
import json
import pickle
import shutil
import sqlite3
import tempfile
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd


def _companion_history(path: Path) -> Path | None:
    """Match the PIT reader's canonical sibling history lookup."""
    # Directory snapshots already contain an in-directory canonical file.
    if path.is_dir() and (path / "full_history_diagnostics.csv").is_file():
        return None
    candidate = path.parent / "full_history_diagnostics.csv"
    return candidate if candidate.is_file() else None

def _file_manifest_fingerprint(path: Path | None, *, timeout: float | None = None) -> str:
    """Fingerprint the contents of a model/data input path.

    SQLite WAL databases can receive a committed update without changing the
    main file's mtime.  Content digests therefore include the database's WAL
    and SHM sidecars when present, as well as every file in a directory.
    """
    if path is None:
        return "none"
    deadline = time.monotonic() + timeout if timeout is not None else None

    def remaining() -> None:
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(f"fingerprint exceeded {timeout}s deadline")

    path = Path(path)
    remaining()
    if not path.exists():
        return f"missing:{path}"
    digest = hashlib.sha256()

    def add_file(label: str, file_path: Path) -> None:
        remaining()
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                remaining()
                digest.update(chunk)
        digest.update(b"\0")

    if path.is_file():
        candidates = [path]
        # WAL sidecar contains committed state that may not yet have been
        # checkpointed into the SQLite main file. Exclude volatile -shm and empty -wal.
        if path.suffix in {".sqlite", ".sqlite3", ".db"}:
            wal_sidecar = path.with_name(path.name + "-wal")
            if wal_sidecar.exists() and wal_sidecar.stat().st_size > 0:
                candidates.append(wal_sidecar)
        for candidate in candidates:
            if candidate.exists():
                add_file(candidate.name, candidate)
    else:
        for child in sorted(
            p
            for p in path.rglob("*")
            if p.is_file() and p.suffix != ".pyc" and "__pycache__" not in p.parts
        ):
            add_file(str(child.relative_to(path)), child)
    companion = _companion_history(path)
    if companion is not None:
        add_file("../full_history_diagnostics.csv", companion)
    remaining()
    return digest.hexdigest()[:16]


def _copy_file_with_deadline(
    source: Path,
    target: Path,
    remaining: Any,
) -> None:
    """Copy one file while checking the caller's absolute deadline per chunk."""
    remaining()
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as source_handle, target.open("wb") as target_handle:
        for chunk in iter(lambda: source_handle.read(1024 * 1024), b""):
            remaining()
            target_handle.write(chunk)
    remaining()
    shutil.copystat(source, target, follow_symlinks=True)


def _copy_directory_with_deadline(
    source: Path,
    target: Path,
    remaining: Any,
) -> None:
    """Copy a directory without an uninterruptible ``copytree`` call."""
    remaining()
    target.mkdir(parents=True, exist_ok=False)
    for child in sorted(source.rglob("*")):
        remaining()
        relative = child.relative_to(source)
        destination = target / relative
        if child.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        elif child.is_file():
            _copy_file_with_deadline(child, destination, remaining)
    remaining()


def _filter_gap_snapshot_after(
    snapshot_path: Path,
    max_trade_date: str | date | datetime | Any,
    remaining: Any,
) -> None:
    """Remove future-dated gap rows from a private SQLite snapshot."""
    cutoff = pd.Timestamp(max_trade_date).normalize().strftime("%Y-%m-%d")
    with sqlite3.connect(str(snapshot_path), timeout=max(0.01, remaining() or 0.01)) as conn:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if "gap_matrices" not in tables:
            return
        rows = conn.execute(
            "SELECT cache_key FROM gap_matrices WHERE substr(trade_date, 1, 10) > ?",
            (cutoff,),
        ).fetchall()
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "DELETE FROM gap_matrices WHERE substr(trade_date, 1, 10) > ?",
            (cutoff,),
        )
        if "cache_store" in tables:
            for (cache_key,) in rows:
                remaining()
                conn.execute("DELETE FROM cache_store WHERE key = ?", (cache_key,))
    conn.commit()
    remaining()


def _cutoff_gap_snapshot_fingerprint(
    snapshot_path: Path,
    max_trade_date: str | date | datetime | Any,
    remaining: Any,
) -> str:
    """Fingerprint only gap records and PIT diagnostics consumed by this replay."""
    cutoff = pd.Timestamp(max_trade_date).normalize().strftime("%Y-%m-%d")
    digest = hashlib.sha256()
    with sqlite3.connect(str(snapshot_path), timeout=max(0.01, remaining() or 0.01)) as conn:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if "gap_matrices" not in tables or "cache_store" not in tables:
            return _file_manifest_fingerprint(snapshot_path)
        rows = conn.execute(
            """
            SELECT trade_date, matrix_type, horizon, cache_key
            FROM gap_matrices
            WHERE substr(trade_date, 1, 10) <= ?
            ORDER BY trade_date, matrix_type, horizon, cache_key
            """,
            (cutoff,),
        ).fetchall()
        for trade_date, matrix_type, horizon, cache_key in rows:
            remaining()
            cached = conn.execute(
                "SELECT value FROM cache_store WHERE key = ?", (cache_key,)
            ).fetchone()
            if cached is None:
                raise ValueError(f"Gap snapshot index points to a missing cache value: {cache_key}")
            for value in (trade_date, matrix_type, horizon, cache_key):
                digest.update(str(value).encode("utf-8"))
                digest.update(b"\0")
            digest.update(bytes(cached[0]))
            digest.update(b"\0")
    companion = _companion_history(snapshot_path)
    if companion is not None:
        remaining()
        with companion.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                remaining()
                digest.update(chunk)
    remaining()
    return digest.hexdigest()[:16]


def _copy_companion_history(
    source: Path,
    target: Path,
    *,
    remaining: Any,
    max_trade_date: str | date | datetime | Any | None,
) -> None:
    """Copy the PIT diagnostics companion, optionally ending at the risk cutoff."""
    if max_trade_date is None:
        _copy_file_with_deadline(source, target, remaining)
        return
    remaining()
    frame = pd.read_csv(source)
    if "trade_date" not in frame.columns:
        raise ValueError(f"PIT diagnostics lacks trade_date: {source}")
    cutoff = pd.Timestamp(max_trade_date).normalize()

    def _normalize(value: Any) -> pd.Timestamp:
        parsed = pd.Timestamp(value)
        if parsed.tzinfo is not None:
            parsed = parsed.tz_convert("Asia/Tokyo").tz_localize(None)
        return parsed.normalize()

    dates = frame["trade_date"].map(_normalize)
    frame = frame.loc[dates <= cutoff].copy()
    remaining()
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    remaining()


def _active_overlay_artifact_fingerprint(path: Path | None) -> str:
    """Fingerprint only the version selected by an overlay ``CURRENT`` pointer.

    Staging directories and inactive versions are intentionally excluded: they
    cannot affect the model used by a VaR/ES backtest and must not invalidate a
    return cache. The selected version id is included even if two versions
    happen to contain identical bytes.
    """
    if path is None:
        return "none"
    path = Path(path)
    if not path.exists():
        return f"missing:{path}"
    if path.is_file():
        return _file_manifest_fingerprint(path)

    current_path = path / "CURRENT"
    if not current_path.exists():
        # The overlay loader rejects this legacy layout. Keep its two relevant
        # files visible in the key so a config repair cannot reuse this cache.
        legacy_files = [candidate for candidate in (path / "model.pkl", path / "metadata.json") if candidate.exists()]
        if not legacy_files:
            return f"missing-current:{path}"
        digest = hashlib.sha256(b"legacy-root-overlay\0")
        for candidate in legacy_files:
            digest.update(candidate.name.encode("utf-8"))
            digest.update(candidate.read_bytes())
        return digest.hexdigest()[:16]
    try:
        version = current_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        return f"unreadable-current:{path}:{type(exc).__name__}"
    if not version or version in {".", ".."} or Path(version).name != version:
        return f"invalid-current:{path}:{version!r}"
    active_dir = path / "versions" / version
    if not active_dir.is_dir():
        return f"missing-active-version:{path}:{version}"
    return f"active:{version}:{_file_manifest_fingerprint(active_dir)}"


def _snapshot_gap_input(
    path: Path | None,
    *,
    timeout: float | None = None,
    max_trade_date: str | date | datetime | Any | None = None,
) -> tuple[Path | None, str, tempfile.TemporaryDirectory[str] | None]:
    """Create one immutable gap-input snapshot and its cache identity.

    A risk cache key must describe the exact input consumed by its backtest.
    SQLite's backup API takes a transactionally consistent copy, including
    committed WAL state.  Directory-backed ``.npy`` inputs are copied only
    after a before/after content fingerprint agrees; otherwise the copy is
    retried so a writer cannot publish a mixed generation during the copy.
    The temporary-directory object is returned to keep the snapshot alive for
    the duration of the caller (including a timeout worker thread).
    """
    if path is None:
        return None, "none", None
    path = Path(path)
    if not path.exists():
        return None, _file_manifest_fingerprint(path), None

    deadline = time.monotonic() + timeout if timeout is not None else None

    def remaining() -> float | None:
        if deadline is None:
            return None
        left = deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError(f"gap-input snapshot exceeded {timeout}s deadline")
        return left

    last_error: Exception | None = None
    for _attempt in range(3):
        temporary = tempfile.TemporaryDirectory(prefix="leadlag-gap-snapshot-")
        try:
            left = remaining()
            before = _file_manifest_fingerprint(path, timeout=remaining())
            if path.is_file() and path.suffix.lower() in {".sqlite", ".sqlite3", ".db"}:
                snapshot_path = Path(temporary.name) / "gap.sqlite"
                connect_timeout = 30.0 if left is None else max(0.01, left)
                source = sqlite3.connect(str(path), timeout=connect_timeout)
                target = sqlite3.connect(str(snapshot_path), timeout=connect_timeout)
                try:
                    # Small pages plus a progress callback bound both the
                    # copy and SQLite's busy waits to the caller's deadline.
                    def backup_progress(_status: int, _remaining: int, _total: int) -> None:
                        remaining()

                    source.backup(
                        target,
                        pages=16,
                        progress=backup_progress,
                        sleep=0.01,
                    )
                finally:
                    target.close()
                    source.close()
                if max_trade_date is not None:
                    _filter_gap_snapshot_after(snapshot_path, max_trade_date, remaining)
            elif path.is_dir():
                snapshot_path = Path(temporary.name) / "gap"
                _copy_directory_with_deadline(path, snapshot_path, remaining)
            else:
                snapshot_path = Path(temporary.name) / path.name
                _copy_file_with_deadline(path, snapshot_path, remaining)
            companion = _companion_history(path)
            if companion is not None:
                _copy_companion_history(
                    companion,
                    snapshot_path.parent / companion.name,
                    remaining=remaining,
                    max_trade_date=max_trade_date,
                )
            remaining()
            after = _file_manifest_fingerprint(path, timeout=remaining())
            if before != after:
                temporary.cleanup()
                continue
            if max_trade_date is not None and path.is_file() and path.suffix.lower() in {
                ".sqlite", ".sqlite3", ".db"
            }:
                snapshot_fingerprint = _cutoff_gap_snapshot_fingerprint(
                    snapshot_path,
                    max_trade_date,
                    remaining,
                )
            else:
                snapshot_fingerprint = _file_manifest_fingerprint(
                    snapshot_path, timeout=remaining()
                )
            if path.is_dir() and max_trade_date is None and snapshot_fingerprint != after:
                # A directory copy may have crossed an atomic publication
                # boundary even when the source is stable again by the final
                # check.  Retry until the copied bytes match one source
                # generation exactly.
                temporary.cleanup()
                continue
            return snapshot_path, snapshot_fingerprint, temporary
        except TimeoutError:
            temporary.cleanup()
            raise
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            temporary.cleanup()
    detail = f": {last_error}" if last_error is not None else ""
    raise RuntimeError(f"Could not obtain a stable gap-input snapshot for {path}{detail}")


def _overlay_model_fingerprint(model: Any | None) -> str:
    """Identify the exact overlay object selected for one risk calculation."""
    if model is None:
        return "none"
    metadata = getattr(model, "metadata", None)
    if isinstance(metadata, dict) and metadata.get("model_sha256"):
        identity = {
            key: metadata.get(key)
            for key in (
                "artifact_version",
                "model_sha256",
                "metadata_version",
                "train_end",
                "data_hash",
                "config_hash",
            )
        }
        return "object:" + hashlib.sha256(
            json.dumps(identity, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:16]
    return "object:" + hashlib.sha256(pickle.dumps(model)).hexdigest()[:16]
