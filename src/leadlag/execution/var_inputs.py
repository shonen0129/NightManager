"""Snapshot acquisition and content identities for a single VaR calculation."""
from __future__ import annotations

import hashlib
import json
import pickle
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any


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
        # WAL/SHM sidecars contain committed state that may not yet have been
        # checkpointed into the SQLite main file.
        if path.suffix in {".sqlite", ".sqlite3", ".db"}:
            candidates.extend(
                path.with_name(path.name + suffix)
                for suffix in ("-wal", "-shm")
                if path.with_name(path.name + suffix).exists()
            )
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
            elif path.is_dir():
                snapshot_path = Path(temporary.name) / "gap"
                _copy_directory_with_deadline(path, snapshot_path, remaining)
            else:
                snapshot_path = Path(temporary.name) / path.name
                _copy_file_with_deadline(path, snapshot_path, remaining)
            companion = _companion_history(path)
            if companion is not None:
                _copy_file_with_deadline(
                    companion, snapshot_path.parent / companion.name, remaining,
                )
            remaining()
            after = _file_manifest_fingerprint(path, timeout=remaining())
            if before != after:
                temporary.cleanup()
                continue
            snapshot_fingerprint = _file_manifest_fingerprint(
                snapshot_path, timeout=remaining()
            )
            if path.is_dir() and snapshot_fingerprint != after:
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
