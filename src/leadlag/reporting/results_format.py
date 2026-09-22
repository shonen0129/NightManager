"""Utilities for standardized results directory formatting.

All run artifacts should be written under ``var/results/`` using a
single naming convention: ``YYYYMMDD_HHMMSS_<run_name>``.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from leadlag.config.paths import results as _results_path

RESULTS_FORMAT_VERSION = "v1"
TIMESTAMP_FMT = "%Y%m%d_%H%M%S"


def _atomic_json_write(path: str, payload: Mapping[str, Any]) -> None:
    """Write JSON without exposing a partially written manifest."""
    temporary = f"{path}.{os.getpid()}.tmp"
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(dict(payload), f, ensure_ascii=False, indent=2, default=str)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def get_default_results_root() -> str:
    """Return the project-level canonical results root directory."""
    return str(_results_path())


def _normalize_component(value: str) -> str:
    token = re.sub(r"[^0-9A-Za-z]+", "_", value.strip().lower())
    token = re.sub(r"_+", "_", token).strip("_")
    return token or "run"


def _build_run_dir_name(run_name: str, run_tag: str | None = None) -> str:
    timestamp = datetime.now().strftime(TIMESTAMP_FMT)
    suffix = _normalize_component(run_tag if run_tag is not None else run_name)
    return f"{timestamp}_{suffix}"


def _allocate_unique_path(base_dir: str, run_dir_name: str) -> str:
    """Return a non-conflicting directory path using numeric suffix if needed."""
    candidate = os.path.join(base_dir, run_dir_name)
    if not os.path.exists(candidate):
        return candidate

    n = 2
    while True:
        bumped = os.path.join(base_dir, f"{run_dir_name}_{n:02d}")
        if not os.path.exists(bumped):
            return bumped
        n += 1


def write_run_manifest(
    output_dir: str,
    run_name: str,
    run_tag: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> str:
    """Write standardized run metadata and return the manifest path."""
    payload: dict[str, Any] = {
        "results_format_version": RESULTS_FORMAT_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": run_name,
        "run_tag": run_tag,
        "output_dir": os.path.abspath(output_dir),
    }
    if extra:
        payload.update(dict(extra))

    manifest_path = os.path.join(output_dir, "run_manifest.json")
    _atomic_json_write(manifest_path, payload)
    return manifest_path


def update_run_manifest(
    output_dir: str,
    extra: Mapping[str, Any],
    *,
    manifest_name: str = "run_manifest.json",
) -> str:
    """Atomically merge runtime evidence into an existing run manifest.

    A failed post-submission checkpoint must leave the evidence already
    collected on disk.  The update is therefore best-effort at the call site,
    but never writes a truncated JSON file.
    """
    manifest_path = os.path.join(output_dir, manifest_name)
    payload: dict[str, Any] = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            raise ValueError(f"Run manifest must contain a JSON object: {manifest_path}")
        payload.update(loaded)
    payload.update(dict(extra))
    _atomic_json_write(manifest_path, payload)
    return manifest_path


def create_results_output_dir(
    run_name: str,
    output_root: str | None = None,
    run_tag: str | None = None,
    manifest_extra: Mapping[str, Any] | None = None,
) -> str:
    """Create a standardized output directory and write ``run_manifest.json``."""
    root = os.path.abspath(output_root) if output_root else get_default_results_root()
    os.makedirs(root, exist_ok=True)

    run_dir_name = _build_run_dir_name(run_name=run_name, run_tag=run_tag)
    output_dir = _allocate_unique_path(root, run_dir_name)
    os.makedirs(output_dir, exist_ok=True)

    write_run_manifest(
        output_dir=output_dir,
        run_name=run_name,
        run_tag=run_tag,
        extra=manifest_extra,
    )
    return output_dir
