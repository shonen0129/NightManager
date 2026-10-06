"""Canonical project path resolution.

All runtime outputs (results, live, artifacts, logs, shadow_runs,
market_data) are consolidated under ``var/`` at the project root.
Callers should prefer the helpers in this module to constructing path
strings manually, which is the root cause of ``results/`` / ``live/``
scattering described in ADR-0006.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def project_root() -> Path:
    """Resolve the deployment root, independently of installed package location.

    Installed wheels require LEADLAG_RUNTIME_ROOT. A source checkout defaults
    to its own root; no unrelated directory is searched for runtime inputs.
    Set the environment once, before importing the application.
    """
    configured = os.environ.get("LEADLAG_RUNTIME_ROOT")
    if configured is not None:
        root = Path(configured).expanduser()
        if not root.is_absolute() or not root.is_dir():
            raise ValueError("LEADLAG_RUNTIME_ROOT must be an existing absolute directory")
        return root.resolve()
    source = Path(__file__).resolve()
    root = source.parents[3]
    if source.parent.parent.parent.name == "src" and (root / "pyproject.toml").is_file():
        return root
    raise RuntimeError("Installed leadlag requires LEADLAG_RUNTIME_ROOT")


@lru_cache(maxsize=1)
def var_dir() -> Path:
    """Return the canonical ``var/`` directory, creating it if needed."""
    p = project_root() / "var"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _sub_dir(name: str, *parts: str | Path) -> Path:
    p = var_dir() / name
    if parts:
        p = p / Path(*parts)
    return p


def results(*parts: str | Path) -> Path:
    """Return a path under ``var/results/``."""
    return _sub_dir("results", *parts)


def live(*parts: str | Path) -> Path:
    """Return a path under ``var/live/``."""
    return _sub_dir("live", *parts)


def artifacts(*parts: str | Path) -> Path:
    """Return a path under ``var/artifacts/``."""
    return _sub_dir("artifacts", *parts)


def logs(*parts: str | Path) -> Path:
    """Return a path under ``var/logs/``."""
    return _sub_dir("logs", *parts)


def shadow_runs(*parts: str | Path) -> Path:
    """Return a path under ``var/shadow_runs/``."""
    return _sub_dir("shadow_runs", *parts)


def outputs(*parts: str | Path) -> Path:
    """Return a path under ``var/outputs/``."""
    return _sub_dir("outputs", *parts)


def market_data(*parts: str | Path) -> Path:
    """Return a path under ``var/market_data/``, creating the directory."""
    canonical = _sub_dir("market_data")
    canonical.mkdir(parents=True, exist_ok=True)
    return canonical / Path(*parts) if parts else canonical


def experiments(*parts: str | Path) -> Path:
    """Return a path under ``var/experiments/``.

    This is the canonical location for the experiment registry and
    experiment artifacts.
    """
    return _sub_dir("experiments", *parts)


def default_registry_path() -> Path:
    """Return the canonical path to the experiment registry JSONL file."""
    return experiments("registry.jsonl")


def gap_distribution_latest() -> Path:
    """Return the canonical path to the latest gap distribution directory.

    This replaces the legacy
    ``live/pipeline_data/gap_adjusted_distribution/latest`` hard-coded string.
    """
    return live("pipeline_data", "gap_adjusted_distribution", "latest")


def gap_store_path() -> Path:
    """Return the canonical SQLite GapStore path."""
    return live("pipeline_data", "gap_adjusted_distribution", "gap_store.sqlite")


def execution_state_path() -> Path:
    """Return the canonical SQLite state store for live execution jobs."""
    return live("pipeline_data", "execution", "execution_state.sqlite")
