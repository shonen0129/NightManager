"""Canonical project path resolution.

All runtime outputs (results, live, artifacts, logs, shadow_runs,
market_data) are consolidated under ``var/`` at the project root.
Callers should prefer the helpers in this module to constructing path
strings manually, which is the root cause of ``results/`` / ``live/``
scattering described in ADR-0006.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def project_root() -> Path:
    """Return the project root directory resolved from this source file.

    ``src/leadlag/config/paths.py`` -> ``src/leadlag`` -> ``src`` -> root.
    This is independent of the current working directory.
    """
    return Path(__file__).resolve().parents[3]


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
    """Return a path under ``var/market_data/``.

    Backward compatibility: if the canonical ``var/market_data/`` does not
    exist yet but a legacy ``market_data/`` directory exists at the project
    root, the legacy path is returned with a deprecation warning. This allows
    ADR-0006 to be adopted without an immediate destructive data migration.
    Run ``tools/migrate_outputs_to_var.py`` to move data permanently.
    """
    canonical = _sub_dir("market_data")
    root_legacy = project_root() / "market_data"
    if canonical.exists() or not root_legacy.exists():
        canonical.mkdir(parents=True, exist_ok=True)
        return canonical / Path(*parts) if parts else canonical
    logger.warning(
        "Legacy market_data/ at project root is being used. "
        "Run tools/migrate_outputs_to_var.py to move it under var/ market_data."
    )
    return root_legacy / Path(*parts) if parts else root_legacy


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
