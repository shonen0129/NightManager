# ADR-0006: Consolidated Runtime Output Directory

- Date: 2026-08-09
- Status: accepted
- Deciders: Devin

## Context

The repository root contains six different output directories
(`outputs/`, `results/`, `logs/`, `shadow_runs/`, `live/`, `artifacts/`)
plus `archive/` and `experiments/`. New contributors cannot tell where to write
output, and cleanup is error-prone.

## Decision

- Create a single `var/` tree with subdirectories `outputs`, `results`, `logs`,
  `shadow_runs`, `live`, `artifacts`, and `experiments`.
- Add `var/` to `.gitignore`.
- Generate `uv.lock` from `pyproject.toml` to replace the deprecated
  `requirements.txt` as the source of truth for exact dependency versions.
- Provide `tools/migrate_outputs_to_var.py` to move existing root-level
  directories and leave symlinks for backward compatibility.

## Consequences

- New scripts default to a single output tree.
- Dependency resolution is reproducible via `uv.lock`.
- Existing tracked `outputs/` files remain until migration is run.
