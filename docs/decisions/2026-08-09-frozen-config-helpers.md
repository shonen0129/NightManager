# ADR-0005: Frozen Config Helpers

- Date: 2026-08-09
- Status: accepted
- Deciders: Devin

## Context

`base_cfg.copy()` on a nested config dict causes shared mutable sub-dicts. In
one experiment, two runs both ended up with `robust_pca=True` because the
nested `blpx` dict was shared.

## Decision

Add `leadlag.config.frozen`:
- `safe_config_copy` uses Pydantic `model_copy(deep=True)` for Pydantic models
  and `copy.deepcopy` for dicts.
- `FrozenConfigDict` is a read-only view that raises `ConfigMutationError` on
  writes.

## Consequences

- Experiment scripts have a single, safe copy function.
- `FrozenConfigDict` can be used for configs that must not be mutated.
- The project continues to move toward Pydantic `AppConfig` as the canonical
  type, but dict-heavy research code is protected now.
