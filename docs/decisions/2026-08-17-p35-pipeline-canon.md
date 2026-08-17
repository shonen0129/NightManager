# ADR-P35: V2 Synchronous Pipeline as Canonical Production Path

- Date: 2026-08-17
- Status: accepted
- Deciders: leadlag-fund development team

## Context

The codebase has been carrying two daily decision paths since Phase 33-34:

1. **V2 synchronous path**: `cli.py decision` → `v2_bridge.py` → `post_decision.py` → `broker_ops.py` → `close.py`
2. **Next-Gen asynchronous path**: `cli.py decision --engine nextgen` → `nextgen_pipeline.py` → `AsyncExecutionEngine` / `async_fsm.py`

Both paths share the same BLPX signal math (`ProductionBLPXModel` / `ProductionV2Model`) and must produce equivalent `w_final` vectors, but they differ in execution engine and PIT-data access pattern. The `production.yaml` contains both V2 and `nextgen_*` parameters, and the CLI exposes an `--engine {v2, nextgen}` choice where `nextgen` is documented as experimental. This dual-track state prevented final cleanup of `models/`, `execution/`, and `config/` because the unused track always had to be preserved.

## Decision

**Adopt the V2 synchronous path as the canonical production execution pipeline.**

The Next-Gen async FSM pipeline (`nextgen_pipeline.py`, `async_fsm.py`, `AsyncExecutionEngine`) is deprecated and will be moved to `archive/legacy_src/execution/` in Phase 41. However, two Next-Gen components are judged worth preserving for later integration in Phase 41:

- `PITDataLake` (`data/pit_lake.py`): a unified as-of data access abstraction
- `ConvexOptimizer` / `core/convex_optimizer.py`: the constrained portfolio optimizer

These components will be decoupled from the async FSM and made available to the V2 path, rather than being deleted with the FSM wrapper.

## Consequences

### Positive

- `cli.py` can remove `--engine nextgen` and default to V2 only.
- `production.yaml` can drop the `nextgen` FSM execution block, simplifying configuration.
- `broker_ops.py`, `post_decision.py`, `close.py`, `v2_bridge.py` become the single source of truth for order execution.
- `import-linter` contracts become simpler because the async path no longer crosses the `execution → cli` boundary.

### Negative / risks

- Async execution (order splitting, rate limiting) is removed. The V2 path uses synchronous order submission; high-volume or slow-broker scenarios may reintroduce the need for async logic later.
- The deprecation is based on one-day shadow evidence. A longer shadow period (≥1 month) is still recommended, but accepting this ADR now unblocks Phase 36-42.

### Migration notes

1. `cli.py`: remove `choices=["nextgen", "v2"]` and default to V2.
2. `production.yaml`: remove `nextgen` execution/FSM block; keep `pit_lake` and `convex_optimizer` settings (relocate to `v2` or a new `math` block in Phase 41).
3. `src/leadlag/execution/nextgen_pipeline.py`, `async_fsm.py`, `AsyncExecutionEngine` and related tests: move to `archive/legacy_src/execution/`.
4. `PITDataLake` and `ConvexOptimizer` stay in `src/leadlag/` and are integrated into the V2 backtest/production path in Phase 41.

## Evidence

Shadow comparison on 2026-08-14 after fixing a PIT-history inclusion bug in `nextgen_pipeline.py`:

- Weight cosine similarity: 0.9410
- Weight correlation: 0.9410
- Net exposure: 0.0 / 0.0
- Gross exposure: 2.0 / 2.0
- Ex-ante return: 151.48 bps / 151.11 bps
- Ex-ante IR: 1.5399 / 1.4673
- Execution time: 0.272 s / 0.908 s

The two paths are close, but the V2 path is simpler, tested, and faster for the dry-run scenario. Long-term divergence is still a risk, so the Next-Gen math components are retained for Phase 41 rather than discarded entirely.
