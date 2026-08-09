# ADR-0001: Point-in-Time Data Access Layer

- Date: 2026-08-09
- Status: accepted
- Deciders: Devin

## Context

`compute_signal` relies on the convention that rolling windows end at
`current_index` and do not include rows at or after the trade decision point.
This is currently enforced by manual slicing and by `ComplianceAuditor` after
signal generation. Manual slicing is error-prone and a future refactor can
reintroduce look-ahead by accident.

## Decision

Introduce `leadlag.core.pit.PITMatrixView`: a read-only numpy view that knows
its `as_of` row and raises `PITAccessError` on any access beyond that row.
`compute_signal` wraps `all_returns` in this view and uses explicit
`historical_slice()`, `historical_range()`, and `asof_row()` methods.

## Consequences

- Look-ahead becomes a runtime exception at the point of access, not just an
  after-the-fact audit failure.
- Callers can still pass plain ndarrays; the function wraps them automatically
  to preserve backward compatibility.
- Existing leakage tests continue to pass, and new randomized property tests
  guard the invariant.
