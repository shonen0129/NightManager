# ADR-0008: Test Markers and Property-Based Leak Tests

- Date: 2026-08-09
- Status: accepted
- Deciders: Devin

## Context

The test suite takes ~8 minutes in parallel and ~32 minutes serial. There is no
quick way to run only fast unit tests, and the leakage invariant is only tested
with a single fixed example.

## Decision

- Add pytest markers in `pyproject.toml`: `unit`, `integration`, `slow`,
  `property`, `leak`.
- Add `scripts/run_tests_unit_only.sh` to run `-m "not integration and not slow"`.
- Add randomized property tests in `tests/unit/test_pit_leak_property.py` that
  verify `compute_signal` is invariant to future-row corruption and that
  `PITMatrixView` rejects any slice ending after `as_of`.

## Consequences

- Local iteration can use the fast unit-only script.
- CI can run the full suite in parallel.
- The PIT / no-look-ahead invariant is tested with many random seeds and
  as-of indices.
