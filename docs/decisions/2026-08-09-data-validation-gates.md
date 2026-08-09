# ADR-0003: Data Validation Gates

- Date: 2026-08-09
- Status: accepted
- Deciders: Devin

## Context

`preprocess_data` silently skips rows with NaN in required columns. `load_gap_matrices`
returns `(None, None, alerts)`. In production, stale or missing data can lead to
flat positions or, worse, trades based on yesterday's gap matrix.

## Decision

Introduce `leadlag.data.validation` with `DataValidationError`,
`validate_raw_data_sources`, `validate_exec_record`, and `validate_gap_matrices`.
`preprocess_data` and `load_gap_matrices` accept a `strict_validation: bool`
parameter. When strict, validation failures raise instead of silently skipping.
The default remains tolerant for backward compatibility.

## Consequences

- New pipelines can opt into fail-fast behavior.
- Gap-matrix shape, finiteness, symmetry, and positive semi-definiteness are
checked explicitly.
- Existing production calls should move to `strict=True` after shadow validation.
