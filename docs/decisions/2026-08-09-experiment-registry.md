# ADR-0002: Experiment Registry with Deflated Sharpe

- Date: 2026-08-09
- Status: accepted
- Deciders: Devin

## Context

The repository has ~30 archived experiment scripts and 264 reports. Trial counts
and reject/adopt decisions are recorded in `AGENTS.md` as free text, making it
impossible to compute the Deflated Sharpe Ratio (DSR) automatically.

## Decision

Add `leadlag.core.experiment_registry.ExperimentRegistry`, an append-only JSONL
store of `ExperimentRecord` objects. Each record carries parameters, metrics,
the number of independent trials, and the DSR computed using the Bailey &
López de Prado (2014) formula.

## Consequences

- Decisions are machine-readable and searchable.
- DSR is computed from the same record, so overfitting adjustments are not
  forgotten.
- Experiment scripts should import `ExperimentRegistry` and record every
  configuration tested.
