# Stage A-C safety boundaries

- Date: 2026-09-13
- Status: accepted

## Decision

The production V2 path uses four explicit safety boundaries:

1. A provisional `df_exec` row always maps a US session to a strictly later JP
   trading date. A same-calendar US close can never become the current JP 09:10
   input.
2. Broker polling keeps `SUBMITTED` and `PARTIALLY_FILLED` orders pending until
   a terminal state or the configured deadline. Partial and unconfirmed orders
   are recorded separately and cannot be reported as completed execution.
3. Gap distributions are read and written as a single μ/Ω/metadata bundle for
   every horizon. The V2 leakage audit receives the bundle's actual `sig_date`
   when available; it does not infer a future-safe date from the wall clock.
4. VaR return-cache identity includes content digests for the effective source
   data, SQLite WAL sidecars, resolved code, configuration, and overlay artifact.
   A model or distribution change therefore creates a new cache key.

5. ML overlay training output is published as an immutable version directory
   containing the pickled model and provenance metadata, then activated by an
   atomic `CURRENT` pointer. Readers verify the model digest and shared
   temporal/data/config fields before accepting the version. Root-level legacy
   files are rejected because their model and metadata cannot be proven to be
   one immutable pair; they must be retrained and republished.

6. An incomplete submission or close is recorded with its result summary before
   the original failure is propagated. Reconciliation of fills, positions,
   wallet, and journal is attempted independently; the close CLI returns 2 for
   an incomplete close and reserves exception paths for processing errors.

7. Decision and research distribution consumers require a committed provenance
   record with a scalar signal date strictly before the trade date and a
   matching horizon. The low-level loader may be used without that requirement
   only for explicitly diagnostic inspection; its three-value compatibility
   wrapper rejects unprovenanced bundles by default.

8. A VaR backtest snapshots its gap input before constructing the cache key and
   passes that same snapshot to the backtest. DataFrame-backed common-input
   caches include schema identity (column labels/order, dtypes, index metadata)
   as well as values. A Tachibana terminal failure/expiry code takes precedence
   over contradictory quantity fields, while amendment/cancellation request
   failures keep the original order pending until it is resolved.

9. Gap provenance is normalized by one utility shared by the low-level loader
   and V2 sources. Both signal-date aliases must agree; trade dates must be
   scalar, valid dates and horizons must be finite integers. VaR's snapshot
   acquisition shares one monotonic deadline with its backtest. SQLite backup
   busy waits are bounded by that deadline, and a timed-out worker owns its
   temporary snapshot until the worker exits.

ML overlays also fail closed when the trade date is within `train_end`; a
backtest must supply a fold-specific artifact or run an explicitly separate
ML-disabled comparison. Drawdown is calculated by one shared function that
starts its high-water mark at wealth 1.0. Regression fixtures use scoped
monkeypatches so they restore imported functions after the test.

## Consequences

The normal execution result distinguishes accepted, filled, partial, and failed
orders. A deadline with residual orders requires reconciliation before a retry or
new order submission. Existing legacy SQLite and `.npy` readers remain callable,
but SQLite production reads use the bundle API so μ and Ω cannot come from
different commits. The `.npy` compatibility path publishes a commit manifest
containing the μ/Ω/metadata digests; readers reject a missing or mixed manifest
and continue to the existing on-demand or flat fallback. A manifest whose
metadata is absent or inconsistent is likewise rejected by production and
research distribution sources, while diagnostic callers can inspect it only by
explicitly opting out of provenance enforcement.
An active versioned overlay is never edited in place; maintenance scripts must
publish a new version when provenance or model fields change.
