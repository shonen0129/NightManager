# ADR: Research-only event replay for execution policy

Date: 2026-10-07

## Decision

Add the adaptive execution simulator under `src/research/`. The current
market-order plan is derived through the pure `build_execution_plan()` helper
in `execution/broker_ops.py`; book math reuses the existing
`execution/microstructure/` snapshot, spread, and depth functions. The
simulator does not instantiate `BrokerClient` or call submission, cancellation,
polling, or retry functions.

The policy rests at the same-side best quote when the observed spread is wide
or displayed opposing depth is below remaining notional, while the execution
deadline is not near. It crosses the visible book when the deadline is near or
when doing so reduces a material pair imbalance. Allowed quantity is recomputed
from observed opposite-side fills and the pair net-exposure cap at every replay
event. A pair's cap cannot be below one legal lot notional, because that would
make an opening fill impossible; the resulting lot-size floor is reported.

Passive limit fills require explicit queue-aware evidence. A quoted-price touch
does not establish queue priority or a fill. Orders larger than the displayed
five-level depth retain an unresolved remainder; the simulator does not extend
the fifth level or infer hidden liquidity. Opportunity cost for unfilled
quantity requires a quote exactly at the configured deadline. Adverse
selection is reported separately from implementation shortfall to avoid double
counting.

### Replay integrity correction (2026-10-08)

`passive_evidence_complete` covers the interval since the preceding replay
event, not just the instant of the update. If a resting interval lacks complete
evidence or its book mark, the affected pair stops further execution and its
residual stays unresolved. A later complete event cannot recover that interval.
The capture adapter requires trade/queue evidence as well as the completeness
declaration. Missing evidence events remain in replay rather than being skipped.

Arrival eligibility requires the valid 09:10:00–09:10:30 JST capture. Later
same-session quote events have a separate eligibility gate and are retained
through the execution deadline and subsequent adverse-selection marks. Invalid
book events within the execution interval reject the comparison rather than
silently bridging the gap. Dates and duplicate timestamps remain validated.

Both modes share the same exposure accounting: revalue known signed filled
quantities at each event, record peak exposure after each simulated fill, and
integrate absolute exposure through the deadline. The last observed midpoint
is held until the next update or the deadline. Sequential fills at one timestamp
contribute to the peak but have zero elapsed time between them. This is a replay
convention, not evidence of actual intratimestamp execution timing. Unresolved
quantities make the reported exposure incomplete; known-fill values are partial
diagnostics, not an upper bound on actual exposure.

## Context

Existing capture records contain timestamped quote cross-sections but do not
provide a queue-aware fill tape. The current stored capture report also records
no valid 09:10 snapshot. Therefore the first run can audit replay eligibility,
but cannot claim a baseline-versus-policy execution result. Synthetic unit
fixtures test simulator mechanics only and are not performance evidence.

## Consequences

- Research replay stays separate from the production order path.
- A valid paired comparison requires same-session orders, 09:10-window
  reference books, a same-date signed-position snapshot, timestamped updates
  through the configured deadline, trade and queue evidence for passive fills,
  and enough marks for adverse selection.
- Initial policy constants are design values, not calibrated recommendations.
  Performance selection requires a new eligible replay study, sensitivity
  analysis, and out-of-sample validation.
