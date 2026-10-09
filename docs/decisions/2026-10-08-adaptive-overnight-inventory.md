# 銘柄別オーバーナイト在庫 sizing の研究契約

**Status:** Research implementation; adoption decision pending. Production configuration is unchanged.

## Purpose

The current fixed carry fractions (`overnight_alpha_long=0.75`, `overnight_alpha_short=0.50`) apply the same proportion to every active name. This research policy estimates a separate carry fraction for each ticker from historical direction persistence, reusable inventory, overnight marks, and carry costs. It remains in `src/research/` and is not part of the production decision path.

## Point-in-time inputs

For a carry decision at close on session `t`, the model may use:

- the current V2 weight and its direction, already known by that close;
- ticker-specific weight transitions whose destination session is no later than `t-1`;
- close-to-09:10 gap marks already observed by session `t`;
- calendar days until the next Japanese session; and
- configured slippage, financing, borrow, and reverse-fee assumptions.

It must not use session `t+1`'s weight, target return, or gap. `adaptive_carry_masks` does not accept target returns and its transition sample ends at `t-1`. A regression test perturbs future weights and marks and asserts the earlier carry decisions do not change. Next-session weights appear only in ex-post reuse and transition attribution.

## Sizing rule

For each active ticker and current direction, estimate same-direction, reversal, and flat transition probabilities from a 252-session rolling history, conditioned on prior origins with the same sign. A symmetric Dirichlet prior of `(1, 1, 1)` is used. Expected overlap is the historical same-direction overlap fraction with a symmetric Beta `(1, 1)` prior.

Let `c` be one-way slippage, `p_same` the estimated continuation probability, `o` expected reusable overlap, `g` the direction-adjusted mean known gap return, `h` expected calendar-day financing/borrow/reverse cost, `p_rev` the reversal probability, and `p_flat` the flat probability. The score is:

```text
edge = p_same * o * 2*c
     + g
     - h
     - p_rev * c * reversal_exit_multiplier
     - p_flat * c
carry_alpha = clip(edge / (2*c), 0, 1)
```

The reversal term estimates the extra carry unwind leg. The next target's actual opening flow remains in the execution-volume ledger. The alpha is zero for a flat current weight and on the final replay row.

## Replay accounting

Both the fixed-alpha baseline and adaptive policy use the same V2 weight stream and the corrected `simulate_daily_pnl` accounting. The replay starts flat, keeps one continuous weight-based inventory state across dates, charges opening plus closing inventory flow at one-way slippage, attributes the next 09:10 mark to the outgoing trade date, accrues financing/borrow/reverse fees by calendar days, and liquidates remaining inventory at the final close with a zero terminal alpha. Effective execution volume is `side_leverage * (opening_flow + closing_flow)`; turnover is half that volume.

This is a continuous notional replay, not a share-level cash and position ledger. Actual fill prices and ticker-specific borrow or reverse charges were unavailable, so the cost rates remain modeled assumptions.

## Evaluation and adoption gate

The comparison keeps the resolved production baseline values as the fixed benchmark but does not edit production configuration. It includes all evaluation sessions, 1–3 day gaps, extended holidays, uniform short-borrow stresses, ticker-level next-session reversal cases, and sensitivity checks. Performance metrics, source availability, and the current PENDING decision are recorded in [the experiment report](../../reports/20261008_adaptive_overnight_inventory/report.md).

The initial 159-session replay was not enough for production adoption: it was a single short interval, ML overlay application was skipped because ADR features were not supplied, macro prices were unavailable, and actual fill/borrow records were missing.

A follow-up replay generated weights from the resolved V2-only production configuration over 2015-01-05–2026-09-25 and evaluated 2,524 sessions from 2016-02-02. It broadens the historical evidence but is not a prospective untouched holdout. Fixed alpha produced net Sharpe 6.952 and net PnL sum 13.1085; adaptive produced 6.576 and 12.9647. Adaptive reduced MDD (-22.41% vs -31.52%) and carry fees (0.1671 vs 0.7882), but increased turnover and slippage; total net costs were slightly higher, reuse was lower, and the paired block-bootstrap interval for daily net difference included zero. Adaptive carry also had materially higher mean overnight net exposure (0.643 vs 0.303). The macro snapshot's historical availability is unproven, and actual fills and ticker-level borrow/reverse costs remain unavailable. See the [long V2-only report](../../reports/20261009_adaptive_overnight_inventory_v2_long/report.md).

**Decision remains pending and production carry sizing remains unchanged.** A future promotion review needs a genuinely prospective or held-out walk-forward period, verified execution/borrow costs, proven PIT macro provenance, and an explicit overnight net/gross exposure policy.
