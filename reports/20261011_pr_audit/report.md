# PR audit and integration

Scope: open PRs #65, #66, #68, #69, #70, #71, #72, #73, #76, #77, #78 in
shonen0129/NightManager. This is a review of these changes and their affected
call paths, not a repository-wide safety certification.

## Findings and repairs

- **P1, #78, `pipeline/gap_publisher.py`:** both resolution checks referenced
  nonexistent `DistributionResult.is_available`. Every otherwise successful
  computation stopped with AttributeError. Replaced these checks with the
  canonical `DistributionStatus.READY`; corrected the other mypy errors.
- **P1, #78, prior TOPIX price:** ordinary preprocessed frames and PIT snapshots
  never carried the TOPIX prior close demanded by frozen-quote publication.
  Added `topix_close_sig` using the same historical JP signal date as sector
  prices, propagated it through the schema/PIT boundary, and bumped the cache
  contract to `pit-topix-close-v3`. The existing loader rebuilds old frames
  from raw ETF cache; model dimensions and traded ticker order are unchanged.
- **P2, #78, requested date:** reject frozen quotes belonging to another trade
  date at the publisher boundary. No prior-day quote is relabeled.
- **P1, #69, final close reconciliation:** a non-split `FILLED` response with
  no broker order ID was reported complete. A regression reproduced this
  false completion before repair. The final persisted-plan reconciliation
  now rejects it, propagating incomplete status to the report and durable
  reconciliation-required state.
- Validate the optional rank-reversal signal before the first write.
- Resolved #70/#78's conflicting test additions by retaining both capture
  terminal/attempt evidence tests and ADR freshness tests.
- Updated the architecture diagram and structured covariance description to
  distinguish the production publisher from research diagnostics.

The publisher regressions use actual OnDemandDistributionSource,
FileCacheDistributionSource and SQLite. Only the expensive model computation
and external input reads are replaced with synthetic fixtures. They cover
h=1/3/5 read-back, a failed h=3 computation causing no publication, prior TOPIX
close and mismatched quote dates. The original rank-reversal current-label
invariance assertion remains.

## Other reviewed boundaries

| PR | Review focus | Decision |
|---|---|---|
| #65 | safe exception/log boundaries; failed reauthentication state; durable post-commit store errors | mergeable; merged separately |
| #66 | explicit research V1 input; no implicit terminal cache in cost regression | integrate |
| #68 | inherited config through close/decision/broker; account/provider and safe manifest | integrate |
| #69 | persisted-plan quantity reconciliation; missing/duplicate response; missing order ID; delayed close gate | repair and integrate |
| #70 | shadow-only environment; registered read-only capture; terminal evidence and preflight | integrate; actual Stage 1 still pending |
| #72 | owner-only legacy credential remediation and explicit operation boundary | integrate; #34 remains open |
| #73 | invalid targets remain missing; missing quote fallback; finite historical label filtering | integrate |
| #76 | historical analysis archive; canonical wallet proxy CLI; no claim of account PnL | integrate |
| #77 | canonical allocator/inventory accounting; weekend costs; shared neutrality helper | integrate |
| #78 | scheduled production/research boundary; source status; quote/PIT/cache provenance | integrate after required CI |
| #71 | Draft account-ledger/readiness/snapshot producer; actual official evidence outstanding | HOLD; omitted from integration |

## Validation

- Integrated checkout before the final missing-order-ID repair: targeted regressions **147 passed**.
- Complete `tests/`, including fixed V2 baseline: **1344 passed, 1 warning**,
  154.28 seconds. Watchdog deadline 1800 seconds, 10-second kill grace,
  exit 0. No timeout occurred. Warning: existing fragmented test DataFrame.
- compileall: PASS; Ruff production/tests/maintained/research: PASS.
- operational imports: 3 Python entry points / 2 shells PASS.
- maintained documentation paths, symbols and links: PASS.
- Each of the nine originally green PRs was checked against its own successful
  full CI logs, including mypy, import contracts, wheel exclusion/smoke and
  complete pytest. Local mypy is unavailable; do not count that local attempt
  as PASS. Final integrated head must pass locked CI before merge.
- The missing-order-ID regression failed before its repair (`close_incomplete`
  was False). The final repaired head is revalidated through complete pytest
  and locked CI; final results are recorded in the completion report.

#65 merged to main as a42c7d0dd47849339b57643de43d545c915bcc84.
Updating main caused #66's immediate merge to be rejected because the required
`quality-and-tests` check was expected for the new combination. Integration
therefore retains every reviewed PR head as an ancestor of #78 and includes
latest main, rather than bypassing branch protection or squashing provenance.
The final GitHub merge and CI result are reported to the user after completion.

## Limits and outstanding acceptance

No broker connection, order, close, resend, credential action, scheduler
installation or live data update was executed. The user's original checkout
and cleanup commit were preserved; changes were developed in isolated /tmp
checkouts. Publication still requires valid owned execution inputs (observed
open-to-09:10 data or positive fallback opens); missing inputs stop publication.
The current-day ADR real-source refresh and market/shadow acceptance are not
proven by offline tests. #25's actual account reconciliation evidence and #34's
owner-only credential/session remediation remain outstanding. #71 is Draft
and must not be promoted merely because its synthetic tests or CI succeed.

Final missing-order-ID repair: the close/config regression set is **31 passed**
(14.12 seconds); compileall and production/research Ruff passed again.
