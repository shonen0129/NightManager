# Issue #25: official-ledger evidence contract and producer boundary (2026-10-10)

## Status and authority

**Producer mechanics are implemented, but actual-account risk-inclusive Stage 1 remains BLOCKED until real broker evidence is reconciled.**

The broker's **official** records are the starting authority for cash, inventory,
executions, expenses and PnL, per issue #25. This does **not** establish that
any currently available CSV or API response is complete. The Sep 27 extraction
has no broker order identity, covers only an 8-share strictly price-matched
subset of 207 requested shares, and does not establish financing, borrow or
reverse charges. Do not treat the CSV's ambiguous 損益/受渡金額 column as net
PnL. Do not substitute collateral, replay returns or estimated charges.

No code path in this change marks real broker data verified automatically.
`verified=true` remains a reconciliation assertion that requires retained
source evidence and a named accountable reviewer.

## Concrete official-source candidates

The current Tachibana adapter and the broker's current v4.10 interface expose
the following **read-only candidates**. They are concrete enough to define the
producer contract, but they are not declared complete until an actual-account
capture proves coverage and field semantics.

| Evidence | Official source candidate | Current use / limitation |
| --- | --- | --- |
| fills / order identity | `CLMOrderList`, `CLMOrderListDetail` | Order detail exposes execution quantity/price and individual execution rows (`aYakuzyouSikkouList`) for partial fills. Must prove every broker execution is captured and mapped one-to-one. |
| positions / lot identity | `CLMShinyouTategyokuList` | Current adapter records position number, side, quantity, open price, evaluation price/PnL and charge-related fields. Historical opening/closing coverage must still be retained. |
| cash / available amount | `CLMZanKaiSummary`, `CLMZanKaiKanougakuSuii` | Useful official cash/available-amount evidence, but margin/collateral must not be reinterpreted as PnL. |
| margin detail | `CLMZanKaiSinyouSinkidateSyousai`, `CLMZanRealHosyoukinRitu` | Supporting account-state evidence only; not a substitute for a cash ledger. |
| commission / financing / borrow / reverse | position/order-detail fields including `sOrderTateTesuryou`, `sOrderZyunHibu`, `sOrderGyakuhibu`, `sOrderKasikaburyou` where supplied | Must prove semantics, settlement timing, closed-lot coverage and explicit-zero behavior. Missing fields are not zero. |
| session mark | broker position evaluation fields plus an explicitly designated broker-authoritative close/session mark | The exact end-of-session valuation source and timestamp policy still require operator sign-off. |
| external cash flows | **no complete implemented source identified** | Deposits, withdrawals and transfers with value dates remain a hard blocker until an official report/API source is designated and reconciled. |

The authoritative interface reference is the Tachibana e支店 API v4.10
documentation: <https://www.e-shiten.jp/e_api/>. Repository wrappers for the
listed calls are in `src/leadlag/broker/tachibana/api.py` and
`src/leadlag/broker/tachibana/client.py`.

## Evidence required for each completed session

For each account and completed session, retain the following official
**source IDs and SHA-256 hashes** and the named person/team accountable for
reconciliation. All source entries remain `verified=false` until the underlying
broker report/API endpoint, its coverage and semantics are confirmed.

| Evidence class | Required proof |
| --- | --- |
| cash | Opening/closing broker cash, settlements, transfers, currency and value date |
| positions | Opening/closing quantity, side, lot identity, realized/unrealized basis |
| fills | Every broker execution ID, parent order ID, quantities, prices, partial fills, amendments |
| fees | Observed commission and taxes itemization, including explicit zero |
| financing | Interest/accrual/settlement evidence, including explicit zero |
| borrow | Stock borrow charges and timing, including explicit zero |
| reverse | Reverse/逆日歩 fees and timing, including explicit zero |
| session_marks | Broker-authoritative end-of-session mark, timezone and valuation timestamp |
| external_cash_flows | Deposits, withdrawals and transfers with value date and independent cash reconciliation |

A source is not complete merely because the endpoint succeeded. Assign a
named reconciliation owner to each of the nine source classes, document the
exact report/API name and extraction date, and verify the original bytes.
A read-only wallet/position capture in `leadlag.execution.reconcile` is useful
supporting evidence but is not a full historical ledger.

## Identity and reconciliation contract

An individual *broker execution ID* maps one-to-one to an internal
*local fill ID*, with the broker order ID retained as context. Enforce both
uniqueness directions, compare quantities/prices/side/date against the original
broker source, include fills not generated by this strategy, and prove both
broker→local and local→broker coverage. Aggregation by ticker/date/side does
not establish identity; a missing runtime row does not prove no historical
trade. Pending, unknown or nonterminal fills block completion. Partial fills
are represented as individual executions.

Prove opening + signed fills = closing quantity per instrument/lot; reconcile
opening cash + settlements - fees + external flows = closing cash (respecting
value dates), including corrections and corporate actions. Establish valuation
policy for long and short inventory using authoritative session marks. Reconcile
all fees (commission, taxes, financing, borrow and reverse), including explicit
zero evidence. Do not silently assign zero to missing fields.

External cash flows are included in the equity/cash bridge but excluded from
investment return. The producer uses opening reconciled equity before external
cash flows as the daily return denominator. Month-to-date return is the
compounded product of every TSE trading day's reconciled daily return; a missing
trading-day ledger blocks snapshot production, including a no-trade day whose
account state has not been reconciled.

## Implemented machine-readable boundary

`leadlag.execution.ledger_readiness.assess_ledger_readiness(manifest)`
checks an `account-ledger-evidence-v1` manifest. Source entries require
`authority=broker_official`, `source_id`, `sha256`, `reconciled_by`,
`verified=true` and a timezone-aware observation timestamp. Crosswalk rows
require unique `broker_execution_id`/`local_fill_id` pairs and explicit
verification. It also requires complete fill coverage, zero pending fills,
cash/position/fee balance matches, external cash-flow reconciliation and
verified session marks.

`leadlag.execution.account_ledger` now provides the fail-closed producer:

- `build_reconciled_execution_ledger` validates the evidence manifest, the
  opening-to-closing equity bridge, all explicit charge categories and the
  return denominator before emitting a content-addressed
  `reconciled-execution-ledger-v1`.
- `build_account_risk_snapshot` validates immutable ledgers, requires the
  immediately completed prior trading session and complete month-to-date TSE
  coverage, and derives daily/monthly return.
- ledger and snapshot artifacts carry source IDs/hashes and stable SHA-256
  identities. Immutable files are written with exclusive-create semantics.
- `latest.json` is only a pointer to the immutable risk snapshot and includes
  the target file hash; `AccountRiskSnapshot.load` validates pointer, file,
  snapshot and source identities before live gating.
- the live decision run manifest records the exact account-risk snapshot ID,
  snapshot hash, source ledger IDs/hashes, account key, observation timestamp
  and PnL basis used by the gate.

The CLI can build artifacts only from prepared verified evidence:

```bash
python -m leadlag.execution.account_ledger ledger \
  --evidence-manifest evidence.json \
  --session session.json \
  --output-dir var/live/pipeline_data/account_ledger

python -m leadlag.execution.account_ledger snapshot \
  --ledger-dir var/live/pipeline_data/account_ledger \
  --valid-for-trade-date YYYY-MM-DD \
  --account-key tachibana:default \
  --output-dir var/live/pipeline_data/account_risk
```

These commands do not authenticate to a broker and cannot place, close,
cancel or resend orders. They intentionally cannot invent missing evidence.

## Offline acceptance coverage

Unit fixtures cover:

- normal ledger and hand-calculated daily return;
- individual execution rows for partial fills;
- missing fee evidence;
- position/cash/fee reconciliation mismatch;
- pending/unsettled fills;
- source/API failure represented as unverified evidence;
- external cash flow in the bridge but excluded from return;
- inconsistent equity bridge rejection;
- complete-month hand-calculated compounding;
- stale prior-session and incomplete month rejection;
- immutable snapshot/pointer identity and tamper rejection;
- decision-manifest linkage to the exact risk snapshot.

The existing live risk gate remains fail-closed when the snapshot is missing or
invalid, and the existing confirmed reduction-only path is unchanged.

## Remaining real-account acceptance work

The remaining work cannot be truthfully completed from repository fixtures:

1. Operator must retain actual official source bytes/exports for the target
   account and sessions, identify a complete official source for external cash
   flows, confirm charge and session-mark semantics, and name the accountable
   reviewer for each source class.
2. Perform and sign the real broker-execution ↔ local-fill crosswalk and the
   opening/closing position and cash bridges. Any historical activity outside
   the new execution database must be included.
3. Run the producer on those verified real records and compare daily/monthly
   returns with a hand-calculated real-account ledger.
4. Save a redacted read-only E2E evidence package and feed its immutable
   account-risk snapshot into issue #27's risk-inclusive Stage 1 acceptance.

Until these evidence steps are complete, `account-risk-snapshot-v1` for the
real account remains missing/invalid and actual-live must continue to block
increased risk while preserving the existing verified reduction-only path.
