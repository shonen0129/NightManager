# Architecture Decision Records (ADRs)

This directory records architectural and strategic decisions for the
lead-lag strategy project. Each ADR is a short markdown file in the format
`YYYY-MM-DD-short-title.md`.

New decisions should be added here, not embedded in `AGENTS.md`. `AGENTS.md`
remains the canonical list of operational invariants (must not break), while
`docs/decisions/` records the reasoning behind structural and tooling choices.

## Template

See [0000-adr-template.md](0000-adr-template.md).

## Index

- [0001 PIT data access layer](2026-08-09-pit-data-access-layer.md)
- [0002 Experiment registry](2026-08-09-experiment-registry.md)
- [0003 Data validation gates](2026-08-09-data-validation-gates.md)
- [0004 Timeouts and transactional cache](2026-08-09-timeouts-and-transactional-cache.md)
- [0005 Frozen config helpers](2026-08-09-frozen-config-helpers.md)
- [0006 Consolidated var output directory](2026-08-09-consolidated-var-output-directory.md)
- [0007 Sensitivity labels from ticker registry](2026-08-09-sensitivity-labels-from-ticker-registry.md)
- [0008 Test markers and property tests](2026-08-09-test-markers-and-property-tests.md)
- [構造改善：再現可能な計算境界と執行状態の分離](2026-09-15-structural-improvement-boundaries.md)
- [VaR入力指紋を既存の全体期限に含める（提案）](2026-09-22-var-input-fingerprint-deadline.md)
- [9:10価格の選択と本番artifact受入](2026-09-22-execution-price-and-artifact-acceptance.md)
