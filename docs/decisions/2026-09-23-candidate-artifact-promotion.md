# Candidate artifact promotion decision (2026-09-23)

## Decision

The user explicitly approved promotion of the acceptance candidate artifact. The active version
`20260922T011723828589Z-7e1a81ab2617` from
`var/results/20260922_production_acceptance/artifacts/evaluation_2025` was copied as an
immutable versioned root at `models/ml_order_overlay/production_20260923`, and
`configs/production/production.yaml` now points to that root.

The shared loader verified `CURRENT`, `metadata_status=verified`, the model digest, and the
embedded/external provenance match before the config change was used. The resolved production
model bundle loads the same active version.

## Numerical gate and override

The fixed evaluation's numerical gate remains `false`. The candidate improves net Sharpe
(2.2047 → 2.2201) but worsens maximum drawdown (-27.75% → -29.41%) and turnover
(1.2971 → 1.3083). This result is preserved in `artifact_evaluation.json`; it was not
rewritten as a performance pass. The promotion is therefore recorded as an explicit operator
override, with approval evidence in
`reports/20260922_production_acceptance/promotion_record.json`.

## Conditions and rollback

This decision promotes the artifact and does not close unrelated operational conditions. Historical
provider-issued `available_at` remains unproven, broker fee components remain unavailable, and a
live scheduler order cycle has not yet been completed. Strict input validation remains fail-closed
for malformed historical cache rows.

To roll back, change the production config to a separately validated versioned root or disable the
overlay and run the loader/config checks again. The former legacy `phase2_8` root is not a valid
rollback target because it has no `CURRENT` pointer and is rejected by the production loader.
