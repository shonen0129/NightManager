# ML overlay operator-directed promotion (2026-09-27)

## Decision

At the user's explicit request, the candidate was promoted to the production ML overlay root
`models/ml_order_overlay/production_20260923`. The immutable version
`20260926T182011981905Z-f8b4ec178e88` is now selected by `CURRENT`; its model SHA-256 is
`616c8ba8564ece95398788450cdc50df0ce3f189c9eaebd53dd6142fd42fca8b`. The previous version
`20260922T011723828589Z-7e1a81ab2617` remains available for rollback. The resolved production
config path did not change.

`ProductionRunner` loaded the selected version through the resolved production config after the
switch. The prior immutable version remains present for rollback. No API client or broker adapter
was created and no orders were sent during promotion.

## Gate status and risk

The original gate was not met. No new forward labels had accumulated (0 of the required 250),
historical provider `available_at` lineage remained unverified, and the known-period diagnostic
walk-forward showed worse maximum drawdown in all five folds. This promotion is an explicit
operator override requested after those facts were disclosed. It is not a numerical-gate pass,
an independent OOS result, or evidence that the candidate outperforms the former active model.

The full metrics and data limits are in
[`reports/20260927_ml_overlay_retrain/report.md`](../../reports/20260927_ml_overlay_retrain/report.md).
The machine-readable gate and promotion records are
[`promotion_gate.json`](../../reports/20260927_ml_overlay_retrain/promotion_gate.json) and
[`20260927_operator_directed_promotion.json`](../../models/ml_order_overlay/production_20260923/promotions/20260927_operator_directed_promotion.json).

## Rollback

To restore the previous model, atomically set `models/ml_order_overlay/production_20260923/CURRENT`
to `20260922T011723828589Z-7e1a81ab2617`, then load the artifact through the resolved production
config and verify the model digest before the next decision run. Do not delete either immutable
version.
