# ML overlay relative-allocation contract and residual-target diagnostic (2026-09-29)

## Decision

The production ML overlay output is a nonnegative **within-side relative allocation multiplier**, not a calibrated trade probability and not a gross-exposure controller. Keep the legacy `p_trade_*` summary aliases for stored-run compatibility, while emitting accurately named allocation-multiplier fields and `ml_trade_gate_status=not_evaluated...` diagnostics. Existing raw-target artifact behavior is numerically unchanged by this naming/reporting correction.

Add `target_type="blpx_residual"` as a research-only training option. Its label is

`sign(score) × realized_return − baseline_residual_scale × sign(score) × mu_gap`.

The existing fixed round-trip charge is restored before this transformation because the residual target is for relative allocation only. The artifact records the target type, baseline residual scale, and target cost contract, and the loader binds those fields to the serialized model metadata.

## Diagnostic outcome

Scales 0.8, 1.0, and 1.2 were compared with a raw-target ML artifact re-trained at the same cutoff for each 2020–2024 walk-forward fold, with a five-trading-day purge. Across 1,180 pooled dates, all three residual candidates had lower net Sharpe than the raw-target candidate and negative 95% intervals for paired daily net-return difference. The current residual-target formulation is therefore rejected as a replacement candidate in this historical diagnostic. The report and per-candidate registry corrections preserve the results.

These years were previously inspected and the historical provider `available_at` evidence remains absent. The diagnostic is not a fresh-OOS promotion gate. The production `CURRENT` pointer, production config, and current raw-target artifact were not changed.

## Separate trade gate

No ML trade/no-trade or gross-scaling gate is implemented. A verified account inventory, order-level incremental cost, and complete execution record are still unavailable; using model weights or fixed 10 bps as substitutes would repeat rejected historical proxies. The ML trade gate remains unassessed until those inputs have an authoritative producer. See [the residual-target report](../../reports/20260929_ml_overlay_blpx_residual/report.md) and [the account-risk forward-evaluation decision](2026-09-29-frozen-0910-account-risk-forward-eval.md).
