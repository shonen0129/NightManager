# ML overlay BLPX-residual target diagnostic plan

## Hypothesis

The current raw directional-return target asks the ML overlay to relearn return levels already forecast by the BLPX distribution. A target equal to the directional realized return minus the same-day directional `mu_gap` forecast may isolate residual cross-sectional information and improve the overlay's relative allocation decisions.

## Fixed comparison

- Primary baseline: existing LightGBM behavior retrained in the same fold on `sign(score) * realized_return - 10 bps`.
- Secondary reference: the same V2 decision with ML overlay disabled.
- Candidate: regression on `sign(score) * realized_return - scale * sign(score) * mu_gap`; the fixed 10 bps is restored because this candidate only changes within-side allocation and does not estimate order-level incremental costs.
- Candidate scales: 0.8, 1.0, and 1.2 (±20% sensitivity around the primary value 1.0).
- Same features, ticker interactions, LightGBM settings, costs, decision inputs, V2 constraints, and seeds as the fixed retraining evaluation. The raw-target ML comparator and all residual variants are trained separately with the same cutoff in each fold.
- Expanding historical folds 2020–2024. Before each fold, remove the last five training dates; the outcome is one trade-date return, so this is conservative relative to the one-day label horizon.
- Preserve all dates, including flat/fallback days. Record net Sharpe, compounded net return, maximum drawdown, turnover, audit counts, and paired net-return differences. Paired uncertainty uses 20-day circular moving blocks, 1,000 samples, seed 42 to match the existing evaluator.

## Decision rules and limitations

The primary paired contrast is each residual target against the raw-target ML overlay; ML-off BLPX is a secondary reference. This is a retrospective diagnostic only. These years and related ML outcomes have already been viewed; no candidate can pass a fresh-OOS adoption gate from this run. DSR will be shown for the four ML variants (raw plus three residual scales) and explicitly marked incomplete because it cannot reconstruct the full prior ML search set.

No target variant will be published to the production artifact root or selected by `CURRENT`. Historical provider `available_at`, complete frozen 09:10 quotes, verified broker inventory, and order-level incremental costs remain unavailable. A separate gross/trade gate is out of scope until those inputs have a verified producer. Any future production decision requires a newly fixed artifact and at least 250 complete forward paired trading days under the existing protocol.
