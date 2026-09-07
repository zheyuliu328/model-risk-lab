# Roadmap: each change should answer a question

## Delivered

- [x] Explicit input/availability contracts with counterexamples.
- [x] European FX prices, five analytical Greeks, 270 finite-difference estimates and boundary tests.
- [x] Synthetic credit generator, five forward validation folds, two baselines and a locked holdout.
- [x] Portable evidence report with data/code hashes and reproducibility tests.

## Next: macroeconomic release delays

Question: how much do conclusions change when an observation exists but was not yet published?

Acceptance: distinguish observation and release quarters; reject unavailable inputs; compare immediate,
one-quarter and two-quarter release assumptions on a prespecified seed set; retain every result.
No real data ingestion is required for this extension.

## Then: robustness across invented paths

Question: does the chosen modelling strategy reliably improve on persistence across different paths?

Acceptance: commit a seed list before running; report win rate, dispersion and regime-specific errors;
separate exploration seeds from evaluation seeds; avoid interpreting synthetic performance as market evidence.

## Then: numerical tail diagnostics

Question: where do analytic option formulas lose precision or cease to be a useful numerical reference?

Acceptance: define extreme synthetic grids, compare against an independently implemented high-precision
or quadrature reference, disclose underflow/cancellation and all excluded domains, and document tolerances.

## Contribution rule

A feature proposal needs a question, public method source, invented-input contract, independent check,
expected failure modes and an interpretable output. A new badge or framework alone is not an improvement.
