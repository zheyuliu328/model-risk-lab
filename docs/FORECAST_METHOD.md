# Monthly OLS candidate producer

This independent producer screens all single-feature and two-feature ordinary least squares (OLS) candidates from caller-supplied monthly observations. With five features it retains 15 candidates and two baselines. It selects one successful OLS candidate using development performance and evaluates an explicit holdout only when requested.

It is a candidate experiment, not model approval, causal analysis, a forecasting platform, or a financial model validation. Inputs, publication delays, target availability and transformations are caller declarations. The supplied target values are used as-is: no transformation, inverse transformation or missing-value imputation occurs.

## Run an experiment

The installed entry point is `model-risk-forecast`. The module invocation is also available:

```bash
model-risk-forecast --example --output /tmp/monthly-development

model-risk-forecast \
  --request /tmp/monthly-development/request.json \
  --reveal-holdout \
  --output /tmp/monthly-holdout

python -m model_risk_lab.forecast_cli \
  --request request.json \
  --output /tmp/external-monthly-development
```

Each output directory must be new and its parent must already exist. The default stage exports development predictions and metrics. `--reveal-holdout` explicitly adds holdout evaluation while preserving the development selection. Full raw observations are retained in both stages; this switch is an evaluation discipline, not access control over the data already supplied by the caller.

Exit codes:

| Code | Meaning |
|---|---|
| 0 | Evidence package published and an OLS candidate selected; if holdout was requested, the selected candidate also has a holdout score. |
| 1 | Evidence package published, but no OLS candidate was eligible, or the requested holdout could not score the selected candidate. Inspect failures and coverage. |
| 2 | Input or output error. This run did not publish a new evidence package. Existing files are preserved. |

The API is `run_experiment(request: dict, reveal_holdout: bool = False) -> dict`. `example_request()` supplies 180 invented months, five declared features, three 12-month development validation folds, and a 24-month holdout. The module depends on NumPy and the standard library only. Its syntax supports Python 3.10; the repository's installation requirement remains controlled by `pyproject.toml`.

## Input contract

The request has `schema_version: 1`, a `title`, `target`, `features`, `rows`, `horizon`, and `development_end`. Optional settings are `n_splits` (default 3), `validation_months` (default 12), `min_train` (default 36), and `source_note`.

| Field | Contract |
|---|---|
| `target` | Nonempty `name`, `unit`, and `transformation` declarations. |
| `features` | One to five unique IDs drawn from `f1` through `f5`; each has a nonempty name, integer lag and optional integer release delay (default zero). |
| `rows` | At most 2,400 monthly observations. Each row contains a strict `YYYY-MM` period, numeric target, and a feature object containing every declared ID. |
| Target numbers | Finite JSON numbers with absolute value at most 10¹². Missing, boolean, string, NaN and infinite targets are rejected. |
| Feature numbers | The same numeric bound; explicit `null` is permitted and tracked as missing. An absent feature key is a schema error. |
| Monthly calendar | Periods must be unique and consecutive. Row order is normalized; duplicate or missing months are rejected rather than merged or inserted. |
| `horizon` | Integer from 1 to 12, in months. |
| `development_end` | A supplied month leaving at least one later holdout month. |
| Splits and sample size | From 1 to 24 folds, at least one month per validation window, and `min_train` from 3 to 2,400. OLS also requires more rows than fitted parameters. |

The command line accepts a UTF-8 JSON request file up to 10 MiB, rejects duplicate JSON keys and nonstandard numeric constants, and reads the source without modifying it. The JSON file contains observations directly; it does not contain instructions to open arbitrary nested file paths or fetch remote data.

## Availability and the common calendar

For target month **t**, prediction origin is **t − horizon**. A feature's raw input value is dated by its observation month. The producer internally uses the observation at **t − lag**, with the mandatory inequality:

```text
lag >= horizon + release_delay
```

For example, horizon 2 and release delay 1 require lag at least 3. The raw feature must not already have been shifted by the caller; applying the lag twice would change the experiment.

Every declared lagged feature must be present for a target month to be usable, including when evaluating a single-feature candidate or either baseline. Warmup months and missing lagged values are excluded from the common sample for all models, with the source month and feature recorded on the original input row. A raw null affects its later lagged target month, not automatically its own observation month.

Development and holdout eligibility are calculated independently. Missing holdout features cannot alter development denominators, coefficients or selection. Target availability is assumed at the declared prediction origin; revised historical data and actual publication timestamps are not independently verified by this tool.

## Development folds and final fit

Validation windows are the final `n_splits × validation_months` consecutive calendar months ending at `development_end`. Windows are defined before excluding unusable observations, so a missing value cannot silently move a fold's calendar boundary.

For each fold starting in month **v**, training uses all common usable rows whose target period is **at or before v − horizon**. This excludes the `horizon − 1` intervening calendar months. It does not merely remove a fixed number of surviving rows after missing-value filtering.

Each fold fits its feature means and population standard deviations using that fold's training rows only. OLS is fitted with an intercept on the standardized design. Validation rows receive the already fitted transformation. The final fit uses common usable rows up to **first holdout month − horizon**. Final coefficients and scaling parameters then remain fixed for the entire holdout.

These are rolling prediction origins with fixed coefficients inside a validation window or holdout. Features newly available at a later origin can be used at that origin. They are not forecasts of every future month made from one initial origin.

## Candidates, failures and baselines

Candidates have stable IDs such as `ols-f1` and `ols-f1-f2`; feature order and performance ranking cannot rename a model. Every single/pair combination is retained, including failures.

Zero training variation, rank deficiency, standardized-design condition number above 10⁸, insufficient training or validation sample, and numerical failures are reported explicitly. Any failed development fold or final fit makes the candidate ineligible for pooled development comparison and selection. Individual fold diagnoses remain available. A final fit can be present even when an earlier fold failed; it does not reverse that failure.

The baseline IDs are:

| ID | Prediction |
|---|---|
| `baseline-mean` | The current training window's target mean; the final training mean is fixed across the holdout. |
| `baseline-persistence` | Actual target `y[t − horizon]` available at each prediction origin. |

Persistence intentionally updates its observed history as origins advance, including earlier validation or holdout actuals once available. This is disclosed rather than presented as a fixed, single-origin forecast. Baselines share the candidates' evaluation months and minimum training requirement, but cannot be selected as the OLS winner. If all OLS candidates fail, selection is `null` while usable baseline results remain visible.

## Metrics and selection

All successful candidates and baselines use identical development target months. Months receive equal weights. Residual is prediction minus actual:

```text
MAE  = mean(abs(residual))
RMSE = sqrt(mean(residual²))
bias = mean(residual)
```

Pooled scores combine individual validation observations; pooled RMSE is not an average of fold RMSEs. The implementation normalizes errors before squaring to avoid overflow from extreme but finite extrapolations. This improves numerical handling; it does not make such extrapolations credible.

Selection minimizes pooled development MAE among successful OLS candidates. Exact floating-point ties are broken by stable model ID. Baselines, holdout metrics and in-sample R2 do not select the winner. Revealing a holdout cannot change a development model's status, ranking, final coefficients or selection.

Each successful fit reports training membership, feature means/scales, original-unit coefficients/intercept, target mean, training R2, adjusted R2, standardized-design condition number and VIF. R2 and adjusted R2 are `null` for constant training targets. Persistence has no estimated coefficients and its regression diagnostics are `null`. No p-values, confidence intervals, significance claims or residual-independence approval are supplied.

## Evidence package and fingerprints

| File | Purpose |
|---|---|
| `request.json` | Complete normalized request, including excluded observations. |
| `result.json` | Full API result, protocol, metrics, fits, predictions and row reasons. |
| `candidates.csv` | Every candidate/baseline, status, failure reason, metrics and final-fit diagnostics. |
| `folds.csv` | Every model/fold, exact training and validation months, scores and fit diagnostics. |
| `predictions.csv` | Common evaluation months, actual target, split/fold and stable model columns. Failed models have blank predictions. |
| `input_rows.csv` | Every raw month, original and lagged values, source months, eligibility and exclusion reasons. |
| `failures.csv` | Model failures, fold failures, row-exclusion reasons and revealed-holdout failures. |
| `REPORT.md` | Offline readable outcome, coverage, candidate scores, failure interpretation and method. |
| `manifest.json` | SHA256 of the original request bytes, all other exported files, both producer source modules, and runtime version information. |

`selection_fingerprint` binds development observations and selection-relevant configuration, including source declarations; it excludes the title and all holdout observations. `data_fingerprint` binds the full normalized input, including holdout observations. Neither fingerprint depends on the reveal switch. Canonical period/feature ordering makes row reorderings immaterial.

Rerunning the same source with the same code and environment produces deterministic artifacts. Rerunning the exported normalized request reproduces the numerical result, but its source filename/hash may differ from the original unnormalized request; that provenance difference is recorded. Floating-point results across different NumPy/BLAS environments may differ slightly. Hashes establish artifact identity, not source authenticity or business approval.

Sources are read only. The output is staged in its destination parent and published using a platform operation that refuses replacement, including a directory created by another process after the initial check. This is supported on macOS, Linux with `renameat2`, and Windows; unsupported platforms fail without publishing. No source or pre-existing output is deleted. Generated staging files are cleaned up on a failed publication. CSV text with formula-significant prefixes receives a protective apostrophe; JSON retains the exact source text. User text is escaped in the Markdown report.

## Public method references

- [NumPy `linalg.lstsq`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html) documents least-squares solutions, rank and singular values. Its returned residual array can be empty for deficient or underdetermined systems; this producer checks rank and computes residuals explicitly rather than treating an empty array as zero error.
- [scikit-learn `TimeSeriesSplit`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) describes ordered expanding training sets, comparable time windows and a train/test gap. The producer implements its own explicit monthly boundaries and horizon cutoff; it does not call this class or claim identical indexing after missing-value exclusions.
- [scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html) explains why preprocessing statistics must be learned from training data and model choices must exclude test data. These principles motivate the fold-local scaler and holdout isolation.

The public references describe general methods. They do not endorse this implementation, the caller's data, or its use in a regulated or production model.
