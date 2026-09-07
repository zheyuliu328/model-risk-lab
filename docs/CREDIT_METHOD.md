# Synthetic credit regression: a protocol before a score

## Prediction question

At the start of quarter t, predict the aggregate default rate in quarter t using information
available through t-1. We assume prior-quarter data are published immediately and never revised.
Test blocks update observed history each quarter, with model coefficients fixed within the block.
This is sequential one-step prediction, not a 12- or 36-quarter forecast at a single origin.

## Invented mechanism

The default seed is 20260907. Generate 148 quarterly rows. Growth, unemployment and rate are
synthetic unitless AR(1) indices with coefficients 0.65, 0.75, 0.55 and independent normal innovations
with standard deviation 0.6. They are not real macroeconomic percentages or historical observations.

```
z[0] = -4.5
z[t] = -4.5 + 0.55*(z[t-1]+4.5) + 0.30*unemployment[t-1]
       - 0.40*growth[max(0,t-2)] + 0.15*rate[t-1]
       + 0.20*normal_innovation[t] + 0.60*I(t>=124)
p[t] = sigmoid(z[t])
defaults[t] ~ Binomial(10000, p[t])
observed_rate[t] = defaults[t]/10000
```

The last 24 quarters contain a predetermined intercept shift. Thirty-two independent normal noise
features are generated after the target process. None of these parameters was estimated from a private
or public portfolio. The report seed is fixed, not selected by its performance.

## Features and estimation

Five prespecified inputs: previous observed default logit; lag-one unemployment; lag-two growth;
lag-one rate; four-quarter growth mean ending at t-1. Rolling features are shifted before averaging.
Each feature carries its latest source quarter, strictly earlier than the prediction quarter.
Tests additionally perturb current/future raw records; provenance metadata alone is not proof.

Target = `logit((defaults+0.5)/(borrowers+1))`, which remains finite at zero or full default.
OLS and Ridge predict this transformed target; sigmoid maps predictions back to a rate.
This is not a grouped binomial GLM, and inverse transformation need not yield an unbiased rate forecast.

- Two baselines: last observed rate; training-window mean observed rate.
- Five candidates: core OLS; Ridge alpha 0.1, 1, 10; OLS with the core plus 32 noise-lag features.
- Every fit uses a new `StandardScaler` → estimator pipeline trained only on its training window.

## Selection and holdout

Drop the first four warm-up rows, leaving 144 observations. The first 108 are development data.
Use five forward folds, 12 quarters per test block, maximum 72 training quarters and no gap.
Training sizes are 48, 60, 72, 72, 72. This gap choice depends on the stated immediate-availability
assumption; a real release delay would change the experiment.

Select one candidate by pooled development fold-out rate MAE. Then fit on quarters 40–111 and
evaluate on quarters 112–147. The holdout never chooses the candidate or its alpha.
Other holdout model scores are diagnostic comparisons, not a second selection stage.

MAE, RMSE and prediction-minus-observation bias are in basis points of observed default rate.
R² uses a consistent logit target scale; negative values are retained. A constant or too-short
target is explicitly marked undefined. No universal statistical pass threshold is asserted.

## Deliberate failure control

Add the current transformed target as a feature while keeping forward splits. The score looks
excellent, but its source quarter equals the prediction quarter. The availability check rejects it
and it is never eligible for selection. The small rate error reflects target smoothing.

This demonstrates why a time split and a high score cannot establish availability by themselves.
It does not imply that every leakage mechanism necessarily increases a score on every dataset.
Random-split and global-scaler ablations remain future work; they are not implemented here.

## Public references

- [TimeSeriesSplit](https://scikit-learn.org/1.8/modules/generated/sklearn.model_selection.TimeSeriesSplit.html): ordered folds and window controls.
- [Common pitfalls](https://scikit-learn.org/1.8/common_pitfalls.html): fitting preprocessing on training data, pipeline boundaries.
- [Lagged-feature forecasting example](https://scikit-learn.org/1.8/auto_examples/applications/plot_time_series_lagged_features.html): why evaluation protocol matters for time series.
- [LinearRegression](https://scikit-learn.org/1.8/modules/generated/sklearn.linear_model.LinearRegression.html): least-squares objective.
- [R² definition and boundaries](https://scikit-learn.org/1.8/modules/generated/sklearn.metrics.r2_score.html).

The generator, chosen parameters and experiment design are original educational choices. These sources
support the statistical tools, not claims about any real borrower population or production ECL model.
