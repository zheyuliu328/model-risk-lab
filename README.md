# Model Risk Lab

**Small models. Explicit assumptions. Reproducible challenges.**

[![Validation](https://github.com/zheyuliu328/model-risk-lab/actions/workflows/validation.yml/badge.svg)](https://github.com/zheyuliu328/model-risk-lab/actions/workflows/validation.yml)

How do you tell whether a model result is trustworthy? This repository turns that question
into two teaching experiments and a configurable monthly regression tool, using public mathematical methods.
The focus is on finding failure modes and explaining evidence, rather than maximizing a score.

| Experiment | Question | Evidence |
|:--|:--|:--|
| **Credit regression** | Does a candidate beat simple baselines when every feature was available at prediction time? | Forward time splits, train-only preprocessing, a locked holdout, deliberately invalid leakage control |
| **FX option sensitivities** | Do derivatives agree with prices, units and no-arbitrage identities? | Analytical Greeks, price-based finite differences, scaling tests, all bump sizes retained |
| **Monthly candidate producer** | Which permitted one/two-feature OLS candidates survive forward validation? | Your declared monthly observations, feature lags/release delays, all attempted candidates, two baselines, development-only selection and an explicit holdout reveal |

**[Read the generated validation report →](docs/sample/REPORT.md)** ·
[Architecture](docs/ARCHITECTURE.md) · [Learning notes / 学习入口](docs/LEARNING_NOTES.md)

[Second review: failure cases and corrections](docs/REVALIDATION.md)

![Credit validation with synthetic data](docs/sample/credit_validation.svg)

## Run locally

Python 3.11 or newer. No API keys, external datasets or network calls are required at runtime.

```bash
git clone https://github.com/zheyuliu328/model-risk-lab.git
cd model-risk-lab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install '.[dev]'
python -m pytest -q
model-risk-lab --seed 20260907 --output outputs/reproduction
```

Open `outputs/reproduction/REPORT.md`. The command produces the synthetic inputs,
predictions, charts, fold metrics, code hashes and dependency versions.
The destination must be new: complete evidence is built in a temporary sibling directory,
then published together. A failed run leaves no report at the requested destination and
never updates an earlier bundle. Verify the saved file inventory and checksums with:

```bash
model-risk-lab --verify outputs/reproduction
```

The checksum file detects missing or changed evidence; it is not a signature or a model approval.
[requirements-reproduce.txt](requirements-reproduce.txt) records the sample's numerical environment;
the package accepts a broader compatible range and CI exercises supported Python versions
using the recorded numerical versions. After changing code, reinstall the package or run
`PYTHONPATH=src python -m model_risk_lab.report` directly from the checkout.

## Screen monthly regression candidates

The new `model-risk-forecast` command accepts a documented JSON observation table. It is a reusable
tool with an optional invented example. For CSV/XLSX field selection and an interactive review,
use [Forecast Review Workbench](https://github.com/zheyuliu328/forecast-review-workbench).

```bash
mkdir -p outputs
model-risk-forecast --example --output outputs/monthly-development
model-risk-forecast --request outputs/monthly-development/request.json \
  --reveal-holdout --output outputs/monthly-holdout
```

Start with `REPORT.md`, then inspect `candidates.csv`, `folds.csv`, `predictions.csv` and
`failures.csv`. The example has 180 invented months, five permitted features, 15 single/pair
OLS candidates, two baselines and a 24-month holdout. All candidate failures are retained.
The first run reports development scores; the second explicitly reveals holdout scores without
reselecting the development winner. Full raw inputs are included in both evidence bundles.

Replace `rows`, feature declarations and cutoffs in the example request to use your own monthly
observations. [The input contract and method](docs/FORECAST_METHOD.md) explain the required unique,
continuous monthly calendar, train-only scaling, common samples, lag/release constraints and
diagnostics. Supplied target values are used as-is. The tool does not infer publication dates,
impute missing features, transform the target, or prove that a human has never viewed the holdout.

Exit 0 means evidence was produced with a usable selected candidate; 1 means evidence was produced
but all OLS candidates failed, or the explicitly requested holdout has no usable selected-candidate
score. Exit 2 means an input/output error. Existing output directories are never replaced.

## What makes the experiments useful

- **Information timing is tested.** Shifting a future observation must not change an earlier feature.
  A time split alone cannot rescue a feature that contains its own target.
- **Selection and evaluation are separate.** Development folds select the credit candidate;
  the final 36 synthetic quarters are scored only after selection. Baselines remain visible.
- **A failing hypothesis is a result.** The report retains worse performance, bias and regime-change effects.
  It does not reselect the model or seed to improve the headline.
- **Numerical agreement has limits.** A volatility-unit mistake can pass put–call parity.
  Bumps that are too small can make gamma worse through floating-point cancellation.
- **Evidence is inspectable.** CSVs and JSON accompany plots; tests check causal timing,
  analytic identities, units, boundaries and reproducibility.

![FX finite-difference convergence](docs/sample/fx_convergence.svg)

## Read the implementation

| Layer | Files |
|:--|:--|
| Contract and pricing | [fx.py](src/model_risk_lab/fx.py), [FX method and units](docs/FX_METHOD.md) |
| Synthetic mechanism and time validation | [credit.py](src/model_risk_lab/credit.py), [credit protocol](docs/CREDIT_METHOD.md) |
| Configurable monthly candidate screening | [forecast.py](src/model_risk_lab/forecast.py), [installed CLI](src/model_risk_lab/forecast_cli.py), [protocol](docs/FORECAST_METHOD.md) |
| Evidence generation | [report.py](src/model_risk_lab/report.py), [sample results](docs/sample/results.json) |
| Independent checks | [FX tests](tests/test_fx.py), [credit tests](tests/test_credit.py), [reproduction test](tests/test_report.py) |

## Scope and authorship

An independent educational project by **Zheyu Liu**, developed with AI-assisted implementation
and explicit reviewable tests. Every bundled dataset and option scenario in this repository is freshly
generated from the documented invented mechanism. No employer/client code, templates,
market-data subscriptions, confidential outputs or private Git history are included.
See the [data and source declaration](docs/DATA_AND_SOURCES.md).

The FX model assumes European exercise and flat rates/volatility. The credit model predicts
synthetic quarterly aggregate default rates, not individual credit decisions or full IFRS 9 ECL.
This project does not implement SIMM aggregation, validate market calibration, establish regulatory
compliance or claim production readiness.

## Next improvements

Each extension must add a falsifiable question and evidence; see [ROADMAP.md](ROADMAP.md).
Related earlier prototypes: [VaR backtesting](https://github.com/zheyuliu328/risk-var-dashboard),
[CreditOne](https://github.com/zheyuliu328/algorithmic-credit-risk-engine).

MIT licensed; public methods and libraries are credited in the method notes.
