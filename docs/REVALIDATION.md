# Revalidation — 2026-09-08

This second review challenged the published experiments with new counterexamples and a
clean, non-editable installation. Passing the original test suite was the starting point.

## Findings and corrections

| Trigger | Previous behavior | Corrected behavior | Regression evidence |
|:--|:--|:--|:--|
| Repeat a report at the same path, then interrupt chart generation | New CSV/JSON could coexist with old charts/report | Existing paths are refused; a new report is built in a temporary sibling and published only when complete | Interrupted-write, existing-bundle and competing-output tests |
| Change or omit a file in a copied report | Only the synthetic-input hash was recorded | A checksum manifest covers all eight evidence files; verification checks inventory and bytes | Changed-chart, missing-file and extra-file tests |
| Request a difference step smaller than the input float's resolution | A zero derivative could be reported; gamma could divide by zero | Unrepresentable steps and invalid denominators raise an explicit error | Five Greeks, tiny steps and a separately representable input with denominator underflow |

The build backend minimum also matches the SPDX license-string metadata used by this package,
following the [Python Packaging guide](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/#license-and-license-files).
The minimum is 83.0.0, which also includes the upstream fix for
[source-distribution exclusion handling on macOS](https://github.com/pypa/setuptools/security/advisories/GHSA-h35f-9h28-mq5c).

## Evidence

- **74 tests passed**, including direct discounted-payoff integration for 324 ordinary-range
  call/put prices and inverse-quote identities. Integration does not reuse the pricing formula's CDF.
- Changes to future synthetic observations leave earlier forecasts and development selection
  unchanged. Holdout metrics remain diagnostics, never selection inputs.
- A clean virtual environment installed a built wheel, with no inherited system packages.
  The installed command was exercised from outside the checkout.
- The committed example uses seed `20260907` and the numerical versions in
  [requirements-reproduce.txt](../requirements-reproduce.txt). Test runner: Python 3.14.5,
  pytest 9.1.1. CI also runs its configured Python-version matrix.
- The credit conclusion is unchanged: the selected ridge model's holdout MAE is **129.43 bp**,
  worse than persistence at **89.67 bp**. No seed or model was changed to improve the result.

```bash
python -m pip install -c requirements-reproduce.txt '.[dev]'
python -m pytest -q
python -m ruff check src tests
# From any directory, choose an output path that does not exist:
model-risk-lab --seed 20260907 --output /tmp/my-new-validation-report
model-risk-lab --verify /tmp/my-new-validation-report
```

## Interpretation

The useful model-risk lesson is to challenge how evidence can fail, including the operational
path that produces it. Numerical agreement alone would miss a stale chart. A positive step size
alone would miss floating-point resolution. Neither a checksum nor a passing test establishes
market calibration, regulatory approval or production readiness.

All inputs remain invented and all implementation is independent of employer/client materials.
Arbitrary extreme-tail pricing, realistic publication lags and revision-aware macro data remain
explicit future work in the [roadmap](../ROADMAP.md).
