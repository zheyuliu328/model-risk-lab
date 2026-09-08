# FX pricing and sensitivity contract

## Public model

We independently implement the European Garman–Kohlhagen formula. See
[Privault's public lecture notes, §16.3](https://personal.ntu.edu.sg/nprivault/MA5182/change-of-numeraire-forward-measures.pdf)
and [Muck (2022), §2.1](https://link.springer.com/article/10.1007/s11147-022-09189-9).
The original paper is [Garman & Kohlhagen (1983)](https://doi.org/10.1016/S0261-5606(83)80001-1)
(bibliographic reference; its full text was not used in this implementation).

Spot S and strike K are DOM per FOR. Foreign notional N is positive. Time T is in years;
domestic/foreign rates rd/rf are continuously compounded annual decimals; sigma is annual volatility
as a decimal. Negative rates are allowed. Dates, settlement conventions and premium adjustment are absent.

`annual_decimal(20, unit="percent")` equals `annual_decimal(0.20, unit="decimal")`.
The library never guesses units. A value of 20 explicitly labelled decimal means 2,000% volatility;
the library cannot infer that the caller intended 20%.

```
Dd = exp(-rd T)                         Df = exp(-rf T)
d1 = [log(S/K) + (rd-rf+sigma²/2) T] / (sigma sqrt(T))
d2 = d1 - sigma sqrt(T)
C  = N [S Df Phi(d1) - K Dd Phi(d2)]
P  = N [K Dd Phi(-d2) - S Df Phi(-d1)]
C-P = N (S Df - K Dd)
```

Phi and phi denote the standard normal CDF and density. With q=+1 for call, q=-1 for put:

| Derivative | Formula | Display units |
|:--|:--|:--|
| Spot delta | N q Df Phi(q d1) | FOR; unadjusted spot delta |
| Gamma | N Df phi(d1) / (S sigma sqrt(T)) | DOM per quote unit² |
| Vega | N S Df phi(d1) sqrt(T) | Raw × 0.01 gives DOM per volatility percentage point |
| Domestic rho | N q K T Dd Phi(q d2) | Raw × 0.0001 gives DOM per domestic-rate bp |
| Foreign rho | -N q S T Df Phi(q d1) | Raw × 0.0001 gives DOM per foreign-rate bp |

T=0 returns payoff. Sigma=0 returns discounted deterministic payoff. Regular Greeks and the
difference experiment explicitly reject these boundaries; near a payoff kink they are not smooth.
Floating-point arithmetic is intended for ordinary demonstration ranges, not arbitrary extreme magnitudes.

## Numerical challenge

For one factor x at a time, central differences use `(V(x+h)-V(x-h))/(2h)`;
gamma uses `(V(S+h)-2V(S)+V(S-h))/h²`. All other factors stay fixed.
See [NIST central difference formulas](https://dlmf.nist.gov/3.4#iii) and
[SciPy's discussion of small-step precision loss](https://docs.scipy.org/doc/scipy/reference/generated/scipy.differentiate.derivative.html).

Three invented scenarios × two payoff types × five Greeks × nine predetermined bumps give
270 estimates. Bump scales are S for spot derivatives, sigma for vega, and 1 for either rate.
The relative sizes are 0.02, 0.01, 0.005, 0.001, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8.
Every up/base/down price, derivative and error is retained. No best bump is selected.
Bumps that cannot change the input float, or yield a zero/non-finite difference denominator,
are rejected explicitly. This prevents an unrepresentable step from masquerading as a zero
derivative; representable but cancellation-prone steps remain visible in the experiment.

The ordinary-scenario derivative tests use a fixed relative bump of 1e-4 and
`|estimate-analytic| <= max(2e-7, 2e-5*|analytic|)` (pytest's absolute/relative rule),
at unit notional. This is an experiment tolerance, not a market or regulatory standard.
Parity is checked separately to absolute precision 1e-14 in the specified cases.
An additional ordinary-range grid compares 324 call/put prices with direct numerical integration
of the discounted lognormal payoff and checks the inverse-quote identity. That integration uses
a predeclared +/-12 standard-normal-shock bound; it does not certify arbitrary extreme tails.

## What the checks do and do not prove

Parity, price bounds, scaling and derivative identities provide different constraints. Price-based
finite differences are independent derivative calculations, but share the same pricing assumptions.
They cannot prove calibration quality. A common unit error can leave both calculations consistent.
Reducing h initially removes truncation error; very small h can magnify floating-point cancellation.

No volatility surface, smile dynamics, premium-adjusted delta, American exercise, barrier feature,
counterparty/funding adjustment or SIMM aggregation is modelled. Tail numerical stability is a
specific future experiment, not an already established property of this implementation.
