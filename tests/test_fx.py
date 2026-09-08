from dataclasses import replace
from itertools import product
from math import exp, log, pi, sqrt

import pytest
from scipy.integrate import quad

from model_risk_lab.fx import (FXOption, annual_decimal, convergence_experiment,
                               display_greeks, finite_difference, greeks, price)

BASE = FXOption(1.2, 1.25, 1.5, .24, .035, .015)


@pytest.mark.parametrize("field,value", [("spot", 0), ("strike", -1), ("notional_foreign", 0),
                                         ("expiry_years", -1), ("volatility_decimal", -.1),
                                         ("domestic_rate", float("nan")), ("foreign_rate", float("inf")),
                                         ("spot", True), ("strike", "1")])
def test_invalid_inputs(field, value):
    with pytest.raises(ValueError):
        replace(BASE, **{field: value})


@pytest.mark.parametrize("spot", [.8, 1.2, 1.6])
@pytest.mark.parametrize("rates", [(.03, .01), (-.01, .02)])
def test_parity_price_bounds_and_direction(spot, rates):
    o = replace(BASE, spot=spot, domestic_rate=rates[0], foreign_rate=rates[1])
    discounted_s = spot * exp(-rates[1] * o.expiry_years)
    discounted_k = o.strike * exp(-rates[0] * o.expiry_years)
    c, p = price(o), price(o, "put")
    assert c-p == pytest.approx(discounted_s-discounted_k, abs=1e-14)
    assert max(discounted_s-discounted_k, 0) - 1e-14 <= c <= discounted_s
    assert max(discounted_k-discounted_s, 0) - 1e-14 <= p <= discounted_k
    assert price(replace(o, spot=spot+.01)) > c
    assert price(replace(o, spot=spot+.01), "put") < p
    for kind in ("call", "put"):
        assert price(replace(o, volatility_decimal=.30), kind) > price(o, kind)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_expiry_and_zero_volatility(kind):
    q = 1 if kind == "call" else -1
    assert price(replace(BASE, expiry_years=0), kind) == max(q*(BASE.spot-BASE.strike), 0)
    forward_pv = BASE.spot*exp(-BASE.foreign_rate*BASE.expiry_years)
    forward_pv -= BASE.strike*exp(-BASE.domestic_rate*BASE.expiry_years)
    assert price(replace(BASE, volatility_decimal=0), kind) == max(q*forward_pv, 0)
    for boundary in (replace(BASE, expiry_years=0), replace(BASE, volatility_decimal=0)):
        with pytest.raises(ValueError):
            greeks(boundary, kind)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_notional_and_quote_scaling(kind):
    raw = greeks(BASE, kind)
    bigger = replace(BASE, notional_foreign=7)
    assert price(bigger, kind) == pytest.approx(7*price(BASE, kind))
    for name, value in greeks(bigger, kind).items():
        assert value == pytest.approx(7*raw[name])
    scaled = replace(BASE, spot=BASE.spot*5, strike=BASE.strike*5)
    assert price(scaled, kind) == pytest.approx(5*price(BASE, kind))
    assert greeks(scaled, kind)["delta"] == pytest.approx(raw["delta"])
    assert greeks(scaled, kind)["gamma"] == pytest.approx(raw["gamma"]/5)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("greek,field", [("delta", "spot"), ("gamma", "spot"),
                                        ("vega", "volatility_decimal"),
                                        ("rho_domestic", "domestic_rate"),
                                        ("rho_foreign", "foreign_rate")])
def test_analytic_derivative_against_prices(kind, greek, field):
    bump = (getattr(BASE, field) if field in ("spot", "volatility_decimal") else 1) * 1e-4
    estimate = finite_difference(BASE, kind, greek, bump)["estimate"]
    assert estimate == pytest.approx(greeks(BASE, kind)[greek], abs=2e-7, rel=2e-5)


def test_greek_identities():
    c, p = greeks(BASE), greeks(BASE, "put")
    assert c["delta"]-p["delta"] == pytest.approx(exp(-BASE.foreign_rate*BASE.expiry_years))
    for name in ("gamma", "vega"):
        assert c[name] == p[name]
    assert c["vega"] == pytest.approx(BASE.spot**2*BASE.volatility_decimal*BASE.expiry_years*c["gamma"])
    assert c["rho_domestic"]-p["rho_domestic"] == pytest.approx(
        BASE.strike*BASE.expiry_years*exp(-BASE.domestic_rate*BASE.expiry_years))
    assert c["rho_foreign"]-p["rho_foreign"] == pytest.approx(
        -BASE.spot*BASE.expiry_years*exp(-BASE.foreign_rate*BASE.expiry_years))


def test_unit_contract_and_display_detect_factor_errors():
    assert annual_decimal(20, unit="percent") == annual_decimal(.2, unit="decimal")
    # Incorrect intent (20 passed as decimal) is not discoverable from parity alone.
    wrong = replace(BASE, volatility_decimal=annual_decimal(20, unit="decimal"))
    assert price(wrong)-price(wrong, "put") == pytest.approx(
        wrong.spot*exp(-wrong.foreign_rate*wrong.expiry_years)
        - wrong.strike*exp(-wrong.domestic_rate*wrong.expiry_years))
    g, shown = greeks(BASE), display_greeks(greeks(BASE))
    assert g["vega"]/shown["vega_domestic_per_vol_point"] == pytest.approx(100)
    assert g["rho_domestic"]/shown["rho_domestic_per_bp"] == pytest.approx(10000)
    with pytest.raises(ValueError):
        annual_decimal(20, unit="auto")
    with pytest.raises(TypeError):
        annual_decimal(20)


def test_second_order_convergence_before_roundoff():
    target = greeks(BASE)["delta"]
    errors = [abs(finite_difference(BASE, "call", "delta", BASE.spot*h)["estimate"]-target)
              for h in (.02, .01, .005)]
    assert 3.5 < errors[0]/errors[1] < 4.5
    assert 3.5 < errors[1]/errors[2] < 4.5
    rows = convergence_experiment()
    assert len(rows) == 3*2*5*9
    assert {r["relative_bump"] for r in rows} == {.02, .01, .005, .001, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8}
    for row in rows:
        if row["relative_bump"] == 1e-4:
            assert row["estimate"] == pytest.approx(row["analytic"], abs=2e-7, rel=2e-5)


@pytest.mark.parametrize("greek,bump", [("delta", 1.2), ("vega", .24), ("delta", 0), ("unknown", .01)])
def test_invalid_bump_or_kind(greek, bump):
    with pytest.raises(ValueError):
        finite_difference(BASE, "call", greek, bump)
    with pytest.raises(ValueError):
        price(BASE, "digital")


@pytest.mark.parametrize("greek", ["delta", "gamma", "vega", "rho_domestic", "rho_foreign"])
@pytest.mark.parametrize("bump", [1e-30, 1e-200])
def test_unrepresentable_bump_is_rejected_instead_of_returning_zero(greek, bump):
    with pytest.raises(ValueError, match="floating-point"):
        finite_difference(BASE, "call", greek, bump)


def test_gamma_denominator_underflow_is_rejected_before_pricing():
    # Here +/- bump are distinct input floats, but bump squared becomes zero.
    tiny_quote = replace(BASE, spot=1e-190, strike=1e-190)
    with pytest.raises(ValueError, match="denominator"):
        finite_difference(tiny_quote, "call", "gamma", 1e-200)


def test_prices_against_independent_payoff_integration_and_inverse_quotes():
    # Integrate the discounted lognormal payoff directly, without d1/d2 or a CDF.
    # The +/-12 normal-shock bound is sufficient for this predeclared ordinary grid;
    # it is not a universal extreme-tail pricing reference.
    for spot, ratio, t, vol, rates in product(
        (.01, 1.2, 26000.), (.7, 1., 1.4), (.01, 1., 5.), (.05, .25, .6),
        ((.05, .02), (-.03, .01)),
    ):
        rd, rf = rates
        strike = spot * ratio
        option = FXOption(spot, strike, t, vol, rd, rf)
        drift, diffusion = (rd - rf - vol**2 / 2) * t, vol * sqrt(t)
        exercise_shock = (log(ratio) - drift) / diffusion
        for kind, sign in (("call", 1), ("put", -1)):
            start, end = ((max(-12., exercise_shock), 12.) if sign == 1
                          else (-12., min(12., exercise_shock)))

            def integrand(z):
                payoff_per_spot = max(sign * (exp(drift + diffusion*z) - ratio), 0)
                return payoff_per_spot * exp(-z*z/2) / sqrt(2*pi)

            integral = (0 if start >= end else
                        quad(integrand, start, end, epsabs=1e-12, epsrel=1e-12)[0])
            reference = spot * exp(-rd*t) * integral
            assert price(option, kind) == pytest.approx(reference, abs=1e-10*spot, rel=0)
        inverse = FXOption(1/spot, 1/strike, t, vol, rf, rd)
        assert price(option) == pytest.approx(spot*strike*price(inverse, "put"), abs=1e-10*spot, rel=0)
