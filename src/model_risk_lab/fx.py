"""European FX options under Garman–Kohlhagen; see docs/FX_METHOD.md.

Spot and strike: DOM per FOR. Prices: DOM. Notional: FOR.
Rates and volatility: annual decimals. Time: years. Delta: unadjusted spot delta.
"""

from dataclasses import dataclass, replace
from math import erfc, exp, isfinite, log, pi, sqrt
from numbers import Real


def _number(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be a finite real number")


@dataclass(frozen=True)
class FXOption:
    spot: float
    strike: float
    expiry_years: float
    volatility_decimal: float
    domestic_rate: float
    foreign_rate: float
    notional_foreign: float = 1.0

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            _number(name, value)
        for name in ("spot", "strike", "notional_foreign"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.expiry_years < 0 or self.volatility_decimal < 0:
            raise ValueError("expiry and volatility must be nonnegative")


def annual_decimal(value: float, *, unit: str) -> float:
    """Explicit unit conversion; never infer a unit from the number's magnitude."""
    _number("value", value)
    if unit == "decimal":
        return float(value)
    if unit == "percent":
        return float(value) / 100
    raise ValueError("unit must be 'decimal' or 'percent'")


def _sign(kind: str) -> int:
    if kind not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'")
    return 1 if kind == "call" else -1


def _cdf(x: float) -> float:
    return 0.5 * erfc(-x / sqrt(2))


def _terms(o: FXOption) -> tuple[float, float, float, float]:
    s, k, t, v = o.spot, o.strike, o.expiry_years, o.volatility_decimal
    d1 = (log(s / k) + (o.domestic_rate - o.foreign_rate + v * v / 2) * t)
    d1 /= v * sqrt(t)
    return d1, d1 - v * sqrt(t), exp(-o.domestic_rate * t), exp(-o.foreign_rate * t)


def price(o: FXOption, kind: str = "call") -> float:
    q = _sign(kind)
    if o.expiry_years == 0:
        return o.notional_foreign * max(q * (o.spot - o.strike), 0.0)
    if o.volatility_decimal == 0:
        forward_pv = o.spot * exp(-o.foreign_rate * o.expiry_years)
        forward_pv -= o.strike * exp(-o.domestic_rate * o.expiry_years)
        return o.notional_foreign * max(q * forward_pv, 0.0)
    d1, d2, dd, df = _terms(o)
    return o.notional_foreign * q * (o.spot * df * _cdf(q * d1)
                                     - o.strike * dd * _cdf(q * d2))


def greeks(o: FXOption, kind: str = "call") -> dict[str, float]:
    """Raw derivatives. Vega per 1.0 volatility; rhos per 1.0 annual rate."""
    q = _sign(kind)
    if o.expiry_years == 0 or o.volatility_decimal == 0:
        raise ValueError("regular Greeks require strictly positive expiry and volatility")
    d1, d2, dd, df = _terms(o)
    n, s, k, t, v = (o.notional_foreign, o.spot, o.strike,
                      o.expiry_years, o.volatility_decimal)
    density = exp(-d1 * d1 / 2) / sqrt(2 * pi)
    return {
        "delta": n * q * df * _cdf(q * d1),
        "gamma": n * df * density / (s * v * sqrt(t)),
        "vega": n * s * df * density * sqrt(t),
        "rho_domestic": n * q * k * t * dd * _cdf(q * d2),
        "rho_foreign": -n * q * s * t * df * _cdf(q * d1),
    }


def display_greeks(raw: dict[str, float]) -> dict[str, float]:
    return {
        "spot_delta_foreign": raw["delta"],
        "gamma_domestic_per_quote_squared": raw["gamma"],
        "vega_domestic_per_vol_point": raw["vega"] * 0.01,
        "rho_domestic_per_bp": raw["rho_domestic"] * 0.0001,
        "rho_foreign_per_bp": raw["rho_foreign"] * 0.0001,
    }


def finite_difference(o: FXOption, kind: str, greek: str, bump: float) -> dict[str, float]:
    """Price-based central difference at a caller-specified absolute bump."""
    _number("bump", bump)
    if bump <= 0:
        raise ValueError("bump must be positive")
    fields = {"delta": "spot", "gamma": "spot", "vega": "volatility_decimal",
              "rho_domestic": "domestic_rate", "rho_foreign": "foreign_rate"}
    if greek not in fields:
        raise ValueError("unsupported Greek")
    if o.expiry_years <= 0 or o.volatility_decimal <= 0:
        raise ValueError("finite difference experiment requires a smooth interior point")
    field = fields[greek]
    value = getattr(o, field)
    up, down = value + bump, value - bump
    if not isfinite(up) or not isfinite(down) or not down < value < up:
        raise ValueError("bump is outside floating-point resolution or range")
    if field in ("spot", "volatility_decimal") and down <= 0:
        raise ValueError("bump must stay inside the positive input domain")
    denominator = bump * bump if greek == "gamma" else 2 * bump
    if not isfinite(denominator) or denominator == 0:
        raise ValueError("finite-difference denominator is outside floating-point range")
    plus = price(replace(o, **{field: up}), kind)
    minus = price(replace(o, **{field: down}), kind)
    base = price(o, kind)
    estimate = ((plus - 2 * base + minus) if greek == "gamma" else (plus - minus)) / denominator
    if not all(isfinite(v) for v in (plus, minus, base, estimate)):
        raise ValueError("finite-difference prices or estimate are outside floating-point range")
    return {"bump": bump, "price_plus": plus, "price_base": base,
            "price_minus": minus, "estimate": estimate}


def convergence_experiment() -> list[dict]:
    """All predeclared bumps are retained, including cancellation at tiny bumps."""
    scenarios = {
        "near_money": FXOption(1.2, 1.25, 1.5, 0.24, 0.035, 0.015),
        "in_money": FXOption(1.4, 1.1, 0.75, 0.18, -0.005, 0.012),
        "short_expiry": FXOption(0.95, 1.0, 0.1, 0.31, 0.02, -0.01),
    }
    rows = []
    for case, o in scenarios.items():
        for kind in ("call", "put"):
            for greek, analytic in greeks(o, kind).items():
                scale = (o.spot if greek in ("delta", "gamma")
                         else o.volatility_decimal if greek == "vega" else 1.0)
                for relative_bump in (0.02, 0.01, 0.005, 0.001, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8):
                    result = finite_difference(o, kind, greek, scale * relative_bump)
                    error = abs(result["estimate"] - analytic)
                    rows.append({"case": case, "kind": kind, "greek": greek,
                                 "relative_bump": relative_bump, "analytic": analytic,
                                 **result, "absolute_error": error,
                                 "relative_error": error / abs(analytic) if analytic else None,
                                 "inputs": vars(o)})
    return rows
